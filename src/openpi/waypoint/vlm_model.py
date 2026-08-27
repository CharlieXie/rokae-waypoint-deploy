"""VLM waypoint prediction model.

Pure PaliGemma autoregressive model for waypoint sequence prediction.
Architecture mirrors Pi0-FAST: SigLIP image encoder + Gemma 2B decoder.
Training: CE loss on waypoint tokens. Inference: constrained AR decoding.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import openpi.models.gemma as _gemma
from openpi.models_pytorch.lora_pytorch import compute_lm_logits
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.waypoint.planner_decode import legacy_constrained_next_token
from openpi.waypoint.planner_decode import legacy_decode_budget
from openpi.waypoint.tokenizer import (
    PALIGEMMA_VOCAB_SIZE,
    _DUR_TOKEN_ID,
    _DURATION_BASE,
    _PROPRIO_BASE,
    _WP_TOKEN_ID,
    DURATION_N_BINS,
    PROPRIO_N_BINS,
)

logger = logging.getLogger(__name__)


class PI0WaypointVLM(nn.Module):
    """Autoregressive waypoint prediction using PaliGemma.

    Only uses the PaliGemma (Gemma 2B + SigLIP) portion of the model.
    No action expert needed — waypoints are predicted as discrete tokens.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_token_len = getattr(config, "max_token_len", 256)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        from transformers import PaliGemmaForConditionalGeneration
        from transformers.models.auto import CONFIG_MAPPING

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = PALIGEMMA_VOCAB_SIZE
        vlm_config_hf.image_token_index = PALIGEMMA_VOCAB_SIZE
        vlm_config_hf.text_config.hidden_size = paligemma_config.width
        vlm_config_hf.text_config.intermediate_size = paligemma_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = paligemma_config.num_heads
        vlm_config_hf.text_config.head_dim = paligemma_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = paligemma_config.depth
        vlm_config_hf.text_config.num_key_value_heads = paligemma_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = PALIGEMMA_VOCAB_SIZE
        vlm_config_hf.text_config.use_adarms = False
        vlm_config_hf.text_config.adarms_cond_dim = None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gradient_checkpointing_enabled = False

        precision = getattr(config, "dtype", "float32")
        if precision == "bfloat16":
            self.paligemma.to(dtype=torch.bfloat16)
            _float32_selectors = [
                "patch_embedding.weight", "patch_embedding.bias",
                "position_embedding.weight",
                "input_layernorm", "post_attention_layernorm",
                "model.norm",
            ]
            for name, param in self.paligemma.named_parameters():
                if any(s in name for s in _float32_selectors):
                    param.data = param.data.to(torch.float32)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma.gradient_checkpointing_disable()

    def _ckpt(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def embed_images(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        """Embed images using SigLIP vision encoder."""
        return [self._ckpt(self.paligemma.model.get_image_features, img) for img in images]

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.paligemma.language_model.embed_tokens(token_ids)
        return emb * math.sqrt(emb.shape[-1])

    def forward(self, batch: dict) -> torch.Tensor:
        """Training forward: compute CE loss on waypoint tokens.

        Args:
            batch: dict with keys:
                images: dict of {model_key: (B, H, W, C) tensor}
                image_masks: dict of {model_key: (B,) bool tensor}
                tokens: (B, L) int64 token IDs
                token_mask: (B, L) bool
                ar_mask: (B, L) int32
                loss_mask: (B, L) bool

        Returns:
            Scalar loss.
        """
        tokens = batch["tokens"]
        token_mask = batch["token_mask"]
        ar_mask = batch["ar_mask"]
        loss_mask = batch["loss_mask"]
        device = tokens.device
        B, L = tokens.shape

        embs_list = []
        pad_list = []
        ar_list = []

        image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        for key in image_keys:
            if key not in batch["images"]:
                continue
            img = batch["images"][key]
            if img.shape[-1] == 3 and img.ndim == 4:
                img = img.permute(0, 3, 1, 2)  # BHWC -> BCHW

            img_emb = self._ckpt(self.paligemma.model.get_image_features, img)
            n_img_tokens = img_emb.shape[1]
            embs_list.append(img_emb)

            img_mask = batch["image_masks"][key]
            pad_list.append(img_mask[:, None].expand(B, n_img_tokens))
            ar_list.append(torch.zeros(B, n_img_tokens, dtype=torch.int32, device=device))

        token_embs = self.embed_tokens(tokens)
        embs_list.append(token_embs)
        pad_list.append(token_mask)
        ar_list.append(ar_mask)

        all_embs = torch.cat(embs_list, dim=1)
        all_pad = torch.cat(pad_list, dim=1)
        all_ar = torch.cat(ar_list, dim=1)

        n_img_total = all_embs.shape[1] - L
        full_loss_mask = torch.cat([
            torch.zeros(B, n_img_total, dtype=torch.bool, device=device),
            loss_mask,
        ], dim=1)

        att_2d = make_att_2d_masks(all_pad, all_ar)
        position_ids = torch.cumsum(all_pad, dim=1) - 1

        att_4d = att_2d[:, None, :, :]
        att_4d = torch.where(att_4d, 0.0, -2.3819763e38)

        model_dtype = self.paligemma.language_model.embed_tokens.weight.dtype
        if all_embs.dtype != model_dtype:
            all_embs = all_embs.to(model_dtype)
        if att_4d.dtype != model_dtype:
            att_4d = att_4d.to(model_dtype)

        outputs = self.paligemma.language_model.forward(
            inputs_embeds=all_embs[:, :-1],
            attention_mask=att_4d[:, :, :-1, :-1],
            position_ids=position_ids[:, :-1],
            use_cache=False,
        )

        hidden = outputs.last_hidden_state

        targets = torch.cat([
            torch.zeros(B, n_img_total, dtype=torch.long, device=device),
            tokens,
        ], dim=1)[:, 1:]

        shift_loss_mask = full_loss_mask[:, 1:]

        # NEVER `F.linear(h, embed_tokens.weight)` here: with LoRA applied,
        # `.weight` is the un-adapted base matrix (see compute_lm_logits).
        lm_head = self.paligemma.language_model.embed_tokens
        mask = shift_loss_mask
        if mask.any():
            active_hidden = hidden[mask]
            active_targets = targets[mask]
            active_logits = compute_lm_logits(lm_head, active_hidden.float())
            loss = F.cross_entropy(active_logits, active_targets)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss

    @torch.no_grad()
    def generate_waypoints(
        self,
        images: dict[str, torch.Tensor],
        image_masks: dict[str, torch.Tensor],
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        wp_tokenizer,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
        forbid_terminal_first_waypoint: bool = False,
    ) -> list[list[tuple]]:
        """AR generation of waypoint tokens.

        ``max_new_tokens=None`` means ``legacy_decode_budget(wp_tokenizer)``
        (header + ``num_waypoints * tokens_per_waypoint``), so a wide proprio
        layout is never silently truncated by a fixed budget.

        Generates tokens autoregressively, injecting <wp>/<dur> delimiters
        at the correct positions (constrained decoding).

        The mask-and-sample step is shared with the joint model's legacy
        decoder (``planner_decode.legacy_constrained_next_token``), so
        ``temperature``, ``generator`` and ``forbid_terminal_first_waypoint``
        mean exactly the same thing on both paths.  ``generator`` matters
        whenever ``temperature > 0``: the evaluator reseeds a dedicated planner
        RNG per trial but never reseeds the global one.

        Returns:
            List (per batch) of decoded waypoint lists [(proprio, duration), ...].
        """
        if max_new_tokens is None:
            max_new_tokens = legacy_decode_budget(wp_tokenizer)
        device = prompt_tokens.device
        B = prompt_tokens.shape[0]

        embs_list = []
        pad_list = []

        image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        for key in image_keys:
            if key not in images:
                continue
            img = images[key]
            if img.shape[-1] == 3 and img.ndim == 4:
                img = img.permute(0, 3, 1, 2)
            img_emb = self.paligemma.model.get_image_features(img)
            n_img = img_emb.shape[1]
            embs_list.append(img_emb)
            pad_list.append(image_masks[key][:, None].expand(B, n_img))

        token_embs = self.embed_tokens(prompt_tokens)
        embs_list.append(token_embs)
        pad_list.append(prompt_mask)

        prefix_embs = torch.cat(embs_list, dim=1)
        prefix_pad = torch.cat(pad_list, dim=1)

        prefix_len = prefix_embs.shape[1]
        prefix_ar = torch.zeros_like(prefix_pad, dtype=torch.int32)
        prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_ar)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1

        total_len = prefix_len + max_new_tokens
        prefix_att_padded = F.pad(prefix_att_2d, (0, max_new_tokens), value=False)
        prefix_att_4d = prefix_att_padded[:, None, :, :]
        prefix_att_4d = torch.where(prefix_att_4d, 0.0, -2.3819763e38)

        model_dtype = self.paligemma.language_model.embed_tokens.weight.dtype
        if prefix_embs.dtype != model_dtype:
            prefix_embs = prefix_embs.to(model_dtype)
        if prefix_att_4d.dtype != model_dtype:
            prefix_att_4d = prefix_att_4d.to(model_dtype)

        outputs = self.paligemma.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        last_hidden = outputs.last_hidden_state[:, -1:]

        lm_head = self.paligemma.language_model.embed_tokens
        last_logits = compute_lm_logits(lm_head, last_hidden.float())

        generated_tokens = []
        tpw = wp_tokenizer.tokens_per_waypoint
        prefill_len = torch.sum(prefix_pad, dim=-1)

        PALIGEMMA_EOS = 1
        action_header = wp_tokenizer._pg_tokenizer.encode("Action: ")
        header_injected = [False] * B
        header_pos = [0] * B

        all_output_tokens = [[] for _ in range(B)]

        # Ranges below are only used by the post-hoc range diagnostics; the
        # decoding masks themselves live in legacy_constrained_next_token.
        proprio_lo = _PROPRIO_BASE - PROPRIO_N_BINS + 1
        proprio_hi = _PROPRIO_BASE
        duration_lo = _DURATION_BASE - DURATION_N_BINS + 1
        duration_hi = _DURATION_BASE

        # Pre-compute position constants from tokenizer
        dur_delim_pos = wp_tokenizer.dur_delimiter_pos_in_wp  # position of <dur>
        dur_val_pos = wp_tokenizer.duration_pos_in_wp   # position of duration value

        for step in range(max_new_tokens):
            # --- Logit masking + sampling (shared with the joint model) ---
            next_token = legacy_constrained_next_token(
                last_logits,
                plan_lens=[len(t) for t in all_output_tokens],
                header_injected=header_injected,
                wp_tokenizer=wp_tokenizer,
                temperature=temperature,
                generator=generator,
                forbid_terminal_first_waypoint=forbid_terminal_first_waypoint,
            )

            # --- Force-inject structural delimiters ---
            for b in range(B):
                if not header_injected[b]:
                    if header_pos[b] < len(action_header):
                        next_token[b, 0] = action_header[header_pos[b]]
                        header_pos[b] += 1
                        continue
                    else:
                        header_injected[b] = True

                wp_count = len(all_output_tokens[b])
                pos_in_wp = wp_count % tpw

                if pos_in_wp == 0:
                    next_token[b, 0] = wp_tokenizer.wp_token_id
                elif pos_in_wp == dur_delim_pos:
                    next_token[b, 0] = wp_tokenizer.dur_token_id

                all_output_tokens[b].append(next_token[b, 0].item())

            token_emb = self.embed_tokens(next_token)
            if token_emb.dtype != self.paligemma.language_model.embed_tokens.weight.dtype:
                token_emb = token_emb.to(self.paligemma.language_model.embed_tokens.weight.dtype)

            positions = prefill_len[:, None] + step
            attn_len = prefix_len + step + 1
            mask = torch.zeros(B, 1, 1, attn_len, device=device, dtype=model_dtype)

            outputs = self.paligemma.language_model.forward(
                inputs_embeds=token_emb,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            last_hidden = outputs.last_hidden_state
            last_logits = compute_lm_logits(lm_head, last_hidden.float())

            all_done = all(len(t) >= tpw * wp_tokenizer.num_waypoints for t in all_output_tokens)
            if all_done:
                break

        results = []
        for b in range(B):
            tids = all_output_tokens[b]
            n_proprio_ok = sum(
                1 for i, t in enumerate(tids)
                if 1 <= (i % tpw) <= wp_tokenizer.proprio_dim
                and proprio_lo <= t <= proprio_hi
            )
            n_duration_ok = sum(
                1 for i, t in enumerate(tids)
                if (i % tpw) == dur_val_pos
                and duration_lo <= t <= duration_hi
            )
            n_proprio_total = sum(
                1 for i in range(len(tids)) if 1 <= (i % tpw) <= wp_tokenizer.proprio_dim
            )
            n_duration_total = sum(
                1 for i in range(len(tids)) if (i % tpw) == dur_val_pos
            )
            if n_proprio_total > 0 or n_duration_total > 0:
                logger.debug(
                    f"VLM raw tokens: proprio_in_range={n_proprio_ok}/{n_proprio_total}, "
                    f"duration_in_range={n_duration_ok}/{n_duration_total}"
                )
            waypoints = wp_tokenizer.decode_waypoints(tids)
            results.append(waypoints)

        return results

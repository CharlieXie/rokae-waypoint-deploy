"""Action Expert model for waypoint VLA.

Extends PI0Pytorch (Pi0.5 mode) with:
  - Duration conditioning via AdaRMSNorm (combined with flow matching timestep)
  - Waypoint proprio conditioning (2 tokens: start_wp + end_wp)
  - Action padding mask for loss computation
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi0_pytorch import (
    PI0Pytorch,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    sample_beta,
)

logger = logging.getLogger(__name__)

DURATION_MAX = 33.0


class PI0WaypointAE(nn.Module):
    """Flow-matching Action Expert with duration conditioning.

    Architecture mirrors Pi0.5 but replaces state conditioning with
    waypoint proprio conditioning and adds duration via AdaRMSNorm.
    """

    def __init__(self, config, aug_cfg: dict | None = None):
        super().__init__()
        self.config = config
        self.aug_cfg = aug_cfg
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.duration_max = getattr(config, "duration_max", DURATION_MAX)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        expert_width = action_expert_config.width

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(self.action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, self.action_dim)

        self.proprio_encoder = nn.Linear(self.action_dim, expert_width)

        # time_mlp_in takes cat(time_emb, dur_emb) = 2*W → W
        # NOTE: the Pi0.5 base checkpoint has time_mlp_in as W→W.
        # We initialize fresh weights for time_mlp_in; all other layers
        # can be loaded from the base checkpoint via strict=False.
        self.time_mlp_in = nn.Linear(2 * expert_width, expert_width)
        self.time_mlp_out = nn.Linear(expert_width, expert_width)

        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

    def _ckpt(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Embed images + language tokens for PaliGemma prefix."""
        embs, pad_masks, att_masks = [], [], []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._ckpt(self.paligemma_with_expert.embed_image, img)
            bsize, n_img = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, n_img))
            att_masks += [0] * n_img

        lang_emb = self._ckpt(self.paligemma_with_expert.embed_language_tokens, lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        bsize = pad_masks.shape[0]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, -1)
        return embs, pad_masks, att_masks

    def embed_suffix(self, start_proprio, end_proprio, noisy_actions, timestep, duration):
        """Embed proprio conditioning + noisy actions + time/duration for expert.

        Args:
            start_proprio: (B, action_dim) start waypoint proprio.
            end_proprio: (B, action_dim) end waypoint proprio.
            noisy_actions: (B, horizon, action_dim) noisy action sequence.
            timestep: (B,) flow matching timestep.
            duration: (B,) waypoint duration (raw, not normalized).
        """
        embs, pad_masks, att_masks = [], [], []
        device = noisy_actions.device
        bsize = noisy_actions.shape[0]
        expert_width = self.action_in_proj.out_features

        proprio_input = torch.stack([start_proprio, end_proprio], dim=1)  # (B, 2, D)
        # Always cast to match weight dtype (bfloat16 in Pi0.5 mode)
        proprio_input = proprio_input.to(self.proprio_encoder.weight.dtype)
        proprio_emb = self._ckpt(self.proprio_encoder, proprio_input)  # (B, 2, W)
        embs.append(proprio_emb)
        pad_masks.append(torch.ones(bsize, 2, dtype=torch.bool, device=device))
        att_masks += [1, 0]

        time_emb = create_sinusoidal_pos_embedding(
            timestep, expert_width, min_period=4e-3, max_period=4.0, device=device,
        ).to(timestep.dtype)

        dur_normalized = duration / self.duration_max
        dur_emb = create_sinusoidal_pos_embedding(
            dur_normalized, expert_width, min_period=4e-3, max_period=4.0, device=device,
        ).to(timestep.dtype)

        combined = torch.cat([time_emb, dur_emb], dim=-1)  # (B, 2*W)
        # Cast to match time_mlp_in weight dtype (bfloat16 in Pi0.5 mode)
        combined = combined.to(self.time_mlp_in.weight.dtype)
        x = self.time_mlp_in(combined)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        adarms_cond = F.silu(x)

        noisy_actions_cast = noisy_actions.to(self.action_in_proj.weight.dtype)
        action_emb = self._ckpt(self.action_in_proj, noisy_actions_cast)
        embs.append(action_emb)
        pad_masks.append(torch.ones(bsize, self.action_horizon, dtype=torch.bool, device=device))
        att_masks += [1] + [0] * (self.action_horizon - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=device)
        att_masks = att_masks[None, :].expand(bsize, -1)

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        observation,
        start_proprio,
        end_proprio,
        actions,
        duration,
        action_pad_mask=None,
        action_dim_mask=None,
        noise=None,
        time=None,
    ):
        """Training forward: compute per-element MSE loss.

        Returns:
            loss: scalar mean loss.
        """
        import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
        obs = _preprocessing.preprocess_observation_pytorch(observation, train=True, aug_cfg=self.aug_cfg)
        images = list(obs.images.values())
        img_masks = list(obs.image_masks.values())
        lang_tokens = obs.tokenized_prompt
        lang_masks = obs.tokenized_prompt_mask

        bsize = actions.shape[0]
        device = actions.device

        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time_beta = sample_beta(1.5, 1.0, bsize, device)
            time = (time_beta * 0.999 + 0.001).float()

        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(
            start_proprio, end_proprio, x_t, time, duration,
        )

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)

        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)

        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_4d = att_2d[:, None, :, :]
        att_4d = torch.where(att_4d, 0.0, -2.3819763e38)

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.action_horizon:].float()
        v_t = self.action_out_proj(suffix_out)

        loss_per_element = F.mse_loss(u_t, v_t, reduction="none")

        # Build a deterministic validity mask.  Denominator must be "the
        # number of elements we want to contribute", NOT "the number of
        # elements that happen to have non-zero loss" (the old nonzero()
        # denominator silently shrinks when predictions match targets or
        # when bf16 rounds small MSEs to zero, which makes the reported
        # loss scale non-comparable across batches).
        mask = torch.ones_like(loss_per_element)
        if action_pad_mask is not None:
            step_mask = (~action_pad_mask).float().unsqueeze(-1)
            mask = mask * step_mask
        if action_dim_mask is not None:
            dim_mask = action_dim_mask[:, None, :].float()
            mask = mask * dim_mask

        loss_per_element = loss_per_element * mask
        denom = mask.sum().clamp(min=1.0)
        return loss_per_element.sum() / denom

    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        start_proprio,
        end_proprio,
        duration,
        num_steps: int = 10,
        noise=None,
    ):
        """Inference: generate action chunk via iterative denoising."""
        import openpi.models_pytorch.preprocessing_pytorch as _prep
        obs = _prep.preprocess_observation_pytorch(observation, train=False)
        images = list(obs.images.values())
        img_masks = list(obs.image_masks.values())
        lang_tokens = obs.tokenized_prompt
        lang_masks = obs.tokenized_prompt_mask

        bsize = start_proprio.shape[0]
        device = start_proprio.device

        if noise is None:
            noise = torch.randn(bsize, self.action_horizon, self.action_dim, device=device)

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        prefix_att_4d = prefix_att_2d[:, None, :, :]
        prefix_att_4d = torch.where(prefix_att_4d, 0.0, -2.3819763e38)

        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if prefix_embs.dtype != model_dtype:
            prefix_embs = prefix_embs.to(model_dtype)
        if prefix_att_4d.dtype != model_dtype:
            prefix_att_4d = prefix_att_4d.to(model_dtype)

        _, past_kv = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        t_val = torch.tensor(1.0, device=device)

        while t_val >= -dt / 2:
            expanded_t = t_val.expand(bsize)

            suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(
                start_proprio, end_proprio, x_t, expanded_t, duration,
            )

            suffix_len = suffix_pad.shape[1]
            prefix_len = prefix_pad.shape[1]

            prefix_2d = prefix_pad[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
            full_att = torch.cat([prefix_2d, suffix_att_2d], dim=2)
            full_att_4d = full_att[:, None, :, :]
            full_att_4d = torch.where(full_att_4d, 0.0, -2.3819763e38)
            if full_att_4d.dtype != model_dtype:
                full_att_4d = full_att_4d.to(model_dtype)

            prefix_offsets = torch.sum(prefix_pad, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1

            if suffix_embs.dtype != model_dtype:
                suffix_embs = suffix_embs.to(model_dtype)

            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_4d,
                position_ids=position_ids,
                past_key_values=past_kv,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = outputs[1][:, -self.action_horizon:].float()
            v_t = self.action_out_proj(suffix_out)

            x_t = x_t + dt * v_t
            t_val = t_val + dt

        return x_t

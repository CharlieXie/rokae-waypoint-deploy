"""PEFT-style LoRA for Waypoint Joint Model (PI0WaypointJoint).

Key design choices (following HuggingFace PEFT):
  - **Wrapper pattern**: LoRALinear wraps the original nn.Linear as
    ``self.base_layer`` — zero-copy, no ``.clone()`` of weights.
  - **Default-all + skip-list**: every nn.Linear gets LoRA unless its
    full path matches an entry in ``modules_to_not_lora``.
  - **LoRA-only save/load**: ``get_lora_state_dict`` / ``load_lora_state_dict``
    only touch trainable parameters (LoRA A/B + whitelisted non-LoRA modules).
  - **DDP-friendly**: frozen base weights don't participate in gradient sync;
    only LoRA + trainable-module params are synced.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file as safetensors_save

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Per-layer LoRA hyper-parameters."""

    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    # rsLoRA: scale by alpha/sqrt(rank) instead of alpha/rank
    use_rslora: bool = False
    # Initialisation for lora_A.
    #   True / "kaiming" → kaiming_uniform (PEFT / LoRA paper default)
    #   "gaussian"       → normal(0, 1/rank)  (PEFT gaussian option)
    init_lora_weights: bool | str = True

    @property
    def scaling(self) -> float:
        if self.use_rslora:
            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank


# ---------------------------------------------------------------------------
# LoRALinear — PEFT-style wrapper (zero-copy)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps an existing ``nn.Linear`` with LoRA adapters.

    The original layer is stored as ``self.base_layer`` **by reference** —
    no ``.clone()`` or memory duplication.  During forward the output is::

        y = base_layer(x) + lora_B(lora_A(dropout(x))) * scaling
    """

    def __init__(self, base_layer: nn.Linear, config: LoRAConfig):
        super().__init__()
        # Store original layer — zero copy
        self.base_layer = base_layer
        self.config = config
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA low-rank matrices
        self.lora_A = nn.Linear(in_features, config.rank, bias=False)
        self.lora_B = nn.Linear(config.rank, out_features, bias=False)
        self.scaling = config.scaling

        if config.dropout > 0:
            self.lora_dropout = nn.Dropout(p=config.dropout)
        else:
            self.lora_dropout = nn.Identity()

        # Initialise: B = zero (always), A depends on config
        # This ensures the initial LoRA contribution is exactly 0.
        nn.init.zeros_(self.lora_B.weight)
        init = config.init_lora_weights
        if init is True or (isinstance(init, str) and init.lower() == "kaiming"):
            # PEFT / LoRA paper default: same as nn.Linear
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif isinstance(init, str) and init.lower() == "gaussian":
            # PEFT gaussian option: normal(0, 1/rank)
            nn.init.normal_(self.lora_A.weight, std=1.0 / config.rank)
        else:
            raise ValueError(f"Unknown init_lora_weights={init!r}")

        # Place LoRA params on same device / dtype as base
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A.to(device=device, dtype=dtype)
        self.lora_B.to(device=device, dtype=dtype)

    # -- forward --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        # Cast input for LoRA branch if needed (e.g. fp32 input, bf16 LoRA)
        lora_input = self.lora_dropout(x)
        if lora_input.dtype != self.lora_A.weight.dtype:
            lora_input = lora_input.to(self.lora_A.weight.dtype)
        lora_out = self.lora_B(self.lora_A(lora_input))
        return result + lora_out * self.scaling

    # -- merge / unmerge for inference ----------------------------------
    def merge(self) -> None:
        """Merge LoRA weights into base layer (irreversible without saved delta).

        Computes delta in float32 to minimize accumulation error from
        bfloat16 matmul, then casts back to the base weight dtype.
        See: torchtune#1101, PEFT#1836 — LoRA merge under bfloat16 has
        unavoidable numerical drift vs LoRA-active forward due to
        different accumulation order.  Float32 merge does NOT eliminate
        this drift (the forward graph difference remains), but it avoids
        *additional* precision loss during the merge computation itself.
        """
        with torch.no_grad():
            A = self.lora_A.weight.to(torch.float32)
            B = self.lora_B.weight.to(torch.float32)
            delta = (B @ A) * self.scaling
            self.base_layer.weight.data += delta.to(self.base_layer.weight.dtype)

    def get_delta_weight(self) -> torch.Tensor:
        """Return B @ A * scaling without modifying base (computed in float32)."""
        A = self.lora_A.weight.to(torch.float32)
        B = self.lora_B.weight.to(torch.float32)
        return (B @ A) * self.scaling

    # -- convenience properties ----------------------------------------
    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"lora_rank={self.config.rank}, lora_alpha={self.config.alpha}"
        )


# ---------------------------------------------------------------------------
# LoRAEmbedding — PEFT-style wrapper for nn.Embedding (zero-copy)
# ---------------------------------------------------------------------------

class LoRAEmbedding(nn.Module):
    """Wraps an existing ``nn.Embedding`` with LoRA adapters.

    Forward (token lookup)::

        y = base_layer(x) + A[x] @ B * scaling

    Also provides ``compute_logits(hidden)`` for using the same weight as
    a linear output head (lm_head), with LoRA correction::

        logits = hidden @ W^T + (hidden @ B^T @ A^T) * scaling
    """

    def __init__(self, base_layer: nn.Embedding, config: LoRAConfig):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        num_embeddings = base_layer.num_embeddings
        embedding_dim = base_layer.embedding_dim

        # LoRA parameters (nn.Parameter, not nn.Linear — following PEFT)
        # A: (num_embeddings, rank) — per-token low-rank vector
        # B: (rank, embedding_dim) — shared projection
        self.lora_embedding_A = nn.Parameter(
            torch.zeros(num_embeddings, config.rank)
        )
        self.lora_embedding_B = nn.Parameter(
            torch.zeros(config.rank, embedding_dim)
        )
        self.scaling = config.scaling

        # Init: A = zeros, B = normal → initial LoRA output is 0 (PEFT convention)
        nn.init.zeros_(self.lora_embedding_A)
        nn.init.normal_(self.lora_embedding_B)

        # Place on same device / dtype as base
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_embedding_A.data = self.lora_embedding_A.data.to(device=device, dtype=dtype)
        self.lora_embedding_B.data = self.lora_embedding_B.data.to(device=device, dtype=dtype)

    # -- forward (embedding lookup) ----------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        # LoRA: look up A rows for each token, project through B
        after_A = F.embedding(x, self.lora_embedding_A)   # (..., rank)
        lora_out = after_A @ self.lora_embedding_B         # (..., embed_dim)
        return result + lora_out * self.scaling

    # -- logits (use embedding weight as lm_head) --------------------------
    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute logits using embedding weight as output projection.

        Equivalent to ``F.linear(hidden, W) + LoRA_correction``
        where W = base_layer.weight.

        This is needed because joint_model uses embed_tokens.weight
        directly for logits via ``F.linear(h, weight)``, bypassing any
        separate lm_head module.
        """
        # Cast hidden to weight dtype (bf16) instead of casting the large
        # weight (257152×2048) to float32 — avoids a ~2 GB temporary copy
        # that would be held alive by the autograd graph until backward.
        # Logits precision is fine in bf16; cross-entropy handles fp32 internally.
        out_dtype = hidden.dtype
        w_dtype = self.base_layer.weight.dtype
        h = hidden.to(w_dtype) if hidden.dtype != w_dtype else hidden
        base = F.linear(h, self.base_layer.weight)
        # LoRA correction: hidden @ B^T @ A^T * scaling
        intermediate = F.linear(h, self.lora_embedding_B)    # (..., rank)
        lora = F.linear(intermediate, self.lora_embedding_A)      # (..., vocab)
        return (base + lora * self.scaling).to(out_dtype)

    def compute_logits_slice(self, hidden: torch.Tensor, lo: int, count: int) -> torch.Tensor:
        """Logits for the contiguous vocabulary rows ``[lo, lo + count)``.

        Equal to ``compute_logits(hidden)[..., lo : lo + count]`` but without
        materialising the full 257k-wide projection.  Only ``lora_embedding_A``
        is vocabulary-indexed, so it narrows exactly like the base weight;
        ``lora_embedding_B`` is shared and must not be sliced.

        Used by ``openpi.waypoint.planner_decode`` for constrained waypoint
        decoding, where at most 300 of the 257 152 rows are ever legal.
        """
        out_dtype = hidden.dtype
        base_weight = self.base_layer.weight.narrow(0, lo, count)
        h = hidden.to(base_weight.dtype) if hidden.dtype != base_weight.dtype else hidden
        base = F.linear(h, base_weight)
        intermediate = F.linear(h, self.lora_embedding_B)                  # (..., rank)
        lora = F.linear(intermediate, self.lora_embedding_A.narrow(0, lo, count))
        return (base + lora * self.scaling).to(out_dtype)

    # -- merge for inference -----------------------------------------------
    def merge(self) -> None:
        """Merge LoRA into base embedding weight.

        Computes delta in float32 (see LoRALinear.merge docstring).
        """
        with torch.no_grad():
            A = self.lora_embedding_A.to(torch.float32)
            B = self.lora_embedding_B.to(torch.float32)
            delta = (A @ B) * self.scaling
            self.base_layer.weight.data += delta.to(self.base_layer.weight.dtype)

    # -- convenience properties -------------------------------------------
    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def num_embeddings(self) -> int:
        return self.base_layer.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.base_layer.embedding_dim

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"lora_rank={self.config.rank}, lora_alpha={self.config.alpha}"
        )


# ---------------------------------------------------------------------------
# Tied output head (PaliGemma has no separate lm_head)
# ---------------------------------------------------------------------------


def compute_lm_logits(embed_layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Project ``hidden`` through the (weight-tied) token embedding head.

    PaliGemma ties the output projection to ``embed_tokens``, so every caller is
    tempted to write ``F.linear(h, embed_tokens.weight)``.  That is wrong as soon
    as LoRA is applied: ``apply_lora_to_model`` *replaces* the module with
    :class:`LoRAEmbedding`, whose ``.weight`` property deliberately returns the
    untouched ``base_layer.weight``.  The A/B factors then reach the *input*
    lookup (which goes through ``forward``) but not the output head -- with no
    error, because the shape and dtype are still right.  Worse, ``merge()``
    folds A@B into the base weight, so the very same ``.weight`` expression
    *starts* including LoRA after a merge: active and merged become structurally
    different models rather than bf16-rounding apart.

    Every logits site must therefore go through this one function.
    """
    fn = getattr(embed_layer, "compute_logits", None)
    if fn is not None:
        return fn(hidden)
    # Cast hidden to the weight dtype rather than the weight to float32: a
    # float32 copy of a 257152x2048 embedding is ~2 GB and autograd keeps it
    # alive until backward.
    w = embed_layer.weight
    out_dtype = hidden.dtype
    h = hidden.to(w.dtype) if hidden.dtype != w.dtype else hidden
    return F.linear(h, w).to(out_dtype)


# ---------------------------------------------------------------------------
# Training-level config
# ---------------------------------------------------------------------------

# Default modules to NOT apply LoRA to (full-train or freeze instead).
# Reasoning for each entry — see docs/03-training.md.
_DEFAULT_MODULES_TO_NOT_LORA: list[str] = [
    # AE-specific projections: input dim too small (32) for LoRA to compress
    "action_in_proj",
    "action_out_proj",
    "proprio_encoder",
    # Time conditioning MLPs: new layers in joint model, must full-train
    "time_mlp_in",
    "time_mlp_out",
    # Goal-conditioning adapter (docs/14 §5.1, v2 in docs/16): small new
    # Linears, full-train; LoRA-wrapping a zero-init layer would also break its
    # zero-output contract.  The v2 null embeddings (null_residual /
    # null_end_embed) are nn.Parameters nested under goal_encoder, so this one
    # substring covers them in both lists.
    "goal_encoder",
    # Vision-language bridge: single small layer
    "multi_modal_projector",
    # NOTE: embed_tokens is NOT here — it gets LoRA via LoRAEmbedding.
    # NOTE: duration_embed is NOT here — nn.Embedding layers that do not
    # end with "embed_tokens" are already skipped by apply_lora_to_model,
    # so they're never LoRA-wrapped.  It's listed in
    # _DEFAULT_TRAINABLE_NON_LORA below to stay trainable.
    # lm_head (nn.Linear) is weight-tied with embed_tokens (same tensor).
    # Joint model never calls lm_head.forward(); logits go through
    # LoRAEmbedding.compute_logits() instead.  LoRA on lm_head would
    # allocate dead params and break weight tying.
    "lm_head",
]

# Default modules among the skip-list that should remain trainable
# (the rest are frozen).
_DEFAULT_TRAINABLE_NON_LORA: list[str] = [
    "action_in_proj",
    "action_out_proj",
    "proprio_encoder",
    "time_mlp_in",
    "time_mlp_out",
    # New nn.Embedding(DURATION_N_BINS, W) layer in joint model; must
    # full-train from random init (not present in Pi0.5 base).
    "duration_embed",
    "multi_modal_projector",
    # Waypoint-block planner (query residual + slot/block embeddings).  Present
    # only when planner_mode="block_ar"; randomly/zero initialised, absent from
    # the Pi0.5 base, and the block decoder is meaningless without it — so it
    # must stay trainable under LoRA rather than falling through to "frozen".
    "block_planner",
    # Goal-conditioning adapter: present only when goal_condition_mode is
    # "adarms_residual_v1"/"adarms_residual_v2"; new module, absent from Pi0.5
    # base, must train.  Covers the v2 null embeddings too (nested parameters).
    "goal_encoder",
    # NOTE: embed_tokens is NOT here — handled by LoRAEmbedding.
    # lm_head is NOT here — frozen (tied to embed_tokens base weight,
    # logits go through LoRAEmbedding.compute_logits()).
]


@dataclass
class LoRATrainingConfig:
    """Training-level LoRA configuration for the Waypoint Joint Model."""

    enabled: bool = False

    # LoRA hyper-parameters (applied uniformly to all LoRA'd layers)
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    use_rslora: bool = False
    # Initialisation for lora_A: True/"kaiming" or "gaussian"
    init_lora_weights: bool | str = True

    # Skip-list: full path substrings — any nn.Linear whose full
    # ``named_modules()`` path contains one of these strings is **not**
    # wrapped with LoRA.
    modules_to_not_lora: list[str] = field(
        default_factory=lambda: list(_DEFAULT_MODULES_TO_NOT_LORA),
    )

    # Among skipped modules, which should remain trainable (the rest freeze).
    trainable_non_lora_modules: list[str] = field(
        default_factory=lambda: list(_DEFAULT_TRAINABLE_NON_LORA),
    )

    # Vision encoder (SigLIP) handling
    vision_encoder_mode: Literal["freeze", "lora", "full"] = "freeze"

    # Which sub-models to apply LoRA to
    apply_to: Literal["all", "backbone_only", "expert_only"] = "all"

    def get_lora_config(self) -> LoRAConfig:
        return LoRAConfig(
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            use_rslora=self.use_rslora,
            init_lora_weights=self.init_lora_weights,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_parent_and_attr(model: nn.Module, dotted_name: str):
    """Return (parent_module, attr_name) for a dotted module path."""
    parts = dotted_name.rsplit(".", 1)
    if len(parts) == 2:
        return model.get_submodule(parts[0]), parts[1]
    return model, dotted_name


def _is_vision_tower(name: str) -> bool:
    return "vision_tower" in name or "vision_model" in name


def _should_skip_lora(name: str, config: LoRATrainingConfig) -> bool:
    """Return True if this module path should NOT be wrapped with LoRA."""
    # Check skip-list
    for pattern in config.modules_to_not_lora:
        if pattern in name:
            return True

    # Vision tower: skip unless vision_encoder_mode == "lora"
    if _is_vision_tower(name) and config.vision_encoder_mode != "lora":
        return True

    # apply_to filter.
    #
    # These must match on *path components*, not on raw substrings.  The joint
    # model mounts both towers under `paligemma_with_expert`, so the unanchored
    # test `"paligemma" in name` matched the expert's own parameters as well
    # (`paligemma_with_expert.gemma_expert...`) and `expert_only` therefore
    # skipped everything, silently producing a run with no LoRA on the expert it
    # was supposed to train.
    parts = name.split(".")
    if config.apply_to == "backbone_only":
        if "gemma_expert" in parts:
            return True
    elif config.apply_to == "expert_only":
        if "paligemma" in parts:
            return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    model: nn.Module,
    config: LoRATrainingConfig,
) -> tuple[int, int]:
    """Apply LoRA to a PI0WaypointJoint model.

    Every ``nn.Linear`` whose full path does **not** match any entry in
    ``config.modules_to_not_lora`` (and passes the vision / apply_to
    filters) gets wrapped with ``LoRALinear``.

    After wrapping, parameters are frozen / unfrozen according to
    ``config.trainable_non_lora_modules`` and ``config.vision_encoder_mode``.

    Returns:
        (frozen_param_count, trainable_param_count)
    """
    if not config.enabled:
        logger.info("LoRA is disabled, skipping")
        total = sum(p.numel() for p in model.parameters())
        return 0, total

    lora_config = config.get_lora_config()
    lora_count = 0
    lora_details: list[str] = []

    # Phase 1: wrap layers with LoRA wrappers
    # Use list() to snapshot because we mutate the tree in-place.
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if _should_skip_lora(name, config):
                continue
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, LoRALinear(module, lora_config))
            lora_count += 1
            lora_details.append(
                f"  LoRA Linear: {name}  ({module.in_features}→{module.out_features})"
            )
        elif isinstance(module, nn.Embedding):
            # Only LoRA embed_tokens, not position/patch embeddings
            if not name.endswith("embed_tokens"):
                continue
            if _should_skip_lora(name, config):
                continue
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, LoRAEmbedding(module, lora_config))
            lora_count += 1
            lora_details.append(
                f"  LoRA Embedding: {name}  ({module.num_embeddings}×{module.embedding_dim})"
            )

    logger.info(f"Applied LoRA (rank={lora_config.rank}) to {lora_count} layers:")
    for line in lora_details:
        logger.info(line)

    # Phase 2: freeze / unfreeze
    frozen, trainable = _freeze_for_lora(model, config)
    return frozen, trainable


def _freeze_for_lora(
    model: nn.Module,
    config: LoRATrainingConfig,
) -> tuple[int, int]:
    """Set requires_grad for LoRA training.

    Rules (in priority order):
      1. LoRA params (lora_A, lora_B)  → trainable
      2. Vision tower                   → depends on vision_encoder_mode
      3. Params matching trainable_non_lora_modules → trainable
      4. Everything else                → frozen
    """
    trainable_set = set(config.trainable_non_lora_modules)

    frozen_count = 0
    trainable_count = 0
    vision_count = 0

    for name, param in model.named_parameters():
        should_train = False

        # Rule 1: LoRA params always trainable
        if "lora_A" in name or "lora_B" in name or "lora_embedding" in name:
            should_train = True

        # Rule 2: vision tower
        elif _is_vision_tower(name):
            if config.vision_encoder_mode == "full":
                should_train = True
            elif config.vision_encoder_mode == "lora":
                # LoRA params handled by rule 1; base weights stay frozen
                should_train = False
            else:  # "freeze"
                should_train = False
            if should_train:
                vision_count += param.numel()

        # Rule 3: trainable non-LoRA modules
        elif any(pattern in name for pattern in trainable_set):
            should_train = True

        param.requires_grad = should_train
        if should_train:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    total = trainable_count + frozen_count
    ratio = trainable_count / total * 100 if total > 0 else 0
    logger.info(
        f"LoRA freeze: {trainable_count:,} trainable, "
        f"{frozen_count:,} frozen  ({ratio:.2f}% trainable)"
    )
    if config.vision_encoder_mode == "freeze":
        logger.info("  Vision encoder (SigLIP): FROZEN")
    elif config.vision_encoder_mode == "full":
        logger.info(f"  Vision encoder (SigLIP): FULL-TRAIN ({vision_count:,} params)")
    else:
        logger.info("  Vision encoder (SigLIP): LoRA-only")

    return frozen_count, trainable_count


# ---------------------------------------------------------------------------
# LoRA-only save / load
# ---------------------------------------------------------------------------

def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect only trainable parameters (LoRA + non-LoRA trainable modules).

    This is the minimal set needed to resume LoRA training or run inference
    (after merging LoRA weights back into the base model).
    """
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data
    return state


def save_lora_checkpoint(
    model: nn.Module,
    path: str,
    *,
    use_safetensors: bool = True,
) -> None:
    """Save only the trainable (LoRA + non-LoRA) parameters.

    Args:
        model: The model (possibly wrapped in DDP).
        path: Output file path (.safetensors or .pt).
        use_safetensors: If True, use safetensors format (recommended).
    """
    # Unwrap DDP if necessary
    raw = model.module if hasattr(model, "module") else model
    state = get_lora_state_dict(raw)

    if use_safetensors:
        safetensors_save(state, path)
    else:
        torch.save(state, path)

    logger.info(f"Saved LoRA checkpoint ({len(state)} tensors) to {path}")


# Model sub-modules whose presence/absence and tensor shapes are a *schema*:
# a LoRA checkpoint must carry exactly the model's tensors under these prefixes
# (no missing, no stray, no shape drift).  ``goal_encoder`` has its own
# per-mode check below; ``block_planner`` is handled by
# ``_check_block_planner_schema``.
BLOCK_PLANNER_PREFIX = "block_planner."


def _check_block_planner_schema(raw: nn.Module, state: dict, path: str) -> None:
    """Fail closed on any block-planner schema drift between model and checkpoint.

    Both directions and the shape are errors, and every one of them used to be
    a warning at most:

    * model has a ``block_planner`` but the checkpoint carries no
      ``block_planner.*`` tensor -> the model would train/evaluate with a fresh
      zero-init planner while the operator believes it warm-started;
    * checkpoint carries ``block_planner.*`` but the model is token-AR -> the
      trained planner would be silently dropped;
    * same keys, different shapes (a P=6 planner loaded into a P=14 model) ->
      the generic loop below used to *skip* the tensor with a warning.
    """
    model_has = getattr(raw, "block_planner", None) is not None
    model_keys = {k for k in raw.state_dict() if k.startswith(BLOCK_PLANNER_PREFIX)}
    ckpt_keys = {k for k in state if k.startswith(BLOCK_PLANNER_PREFIX)}
    if model_has and not ckpt_keys:
        raise RuntimeError(
            f"this model was built with planner_mode='block_ar' but {path} contains no "
            f"{BLOCK_PLANNER_PREFIX}* tensor: it is a token-AR checkpoint (or a planner-less "
            "merge). Loading it would leave the block planner at its zero init while "
            "claiming a warm start. Refusing."
        )
    if ckpt_keys and not model_has:
        raise RuntimeError(
            f"{path} carries trained {sorted(ckpt_keys)} but the model was built with "
            "planner_mode='token_ar' (block_planner=None); loading would silently drop "
            "the trained planner. Rebuild with the planner_mode recorded in the "
            "checkpoint metadata."
        )
    if not model_has:
        return
    missing = sorted(model_keys - ckpt_keys)
    stray = sorted(ckpt_keys - model_keys)
    if missing or stray:
        raise RuntimeError(
            f"block planner schema mismatch loading {path}: the model expects planner "
            f"keys the checkpoint lacks {missing}; the checkpoint carries planner keys "
            f"the model lacks {stray} (planner_use_block_embed disagreement?)."
        )
    model_state = raw.state_dict()
    for key in sorted(ckpt_keys):
        if tuple(model_state[key].shape) != tuple(state[key].shape):
            raise RuntimeError(
                f"block planner geometry mismatch loading {path}: {key} is "
                f"{tuple(state[key].shape)} in the checkpoint but {tuple(model_state[key].shape)} "
                "in the model. The planner slot rows are indexed by "
                "1 + proprio_dim + gripper_slots + 2, so this checkpoint was trained for a "
                "different robot layout; rebuild the model with the geometry recorded in "
                "its metadata.pt (planner_proprio_dim / planner_use_gripper_token / "
                "planner_num_waypoints / planner_block_width)."
            )


def load_lora_checkpoint(
    model: nn.Module,
    path: str,
    *,
    allow_shape_mismatch: tuple[str, ...] = (),
) -> None:
    """Load a LoRA checkpoint (trainable params only) into *model*.

    The model must already have LoRA layers injected (via ``apply_lora_to_model``).
    Keys the model has but the checkpoint lacks keep their init values (a fresh
    LoRA start from base weights); keys the checkpoint has but the model lacks
    are reported as *unexpected* (warning).  Both of those are silent for the
    ``block_planner`` / ``goal_encoder`` schemas, which fail closed instead.

    Shapes are **strict**: a tensor present in both with different shapes
    raises, because the historical "skip with a warning" left the model with
    its init in exactly the place the operator believed was warm-started.
    ``allow_shape_mismatch`` names key prefixes for which a mismatch may be
    skipped (warning) instead -- an explicit opt-in, empty by default.

    Args:
        model: The model with LoRA layers already applied.
        path: Path to ``.safetensors`` or ``.pt`` file.
        allow_shape_mismatch: key prefixes whose shape mismatch is skipped.
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)

    # Fail-closed goal-adapter schema check (docs/14 §5.1): a goal-conditioned
    # model must never silently keep its fresh adapter because the checkpoint
    # predates the feature, and a mode="none" model must never silently drop a
    # checkpoint's trained adapter.  The generic missing/unexpected handling
    # below only warns, which is exactly the silent-degradation path docs/14
    # forbids.
    raw = model.module if hasattr(model, "module") else model
    model_has_goal = getattr(raw, "goal_encoder", None) is not None
    ckpt_has_goal = any("goal_encoder." in key for key in state)
    if model_has_goal and not ckpt_has_goal:
        raise RuntimeError(
            f"goal_condition_mode is enabled on this model but {path} contains no "
            "goal_encoder.* tensors — this checkpoint was not trained with goal "
            "conditioning. Refusing to load it into a goal-conditioned model."
        )
    if ckpt_has_goal and not model_has_goal:
        raise RuntimeError(
            f"{path} contains trained goal_encoder.* tensors but the model was "
            "built with goal_condition_mode='none' — loading would silently "
            "evaluate the checkpoint without its goal adapter."
        )
    if model_has_goal and ckpt_has_goal:
        # Per-key schema check: v2 adds goal_encoder.null_* parameters, so a v1
        # checkpoint loaded into a v2 model would silently keep fresh nulls (and
        # the reverse would silently drop trained ones) if we only checked
        # presence.  Missing is judged against *parameters* (the delta_scale
        # buffer is intentionally not in lora.safetensors — it travels via
        # checkpoint metadata); stray is judged against the full state_dict (a
        # checkpoint that does carry the buffer is not a schema violation).
        model_goal_params = {
            name for name, _ in raw.named_parameters() if "goal_encoder." in name
        }
        model_goal_state = {key for key in raw.state_dict() if "goal_encoder." in key}
        ckpt_goal_keys = {key for key in state if "goal_encoder." in key}
        missing_goal = sorted(model_goal_params - ckpt_goal_keys)
        stray_goal = sorted(ckpt_goal_keys - model_goal_state)
        if missing_goal or stray_goal:
            raise RuntimeError(
                f"goal adapter schema mismatch loading {path}: the model expects "
                f"goal keys the checkpoint lacks {missing_goal}; the checkpoint "
                f"carries goal keys the model lacks {stray_goal}. A v1 checkpoint "
                "cannot load into a v2 model or vice versa — rebuild the model "
                "with the goal_condition_mode recorded in the checkpoint metadata."
            )

    _check_block_planner_schema(raw, state, path)

    missing, unexpected = [], []
    model_state = model.state_dict()
    for key, val in state.items():
        if key in model_state:
            if model_state[key].shape == val.shape:
                model_state[key].copy_(val)
            elif allow_shape_mismatch and key.startswith(allow_shape_mismatch):
                logger.warning(
                    f"Shape mismatch for {key}: model={tuple(model_state[key].shape)}, "
                    f"ckpt={tuple(val.shape)} — skipping (explicitly allowed)"
                )
                missing.append(key)
            else:
                raise RuntimeError(
                    f"shape mismatch loading {key} from {path}: checkpoint "
                    f"{tuple(val.shape)} vs model {tuple(model_state[key].shape)}. "
                    "Skipping it would leave this tensor at its init while reporting a "
                    "warm start; rebuild the model with the checkpoint's geometry, or "
                    "pass allow_shape_mismatch=(prefix,) to skip it deliberately."
                )
        else:
            unexpected.append(key)

    if missing:
        logger.warning(f"Missing / shape-mismatch keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in LoRA checkpoint: {unexpected}")
    logger.info(
        f"Loaded LoRA checkpoint from {path} "
        f"({len(state) - len(missing) - len(unexpected)} tensors applied)"
    )


# ---------------------------------------------------------------------------
# Inference-time slim forward (zero-drift, algebraically exact speedup)
# ---------------------------------------------------------------------------

def _slim_lora_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Inference-only slim forward for LoRALinear.

    Drops the per-call ``dtype`` safety cast and the ``* scaling`` op.
    Correctness is preserved because:
      * ``install_inference_slim_forward`` folds ``scaling`` into
        ``lora_B.weight`` (``B <- B * s``, ``scaling = 1.0``), which is
        algebraically exact:   ``B @ (A x) * s  ==  (s B) @ (A x)``.
      * ``install_inference_slim_forward`` also asserts ``lora_A.weight.dtype``
        equals ``base_layer.weight.dtype``, so the runtime dtype branch is dead.
    """
    lora_input = self.lora_dropout(x)
    lora_out = self.lora_B(self.lora_A(lora_input))
    return self.base_layer(x) + lora_out


def install_inference_slim_forward(model: nn.Module) -> int:
    """Zero-drift inference speedup for LoRA-active forward passes.

    For every ``LoRALinear`` in ``model``:
      1. Pre-scale ``lora_B.weight`` by ``self.scaling`` and set
         ``self.scaling = 1.0``.  This is algebraically exact
         (``(s·B)(A x) == s · B(A x)``), so **zero numerical drift**.
      2. Replace ``forward`` with a slim version that skips the defensive
         ``if lora_input.dtype != ...`` cast branch and the ``* scaling`` op.
         We first assert dtype consistency so the skipped branch is truly dead.

    Why this helps:
      * fewer ops per layer (2 elementwise ops removed) → smaller compiled
        graph, better kernel fusion under ``torch.compile``;
      * the ``if ... dtype ...`` branch caused ``torch.compile`` to emit
        dynamic-shape specialisation guards that are unnecessary at inference.

    Call **after** ``load_lora_checkpoint`` and **before** ``torch.compile``.
    Only applied to ``LoRALinear`` (the hot path in the denoising loop);
    ``LoRAEmbedding`` is left untouched because it runs at most once per
    sample_actions call.

    Returns:
        Number of layers patched.
    """
    n = 0
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            assert mod.lora_A.weight.dtype == mod.base_layer.weight.dtype, (
                f"install_inference_slim_forward requires lora_A dtype "
                f"({mod.lora_A.weight.dtype}) == base dtype "
                f"({mod.base_layer.weight.dtype})"
            )
            if mod.scaling != 1.0:
                with torch.no_grad():
                    mod.lora_B.weight.data.mul_(mod.scaling)
                mod.scaling = 1.0
            mod.forward = _slim_lora_linear_forward.__get__(mod, LoRALinear)
            n += 1
    if n > 0:
        logger.info(
            f"Installed inference-slim forward on {n} LoRALinear layers "
            f"(zero drift; dropped dtype-cast and *scaling ops)."
        )
    return n


# ---------------------------------------------------------------------------
# Merge all LoRA layers (for inference deployment)
# ---------------------------------------------------------------------------

def merge_lora_weights(model: nn.Module) -> int:
    """Merge all LoRA layers (LoRALinear + LoRAEmbedding) back into base layers.

    After merging, the model behaves identically to a fully fine-tuned
    model with no LoRA overhead.  This is **irreversible** unless you
    saved the LoRA checkpoint separately.

    Returns:
        Number of layers merged.
    """
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (LoRALinear, LoRAEmbedding)):
            module.merge()
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, module.base_layer)
            count += 1
    logger.info(f"Merged {count} LoRA layers into base weights")
    return count


# ---------------------------------------------------------------------------
# Utility: print model LoRA summary
# ---------------------------------------------------------------------------

def print_lora_summary(model: nn.Module) -> None:
    """Print a summary of LoRA layers and trainable parameter counts."""
    lora_params = 0
    trainable_non_lora = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "lora_embedding" in name:
            lora_params += param.numel()
        elif param.requires_grad:
            trainable_non_lora += param.numel()
        else:
            frozen_params += param.numel()

    total = lora_params + trainable_non_lora + frozen_params
    logger.info("=" * 60)
    logger.info("LoRA Summary")
    logger.info(f"  LoRA params:            {lora_params:>12,}")
    logger.info(f"  Trainable non-LoRA:     {trainable_non_lora:>12,}")
    logger.info(f"  Frozen params:          {frozen_params:>12,}")
    logger.info(f"  Total params:           {total:>12,}")
    logger.info(f"  Trainable ratio:        {(lora_params + trainable_non_lora) / total * 100:.2f}%")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Shared config builder: YAML dict → LoRATrainingConfig
# ---------------------------------------------------------------------------

def build_lora_training_config(cfg: dict) -> LoRATrainingConfig:
    """Build a ``LoRATrainingConfig`` from a flat YAML config dict.

    This centralises the logic duplicated in train_waypoint.py,
    train_waypoint_joint.py, merge_lora.py, and eval_libero.py.
    """
    vision_mode = cfg.get("vision_encoder_mode", None)
    if vision_mode is None:
        vision_mode = "full" if cfg.get("train_vision_encoder", False) else "freeze"

    kwargs: dict = dict(
        enabled=True,
        rank=cfg.get("lora_rank", 16),
        alpha=cfg.get("lora_alpha", 16.0),
        dropout=cfg.get("lora_dropout", 0.0),
        use_rslora=cfg.get("lora_use_rslora", False),
        init_lora_weights=cfg.get("lora_init", True),
        vision_encoder_mode=vision_mode,
        apply_to=cfg.get("lora_apply_to", "all"),
    )
    if "lora_modules_to_skip" in cfg:
        kwargs["modules_to_not_lora"] = cfg["lora_modules_to_skip"]
    if "lora_trainable_modules" in cfg:
        kwargs["trainable_non_lora_modules"] = cfg["lora_trainable_modules"]
    return LoRATrainingConfig(**kwargs)


# ---------------------------------------------------------------------------
# Checkpoint-type detection helpers
# ---------------------------------------------------------------------------

_MERGE_DRIFT_WARNING = """
╔══════════════════════════════════════════════════════════════════════╗
║  ⚠️  WARNING: LOADING A MERGED LoRA CHECKPOINT                     ║
║                                                                      ║
║  The checkpoint at                                                   ║
║    {path}
║  contains a merged model.safetensors (LoRA folded into base).        ║
║                                                                      ║
║  LoRA merge introduces UNAVOIDABLE numerical drift in bfloat16       ║
║  due to different matmul accumulation order vs LoRA-active forward.  ║
║  (See torchtune#1101, PEFT#1836)                                     ║
║                                                                      ║
║  Recommended: use 'resume_lora_from' to load the LoRA checkpoint    ║
║  directly (zero drift).  Set pretrained_weight_path to the ORIGINAL  ║
║  base model and resume_lora_from to the lora.safetensors path.       ║
╚══════════════════════════════════════════════════════════════════════╝
""".strip()


def detect_checkpoint_type(ckpt_dir: str) -> str:
    """Detect the type of checkpoint in a directory.

    Returns:
        ``"lora"`` — ``lora.safetensors`` exists (prefer LoRA-active load)
        ``"merged"`` — only ``model.safetensors`` exists (merged / full-train)
        ``"both"`` — both exist (prefer LoRA-active)
        ``"empty"`` — neither exists
    """
    import os
    has_lora = os.path.isfile(os.path.join(ckpt_dir, "lora.safetensors"))
    has_model = os.path.isfile(os.path.join(ckpt_dir, "model.safetensors"))
    if has_lora and has_model:
        return "both"
    if has_lora:
        return "lora"
    if has_model:
        return "merged"
    return "empty"


def warn_merged_checkpoint(ckpt_path: str) -> None:
    """Emit a prominent warning that a merged checkpoint is being loaded."""
    logger.warning(_MERGE_DRIFT_WARNING.format(path=ckpt_path))

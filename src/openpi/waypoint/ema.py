"""Exponential Moving Average (EMA) of trainable parameters.

Maintains a shadow copy of every ``requires_grad=True`` parameter and updates
it after each optimizer step::

    shadow = decay * shadow + (1 - decay) * param

Design choices
--------------
* **Shadow stays in float32** (regardless of training dtype) to avoid
  bfloat16 accumulation drift over thousands of steps.  At save time the
  shadow is cast back to the model's dtype so the resulting checkpoint is
  byte-compatible with the regular LoRA checkpoint loader.
* **Tracks only ``requires_grad`` params**, matching ``get_lora_state_dict``.
  In LoRA mode this means LoRA A/B + non-LoRA trainable modules; the frozen
  base weights are loaded fresh at eval time, so EMA-averaging them is
  unnecessary (and wrong: their value is constant, EMA == identity).
* **DDP-safe**: every rank keeps its own shadow.  Because DDP guarantees
  identical params across ranks after each optimizer step, all shadows are
  identical too.  Save only from rank 0.
* **Optional warmup**: during the first ``warmup_steps`` the shadow simply
  copies the latest params (no averaging).  This sidesteps the well-known
  "EMA is biased toward initialisation" problem that would otherwise leave
  the shadow stuck near random LoRA init for thousands of steps.

Usage
-----
::

    ema = EMAState(model, decay=0.9999, warmup_steps=2000)

    for step in range(num_steps):
        ...
        optimizer.step()
        ema.update(model, step=step)

    # At checkpoint time (rank 0 only):
    ema.save_lora_safetensors(ckpt_dir / "lora_ema.safetensors", model)
"""

from __future__ import annotations

from collections.abc import Iterable
import logging
import math
from pathlib import Path

from safetensors.torch import save_file as safetensors_save
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# A finite EMA run is not a stationary ``1 / (1 - decay)`` window.  The
# parameter snapshot at the end of warmup keeps a point mass of
# ``decay ** (num_train_steps - warmup_steps)``.  A large point mass is useful
# for a deliberate forensic reproduction, but is a dangerous production
# default: it can turn a final checkpoint into an early-training model.
EMA_STALE_WARN_THRESHOLD = 0.10
EMA_STALE_ERROR_THRESHOLD = 0.50
EMA_FORENSIC_OVERRIDE_KEY = "allow_stale_ema_forensics"


def ema_retained_anchor_weight(
    decay: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Return the final weight of the last warmup-copy snapshot.

    ``EMAState.update`` copies the current parameters while
    ``step < warmup_steps``.  With the trainer's zero-based step counter, the
    last copied snapshot is therefore checkpoint step ``warmup_steps`` and it
    is followed by ``total_steps - warmup_steps`` averaging updates.

    If training ends during warmup, the final EMA is an exact copy of the final
    raw parameters.  There is no *stale* anchor in that case, so this function
    returns zero rather than the algebraic point mass of one at the current
    (non-stale) parameters.
    """
    decay = float(decay)
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    if not (0.0 < decay < 1.0):
        raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
    if warmup_steps < 0:
        raise ValueError(f"EMA warmup_steps must be >= 0, got {warmup_steps}")
    if total_steps < 0:
        raise ValueError(f"EMA total_steps must be >= 0, got {total_steps}")
    averaging_updates = total_steps - warmup_steps
    if averaging_updates <= 0:
        return 0.0
    # exp/log is stable for decays close to one and long training horizons.
    return math.exp(averaging_updates * math.log(decay))


def validate_ema_staleness(
    *,
    decay: float,
    warmup_steps: int,
    total_steps: int,
    allow_stale_ema_forensics: bool = False,
    context: str = "EMA",
) -> float:
    """Warn on a mildly stale EMA and reject a severely stale one.

    The sole escape hatch is intentionally named for forensic reproduction.
    It must be the boolean ``True``; truthy strings such as ``"false"`` are
    rejected so a YAML quoting mistake cannot silently disable the guard.

    Returns:
        The retained warmup-anchor weight.
    """
    if not isinstance(allow_stale_ema_forensics, bool):
        raise TypeError(
            f"{EMA_FORENSIC_OVERRIDE_KEY} must be a boolean, got "
            f"{allow_stale_ema_forensics!r}"
        )

    retained = ema_retained_anchor_weight(decay, warmup_steps, total_steps)
    averaging_updates = max(int(total_steps) - int(warmup_steps), 0)
    detail = (
        f"{context}: decay={float(decay):.8g}, warmup_steps={int(warmup_steps)}, "
        f"total_steps={int(total_steps)} retains {retained:.2%} of the "
        f"warmup-boundary parameters after {averaging_updates} averaging updates"
    )

    if retained > EMA_STALE_ERROR_THRESHOLD:
        message = (
            f"{detail}, exceeding the severe-staleness threshold "
            f"({EMA_STALE_ERROR_THRESHOLD:.0%}). This EMA is dominated by an "
            "early checkpoint. Use raw weights or choose a shorter EMA horizon. "
            f"Only an intentional forensic reproduction may set "
            f"`{EMA_FORENSIC_OVERRIDE_KEY}: true`."
        )
        if not allow_stale_ema_forensics:
            raise ValueError(message)
        logger.warning("%s Forensic override accepted; do not use this as a production default.", message)
    elif retained > EMA_STALE_WARN_THRESHOLD:
        logger.warning(
            "%s, exceeding the mild-staleness threshold (%.0f%%). Validate raw and EMA "
            "on paired multi-seed rollouts before selecting EMA.",
            detail,
            EMA_STALE_WARN_THRESHOLD * 100,
        )
    else:
        logger.info("%s", detail)
    return retained


def _unwrap(model: nn.Module) -> nn.Module:
    """Strip DDP / compile wrappers to get clean parameter names."""
    raw = model
    if hasattr(raw, "module"):
        raw = raw.module
    if hasattr(raw, "_orig_mod"):  # torch.compile
        raw = raw._orig_mod
    return raw


def _trainable_named_params(model: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield (name, param) for parameters with requires_grad=True."""
    raw = _unwrap(model)
    for name, param in raw.named_parameters():
        if param.requires_grad:
            yield name, param


class EMAState:
    """Maintain an EMA shadow of every trainable parameter.

    Args:
        model: Model to mirror.  Iterates ``requires_grad=True`` params.
        decay: Target EMA decay (e.g. 0.9999 ≈ averaging window of 10 000 steps).
        warmup_steps: For the first ``warmup_steps`` updates, copy params
            directly into the shadow (no averaging).  Set to 0 to start
            averaging immediately.  Defaults to 0.
        dtype: Storage dtype of the shadow.  ``float32`` (default) avoids
            bfloat16 drift; ``bfloat16`` saves memory at the cost of precision.
        device: Where to keep the shadow.  Defaults to the same device as
            each tracked parameter.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.dtype = dtype
        self._device_override = torch.device(device) if device is not None else None

        self.shadow: dict[str, torch.Tensor] = {}
        self.num_updates: int = 0

        for name, param in _trainable_named_params(model):
            target_device = self._device_override or param.device
            self.shadow[name] = (
                param.data.detach().to(dtype=dtype, device=target_device).clone()
            )

        n_params = sum(s.numel() for s in self.shadow.values())
        bytes_each = torch.tensor([], dtype=dtype).element_size()
        mb = n_params * bytes_each / 1024**2
        logger.info(
            f"EMAState: tracking {len(self.shadow)} tensors "
            f"({n_params:,} params, ~{mb:.1f} MB in {dtype}), "
            f"decay={decay}, warmup_steps={warmup_steps}"
        )

    @torch.no_grad()
    def update(self, model: nn.Module, step: int | None = None) -> float:
        """Pull the latest params into the shadow.

        Args:
            model: The trainable model (DDP-wrapped or not).
            step: Optional global step.  If provided and ``step < warmup_steps``,
                the shadow is set equal to the current params (no averaging).
                If ``step is None``, ``self.num_updates`` is used.

        Returns:
            The effective decay used for this update (useful for logging).
        """
        s = step if step is not None else self.num_updates
        in_warmup = s < self.warmup_steps
        decay = 0.0 if in_warmup else self.decay
        one_minus = 1.0 - decay

        for name, param in _trainable_named_params(model):
            shadow = self.shadow.get(name)
            if shadow is None:
                # Could happen if requires_grad changed mid-training; rebuild lazily.
                target_device = self._device_override or param.device
                self.shadow[name] = (
                    param.data.detach().to(dtype=self.dtype, device=target_device).clone()
                )
                continue
            src = param.data
            if src.dtype != self.dtype or src.device != shadow.device:
                src = src.to(dtype=self.dtype, device=shadow.device)
            if in_warmup:
                shadow.copy_(src)
            else:
                # shadow = decay * shadow + (1 - decay) * src   (fused via lerp)
                # torch.lerp(input, end, weight) = input + weight * (end - input)
                #                                = (1 - weight) * input + weight * end
                shadow.lerp_(src, one_minus)

        self.num_updates += 1
        return decay

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(
        self,
        *,
        dst_dtype: torch.dtype | None = None,
        dst_dtypes: dict[str, torch.dtype] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a serialisable copy of the shadow.

        Args:
            dst_dtype: Optional single target dtype for every tensor.  Defaults
                to the shadow's storage dtype.
            dst_dtypes: Optional *per-tensor* target dtypes, which is what a
                mixed-precision model needs: the backbone LoRA adapters are
                bfloat16 while some AE heads are float32, so a single dtype
                (taken from whichever parameter happened to be enumerated
                first) silently downcasts the float32 ones.  Takes precedence
                over ``dst_dtype`` for keys it contains.
        """
        out = {}
        for k, v in self.shadow.items():
            t = v.detach()
            target = (dst_dtypes or {}).get(k, dst_dtype)
            if target is not None and t.dtype != target:
                t = t.to(dtype=target)
            out[k] = t.contiguous().clone()
        return out

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        num_updates: int = 0,
    ) -> None:
        for k, v in state_dict.items():
            if k not in self.shadow:
                logger.warning(f"EMA load_state_dict: unexpected key {k}, skipping")
                continue
            self.shadow[k].copy_(v.to(dtype=self.dtype, device=self.shadow[k].device))
        self.num_updates = int(num_updates)

    def save_lora_safetensors(
        self,
        path: str | Path,
        model: nn.Module | None = None,
    ) -> None:
        """Save the EMA shadow as a ``lora.safetensors``-compatible file.

        Each tensor is written back in **its own** parameter's dtype, so a
        bfloat16 LoRA adapter stays bfloat16 and a float32 AE head stays
        float32.  Deriving one dtype from the first trainable parameter (as this
        used to) downcast every float32 head to bfloat16, which is a silent
        precision loss that only shows up as a raw-vs-EMA eval discrepancy.
        """
        dst_dtypes: dict[str, torch.dtype] = {}
        fallback = self.dtype
        if model is not None:
            for name, p in _trainable_named_params(model):
                dst_dtypes[name] = p.dtype
            if dst_dtypes:
                fallback = next(iter(dst_dtypes.values()))
            missing = set(self.shadow) - set(dst_dtypes)
            if missing:
                logger.warning(
                    "EMA save: %d shadow tensor(s) have no matching trainable parameter "
                    "(e.g. %s); writing them as %s",
                    len(missing),
                    sorted(missing)[:3],
                    fallback,
                )

        state = self.state_dict(dst_dtype=fallback, dst_dtypes=dst_dtypes)
        path = str(path)
        safetensors_save(state, path)
        dtypes_used = sorted({str(t.dtype) for t in state.values()})
        logger.info(
            f"Saved EMA checkpoint ({len(state)} tensors, dtypes={dtypes_used}, "
            f"updates={self.num_updates}) to {path}"
        )

    # ------------------------------------------------------------------
    # Optional in-place swap (for evaluating EMA mid-training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite ``model``'s trainable params with the EMA shadow.

        Useful for in-process eval after training without restarting.
        Note: this is destructive; back up params first if you need them.
        """
        for name, param in _trainable_named_params(model):
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            param.data.copy_(shadow.to(dtype=param.dtype, device=param.device))

    @torch.no_grad()
    def store_and_swap_in(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Save current params, then load EMA into model.

        Returns a backup dict (suitable for ``swap_back``).
        """
        backup: dict[str, torch.Tensor] = {}
        for name, param in _trainable_named_params(model):
            backup[name] = param.data.detach().clone()
        self.copy_to(model)
        return backup

    @torch.no_grad()
    def swap_back(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, param in _trainable_named_params(model):
            saved = backup.get(name)
            if saved is not None:
                param.data.copy_(saved)


def build_ema_from_cfg(model: nn.Module, cfg: dict) -> EMAState | None:
    """Construct an EMAState from a training config dict.

    Recognised config keys (all optional):
      * ``ema_enabled`` (bool, default False)
      * ``ema_decay`` (float, default 0.9999)
      * ``ema_warmup_steps`` (int, default 0 — start averaging immediately)
      * ``ema_dtype`` ("float32" | "bfloat16" | "float16", default "float32")
      * ``ema_device`` (str | null, default null = same as params)
      * ``allow_stale_ema_forensics`` (bool, default False; forensic only)

    Returns:
        EMAState if ``ema_enabled`` is true, else ``None``.
    """
    if not cfg.get("ema_enabled", False):
        return None

    decay = float(cfg.get("ema_decay", 0.9999))
    warmup_steps = int(cfg.get("ema_warmup_steps", 0))
    # ``train_waypoint_joint.py`` defaults to 5k steps when the key is absent,
    # so use the same value here.  This makes the training path fail before it
    # allocates the shadow or consumes a single batch.
    total_steps = int(cfg.get("num_train_steps", 5000))
    validate_ema_staleness(
        decay=decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        allow_stale_ema_forensics=cfg.get(EMA_FORENSIC_OVERRIDE_KEY, False),
        context="training EMA plan",
    )

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    dtype_str = str(cfg.get("ema_dtype", "float32")).lower()
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown ema_dtype: {dtype_str}")

    return EMAState(
        model,
        decay=decay,
        warmup_steps=warmup_steps,
        dtype=dtype_map[dtype_str],
        device=cfg.get("ema_device", None),
    )

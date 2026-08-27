"""Two-stage LIBERO evaluation for Waypoint VLA.

Pipeline:
  1. VLM predicts M=7 waypoints autoregressively.
  2. Action Expert fills actions between each waypoint pair via flow matching.
  3. Execute actions in LIBERO environment.
  4. Replan after exhausting all waypoints.

Usage:
  python -m openpi.waypoint.eval_libero --config configs/eval_waypoint_libero.yaml
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
import time

import imageio
import numpy as np
import safetensors.torch
import torch
import yaml
from PIL import Image

try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
    torch.serialization.add_safe_globals([np.ndarray])
    torch.serialization.add_safe_globals([np.dtype])
except Exception:
    pass

from openpi.waypoint.normalize import (
    NormalizationHelper,
    load_dataset_statistics,
    pad_to_dim,
    unnormalize_q99,
    unnormalize_normal,
)
from openpi.waypoint.planner_decode import DECODE_IMPLS, WaypointDecodeConfig
from openpi.waypoint.robot_config import get_robot_config
from openpi.waypoint.tokenizer import WaypointTokenizer

logger = logging.getLogger(__name__)

_image_save_dir = None
_image_frame_idx = 0

MAX_STEPS_MAP = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def resolve_max_steps(cfg) -> int:
    """Suite step budget, with an opt-in override.

    ``eval.max_steps_override`` exists for one experiment: re-running an arm on a
    *larger* budget to test whether its failures are "too slow" or "wrong".  It is
    deliberately not a routine protocol knob -- every comparable number in
    ``docs/12-libero10-root-cause.md`` was measured at the suite default, so a run
    that sets this is a different protocol and has to be reported as one.

    The guard clause is the contract: with the key absent (or explicitly ``None``)
    this evaluates exactly ``MAX_STEPS_MAP.get(cfg.get("task_suite",
    "libero_object"), 280)`` -- the expression that used to be inlined at the call
    site -- so the default path never touches any of the new code.
    """
    default = MAX_STEPS_MAP.get(cfg.get("task_suite", "libero_object"), 280)
    override = cfg.get("max_steps_override")
    if override is None:
        return default
    override = int(override)
    if override <= 0:
        raise ValueError(
            f"eval.max_steps_override must be a positive step count, got {override!r}"
        )
    return override


# def quat2axisangle(quat):
#     """Convert quaternion to axis-angle."""
#     from transforms3d.quaternions import quat2axangle
#     axis, angle = quat2axangle(quat)
#     return (axis * angle).astype(np.float32)

def quat2axisangle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle, matching robosuite convention."""
    import math
    q = quat.copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)
    
def get_proprio_from_obs(obs):
    """Extract 8d LIBERO proprio: [EEF_pos(3), EEF_axisangle(3), gripper_qpos(2)]."""
    eef_pos = obs["robot0_eef_pos"]
    eef_rot = quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]
    return np.concatenate([eef_pos, eef_rot, gripper]).astype(np.float32)


def center_crop_and_resize(img_array: np.ndarray, crop_scale: float, target_size: int = 224) -> np.ndarray:
    """Center crop with area ratio ``crop_scale``, then resize back to ``target_size``.

    Inference-time deterministic equivalent of training's
    ``RandomResizedCrop(target_size, scale=(lo, hi), ratio=(1, 1))``.
    Set ``crop_scale`` to the midpoint of ``(lo, hi)`` for a representative crop.
    """
    from PIL import Image as PILImage
    h, w = img_array.shape[:2]
    side_ratio = math.sqrt(crop_scale)
    crop_h = int(round(h * side_ratio))
    crop_w = int(round(w * side_ratio))
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    cropped = img_array[start_h:start_h + crop_h, start_w:start_w + crop_w]
    if cropped.shape[0] != target_size or cropped.shape[1] != target_size:
        cropped = np.array(
            PILImage.fromarray(cropped).resize((target_size, target_size), PILImage.BILINEAR),
            dtype=np.uint8,
        )
    return cropped


def get_libero_images(env, obs, size=224, center_crop_scale=None):
    """Extract camera images from LIBERO observation.

    If ``center_crop_scale`` is not None, a center crop with the given area
    ratio is applied after resize — matching the inference-time equivalent of
    training's ``RandomResizedCrop`` augmentation.
    """
    global _image_frame_idx
    from PIL import Image as PILImage
    images = {}
    agentview = obs.get("agentview_image", obs.get("agentview_rgb"))
    if agentview is not None:
        img = PILImage.fromarray(agentview[::-1, ::-1])
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        if center_crop_scale is not None:
            arr = center_crop_and_resize(arr, center_crop_scale, size)
        images["base_0_rgb"] = arr

    wrist = obs.get("robot0_eye_in_hand_image", obs.get("robot0_eye_in_hand_rgb"))
    if wrist is not None:
        img = PILImage.fromarray(wrist[::-1, ::-1])
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        if center_crop_scale is not None:
            arr = center_crop_and_resize(arr, center_crop_scale, size)
        images["left_wrist_0_rgb"] = arr

    return images


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def skip_init_weights():
    """Bypass random weight initialization when loading from checkpoint.

    Saves ~30-60s for large models (3B+ params) by replacing torch.nn.init
    functions with no-ops. Safe when all weights are overwritten by a checkpoint.
    """
    saved = {}
    _noop = lambda x, *a, **kw: x
    for name in (
        "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_", "xavier_normal_",
        "uniform_", "normal_", "zeros_", "ones_", "constant_", "orthogonal_",
        "trunc_normal_",
    ):
        if hasattr(torch.nn.init, name):
            saved[name] = getattr(torch.nn.init, name)
            setattr(torch.nn.init, name, _noop)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.nn.init, name, fn)


def _apply_sdpa(gemma_model, label: str):
    """Enable SDPA (Scaled Dot-Product Attention) on a GemmaModel.

    Falls back to eager if SDPA is unavailable or fails.
    """
    try:
        gemma_model.config._attn_implementation = "sdpa"
        logger.info(f"{label}: enabled SDPA attention")
    except Exception as e:
        logger.warning(f"{label}: failed to enable SDPA: {e}")


def _apply_compile(parent_module, attr_name: str, label: str, mode: str | None = None):
    """Apply ``torch.compile`` to a sub-module, replacing the attribute in-place.

    Args:
        parent_module: module owning the attribute to compile.
        attr_name: attribute name to replace with the compiled module.
        label: human-readable log label.
        mode: ``torch.compile`` mode.  Defaults to the PyTorch default.  For
            the joint-model denoising loop we use ``"reduce-overhead"`` to
            enable CUDA Graphs, which cuts kernel-launch overhead by ~40% in
            the 10-step denoise loop (LoRA-active: 270→170 ms; merged:
            143→114 ms on RTX 4090).  ``"reduce-overhead"`` is **only** safe
            here because:
              * we compile **gemma_expert.model** only, not paligemma — so
                past-kv tensors flowing from paligemma (eager) into
                gemma_expert are always freshly allocated, avoiding the
                CUDAGraph output-aliasing error that would otherwise trip
                when the graph's static buffers are overwritten between
                denoise steps.
    """
    try:
        original = getattr(parent_module, attr_name)
        kwargs = {"mode": mode} if mode else {}
        compiled = torch.compile(original, **kwargs)
        setattr(parent_module, attr_name, compiled)
        logger.info(f"{label}: enabled torch.compile (mode={mode or 'default'})")
    except Exception as e:
        logger.warning(f"{label}: torch.compile failed: {e}")


def load_vlm(cfg, device):
    """Load VLM waypoint predictor from checkpoint.

    Supports LoRA-active inference (zero drift) when lora.safetensors is
    present.  Falls back to loading merged model.safetensors.

    Handles checkpoints saved from PI0WaypointAE structure by remapping
    paligemma_with_expert.paligemma.* -> paligemma.* keys.
    """
    from openpi.waypoint.vlm_model import PI0WaypointVLM
    import openpi.models.pi0_config as pi0_config
    import gc

    t0 = time.time()
    vlm_precision = cfg.get("vlm_precision", "bfloat16")
    model_cfg = pi0_config.Pi0Config(
        pi05=False,
        max_token_len=cfg.get("vlm_max_token_len", cfg.get("max_token_len", 256)),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        dtype=vlm_precision,
    )
    logger.info(f"VLM precision: {vlm_precision}")
    with skip_init_weights():
        model = PI0WaypointVLM(model_cfg)
    logger.info(f"VLM model init: {time.time() - t0:.1f}s")

    ckpt_path = cfg["vlm_checkpoint"]
    use_lora_active, lora_file = _detect_lora_active(ckpt_path, cfg)

    if use_lora_active:
        # LoRA-active inference: base weights + LoRA adapters (zero drift)
        base_weight_path = cfg.get("pretrained_weight_path")
        if not base_weight_path:
            raise ValueError(
                "LoRA-active VLM eval requires 'pretrained_weight_path' in config."
            )
        logger.info(f"LoRA-active VLM eval: loading base PaliGemma weights...")
        t0 = time.time()
        import openpi.models_pytorch.pi0_pytorch as _pi0
        full_model = _pi0.PI0Pytorch(pi0_config.Pi0Config(
            pi05=True,
            dtype=cfg.get("precision", "bfloat16"),
            paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        ))
        base_file = os.path.join(base_weight_path, "model.safetensors") \
            if os.path.isdir(base_weight_path) else base_weight_path
        safetensors.torch.load_model(full_model, base_file, strict=False)
        pg_state = {}
        for name, param in full_model.paligemma_with_expert.paligemma.named_parameters():
            pg_state[f"paligemma.{name}"] = param.data
        model.load_state_dict(pg_state, strict=False)
        logger.info(f"  Base PaliGemma weights loaded: {len(pg_state)} tensors ({time.time() - t0:.1f}s)")
        del full_model, pg_state
        gc.collect()

        import openpi.models_pytorch.lora_pytorch as lora_utils
        lora_cfg = lora_utils.build_lora_training_config(cfg)
        logger.info("  Injecting LoRA structure...")
        lora_utils.apply_lora_to_model(model, lora_cfg)

        logger.info(f"  Loading LoRA weights from {lora_file}")
        t0 = time.time()
        lora_utils.load_lora_checkpoint(model, lora_file)
        logger.info(f"  LoRA weights loaded: {time.time() - t0:.1f}s (zero drift)")
    else:
        # Merged model (fallback)
        ckpt_file = os.path.join(ckpt_path, "model.safetensors")
        logger.info(f"Loading VLM from {ckpt_file}")

        with open(ckpt_file, "rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            header_keys = list(json.loads(f.read(header_size)).keys())

        has_pg_direct = any(k.startswith("paligemma.") for k in header_keys)
        has_pg_nested = any(k.startswith("paligemma_with_expert.paligemma.") for k in header_keys)

        t0 = time.time()
        if has_pg_direct:
            safetensors.torch.load_model(model, ckpt_file)
        elif has_pg_nested:
            PREFIX = "paligemma_with_expert.paligemma."
            state_dict = safetensors.torch.load_file(ckpt_file, device="cpu")
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith(PREFIX):
                    new_key = "paligemma." + k[len(PREFIX):]
                    remapped[new_key] = v
            own_state = model.state_dict()
            loaded, skipped = 0, 0
            for k, v in remapped.items():
                if k in own_state and own_state[k].shape == v.shape:
                    own_state[k].copy_(v)
                    loaded += 1
                else:
                    skipped += 1
            model.load_state_dict(own_state, strict=False)
            del state_dict
            logger.info(f"VLM: loaded {loaded} params, skipped {skipped} (remapped from AE checkpoint)")
        else:
            raise ValueError(f"Cannot find PaliGemma weights in checkpoint: {ckpt_file}")
        logger.info(f"VLM weight load: {time.time() - t0:.1f}s")

    t0 = time.time()
    model = model.to(device).eval()
    logger.info(f"VLM to {device}: {time.time() - t0:.1f}s")

    _apply_sdpa(model.paligemma.model.language_model, "VLM language_model")

    if cfg.get("torch_compile", False):
        _apply_compile(model.paligemma.model, "language_model", "VLM language_model")

    return model


def load_ae(cfg, device):
    """Load Action Expert from checkpoint.

    Supports LoRA-active inference (zero drift) when lora.safetensors is
    present.  Falls back to loading merged model.safetensors.
    """
    from openpi.waypoint.ae_model import PI0WaypointAE
    import openpi.models.pi0_config as pi0_config

    t0 = time.time()
    model_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=cfg.get("model_action_dim", 32),
        action_horizon=cfg.get("horizon_steps", 32),
        max_token_len=cfg.get("ae_max_token_len", cfg.get("max_token_len", 64)),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        dtype=cfg.get("precision", "bfloat16"),
    )
    with skip_init_weights():
        model = PI0WaypointAE(model_cfg)
    logger.info(f"AE model init: {time.time() - t0:.1f}s")

    ckpt_path = cfg["ae_checkpoint"]
    use_lora_active, lora_file = _detect_lora_active(ckpt_path, cfg)

    if use_lora_active:
        # LoRA-active inference: base weights + LoRA adapters (zero drift)
        base_weight_path = cfg.get("pretrained_weight_path")
        if not base_weight_path:
            raise ValueError(
                "LoRA-active AE eval requires 'pretrained_weight_path' in config."
            )
        base_file = os.path.join(base_weight_path, "model.safetensors") \
            if os.path.isdir(base_weight_path) else base_weight_path
        logger.info(f"LoRA-active AE eval: loading base weights from {base_file}")
        t0 = time.time()
        state_dict = safetensors.torch.load_file(base_file, device="cpu")
        own_state = model.state_dict()
        loaded, skipped = 0, 0
        for name, param in state_dict.items():
            if name not in own_state:
                skipped += 1
                continue
            if own_state[name].shape != param.shape:
                skipped += 1
                continue
            own_state[name].copy_(param)
            loaded += 1
        logger.info(f"  Base weights loaded: {loaded} tensors, {skipped} skipped ({time.time() - t0:.1f}s)")
        del state_dict

        import openpi.models_pytorch.lora_pytorch as lora_utils
        lora_cfg = lora_utils.build_lora_training_config(cfg)
        logger.info("  Injecting LoRA structure...")
        lora_utils.apply_lora_to_model(model, lora_cfg)

        logger.info(f"  Loading LoRA weights from {lora_file}")
        t0 = time.time()
        lora_utils.load_lora_checkpoint(model, lora_file)
        logger.info(f"  LoRA weights loaded: {time.time() - t0:.1f}s (zero drift)")

        if cfg.get("lora_inference_slim", True):
            lora_utils.install_inference_slim_forward(model)
    else:
        # Merged model (fallback)
        ckpt_file = os.path.join(ckpt_path, "model.safetensors")
        logger.info(f"Loading Action Expert from {ckpt_file}")
        t0 = time.time()
        safetensors.torch.load_model(model, ckpt_file)
        logger.info(f"AE weight load: {time.time() - t0:.1f}s")

    t0 = time.time()
    model = model.to(device).eval()
    logger.info(f"AE to {device}: {time.time() - t0:.1f}s")

    _apply_sdpa(model.paligemma_with_expert.paligemma.model.language_model, "AE paligemma")
    _apply_sdpa(model.paligemma_with_expert.gemma_expert.model, "AE gemma_expert")

    if cfg.get("torch_compile", False):
        ae_compile_mode = cfg.get("ae_compile_mode", "reduce-overhead")
        _apply_compile(
            model.paligemma_with_expert.gemma_expert, "model", "AE gemma_expert",
            mode=ae_compile_mode,
        )

    return model


def _assert_block_planner_weights_loaded(model, ckpt_path, lora_file, cfg) -> None:
    """Fail if a `planner_mode: block_ar` config points at a token-AR checkpoint.

    Both loaders iterate the *checkpoint's* keys, so weights the model has but the
    file lacks are never reported.  A token-AR checkpoint therefore loads into a
    block_ar model without a single warning, leaving the query/slot/block
    residuals at their zero initialisation -- and the eval then block-decodes a
    backbone that never saw the block-causal mask.  That produces a plausible
    success rate from meaningless plans, i.e. the block-AR gate fails for the
    wrong reason.  This is the schema check the state-dict key set is supposed to
    provide, made explicit.
    """
    import safetensors.torch

    candidates = [lora_file] if lora_file else []
    candidates.append(os.path.join(ckpt_path, "model.safetensors"))
    seen_any = False
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        seen_any = True
        with safetensors.torch.safe_open(path, framework="pt") as f:
            keys = list(f.keys())
            if any(k.startswith("block_planner.") for k in keys):
                # Presence is not enough: the slot rows are indexed by the
                # robot layout (1 + proprio_dim + gripper + 2), so a planner
                # trained for another layout has the right keys with the wrong
                # row count.  Both loaders now refuse the mismatch, but check
                # the file itself so the error names the geometry, not a tensor.
                qr = "block_planner.query_resid"
                if qr in keys:
                    ckpt_width = int(f.get_slice(qr).get_shape()[0])
                    if ckpt_width != int(model.planner_block_width):
                        raise ValueError(
                            f"block planner geometry mismatch: {os.path.basename(path)} "
                            f"has {qr} with {ckpt_width} slot rows but the model built "
                            f"from this eval config has planner_block_width="
                            f"{model.planner_block_width} (proprio_dim="
                            f"{model.planner_proprio_dim}, gripper_token="
                            f"{model.planner_use_gripper_token}). Fix robot_type / "
                            "robot_config_kwargs to the layout the checkpoint was "
                            "trained with (its metadata.pt records it)."
                        )
                logger.info(
                    f"  block planner weights found in {os.path.basename(path)} "
                    f"(block_width={model.planner_block_width})"
                )
                return
    if not seen_any:  # pragma: no cover - load would already have failed
        return
    raise ValueError(
        f"planner_mode='block_ar' but no `block_planner.*` tensor exists in "
        f"{[os.path.basename(p) for p in candidates if p]}: this looks like a token-AR "
        "checkpoint. Block-decoding it would run with zero-initialised query/slot "
        "embeddings against a backbone that never saw the block-causal mask, and would "
        "report a plausible success rate from meaningless plans. Point "
        "eval.joint_checkpoint at a block_ar checkpoint, or set planner_mode: token_ar "
        "and waypoint_decode.impl: compact for the control arm."
    )


def _detect_lora_active(ckpt_path, cfg):
    """Decide whether to use LoRA-active inference for a checkpoint.

    Returns (use_lora_active: bool, lora_file: str | None).
    Prefers LoRA-active when lora.safetensors is present.

    EMA selection:
      * ``cfg["use_ema"] == True`` → require and use
        ``lora_ema.safetensors`` instead of ``lora.safetensors``.
      * ``cfg["use_ema"] == False`` (default) → always use the raw weights.

    The selected adapter's SHA-256 is logged before it is loaded.  Explicit
    EMA selection is fail-closed: a missing file, unsupported full-model EMA,
    or severely stale finite-horizon EMA raises instead of silently producing
    a mislabeled result.
    """
    import openpi.models_pytorch.lora_pytorch as lora_utils
    from openpi.waypoint.ema import EMA_FORENSIC_OVERRIDE_KEY
    from openpi.waypoint.ema import validate_ema_staleness

    use_ema = cfg.get("use_ema", False)
    if not isinstance(use_ema, bool):
        raise TypeError(f"use_ema must be a boolean, got {use_ema!r}")

    if not cfg.get("lora_enabled", False):
        if use_ema:
            raise ValueError(
                "use_ema=true is currently supported only for LoRA-active checkpoints. "
                "The trainer's model_ema.safetensors full-model artifact has no eval "
                "overlay loader; refusing to silently load model.safetensors instead."
            )
        return False, None

    ckpt_type = lora_utils.detect_checkpoint_type(ckpt_path)
    lora_file = os.path.join(ckpt_path, "lora.safetensors")
    ema_file = os.path.join(ckpt_path, "lora_ema.safetensors")

    if use_ema and not os.path.isfile(ema_file):
        raise FileNotFoundError(
            f"use_ema=true requires {ema_file}, but the file does not exist. "
            "Use --no-ema for raw lora.safetensors; no implicit fallback is allowed."
        )

    if ckpt_type in ("lora", "both"):
        if ckpt_type == "both":
            logger.info(
                "Checkpoint has both lora.safetensors and model.safetensors. "
                "Using LoRA-active inference (zero drift, recommended)."
            )
        selected_file = ema_file if use_ema else lora_file
        selected_role = "ema" if use_ema else "raw"

        if use_ema:
            # The EMA file is meaningful only together with the recipe that
            # produced it.  Never guess decay/warmup from an eval config: that
            # can make an unknown or stale historical EMA look safe.
            forensic_override = cfg.get(EMA_FORENSIC_OVERRIDE_KEY, False)
            if not isinstance(forensic_override, bool):
                raise TypeError(
                    f"{EMA_FORENSIC_OVERRIDE_KEY} must be a boolean, got "
                    f"{forensic_override!r}"
                )

            metadata = None
            metadata_path = os.path.join(ckpt_path, "metadata.pt")
            provenance_error = None
            if os.path.isfile(metadata_path):
                try:
                    loaded = torch.load(metadata_path, map_location="cpu", weights_only=True)
                    if isinstance(loaded, dict):
                        metadata = loaded
                    else:
                        provenance_error = f"EMA metadata is not a dict: {metadata_path}"
                except Exception as exc:
                    provenance_error = f"Could not read EMA metadata {metadata_path}: {exc}"
            else:
                provenance_error = f"EMA metadata is missing: {metadata_path}"

            required_fields = {"global_step", "ema_decay", "ema_warmup_steps"}
            if metadata is not None:
                missing_fields = sorted(required_fields - metadata.keys())
                if missing_fields:
                    provenance_error = (
                        f"EMA metadata {metadata_path} is missing required fields: "
                        f"{missing_fields}"
                    )

            if provenance_error is not None:
                message = (
                    f"Cannot verify EMA provenance: {provenance_error}. Refusing to infer the "
                    "training recipe from the eval config or checkpoint directory name. Only an "
                    f"intentional forensic reproduction may set "
                    f"`{EMA_FORENSIC_OVERRIDE_KEY}: true`."
                )
                if not forensic_override:
                    raise ValueError(message)
                logger.warning(
                    "%s Forensic override accepted; staleness is unknown and this artifact "
                    "must not be used as a production default.",
                    message,
                )
            else:
                assert metadata is not None
                validate_ema_staleness(
                    decay=float(metadata["ema_decay"]),
                    warmup_steps=int(metadata["ema_warmup_steps"]),
                    total_steps=int(metadata["global_step"]),
                    allow_stale_ema_forensics=forensic_override,
                    context=f"EMA checkpoint {ckpt_path}",
                )

        digest = hashlib.sha256()
        with open(selected_file, "rb") as adapter_file:
            for chunk in iter(lambda: adapter_file.read(1024 * 1024), b""):
                digest.update(chunk)
        logger.info(
            "Adapter selected: role=%s path=%s sha256=%s",
            selected_role,
            selected_file,
            digest.hexdigest(),
        )
        if use_ema:
            logger.info(
                "  EMA mode: using %s (overrides lora.safetensors)",
                ema_file,
            )
        return True, selected_file
    elif ckpt_type == "merged":
        lora_utils.warn_merged_checkpoint(ckpt_path)
        return False, None
    return False, None


def _assert_planner_schema_matches_checkpoint(ckpt_path, cfg) -> None:
    """Refuse an eval config whose planner schema contradicts the checkpoint.

    ``planner_block0_cond`` is not visible in the state dict -- it changes what
    block 0 is *fed*, not which parameters exist -- so a checkpoint trained with
    ``current_state`` loads with zero warnings under a ``none`` config and then
    plans from an input it never saw in training.  That is the same silent-garbage
    class as decoding a block checkpoint token-at-a-time
    (``docs/11-open-defects.md`` P0-2), so check it against the recorded metadata
    instead of trusting the caller.

    The guard is asymmetric on purpose (round3 OPEN_ITEMS item 1):

    * a config requesting a **non-default** ``planner_block0_cond`` needs the
      metadata to exist, be readable, and carry a confirming value -- a
      conditioned model built without proof would be fed a condition it may
      never have seen, silently;
    * a config requesting ``none`` accepts a checkpoint whose metadata lacks
      the key, because the key only became recordable in 42dc32c and no
      checkpoint predating it can have been trained with conditioning.  A
      metadata file that *does* carry the key must still agree.
    """
    requested_cond = cfg.get("planner_block0_cond", "none")
    metadata_path = os.path.join(ckpt_path, "metadata.pt")
    metadata = None
    if os.path.isfile(metadata_path):
        try:
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # pragma: no cover - unreadable metadata
            logger.warning("Could not read planner metadata %s: %s", metadata_path, exc)
            metadata = None
        if not isinstance(metadata, dict):
            metadata = None
    if metadata is None:
        if requested_cond != "none":
            raise ValueError(
                f"planner_block0_cond={requested_cond!r} requested, but checkpoint "
                f"{ckpt_path} has no readable metadata.pt to confirm it was trained "
                "that way. Refusing: a conditioned block-0 fed an unconfirmed input "
                "decodes silent garbage (docs/11-open-defects.md P0-2 class)."
            )
        return
    if "planner_block0_cond" not in metadata and requested_cond != "none":
        raise ValueError(
            f"planner_block0_cond={requested_cond!r} requested, but the metadata of "
            f"{ckpt_path} does not record planner_block0_cond, so the request cannot "
            "be confirmed. Checkpoints trained with conditioning always record the "
            "key (it landed together with the feature in 42dc32c)."
        )
    for key, default in (("planner_mode", "token_ar"), ("planner_block0_cond", "none")):
        if key not in metadata:
            continue
        trained = metadata[key]
        requested = cfg.get(key, default)
        if trained != requested:
            raise ValueError(
                f"{key} mismatch: checkpoint {ckpt_path} was trained with {trained!r} "
                f"but this eval config requests {requested!r}. Loading anyway would "
                "silently evaluate a different model than the one that was trained."
            )
    # Planner geometry (recorded since the dual-arm hardening; older checkpoints
    # lack the keys and must still load -- warn, don't fail).  Compared against
    # the same derivation the model is built from, so a robot_type /
    # robot_config_kwargs / num_waypoints edit in the eval config is caught
    # before a mis-shaped planner is loaded.
    from openpi.waypoint.joint_model import PLANNER_GEOMETRY_METADATA_KEYS
    from openpi.waypoint.joint_model import planner_block_width
    from openpi.waypoint.joint_model import planner_kwargs_from_config

    requested_kwargs = planner_kwargs_from_config(cfg)
    requested_geom = {
        "planner_proprio_dim": requested_kwargs["planner_proprio_dim"],
        "planner_num_waypoints": requested_kwargs["planner_num_waypoints"],
        "planner_use_gripper_token": requested_kwargs["planner_use_gripper_token"],
        "planner_n_gripper_slots": requested_kwargs["planner_n_gripper_slots"],
        "planner_block_width": planner_block_width(
            requested_kwargs["planner_proprio_dim"],
            requested_kwargs["planner_use_gripper_token"],
            requested_kwargs["planner_n_gripper_slots"],
        ),
    }
    absent = [k for k in PLANNER_GEOMETRY_METADATA_KEYS if k not in metadata]
    if absent:
        logger.warning(
            "checkpoint %s metadata does not record planner geometry %s (written "
            "before the keys existed); trusting the eval config: %s",
            ckpt_path, absent, requested_geom,
        )
    for key in PLANNER_GEOMETRY_METADATA_KEYS:
        if key not in metadata:
            continue
        trained = metadata[key]
        requested = requested_geom[key]
        if type(requested)(trained) != requested:
            raise ValueError(
                f"{key} mismatch: checkpoint {ckpt_path} was trained with {trained!r} "
                f"but this eval config implies {requested!r} (robot_type="
                f"{cfg.get('robot_type', 'libero')!r}, robot_config_kwargs="
                f"{cfg.get('robot_config_kwargs')!r}, num_waypoints="
                f"{cfg.get('num_waypoints', 7)!r}). The block planner's slot rows are "
                "indexed by this geometry; loading anyway would evaluate a misaligned "
                "or truncated planner."
            )

def _resolve_goal_conditioning(ckpt_path, cfg):
    """Resolve goal-conditioning constructor kwargs from checkpoint metadata.

    Fail-closed in both directions (docs/14 §5.1, docs/16):

    * an eval config requesting a goal-conditioned mode needs the checkpoint
      metadata to confirm the mode **and** carry the trained contract — the
      ``goal_delta_scale`` and (for v2) the gate/noise/dropout hyperparameters
      are model contracts, so they are read back from metadata rather than
      from a config or stats file that may have drifted since training;
    * an eval config requesting ``none`` refuses a checkpoint whose metadata
      records goal conditioning: loading it without the adapter would silently
      evaluate a different model (``load_lora_checkpoint`` would also reject
      the stray ``goal_encoder.*`` keys, but failing here names the cause);
    * a v2 key set explicitly in the eval config must agree with the metadata
      value (互验): a mismatch means the operator believes they are evaluating
      a different contract than the one the checkpoint was trained with.

    Returns a dict of ``PI0WaypointJoint`` constructor kwargs.
    """
    from openpi.waypoint.joint_model import GOAL_V2_METADATA_KEYS
    from openpi.waypoint.joint_model import goal_condition_kwargs_from_metadata

    requested = cfg.get("goal_condition_mode", "none")
    metadata = None
    metadata_path = os.path.join(ckpt_path, "metadata.pt")
    if os.path.isfile(metadata_path):
        try:
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # pragma: no cover - unreadable metadata
            logger.warning("Could not read goal metadata %s: %s", metadata_path, exc)
            metadata = None
        if not isinstance(metadata, dict):
            metadata = None
    trained = (metadata or {}).get("goal_condition_mode", "none")
    if requested == "none":
        if trained != "none":
            raise ValueError(
                f"checkpoint {ckpt_path} was trained with goal_condition_mode="
                f"{trained!r} but this eval config requests 'none'. Loading anyway "
                "would silently drop the trained goal adapter."
            )
        return {"goal_condition_mode": "none", "goal_delta_scale": None}
    if metadata is None:
        raise ValueError(
            f"goal_condition_mode={requested!r} requested, but checkpoint {ckpt_path} "
            "has no readable metadata.pt to confirm it was trained that way."
        )
    if trained != requested:
        raise ValueError(
            f"goal_condition_mode mismatch: checkpoint {ckpt_path} was trained with "
            f"{trained!r} but this eval config requests {requested!r}."
        )
    kwargs = goal_condition_kwargs_from_metadata(
        metadata, source=f"checkpoint {ckpt_path}"
    )
    for key in GOAL_V2_METADATA_KEYS:
        if key in cfg and float(cfg[key]) != float(kwargs.get(key, float("nan"))):
            raise ValueError(
                f"{key} mismatch: checkpoint {ckpt_path} was trained with "
                f"{kwargs.get(key)!r} but this eval config requests {cfg[key]!r}. "
                "The v2 contract travels with the checkpoint; remove the key from "
                "the eval config or evaluate the matching checkpoint."
            )
    return kwargs


def load_joint(cfg, device):
    """Load joint VLM+AE model from a single checkpoint.

    The joint model shares a single PaliGemma backbone between VLM and AE,
    saving ~50% VRAM compared to loading two separate models.

    Supports two loading modes:
      - **LoRA-active** (preferred): when ``lora.safetensors`` exists in the
        checkpoint dir, loads base weights from ``pretrained_weight_path``,
        injects LoRA structure, and loads LoRA weights directly.  This gives
        **zero numerical drift** compared to training.
      - **Merged** (fallback): loads ``model.safetensors`` directly.  Has
        unavoidable bfloat16 precision drift from LoRA merge.
    """
    from openpi.waypoint.joint_model import PI0WaypointJoint
    import openpi.models.pi0_config as pi0_config

    t0 = time.time()
    model_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=cfg.get("model_action_dim", 32),
        action_horizon=cfg.get("horizon_steps", 32),
        max_token_len=cfg.get("ae_max_token_len", cfg.get("max_token_len", 64)),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        dtype=cfg.get("precision", "bfloat16"),
    )
    # eval_calvin imports and calls this same loader, so the planner geometry must
    # come from the config, not from a hardcoded "libero" (both happen to report
    # continuous_proprio_dim == 6 today, which is exactly why this was invisible).
    # planner_kwargs_from_config is the single derivation shared with the trainer
    # and scripts/merge_lora.py (robot_type + robot_config_kwargs ->
    # continuous_proprio_dim, num_waypoints, planner_mode, ...).
    from openpi.waypoint.joint_model import planner_kwargs_from_config

    planner_kwargs = planner_kwargs_from_config(cfg)
    ckpt_path = cfg["joint_checkpoint"]
    goal_kwargs = _resolve_goal_conditioning(ckpt_path, cfg)
    with skip_init_weights():
        model = PI0WaypointJoint(
            config=model_cfg,
            vlm_max_token_len=cfg.get("vlm_max_token_len", cfg.get("max_token_len", 256)),
            gradient_strategy="none",
            **planner_kwargs,
            # 2026-08-24 训评错配修复(与 openpi-ctrl12k 工作区同日修复逐字对齐):
            # 训练端 train_waypoint_joint.py:575 传本开关(E060/E062 配方均为 true,只编码
            # 2 路真实相机);评测端此前漏传恒 False(多编码 1 路全 mask 空相机)。
            # 实测同权重同输入单步去噪动作差 |a| 的 3.69%。评测配置需显式给出该键。
            skip_masked_cameras=cfg.get("skip_masked_cameras", False),
            **goal_kwargs,
        )
    logger.info(f"Joint model init: {time.time() - t0:.1f}s")
    if model.block_planner is not None:
        # skip_init_weights() leaves fresh modules uninitialised; the block
        # planner's zero-init residuals are part of its contract, so re-apply
        # them explicitly and let the checkpoint overwrite them.
        with torch.no_grad():
            torch.nn.init.zeros_(model.block_planner.query_resid)
            torch.nn.init.zeros_(model.block_planner.slot_embed.weight)
            if model.block_planner.block_embed is not None:
                torch.nn.init.zeros_(model.block_planner.block_embed.weight)
    if getattr(model, "goal_encoder", None) is not None:
        # Same skip_init_weights() caveat: re-apply the zero-init contract and
        # let the checkpoint overwrite every goal_encoder tensor (the
        # fail-closed check in load_lora_checkpoint guarantees they exist).
        model.goal_encoder.reset_zero_init()
    _assert_planner_schema_matches_checkpoint(ckpt_path, cfg)
    use_lora_active, lora_file = _detect_lora_active(ckpt_path, cfg)

    if use_lora_active:
        # LoRA-active inference: base weights + LoRA adapters (zero drift)
        base_weight_path = cfg.get("pretrained_weight_path")
        if not base_weight_path:
            raise ValueError(
                "LoRA-active eval requires 'pretrained_weight_path' in config "
                "(the original base model used for training)."
            )
        base_file = os.path.join(base_weight_path, "model.safetensors") \
            if os.path.isdir(base_weight_path) else base_weight_path
        logger.info(f"LoRA-active eval: loading base weights from {base_file}")
        t0 = time.time()
        PI0WaypointJoint.load_pretrained_weights(model, base_file, "cpu")
        logger.info(f"  Base weights loaded: {time.time() - t0:.1f}s")

        import openpi.models_pytorch.lora_pytorch as lora_utils
        lora_cfg = lora_utils.build_lora_training_config(cfg)
        logger.info("  Injecting LoRA structure...")
        lora_utils.apply_lora_to_model(model, lora_cfg)

        logger.info(f"  Loading LoRA weights from {lora_file}")
        t0 = time.time()
        lora_utils.load_lora_checkpoint(model, lora_file)
        logger.info(f"  LoRA weights loaded: {time.time() - t0:.1f}s (zero drift)")

        if cfg.get("runtime_merge", False):
            t0 = time.time()
            merge_count = lora_utils.merge_lora_weights(model)
            logger.info(
                f"  Runtime merge: {merge_count} LoRA layers merged into base "
                f"({time.time() - t0:.1f}s). Same speed as pre-saved merged model "
                f"with identical numerical behavior."
            )
        elif cfg.get("lora_inference_slim", True):
            # Zero-drift speedup: fold scaling into lora_B and slim the
            # LoRALinear.forward.  Must come *after* load_lora_checkpoint
            # (so scaling is applied to the real weights) and *before*
            # torch.compile (so the compiled graph captures the slim form).
            lora_utils.install_inference_slim_forward(model)
    else:
        # Merged model (fallback)
        ckpt_file = os.path.join(ckpt_path, "model.safetensors")
        logger.info(f"Loading joint model from {ckpt_file}")
        t0 = time.time()
        PI0WaypointJoint.load_pretrained_weights(model, ckpt_file, "cpu")
        logger.info(f"Joint weight load: {time.time() - t0:.1f}s")

    if model.block_planner is not None:
        _assert_block_planner_weights_loaded(model, ckpt_path, lora_file, cfg)

    t0 = time.time()
    model = model.to(device).eval()
    logger.info(f"Joint model to {device}: {time.time() - t0:.1f}s")

    _apply_sdpa(model.paligemma_with_expert.paligemma.model.language_model, "Joint paligemma")
    _apply_sdpa(model.paligemma_with_expert.gemma_expert.model, "Joint gemma_expert")

    if cfg.get("torch_compile", False):
        # reduce-overhead uses CUDA Graphs → ~40% fewer kernel-launch cycles in
        # the 10-step denoise loop.  Safe because only gemma_expert is
        # compiled; paligemma stays eager so its past-kv outputs aren't
        # aliased across denoise steps (see _apply_compile docstring).
        ae_compile_mode = cfg.get("ae_compile_mode", "reduce-overhead")
        _apply_compile(
            model.paligemma_with_expert.gemma_expert, "model", "Joint gemma_expert",
            mode=ae_compile_mode,
        )

    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def accumulate_decode_stats(sink: dict, decode_stats) -> dict:
    """Fold one planning call's ``DecodeStats`` into a per-episode accumulator.

    Shared by both evaluators, which carried byte-identical copies of this loop.
    Counters add up; identity and tri-state fields (``impl``,
    ``fixed_point_verified``) do not -- ``None`` would raise on ``+``, and
    summing a bool produced a "count" nobody could interpret (a run of 40
    replans reporting ``fixed_point_verified: 37`` says nothing about the three
    that were unverified).
    """
    if decode_stats is None:
        return sink
    for key, value in decode_stats.as_dict().items():
        if value is None or isinstance(value, (str, bool)):
            sink[key] = value
        else:
            sink[key] = sink.get(key, 0) + value
    return sink


def _generate_waypoints_params(model) -> frozenset[str]:
    """The parameter names ``model.generate_waypoints`` actually accepts."""
    import inspect

    try:
        return frozenset(inspect.signature(model.generate_waypoints).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return frozenset()


def _supports_decode_cfg(model) -> bool:
    """Whether ``model.generate_waypoints`` accepts the pluggable-decoder kwargs."""
    return "decode_cfg" in _generate_waypoints_params(model)


def _decode_kwargs_for(model, decode_cfg, generator) -> dict:
    """Forward the decode settings *parameter by parameter*.

    "Supports decode_cfg" used to be treated as all-or-nothing: a model without
    that parameter got ``{}``, which silently dropped ``temperature`` (and the
    planner RNG) on the standalone VLM+AE path -- a config asking for
    ``temperature: 1.25`` decoded greedily with no warning.  The standalone
    decoder does accept ``temperature`` / ``generator`` /
    ``forbid_terminal_first_waypoint``, so pass whatever the callee declares and
    refuse loudly if a non-default setting has nowhere to go.
    """
    params = _generate_waypoints_params(model)
    if "decode_cfg" in params:
        return {"decode_cfg": decode_cfg, "generator": generator}

    kwargs = {}
    if generator is not None and "generator" in params:
        kwargs["generator"] = generator
    if decode_cfg is None:
        return kwargs

    # name on WaypointDecodeConfig -> callee parameter name (same here, but the
    # mapping is what makes "unhonourable" below checkable rather than assumed).
    forwardable = {"temperature": "temperature",
                   "forbid_terminal_first_waypoint": "forbid_terminal_first_waypoint"}
    defaults = type(decode_cfg)()
    unhonourable = []
    for field, param in forwardable.items():
        value = getattr(decode_cfg, field)
        if param in params:
            kwargs[param] = value
        elif value != getattr(defaults, field):
            unhonourable.append(f"{field}={value!r}")
    if generator is not None and "generator" not in params and decode_cfg.temperature:
        unhonourable.append("temperature sampling without a planner generator")
    if unhonourable:
        raise ValueError(
            f"{type(model).__name__}.generate_waypoints cannot honour "
            f"{', '.join(unhonourable)}; it would have been dropped silently."
        )
    return kwargs


@torch.no_grad()
def predict_waypoints(vlm, images, instruction, wp_tokenizer, state_continuous_norm, gripper_binary, device,
                      decode_cfg=None, generator=None):
    """VLM autoregressive waypoint prediction.

    Args:
        state_continuous_norm: normalized continuous proprio (e.g. 6D for LIBERO).
        gripper_binary: 0=close, 1=open.
        decode_cfg: ``WaypointDecodeConfig``; ``None`` keeps the legacy decoder.
        generator: RNG for temperature sampling (unused when temperature == 0).
            Deliberately NOT the Action Expert's noise generator: sharing it would
            make the planner's draws shift the AE's noise stream, and it is a CPU
            generator while the planner logits live on the GPU.
    """
    img_tensors = {}
    img_masks = {}
    for key, arr in images.items():
        t = torch.from_numpy(arr).float() / 127.5 - 1.0
        img_tensors[key] = t.unsqueeze(0).to(device)
        img_masks[key] = torch.ones(1, dtype=torch.bool, device=device)

    for model_key in ["base_0_rgb", "left_wrist_0_rgb"]:
        if model_key not in img_tensors:
            img_tensors[model_key] = torch.zeros(1, 224, 224, 3, device=device)
            img_masks[model_key] = torch.zeros(1, dtype=torch.bool, device=device)

    from openpi.waypoint.tokenizer import PROPRIO_N_BINS

    proprio_dim = wp_tokenizer.proprio_dim
    state_for_prompt = state_continuous_norm[:proprio_dim]
    discretized = np.digitize(np.clip(state_for_prompt, -1, 1), np.linspace(-1, 1, PROPRIO_N_BINS + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized.astype(int)))

    if wp_tokenizer.use_gripper_token:
        grip_str = "open" if gripper_binary else "closed"
        prompt_text = f"Task: {instruction.strip().replace('_', ' ').lower()}, State: {state_str}, Gripper: {grip_str};\n"
    else:
        prompt_text = f"Task: {instruction.strip().replace('_', ' ').lower()}, State: {state_str};\n"

    prompt_tokens_list = wp_tokenizer._pg_tokenizer.encode(prompt_text, add_bos=True)
    prompt_tokens = torch.tensor([prompt_tokens_list], dtype=torch.long, device=device)
    prompt_mask = torch.ones_like(prompt_tokens, dtype=torch.bool)

    # The standalone PI0WaypointVLM (used by the separate VLM+AE configs) has no
    # decode_cfg parameter and its own prefix embedding convention, so only the
    # joint model gets the pluggable decoders -- but it does honour the individual
    # legacy knobs, so forward them one by one instead of dropping the lot.
    extra = _decode_kwargs_for(vlm, decode_cfg, generator)
    # planner_block0_cond='current_state' conditions block 0 on the measured
    # current state through the same token families as every later block.  The
    # model refuses to decode without it (and refuses it when unconfigured), so
    # this is supplied whenever the loaded model asks for it.
    if getattr(vlm, "planner_block0_cond", "none") == "current_state":
        block0 = wp_tokenizer.encode_current_state_block(
            state_for_prompt, bool(gripper_binary)
        )
        extra["block0_cond_tokens"] = torch.as_tensor(
            block0, dtype=torch.long, device=device
        )[None]
    waypoints = vlm.generate_waypoints(
        images=img_tensors,
        image_masks=img_masks,
        prompt_tokens=prompt_tokens,
        prompt_mask=prompt_mask,
        wp_tokenizer=wp_tokenizer,
        **extra,
    )
    return waypoints[0]


def load_pg_tokenizer():
    """Load PaliGemma SentencePiece tokenizer (cached, call once)."""
    import sentencepiece
    import openpi.shared.download as download

    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


@torch.no_grad()
def predict_actions(ae_model, images, instruction, start_wp, end_wp, duration, device, pg_tok,
                    noise_generator=None):
    """Action Expert flow matching inference."""
    img_tensors = {}
    img_masks = {}
    for key, arr in images.items():
        t = torch.from_numpy(arr).float() / 127.5 - 1.0  # (H, W, C)
        t = t.permute(2, 0, 1)  # -> (C, H, W)
        img_tensors[key] = t.unsqueeze(0).to(device)  # (1, C, H, W)
        img_masks[key] = torch.ones(1, dtype=torch.bool, device=device)

    for model_key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        if model_key not in img_tensors:
            img_tensors[model_key] = torch.zeros(1, 3, 224, 224, device=device)
            img_masks[model_key] = torch.zeros(1, dtype=torch.bool, device=device)

    text = f"Task: {instruction.strip().replace('_', ' ').lower()}, \n"
    tids = pg_tok.encode(text, add_bos=True)
    max_len = 64
    tids_len = len(tids)
    if tids_len < max_len:
        tids = tids + [0] * (max_len - tids_len)
        mask = [True] * tids_len + [False] * (max_len - tids_len)
    else:
        tids = tids[:max_len]
        mask = [True] * max_len

    prompt_tokens = torch.tensor([tids[:max_len]], dtype=torch.long, device=device)
    prompt_mask = torch.tensor([mask[:max_len]], dtype=torch.bool, device=device)

    class _Obs:
        def __init__(self):
            self.images = img_tensors
            self.image_masks = img_masks
            self.state = torch.zeros(1, 32, device=device)
            self.tokenized_prompt = prompt_tokens
            self.tokenized_prompt_mask = prompt_mask
            self.token_ar_mask = None
            self.token_loss_mask = None
    obs = _Obs()

    start_t = torch.from_numpy(start_wp).float().unsqueeze(0).to(device)
    end_t = torch.from_numpy(end_wp).float().unsqueeze(0).to(device)
    dur_t = torch.tensor([float(duration)], dtype=torch.float32, device=device)

    noise = None
    if noise_generator is not None:
        noise = torch.randn(
            1, ae_model.action_horizon, ae_model.action_dim,
            generator=noise_generator, device="cpu",
        ).to(device)

    actions = ae_model.sample_actions(obs, start_t, end_t, dur_t, noise=noise)
    return actions.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _fmt_array(a, n=4):
    """Format first n elements of an array for logging."""
    flat = np.asarray(a).flatten()
    vals = " ".join(f"{v:.4f}" for v in flat[:n])
    if len(flat) > n:
        vals += " ..."
    return f"[{vals}]"


def _episode_stats() -> dict:
    """Fresh per-episode diagnostics record (always collected, cheap)."""
    return {
        "steps": 0,
        "replans": 0,
        "segments": 0,
        "budget_truncated": False,
        "duration_overflow_count": 0,
        "d0_terminations": 0,
        "deviation_replans": 0,
        "segment_deviation_l2": [],
        "segment_deviation_linf": [],
        "segment_deviation_l2_with_grip": [],
        "planner_ms": [],
        "planner": {},
    }


class _TrajectoryRecorder:
    """E062 collection sink: records (obs, action) pairs in the RLDS storage
    convention of ``libero_10_no_noops`` so collected rollouts can be packed
    into a merged training dataset without any further transformation.

    Conventions (corrected 2026-08-25, E062 erratum — see E062_PREREG_NOTE §7
    and E063_PREREG_NOTE §3): images stored 180°-rotated relative to the raw
    robosuite render (``frame[::-1, ::-1]``), 256x256 JPEG; ``state`` is the 8D
    proprio [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]; ``action`` is the
    7D env action stored AS-IS.  The gripper convention is identical on both
    sides — robosuite PandaGripper: +1 == close, -1 == open, and the dataset
    stores the same sign (the loader's ``1-x`` and the execution chain's
    negation cancel, so ``action_env[-1] == action_dataset[-1]``).  The
    previous version negated the gripper here, which sign-inverted all 600
    E062 collected episodes (root cause of the E062 kitchen collapse).

    Only real policy steps are recorded: the no-op wait prefix and injected
    no-op advances are excluded (the source dataset is the *no_noops* variant).
    """

    def __init__(self, jpeg_quality: int = 95):
        from io import BytesIO
        from PIL import Image as PILImage
        self._BytesIO = BytesIO
        self._PILImage = PILImage
        self.jpeg_quality = jpeg_quality
        self.head_jpeg: list[bytes] = []
        self.wrist_jpeg: list[bytes] = []
        self.state: list[np.ndarray] = []
        self.joint: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.plans: list[dict] = []

    def _encode(self, frame: np.ndarray) -> bytes:
        buf = self._BytesIO()
        self._PILImage.fromarray(np.ascontiguousarray(frame)).save(
            buf, format="JPEG", quality=self.jpeg_quality
        )
        return buf.getvalue()

    def on_plan(self, replan_idx: int, waypoints) -> None:
        self.plans.append({
            "replan": int(replan_idx),
            "t": None,  # filled by on_step count at record time
            "step_index": len(self.action),
            "waypoints": [
                {"proprio": [float(x) for x in np.asarray(p).ravel()], "duration": int(d)}
                for p, d in waypoints
            ],
        })

    def on_step(self, obs, action_env: np.ndarray) -> None:
        head = obs.get("agentview_image", obs.get("agentview_rgb"))
        wrist = obs.get("robot0_eye_in_hand_image", obs.get("robot0_eye_in_hand_rgb"))
        if head is None or wrist is None:
            raise RuntimeError("collection requires both agentview and wrist camera observations")
        self.head_jpeg.append(self._encode(head[::-1, ::-1]))
        self.wrist_jpeg.append(self._encode(wrist[::-1, ::-1]))
        self.state.append(get_proprio_from_obs(obs).astype(np.float32))
        joint = obs.get("robot0_joint_pos")
        self.joint.append(
            np.asarray(joint, dtype=np.float32) if joint is not None
            else np.zeros(7, dtype=np.float32)
        )
        act = np.asarray(action_env, dtype=np.float32).copy()
        # Record what the env actually executed: robosuite clips control inputs
        # to [-1, 1], and the source dataset's actions live inside that box.
        # The q99-unnormalized model output occasionally pokes past 1.0.
        act[:6] = np.clip(act[:6], -1.0, 1.0)
        # Gripper recorded AS-IS: env sign == dataset sign (+1 close / -1 open;
        # frozen contract table in E063_PREREG_NOTE §3).  A negation here is
        # exactly the E062 data bug — do not reintroduce it.
        self.action.append(act)


def _dump_collected_episode(collect_dir, task_idx, task_name, task_desc, trial,
                            success, ep_stats, initial_state, recorder, cfg):
    import hashlib
    import pickle
    init_arr = np.asarray(initial_state, dtype=np.float64)
    payload = {
        "schema_version": 1,
        "task_idx": int(task_idx),
        "task_name": str(task_name),
        "language_instruction": str(task_desc),
        "trial": int(trial),
        "success": bool(success),
        "steps_recorded": len(recorder.action),
        "steps_env": int(ep_stats.get("steps", -1)),
        "replans": int(ep_stats.get("replans", -1)),
        "budget_truncated": bool(ep_stats.get("budget_truncated", False)),
        "init_state": init_arr,
        "init_state_sha1": hashlib.sha1(init_arr.tobytes()).hexdigest(),
        "collector_checkpoint": str(cfg.get("joint_checkpoint", "")),
        "eval_seed": cfg.get("eval_seed"),
        "head_jpeg": recorder.head_jpeg,
        "wrist_jpeg": recorder.wrist_jpeg,
        "state": np.stack(recorder.state) if recorder.state else np.zeros((0, 8), np.float32),
        "joint_state": np.stack(recorder.joint) if recorder.joint else np.zeros((0, 7), np.float32),
        "action": np.stack(recorder.action) if recorder.action else np.zeros((0, 7), np.float32),
        "plans": recorder.plans,
    }
    out_dir = pathlib.Path(collect_dir) / f"task{int(task_idx):02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "succ" if success else "fail"
    out = out_dir / f"trial{int(trial):03d}_{suffix}.pkl"
    tmp = out.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, out)
    logger.info(f"  collected episode -> {out} ({len(recorder.action)} steps, success={success})")


def run_episode(
    vlm, ae_model, wp_tokenizer, norm_helper,
    env, initial_state, task_desc, cfg, device, pg_tok,
    noise_generator=None, decode_cfg=None, planner_generator=None,
    recorder=None,
):
    """Run one LIBERO episode with the two-stage waypoint pipeline.

    Behaviour flags (all default to the historical behaviour so that any change
    is an explicit, separately measurable arm -- see docs/10-waypoint-v2.md):

    ``eval.strict_step_budget`` (default True, **bug fix**)
        The horizon used to be checked only at replanning-cycle boundaries, so an
        episode could run up to ``num_waypoints * horizon_steps`` env steps past
        ``max_steps``.  Every ``env.step`` is now budget-checked.
    ``eval.segment_start_from`` (``predicted`` | ``actual``)
        ``predicted`` reuses the VLM's previous endpoint as the next segment's
        start condition (visual closed loop, proprioceptive *open* loop);
        ``actual`` re-reads proprio from the environment, which is what the AE
        actually saw during training.
    ``eval.stop_on_d0``
        Treat a terminal duration as an episode terminator rather than only
        breaking the inner waypoint loop.
    ``eval.replan_on_deviation`` / ``deviation_threshold`` / ``deviation_metric``
        Replan early when the achieved proprio drifts from the planned waypoint.
        The per-segment deviation histogram is recorded either way.

    Returns:
        success: bool
        replay_images: list of uint8 numpy images for video recording
        stats: dict of per-episode diagnostics
    """
    env.reset()
    obs = env.set_init_state(initial_state)

    rc = get_robot_config("libero")
    model_proprio_dim = cfg.get("model_proprio_dim", 32)
    actual_action_dim = rc.actual_action_dim
    max_steps = resolve_max_steps(cfg)
    num_steps_wait = cfg.get("num_steps_wait", 10)
    crop_scale = cfg.get("center_crop_scale") if cfg.get("center_crop", False) else None

    strict_budget = cfg.get("strict_step_budget", True)
    segment_start_from = cfg.get("segment_start_from", "predicted")
    if segment_start_from not in ("predicted", "actual"):
        raise ValueError(f"eval.segment_start_from must be 'predicted' or 'actual', got {segment_start_from!r}")
    stop_on_d0 = cfg.get("stop_on_d0", False)
    replan_on_deviation = cfg.get("replan_on_deviation", False)
    deviation_threshold = float(cfg.get("deviation_threshold", 0.15))
    deviation_metric = cfg.get("deviation_metric", "l2")
    if deviation_metric not in ("l2", "linf"):
        raise ValueError(f"eval.deviation_metric must be 'l2' or 'linf', got {deviation_metric!r}")
    # Diagnostics only, off unless asked for: the flag is read once here, so when
    # it is absent the per-step loop below does no extra work at all.
    record_step_states = bool(cfg.get("record_step_states", False))
    segment_records: list[dict] = []
    n_wp_dims = rc.continuous_proprio_dim + 1  # 6 continuous dims + binary gripper

    budget = max_steps + num_steps_wait
    ep_stats = _episode_stats()

    t = 0
    done = False
    reward = 0.0
    replay_images = []
    collect_video = not cfg.get("disable_video", False)
    replan_count = 0
    consecutive_d1_zero = 0
    terminate_episode = False

    dummy_action = np.zeros(7)
    while t < num_steps_wait:
        obs, _, done, _ = env.step(dummy_action)
        t += 1
        if done:
            ep_stats["steps"] = t
            return True, replay_images, ep_stats

    while t < budget and not done and not terminate_episode:
        agentview = obs.get("agentview_image", obs.get("agentview_rgb"))
        if collect_video and agentview is not None:
            head_frame = np.ascontiguousarray(agentview[::-1, ::-1])
            wrist_raw = obs.get("robot0_eye_in_hand_image", obs.get("robot0_eye_in_hand_rgb"))
            if wrist_raw is not None:
                wrist_frame = np.ascontiguousarray(wrist_raw[::-1, ::-1])
                if wrist_frame.shape[0] != head_frame.shape[0]:
                    from PIL import Image as PILImage
                    wrist_frame = np.array(
                        PILImage.fromarray(wrist_frame).resize(
                            (wrist_frame.shape[1], head_frame.shape[0]), PILImage.BILINEAR
                        )
                    )
                replay_images.append(np.concatenate([head_frame, wrist_frame], axis=1))
            else:
                replay_images.append(head_frame)

        images = get_libero_images(env, obs, center_crop_scale=crop_scale)
        proprio_raw = get_proprio_from_obs(obs)
        # Split into continuous + binary gripper
        continuous_raw, gripper_binary = rc.split_proprio(proprio_raw)
        continuous_norm = norm_helper.normalize_proprio(continuous_raw)

        t_vlm = time.time()
        waypoints = predict_waypoints(
            vlm, images, task_desc, wp_tokenizer,
            continuous_norm, gripper_binary, device,
            decode_cfg=decode_cfg, generator=planner_generator,
        )
        vlm_ms = (time.time() - t_vlm) * 1000
        ep_stats["planner_ms"].append(vlm_ms)
        accumulate_decode_stats(ep_stats["planner"], getattr(vlm, "last_decode_stats", None))

        replan_count += 1
        if recorder is not None and waypoints:
            recorder.on_plan(replan_count, waypoints)
        if not waypoints:
            logger.info(f"  [replan {replan_count}] VLM returned empty waypoints ({vlm_ms:.0f}ms), stopping")
            break

        valid_wps = [(p, d) for p, d in waypoints if d > 0]
        durations = [d for _, d in valid_wps]
        logger.info(
            f"  [replan {replan_count}] VLM: {len(waypoints)} waypoints, "
            f"{len(valid_wps)} valid, durations={durations}, vlm_time={vlm_ms:.0f}ms"
        )
        for wi, (pv, dur) in enumerate(waypoints):
            logger.info(f"    wp[{wi}]: proprio={_fmt_array(pv, 6)}, duration={dur}")

        # Build 7D start_wp: [continuous_norm(6D), gripper_binary(1D)]
        start_wp_7d = np.concatenate([continuous_norm, [float(gripper_binary)]])
        start_wp = pad_to_dim(start_wp_7d, model_proprio_dim)
        steps_this_cycle = 0
        budget_hit = False

        max_dur = cfg.get("horizon_steps", 32)
        for wp_idx, (proprio_values, duration) in enumerate(waypoints):
            # proprio_values is 7D: [6D continuous_norm, 1D gripper_binary]
            # `done` first: the padded tail of a plan legitimately carries
            # duration 0, so testing it before `done` inflated d0_terminations by
            # one on every episode that finished mid-segment.
            if done:
                break
            if duration == 0:
                ep_stats["d0_terminations"] += 1
                if stop_on_d0:
                    if wp_idx == 0:
                        # d1 == 0 on the very first waypoint is out of the training
                        # distribution; allow one more replan before giving up so a
                        # single bad plan cannot end an otherwise healthy episode.
                        consecutive_d1_zero += 1
                        if consecutive_d1_zero >= 2:
                            logger.info("    terminal duration twice in a row on wp[0]: ending episode")
                            terminate_episode = True
                    else:
                        terminate_episode = True
                break
            if duration < 0:
                logger.warning(f"    wp[{wp_idx}]: negative duration {duration}, skipping")
                continue
            if duration > max_dur:
                # Clamping is the correct guard, not a bug: `duration` is the
                # "reach end_wp in d steps" knob and end_wp is purely spatial, so
                # d=32 simply paces arrival at the last executable action slot.
                # Passing d=33 through unclamped would leave the endpoint short
                # and then poison the next segment via start_wp = end_wp.
                # What it *does* mean is a small speed-distribution shift on a
                # duration the AE never trained on (ae_dataset keeps dur <= 32),
                # so count it: the real fix is data-side (see
                # scripts/migrate_waypoint_indices.py).
                ep_stats["duration_overflow_count"] += 1
                logger.warning(f"    wp[{wp_idx}]: duration {duration} exceeds max {max_dur}, clamping")
                duration = max_dur

            if strict_budget and t >= budget:
                budget_hit = True
                break

            if segment_start_from == "actual" and wp_idx > 0:
                # The AE was trained with start_proprio measured from the same
                # frame as the images; reuse that convention instead of the
                # previous prediction.
                seg_raw = get_proprio_from_obs(obs)
                seg_cont_raw, seg_grip = rc.split_proprio(seg_raw)
                seg_cont_norm = norm_helper.normalize_proprio(seg_cont_raw)
                start_wp = pad_to_dim(
                    np.concatenate([seg_cont_norm, [float(seg_grip)]]), model_proprio_dim
                )

            end_wp = pad_to_dim(proprio_values, model_proprio_dim)

            fresh_images = get_libero_images(env, obs, center_crop_scale=crop_scale)
            t_ae = time.time()
            actions_norm = predict_actions(
                ae_model, fresh_images, task_desc, start_wp, end_wp, duration, device, pg_tok,
                noise_generator=noise_generator,
            )
            ae_ms = (time.time() - t_ae) * 1000

            num_execute = min(int(duration), actions_norm.shape[0])
            segment_clamped = False
            if strict_budget and num_execute > budget - t:
                # The budget cut this segment short.  Record it here: setting the
                # flag only when a *later* loop re-enters with t >= budget reads
                # `budget_truncated=0` in exactly the runs where the fix bit
                # (a 7x32 plan against a 230-step budget lands on t == budget
                # with no eighth waypoint to trip the entry check).
                num_execute = budget - t
                segment_clamped = True
                budget_hit = True
            logger.info(
                f"    ae[{wp_idx}]: shape={actions_norm.shape}, execute={num_execute}, "
                f"range=[{actions_norm.min():.3f}, {actions_norm.max():.3f}], ae_time={ae_ms:.0f}ms"
            )

            step_states: list[list[float]] = []
            for step_i in range(num_execute):
                if strict_budget and t >= budget:
                    budget_hit = True
                    break
                action_raw = norm_helper.unnormalize_actions(actions_norm[step_i, :actual_action_dim])

                gripper = action_raw[-1]
                gripper = gripper * 2.0 - 1.0
                gripper = np.sign(gripper)
                gripper = -gripper
                action_raw[-1] = gripper

                if recorder is not None:
                    recorder.on_step(obs, action_raw)
                obs, reward, done, info = env.step(action_raw)
                t += 1
                steps_this_cycle += 1

                if record_step_states:
                    # The same normalisation the planner conditions on and the
                    # deviation is measured in, so a step state is directly
                    # comparable with `proprio_values` (the planned endpoint).
                    st_raw = get_proprio_from_obs(obs)
                    st_cont_raw, st_grip = rc.split_proprio(st_raw)
                    st_norm = norm_helper.normalize_proprio(st_cont_raw)
                    step_states.append(
                        [*(round(float(v), 5) for v in st_norm), float(st_grip)]
                    )

                agentview = obs.get("agentview_image", obs.get("agentview_rgb"))
                if collect_video and agentview is not None:
                    head_frame = np.ascontiguousarray(agentview[::-1, ::-1])
                    wrist_raw = obs.get("robot0_eye_in_hand_image", obs.get("robot0_eye_in_hand_rgb"))
                    if wrist_raw is not None:
                        wrist_frame = np.ascontiguousarray(wrist_raw[::-1, ::-1])
                        if wrist_frame.shape[0] != head_frame.shape[0]:
                            from PIL import Image as PILImage
                            wrist_frame = np.array(
                                PILImage.fromarray(wrist_frame).resize(
                                    (wrist_frame.shape[1], head_frame.shape[0]), PILImage.BILINEAR
                                )
                            )
                        replay_images.append(np.concatenate([head_frame, wrist_frame], axis=1))
                    else:
                        replay_images.append(head_frame)

                if done:
                    break

            ep_stats["segments"] += 1
            consecutive_d1_zero = 0

            # --- achieved-vs-planned deviation (recorded regardless of the flag) ---
            # A budget-clamped segment never got the steps to reach its endpoint,
            # so its deviation is a truncation artefact -- and systematically the
            # largest one, which would land straight in the P95 used to calibrate
            # `deviation_threshold`.  Skip it.
            dev_l2 = dev_linf = float("nan")
            if not done and not segment_clamped:
                achieved_raw = get_proprio_from_obs(obs)
                achieved_cont_raw, achieved_grip = rc.split_proprio(achieved_raw)
                achieved_norm = norm_helper.normalize_proprio(achieved_cont_raw)
                n_cont = min(len(achieved_norm), len(proprio_values) - 1)
                delta = np.asarray(achieved_norm[:n_cont]) - np.asarray(proprio_values[:n_cont])
                dev_l2 = float(np.linalg.norm(delta))
                dev_linf = float(np.max(np.abs(delta))) if n_cont > 0 else float("nan")
                ep_stats["segment_deviation_l2"].append(dev_l2)
                ep_stats["segment_deviation_linf"].append(dev_linf)
                # NEW FIELD ONLY.  The two numbers above keep their exact values and
                # the early-replan decision below still reads dev_l2/dev_linf, so no
                # control flow depends on this.  Both terms are already on one scale:
                # the continuous dims are q99-normalised to ~[-1, 1] and the gripper
                # is 0/1, so an open-vs-closed miss contributes 1.0 -- deliberately
                # large, because that is the failure this field exists to see.
                grip_delta = float(achieved_grip) - float(proprio_values[-1])
                ep_stats["segment_deviation_l2_with_grip"].append(
                    float(math.hypot(dev_l2, grip_delta))
                )

            if record_step_states:
                # Unlike the deviation histogram, a budget-clamped segment's
                # *trajectory* is exactly what the "blocked vs oscillating"
                # question needs, so keep it -- and flag it, so a reader cannot
                # mistake it for a segment that ran to completion.
                segment_records.append({
                    "replan": replan_count,
                    "wp_idx": wp_idx,
                    "duration": int(duration),
                    "num_execute": int(num_execute),
                    "budget_clamped": bool(segment_clamped),
                    "t_end": int(t),
                    "start": [round(float(v), 5) for v in np.asarray(start_wp)[:n_wp_dims]],
                    "planned": [round(float(v), 5) for v in np.asarray(proprio_values)[:n_wp_dims]],
                    "achieved_steps": step_states,
                })

            start_wp = end_wp.copy()

            if budget_hit:
                break

            dev = dev_l2 if deviation_metric == "l2" else dev_linf
            if replan_on_deviation and not done and dev == dev and dev > deviation_threshold:
                logger.info(
                    f"    wp[{wp_idx}]: deviation {deviation_metric}={dev:.3f} > "
                    f"{deviation_threshold:.3f}, replanning early"
                )
                ep_stats["deviation_replans"] += 1
                break

        if budget_hit:
            ep_stats["budget_truncated"] = True
            break

        if steps_this_cycle == 0 and not done and not terminate_episode:
            if strict_budget and t >= budget:
                ep_stats["budget_truncated"] = True
                break
            logger.warning(f"  [replan {replan_count}] no actions executed, advancing with no-op")
            obs, reward, done, info = env.step(np.zeros(actual_action_dim))
            t += 1

    # Diagnostic-only fix (protocol unchanged: nothing below alters control flow
    # or the returned success bit).  The two `budget_truncated = True` sites above
    # both hang off `budget_hit`, which is only raised when a segment asks for
    # *strictly* more steps than remain (`num_execute > budget - t`).  A segment
    # that consumes exactly the remainder leaves `budget_hit` False; if it is also
    # the last waypoint of that plan, the waypoint loop ends normally, the no-op
    # branch is skipped (`steps_this_cycle > 0`) and the outer `while t < budget`
    # simply falls through -- so an episode that plainly ran out of budget was
    # recorded with `budget_truncated = False`.  Restate the flag from the exit
    # condition itself, which covers every path.
    if strict_budget and t >= budget and not (done and reward > 0):
        ep_stats["budget_truncated"] = True

    ep_stats["steps"] = t
    ep_stats["replans"] = replan_count
    if record_step_states:
        ep_stats["segment_states"] = segment_records
    if strict_budget:
        assert t <= budget, f"step budget violated: {t} > {budget}"
    logger.info(f"  episode done: steps={t}, replans={replan_count}, success={done and reward > 0}")
    return done and reward > 0, replay_images, ep_stats


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def _setup_seed(seed, deterministic_sdpa=False):
    """Set global RNG seeds for reproducible evaluation."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    if deterministic_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        logger.info(f"Eval seed set: {seed} (all RNGs seeded, CUDA deterministic, math SDPA only)")
    else:
        logger.info(f"Eval seed set: {seed} (all RNGs seeded, CUDA deterministic)")


def evaluate(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    eval_seed = cfg.get("eval_seed", None)
    noise_generator = None
    planner_generator = None
    if eval_seed is not None:
        _setup_seed(eval_seed, deterministic_sdpa=cfg.get("deterministic_sdpa", False))
        noise_generator = torch.Generator(device="cpu")
        noise_generator.manual_seed(eval_seed)
        # A *separate* stream for planner temperature sampling.  Sharing the AE's
        # generator would make the planner's draws shift the AE's noise (and vice
        # versa), so sharded (--task-start/--task-end) and resumed runs would not
        # be the same experiment as a single sequential run.
        planner_generator = torch.Generator(device="cpu")
        planner_generator.manual_seed(eval_seed + 1_000_003)

    t_total = time.time()
    use_joint = "joint_checkpoint" in cfg
    if use_joint:
        joint_model = load_joint(cfg, device)
        vlm = joint_model      # generate_waypoints() lives on joint model
        ae_model = joint_model  # sample_actions() lives on joint model
        logger.info(f"Joint model loaded (shared backbone): {time.time() - t_total:.1f}s")
    else:
        vlm = load_vlm(cfg, device)
        ae_model = load_ae(cfg, device)
        logger.info(f"Total model loading (separate VLM+AE): {time.time() - t_total:.1f}s")

    t0 = time.time()
    pg_tok = load_pg_tokenizer()
    logger.info(f"PaliGemma tokenizer loaded: {time.time() - t0:.1f}s")

    rc = get_robot_config("libero")
    stats = load_dataset_statistics(cfg["dataset_statistics_path"])
    norm_helper = NormalizationHelper(stats, cfg.get("norm_type", "q99"))
    if rc.action_norm_mask is not None:
        norm_helper.action_norm_mask = rc.action_norm_mask

    wp_tokenizer = WaypointTokenizer(
        proprio_dim=rc.continuous_proprio_dim,
        num_waypoints=cfg.get("num_waypoints", 7),
        max_token_len=cfg.get("vlm_max_token_len", cfg.get("max_token_len", 256)),
        use_gripper_token=True,
        n_gripper_slots=rc.num_grippers(),
    )

    if cfg.get("center_crop", False):
        if cfg.get("center_crop_scale") is None:
            # run_episode passes cfg["center_crop_scale"] straight through, so a
            # missing value used to mean "crop with whatever default that call
            # site happens to have".  Fail instead of silently cropping
            # differently from training.
            raise ValueError("eval.center_crop is true but eval.center_crop_scale is not set")
        logger.info(
            f"Center crop enabled: area_scale={cfg['center_crop_scale']}, "
            f"side_ratio={math.sqrt(cfg['center_crop_scale']):.4f}"
        )

    decode_cfg = WaypointDecodeConfig.from_dict(cfg)
    logger.info(f"Planner decode: {decode_cfg}")
    _check_decode_compatibility(decode_cfg, cfg, vlm, use_joint)

    video_out_path = pathlib.Path(cfg.get("video_out_path", "data/libero/videos"))
    video_out_path.mkdir(parents=True, exist_ok=True)

    # Per-step achieved states get their own file: appended per episode so a shard
    # that dies still leaves the episodes it finished, and kept out of the
    # diagnostics JSON that arm_report.py and the probes load whole.
    step_states_file = None
    if cfg.get("record_step_states", False):
        step_states_file = cfg.get("step_states_file")
        if step_states_file is None and cfg.get("diagnostics_file"):
            step_states_file = (
                str(pathlib.Path(cfg["diagnostics_file"]).with_suffix("")) + "_steps.jsonl"
            )
        if step_states_file is None:
            raise ValueError(
                "eval.record_step_states requires eval.step_states_file or --diagnostics-file"
            )
        step_states_file = pathlib.Path(step_states_file)
        step_states_file.parent.mkdir(parents=True, exist_ok=True)
        step_states_file.write_text("")  # truncate; the episode loop appends
        logger.info(f"Recording per-step achieved states to {step_states_file}")

    global _image_save_dir, _image_frame_idx
    _image_save_dir = pathlib.Path("image")
    _image_save_dir.mkdir(parents=True, exist_ok=True)
    _image_frame_idx = 0
    logger.info(f"Inference images will be saved to: {_image_save_dir.resolve()}")

    from libero.libero import benchmark

    task_suite_name = cfg.get("task_suite", "libero_object")
    bm = benchmark.get_benchmark_dict()[task_suite_name]()
    num_tasks = bm.n_tasks
    num_trials = cfg.get("num_trials_per_task", 3)

    task_start = cfg.get("task_start", 0)
    task_end = cfg.get("task_end", num_tasks)
    task_end = min(task_end, num_tasks)
    logger.info(f"Evaluating tasks [{task_start}, {task_end}) out of {num_tasks} total")

    # Load existing results for continuation (skip already-completed trials)
    existing_results = {}
    existing_results_file = cfg.get("existing_results_file")
    if existing_results_file and os.path.isfile(existing_results_file):
        with open(existing_results_file) as f:
            existing_results = json.load(f)
        logger.info(f"Loaded existing results from {existing_results_file} ({len(existing_results)} tasks)")

    results = {}
    total_success = 0
    total_episodes = 0
    episode_stats: list[dict] = []

    for task_idx in range(task_start, task_end):
        task = bm.get_task(task_idx)
        task_name = task.name
        task_desc = task.language
        initial_states = bm.get_task_init_states(task_idx)
        if cfg.get("init_states_file"):
            _custom = torch.load(cfg["init_states_file"], map_location="cpu", weights_only=False)
            if task_name not in _custom:
                raise KeyError(f"init_states_file has no entry for task {task_name!r}")
            initial_states = np.asarray(_custom[task_name], dtype=np.float64)
            logger.info(
                f"Task {task_idx}: initial states OVERRIDDEN from {cfg['init_states_file']} "
                f"({initial_states.shape[0]} states)"
            )
        target_trials = min(num_trials, len(initial_states))

        existing = existing_results.get(task_name, {})
        existing_trials = existing.get("trials", 0)
        existing_successes = existing.get("successes", 0)
        start_trial = min(existing_trials, target_trials)

        if start_trial >= target_trials:
            logger.info(
                f"Task {task_idx}: {task_name} already has {existing_trials} trials "
                f"(>= {target_trials}), skipping"
            )
            results[task_name] = existing
            total_success += existing_successes
            total_episodes += existing_trials
            continue

        if start_trial > 0:
            logger.info(
                f"Task {task_idx}: {task_name} continuing from trial {start_trial} "
                f"(existing: {existing_successes}/{existing_trials})"
            )

        env_args = {
            "bddl_file_name": bm.get_task_bddl_file_path(task_idx),
            "camera_heights": 256,
            "camera_widths": 256,
        }

        from libero.libero.envs import OffScreenRenderEnv
        t0 = time.time()
        env = OffScreenRenderEnv(**env_args)
        env.seed(7)
        logger.info(f"Env init for task {task_idx}: {time.time() - t0:.1f}s")

        successes = existing_successes
        total_success += existing_successes
        total_episodes += existing_trials

        trial_lo = int(cfg.get("trial_start", 0))
        stop_after = cfg.get("stop_after_successes")
        range_successes = 0
        range_trials = 0
        for trial in range(max(start_trial, trial_lo), target_trials):
            t_ep = time.time()
            logger.info(f"Task {task_idx}/{num_tasks}: {task_name} trial {trial}")
            if noise_generator is not None:
                noise_generator.manual_seed(eval_seed + task_idx * 1000 + trial)
            if planner_generator is not None:
                planner_generator.manual_seed(eval_seed + 1_000_003 + task_idx * 1000 + trial)
            recorder = None
            if cfg.get("collect_dir"):
                recorder = _TrajectoryRecorder(jpeg_quality=int(cfg.get("collect_jpeg_quality", 95)))
            success, replay_images, ep_stats = run_episode(
                vlm, ae_model, wp_tokenizer, norm_helper,
                env, initial_states[trial], task_desc, cfg, device, pg_tok,
                noise_generator=noise_generator, decode_cfg=decode_cfg,
                planner_generator=planner_generator, recorder=recorder,
            )
            if recorder is not None:
                _dump_collected_episode(
                    cfg["collect_dir"], task_idx, task_name, task_desc, trial,
                    bool(success), ep_stats, initial_states[trial], recorder, cfg,
                )
            segment_records = ep_stats.pop("segment_states", None)
            episode_stats.append({"task": task_name, "trial": trial, "success": bool(success), **ep_stats})
            if segment_records is not None and step_states_file is not None:
                with open(step_states_file, "a") as f:
                    f.write(json.dumps({
                        "task": task_name, "trial": trial, "success": bool(success),
                        "steps": ep_stats["steps"], "segments": segment_records,
                    }) + "\n")
            ep_secs = time.time() - t_ep
            if success:
                successes += 1
                total_success += 1
                range_successes += 1
            total_episodes += 1
            range_trials += 1

            suffix = "success" if success else "failure"
            task_segment = task_desc.replace(" ", "_")
            if replay_images and not cfg.get("disable_video", False):
                video_file = video_out_path / f"rollout_{task_segment}_t{trial}_{suffix}.mp4"
                imageio.mimwrite(
                    str(video_file),
                    [np.asarray(x) for x in replay_images],
                    fps=20,
                )
                logger.info(f"  -> {suffix.upper()} ({ep_secs:.1f}s, video: {video_file})")
            else:
                logger.info(f"  -> {suffix.upper()} ({ep_secs:.1f}s)")

            trials_done = trial + 1
            task_sr = successes / trials_done
            overall_sr = total_success / max(total_episodes, 1)
            logger.info(
                f"  [成功率] 当前任务: {successes}/{trials_done} = {task_sr:.2%} | "
                f"整体: {total_success}/{total_episodes} = {overall_sr:.2%}"
            )
            if stop_after is not None and range_successes >= int(stop_after):
                logger.info(
                    f"  stop_after_successes={stop_after} reached "
                    f"({range_successes} successes in {range_trials} trials); ending this shard early"
                )
                break

        if trial_lo > 0 or stop_after is not None:
            executed = existing_trials + range_trials
            results[task_name] = {
                "success_rate": (successes / executed) if executed else 0.0,
                "successes": successes,
                "trials": executed,
                "trial_start": trial_lo,
                "trial_end_exclusive": max(start_trial, trial_lo) + range_trials,
                "stop_after_successes": stop_after,
            }
        else:
            results[task_name] = {
                "success_rate": successes / target_trials,
                "successes": successes,
                "trials": target_trials,
            }
        env.close()

    overall_rate = total_success / max(total_episodes, 1)
    logger.info(f"\n{'='*60}")
    logger.info(f"Overall success rate: {overall_rate:.2%} ({total_success}/{total_episodes})")
    for name, r in results.items():
        logger.info(f"  {name}: {r['success_rate']:.2%} ({r['successes']}/{r['trials']})")
    logger.info(f"Total eval time: {time.time() - t_total:.1f}s")

    diagnostics = summarize_episode_stats(episode_stats, decode_cfg)
    if cfg.get("max_steps_override") is not None:
        # Written only when the override is in force: an arm run at the suite
        # default must produce the same diagnostics bytes as before this patch,
        # and a reader must never have to guess which budget a JSON came from.
        _suite_default = MAX_STEPS_MAP.get(cfg.get("task_suite", "libero_object"), 280)
        _max_steps = resolve_max_steps(cfg)
        _wait = cfg.get("num_steps_wait", 10)
        diagnostics["step_budget"] = {
            "suite_default_max_steps": _suite_default,
            "max_steps": _max_steps,
            "max_steps_override": int(cfg["max_steps_override"]),
            "num_steps_wait": _wait,
            "budget": _max_steps + _wait,
        }
        logger.warning(
            f"NON-DEFAULT STEP BUDGET: max_steps={_max_steps} (suite default "
            f"{_suite_default}), total budget {_max_steps + _wait}"
        )
    log_diagnostics(diagnostics)
    diag_file = cfg.get("diagnostics_file")
    if diag_file:
        with open(diag_file, "w") as f:
            json.dump({"summary": diagnostics, "episodes": episode_stats}, f, indent=2)
        logger.info(f"Planner diagnostics saved to {diag_file}")

    return results


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _check_decode_compatibility(decode_cfg, cfg, vlm, use_joint: bool) -> None:
    """Reject planner/decoder mismatches *before* the first episode runs.

    Both directions matter and only one used to be checked:
      * a block decoder needs the block planner parameters;
      * a block-trained checkpoint decoded token-at-a-time produces meaningless
        waypoints with no error at all (its hidden state at position p predicts
        the token AT p, not at p+1) -- so plausible success/latency numbers from
        garbage plans, which is worse than a crash.
    """
    planner_mode = cfg.get("planner_mode", "token_ar")
    if decode_cfg.impl == "block" and planner_mode != "block_ar":
        raise ValueError(
            "waypoint_decode.impl='block' requires planner_mode='block_ar' "
            "(the block query/slot parameters must exist in the checkpoint)"
        )
    if planner_mode == "block_ar" and decode_cfg.impl != "block":
        raise ValueError(
            f"planner_mode='block_ar' but waypoint_decode.impl='{decode_cfg.impl}': a "
            "block-trained checkpoint cannot be decoded token-at-a-time (it was trained "
            "with block-shifted inputs under a block-causal mask, so token-AR decoding "
            "silently returns meaningless waypoints). Set impl='block', or compare "
            "against a separately trained planner_mode='token_ar' checkpoint."
        )
    if not use_joint and decode_cfg.impl != "legacy":
        raise ValueError(
            f"waypoint_decode.impl='{decode_cfg.impl}' is only implemented for the joint "
            "model; this config uses the separate VLM+AE checkpoints, whose "
            "PI0WaypointVLM.generate_waypoints has no pluggable decoder. Use "
            "impl='legacy' here, or evaluate a joint checkpoint."
        )
    if use_joint and not _supports_decode_cfg(vlm):  # pragma: no cover - defensive
        raise ValueError("joint model does not accept decode_cfg; is the tree consistent?")


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray([v for v in values if v == v], dtype=np.float64)  # drop NaN
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def summarize_episode_stats(episode_stats: list[dict], decode_cfg=None) -> dict:
    """Aggregate the per-episode diagnostics into a reportable summary.

    Latency is reported as P50/P95 (not just the mean) because deployment cares
    about the tail, and the GPU-section and wall-clock numbers are kept apart
    because they are different clocks over overlapping intervals.
    """
    if not episode_stats:
        return {}
    planner_ms: list[float] = []
    dev_l2: list[float] = []
    dev_linf: list[float] = []
    dev_l2_grip: list[float] = []
    counters: dict[str, float] = {}
    for ep in episode_stats:
        planner_ms.extend(ep.get("planner_ms", []))
        dev_l2.extend(ep.get("segment_deviation_l2", []))
        dev_linf.extend(ep.get("segment_deviation_linf", []))
        # `.get(..., [])`: a diagnostics JSON written before this patch has no such
        # key, and _percentiles([]) is {}, so old files still summarise cleanly.
        dev_l2_grip.extend(ep.get("segment_deviation_l2_with_grip", []))
        for key in ("steps", "replans", "segments", "duration_overflow_count",
                    "d0_terminations", "deviation_replans"):
            counters[key] = counters.get(key, 0) + ep.get(key, 0)
        counters["budget_truncated"] = counters.get("budget_truncated", 0) + int(ep.get("budget_truncated", False))
        for key, value in ep.get("planner", {}).items():
            # `bool` is an `int` subclass, so exclude it explicitly: tri-state
            # flags must not be silently turned into totals.
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                counters[f"planner_{key}"] = counters.get(f"planner_{key}", 0) + value

    n_replans = max(counters.get("replans", 0), 1)
    per_call = {
        key[len("planner_"):]: value / n_replans
        for key, value in counters.items()
        if key.startswith("planner_")
    }
    return {
        "episodes": len(episode_stats),
        "decode": dataclasses.asdict(decode_cfg) if decode_cfg is not None else None,
        "planner_latency_ms": _percentiles(planner_ms),
        "deviation_l2": _percentiles(dev_l2),
        "deviation_linf": _percentiles(dev_linf),
        "deviation_l2_with_grip": _percentiles(dev_l2_grip),
        "totals": counters,
        "per_planner_call": per_call,
    }


def log_diagnostics(diagnostics: dict) -> None:
    if not diagnostics:
        return
    logger.info(f"{'-'*60}")
    logger.info("Planner diagnostics")
    lat = diagnostics.get("planner_latency_ms") or {}
    if lat:
        logger.info(
            f"  planner latency ms: P50={lat['p50']:.1f} P95={lat['p95']:.1f} "
            f"mean={lat['mean']:.1f} n={lat['n']}"
        )
    per_call = diagnostics.get("per_planner_call") or {}
    for key in ("transformer_forward_count", "lm_head_invocation_count",
                "generated_value_tokens", "generated_forced_tokens", "jacobi_iters", "plan_len"):
        if key in per_call:
            logger.info(f"  per call {key}: {per_call[key]:.2f}")
    for key in ("decode_transformer_ms", "decode_lm_head_ms", "prefix_prefill_ms",
                "image_encode_ms", "decode_python_ms"):
        if per_call.get(key):
            logger.info(f"  per call {key}: {per_call[key]:.2f}")
    for name in ("deviation_l2", "deviation_linf", "deviation_l2_with_grip"):
        stat = diagnostics.get(name) or {}
        if stat:
            logger.info(f"  {name}: P50={stat['p50']:.4f} P95={stat['p95']:.4f} max={stat['max']:.4f}")
    totals = diagnostics.get("totals") or {}
    logger.info(
        f"  totals: steps={totals.get('steps', 0)} replans={totals.get('replans', 0)} "
        f"segments={totals.get('segments', 0)} budget_truncated={totals.get('budget_truncated', 0)} "
        f"duration_overflow={totals.get('duration_overflow_count', 0)} "
        f"d0={totals.get('d0_terminations', 0)} dev_replans={totals.get('deviation_replans', 0)}"
    )


def _resolve_config(cfg):
    """Merge eval section into top-level for unified config support.

    Training configs can include an ``eval:`` section with eval-specific
    params.  This function merges those into the top-level dict so the
    rest of the eval code can access them uniformly.  Standalone eval
    configs (without ``eval:``) are unaffected.
    """
    eval_section = cfg.pop("eval", None)
    if eval_section:
        for key, value in eval_section.items():
            if value is not None:
                cfg[key] = value
    return cfg


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task-start", type=int, default=None, help="First task index (inclusive)")
    parser.add_argument("--task-end", type=int, default=None, help="Last task index (exclusive)")
    parser.add_argument("--results-file", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--existing-results", type=str, default=None,
                        help="Path to existing results JSON for trial continuation")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=None,
                        help="Prefer lora_ema.safetensors over lora.safetensors")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false",
                        help="Force raw LoRA weights even if eval.use_ema is true in the config")
    parser.add_argument("--decode-impl", type=str, default=None,
                        choices=list(DECODE_IMPLS), help="Planner decode implementation")
    parser.add_argument("--profile", action="store_true", help="Enable CUDA-event planner timing")
    parser.add_argument("--diagnostics-file", type=str, default=None,
                        help="Save the planner latency / deviation diagnostics JSON here")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_config(cfg)

    if args.task_start is not None:
        cfg["task_start"] = args.task_start
    if args.task_end is not None:
        cfg["task_end"] = args.task_end
    if args.existing_results is not None:
        cfg["existing_results_file"] = args.existing_results
    if args.use_ema is not None:
        cfg["use_ema"] = args.use_ema
    if args.decode_impl is not None:
        cfg.setdefault("waypoint_decode", {})
        cfg["waypoint_decode"] = {**cfg.get("waypoint_decode", {}), "impl": args.decode_impl}
    if args.profile:
        cfg["waypoint_decode"] = {**cfg.get("waypoint_decode", {}), "profile": True}
    if args.diagnostics_file is not None:
        cfg["diagnostics_file"] = args.diagnostics_file

    results = evaluate(cfg)

    if args.results_file:
        with open(args.results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()

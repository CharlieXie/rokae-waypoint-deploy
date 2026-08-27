"""Joint VLM + Action Expert model for waypoint VLA.

Shares a single PaliGemma backbone between:
  - VLM: autoregressive waypoint token prediction (CE loss)
  - AE: flow-matching action prediction (MSE loss)

Gradient strategy options:
  - "none": both losses freely update all parameters
  - "stop_gradient": Knowledge Insulation (Pi0.5 §5.2) — detach backbone
    K/V inside every attention layer so MSE loss cannot update backbone
    weights through cross-attention. Backbone only receives VLM CE gradients.
  - "scale_gradient": Soft KI — scale AE gradients flowing into backbone
    K/V by ``gradient_scale``.  scale=0 is full detach (same as
    stop_gradient), 0<s<1 attenuates, s=1 is equivalent to "none",
    s>1 amplifies AE gradient contribution to backbone K/V.
  - "freeze_backbone": freeze all paligemma backbone parameters entirely
"""

import dataclasses
import logging
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.lora_pytorch import compute_lm_logits as lora_compute_lm_logits
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_pytorch import sample_beta
from openpi.waypoint.ae_model import DURATION_MAX as _AE_DURATION_MAX
from openpi.waypoint.distributed import global_mean_loss
from openpi.waypoint.distributed import require_unweighted_global_mean
from openpi.waypoint.planner_block import WaypointBlockPlanner
from openpi.waypoint.planner_block import block_ce_loss
from openpi.waypoint.planner_block import build_block_ar_mask
from openpi.waypoint.planner_block import build_block_shifted_embeddings
from openpi.waypoint.planner_block import jitter_condition_proprio_tokens
from openpi.waypoint.planner_block import generate_waypoints_block
from openpi.waypoint.planner_block import locate_block_layout
from openpi.waypoint.planner_decode import DecodeStats
from openpi.waypoint.planner_decode import WaypointDecodeConfig
from openpi.waypoint.planner_decode import compute_family_logits
from openpi.waypoint.planner_decode import generate_waypoints_compact
from openpi.waypoint.planner_decode import generate_waypoints_jacobi
from openpi.waypoint.planner_decode import legacy_constrained_next_token
from openpi.waypoint.planner_decode import legacy_decode_budget
from openpi.waypoint.tokenizer import _DUR_TOKEN_ID
from openpi.waypoint.tokenizer import _DURATION_BASE
from openpi.waypoint.tokenizer import _GRIP_CLOSE_ID
from openpi.waypoint.tokenizer import _GRIP_OPEN_ID
from openpi.waypoint.tokenizer import _PROPRIO_BASE
from openpi.waypoint.tokenizer import _WP_TOKEN_ID
from openpi.waypoint.tokenizer import CURRENT_STATE_DURATION_CODE
from openpi.waypoint.tokenizer import DURATION_MAX
from openpi.waypoint.tokenizer import DURATION_N_BINS
from openpi.waypoint.tokenizer import PROPRIO_N_BINS

logger = logging.getLogger(__name__)

PLANNER_MODES = ("token_ar", "block_ar")
# Block-0 conditioning for waypoint-block AR.  "none" is the historical
# constant query block; "current_state" feeds the measured current state as an
# input-only "waypoint -1" block so block 0 is conditioned through the same
# token families and slot positions as every later block.
PLANNER_BLOCK0_COND_MODES = ("none", "current_state")


def validate_planner_conditioning(
    *,
    planner_mode: str,
    planner_block0_cond: str,
    planner_cond_jitter_prob: float,
    planner_cond_jitter_max_bins: int,
) -> None:
    """Reject conditioning options that cannot mean anything for this planner.

    Split out of ``PI0WaypointJoint.__init__`` so the argument contract can be
    unit-tested without allocating a 3B-parameter backbone -- otherwise the only
    way to cover it on CPU would be a hand-copied mirror that silently drifts.
    """
    if planner_block0_cond not in PLANNER_BLOCK0_COND_MODES:
        raise ValueError(
            f"planner_block0_cond must be one of {sorted(PLANNER_BLOCK0_COND_MODES)}, "
            f"got {planner_block0_cond!r}"
        )
    if planner_block0_cond != "none" and planner_mode != "block_ar":
        raise ValueError(
            f"planner_block0_cond={planner_block0_cond!r} only applies to "
            "planner_mode='block_ar' (token-AR has no block-0 query)"
        )
    if not 0.0 <= float(planner_cond_jitter_prob) <= 1.0:
        raise ValueError(
            f"planner_cond_jitter_prob must be in [0, 1], got {planner_cond_jitter_prob!r}"
        )
    if int(planner_cond_jitter_max_bins) < 1:
        raise ValueError(
            f"planner_cond_jitter_max_bins must be >= 1, got {planner_cond_jitter_max_bins!r}"
        )
    if float(planner_cond_jitter_prob) > 0.0 and planner_mode != "block_ar":
        raise ValueError(
            "planner_cond_jitter_prob only applies to planner_mode='block_ar'; the "
            "token-AR forward has no separate conditioning tensor to perturb"
        )

# DURATION_MAX is declared independently in tokenizer.py (int, defines the token
# range) and ae_model.py (float, used as a continuous divisor).  They must agree:
# tokenizer.DURATION_MAX sets which duration tokens exist, ae_model's value sets
# how the AE clamps/normalises them, and a silent divergence mis-clamps the
# duration embedding with no shape error to catch it.
assert int(_AE_DURATION_MAX) == DURATION_MAX, (
    f"duration bound mismatch: ae_model.DURATION_MAX={_AE_DURATION_MAX} vs "
    f"tokenizer.DURATION_MAX={DURATION_MAX}"
)

# Goal conditioning for the action expert (docs/14-waypoint-condition-authority-
# review.md §5.1, docs/16 round-5 revision).  "none" is the historical model:
# the endpoint reaches the expert only as an absolute-pose suffix token.
# "adarms_residual_v1" adds a zero-init residual to the per-layer AdaRMS
# condition, fed by the z-scored relative goal delta6 = end[:6] - start[:6]
# plus a target-gripper embedding.  "adarms_residual_v2" keeps that encoder
# but breaks the E050 label-leak shortcut (docs/15 §4, the delta6 ≈ action
# integral identity) with three optimizer-proof constraints: a fixed flow-time
# phase gate on the residual, goal noise at the input (supervision stays
# ground truth), and goal dropout to a learned null — noise and null both
# applied to the goal encoder input AND the legacy end suffix token, because
# leaving a clean end on either path is the docs/14 §5.6 leak backdoor.
GOAL_CONDITION_MODES = ("none", "adarms_residual_v1", "adarms_residual_v2")

# The v2 training/inference contract travels with the checkpoint metadata under
# exactly these keys; eval/merge reconstruction fails closed when a v2
# checkpoint does not record all of them (see goal_condition_kwargs_from_metadata).
GOAL_V2_METADATA_KEYS = (
    "goal_gate_t_lo",
    "goal_gate_t_hi",
    "goal_noise_sigma",
    "goal_dropout_prob",
    "goal_duration_jitter_steps",
)


def goal_gate(timestep: torch.Tensor, t_lo: float, t_hi: float) -> torch.Tensor:
    """Fixed (non-learnable) phase gate g(t) = clamp((t - t_lo)/(t_hi - t_lo), 0, 1).

    ``timestep`` is the flow-matching time of this repo's convention
    ``x_t = t*noise + (1-t)*actions`` (ae_forward), i.e. t=1 is the pure-noise
    /coarse end and t=0 the clean/fine end; ``sample_actions`` denoises from
    t=1 down to t=0.  With the pre-registered (t_lo, t_hi) = (0.2, 0.5) the
    goal residual is fully injected for t >= 0.5 (sketch phase) and exactly
    zero for t <= 0.2, so the refinement steps see vision/instruction only.
    The gate is a pure function of t with no parameters, so the optimizer
    cannot re-open the fine phase; training and the 10-step inference loop go
    through the same ``embed_suffix`` and therefore the same gate.
    """
    return ((timestep - t_lo) / (t_hi - t_lo)).clamp(0.0, 1.0)


def validate_goal_conditioning(
    *,
    goal_condition_mode: str,
    goal_delta_scale,
    goal_gate_t_lo: float = 0.2,
    goal_gate_t_hi: float = 0.5,
    goal_noise_sigma: float = 0.7,
    goal_dropout_prob: float = 0.15,
    goal_duration_jitter_steps: int = 0,
) -> None:
    """Reject goal-conditioning options that cannot be constructed.

    Split out of ``PI0WaypointJoint.__init__`` for the same reason as
    ``validate_planner_conditioning``: the argument contract must be coverable
    on CPU without allocating the 3B backbone.  The v2 knobs are validated
    unconditionally (they are constructor state even when inert) so a typo'd
    value can never ride along silently with any mode.
    """
    if goal_condition_mode not in GOAL_CONDITION_MODES:
        raise ValueError(
            f"goal_condition_mode must be one of {sorted(GOAL_CONDITION_MODES)}, "
            f"got {goal_condition_mode!r}"
        )
    if not 0.0 <= float(goal_gate_t_lo) < float(goal_gate_t_hi) <= 1.0:
        raise ValueError(
            "goal gate must satisfy 0 <= t_lo < t_hi <= 1, got "
            f"(t_lo={goal_gate_t_lo!r}, t_hi={goal_gate_t_hi!r})"
        )
    if float(goal_noise_sigma) < 0.0:
        raise ValueError(f"goal_noise_sigma must be >= 0, got {goal_noise_sigma!r}")
    if not 0.0 <= float(goal_dropout_prob) <= 1.0:
        raise ValueError(
            f"goal_dropout_prob must be in [0, 1], got {goal_dropout_prob!r}"
        )
    if int(goal_duration_jitter_steps) < 0:
        raise ValueError(
            f"goal_duration_jitter_steps must be >= 0, got {goal_duration_jitter_steps!r}"
        )
    if goal_condition_mode == "none":
        return
    if goal_delta_scale is None:
        raise ValueError(
            f"goal_condition_mode={goal_condition_mode!r} requires goal_delta_scale "
            "(the 6 per-dim robust displacement scales from "
            "scripts/compute_goal_delta_stats.py); refusing to guess a scale"
        )
    scale = [float(v) for v in goal_delta_scale]
    if len(scale) != 6:
        raise ValueError(f"goal_delta_scale must have 6 entries, got {len(scale)}")
    if any(v <= 0 for v in scale):
        raise ValueError(f"goal_delta_scale entries must be positive, got {scale}")


def goal_condition_kwargs_from_metadata(metadata, source: str) -> dict:
    """Constructor kwargs for the goal-conditioning contract a checkpoint records.

    Single shared reconstruction path for eval (``eval_libero.load_joint``) and
    ``scripts/merge_lora.py``, so the two can never drift apart.  Fail-closed:

    * a goal-conditioned checkpoint without a valid ``goal_delta_scale`` refuses
      to load (the scale is a model contract, docs/14 §5.1);
    * a v2 checkpoint whose metadata lacks any of ``GOAL_V2_METADATA_KEYS``
      refuses to load — the phase gate is part of the deployed forward pass, so
      guessing it would silently evaluate a different model (docs/16).
    """
    meta = metadata if isinstance(metadata, dict) else {}
    mode = meta.get("goal_condition_mode", "none")
    if mode == "none":
        return {"goal_condition_mode": "none", "goal_delta_scale": None}
    scale = meta.get("goal_delta_scale")
    if not scale or len(scale) != 6:
        raise ValueError(
            f"{source}: metadata records goal_condition_mode={mode!r} but no valid "
            f"goal_delta_scale (got {scale!r}); the scale travels with the "
            "checkpoint and cannot be guessed."
        )
    kwargs = {
        "goal_condition_mode": mode,
        "goal_delta_scale": [float(v) for v in scale],
    }
    if mode == "adarms_residual_v2":
        missing = [k for k in GOAL_V2_METADATA_KEYS if k not in meta]
        if missing:
            raise ValueError(
                f"{source}: goal_condition_mode='adarms_residual_v2' but metadata is "
                f"missing {missing}; the gate/noise/dropout contract travels with "
                "the checkpoint (v2 checkpoints always record it) and the phase "
                "gate changes the deployed forward pass, so it cannot be defaulted."
            )
        kwargs["goal_gate_t_lo"] = float(meta["goal_gate_t_lo"])
        kwargs["goal_gate_t_hi"] = float(meta["goal_gate_t_hi"])
        kwargs["goal_noise_sigma"] = float(meta["goal_noise_sigma"])
        kwargs["goal_dropout_prob"] = float(meta["goal_dropout_prob"])
        kwargs["goal_duration_jitter_steps"] = int(meta["goal_duration_jitter_steps"])
    return kwargs


# ---------------------------------------------------------------------------
# Planner geometry: ONE derivation shared by train / eval / merge
# ---------------------------------------------------------------------------
# Geometry keys the trainer records in metadata.pt (scripts/train_waypoint_joint.py)
# and that eval / merge read back.  Checkpoints written before these keys
# existed lack them; readers must fall back to the config and warn, not fail.
PLANNER_GEOMETRY_METADATA_KEYS = (
    "planner_proprio_dim",
    "planner_num_waypoints",
    "planner_use_gripper_token",
    "planner_n_gripper_slots",
    "planner_block_width",
)
# Every planner kwarg a checkpoint can pin (schema + geometry).
PLANNER_METADATA_KEYS = (
    "planner_mode",
    "planner_block0_cond",
    "planner_use_block_embed",
    *PLANNER_GEOMETRY_METADATA_KEYS,
)


def planner_block_width(
    proprio_dim: int, use_gripper_token: bool, n_gripper_slots: int = 1
) -> int:
    """Tokens per waypoint block: ``<wp>`` + proprio + gripper slots + ``<dur>`` + duration.

    Must equal ``WaypointTokenizer.tokens_per_waypoint`` for the same layout; the
    block planner's ``query_resid`` / ``slot_embed`` rows are indexed by it, so a
    P=6 single-gripper layout is 10 wide and a P=14 layout is 18 (19 with a
    second gripper slot, ``n_gripper_slots=2``).  Nothing may hardcode 10.
    """
    slots = int(n_gripper_slots) if use_gripper_token else 0
    return 1 + int(proprio_dim) + slots + 1 + 1


def planner_kwargs_from_config(cfg: dict, robot_config=None) -> dict:
    """Planner constructor kwargs derived from a training/eval config dict.

    The single derivation shared by ``scripts/train_waypoint_joint.py``,
    ``eval_libero.load_joint`` (and through it ``eval_calvin``) and
    ``scripts/merge_lora.py``.  The planner's slot geometry is a function of the
    robot layout (``robot_type`` + ``robot_config_kwargs`` ->
    ``RobotConfig.continuous_proprio_dim``), so any script that rebuilds the
    model from a config must take it from here rather than re-derive it by
    hand: the merge script used to omit these kwargs entirely and silently
    built a token-AR model (no ``block_planner``) for every block_ar checkpoint.
    """
    if robot_config is None:
        from openpi.waypoint.robot_config import get_robot_config

        robot_config = get_robot_config(
            cfg.get("robot_type", "libero"), **(cfg.get("robot_config_kwargs") or {})
        )
    return {
        "planner_mode": cfg.get("planner_mode", "token_ar"),
        "planner_num_waypoints": int(cfg.get("num_waypoints", 7)),
        "planner_proprio_dim": int(robot_config.continuous_proprio_dim),
        "planner_use_gripper_token": bool(cfg.get("use_gripper_token", True)),
        # One gripper slot per robot gripper (1 for LIBERO/CALVIN/single-arm,
        # 2 for dual-arm Rokae); a slot count that differs from the robot's
        # gripper count is exactly the "silently drop one arm" defect.
        "planner_n_gripper_slots": int(robot_config.num_grippers()),
        "planner_use_block_embed": bool(cfg.get("planner_use_block_embed", True)),
        "planner_block0_cond": cfg.get("planner_block0_cond", "none"),
    }


def planner_kwargs_from_metadata(metadata, fallback: dict, source: str) -> dict:
    """Overlay the planner schema/geometry a checkpoint recorded onto config kwargs.

    ``metadata.pt`` is the higher-priority source: it records what the trainer
    *actually* built (``PLANNER_METADATA_KEYS``), whereas the config handed to a
    merge may be a later edit.  Keys absent from the metadata (checkpoints that
    predate the geometry keys) fall back to ``fallback`` (normally
    ``planner_kwargs_from_config``).  Every override is logged as a warning so
    a config/checkpoint disagreement is visible.  Fail-closed on an internally
    inconsistent record: a recorded ``planner_block_width`` that disagrees with
    the recorded proprio/gripper layout cannot be trusted either way.
    """
    meta = metadata if isinstance(metadata, dict) else {}
    kwargs = dict(fallback)
    overridden = {}
    for key in PLANNER_METADATA_KEYS:
        if key == "planner_block_width" or key not in meta:
            continue
        value = meta[key]
        if key in ("planner_num_waypoints", "planner_proprio_dim", "planner_n_gripper_slots"):
            value = int(value)
        elif key in ("planner_use_block_embed", "planner_use_gripper_token"):
            value = bool(value)
        if key in kwargs and kwargs[key] != value:
            overridden[key] = (kwargs[key], value)
        kwargs[key] = value
    if "planner_block_width" in meta:
        expected = planner_block_width(
            kwargs["planner_proprio_dim"], kwargs["planner_use_gripper_token"],
            kwargs.get("planner_n_gripper_slots", 1),
        )
        if int(meta["planner_block_width"]) != expected:
            raise ValueError(
                f"{source}: metadata records planner_block_width="
                f"{int(meta['planner_block_width'])} but its own "
                f"planner_proprio_dim={kwargs['planner_proprio_dim']} / "
                f"planner_use_gripper_token={kwargs['planner_use_gripper_token']} / "
                f"planner_n_gripper_slots={kwargs.get('planner_n_gripper_slots', 1)} "
                f"imply {expected}; the record is inconsistent and cannot be trusted."
            )
    for key, (was, now) in overridden.items():
        logger.warning(
            "%s: metadata records %s=%r, overriding the config-derived %r",
            source, key, now, was,
        )
    return kwargs


class GoalConditionEncoder(nn.Module):
    """Relative-goal encoder for the AdaRMS residual (docs/14 §5.1, v1).

    ``forward(start_proprio, end_proprio)`` consumes the same padded 32-dim
    normalized proprio tensors that ``embed_suffix`` already receives:
    dims [0:6] are the q99-normalized continuous pose, dim 6 is the binary
    gripper.  It returns a ``[B, width]`` residual to be *added* to the
    flow-time ``adarms_cond``.

    Contract highlights (each has a unit test in
    ``tests/waypoint/test_goal_conditioning.py``):

    * ``delta6`` is z-scored by ``robust_delta_scale`` (buffer, recorded in the
      checkpoint metadata, produced by ``scripts/compute_goal_delta_stats.py``)
      so the pose signal enters at O(1) instead of the raw ~0.065 RMS floor
      documented in docs/13 §2.2(d).
    * pose and gripper branches are separate before fusion so the O(1) gripper
      bit cannot drown the pose delta inside a shared projection.
    * The final ``out`` layer is zero-initialised (adaLN-Zero / ControlNet
      style): at step 0 the residual is exactly zero, so a v1 model with
      otherwise identical weights is bitwise-equal to a "none" model.
    """

    def __init__(self, width: int, delta_scale) -> None:
        super().__init__()
        scale = torch.as_tensor([float(v) for v in delta_scale], dtype=torch.float32)
        if scale.shape != (6,) or bool((scale <= 0).any()):
            raise ValueError(f"delta_scale must be 6 positive floats, got {delta_scale!r}")
        self.register_buffer("delta_scale", scale)
        self.pose_in = nn.Linear(6, width)
        self.grip_embed = nn.Embedding(2, width)
        self.fuse = nn.Linear(2 * width, width)
        self.out = nn.Linear(width, width)
        self.reset_zero_init()

    def reset_zero_init(self) -> None:
        """(Re)apply the zero-init contract on the output layer.

        Called from ``__init__`` and again by eval's ``skip_init_weights()``
        construction path, which deliberately skips module ``reset_parameters``
        (mirrors the ``block_planner`` zero-init handling in
        ``eval_libero.load_joint``).
        """
        with torch.no_grad():
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, start_proprio: torch.Tensor, end_proprio: torch.Tensor) -> torch.Tensor:
        dtype = self.pose_in.weight.dtype
        delta6 = (end_proprio[:, :6] - start_proprio[:, :6]).to(dtype) / self.delta_scale
        pose_h = F.silu(self.pose_in(delta6))
        target_grip = end_proprio[:, 6].round().clamp(0, 1).long()
        grip_h = self.grip_embed(target_grip).to(dtype)
        fused = F.silu(self.fuse(torch.cat([pose_h, grip_h], dim=-1)))
        return self.out(fused)


class GoalConditionEncoderV2(GoalConditionEncoder):
    """v1 encoder plus the two learned nulls the v2 dropout contract needs.

    Goal dropout replaces the whole goal (pose + target gripper) with a learned
    null, and the null must land on BOTH ingestion paths (docs/14 §5.6):

    * ``null_residual`` stands in for the encoder's AdaRMS residual output.  It
      is injected through the same phase gate as a real goal, so at inference a
      CFG null branch sees exactly the geometry a dropped training sample saw.
    * ``null_end_embed`` stands in for the *legacy end suffix token* embedding
      (``proprio_encoder(end)``).  Noising or nulling only the residual while a
      clean end stays in the suffix token would leave a leak backdoor through
      the historical channel.

    Both are zero-init so the v1 step-0 bitwise-equivalence contract extends to
    v2 unchanged; the extra ``goal_encoder.null_*`` keys make the v1/v2 schema
    difference visible to the fail-closed loader.
    """

    def __init__(self, width: int, delta_scale) -> None:
        super().__init__(width, delta_scale)
        self.null_residual = nn.Parameter(torch.zeros(width))
        self.null_end_embed = nn.Parameter(torch.zeros(width))

    def reset_zero_init(self) -> None:
        super().reset_zero_init()
        with torch.no_grad():
            # getattr: the base __init__ calls reset_zero_init before the null
            # parameters exist; the eval skip_init path calls it again after.
            for name in ("null_residual", "null_end_embed"):
                param = getattr(self, name, None)
                if param is not None:
                    nn.init.zeros_(param)


class PI0WaypointJoint(nn.Module):
    """Joint VLM + AE model sharing a single PaliGemma backbone.

    The VLM head uses ``paligemma_with_expert.paligemma`` for autoregressive
    waypoint token prediction. The AE head uses the full
    ``paligemma_with_expert`` (backbone + gemma_expert) for flow-matching
    action prediction, exactly as in the standalone AE model.
    """

    # Class-level defaults, not just __init__ assignments: tests/waypoint/conftest.py
    # builds this class with ``PI0WaypointJoint.__new__`` to populate only the
    # planner parts, and a checkpoint unpickled from older code has no such
    # attribute either.  Both flags must read False on every construction path.
    skip_masked_cameras: bool = False
    trim_padded_tokens: bool = False
    # Same reasoning for the block-AR conditioning treatments: every construction
    # path must read the historical behaviour when the attribute was never set.
    planner_block0_cond: str = "none"
    planner_cond_jitter_prob: float = 0.0
    planner_cond_jitter_max_bins: int = 1
    planner_cond_jitter_seed: int | None = None
    _cond_jitter_generator = None
    # Same reasoning for goal conditioning: __new__-built test instances and
    # checkpoints unpickled from older code must read the historical behaviour
    # ("none") without touching __init__.  Deliberately NOT paired with a
    # class-level ``goal_encoder = None``: a plain class attribute wins the
    # standard attribute lookup over nn.Module.__getattr__, so it would shadow
    # the registered submodule and every read would return None.  Callers that
    # must tolerate pre-feature instances use getattr(model, "goal_encoder",
    # None) instead.
    goal_condition_mode: str = "none"
    # v2 contract knobs (docs/16).  Class-level for the same __new__/unpickle
    # reason; all of them are inert unless goal_condition_mode is v2.  The
    # defaults are the pre-registered E053 values.
    goal_gate_t_lo: float = 0.2
    goal_gate_t_hi: float = 0.5
    goal_noise_sigma: float = 0.7
    goal_dropout_prob: float = 0.15
    goal_duration_jitter_steps: int = 0
    goal_cond_seed: int | None = None
    _goal_cond_generator = None

    def __init__(
        self,
        config,
        vlm_max_token_len: int = 256,
        gradient_strategy: str = "none",
        gradient_scale: float = 0.1,
        aug_cfg: dict | None = None,
        planner_mode: str = "token_ar",
        planner_num_waypoints: int = 7,
        planner_proprio_dim: int = 6,
        planner_use_gripper_token: bool = True,
        planner_n_gripper_slots: int = 1,
        planner_use_block_embed: bool = True,
        planner_family_weights: dict[str, float] | None = None,
        planner_block0_cond: str = "none",
        planner_cond_jitter_prob: float = 0.0,
        planner_cond_jitter_max_bins: int = 1,
        planner_cond_jitter_seed: int | None = None,
        skip_masked_cameras: bool = False,
        trim_padded_tokens: bool = False,
        goal_condition_mode: str = "none",
        goal_delta_scale=None,
        goal_gate_t_lo: float = 0.2,
        goal_gate_t_hi: float = 0.5,
        goal_noise_sigma: float = 0.7,
        goal_dropout_prob: float = 0.15,
        goal_duration_jitter_steps: int = 0,
        goal_cond_seed: int | None = None,
    ):
        super().__init__()
        assert gradient_strategy in ("none", "stop_gradient", "scale_gradient", "freeze_backbone"), (
            f"Unknown gradient_strategy: {gradient_strategy}"
        )
        assert planner_mode in PLANNER_MODES, f"Unknown planner_mode: {planner_mode}"
        self.config = config
        self.vlm_max_token_len = vlm_max_token_len
        self.gradient_strategy = gradient_strategy
        self.gradient_scale = gradient_scale
        self.aug_cfg = aug_cfg
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.duration_max = getattr(config, "duration_max", DURATION_MAX)

        # --- planner geometry (must mirror the WaypointTokenizer layout) ---
        self.planner_mode = planner_mode
        self.planner_num_waypoints = planner_num_waypoints
        self.planner_proprio_dim = planner_proprio_dim
        self.planner_use_gripper_token = planner_use_gripper_token
        if planner_use_gripper_token and int(planner_n_gripper_slots) < 1:
            raise ValueError(f"planner_n_gripper_slots must be >= 1, got {planner_n_gripper_slots}")
        self.planner_n_gripper_slots = int(planner_n_gripper_slots) if planner_use_gripper_token else 0
        self.planner_family_weights = planner_family_weights
        # --- block-AR conditioning options (both default to the historical path) ---
        validate_planner_conditioning(
            planner_mode=planner_mode,
            planner_block0_cond=planner_block0_cond,
            planner_cond_jitter_prob=planner_cond_jitter_prob,
            planner_cond_jitter_max_bins=planner_cond_jitter_max_bins,
        )
        self.planner_block0_cond = planner_block0_cond
        self.planner_cond_jitter_prob = float(planner_cond_jitter_prob)
        self.planner_cond_jitter_max_bins = int(planner_cond_jitter_max_bins)
        self.planner_cond_jitter_seed = planner_cond_jitter_seed
        # Lazily built on the first jittered forward so the draws come from a
        # stream of their own: sharing the process generator would make the
        # augmentation shift every other torch draw, and a matched control needs
        # the treatment arm's *data* stream to stay identical to the baseline's.
        self._cond_jitter_generator = None
        # Opt-in (default False keeps the historical graph): see embed_prefix().
        self.skip_masked_cameras = skip_masked_cameras
        # Opt-in (default False keeps the historical graph): see _trim_len().
        self.trim_padded_tokens = trim_padded_tokens
        # Mirrors WaypointTokenizer: <wp> + P proprio + N gripper slots + <dur> + d
        grip = self.planner_n_gripper_slots
        self.planner_block_width = planner_block_width(
            planner_proprio_dim, planner_use_gripper_token, planner_n_gripper_slots
        )
        self.planner_dur_delim_pos = 1 + planner_proprio_dim + grip
        self.last_decode_stats = None
        # Counters for the legacy decoder, so the benchmark table can show what
        # the compact/block paths removed rather than asserting it.
        self._legacy_forward_count = 0
        self._legacy_lm_head_count = 0
        self._legacy_value_token_count = 0
        self._legacy_forced_token_count = 0

        # --- Shared backbone + expert (single instance) ---
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        expert_width = action_expert_config.width

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        # --- AE-specific projections ---
        self.action_in_proj = nn.Linear(self.action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, self.action_dim)
        self.proprio_encoder = nn.Linear(self.action_dim, expert_width)
        # time_mlp_in is W->W (matches Pi0.5 base exactly, so pretrained
        # weights load without shape-mismatch skipping). Duration is now
        # conditioned via a learned token inserted into the expert suffix
        # sequence (see embed_suffix), not via AdaRMSNorm.
        self.time_mlp_in = nn.Linear(expert_width, expert_width)
        self.time_mlp_out = nn.Linear(expert_width, expert_width)
        self.duration_embed = nn.Embedding(DURATION_N_BINS, expert_width)

        # --- block planner (waypoint-block AR) ---
        # Only instantiated in block mode so that a token-AR checkpoint never
        # carries silently-uninitialised planner parameters and vice versa: the
        # state-dict key set is itself the schema check.
        self.block_planner = None
        if planner_mode == "block_ar":
            self.block_planner = WaypointBlockPlanner(
                hidden_width=paligemma_config.width,
                block_width=self.planner_block_width,
                max_blocks=planner_num_waypoints,
                use_block_embed=planner_use_block_embed,
            )

        # --- goal conditioning (docs/14 §5.1) ---
        # Constructed AFTER every legacy module on purpose: mode="none" keeps
        # the historical state-dict key set (the key set is the schema check,
        # same as block_planner), and in v1 mode all legacy parameters consume
        # the model_init RNG stream in the historical order, so they initialise
        # bit-identically to a mode="none" run with the same training seed.
        validate_goal_conditioning(
            goal_condition_mode=goal_condition_mode,
            goal_delta_scale=goal_delta_scale,
            goal_gate_t_lo=goal_gate_t_lo,
            goal_gate_t_hi=goal_gate_t_hi,
            goal_noise_sigma=goal_noise_sigma,
            goal_dropout_prob=goal_dropout_prob,
            goal_duration_jitter_steps=goal_duration_jitter_steps,
        )
        self.goal_condition_mode = goal_condition_mode
        self.goal_gate_t_lo = float(goal_gate_t_lo)
        self.goal_gate_t_hi = float(goal_gate_t_hi)
        self.goal_noise_sigma = float(goal_noise_sigma)
        self.goal_dropout_prob = float(goal_dropout_prob)
        self.goal_duration_jitter_steps = int(goal_duration_jitter_steps)
        self.goal_cond_seed = goal_cond_seed
        # Same dedicated-stream reasoning as _cond_jitter_generator: the v2 goal
        # noise / dropout / duration-jitter draws must not shift any other torch
        # draw, or the matched control's data stream stops being identical.
        self._goal_cond_generator = None
        self.goal_encoder = None
        if goal_condition_mode == "adarms_residual_v1":
            self.goal_encoder = GoalConditionEncoder(expert_width, goal_delta_scale)
        elif goal_condition_mode == "adarms_residual_v2":
            # Shares the v1 encoder body, so with the same training seed every
            # shared parameter draws the same model_init RNG values as a v1 (or
            # none) run; the extra nulls are zeros and consume no RNG.
            self.goal_encoder = GoalConditionEncoderV2(expert_width, goal_delta_scale)

        self.gradient_checkpointing_enabled = False

        # Apply freeze_backbone strategy
        if self.gradient_strategy == "freeze_backbone":
            for param in self.paligemma_with_expert.paligemma.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        # VLM path: use HF API so that the inner GemmaModel / SiglipEncoder
        # (which actually check the flag) get gradient_checkpointing=True.
        # Just setting the attribute on the outer wrapper does NOT propagate.
        self.paligemma_with_expert.paligemma.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # AE path: the custom paligemma_with_expert.forward() uses its own
        # torch.utils.checkpoint.checkpoint() calls, keyed off this attribute.
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        self.paligemma_with_expert.allow_gradient_checkpointing_disable = False

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.gradient_checkpointing_disable()
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        # PaliGemmaWithExpertModel.forward() force-enables checkpointing on the
        # expert on every training forward; without this opt-in the two lines
        # above are undone before they are ever read (gemma_pytorch.py).
        self.paligemma_with_expert.allow_gradient_checkpointing_disable = True

    def _ckpt(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    # ------------------------------------------------------------------
    # LM logits helper (supports LoRAEmbedding)
    # ------------------------------------------------------------------
    def _compute_lm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute LM logits from hidden states.

        Thin wrapper over the shared :func:`lora_compute_lm_logits`; see there
        for why no call site may use ``embed_tokens.weight`` directly.
        """
        embed_layer = self.paligemma_with_expert.paligemma.language_model.embed_tokens
        return lora_compute_lm_logits(embed_layer, hidden)

    # ------------------------------------------------------------------
    # DDP-compatible forward dispatcher
    # ------------------------------------------------------------------
    def forward(self, mode: str, **kwargs):
        """Dispatch to vlm_forward or ae_forward.

        This method exists so that DDP can route through ``__call__`` and
        properly track which parameters are used in each forward pass.

        Args:
            mode: ``"vlm"`` or ``"ae"``.
            **kwargs: forwarded to the corresponding method.
        """
        if mode == "vlm":
            return self.vlm_forward(
                kwargs["batch"],
                return_breakdown=kwargs.get("return_breakdown", False),
            )
        elif mode == "ae":
            return self.ae_forward(**kwargs)
        else:
            raise ValueError(f"Unknown forward mode: {mode}")

    # ------------------------------------------------------------------
    # AE embedding helpers (from ae_model.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _trim_len(*masks) -> int:
        """Length of the shortest prefix that still contains every occupied slot.

        The tokenizer pads both the planner token block (256 slots, ~125 used) and
        the AE prompt (64 slots, 24 used) to a fixed width.  A padded slot is
        False in ``token_mask``, so ``make_att_2d_masks`` gives it additive
        -2.38e38 as an attention key and its softmax weight underflows to exactly
        0; ``position_ids = cumsum(pad_mask) - 1`` is likewise unchanged by it, and
        it carries no supervision.  Dropping the tail is therefore exact, and it
        takes the whole padded region out of every GEMM in the backbone.

        Computed from the *last* occupied index rather than from a count, so the
        result stays correct even if a mask were ever not trailing-padded.  The
        length is rounded up to a multiple of 8 to keep GEMM shapes tensor-core
        friendly, and the union over every supplied mask is used so a position
        that is supervised or attention-relevant can never be cut.
        """
        occupied = None
        for m in masks:
            if m is None:
                continue
            mb = m.bool() if m.dtype != torch.bool else m
            occupied = mb if occupied is None else (occupied | mb)
        if occupied is None:
            return 0
        length = occupied.shape[-1]
        idx = torch.arange(length, device=occupied.device)
        last = int((occupied.any(0).long() * (idx + 1)).max().item())
        return min(length, ((last + 7) // 8) * 8)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Embed images + language tokens for PaliGemma prefix.

        When ``skip_masked_cameras`` is set, a camera whose pad mask is False for
        *every* sample in the batch is dropped instead of encoded.  This is exact,
        not an approximation:

          * its 256 prefix positions are all-False in ``pad_masks``, so
            ``make_att_2d_masks`` gives them additive -2.38e38 as attention keys
            and their softmax weight underflows to exactly 0;
          * ``position_ids`` is ``cumsum(pad_masks) - 1``, and an all-False block
            adds 0 to the cumsum, so every surviving token keeps its position id;
          * the prefix hidden states are discarded by ``ae_forward`` (only
            ``suffix_out`` is read), so the dropped queries' outputs are unused.

        The skip happens *after* ``preprocess_observation_pytorch``, deliberately:
        image augmentation draws from the global RNG once per camera key, so
        skipping earlier would shift the RNG stream and change the augmentation of
        the *real* cameras.  Leaving preprocessing untouched keeps the surviving
        cameras bit-identical.

        This matters for single-arm robots: WaypointAECollator pads every absent
        camera up to the model's 3-camera layout with a zero image and an all-False
        mask (ae_dataset.py), so LIBERO spends a full 27-layer SigLIP encode and
        256 extra sequence positions per step on a blank frame.
        """
        embs, pad_masks, att_masks = [], [], []

        for img, img_mask in zip(images, img_masks, strict=True):
            if self.skip_masked_cameras and not bool(img_mask.any()):
                continue
            img_emb = self._ckpt(self.paligemma_with_expert.embed_image, img)
            bsize, n_img = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, n_img))
            att_masks += [0] * n_img

        if self.trim_padded_tokens and lang_masks is not None:
            keep = self._trim_len(lang_masks)
            if 0 < keep < lang_tokens.shape[1]:
                lang_tokens = lang_tokens[:, :keep]
                lang_masks = lang_masks[:, :keep]
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

    def embed_suffix(self, start_proprio, end_proprio, noisy_actions, timestep, duration,
                     *, goal_null_mask=None):
        """Embed proprio conditioning + duration token + noisy actions for expert.

        Suffix sequence: ``[start_wp, end_wp, dur_token, action_1, ..., action_H]``
        AdaRMSNorm conditioning uses the flow-matching timestep only (no duration);
        duration is injected into the sequence as a learned discrete token.

        ``goal_null_mask`` (v2 only, ``[B]`` bool) marks samples whose goal is
        replaced by the learned null — on BOTH paths: the end suffix token
        becomes ``null_end_embed`` and the AdaRMS residual becomes
        ``null_residual`` (docs/14 §5.6: a clean end left on either path is a
        leak backdoor).  Callers pass it during v2 training dropout and for the
        CFG null branch at inference; ``None`` means every sample keeps its goal.
        """
        if goal_null_mask is not None and self.goal_condition_mode != "adarms_residual_v2":
            raise ValueError(
                "goal_null_mask requires goal_condition_mode='adarms_residual_v2' "
                f"(got {self.goal_condition_mode!r}); other modes have no learned null"
            )
        embs, pad_masks, att_masks = [], [], []
        device = noisy_actions.device
        bsize = noisy_actions.shape[0]
        expert_width = self.action_in_proj.out_features

        proprio_input = torch.stack([start_proprio, end_proprio], dim=1)
        proprio_input = proprio_input.to(self.proprio_encoder.weight.dtype)
        proprio_emb = self._ckpt(self.proprio_encoder, proprio_input)
        if goal_null_mask is not None:
            # Null path (i) of two: replace the legacy end suffix token embedding.
            null_tok = self.goal_encoder.null_end_embed.to(proprio_emb.dtype)
            proprio_emb = torch.cat(
                [
                    proprio_emb[:, :1],
                    torch.where(
                        goal_null_mask[:, None, None],
                        null_tok.expand(bsize, 1, expert_width),
                        proprio_emb[:, 1:2],
                    ),
                ],
                dim=1,
            )
        embs.append(proprio_emb)
        pad_masks.append(torch.ones(bsize, 2, dtype=torch.bool, device=device))
        att_masks += [1, 0]

        dur_int = duration.clamp(0, DURATION_MAX).long()
        dur_tok = self.duration_embed(dur_int).unsqueeze(1)  # [B, 1, W]
        dur_tok = dur_tok.to(proprio_emb.dtype)
        embs.append(dur_tok)
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
        att_masks += [0]

        time_emb = create_sinusoidal_pos_embedding(
            timestep, expert_width, min_period=4e-3, max_period=4.0, device=device,
        ).to(self.time_mlp_in.weight.dtype)
        x = self.time_mlp_in(time_emb)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        adarms_cond = F.silu(x)
        if self.goal_condition_mode == "adarms_residual_v1":
            # docs/14 §5.1: adarms_cond = time_cond + goal_residual.  The
            # residual is zero at init (GoalConditionEncoder.out is zero-init),
            # so mode=v1 at step 0 is bitwise-equal to mode=none.  Every
            # AdaRMS layer has its own modulation dense, so this single vector
            # conditions all expert layers.
            goal_res = self.goal_encoder(start_proprio, end_proprio)
            adarms_cond = adarms_cond + goal_res.to(adarms_cond.dtype)
        elif self.goal_condition_mode == "adarms_residual_v2":
            # docs/16: adarms_cond = time_cond + g(t) * goal_residual, with the
            # fixed phase gate g(t) of goal_gate() — full injection in the
            # coarse/noise phase, exactly zero in the refinement phase, in
            # training and inference alike.  Zero-init keeps step-0 bitwise
            # equal to mode=none.  Null path (ii) of two: a dropped sample's
            # residual is the learned null, gated identically to a real goal.
            goal_res = self.goal_encoder(start_proprio, end_proprio)
            if goal_null_mask is not None:
                goal_res = torch.where(
                    goal_null_mask[:, None],
                    self.goal_encoder.null_residual.to(goal_res.dtype).expand_as(goal_res),
                    goal_res,
                )
            gate = goal_gate(
                timestep.to(adarms_cond.dtype), self.goal_gate_t_lo, self.goal_gate_t_hi
            )[:, None]
            adarms_cond = adarms_cond + gate * goal_res.to(adarms_cond.dtype)

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

    # ------------------------------------------------------------------
    # VLM forward (CE loss on waypoint tokens)
    # ------------------------------------------------------------------
    @staticmethod
    def _vlm_token_loss_breakdown(
        loss_per_token: torch.Tensor,
        targets: torch.Tensor,
        block_indices: torch.Tensor,
        num_blocks: int,
    ) -> dict[str, torch.Tensor]:
        """Return additive VLM CE statistics by family and waypoint block.

        Every ``vlm_ce_sum_*`` value is the numerator of a full-vocabulary
        cross-entropy and its matching ``vlm_tokens_*`` value is the
        denominator.  These are deliberately *not* local means: callers can
        accumulate them across arbitrary batches and ranks before dividing,
        without biasing the result toward a small batch/rank.

        ``block_indices`` is aligned one-to-one with the active targets.  It is
        derived from the tokenizer's structural waypoint grid in both token-AR
        and block-AR mode, so the reported block slices have identical meaning.
        The returned tensors are detached inside this helper and never
        participate in the training objective.
        """
        if loss_per_token.ndim != 1 or targets.ndim != 1 or block_indices.ndim != 1:
            raise ValueError("loss_per_token, targets and block_indices must all be 1-D")
        if not (loss_per_token.numel() == targets.numel() == block_indices.numel()):
            raise ValueError(
                "VLM breakdown inputs must have equal lengths: "
                f"loss={loss_per_token.numel()}, targets={targets.numel()}, "
                f"blocks={block_indices.numel()}"
            )
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        proprio_lo = _PROPRIO_BASE - PROPRIO_N_BINS + 1
        proprio_hi = _PROPRIO_BASE
        duration_lo = _DURATION_BASE - DURATION_N_BINS + 1
        duration_hi = _DURATION_BASE

        is_proprio = (targets >= proprio_lo) & (targets <= proprio_hi)
        is_duration = (targets >= duration_lo) & (targets <= duration_hi)
        is_gripper = (targets == _GRIP_OPEN_ID) | (targets == _GRIP_CLOSE_ID)
        is_other = ~(is_proprio | is_duration | is_gripper)

        metrics: dict[str, torch.Tensor] = {}
        ce = loss_per_token.detach()

        def add(name: str, mask: torch.Tensor) -> None:
            # Accumulate before any division, and use float64 at the point of
            # summation rather than after a rounded float32 batch sum.  These
            # tensors are detached observability data only.
            metrics[f"vlm_ce_sum_{name}"] = ce[mask].sum(dtype=torch.float64)
            metrics[f"vlm_tokens_{name}"] = mask.sum()

        add("all", torch.ones_like(targets, dtype=torch.bool))
        for name, mask in (
            ("proprio", is_proprio),
            ("gripper", is_gripper),
            ("duration", is_duration),
            ("other", is_other),
        ):
            add(name, mask)
            count = metrics[f"vlm_tokens_{name}"].to(torch.float64)
            # Compatibility aliases for the offline diagnostics packaged with
            # older checkpoints.  The trainer intentionally ignores these
            # local means and reduces the additive pairs above instead.
            metrics[f"vlm_loss_{name}"] = metrics[f"vlm_ce_sum_{name}"] / count.clamp(min=1)

        for block_idx in range(num_blocks):
            add(f"block_{block_idx}", block_indices == block_idx)
        add("block_1plus", block_indices >= 1)
        return metrics

    def vlm_forward(
        self,
        batch: dict,
        return_breakdown: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Planner training forward; dispatches on ``planner_mode``.

        Both modes consume *the same* batch produced by ``WaypointVLMDataset``,
        so a token-AR vs block-AR comparison changes only the architecture.
        """
        if self.planner_mode == "block_ar":
            return self._vlm_forward_block(batch, return_breakdown)
        return self._vlm_forward_token_ar(batch, return_breakdown)

    def _jitter_generator(self, device) -> "torch.Generator | None":
        """Dedicated RNG for conditioning jitter, seeded once per process.

        Kept separate from the process-global stream so that enabling the
        augmentation does not shift any other torch draw; with
        ``planner_cond_jitter_seed`` unset it falls back to the ambient stream
        (``None``), which is the same nondeterminism contract an unseeded run
        already has.
        """
        if self.planner_cond_jitter_seed is None:
            return None
        cached = self._cond_jitter_generator
        if cached is None or cached.device != torch.device(device):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.planner_cond_jitter_seed))
            self._cond_jitter_generator = generator
        return self._cond_jitter_generator

    def _goal_cond_generator_for(self, device) -> "torch.Generator | None":
        """Dedicated RNG for the v2 goal corruptions, seeded once per process.

        Same contract as ``_jitter_generator``: with ``goal_cond_seed`` unset it
        falls back to the ambient stream (``None``); seeded, it keeps the flow
        noise / time / augmentation draws of a v2 run bitwise identical to the
        matched control's.
        """
        if self.goal_cond_seed is None:
            return None
        cached = self._goal_cond_generator
        if cached is None or cached.device != torch.device(device):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.goal_cond_seed))
            self._goal_cond_generator = generator
        return self._goal_cond_generator

    def _sample_goal_corruptions(self, end_proprio, duration):
        """Draw the v2 training-contract corruptions for one batch (docs/16 B/C).

        Returns ``(end_for_cond, dur_for_cond, goal_null_mask)``.  These feed
        ``embed_suffix`` (conditioning) ONLY — the supervised actions, the
        action pad mask and the loss keep the ground truth, which is the whole
        point: the goal input is made unreliable while the target stays exact,
        so the delta6-integrator shortcut of docs/15 §4 stops being available.

        * Goal noise: per sample, per dim, ``N(0, sigma)`` in the z-scored
          delta6 domain (``sigma * delta_scale`` in normalized proprio units),
          added to ``end[:6]``.  One tensor is returned and used by BOTH the
          goal encoder and the legacy end suffix token — noising only one path
          would leave the clean end readable through the other (docs/14 §5.6).
        * Goal dropout: with prob ``goal_dropout_prob`` the sample's goal is
          replaced by the learned null via ``goal_null_mask`` (both paths, see
          ``embed_suffix``).
        * Duration jitter (optional, off when ``goal_duration_jitter_steps=0``):
          integer jitter on the duration *input token* only, clamped to
          ``[1, duration_max]``.
        """
        bsize = end_proprio.shape[0]
        device = end_proprio.device
        generator = self._goal_cond_generator_for(device)

        end_for_cond = end_proprio
        sigma = float(self.goal_noise_sigma)
        if sigma > 0.0:
            eps = torch.randn(
                bsize, 6, device=device, dtype=torch.float32, generator=generator
            ) * sigma
            scale = self.goal_encoder.delta_scale.to(device=device)
            end_for_cond = end_proprio.clone()
            end_for_cond[:, :6] = end_for_cond[:, :6] + (eps * scale).to(end_proprio.dtype)

        goal_null_mask = None
        p = float(self.goal_dropout_prob)
        if p > 0.0:
            goal_null_mask = (
                torch.rand(bsize, device=device, generator=generator) < p
            )

        dur_for_cond = duration
        k = int(self.goal_duration_jitter_steps)
        if k > 0:
            jit = torch.randint(
                -k, k + 1, (bsize,), device=device, generator=generator
            )
            dur_for_cond = (duration + jit.to(duration.dtype)).clamp(
                1.0, float(self.duration_max)
            )
        return end_for_cond, dur_for_cond, goal_null_mask

    def _validate_distributed_vlm_reduction(self) -> None:
        require_unweighted_global_mean(
            self.planner_family_weights,
            context="PI0WaypointJoint block-AR loss",
        )

    def _vlm_image_prefix(self, batch: dict) -> tuple[list, list, list]:
        """Encode the camera views into embedding / pad / ar-mask fragments."""
        device = batch["tokens"].device
        b = batch["tokens"].shape[0]
        embs_list, pad_list, ar_list = [], [], []
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            if key not in batch["images"]:
                continue
            img = batch["images"][key]
            if img.shape[-1] == 3 and img.ndim == 4:
                img = img.permute(0, 3, 1, 2)  # BHWC -> BCHW
            img_emb = self._ckpt(self.paligemma_with_expert.paligemma.model.get_image_features, img)
            n_img_tokens = img_emb.shape[1]
            embs_list.append(img_emb)
            pad_list.append(batch["image_masks"][key][:, None].expand(b, n_img_tokens))
            ar_list.append(torch.zeros(b, n_img_tokens, dtype=torch.int32, device=device))
        return embs_list, pad_list, ar_list

    def _vlm_forward_block(
        self,
        batch: dict,
        return_breakdown: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Waypoint-block AR training forward.

        Differences from token-AR, all confined to the plan region:
          * inputs are shifted by one *block* (block 0 gets a learned query block);
          * the ar-mask is block-causal instead of strictly causal;
          * labels are aligned to the *same* position rather than ``tokens[:, 1:]``.
        """
        self._validate_distributed_vlm_reduction()
        tokens = batch["tokens"]
        token_mask = batch["token_mask"]
        ar_mask = batch["ar_mask"]
        loss_mask = batch["loss_mask"]

        if self.trim_padded_tokens:
            keep = self._trim_len(token_mask, loss_mask, ar_mask)
            if 0 < keep < tokens.shape[1]:
                tokens = tokens[:, :keep]
                token_mask = token_mask[:, :keep]
                ar_mask = ar_mask[:, :keep]
                loss_mask = loss_mask[:, :keep]

        layout = locate_block_layout(
            tokens,
            wp_token_id=_WP_TOKEN_ID,
            dur_token_id=_DUR_TOKEN_ID,
            num_blocks=self.planner_num_waypoints,
            block_width=self.planner_block_width,
            dur_delim_pos=self.planner_dur_delim_pos,
        )

        embed_layer = self.paligemma_with_expert.paligemma.language_model.embed_tokens
        token_embs = embed_layer(tokens)
        token_embs = token_embs * math.sqrt(token_embs.shape[-1])

        block0_cond = None
        if self.planner_block0_cond == "current_state":
            block0_cond = batch.get("current_wp_tokens")
            if block0_cond is None:
                raise KeyError(
                    "planner_block0_cond='current_state' needs batch['current_wp_tokens'], "
                    "which WaypointVLMDataset emits for every sample; this batch came from "
                    "an older loader or a collator that dropped the key"
                )
            block0_cond = block0_cond.to(device=tokens.device, dtype=torch.long)
            # ``CURRENT_STATE_DURATION_CODE`` is reserved as an *input-only*
            # marker; ``tokenizer.encode_current_state_block`` promises it never
            # appears in a label, and the tokenizer makes the feature's caller
            # responsible for verifying that.  Enforce it here -- the single
            # choke point where the feature is active -- rather than trusting a
            # dataset-side comment about the current index's maximum gap.
            marker = _DURATION_BASE - CURRENT_STATE_DURATION_CODE
            region_ids = tokens.gather(1, layout.flat_index)
            collision = (region_ids == marker) & loss_mask.gather(1, layout.flat_index)
            if bool(collision.any()):
                raise ValueError(
                    f"planner_block0_cond='current_state' reserves duration "
                    f"{CURRENT_STATE_DURATION_CODE} as an input-only marker, but "
                    f"{int(collision.sum())} supervised duration target(s) in this "
                    "batch equal it; filter such windows from the dataset or change "
                    "CURRENT_STATE_DURATION_CODE before training with this option"
                )

        condition_ids = None
        if self.training and self.planner_cond_jitter_prob > 0.0:
            condition_ids = jitter_condition_proprio_tokens(
                tokens.gather(1, layout.flat_index),
                block_width=self.planner_block_width,
                proprio_dim=self.planner_proprio_dim,
                prob=self.planner_cond_jitter_prob,
                max_bins=self.planner_cond_jitter_max_bins,
                generator=self._jitter_generator(tokens.device),
            )

        token_embs = build_block_shifted_embeddings(
            token_embs,
            layout,
            self.block_planner,
            embed_layer,
            _WP_TOKEN_ID,
            block0_cond_tokens=block0_cond,
            condition_token_ids=condition_ids,
        )
        block_ar = build_block_ar_mask(ar_mask, layout)

        embs_list, pad_list, ar_list = self._vlm_image_prefix(batch)
        embs_list.append(token_embs)
        pad_list.append(token_mask)
        ar_list.append(block_ar)

        all_embs = torch.cat(embs_list, dim=1)
        all_pad = torch.cat(pad_list, dim=1)
        all_ar = torch.cat(ar_list, dim=1)
        n_img_total = all_embs.shape[1] - tokens.shape[1]

        att_2d = make_att_2d_masks(all_pad, all_ar)
        position_ids = torch.cumsum(all_pad, dim=1) - 1
        att_4d = torch.where(att_2d[:, None], 0.0, -2.3819763e38)

        # LoRAEmbedding has no `.weight`; the real tensor lives on `.base_layer`.
        base_embed = getattr(embed_layer, "base_layer", embed_layer)
        model_dtype = base_embed.weight.dtype
        if all_embs.dtype != model_dtype:
            all_embs = all_embs.to(model_dtype)
        if att_4d.dtype != model_dtype:
            att_4d = att_4d.to(model_dtype)

        outputs = self.paligemma_with_expert.paligemma.language_model.forward(
            inputs_embeds=all_embs,
            attention_mask=att_4d,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state

        flat = layout.flat_index
        hidden_idx = (flat + n_img_total)[..., None].expand(-1, -1, hidden.shape[-1])
        region_hidden = hidden.gather(1, hidden_idx)
        region_targets = tokens.gather(1, flat)
        region_mask = loss_mask.gather(1, flat)

        if bool(region_mask.any()):
            active_logits = self._compute_lm_logits(region_hidden[region_mask].float())
            local_loss, loss_per_token, active_targets = block_ce_loss(
                active_logits,
                region_targets[region_mask],
                torch.ones(active_logits.shape[0], dtype=torch.bool, device=active_logits.device),
                self.planner_family_weights,
            )
        else:
            # Keep the empty-rank zero connected to this forward graph.  Every
            # rank must still enter global_mean_loss's scalar collective.
            local_loss = region_hidden.sum() * 0.0
            loss_per_token = torch.empty(0, device=tokens.device)
            active_targets = torch.empty(0, dtype=torch.long, device=tokens.device)

        if self.planner_family_weights is None:
            local_sum = loss_per_token.sum() if loss_per_token.numel() else local_loss
            loss = global_mean_loss(local_sum, region_mask.sum())
        else:
            # Multi-rank use was rejected above.  Preserve the historical
            # single-rank weighted sum of per-family means exactly.
            loss = local_loss

        if return_breakdown:
            with torch.no_grad():
                block_indices = (
                    torch.arange(layout.num_blocks, device=tokens.device)
                    .view(1, layout.num_blocks, 1)
                    .expand(tokens.shape[0], layout.num_blocks, layout.block_width)
                    .reshape(tokens.shape[0], -1)
                )
                active_block_indices = block_indices[region_mask]
                metrics = self._vlm_token_loss_breakdown(
                    loss_per_token.detach(),
                    active_targets.detach(),
                    active_block_indices,
                    layout.num_blocks,
                )
            return loss, metrics
        return loss

    def _vlm_forward_token_ar(
        self,
        batch: dict,
        return_breakdown: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """VLM training forward: compute CE loss on waypoint tokens.

        Routes through ``self.paligemma_with_expert.paligemma`` (the shared
        backbone). Identical logic to PI0WaypointVLM.forward().
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

            img_emb = self._ckpt(
                self.paligemma_with_expert.paligemma.model.get_image_features, img
            )
            n_img_tokens = img_emb.shape[1]
            embs_list.append(img_emb)

            img_mask = batch["image_masks"][key]
            pad_list.append(img_mask[:, None].expand(B, n_img_tokens))
            ar_list.append(torch.zeros(B, n_img_tokens, dtype=torch.int32, device=device))

        token_embs = self.paligemma_with_expert.paligemma.language_model.embed_tokens(tokens)
        token_embs = token_embs * math.sqrt(token_embs.shape[-1])
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

        model_dtype = self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.dtype
        if all_embs.dtype != model_dtype:
            all_embs = all_embs.to(model_dtype)
        if att_4d.dtype != model_dtype:
            att_4d = att_4d.to(model_dtype)

        outputs = self.paligemma_with_expert.paligemma.language_model.forward(
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

        mask = shift_loss_mask
        if mask.any():
            active_hidden = hidden[mask]
            active_targets = targets[mask]
            active_logits = self._compute_lm_logits(active_hidden.float())
            loss_per_token = F.cross_entropy(
                active_logits,
                active_targets,
                reduction="none",
            )
            local_sum = loss_per_token.sum()
        else:
            local_sum = hidden.sum() * 0.0
            loss_per_token = torch.empty(0, device=device)
            active_targets = torch.empty(0, dtype=torch.long, device=device)

        loss = global_mean_loss(local_sum, mask.sum())

        if return_breakdown:
            with torch.no_grad():
                # ``hidden[t]`` predicts the token at full-sequence position
                # ``t + 1``.  Build block ids in that same shifted coordinate
                # system, including the image prefix, then select active labels.
                layout = locate_block_layout(
                    tokens,
                    wp_token_id=_WP_TOKEN_ID,
                    dur_token_id=_DUR_TOKEN_ID,
                    num_blocks=self.planner_num_waypoints,
                    block_width=self.planner_block_width,
                    dur_delim_pos=self.planner_dur_delim_pos,
                )
                token_block_indices = torch.full_like(tokens, -1)
                grid_block_indices = (
                    torch.arange(layout.num_blocks, device=device)
                    .view(1, layout.num_blocks, 1)
                    .expand(B, layout.num_blocks, layout.block_width)
                    .reshape(B, -1)
                )
                token_block_indices.scatter_(1, layout.flat_index, grid_block_indices)
                full_block_indices = torch.cat(
                    [
                        torch.full((B, n_img_total), -1, dtype=torch.long, device=device),
                        token_block_indices,
                    ],
                    dim=1,
                )[:, 1:]
                active_block_indices = full_block_indices[mask]
                if bool((active_block_indices < 0).any()):
                    raise RuntimeError("a supervised VLM target lies outside the waypoint block grid")
                metrics = self._vlm_token_loss_breakdown(
                    loss_per_token.detach(),
                    active_targets.detach(),
                    active_block_indices,
                    layout.num_blocks,
                )
            return loss, metrics

        return loss

    # ------------------------------------------------------------------
    # AE forward (MSE loss on flow-matching)
    # ------------------------------------------------------------------
    def ae_forward(
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
        """AE training forward: compute per-element MSE loss.

        When ``gradient_strategy == "stop_gradient"``, prefix embeddings are
        detached so that the MSE loss does not propagate gradients into the
        shared PaliGemma backbone.
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

        # v2 training contract (docs/16 B/C): corrupt the goal *inputs* while
        # every supervision tensor (u_t, action_pad_mask) keeps ground truth.
        # Gated on self.training so eval/probe forwards stay clean; active from
        # step 0 by construction (no schedule, no warmup exemption).
        end_for_cond, dur_for_cond, goal_null_mask = end_proprio, duration, None
        if self.goal_condition_mode == "adarms_residual_v2" and self.training:
            end_for_cond, dur_for_cond, goal_null_mask = self._sample_goal_corruptions(
                end_proprio, duration
            )

        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(
            start_proprio, end_for_cond, x_t, time, dur_for_cond,
            goal_null_mask=goal_null_mask,
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

        # Knowledge Insulation: control gradient flow from AE MSE loss into
        # backbone weights through cross-attention K/V.
        #   - "stop_gradient": full detach (True)
        #   - "scale_gradient" + scale=0: full detach (True)
        #   - "scale_gradient" + scale>0: gradient scaling (<1 attenuate, >1 amplify)
        #   - "none" / "freeze_backbone": no insulation (False)
        if self.gradient_strategy == "stop_gradient":
            ki_value: bool | float = True
        elif self.gradient_strategy == "scale_gradient":
            ki_value = True if self.gradient_scale == 0 else self.gradient_scale
        else:
            ki_value = False

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            knowledge_insulation=ki_value,
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
        return global_mean_loss(loss_per_element.sum(), mask.sum())

    # ------------------------------------------------------------------
    # VLM inference: constrained AR generation
    # ------------------------------------------------------------------
    def _compute_family_logits(self, hidden: torch.Tensor, family) -> torch.Tensor:
        """Logits restricted to one contiguous token family (see planner_decode)."""
        embed_layer = self.paligemma_with_expert.paligemma.language_model.embed_tokens
        return compute_family_logits(embed_layer, hidden, family)

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
        decode_cfg: WaypointDecodeConfig | None = None,
        generator: torch.Generator | None = None,
        block0_cond_tokens: torch.Tensor | None = None,
    ) -> list[list[tuple]]:
        """Planner inference; dispatches on ``decode_cfg.impl``.

        ``decode_cfg=None`` (or ``impl="legacy"``) runs the original
        token-at-a-time full-vocabulary decoder unchanged.  The chosen
        implementation's timing/counter record is left on
        ``self.last_decode_stats`` for the evaluator to aggregate.

        ``max_new_tokens`` only bounds the legacy decoder (compact / jacobi /
        block size their buffers from the tokenizer).  ``None`` means the exact
        budget a full plan needs, ``legacy_decode_budget(wp_tokenizer)`` =
        header + ``num_waypoints * tokens_per_waypoint``; the old fixed default
        of 128 silently truncated a P=14 plan (3 + 7*18 = 129) to 6 waypoints.
        """
        cfg = decode_cfg or WaypointDecodeConfig()
        if max_new_tokens is None:
            max_new_tokens = legacy_decode_budget(wp_tokenizer)
        if temperature and not cfg.temperature:
            cfg = dataclasses.replace(cfg, temperature=temperature)

        # Both directions of the planner/decoder pairing are errors, and the one
        # that used to be unguarded is the dangerous one: a block-trained model
        # decoded token-at-a-time returns *plausible garbage* rather than
        # crashing, because its hidden state at position p predicts the token AT
        # p, not at p+1.  Guard here so every entry point is covered (the two
        # evaluators, the benchmark script, and any notebook).
        if self.block_planner is not None and cfg.impl != "block":
            raise RuntimeError(
                f"this model was built with planner_mode='block_ar' but decode impl is "
                f"'{cfg.impl}'. Token-AR decoding of a block-trained planner silently "
                "produces meaningless waypoints; use impl='block'."
            )
        if self.block_planner is None and cfg.impl == "block":
            raise RuntimeError(
                "decode impl='block' requires a model built with planner_mode='block_ar'"
            )
        # A checkpoint trained with a conditioned block 0 decodes without any
        # error if the condition is dropped -- it just silently plans from the
        # zero-residual query block it never saw in training.  That is the same
        # failure class as decoding a block checkpoint token-at-a-time, so guard
        # both directions rather than trusting the caller.
        if self.planner_block0_cond == "current_state" and block0_cond_tokens is None:
            raise RuntimeError(
                "this model was built with planner_block0_cond='current_state' but no "
                "block0_cond_tokens were passed; decoding without the condition silently "
                "produces a plan from an input the model never saw in training"
            )
        if self.planner_block0_cond == "none" and block0_cond_tokens is not None:
            raise RuntimeError(
                "block0_cond_tokens were passed but this model was built with "
                "planner_block0_cond='none'; the query block ignores them"
            )

        if cfg.impl == "legacy":
            wall0 = time.perf_counter()
            plans = self._generate_waypoints_legacy(
                images, image_masks, prompt_tokens, prompt_mask, wp_tokenizer,
                max_new_tokens=max_new_tokens, temperature=cfg.temperature,
                generator=generator,
                forbid_terminal_first_waypoint=cfg.forbid_terminal_first_waypoint,
            )
            stats = DecodeStats(impl="legacy")
            stats.planner_total_ms = (time.perf_counter() - wall0) * 1000.0
            stats.plan_len = max((len(p) for p in plans), default=0)
            stats.transformer_forward_count = self._legacy_forward_count
            stats.lm_head_invocation_count = self._legacy_lm_head_count
            stats.generated_value_tokens = self._legacy_value_token_count
            stats.generated_forced_tokens = self._legacy_forced_token_count
            # The legacy loop always computes one more LM head after the last
            # useful token and then breaks; record it so the benchmark table
            # shows where the removed work went.
            stats.unused_final_logit_count = 1
            self.last_decode_stats = stats
            return plans

        if cfg.impl == "compact":
            fn = generate_waypoints_compact
        elif cfg.impl == "block":
            fn = generate_waypoints_block
        elif cfg.impl == "jacobi":
            fn = generate_waypoints_jacobi
        else:  # pragma: no cover - guarded by WaypointDecodeConfig
            raise ValueError(f"Unknown decode impl: {cfg.impl}")

        extra_kwargs = {}
        if cfg.impl == "block":
            extra_kwargs["block0_cond_tokens"] = block0_cond_tokens
        plans, stats = fn(
            self, images, image_masks, prompt_tokens, prompt_mask, wp_tokenizer, cfg,
            generator=generator,
            **extra_kwargs,
        )
        self.last_decode_stats = stats
        return plans

    @torch.no_grad()
    def _generate_waypoints_legacy(
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
        """AR generation of waypoint tokens using the shared backbone.

        Identical logic to PI0WaypointVLM.generate_waypoints(), routing
        through ``self.paligemma_with_expert.paligemma``.  The masking and
        sampling step itself lives in ``planner_decode`` so both models share
        one copy (see ``legacy_constrained_next_token``).  ``max_new_tokens=None``
        means ``legacy_decode_budget(wp_tokenizer)`` (a full plan, no
        truncation for wide proprio layouts).
        """
        if max_new_tokens is None:
            max_new_tokens = legacy_decode_budget(wp_tokenizer)
        paligemma = self.paligemma_with_expert.paligemma

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
            img_emb = paligemma.model.get_image_features(img)
            n_img = img_emb.shape[1]
            embs_list.append(img_emb)
            pad_list.append(image_masks[key][:, None].expand(B, n_img))

        token_embs = paligemma.language_model.embed_tokens(prompt_tokens)
        token_embs = token_embs * math.sqrt(token_embs.shape[-1])
        embs_list.append(token_embs)
        pad_list.append(prompt_mask)

        prefix_embs = torch.cat(embs_list, dim=1)
        prefix_pad = torch.cat(pad_list, dim=1)

        prefix_len = prefix_embs.shape[1]
        prefix_ar = torch.zeros_like(prefix_pad, dtype=torch.int32)
        prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_ar)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1

        prefix_att_padded = F.pad(prefix_att_2d, (0, max_new_tokens), value=False)
        prefix_att_4d = prefix_att_padded[:, None, :, :]
        prefix_att_4d = torch.where(prefix_att_4d, 0.0, -2.3819763e38)

        model_dtype = paligemma.language_model.embed_tokens.weight.dtype
        if prefix_embs.dtype != model_dtype:
            prefix_embs = prefix_embs.to(model_dtype)
        if prefix_att_4d.dtype != model_dtype:
            prefix_att_4d = prefix_att_4d.to(model_dtype)

        outputs = paligemma.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        last_hidden = outputs.last_hidden_state[:, -1:]

        self._legacy_forward_count = 1  # the prefix prefill
        self._legacy_lm_head_count = 1
        self._legacy_value_token_count = 0
        self._legacy_forced_token_count = 0
        last_logits = self._compute_lm_logits(last_hidden.float())

        tpw = wp_tokenizer.tokens_per_waypoint
        prefill_len = torch.sum(prefix_pad, dim=-1)

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
        dur_delim_pos = wp_tokenizer.dur_delimiter_pos_in_wp
        dur_val_pos = wp_tokenizer.duration_pos_in_wp

        for step in range(max_new_tokens):
            # --- Logit masking + sampling (shared with PI0WaypointVLM) ---
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
                    self._legacy_forced_token_count += 1
                elif pos_in_wp == dur_delim_pos:
                    next_token[b, 0] = wp_tokenizer.dur_token_id
                    self._legacy_forced_token_count += 1
                else:
                    # pose / gripper / duration: the tokens the model actually
                    # chose.  Counted the same way as the compact runner's
                    # `record(forced=...)`, and neither counts the header.
                    self._legacy_value_token_count += 1

                all_output_tokens[b].append(next_token[b, 0].item())

            token_emb = paligemma.language_model.embed_tokens(next_token)
            token_emb = token_emb * math.sqrt(token_emb.shape[-1])
            if token_emb.dtype != model_dtype:
                token_emb = token_emb.to(model_dtype)

            positions = prefill_len[:, None] + step
            attn_len = prefix_len + step + 1
            mask = torch.zeros(B, 1, 1, attn_len, device=device, dtype=model_dtype)

            outputs = paligemma.language_model.forward(
                inputs_embeds=token_emb,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            last_hidden = outputs.last_hidden_state
            self._legacy_forward_count += 1
            self._legacy_lm_head_count += 1
            last_logits = self._compute_lm_logits(last_hidden.float())

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

    # ------------------------------------------------------------------
    # AE inference: iterative denoising
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        start_proprio,
        end_proprio,
        duration,
        num_steps: int = 10,
        noise=None,
        goal_null: bool = False,
    ):
        """Inference: generate action chunk via iterative denoising.

        ``goal_null=True`` (v2 only) runs the learned-null branch — the goal is
        replaced by the null on both ingestion paths at every denoising step.
        This is the CFG hook (docs/14 §5.6): guidance composes two calls as
        ``v_guided = v_null + w * (v_goal - v_null)``.  Default inference is
        the plain conditional branch; no noise or dropout is ever applied here
        (those are training-contract corruptions), while the v2 phase gate is
        part of the forward pass and applies at each step's t.
        """
        if goal_null and self.goal_condition_mode != "adarms_residual_v2":
            raise ValueError(
                "goal_null=True requires goal_condition_mode='adarms_residual_v2' "
                f"(got {self.goal_condition_mode!r})"
            )
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
        goal_null_mask = (
            torch.ones(bsize, dtype=torch.bool, device=device) if goal_null else None
        )

        while t_val >= -dt / 2:
            expanded_t = t_val.expand(bsize)

            suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(
                start_proprio, end_proprio, x_t, expanded_t, duration,
                goal_null_mask=goal_null_mask,
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

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    # Tensors for which a *prefix* copy of a narrower checkpoint tensor is
    # meaningful, and therefore the only ones ``load_pretrained_weights`` may
    # partially load.  They are the action/proprio-dim-indexed projections: the
    # Pi0.5 base ships ``action_in_proj`` ``[W, 32]`` / ``action_out_proj``
    # ``[32, W]`` (the base has no ``proprio_encoder``; it is listed because it
    # is indexed by the same ``action_dim`` and a waypoint checkpoint may carry
    # it at a narrower width), so when ``model_action_dim`` grows beyond 32 the
    # first 32 columns/rows keep their pretrained meaning and the new ones start
    # fresh.  ``time_mlp_in`` / ``time_mlp_out`` are kept for the historical
    # ``Linear(2W, W)`` -> ``Linear(W, W)`` migration of legacy checkpoints.
    #
    # Everything else is position-indexed (planner slot rows, goal-adapter
    # embeddings, backbone weights): a prefix copy there silently misaligns
    # rows -- e.g. the gripper-slot row of a P=6 ``block_planner.slot_embed``
    # landing on a proprio slot of a P=14 planner -- so a mismatch is an error.
    PARTIAL_LOAD_ALLOWLIST = (
        "action_in_proj.",
        "action_out_proj.",
        "proprio_encoder.",
        "time_mlp_in.",
        "time_mlp_out.",
    )

    @staticmethod
    def load_pretrained_weights(model, weight_path, device):
        """Load Pi0.5 base (or a merged waypoint) checkpoint into ``model``.

        * Keys the checkpoint has but the model lacks are skipped (counted).
        * Keys the model has but the checkpoint lacks keep their init
          (``duration_embed``, a fresh ``block_planner`` on a base warm start).
        * A shape mismatch is tolerated -- as a prefix copy that preserves the
          pretrained region -- **only** for ``PARTIAL_LOAD_ALLOWLIST`` tensors
          whose checkpoint shape fits inside the model's.  Any other mismatch
          raises: the historical silent prefix copy misaligned planner slot
          rows across robot layouts (see the allowlist comment).
        """
        import safetensors.torch

        state_dict = safetensors.torch.load_file(weight_path, device=str(device))
        own_state = model.state_dict()
        loaded, skipped, partial = 0, 0, 0
        for name, param in state_dict.items():
            if name not in own_state:
                skipped += 1
                continue
            own = own_state[name]
            if own.shape != param.shape:
                allowlisted = name.startswith(PI0WaypointJoint.PARTIAL_LOAD_ALLOWLIST)
                fits = param.dim() == own.dim() and all(
                    ps <= ns for ps, ns in zip(param.shape, own.shape)
                )
                if not (allowlisted and fits):
                    raise RuntimeError(
                        f"shape mismatch loading {name} from {weight_path}: checkpoint "
                        f"{tuple(param.shape)} vs model {tuple(own.shape)}. "
                        + (
                            "A prefix copy is only meaningful for "
                            f"{PI0WaypointJoint.PARTIAL_LOAD_ALLOWLIST} (action-dim-indexed "
                            "projections); this tensor is position-indexed, so partially "
                            "loading it would silently misalign rows. Rebuild the model "
                            "with the geometry the checkpoint was trained with "
                            "(metadata.pt: planner_proprio_dim / planner_use_gripper_token / "
                            "planner_num_waypoints / planner_block_width)."
                            if not allowlisted
                            else "The checkpoint tensor does not fit inside the model tensor, "
                            "so no pretrained-region-preserving copy exists."
                        )
                    )
                slices = tuple(slice(0, s) for s in param.shape)
                own[slices].copy_(param)
                logger.info(
                    f"  Partial load {name}: pretrained {tuple(param.shape)} "
                    f"-> new {tuple(own.shape)} (preserved pretrained region)"
                )
                partial += 1
                continue
            own.copy_(param)
            loaded += 1
        logger.info(f"Loaded {loaded} weight tensors, {partial} partial, skipped {skipped}")

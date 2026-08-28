"""Real-robot inference wrapper for the Rokae dual-arm waypoint policy.

One object, ``RokaeWaypointPolicy``, turns a raw robot observation into the joint
targets for the next segment, on exactly the code path the offline validation
(``scripts/eval_rokae_offline.py``) measured:

    observation ──> planner prefix (tokenizer.encode_prefix, byte-identical to training)
                ──> generate_waypoints (compact / block decoder from the training config)
                ──> first waypoint (14 joints normalized + 2 grippers, duration d)
                ──> action expert sample_actions(start=current state, end=waypoint, d)
                ──> d dense actions, unnormalized: [L joints(7), L grip, R joints(7), R grip]

Observation contract (``infer(obs)``):
    obs["external"], obs["left_wrist"], obs["right_wrist"]: uint8 RGB images of any size
        (they are squashed to 224x224 with PIL BILINEAR, no crop -- the same transform the
        training data went through, decision D3); model-side names (``base_0_rgb``,
        ``left_wrist_0_rgb``, ``right_wrist_0_rgb``) are accepted too.
    obs["state"]: float32[30], the bi_rokae ``observation.state`` layout
        (left joints 0..6, left cart 7..12, left psi 13, left gripper 14, right likewise 15..29).
    obs["prompt"]: the task instruction (one of the training sentences).
Returns a dict:
    Optional request keys: "reset" (true on the first request of an episode: clears the budget
    counters and re-seeds the action-expert noise -- the upstream websocket client's ``reset()`` is
    a local no-op and the server never calls ``policy.reset()``, so this key is the only way to reset
    over the wire), "execute_waypoints" (1 | 2, see __init__), "max_steps" / "max_replans"
    (per-episode budget overrides; fixed at the first call after a reset).

    "actions": float32[d, 16] joint targets for the next d control ticks (30 Hz):
        [left joint 0..6, left gripper, right joint 0..6, right gripper]; grippers are 0=closed /
        1=open commands (no sign inversion on Rokae).
    "duration": d (frames) = sum of "segment_durations"; "done": execute the returned actions and
    then stop -- "done_reason" says why: "terminal_plan" (the plan's first waypoint is the planner's
    end marker, so there is nothing to execute; or -- only with terminal_stop_agree > 0 -- the
    agreement rule of ``TerminalAgreement`` fired), "step_budget" / "replan_budget"
    (training-derived per-task budgets, TASK_BUDGETS), "stalled" (only if stall_stop_replans > 0);
    "budget": the counters (episode budget + terminal-agreement state);
    "plan_ends_in": how many real waypoints the plan contains before the planner's end marker,
    None when the plan carries no marker (token_ar checkpoints never emit one on held-out data;
    block_ar checkpoints do -- reported only, by default, see ``TerminalAgreement``);
    "waypoints": the whole decoded plan in robot units [[joints(14) + grippers(2), duration], ...],
    "planner_ms", "ae_ms".

The wrapper implements ``openpi_client.base_policy.BasePolicy`` so the upstream
``WebsocketPolicyServer`` can serve it unchanged (``python -m openpi.waypoint.rokae_policy serve``);
``replay`` runs it open-loop over a recorded validation episode as a plumbing self-check.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import torch
import yaml

from openpi.waypoint import transformers_guard

logger = logging.getLogger(__name__)

IMAGE_KEY_MAP = {"external": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb",
                 "base_0_rgb": "base_0_rgb", "left_wrist_0_rgb": "left_wrist_0_rgb", "right_wrist_0_rgb": "right_wrist_0_rgb"}
MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

# --- stop conditions ------------------------------------------------------------------------
# The token_ar planner has never emitted a terminal plan on held-out data (docs/04 §8.2), so
# the only stop the policy itself can guarantee is a budget.  Budgets come from the training
# demonstrations (3-task train split, 120 episodes, 2026-08-26): frames = ceil(1.5 x the
# longest demonstration of that task), replans = ceil(1.5 x the most waypoints of that task).
# The robot side must add its own hard stop on top (see docs/17-rokae-robot-client.md).
# The block_ar planner does emit an end marker (194 of the 2172 validation windows, 164 of them
# real endings) but never in the first slot, and the marker's position is exact in only 46 of
# the 164 -- too noisy to stop on directly.  It is therefore only REPORTED ("plan_ends_in")
# unless the operator opts in with terminal_stop_agree > 0 (``TerminalAgreement`` below).
TASK_BUDGETS = {
    "shelf": {"max_steps": 2373, "max_replans": 95},          # demos 502..1582 frames, 37..63 waypoints
    "pepper_banana": {"max_steps": 2013, "max_replans": 150},  # demos 404..1342 frames, 40..100 waypoints
    "swap": {"max_steps": 1749, "max_replans": 138},           # demos 433..1166 frames, 40..92 waypoints
}
DEFAULT_BUDGET = {"max_steps": 2373, "max_replans": 150}
STALL_EPS_RAD = 0.01          # a plan whose first waypoint is within this of the current joints
DONE_REASONS = ("terminal_plan", "step_budget", "replan_budget", "stalled")


def task_label(prompt: str) -> str:
    """Map an instruction to its task family (same rule as scripts/eval_rokae_offline.py)."""
    t = prompt.lower()
    if "shelf" in t:
        return "shelf"
    if "banana" in t:
        return "pepper_banana"
    if "plate" in t or "swap" in t:
        return "swap"
    return "unknown"


def task_budget(prompt: str, max_steps: int | None = None, max_replans: int | None = None) -> dict:
    """Budget for one episode: the task's training-derived default, overridden per field."""
    b = dict(TASK_BUDGETS.get(task_label(prompt), DEFAULT_BUDGET))
    if max_steps is not None:
        b["max_steps"] = int(max_steps)
    if max_replans is not None:
        b["max_replans"] = int(max_replans)
    if b["max_steps"] < 1 or b["max_replans"] < 1:
        raise ValueError(f"budget must be positive, got {b}")
    return b


class EpisodeBudget:
    """Counts what has been handed to the robot since reset() and decides when to stop.

    ``note(duration, stalled)`` is called once per plan with the number of frames handed
    out; ``reason()`` returns the stop reason once a limit is reached (None otherwise).
    ``stall_stop_replans`` = 0 disables the stall stop (default): a run of plans whose
    first waypoint equals the current pose may be a legitimate wait, so stalls are only
    reported unless the robot side opts in.
    """

    def __init__(self, max_steps: int, max_replans: int, stall_stop_replans: int = 0):
        self.max_steps, self.max_replans, self.stall_stop_replans = int(max_steps), int(max_replans), int(stall_stop_replans)
        self.steps = 0
        self.replans = 0
        self.stalled = 0

    def note(self, duration: int, stalled: bool = False) -> None:
        self.replans += 1
        self.steps += int(duration)
        self.stalled = self.stalled + 1 if stalled else 0

    def reason(self) -> str | None:
        if self.steps >= self.max_steps:
            return "step_budget"
        if self.replans >= self.max_replans:
            return "replan_budget"
        if self.stall_stop_replans > 0 and self.stalled >= self.stall_stop_replans:
            return "stalled"
        return None

    def as_dict(self) -> dict:
        return {"max_steps": self.max_steps, "max_replans": self.max_replans, "steps_executed": self.steps,
                "replans": self.replans, "stalled_replans": self.stalled}


def plan_ends_in(plan) -> int | None:
    """Real waypoints before the planner's first end marker (duration 0); None if the plan has none."""
    for k, (_, d) in enumerate(plan):
        if int(d) <= 0:
            return k
    return None


def _hist(values) -> dict:
    """{"none": n, "0": n, "1": n, ...} for a list of ints / Nones (JSON-friendly)."""
    out: dict = {}
    for v in values:
        key = "none" if v is None else str(int(v))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


class TerminalAgreement:
    """Opt-in stop on the planner's own end marker, robust to single-plan noise.

    Every plan that carries an end marker predicts an absolute end: the number of waypoints the
    robot has executed so far in this episode + ``plan_ends_in``.  ``note()`` records one plan and
    returns True when the rule fires: the last ``agree`` plans all carried a marker AND predicted
    the same absolute end AND the current plan's end lies inside the segments being returned now
    (``plan_ends_in <= n_exec``).  ``agree = 0`` (default) never fires -- the marker is reported only.

    Why not stop on the first marker: simulated on the 15 validation demonstrations (block_ar
    8800, 45 trajectories = 15 demos x 3 start offsets, stopping at the FIRST firing):
    agree=1 stops exactly at the end in 18% of trajectories but EARLY in 40% (13% by more than
    three waypoints); agree=2 stops exactly in 11%, one waypoint early in 2%, and otherwise leaves
    the stop to the budget; agree=3 almost never fires.  Use 2 if you enable it at all.
    """

    def __init__(self, agree: int = 0):
        if int(agree) < 0:
            raise ValueError("terminal_stop_agree must be >= 0")
        self.agree = int(agree)
        self.executed_wps = 0
        self.history: list[int | None] = []       # predicted absolute end per plan (None = no marker)

    def note(self, ends_in: int | None, n_exec: int, executed_now: int) -> bool:
        end_abs = None if ends_in is None else self.executed_wps + int(ends_in)
        self.history.append(end_abs)
        fire = (self.agree > 0 and ends_in is not None and int(ends_in) <= int(n_exec)
                and len(self.history) >= self.agree
                and all(h is not None and h == end_abs for h in self.history[-self.agree:]))
        self.executed_wps += int(executed_now)
        return bool(fire)

    def run_length(self) -> int:
        """How many trailing plans agree on the same predicted absolute end (0 if the last had none)."""
        if not self.history or self.history[-1] is None:
            return 0
        n = 0
        for h in reversed(self.history):
            if h != self.history[-1]:
                break
            n += 1
        return n

    def as_dict(self) -> dict:
        return {"terminal_stop_agree": self.agree, "executed_waypoints": self.executed_wps,
                "terminal_agree_run": self.run_length()}


def assemble_segments(segments: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, int, list[int]]:
    """Concatenate per-segment action blocks -> (actions, total frames, per-segment frames)."""
    durs = [int(d) for _, d in segments]
    acts = np.concatenate([a[:d] for a, d in zip((seg for seg, _ in segments), durs)], axis=0) if segments else np.zeros((0, 0), np.float32)
    return acts, int(sum(durs)), durs


def squash_224(img: np.ndarray, size: int = 224) -> np.ndarray:
    """640x480 (or anything) -> 224x224 by PIL BILINEAR squash, the training-data transform."""
    from PIL import Image
    img = np.asarray(img)
    if img.shape[0] == size and img.shape[1] == size:
        return np.ascontiguousarray(img, dtype=np.uint8)
    return np.asarray(Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).resize((size, size), Image.BILINEAR))


def checkpoint_has_block_planner(checkpoint_dir: str) -> bool | None:
    """True if the checkpoint's safetensors file carries ``block_planner.*`` tensors (a block_ar
    checkpoint), False if not, None if no safetensors file is found.  Reads only the header."""
    import os
    import struct
    for name in ("lora.safetensors", "model.safetensors"):
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        return any(k.startswith("block_planner.") for k in header if k != "__metadata__")
    return None


def check_checkpoint_matches_planner_mode(cfg: dict, checkpoint_dir: str) -> None:
    """Refuse the one mismatch the model loader does NOT catch.

    A block_ar checkpoint loaded with a token_ar config is silently wrong: the loader skips the
    ``block_planner.*`` tensors (log line ``skipped 3``) and decodes token-by-token with a
    backbone that was trained under the block-causal mask -- plausible-looking, meaningless plans.
    (The other direction, block_ar config + token_ar checkpoint, already raises in
    ``eval_libero.load_joint``.)"""
    has_block = checkpoint_has_block_planner(checkpoint_dir)
    mode = cfg.get("planner_mode", "token_ar")
    if has_block is True and mode != "block_ar":
        raise ValueError(
            f"checkpoint {checkpoint_dir} contains block_planner.* tensors (a block_ar checkpoint) but the "
            f"config says planner_mode={mode!r}. Loading it this way silently drops those tensors and "
            "produces meaningless plans. Use the block_ar inference config (planner_mode: block_ar, "
            "eval.waypoint_decode.impl: block), e.g. configs/rokae_blockar_infer.yaml.")
    if has_block is False and mode == "block_ar":
        raise ValueError(
            f"config says planner_mode='block_ar' but checkpoint {checkpoint_dir} has no block_planner.* "
            "tensors (a token_ar checkpoint). Use configs/rokae_tokenar_infer.yaml for this checkpoint.")
    logger.info(f"checkpoint {checkpoint_dir}: block planner tensors={'yes' if has_block else 'no'}, planner_mode={mode}")


class RokaeWaypointPolicy:
    """Waypoint planner + action expert for the Rokae AR5-5 dual arm (see module docstring)."""

    def __init__(self, config_path: str, checkpoint_dir: str, device: str = "cuda:0",
                 num_denoise_steps: int = 10, seed: int = 0, execute_waypoints: int = 1,
                 stall_stop_replans: int = 0, max_steps: int | None = None, max_replans: int | None = None,
                 terminal_stop_agree: int = 0):
        """``execute_waypoints``: how many plan segments to return per call (1 = the validated
        protocol, replan after every waypoint; 2 = also the second segment, generated open-loop
        from the predicted first waypoint, halving the number of planner calls).
        ``stall_stop_replans`` / ``max_steps`` / ``max_replans``: stop-condition defaults, see
        ``EpisodeBudget``; per-request keys in ``obs`` override them.
        ``terminal_stop_agree``: 0 (default) = the planner's end marker is only reported
        (``plan_ends_in``); N > 0 = stop when the last N plans agree on it (``TerminalAgreement``)."""
        # 投产推理路径此前不做任何 transformers 替换校验：上游那道检查只在
        # Pi0Pytorch.__init__ 里，而 waypoint 这条线从不构造 Pi0Pytorch。全新 venv 或
        # 重装 transformers 都会静默改掉注意力语义，而重装正是恢复流程的默认步骤。
        self.transformers_provenance = transformers_guard.assert_transformers_replacement(
            caller="RokaeWaypointPolicy"
        )
        if execute_waypoints not in (1, 2):
            raise ValueError("execute_waypoints must be 1 or 2")
        self.execute_waypoints = int(execute_waypoints)
        self.stall_stop_replans = int(stall_stop_replans)
        self.default_max_steps, self.default_max_replans = max_steps, max_replans
        self.terminal_stop_agree = int(terminal_stop_agree)
        self.terminal = TerminalAgreement(self.terminal_stop_agree)
        self.budget: EpisodeBudget | None = None
        from openpi.waypoint import eval_libero
        from openpi.waypoint.normalize import NormalizationHelper, load_dataset_statistics, pad_to_dim
        from openpi.waypoint.planner_decode import WaypointDecodeConfig
        from openpi.waypoint.robot_config import get_robot_config
        from openpi.waypoint.tokenizer import WaypointTokenizer

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        cfg = self.cfg
        check_checkpoint_matches_planner_mode(cfg, checkpoint_dir)
        self.rc = get_robot_config(cfg["robot_type"], **(cfg.get("robot_config_kwargs") or {}))
        if self.rc.num_grippers() != 2 or self.rc.continuous_proprio_dim != 14:
            raise ValueError("RokaeWaypointPolicy expects the dual-arm rokae config (14 joints + 2 grippers)")
        stats = load_dataset_statistics(cfg["dataset_statistics_path"])
        # for_robot: keep the two gripper action columns un-normalized (raw {0,1}
        # flags, as trained); the default mask would map closed 0.0 -> 0.5.
        self.norm = NormalizationHelper.for_robot(stats, cfg.get("norm_type", "q99"), self.rc)
        self.tok = WaypointTokenizer(
            proprio_dim=self.rc.continuous_proprio_dim, num_waypoints=cfg.get("num_waypoints", 7),
            max_token_len=cfg.get("vlm_max_token_len", cfg.get("max_token_len", 256)),
            use_gripper_token=True, n_gripper_slots=self.rc.num_grippers(),
        )
        self.decode_cfg = WaypointDecodeConfig.from_dict(cfg.get("eval") or {})
        self.device = torch.device(device)
        self.model_proprio_dim = cfg.get("model_proprio_dim", 32)
        self.actual_action_dim = self.rc.actual_action_dim
        self.max_duration = int(cfg.get("max_duration", 32))
        self.num_denoise_steps = num_denoise_steps
        self._pad_to_dim = pad_to_dim
        self._seed = int(seed)
        self._noise_gen = torch.Generator(device="cpu").manual_seed(self._seed)
        t0 = time.time()
        cfg_eval = dict(cfg); cfg_eval["joint_checkpoint"] = checkpoint_dir
        self.model = eval_libero.load_joint(cfg_eval, self.device)          # base + LoRA-active, slim forward
        self._decode_extra = eval_libero._decode_kwargs_for(self.model, self.decode_cfg, None)
        self.checkpoint_dir = checkpoint_dir
        self.replans = 0
        logger.info(f"RokaeWaypointPolicy ready: planner_mode={cfg.get('planner_mode')} decode={self.decode_cfg.impl} "
                    f"tokens/wp={self.tok.tokens_per_waypoint} ckpt={checkpoint_dir} ({time.time() - t0:.1f}s)")

    # -- openpi_client.BasePolicy -------------------------------------------------------------
    @property
    def metadata(self) -> dict:
        return {"robot": "rokae_dual_arm", "planner_mode": self.cfg.get("planner_mode"), "decode_impl": self.decode_cfg.impl,
                "action_layout": "left joints 0..6, left gripper, right joints 0..6, right gripper (0=closed, 1=open)",
                "control_hz": 30, "image_transform": "PIL BILINEAR squash to 224x224, no crop",
                "checkpoint": self.checkpoint_dir,
                "request_keys": {"required": ["external", "left_wrist", "right_wrist", "state", "prompt"],
                                 "optional": ["reset", "execute_waypoints", "max_steps", "max_replans"]},
                # NOTE: "server_timing" is appended by the transport layer (openpi.serving.
                # websocket_policy_server), not by this class -- it carries {"infer_ms": ...} and,
                # from the second request of a connection on, "prev_total_ms".  It is listed here
                # because clients validating a response against this contract would otherwise see
                # an undeclared key.
                "response_keys": ["actions", "duration", "segment_durations", "done", "done_reason", "waypoints",
                                  "planner_ms", "ae_ms", "budget", "plan_ends_in", "server_timing"],
                "execute_waypoints_default": self.execute_waypoints,
                "task_budgets": TASK_BUDGETS, "default_budget": DEFAULT_BUDGET,
                "stall_stop_replans": self.stall_stop_replans, "done_reasons": list(DONE_REASONS),
                "terminal_stop_agree": self.terminal_stop_agree}

    def reset(self) -> None:
        """Start of an episode: clears the step / replan budget counters and re-seeds the action-expert
        noise so two replays of the same recording give the same numbers.  Reached over the wire by
        sending ``obs["reset"] = True`` (``WebsocketClientPolicy.reset()`` does not talk to the server)."""
        self.replans = 0
        self.budget = None
        self.terminal = TerminalAgreement(getattr(self, "terminal_stop_agree", 0))
        self._noise_gen = torch.Generator(device="cpu").manual_seed(self._seed)

    @staticmethod
    def wants_reset(obs: dict) -> bool:
        """True when the request asks for an episode reset (``reset`` key truthy)."""
        return bool(obs.get("reset", False))

    @torch.inference_mode()
    def infer(self, obs: dict) -> dict:
        if self.wants_reset(obs):
            self.reset()
        images = self._images(obs)
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] != self.rc.actual_proprio_dim:
            raise ValueError(f"state must have {self.rc.actual_proprio_dim} dims, got {state.shape}")
        prompt = str(obs["prompt"])
        n_exec = int(obs.get("execute_waypoints", self.execute_waypoints))
        if n_exec not in (1, 2):
            raise ValueError("execute_waypoints must be 1 or 2")
        if self.budget is None:                       # first call after reset(): fix this episode's budget
            b = task_budget(prompt, obs.get("max_steps", self.default_max_steps), obs.get("max_replans", self.default_max_replans))
            self.budget = EpisodeBudget(b["max_steps"], b["max_replans"], self.stall_stop_replans)
        cont, grips = self.rc.split_proprio_grippers(state)
        cont_norm = self.norm.normalize_proprio(cont.astype(np.float32))
        # ---- planner ----
        t0 = time.time()
        plan = self.plan(images, prompt, cont_norm, grips)
        planner_ms = (time.time() - t0) * 1000
        self.replans += 1
        ends_in = plan_ends_in(plan)
        out = {"waypoints": [[self._denorm_wp(p).tolist(), int(d)] for p, d in plan],
               "planner_ms": planner_ms, "ae_ms": 0.0, "done": False, "done_reason": None,
               "duration": 0, "segment_durations": [], "plan_ends_in": ends_in,
               "actions": np.zeros((0, self.actual_action_dim), dtype=np.float32)}
        if not plan or int(plan[0][1]) <= 0:
            out["done"], out["done_reason"] = True, "terminal_plan"   # the planner says the task is finished
            self.budget.note(0)
            self.terminal.note(ends_in, n_exec, 0)
            out["budget"] = {**self.budget.as_dict(), **self.terminal.as_dict()}
            return out
        # ---- action expert: first segment from the measured state, optionally the second
        # segment open-loop from the predicted first waypoint (same images) ----
        t1 = time.time()
        P = self.rc.continuous_proprio_dim
        real = [(np.asarray(p, dtype=np.float32), int(min(max(int(d), 1), self.max_duration))) for p, d in plan if int(d) > 0]
        start_vec = np.concatenate([cont_norm, np.asarray(grips, dtype=np.float32)])
        segments = []
        for k in range(min(n_exec, len(real))):
            end_vec, dur = real[k]
            actions_norm = self.act(images, prompt, start_vec, end_vec, dur)
            actions = self.norm.unnormalize_actions(actions_norm[:dur, :self.actual_action_dim])
            for gi in (7, 15):                        # gripper columns of the 16-dim layout -> {0,1} commands
                actions[:, gi] = np.clip(np.round(actions[:, gi]), 0, 1)
            segments.append((actions.astype(np.float32), dur))
            start_vec = np.concatenate([end_vec[:P], np.round(end_vec[P:P + 2])]).astype(np.float32)
        out["ae_ms"] = (time.time() - t1) * 1000
        actions, total, durs = assemble_segments(segments)
        out["actions"], out["duration"], out["segment_durations"] = actions, total, durs
        # ---- stop conditions ----
        wp1_joints = self._denorm_wp(real[0][0])[:P]
        stalled = bool(np.abs(wp1_joints - cont[:P]).max() < STALL_EPS_RAD and
                       np.array_equal(np.round(real[0][0][P:P + 2]), np.round(np.asarray(grips, dtype=np.float32))))
        self.budget.note(total, stalled)
        agreed_end = self.terminal.note(ends_in, n_exec, len(segments))
        reason = self.budget.reason()
        if reason is None and agreed_end:
            reason = "terminal_plan"                              # opt-in only: terminal_stop_agree > 0
        if reason is not None:
            out["done"], out["done_reason"] = True, reason        # execute these actions, then stop
        out["budget"] = {**self.budget.as_dict(), **self.terminal.as_dict()}
        return out

    # -- pieces (also used by the replay self-check) ---------------------------------------------
    def _images(self, obs: dict) -> dict[str, np.ndarray]:
        images = {}
        for k, v in obs.items():
            if k in IMAGE_KEY_MAP and v is not None:
                images[IMAGE_KEY_MAP[k]] = squash_224(v)
        missing = [k for k in MODEL_IMAGE_KEYS if k not in images]
        if missing:
            raise ValueError(f"missing camera images {missing}; got {sorted(images)}")
        return images

    def plan(self, images: dict[str, np.ndarray], prompt: str, cont_norm: np.ndarray, grips) -> list:
        """Decode the plan: list of (proprio16 normalized [14 joints + 2 grippers], duration)."""
        img_t, mask_t = {}, {}
        for k in MODEL_IMAGE_KEYS:
            t = torch.from_numpy(images[k]).float() / 127.5 - 1.0          # (H, W, C), as eval's planner path
            img_t[k] = t.unsqueeze(0).to(self.device)
            mask_t[k] = torch.ones(1, dtype=torch.bool, device=self.device)
        prefix = self.tok.encode_prefix(prompt, cont_norm, list(grips))
        prompt_tokens = torch.as_tensor(prefix, dtype=torch.long, device=self.device)[None]
        prompt_mask = torch.ones_like(prompt_tokens, dtype=torch.bool)
        wps = self.model.generate_waypoints(images=img_t, image_masks=mask_t, prompt_tokens=prompt_tokens,
                                            prompt_mask=prompt_mask, wp_tokenizer=self.tok, **self._decode_extra)[0]
        return [(np.asarray(p, dtype=np.float32), int(d)) for p, d in wps]

    def act(self, images: dict[str, np.ndarray], prompt: str, start16: np.ndarray, end16: np.ndarray, duration: int) -> np.ndarray:
        """Action expert: dense normalized actions (horizon, model_action_dim) for one segment."""
        img_t, mask_t = {}, {}
        for k in MODEL_IMAGE_KEYS:
            t = torch.from_numpy(images[k]).float() / 127.5 - 1.0
            img_t[k] = t.permute(2, 0, 1).unsqueeze(0).to(self.device)     # (1, C, H, W), as eval's AE path
            mask_t[k] = torch.ones(1, dtype=torch.bool, device=self.device)
        max_len = self.cfg.get("ae_max_token_len", 64)
        tids = self.tok._pg_tokenizer.encode(f"Task: {prompt.strip().replace('_', ' ').lower()}, \n", add_bos=True)[:max_len]
        ptok = torch.zeros(1, max_len, dtype=torch.long, device=self.device); ptok[0, :len(tids)] = torch.tensor(tids)
        pmask = torch.zeros(1, max_len, dtype=torch.bool, device=self.device); pmask[0, :len(tids)] = True
        start = torch.from_numpy(self._pad_to_dim(start16.astype(np.float32), self.model_proprio_dim)).unsqueeze(0).to(self.device)
        end = torch.from_numpy(self._pad_to_dim(end16.astype(np.float32), self.model_proprio_dim)).unsqueeze(0).to(self.device)
        dur = torch.tensor([float(duration)], device=self.device)

        class _Obs:
            pass
        obs = _Obs()
        obs.images, obs.image_masks, obs.state = img_t, mask_t, start
        obs.tokenized_prompt, obs.tokenized_prompt_mask = ptok, pmask
        obs.token_ar_mask = obs.token_loss_mask = None
        noise = torch.randn(1, self.model.action_horizon, self.model.action_dim, generator=self._noise_gen).to(self.device)
        actions = self.model.sample_actions(obs, start, end, dur, num_steps=self.num_denoise_steps, noise=noise)
        return actions.squeeze(0).float().cpu().numpy()

    def _denorm_wp(self, p16: np.ndarray) -> np.ndarray:
        """Normalized waypoint -> robot units: 14 joints in radians + 2 gripper flags."""
        P = self.rc.continuous_proprio_dim
        joints = self.norm.unnormalize_proprio(np.asarray(p16[:P], dtype=np.float32))
        return np.concatenate([joints, np.round(np.asarray(p16[P:P + 2], dtype=np.float32))])


# ---------------------------------------------------------------------------------------------
# self-check: open-loop replay over a recorded validation episode
# ---------------------------------------------------------------------------------------------

def replay_episode(policy: RokaeWaypointPolicy, rlds_dir: str, episode: int, max_replans: int = 200,
                   execute_waypoints: int | None = None) -> dict:
    """Feed the recorded observations to the policy at the times it would ask for them.

    Open loop: after each plan the recording is advanced by the *predicted* duration (the robot
    is assumed to reach the waypoint), so this checks plumbing, latency and the gap between
    predicted and recorded joints -- not closed-loop success.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder_from_directory(rlds_dir)
    ds = builder.as_dataset(split="train", read_config=tfds.ReadConfig(interleave_cycle_length=16, interleave_block_length=16))
    ep = None
    for i, e in enumerate(ds):
        if i == episode:
            ep = e
            break
    if ep is None:
        raise ValueError(f"episode {episode} not found")
    steps = list(ep["steps"])
    T = len(steps)
    instruction = steps[0]["language_instruction"].numpy().decode("utf-8")
    states = np.stack([s["observation"]["state"].numpy() for s in steps]).astype(np.float32)
    actions = np.stack([s["action"].numpy() for s in steps]).astype(np.float32)
    A = policy.rc.action_dim_indices
    t, log, errs, ms = 0, [], [], []
    policy.reset()
    while t < T - 1 and len(log) < max_replans:
        obs = {"external": steps[t]["observation"]["external"].numpy(), "left_wrist": steps[t]["observation"]["left_wrist"].numpy(),
               "right_wrist": steps[t]["observation"]["right_wrist"].numpy(), "state": states[t], "prompt": instruction}
        if execute_waypoints is not None:
            obs["execute_waypoints"] = int(execute_waypoints)
        out = policy.infer(obs)
        ms.append((out["planner_ms"], out["ae_ms"]))
        d = out["duration"]
        if d <= 0:
            log.append({"t": t, "done": bool(out["done"]), "done_reason": out.get("done_reason")}); break
        seg_gt = actions[t: t + d][:, A]                       # recorded 16-dim actions for the same ticks
        n = min(len(seg_gt), d)
        err = np.abs(out["actions"][:n] - seg_gt[:n])
        d1 = out["segment_durations"][0]
        rec = {"t": t, "d": d, "segment_durations": list(out["segment_durations"]),
               "left_joint_mae": float(err[:, 0:7].mean()), "right_joint_mae": float(err[:, 8:15].mean()),
               "grip_acc": float((out["actions"][:n][:, [7, 15]] == np.round(seg_gt[:n][:, [7, 15]])).mean()),
               "wp1_joint_maxabs": float(np.abs(out["waypoints"][0][0][:14] - states[min(t + d1, T - 1)][policy.rc.continuous_proprio_indices]).max())}
        if len(out["segment_durations"]) > 1 and n > d1:      # second, open-loop segment reported separately
            e2 = err[d1:n]
            rec["seg2_left_joint_mae"], rec["seg2_right_joint_mae"] = float(e2[:, 0:7].mean()), float(e2[:, 8:15].mean())
            rec["seg2_grip_acc"] = float((out["actions"][d1:n][:, [7, 15]] == np.round(seg_gt[d1:n][:, [7, 15]])).mean())
        errs.append(rec)
        log.append({"t": t, "d": d, "n_wp": len(out["waypoints"]), "plan_ends_in": out.get("plan_ends_in"),
                    "planner_ms": round(out["planner_ms"]), "ae_ms": round(out["ae_ms"])})
        t += d
        if out["done"]:
            log.append({"t": t, "done": True, "done_reason": out.get("done_reason")}); break
    seg2 = [e for e in errs if "seg2_left_joint_mae" in e]
    summary = {"episode": episode, "frames": T, "instruction": instruction, "replans": len([l for l in log if "d" in l]), "reached_t": t,
               "done_emitted": bool(log and log[-1].get("done")), "done_reason": (log[-1].get("done_reason") if log else None),
               "plan_ends_in_hist": _hist([l.get("plan_ends_in") for l in log if "d" in l]),
               "execute_waypoints": execute_waypoints or policy.execute_waypoints,
               "seg2_left_joint_mae_rad": float(np.mean([e["seg2_left_joint_mae"] for e in seg2])) if seg2 else None,
               "seg2_right_joint_mae_rad": float(np.mean([e["seg2_right_joint_mae"] for e in seg2])) if seg2 else None,
               "seg2_grip_acc": float(np.mean([e["seg2_grip_acc"] for e in seg2])) if seg2 else None,
               "planner_ms_mean": float(np.mean([m[0] for m in ms])) if ms else None,
               "ae_ms_mean": float(np.mean([m[1] for m in ms])) if ms else None,
               "left_joint_mae_rad": float(np.mean([e["left_joint_mae"] for e in errs])) if errs else None,
               "right_joint_mae_rad": float(np.mean([e["right_joint_mae"] for e in errs])) if errs else None,
               "grip_acc": float(np.mean([e["grip_acc"] for e in errs])) if errs else None,
               "wp1_vs_recorded_maxabs_rad_mean": float(np.mean([e["wp1_joint_maxabs"] for e in errs])) if errs else None,
               "segments": errs[:5]}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--denoise-steps", type=int, default=10)
        p.add_argument("--execute-waypoints", type=int, default=1, choices=(1, 2),
                       help="plan segments returned per call (1 = validated protocol; 2 halves planner calls, second segment open-loop)")
        p.add_argument("--stall-stop-replans", type=int, default=0, help="stop after this many consecutive no-motion plans (0 = report only)")
        p.add_argument("--max-steps", type=int, default=None, help="override the per-task frame budget")
        p.add_argument("--max-replans", type=int, default=None, help="override the per-task replan budget")
        p.add_argument("--terminal-stop-agree", type=int, default=0,
                       help="stop when the last N plans agree on the planner's end marker (0 = report only, the default; "
                            "2 is the recommended value if enabled, see docs/17 §6)")
    sub.choices["serve"].add_argument("--host", default="0.0.0.0")
    sub.choices["serve"].add_argument("--port", type=int, default=8000)
    sub.choices["replay"].add_argument("--rlds", required=True)
    sub.choices["replay"].add_argument("--episode", type=int, default=0)
    sub.choices["replay"].add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy = RokaeWaypointPolicy(args.config, args.checkpoint, device=args.device, num_denoise_steps=args.denoise_steps,
                                 execute_waypoints=args.execute_waypoints, stall_stop_replans=args.stall_stop_replans,
                                 max_steps=args.max_steps, max_replans=args.max_replans,
                                 terminal_stop_agree=args.terminal_stop_agree)
    if args.cmd == "serve":
        from openpi.serving import websocket_policy_server
        server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata)
        logger.info(f"serving RokaeWaypointPolicy on {args.host}:{args.port}")
        server.serve_forever()
    else:
        summary = replay_episode(policy, args.rlds, args.episode, execute_waypoints=args.execute_waypoints)
        print(json.dumps(summary, indent=1, ensure_ascii=False))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

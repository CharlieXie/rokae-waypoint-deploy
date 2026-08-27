"""Two-stage CALVIN evaluation for Waypoint VLA.

Pipeline:
  1. VLM predicts M=7 waypoints autoregressively.
  2. Action Expert fills actions between each waypoint pair via flow matching.
  3. Execute actions in CALVIN environment.
  4. Replan after exhausting all waypoints.

Uses CALVIN's chain-task evaluation protocol:
  - 1000 evaluation sequences, each with 5 subtasks in a row.
  - Report avg_seq_len and chain success rates (1/5 through 5/5).

Usage:
  python -m openpi.waypoint.eval_calvin --config configs/eval_waypoint_joint_calvin.yaml
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import argparse
import copy
import json
import logging
import math
import pathlib
import time
from collections import Counter
from pathlib import Path

import imageio
import numpy as np
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
)
from openpi.waypoint.robot_config import get_robot_config
from openpi.waypoint.tokenizer import WaypointTokenizer

# Reuse model loading utilities from eval_libero
from openpi.waypoint.eval_libero import (
    _check_decode_compatibility,
    _episode_stats,
    _setup_seed,
    center_crop_and_resize,
    load_ae,
    load_joint,
    load_pg_tokenizer,
    load_vlm,
    log_diagnostics,
    predict_actions,
    accumulate_decode_stats,
    predict_waypoints,
    summarize_episode_stats,
)
from openpi.waypoint.planner_decode import DECODE_IMPLS, WaypointDecodeConfig

logger = logging.getLogger(__name__)

CALVIN_ROOT = os.environ.get("CALVIN_ROOT", "calvin")


# ---------------------------------------------------------------------------
# CALVIN observation helpers
# ---------------------------------------------------------------------------

def get_proprio_from_obs_calvin(obs):
    """Extract proprio from CALVIN environment observation.

    CALVIN env returns obs["robot_obs"] as a 15D vector:
      [0:3]  TCP position (x, y, z)
      [3:6]  TCP orientation (euler x, y, z)
      [6]    Gripper width (meters, 0~0.08)
      [7:14] 7 joint angles (radians)
      [14]   Gripper action (-1/+1, +1=open)

    We return the full 15D; RobotConfig.split_proprio() handles
    extracting continuous_proprio (dims 0:6) and binarizing gripper (dim 6).
    """
    robot_obs = obs["robot_obs"]
    return np.array(robot_obs, dtype=np.float32)


def get_calvin_images(obs, size=224, center_crop_scale=None):
    """Extract camera images from CALVIN observation.

    CALVIN env returns:
      obs["rgb_obs"]["rgb_static"]  — (200, 200, 3) uint8, static camera
      obs["rgb_obs"]["rgb_gripper"] — (84, 84, 3) uint8, wrist camera
    """
    from PIL import Image as PILImage
    images = {}

    static = obs["rgb_obs"]["rgb_static"]
    if static is not None:
        img = PILImage.fromarray(static)
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        if center_crop_scale is not None:
            arr = center_crop_and_resize(arr, center_crop_scale, size)
        images["base_0_rgb"] = arr

    gripper_img = obs["rgb_obs"]["rgb_gripper"]
    if gripper_img is not None:
        img = PILImage.fromarray(gripper_img)
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        if center_crop_scale is not None:
            arr = center_crop_and_resize(arr, center_crop_scale, size)
        images["left_wrist_0_rgb"] = arr

    return images


# ---------------------------------------------------------------------------
# CALVIN environment setup
# ---------------------------------------------------------------------------

def make_calvin_env(dataset_path, device):
    """Create CALVIN environment using CalvinEnvWrapperRaw."""
    val_folder = Path(dataset_path) / "validation"
    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }

    # Import CalvinEnvWrapperRaw — available from calvin_env package
    # Try the VLA-Adapter path first, then fall back to direct import
    try:
        from calvin_env_wrapper import CalvinEnvWrapperRaw
    except ImportError:
        from calvin_env.envs.play_table_env import get_env
        import gym

        class CalvinEnvWrapperRaw(gym.Wrapper):
            def __init__(self, abs_datasets_dir, obs_space, dev, **kwargs):
                env = get_env(abs_datasets_dir, show_gui=False, obs_space=obs_space, **kwargs)
                super().__init__(env)
                self.observation_space_keys = obs_space
                self.device = dev
                self.relative_actions = "rel_actions" in obs_space["actions"]

            def step(self, action):
                if self.relative_actions:
                    assert len(action) == 7
                o, r, d, i = self.env.step(action)
                return o, r, d, i

            def reset(self, robot_obs=None, scene_obs=None, **kwargs):
                if scene_obs is not None or robot_obs is not None:
                    return self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
                return self.env.reset()

            def get_info(self):
                return self.env.get_info()

            def get_obs(self):
                return self.env.get_obs()

    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _fmt_array(a, n=4):
    """Format first n elements of an array for logging."""
    flat = np.asarray(a).flatten()
    vals = " ".join(f"{v:.4f}" for v in flat[:n])
    if len(flat) > n:
        vals += " ..."
    return f"[{vals}]"


# ---------------------------------------------------------------------------
# Episode runner (single subtask)
# ---------------------------------------------------------------------------

def run_calvin_subtask(
    vlm, ae_model, wp_tokenizer, norm_helper, rc,
    env, task_desc, cfg, device, pg_tok,
    ep_len=360,
    task_oracle=None, start_info=None, subtask=None,
    decode_cfg=None, noise_generator=None, stats_sink=None, planner_generator=None,
):
    """Run one CALVIN subtask with the two-stage waypoint pipeline.

    Mirrors the LIBERO runner's behaviour flags (see
    ``eval_libero.run_episode``): ``strict_step_budget``,
    ``segment_start_from``, ``stop_on_d0``, ``replan_on_deviation``.
    Note CALVIN does NOT negate the gripper sign (LIBERO does).

    Returns:
        success: bool
        replay_images: list of (static, gripper) image pairs for video
    """
    model_proprio_dim = cfg.get("model_proprio_dim", 32)
    actual_action_dim = rc.actual_action_dim
    crop_scale = cfg.get("center_crop_scale") if cfg.get("center_crop", False) else None

    strict_budget = cfg.get("strict_step_budget", True)
    # Validate exactly as eval_libero does: a typo used to silently select the
    # other arm here while the results were filed under the requested one.
    segment_start_from = cfg.get("segment_start_from", "predicted")
    if segment_start_from not in ("predicted", "actual"):
        raise ValueError(f"eval.segment_start_from must be 'predicted' or 'actual', got {segment_start_from!r}")
    stop_on_d0 = cfg.get("stop_on_d0", False)
    replan_on_deviation = cfg.get("replan_on_deviation", False)
    deviation_threshold = float(cfg.get("deviation_threshold", 0.15))
    deviation_metric = cfg.get("deviation_metric", "l2")
    if deviation_metric not in ("l2", "linf"):
        raise ValueError(f"eval.deviation_metric must be 'l2' or 'linf', got {deviation_metric!r}")

    t = 0
    done = False
    replay_images = []
    replan_count = 0
    consecutive_d1_zero = 0
    terminate_subtask = False

    obs = env.get_obs()

    while t < ep_len and not done and not terminate_subtask:
        # Collect replay frames
        static_frame = obs["rgb_obs"]["rgb_static"]
        gripper_frame = obs["rgb_obs"]["rgb_gripper"]
        if static_frame is not None:
            replay_images.append(copy.deepcopy(static_frame))

        images = get_calvin_images(obs, center_crop_scale=crop_scale)
        proprio_raw = get_proprio_from_obs_calvin(obs)
        continuous_raw, gripper_binary = rc.split_proprio(proprio_raw)
        continuous_norm = norm_helper.normalize_proprio(continuous_raw)

        t_vlm = time.time()
        waypoints = predict_waypoints(
            vlm, images, task_desc, wp_tokenizer,
            continuous_norm, gripper_binary, device,
            decode_cfg=decode_cfg, generator=planner_generator,
        )
        vlm_ms = (time.time() - t_vlm) * 1000
        if stats_sink is not None:
            stats_sink["planner_ms"].append(vlm_ms)
            accumulate_decode_stats(
                stats_sink["planner"], getattr(vlm, "last_decode_stats", None)
            )

        replan_count += 1
        if not waypoints:
            logger.info(f"  [replan {replan_count}] VLM returned empty waypoints ({vlm_ms:.0f}ms), stopping")
            break

        valid_wps = [(p, d) for p, d in waypoints if d > 0]
        durations = [d for _, d in valid_wps]
        logger.info(
            f"  [replan {replan_count}] VLM: {len(waypoints)} waypoints, "
            f"{len(valid_wps)} valid, durations={durations}, vlm_time={vlm_ms:.0f}ms"
        )

        # Build 7D start_wp: [continuous_norm(6D), gripper_binary(1D)]
        start_wp_7d = np.concatenate([continuous_norm, [float(gripper_binary)]])
        start_wp = pad_to_dim(start_wp_7d, model_proprio_dim)
        steps_this_cycle = 0

        max_dur = cfg.get("horizon_steps", 32)
        budget_hit = False
        for wp_idx, (proprio_values, duration) in enumerate(waypoints):
            if done:
                break
            if duration == 0:
                if stats_sink is not None:
                    stats_sink["d0_terminations"] += 1
                if stop_on_d0:
                    if wp_idx == 0:
                        consecutive_d1_zero += 1
                        if consecutive_d1_zero >= 2:
                            terminate_subtask = True
                    else:
                        terminate_subtask = True
                break
            if duration < 0:
                logger.warning(f"    wp[{wp_idx}]: negative duration {duration}, skipping")
                continue
            if duration > max_dur:
                if stats_sink is not None:
                    stats_sink["duration_overflow_count"] += 1
                logger.warning(f"    wp[{wp_idx}]: duration {duration} exceeds max {max_dur}, clamping")
                duration = max_dur

            if strict_budget and t >= ep_len:
                budget_hit = True
                break

            if segment_start_from == "actual" and wp_idx > 0:
                seg_raw = get_proprio_from_obs_calvin(obs)
                seg_cont_raw, seg_grip = rc.split_proprio(seg_raw)
                seg_cont_norm = norm_helper.normalize_proprio(seg_cont_raw)
                start_wp = pad_to_dim(
                    np.concatenate([seg_cont_norm, [float(seg_grip)]]), model_proprio_dim
                )

            end_wp = pad_to_dim(proprio_values, model_proprio_dim)

            fresh_images = get_calvin_images(obs, center_crop_scale=crop_scale)
            t_ae = time.time()
            actions_norm = predict_actions(
                ae_model, fresh_images, task_desc, start_wp, end_wp, duration, device, pg_tok,
                noise_generator=noise_generator,
            )
            ae_ms = (time.time() - t_ae) * 1000

            num_execute = min(int(duration), actions_norm.shape[0])
            segment_clamped = False
            if strict_budget and num_execute > ep_len - t:
                num_execute = ep_len - t
                segment_clamped = True
                budget_hit = True
            logger.info(
                f"    ae[{wp_idx}]: shape={actions_norm.shape}, execute={num_execute}, "
                f"range=[{actions_norm.min():.3f}, {actions_norm.max():.3f}], ae_time={ae_ms:.0f}ms"
            )

            for step_i in range(num_execute):
                if strict_budget and t >= ep_len:
                    budget_hit = True
                    break
                action_raw = norm_helper.unnormalize_actions(actions_norm[step_i, :actual_action_dim])

                # CALVIN gripper post-processing:
                # Model output is in [0,1] (0=close, 1=open).
                # CALVIN env expects [-1,+1] with +1=open, -1=close.
                # So: x*2-1 -> sign (NO negate, unlike LIBERO).
                gripper = action_raw[-1]
                gripper = gripper * 2.0 - 1.0
                gripper = np.sign(gripper)
                action_raw[-1] = gripper

                obs, reward, done, info = env.step(action_raw.tolist())
                t += 1
                steps_this_cycle += 1

                # Collect replay frames
                static_frame = obs["rgb_obs"]["rgb_static"]
                if static_frame is not None:
                    replay_images.append(copy.deepcopy(static_frame))

                # Check task completion via oracle
                if task_oracle is not None and start_info is not None and subtask is not None:
                    current_info = env.get_info()
                    current_task_info = task_oracle.get_task_info_for_set(
                        start_info, current_info, {subtask}
                    )
                    if len(current_task_info) > 0:
                        logger.info(f"  subtask '{subtask}' succeeded at step {t}")
                        if stats_sink is not None:
                            stats_sink["steps"] += t
                            stats_sink["replans"] += replan_count
                            stats_sink["segments"] += 1
                        return True, replay_images

                if done:
                    break

            if stats_sink is not None:
                stats_sink["segments"] += 1
            consecutive_d1_zero = 0

            dev_l2 = dev_linf = float("nan")
            if not done and not segment_clamped:
                achieved_raw = get_proprio_from_obs_calvin(obs)
                achieved_cont_raw, _ = rc.split_proprio(achieved_raw)
                achieved_norm = norm_helper.normalize_proprio(achieved_cont_raw)
                n_cont = min(len(achieved_norm), len(proprio_values) - 1)
                delta = np.asarray(achieved_norm[:n_cont]) - np.asarray(proprio_values[:n_cont])
                dev_l2 = float(np.linalg.norm(delta))
                dev_linf = float(np.max(np.abs(delta))) if n_cont > 0 else float("nan")
                if stats_sink is not None:
                    stats_sink["segment_deviation_l2"].append(dev_l2)
                    stats_sink["segment_deviation_linf"].append(dev_linf)

            start_wp = end_wp.copy()

            if budget_hit:
                break

            dev = dev_l2 if deviation_metric == "l2" else dev_linf
            if replan_on_deviation and not done and dev == dev and dev > deviation_threshold:
                if stats_sink is not None:
                    stats_sink["deviation_replans"] += 1
                break

        if budget_hit:
            if stats_sink is not None:
                stats_sink["budget_truncated"] = True
            break

        if steps_this_cycle == 0 and not done and not terminate_subtask:
            if strict_budget and t >= ep_len:
                if stats_sink is not None:
                    stats_sink["budget_truncated"] = True
                break
            logger.warning(f"  [replan {replan_count}] no actions executed, advancing with no-op")
            obs, reward, done, info = env.step(np.zeros(actual_action_dim).tolist())
            t += 1

    if stats_sink is not None:
        stats_sink["steps"] += t
        stats_sink["replans"] += replan_count
    if strict_budget:
        assert t <= ep_len, f"CALVIN step budget violated: {t} > {ep_len}"
    logger.info(f"  subtask done: steps={t}, replans={replan_count}, success=False")
    return False, replay_images


# ---------------------------------------------------------------------------
# Sequence evaluation (chain of 5 subtasks)
# ---------------------------------------------------------------------------

def evaluate_sequence(
    env, vlm, ae_model, wp_tokenizer, norm_helper, rc,
    task_oracle, initial_state, eval_sequence, val_annotations,
    cfg, device, pg_tok,
    eval_dir=None, sequence_i=0,
    ep_len=360, decode_cfg=None, noise_generator=None, stats_sink=None,
    planner_generator=None,
):
    """Evaluate a chain of 5 subtasks. Returns number of consecutive successes."""
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    for subtask_i, subtask in enumerate(eval_sequence):
        lang_annotation = val_annotations[subtask][0]
        start_info = env.get_info()

        success, replay_images = run_calvin_subtask(
            vlm, ae_model, wp_tokenizer, norm_helper, rc,
            env, lang_annotation, cfg, device, pg_tok,
            ep_len=ep_len,
            task_oracle=task_oracle,
            start_info=start_info,
            subtask=subtask,
            decode_cfg=decode_cfg,
            noise_generator=noise_generator,
            stats_sink=stats_sink,
            planner_generator=planner_generator,
        )

        # Save video
        if eval_dir is not None and replay_images:
            suffix = "succ" if success else "fail"
            video_file = os.path.join(
                eval_dir, f"{sequence_i}-{subtask_i}-{subtask}-static-{suffix}.mp4"
            )
            try:
                imageio.mimwrite(
                    video_file,
                    [np.asarray(x) for x in replay_images],
                    fps=20,
                )
            except Exception as e:
                logger.warning(f"Failed to save video: {e}")

        if success:
            success_counter += 1
        else:
            return success_counter

    return success_counter


# ---------------------------------------------------------------------------
# CALVIN chain-task metrics
# ---------------------------------------------------------------------------

def count_success(results):
    """Compute chain success rates from list of success counts (0-5)."""
    n = len(results)
    if n == 0:
        return []
    success_rates = []
    for i in range(1, 6):
        success_rates.append(sum(r >= i for r in results) / n)
    return success_rates


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # CALVIN used to have no seeding at all, so its numbers were not reproducible
    # and not comparable with the LIBERO arm.  Mirror eval_libero exactly.
    eval_seed = cfg.get("eval_seed", None)
    noise_generator = None
    planner_generator = None
    if eval_seed is not None:
        _setup_seed(eval_seed, deterministic_sdpa=cfg.get("deterministic_sdpa", False))
        noise_generator = torch.Generator(device="cpu")
        noise_generator.manual_seed(eval_seed)
        planner_generator = torch.Generator(device="cpu")
        planner_generator.manual_seed(eval_seed + 1_000_003)

    t_total = time.time()

    # Load model
    use_joint = "joint_checkpoint" in cfg
    if use_joint:
        joint_model = load_joint(cfg, device)
        vlm = joint_model
        ae_model = joint_model
        logger.info(f"Joint model loaded (shared backbone): {time.time() - t_total:.1f}s")
    else:
        vlm = load_vlm(cfg, device)
        ae_model = load_ae(cfg, device)
        logger.info(f"Total model loading (separate VLM+AE): {time.time() - t_total:.1f}s")

    t0 = time.time()
    pg_tok = load_pg_tokenizer()
    logger.info(f"PaliGemma tokenizer loaded: {time.time() - t0:.1f}s")

    rc = get_robot_config("calvin")
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
            raise ValueError("eval.center_crop is true but eval.center_crop_scale is not set")
        logger.info(
            f"Center crop enabled: area_scale={cfg['center_crop_scale']}, "
            f"side_ratio={math.sqrt(cfg['center_crop_scale']):.4f}"
        )

    decode_cfg = WaypointDecodeConfig.from_dict(cfg)
    logger.info(f"Planner decode: {decode_cfg}")
    _check_decode_compatibility(decode_cfg, cfg, vlm, use_joint)
    episode_stats: list[dict] = []

    # Set up CALVIN environment
    calvin_dataset_path = cfg.get("calvin_dataset_path", os.path.join(CALVIN_ROOT, "dataset/task_ABC_D"))
    logger.info(f"CALVIN dataset path: {calvin_dataset_path}")

    t0 = time.time()
    env = make_calvin_env(calvin_dataset_path, device)
    logger.info(f"CALVIN env initialized: {time.time() - t0:.1f}s")

    # Load task oracle and annotations
    import hydra
    from omegaconf import OmegaConf

    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # Get evaluation sequences
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import count_success as calvin_count_success

    num_sequences = cfg.get("num_sequences", 1000)
    ep_len = cfg.get("ep_len", 360)
    eval_sequences = get_sequences(num_sequences)

    # Set up output directory
    eval_dir = cfg.get("video_out_path", "data/calvin/videos_wp")
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Videos will be saved to: {eval_dir}")

    # Run evaluation
    results = []
    for seq_i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        t_seq = time.time()
        logger.info(f"Sequence {seq_i}/{num_sequences}: {' -> '.join(eval_sequence)}")

        if noise_generator is not None:
            noise_generator.manual_seed(eval_seed + seq_i)
        if planner_generator is not None:
            planner_generator.manual_seed(eval_seed + 1_000_003 + seq_i)
        seq_stats = _episode_stats()
        result = evaluate_sequence(
            env, vlm, ae_model, wp_tokenizer, norm_helper, rc,
            task_oracle, initial_state, eval_sequence, val_annotations,
            cfg, device, pg_tok,
            eval_dir=eval_dir, sequence_i=seq_i,
            ep_len=ep_len, decode_cfg=decode_cfg,
            noise_generator=noise_generator, stats_sink=seq_stats,
            planner_generator=planner_generator,
        )
        episode_stats.append({"sequence": seq_i, "chain_len": int(result), **seq_stats})
        results.append(result)
        seq_secs = time.time() - t_seq

        # Log running metrics
        success_rates = count_success(results)
        avg_seq_len = np.mean(results)
        sr_str = " | ".join(f"{i+1}/5: {sr:.1%}" for i, sr in enumerate(success_rates))
        logger.info(
            f"  -> {result}/5 subtasks ({seq_secs:.1f}s) | "
            f"avg_seq_len: {avg_seq_len:.2f} | {sr_str}"
        )

    # Final results
    avg_seq_len = np.mean(results)
    chain_sr = count_success(results)

    logger.info(f"\n{'='*60}")
    logger.info(f"CALVIN Evaluation Results ({num_sequences} sequences)")
    logger.info(f"Average successful sequence length: {avg_seq_len:.3f}")
    logger.info("Chain success rates:")
    for i, sr in enumerate(chain_sr):
        logger.info(f"  {i+1}/5: {sr:.1%}")
    logger.info(f"Total eval time: {time.time() - t_total:.1f}s")

    # Save results to JSON
    results_path = cfg.get("_results_file_override") or os.path.join(eval_dir, "eval_results.json")
    diagnostics = summarize_episode_stats(episode_stats, decode_cfg)
    log_diagnostics(diagnostics)
    results_data = {
        "avg_seq_len": float(avg_seq_len),
        "chain_sr": {str(i+1): float(sr) for i, sr in enumerate(chain_sr)},
        "num_sequences": num_sequences,
        "eval_seed": eval_seed,
        "per_sequence_results": results,
        "diagnostics": diagnostics,
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return results_data


def _resolve_config(cfg):
    """Merge eval section into top-level for unified config support."""
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
    parser.add_argument("--results-file", type=str, default=None,
                        help="Override path for results JSON output")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=None,
                        help="Prefer lora_ema.safetensors over lora.safetensors")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false",
                        help="Force raw LoRA weights even if eval.use_ema is true")
    parser.add_argument("--decode-impl", type=str, default=None, choices=list(DECODE_IMPLS),
                        help="Planner decode implementation")
    parser.add_argument("--profile", action="store_true", help="Enable CUDA-event planner timing")
    parser.add_argument("--eval-seed", type=int, default=None, help="Seed all RNGs for reproducibility")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_config(cfg)

    if args.results_file:
        cfg["_results_file_override"] = args.results_file
    if args.use_ema is not None:
        cfg["use_ema"] = args.use_ema
    if args.decode_impl is not None:
        cfg["waypoint_decode"] = {**cfg.get("waypoint_decode", {}), "impl": args.decode_impl}
    if args.profile:
        cfg["waypoint_decode"] = {**cfg.get("waypoint_decode", {}), "profile": True}
    if args.eval_seed is not None:
        cfg["eval_seed"] = args.eval_seed

    evaluate(cfg)


if __name__ == "__main__":
    main()

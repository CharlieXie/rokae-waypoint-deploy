"""VLM waypoint prediction dataset.

Reads waypoint-filtered RLDS data to produce autoregressive training samples
for the VLM waypoint predictor.

Each sample contains tokenized: images + "Task: instruction, State: s;\nAction: <wp>p...<dur>d...|"
"""

import gc
import hashlib
import json
import logging
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms as T

from openpi.waypoint.normalize import NormalizationHelper
from openpi.waypoint.normalize import extract_proprio_from_obs
from openpi.waypoint.normalize import pad_to_dim
from openpi.waypoint.robot_config import RobotConfig
from openpi.waypoint.tokenizer import DURATION_MAX
from openpi.waypoint.tokenizer import WaypointTokenizer

logger = logging.getLogger(__name__)


def _derive_shuffle_seed(
    base_seed: int | None,
    *,
    rank: int,
    epoch: int,
    stream: str,
) -> int | None:
    """Derive a stable, independent 31-bit seed for one shuffle stream.

    Python's built-in ``hash`` is process-randomized, so it cannot be used for
    a reproducibility contract.  BLAKE2s gives the same result across processes
    and keeps the rank / epoch / stream namespaces explicit.  TensorFlow accepts
    the resulting signed-int32-range value directly; NumPy accepts it as well.
    ``None`` deliberately stays ``None`` so existing configs remain
    nondeterministic.
    """
    if base_seed is None:
        return None
    payload = (
        f"openpi-waypoint-shuffle-v1\0{int(base_seed)}\0{int(rank)}\0"
        f"{int(epoch)}\0{stream}"
    ).encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little") & 0x7FFF_FFFF


def validate_shard(shard) -> tuple[int, int] | None:
    """Validate an explicit ``(world_size, rank)`` episode-sharding override.

    ``None`` keeps the DDP-derived sharding (the training default).  The
    override exists for single-process consumers (the sharded offline
    evaluator) that want the exact episode split DDP rank ``rank`` of
    ``world_size`` would see, without a process group.
    """
    if shard is None:
        return None
    try:
        world_size, rank = (int(shard[0]), int(shard[1]))
    except (TypeError, ValueError, IndexError) as e:
        raise ValueError(f"shard must be a (world_size, rank) pair, got {shard!r}") from e
    if world_size < 1 or not (0 <= rank < world_size):
        raise ValueError(f"shard=(world_size={world_size}, rank={rank}) is out of range")
    return (world_size, rank)


def _distributed_rank() -> int:
    """Return the live DDP rank, or zero outside a process group."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def build_image_augmentation(aug_cfg: dict | None = None) -> T.Compose:
    """Build a torchvision augmentation pipeline from config.

    Default parameters match the galaxea_0 reference:
      brightness=0.2, contrast=[0.8,1.2], saturation=[0.8,1.2], hue=0.05,
      random_resized_crop scale=[0.9,1.0].
    """
    if aug_cfg is None:
        aug_cfg = {}

    enable_crop = aug_cfg.get("enable_random_resized_crop", True)
    crop_scale = tuple(aug_cfg.get("random_resized_crop_scale", [0.9, 1.0]))
    brightness = aug_cfg.get("brightness", 0.2)
    contrast = tuple(aug_cfg.get("contrast", [0.8, 1.2]))
    saturation = tuple(aug_cfg.get("saturation", [0.8, 1.2]))
    hue = aug_cfg.get("hue", 0.05)

    tfms = []
    if enable_crop:
        tfms.append(T.RandomResizedCrop(224, scale=crop_scale, ratio=(1.0, 1.0), antialias=True))
    tfms.append(T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    ))
    return T.Compose(tfms)


class WaypointVLMDataset(IterableDataset):
    """RLDS-based VLM waypoint training dataset.

    Reads waypoint-filtered RLDS (where each step IS a waypoint, with
    `waypoint_duration` and `is_waypoint_end` fields) and produces
    tokenized AR training samples.
    """

    def __init__(
        self,
        wp_rlds_dir: str,
        robot_config: RobotConfig,
        dataset_statistics: dict[str, Any],
        wp_tokenizer: WaypointTokenizer,
        norm_type: str = "q99",
        num_waypoints: int = 7,
        stride: int = 1,
        image_size: tuple[int, int] = (224, 224),
        shuffle_buffer_size: int = 5000,
        image_aug: bool = False,
        image_aug_cfg: dict | None = None,
        episode_shuffle_buffer: int = 0,
        vlm_max_shift: int = 0,
        original_rlds_dir: str | None = None,
        wp_indices_path: str | None = None,
        shuffle_seed: int | None = None,
        shard: tuple[int, int] | None = None,
        max_epochs: int | None = None,
    ):
        """``shard``: explicit ``(world_size, rank)`` episode split for
        non-DDP consumers (see :func:`validate_shard`); ``None`` = derive from
        ``torch.distributed`` as before.  ``max_epochs``: stop iterating after
        this many passes over the data (``None`` = endless, the training
        default).  Both default to the historical behaviour."""
        super().__init__()
        self.shard = validate_shard(shard)
        if max_epochs is not None and int(max_epochs) < 1:
            raise ValueError(f"max_epochs must be >= 1 or None, got {max_epochs!r}")
        self.max_epochs = None if max_epochs is None else int(max_epochs)
        self.wp_rlds_dir = wp_rlds_dir
        self.robot_config = robot_config
        self.wp_tokenizer = wp_tokenizer
        self.norm_type = norm_type
        self.num_waypoints = num_waypoints
        self.stride = stride
        self.image_size = image_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.episode_shuffle_buffer = episode_shuffle_buffer
        self.shuffle_seed = shuffle_seed
        self._sample_shuffle_rng: np.random.Generator | None = None

        self.norm_helper = NormalizationHelper(dataset_statistics, norm_type)

        # The planner names every gripper of the robot in every waypoint: the
        # tokenizer's gripper slots must equal the robot's gripper count, or a
        # dual-arm robot would silently train on its first gripper only (the
        # bypass that ``split_proprio``'s guard could never see from here).
        n_slots = getattr(wp_tokenizer, "n_gripper_slots", 1)
        if wp_tokenizer.use_gripper_token and n_slots != robot_config.num_grippers():
            raise ValueError(
                f"WaypointTokenizer has {n_slots} gripper slot(s) but robot "
                f"{robot_config.robot_type!r} declares {robot_config.num_grippers()} "
                f"gripper(s) {robot_config.gripper_dim_indices}; build the tokenizer with "
                "n_gripper_slots=robot_config.num_grippers()"
            )

        # Counters for windows dropped by the duration bounds check.  Silent
        # `continue`s are how a dataset quietly stops covering part of its own
        # distribution, so they are counted and logged per epoch.
        self._skipped_bad_duration = 0
        self._yielded_samples = 0

        self.image_aug_transform = None
        if image_aug:
            self.image_aug_transform = build_image_augmentation(image_aug_cfg)
            logger.info(f"Image augmentation enabled: {self.image_aug_transform}")

        logger.info(
            f"WaypointVLMDataset: dir={wp_rlds_dir}, M={num_waypoints}, stride={stride}, "
            f"robot={robot_config.robot_type}, image_aug={image_aug}, "
            f"episode_shuffle_buffer={episode_shuffle_buffer}, shuffle_seed={shuffle_seed}"
        )

        # --- VLM shift augmentation (reads from original RLDS + waypoint indices) ---
        self.vlm_max_shift = vlm_max_shift
        self.original_rlds_dir = original_rlds_dir
        self.episode_wp_indices_map: dict[int, list[int]] = {}

        if vlm_max_shift > 0:
            assert original_rlds_dir is not None, (
                "vlm_max_shift > 0 requires original_rlds_dir"
            )
            assert wp_indices_path is not None, (
                "vlm_max_shift > 0 requires wp_indices_path"
            )

            logger.info(f"Loading waypoint indices for VLM shift from {wp_indices_path}")
            with open(wp_indices_path) as f:
                wp_data = json.load(f)

            episode_lengths: dict[int, int] = {}
            for ep in wp_data["episodes"]:
                ep_idx = ep["src_ep_idx"]
                ep_len = ep.get("original_steps", 0)
                episode_lengths[ep_idx] = ep_len
                self.episode_wp_indices_map[ep_idx] = ep["waypoint_indices"]

            M = self.num_waypoints
            total_original = 0
            total_augmented = 0
            skipped_50pct = 0
            skipped_bounds = 0
            for ep in wp_data["episodes"]:
                ep_idx = ep["src_ep_idx"]
                ep_len = episode_lengths[ep_idx]
                wpi = ep["waypoint_indices"]
                num_wps = len(wpi)
                for i in range(0, num_wps - 1, self.stride):
                    actual_wps = min(M, num_wps - 1 - i)
                    if actual_wps < 1:
                        continue
                    total_original += 1
                    d1 = wpi[i + 1] - wpi[i]
                    if d1 >= 2 * vlm_max_shift:
                        for s in range(-vlm_max_shift, vlm_max_shift + 1):
                            ns = wpi[i] + s
                            if ns < 0 or ns >= ep_len:
                                skipped_bounds += 1
                                continue
                            d1n = wpi[i + 1] - ns
                            if d1n < 1 or d1n > DURATION_MAX:
                                skipped_bounds += 1
                                continue
                            total_augmented += 1
                    else:
                        skipped_50pct += 1
                        total_augmented += 1

            # NOTE: this is an *estimate*, not a count.  It replays only the
            # window/shift guards, not the runtime duration bounds check or the
            # per-episode index/length validation, and it degenerates to
            # "everything skipped" when `original_steps` is absent (ep_len = 0).
            # Use the per-epoch `yielded=` line below as the ground truth.
            logger.info(
                f"VLM shift augmentation (estimate): {total_original} starting points, "
                f"vlm_max_shift={vlm_max_shift}, "
                f"d1>=2N passed: {total_original - skipped_50pct}, "
                f"skipped_50pct: {skipped_50pct}, "
                f"skipped_bounds: {skipped_bounds} -> "
                f"~{total_augmented} samples"
            )

    # ------------------------------------------------------------------
    # Shifted-path helpers (original RLDS + waypoint_indices.json)
    # ------------------------------------------------------------------

    def _process_episode_shifted(self, steps, wp_indices, rc, ep_idx_for_log=-1):
        """Extract VLM samples from an original-RLDS episode with shift augmentation.

        For each waypoint starting point (with stride), generates shifted
        versions where the observation frame is moved ±N steps along the
        original timeline.  Only the first duration (d₁) changes; all
        subsequent waypoint targets remain identical.
        """
        num_steps = len(steps)
        num_wps = len(wp_indices)
        M = self.num_waypoints
        max_s = self.vlm_max_shift

        if wp_indices[-1] >= num_steps:
            logger.warning(
                f"Episode {ep_idx_for_log}: last wp_index {wp_indices[-1]} >= "
                f"num_steps {num_steps}, skipping entire episode"
            )
            return

        all_proprios_cont_norm = np.empty(
            (num_steps, rc.continuous_proprio_dim), dtype=np.float32
        )
        # [T, n_grippers]: EVERY gripper, in gripper_dim_indices order.
        all_grippers = np.empty((num_steps, rc.num_grippers()), dtype=np.int32)

        for t in range(num_steps):
            obs_dict = {k: steps[t]["observation"][k] for k in steps[t]["observation"]}
            raw = extract_proprio_from_obs(obs_dict, rc.state_obs_keys)
            all_proprios_cont_norm[t] = self.norm_helper.normalize_proprio(
                rc.select_continuous(raw)
            )
            all_grippers[t] = rc.binarize_grippers(raw)

        instruction = steps[0]["language_instruction"].numpy()
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")

        for i in range(0, num_wps - 1, self.stride):
            actual_wps = min(M, num_wps - 1 - i)
            if actual_wps < 1:
                continue

            d1_orig = wp_indices[i + 1] - wp_indices[i]

            # Durations 2..M are shift-invariant, so check them ONCE per window
            # rather than once per shift: doing it inside the shift loop decoded
            # and augmented three sets of camera images before discarding them,
            # and made the counter report (window, shift) pairs while the log
            # presents it as windows dropped.  d1 stays per-shift below.
            later_gaps = [
                wp_indices[i + j + 1] - wp_indices[i + j] for j in range(1, actual_wps)
            ]
            if any(not (1 <= g <= DURATION_MAX) for g in later_gaps):
                self._skipped_bad_duration += 1
                continue

            if max_s > 0 and d1_orig >= 2 * max_s:
                shifts = range(-max_s, max_s + 1)
            else:
                shifts = [0]

            for s in shifts:
                new_start = wp_indices[i] + s

                if new_start < 0 or new_start >= num_steps:
                    continue
                d1_new = wp_indices[i + 1] - new_start
                if d1_new < 1 or d1_new > DURATION_MAX:
                    continue

                images = {}
                image_masks = {}
                for view, rlds_key in rc.camera_rlds_keys.items():
                    model_key = rc.camera_model_keys[view]
                    img_data = steps[new_start]["observation"][rlds_key].numpy()
                    img = Image.fromarray(img_data)
                    if img.size != self.image_size:
                        img = img.resize(self.image_size, Image.BILINEAR)
                    if self.image_aug_transform is not None:
                        img = self.image_aug_transform(img)
                    images[model_key] = np.array(img, dtype=np.uint8)
                    image_masks[model_key] = True

                current_cont_norm = all_proprios_cont_norm[new_start]
                state_padded = np.zeros(
                    self.wp_tokenizer.proprio_dim, dtype=np.float32
                )
                state_padded[:rc.continuous_proprio_dim] = current_cont_norm
                current_gripper = all_grippers[new_start]  # [n_grippers]

                wp_proprios = np.zeros(
                    (M, rc.continuous_proprio_dim), dtype=np.float32
                )
                wp_grippers_arr = np.zeros((M, rc.num_grippers()), dtype=np.int32)
                wp_durations = np.zeros(M, dtype=np.int32)
                wp_pad_mask_proprio = np.ones(M, dtype=bool)
                wp_pad_mask_duration = np.ones(M, dtype=bool)

                # d1_new was range-checked above; later gaps were validated once
                # per window before the shift loop.
                for j in range(actual_wps):
                    wp_frame = wp_indices[i + 1 + j]
                    wp_proprios[j] = all_proprios_cont_norm[wp_frame]
                    wp_grippers_arr[j] = all_grippers[wp_frame]
                    wp_pad_mask_proprio[j] = False
                    wp_durations[j] = d1_new if j == 0 else later_gaps[j - 1]
                    wp_pad_mask_duration[j] = False

                is_episode_end = (i + actual_wps >= num_wps - 1)
                if is_episode_end and actual_wps < M:
                    wp_durations[actual_wps] = 0
                    wp_pad_mask_duration[actual_wps] = False

                wp_proprios_padded = np.zeros(
                    (M, self.wp_tokenizer.proprio_dim), dtype=np.float32
                )
                wp_proprios_padded[:, :rc.continuous_proprio_dim] = wp_proprios

                tokens, token_mask, ar_mask, loss_mask = self.wp_tokenizer.tokenize(
                    prompt=instruction,
                    state=state_padded,
                    wp_proprios=wp_proprios_padded,
                    wp_durations=wp_durations,
                    current_gripper=current_gripper,
                    wp_grippers=wp_grippers_arr,
                    wp_pad_mask_proprio=wp_pad_mask_proprio,
                    wp_pad_mask_duration=wp_pad_mask_duration,
                )

                self._yielded_samples += 1
                yield {
                    "images": images,
                    "image_masks": image_masks,
                    "tokens": tokens,
                    "token_mask": token_mask,
                    "ar_mask": ar_mask,
                    "loss_mask": loss_mask,
                    # Input-only "waypoint -1" block; consumed only when the
                    # model is built with planner_block0_cond='current_state'.
                    # It carries no label, so emitting it unconditionally cannot
                    # change what any other configuration trains on.
                    "current_wp_tokens": self.wp_tokenizer.encode_current_state_block(
                        current_cont_norm, current_gripper
                    ),
                }

    def _raw_sample_iter_shifted(self):
        """VLM sample iterator using original RLDS + waypoint indices.

        Follows the same enumerate-shard-shuffle pattern as ae_dataset to
        ensure correct episode indexing with DDP and episode-level shuffle.
        """
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        try:
            import torch.distributed as dist
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size, rank = 1, 0
        except Exception:
            world_size, rank = 1, 0
        if self.shard is not None:
            # Explicit override for single-process (non-DDP) consumers such as
            # the sharded offline evaluator: episode i goes to rank i % world_size,
            # exactly as under DDP.
            world_size, rank = self.shard

        rc = self.robot_config
        use_episode_shuffle = self.episode_shuffle_buffer > 0

        # See ae_dataset.py for the rationale: rebuild TFDS per epoch so the
        # interleave reader pool / iterator state is released, while keeping
        # cycle_length=16 to preserve waypoint_indices.json alignment.
        epoch_count = 0
        episodes_seen = 0
        while self.max_epochs is None or epoch_count < self.max_epochs:
            epoch_count += 1
            builder = tfds.builder_from_directory(self.original_rlds_dir)
            read_config = tfds.ReadConfig(
                interleave_cycle_length=16,
                interleave_block_length=16,
            )
            dataset = builder.as_dataset(split="train", read_config=read_config)
            options = tf.data.Options()
            options.autotune.enabled = False
            options.threading.private_threadpool_size = 1
            options.experimental_optimization.apply_default_optimizations = False
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            dataset = dataset.with_options(options)
            dataset = dataset.enumerate()
            if world_size > 1:
                dataset = dataset.shard(world_size, rank)
            if use_episode_shuffle:
                episode_seed = _derive_shuffle_seed(
                    self.shuffle_seed,
                    rank=rank,
                    epoch=epoch_count,
                    stream="vlm_episode",
                )
                if episode_seed is None:
                    dataset = dataset.shuffle(buffer_size=self.episode_shuffle_buffer)
                else:
                    dataset = dataset.shuffle(
                        buffer_size=self.episode_shuffle_buffer,
                        seed=episode_seed,
                        reshuffle_each_iteration=False,
                    )

            for orig_idx_tensor, episode in dataset:
                ep_idx = int(orig_idx_tensor.numpy())
                wpi = self.episode_wp_indices_map.get(ep_idx)
                if wpi is None:
                    continue
                steps = list(episode["steps"])
                if len(steps) < 2:
                    continue
                yield from self._process_episode_shifted(
                    steps, wpi, rc, ep_idx_for_log=ep_idx,
                )
                del steps
                episodes_seen += 1
                if episodes_seen % 100 == 0:
                    gc.collect()

            del dataset, builder
            gc.collect()
            logger.info(
                f"VLM(shift) dataset epoch {epoch_count} complete "
                f"(rank={rank}, episodes_seen={episodes_seen}, "
                f"yielded={self._yielded_samples}, "
                f"skipped_bad_duration={self._skipped_bad_duration}); "
                "rebuilding TFDS graph."
            )

    def _raw_sample_iter(self):
        """Yield tokenized samples.  Dispatches to shifted path when enabled."""
        if self.vlm_max_shift > 0:
            yield from self._raw_sample_iter_shifted()
            return

        # === Original wp_rlds path (unchanged) ===
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        try:
            import torch.distributed as dist
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size, rank = 1, 0
        except Exception:
            world_size, rank = 1, 0
        if self.shard is not None:
            # Explicit override for single-process (non-DDP) consumers such as
            # the sharded offline evaluator: episode i goes to rank i % world_size,
            # exactly as under DDP.
            world_size, rank = self.shard

        rc = self.robot_config
        M = self.num_waypoints
        use_episode_shuffle = self.episode_shuffle_buffer > 0

        # See _raw_sample_iter_shifted above for the rebuild-per-epoch
        # rationale.  This wp_rlds path is only used when vlm_max_shift==0;
        # current Calvin configs use vlm_max_shift>=1 (shifted path).
        epoch_count = 0
        episodes_seen = 0
        while self.max_epochs is None or epoch_count < self.max_epochs:
            epoch_count += 1
            builder = tfds.builder_from_directory(self.wp_rlds_dir)
            read_config = tfds.ReadConfig(
                interleave_cycle_length=16,
                interleave_block_length=16,
            )
            dataset = builder.as_dataset(split="train", read_config=read_config)
            options = tf.data.Options()
            options.autotune.enabled = False
            options.threading.private_threadpool_size = 1
            options.experimental_optimization.apply_default_optimizations = False
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            dataset = dataset.with_options(options)
            if world_size > 1:
                dataset = dataset.shard(world_size, rank)
            if use_episode_shuffle:
                episode_seed = _derive_shuffle_seed(
                    self.shuffle_seed,
                    rank=rank,
                    epoch=epoch_count,
                    stream="vlm_episode",
                )
                if episode_seed is None:
                    dataset = dataset.shuffle(buffer_size=self.episode_shuffle_buffer)
                else:
                    dataset = dataset.shuffle(
                        buffer_size=self.episode_shuffle_buffer,
                        seed=episode_seed,
                        reshuffle_each_iteration=False,
                    )

            for episode in dataset:
                steps = list(episode["steps"])
                num_steps = len(steps)
                if num_steps < 2:
                    continue

                all_proprios_raw = []
                all_grippers = []
                all_durations = []
                all_is_end = []
                for s in steps:
                    proprio_raw = extract_proprio_from_obs(
                        {k: s["observation"][k] for k in s["observation"]},
                        rc.state_obs_keys,
                    )
                    all_proprios_raw.append(proprio_raw)
                    # Binarize EVERY gripper from raw state -> [n_grippers]
                    all_grippers.append(rc.binarize_grippers(proprio_raw))
                    all_durations.append(int(s["waypoint_duration"].numpy()))
                    all_is_end.append(bool(s.get("is_waypoint_end", s["is_last"]).numpy()))

                # Normalize only continuous dims (exclude gripper)
                all_proprios_continuous_norm = np.stack([
                    self.norm_helper.normalize_proprio(rc.select_continuous(p))
                    for p in all_proprios_raw
                ])

                instruction = steps[0]["language_instruction"].numpy()
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")

                for start_idx in range(0, num_steps - 1, self.stride):
                    end_idx = min(start_idx + 1 + M, num_steps)
                    actual_wps = end_idx - start_idx - 1
                    if actual_wps < 1:
                        continue

                    wp_proprios = np.zeros((M, rc.continuous_proprio_dim), dtype=np.float32)
                    wp_grippers = np.zeros((M, rc.num_grippers()), dtype=np.int32)
                    wp_durations = np.zeros(M, dtype=np.int32)
                    wp_pad_mask_proprio = np.ones(M, dtype=bool)
                    wp_pad_mask_duration = np.ones(M, dtype=bool)

                    bad_duration = False
                    for j in range(actual_wps):
                        wp_proprios[j] = all_proprios_continuous_norm[start_idx + 1 + j]
                        wp_grippers[j] = all_grippers[start_idx + 1 + j]
                        wp_pad_mask_proprio[j] = False
                        wp_durations[j] = all_durations[start_idx + j]
                        # This path never range-checked any slot: waypoint_duration
                        # comes straight from the filtered RLDS and was clipped to
                        # DURATION_MAX by the tokenizer.  See the shifted path above.
                        if not (1 <= wp_durations[j] <= DURATION_MAX):
                            self._skipped_bad_duration += 1
                            bad_duration = True
                            break
                        wp_pad_mask_duration[j] = False
                    if bad_duration:
                        continue

                    last_wp_step = start_idx + actual_wps
                    if all_is_end[min(last_wp_step, num_steps - 1)]:
                        if actual_wps < M:
                            wp_durations[actual_wps] = 0
                            wp_pad_mask_duration[actual_wps] = False

                    # Pad continuous proprio to tokenizer's proprio_dim
                    wp_proprios_padded = np.zeros((M, self.wp_tokenizer.proprio_dim), dtype=np.float32)
                    wp_proprios_padded[:, :rc.continuous_proprio_dim] = wp_proprios

                    # Current state: continuous dims only
                    current_continuous_norm = all_proprios_continuous_norm[start_idx]
                    state_padded = np.zeros(self.wp_tokenizer.proprio_dim, dtype=np.float32)
                    state_padded[:rc.continuous_proprio_dim] = current_continuous_norm
                    current_gripper = all_grippers[start_idx]

                    images = {}
                    image_masks = {}
                    for view, rlds_key in rc.camera_rlds_keys.items():
                        model_key = rc.camera_model_keys[view]
                        img_data = steps[start_idx]["observation"][rlds_key].numpy()
                        img = Image.fromarray(img_data)
                        if img.size != self.image_size:
                            img = img.resize(self.image_size, Image.BILINEAR)
                        if self.image_aug_transform is not None:
                            img = self.image_aug_transform(img)
                        images[model_key] = np.array(img, dtype=np.uint8)
                        image_masks[model_key] = True

                    tokens, token_mask, ar_mask, loss_mask = self.wp_tokenizer.tokenize(
                        prompt=instruction,
                        state=state_padded,
                        wp_proprios=wp_proprios_padded,
                        wp_durations=wp_durations,
                        current_gripper=current_gripper,
                        wp_grippers=wp_grippers,
                        wp_pad_mask_proprio=wp_pad_mask_proprio,
                        wp_pad_mask_duration=wp_pad_mask_duration,
                    )

                    self._yielded_samples += 1
                    yield {
                        "images": images,
                        "image_masks": image_masks,
                        "tokens": tokens,
                        "token_mask": token_mask,
                        "ar_mask": ar_mask,
                        "loss_mask": loss_mask,
                        # See the shifted iterator: input-only, never a label.
                        "current_wp_tokens": self.wp_tokenizer.encode_current_state_block(
                            current_continuous_norm, current_gripper
                        ),
                    }

                del steps, all_proprios_raw, all_proprios_continuous_norm
                episodes_seen += 1
                if episodes_seen % 100 == 0:
                    gc.collect()

            del dataset, builder
            gc.collect()
            logger.info(
                f"VLM(wp_rlds) dataset epoch {epoch_count} complete "
                f"(rank={rank}, episodes_seen={episodes_seen}, "
                f"yielded={self._yielded_samples}, "
                f"skipped_bad_duration={self._skipped_bad_duration}); "
                "rebuilding TFDS graph."
            )

    def __iter__(self):
        """Yields shuffled samples via a sliding-window buffer.

        Fill buffer up to shuffle_buffer_size, then for every new sample
        yield one random item from the buffer (keeps buffer size constant).
        """
        if self._sample_shuffle_rng is None:
            seed = _derive_shuffle_seed(
                self.shuffle_seed,
                rank=_distributed_rank(),
                epoch=0,
                stream="vlm_sample",
            )
            self._sample_shuffle_rng = np.random.default_rng(seed)

        buffer = []
        logged_fill = False
        for sample in self._raw_sample_iter():
            buffer.append(sample)
            if len(buffer) < self.shuffle_buffer_size:
                if not logged_fill and len(buffer) == 1:
                    logger.info(
                        f"VLM shuffle buffer filling: target={self.shuffle_buffer_size} samples"
                    )
                continue
            if not logged_fill:
                logger.info(f"VLM shuffle buffer full ({len(buffer)} samples), training starts")
                logged_fill = True
            idx = int(self._sample_shuffle_rng.integers(len(buffer)))
            yield buffer.pop(idx)


class WaypointVLMCollator:
    """Collates VLM samples into batched tensors."""

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)

        all_images = {}
        all_image_masks = {}
        image_keys = sorted(batch[0]["images"].keys())
        for key in image_keys:
            imgs = np.stack([s["images"][key] for s in batch])  # (B, H, W, C)
            imgs = imgs.astype(np.float32) / 127.5 - 1.0
            imgs = imgs.transpose(0, 3, 1, 2)  # (B, C, H, W)
            all_images[key] = torch.from_numpy(imgs)
            masks = [s["image_masks"].get(key, False) for s in batch]
            all_image_masks[key] = torch.tensor(masks, dtype=torch.bool)

        tokens = torch.from_numpy(np.stack([s["tokens"] for s in batch]))
        token_mask = torch.from_numpy(np.stack([s["token_mask"] for s in batch]))
        ar_mask = torch.from_numpy(np.stack([s["ar_mask"] for s in batch]))
        loss_mask = torch.from_numpy(np.stack([s["loss_mask"] for s in batch]))

        out = {
            "images": all_images,
            "image_masks": all_image_masks,
            "tokens": tokens,
            "token_mask": token_mask,
            "ar_mask": ar_mask,
            "loss_mask": loss_mask,
        }
        if "current_wp_tokens" in batch[0]:
            out["current_wp_tokens"] = torch.from_numpy(
                np.stack([s["current_wp_tokens"] for s in batch])
            )
        return out

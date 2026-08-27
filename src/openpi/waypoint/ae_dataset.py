"""Action Expert training dataset for waypoint VLA.

Reads original RLDS data + waypoint_indices.json to produce waypoint-pair
training samples for the flow-matching Action Expert.

Each sample: (images, instruction, start_proprio, end_proprio, duration,
              padded_actions[horizon, model_dim], action_pad_mask, action_dim_mask)
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import IterableDataset

from openpi.waypoint.normalize import NormalizationHelper
from openpi.waypoint.normalize import extract_proprio_from_obs
from openpi.waypoint.normalize import make_dim_mask
from openpi.waypoint.normalize import pad_to_dim
from openpi.waypoint.robot_config import RobotConfig
from openpi.waypoint.vlm_dataset import _derive_shuffle_seed
from openpi.waypoint.vlm_dataset import _distributed_rank, validate_shard
from openpi.waypoint.vlm_dataset import build_image_augmentation

logger = logging.getLogger(__name__)

MODEL_ACTION_DIM = 32
MODEL_PROPRIO_DIM = 32


class WaypointAEDataset(IterableDataset):
    """RLDS-based Action Expert dataset with global shuffle buffer.

    Iterates over episodes, extracts waypoint pairs, normalizes, pads to
    unified model dimensions, and yields training samples.
    """

    def __init__(
        self,
        original_rlds_dir: str,
        wp_indices_path: str,
        robot_config: RobotConfig,
        dataset_statistics: dict[str, Any],
        norm_type: str = "q99",
        max_duration: int = 32,
        horizon_steps: int = 32,
        model_action_dim: int = MODEL_ACTION_DIM,
        model_proprio_dim: int = MODEL_PROPRIO_DIM,
        shuffle_buffer_size: int = 10000,
        image_size: tuple[int, int] = (224, 224),
        episode_shuffle_buffer: int = 0,
        image_aug: bool = False,
        image_aug_cfg: dict | None = None,
        ae_max_shift: int = 0,
        shuffle_seed: int | None = None,
        shard: tuple[int, int] | None = None,
        max_epochs: int | None = None,
    ):
        """``shard`` / ``max_epochs``: see ``WaypointVLMDataset`` -- explicit
        ``(world_size, rank)`` episode split for non-DDP consumers, and an
        optional cap on the number of passes (``None`` = endless).  Note that
        ``__len__`` still reports the whole (unsharded) sample count."""
        super().__init__()
        self.shard = validate_shard(shard)
        if max_epochs is not None and int(max_epochs) < 1:
            raise ValueError(f"max_epochs must be >= 1 or None, got {max_epochs!r}")
        self.max_epochs = None if max_epochs is None else int(max_epochs)
        self.original_rlds_dir = original_rlds_dir
        self.robot_config = robot_config
        self.norm_type = norm_type
        self.max_duration = max_duration
        self.horizon_steps = horizon_steps
        self.model_action_dim = model_action_dim
        self.model_proprio_dim = model_proprio_dim
        self.shuffle_buffer_size = shuffle_buffer_size
        self.image_size = image_size
        self.episode_shuffle_buffer = episode_shuffle_buffer
        self.ae_max_shift = ae_max_shift
        self.shuffle_seed = shuffle_seed
        self._sample_shuffle_rng: np.random.Generator | None = None

        self.norm_helper = NormalizationHelper(dataset_statistics, norm_type)

        self.image_aug_transform = None
        if image_aug:
            self.image_aug_transform = build_image_augmentation(image_aug_cfg)
            logger.info(f"AE image augmentation enabled: {self.image_aug_transform}")

        # Override action_norm_mask from robot_config if provided (e.g. gripper not normalized)
        if robot_config.action_norm_mask is not None:
            self.norm_helper.action_norm_mask = robot_config.action_norm_mask

        self.action_dim_mask = make_dim_mask(robot_config.actual_action_dim, model_action_dim)
        # AE proprio endpoint = continuous dims + one binary value PER gripper
        # (1 for LIBERO/CALVIN/single-arm, 2 for dual-arm Rokae).
        ae_proprio_dim = robot_config.continuous_proprio_dim + robot_config.num_grippers()
        self.proprio_dim_mask = make_dim_mask(ae_proprio_dim, model_proprio_dim)

        logger.info(f"Loading waypoint indices from {wp_indices_path}")
        with open(wp_indices_path) as f:
            wp_data = json.load(f)

        self.episode_wp_map: dict[int, list[tuple[int, int, int]]] = {}
        self.episode_lengths: dict[int, int] = {}
        total_pairs = 0
        total_augmented = 0
        skipped = 0
        skipped_nonpositive = 0
        # Runtime skip counters.  Every one of these used to be a bare
        # `continue`, i.e. an invisible reduction in dataset coverage.
        self._skipped_unknown_episode = 0
        self._skipped_out_of_range = 0
        self._skipped_length_mismatch = 0
        self._episodes_checked = 0
        self._mismatch_deltas: list[int] = []
        self._yielded_samples = 0
        self.wp_indices_path_for_log = str(wp_indices_path)
        for ep in wp_data["episodes"]:
            ep_idx = ep["src_ep_idx"]
            ep_len = ep.get("original_steps", 0)
            self.episode_lengths[ep_idx] = ep_len
            wp_indices = ep["waypoint_indices"]
            valid_pairs = []
            for i in range(len(wp_indices) - 1):
                dur = wp_indices[i + 1] - wp_indices[i]
                if dur < 1:
                    # A duplicated or non-monotonic index yields dur <= 0, which
                    # produced an empty action slice, an all-True pad mask and a
                    # `duration: 0.0` sample -- a flow-matching target with zero
                    # supervised elements.  Reject it at load time.
                    skipped_nonpositive += 1
                    continue
                if dur <= max_duration:
                    w_start = wp_indices[i]
                    w_end = wp_indices[i + 1]
                    valid_pairs.append((w_start, w_end, dur))
                    if ae_max_shift > 0 and dur >= 2 * ae_max_shift:
                        min_s = max(-ae_max_shift, -w_start)
                        max_s = min(ae_max_shift, ep_len - w_end - 1)
                        total_augmented += max(0, max_s - min_s + 1)
                    elif ae_max_shift > 0:
                        total_augmented += 1
                else:
                    skipped += 1
            if valid_pairs:
                self.episode_wp_map[ep_idx] = valid_pairs
                total_pairs += len(valid_pairs)

        if ae_max_shift > 0:
            self.total_pairs = total_augmented
        else:
            self.total_pairs = total_pairs
        logger.info(
            f"WaypointAEDataset: {len(self.episode_wp_map)} episodes, "
            f"{total_pairs} valid pairs, {skipped} skipped (dur>{max_duration}), "
            f"{skipped_nonpositive} skipped (dur<1), "
            f"ae_max_shift={ae_max_shift} → {self.total_pairs} total samples, "
            f"actual_action={robot_config.actual_action_dim}→{model_action_dim}, "
            f"proprio={robot_config.continuous_proprio_dim}+{robot_config.num_grippers()}grip={ae_proprio_dim}→{model_proprio_dim}, "
            f"episode_shuffle_buffer={episode_shuffle_buffer}, shuffle_seed={shuffle_seed}"
        )

    def __len__(self) -> int:
        return self.total_pairs

    def _process_episode(self, steps, wp_pairs, rc, ep_idx_for_sample=-1):
        """Extract waypoint-pair samples from a single episode's steps.

        When ``ae_max_shift`` > 0, each waypoint pair is augmented by
        shifting the action window by -max_shift .. +max_shift steps
        (bidirectional) while keeping duration constant.  shift=0 is the
        original unaugmented sample.  Invalid shifts (window out of
        episode bounds) are silently skipped.
        """
        num_steps = len(steps)

        all_actions_raw = np.stack([s["action"].numpy() for s in steps])
        all_actions = rc.normalize_gripper(all_actions_raw)
        all_actions = all_actions[:, rc.action_dim_indices]
        all_actions = self.norm_helper.normalize_actions(all_actions)

        n_grip = rc.num_grippers()
        all_proprios_7d = np.empty((num_steps, rc.continuous_proprio_dim + n_grip), dtype=np.float32)
        for t in range(num_steps):
            obs_dict = {k: steps[t]["observation"][k] for k in steps[t]["observation"]}
            raw = extract_proprio_from_obs(obs_dict, rc.state_obs_keys)
            # Plural accessor: every gripper, in gripper_dim_indices order
            # (``split_proprio`` keeps rejecting >1 gripper for single-slot callers).
            cont, grips = rc.split_proprio_grippers(raw)
            cont_norm = self.norm_helper.normalize_proprio(cont)
            all_proprios_7d[t, :rc.continuous_proprio_dim] = cont_norm
            all_proprios_7d[t, rc.continuous_proprio_dim:] = np.asarray(grips, dtype=np.float32)

        instruction = steps[0]["language_instruction"].numpy()
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")

        max_s = self.ae_max_shift
        for w_start, w_end, duration in wp_pairs:
            if max_s > 0 and duration >= 2 * max_s:
                shifts = range(-max_s, max_s + 1)
            else:
                shifts = range(1)
            for shift in shifts:
                new_start = w_start + shift
                new_end = new_start + duration
                if new_start < 0 or new_end >= num_steps:
                    self._skipped_out_of_range += 1
                    continue

                images = {}
                for view, rlds_key in rc.camera_rlds_keys.items():
                    model_key = rc.camera_model_keys[view]
                    img_data = steps[new_start]["observation"][rlds_key].numpy()
                    img = Image.fromarray(img_data)
                    if img.size != self.image_size:
                        img = img.resize(self.image_size, Image.BILINEAR)
                    if self.image_aug_transform is not None:
                        img = self.image_aug_transform(img)
                    images[model_key] = np.array(img, dtype=np.uint8)

                start_proprio = pad_to_dim(all_proprios_7d[new_start], self.model_proprio_dim)
                end_proprio = pad_to_dim(all_proprios_7d[new_end], self.model_proprio_dim)

                seg_actions = all_actions[new_start:new_end]
                actual_len = len(seg_actions)
                padded_actions = np.zeros(
                    (self.horizon_steps, self.model_action_dim), dtype=np.float32
                )
                padded_actions[:actual_len, :rc.actual_action_dim] = seg_actions[:actual_len]

                action_pad_mask = np.zeros(self.horizon_steps, dtype=bool)
                action_pad_mask[actual_len:] = True

                self._yielded_samples += 1
                yield {
                    "images": images,
                    "instruction": instruction,
                    "start_proprio": start_proprio.astype(np.float32),
                    "end_proprio": end_proprio.astype(np.float32),
                    "duration": float(duration),
                    "actions": padded_actions.astype(np.float32),
                    "action_pad_mask": action_pad_mask,
                    "action_dim_mask": self.action_dim_mask.copy(),
                    "proprio_dim_mask": self.proprio_dim_mask.copy(),
                    "_ep_idx": int(ep_idx_for_sample),
                    "_wp_start": int(w_start),
                    "_wp_end": int(w_end),
                    "_shift": int(shift),
                }

    def _raw_sample_iter(self):
        """Yield raw samples from RLDS. DDP-aware, repeats indefinitely.

        When episode_shuffle_buffer > 0, episodes are shuffled at the TF
        dataset level to mix data across task suites.  We use
        ``dataset.enumerate()`` **before** shard/shuffle so that each
        episode carries its original sequential index (== src_ep_idx in
        waypoint_indices.json).  This keeps the ``episode_wp_map`` lookup
        correct regardless of iteration order.
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

        # Outer epoch loop rebuilds the TFDS graph from scratch every pass.
        # This is the structure committed in 4c7fd1c (which works correctly
        # w.r.t. waypoint_indices.json ordering).  The "repeatfix" attempt
        # that replaced this with a single .repeat() call was the source
        # of the accelerated leak in E043 — keeping the interleave reader
        # pool alive forever accumulates state.  Rebuilding per epoch
        # guarantees that pool is torn down and recreated, releasing C++
        # state that .repeat() would otherwise hold forever.
        epoch_count = 0
        episodes_seen = 0
        while self.max_epochs is None or epoch_count < self.max_epochs:
            epoch_count += 1
            builder = tfds.builder_from_directory(self.original_rlds_dir)
            # IMPORTANT: keep interleave_cycle_length=16 — the natural
            # episode order produced by this setting is what
            # waypoint_indices.json's src_ep_idx assumes (4c7fd1c notes:
            # "(cycle=16, block=16) produces BIT-IDENTICAL episode order
            # to the TFDS default").  Lowering it (e.g. cycle=1) broke
            # dataset.enumerate() <-> episode_wp_map alignment in E044
            # (1320 skip-warnings in 23 min vs 0 in E043).
            read_config = tfds.ReadConfig(
                interleave_cycle_length=16,
                interleave_block_length=16,
            )
            dataset = builder.as_dataset(split="train", read_config=read_config)
            # Leak fix: disable tf.data autotune / private threadpool /
            # default optimizations / implicit auto-shard.  We shard
            # manually below and want a strictly minimal background state.
            options = tf.data.Options()
            options.autotune.enabled = False
            options.threading.private_threadpool_size = 1
            options.experimental_optimization.apply_default_optimizations = False
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            dataset = dataset.with_options(options)
            # enumerate() binds the original sequential index BEFORE
            # shard/shuffle so each episode keeps its src_ep_idx key.
            dataset = dataset.enumerate()
            if world_size > 1:
                dataset = dataset.shard(world_size, rank)
            if use_episode_shuffle:
                episode_seed = _derive_shuffle_seed(
                    self.shuffle_seed,
                    rank=rank,
                    epoch=epoch_count,
                    stream="ae_episode",
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
                wp_pairs = self.episode_wp_map.get(ep_idx)
                if wp_pairs is None:
                    self._skipped_unknown_episode += 1
                    continue

                steps = list(episode["steps"])
                if not steps:
                    continue

                # The `src_ep_idx` <-> TFDS-enumerate contract is what binds
                # waypoint_indices.json to the actual frames; if the reader order
                # ever changes, every index in this episode silently points at the
                # wrong frame.  `original_steps` is the one cheap witness we have,
                # so check it instead of trusting it.  (E044 already hit a real
                # TFDS ordering mismatch.)
                declared = self.episode_lengths.get(ep_idx, 0)
                if declared:
                    self._episodes_checked += 1
                    if declared != len(steps):
                        self._skipped_length_mismatch += 1
                        self._mismatch_deltas.append(declared - len(steps))
                        if self._skipped_length_mismatch <= 5:
                            logger.error(
                                "AE dataset: episode %d has %d steps but waypoint_indices.json "
                                "declares original_steps=%d -- src_ep_idx/TFDS order mismatch, "
                                "skipping this episode",
                                ep_idx, len(steps), declared,
                            )
                        # A *systematic* mismatch would otherwise yield a silently
                        # empty dataset with five stray log lines.  Fail fast, and
                        # name the likely cause: a constant delta means the two
                        # sides disagree on the `original_steps` convention, a
                        # varying delta means the src_ep_idx <-> TFDS order broke.
                        # Ratio, not all-or-nothing: a read-order break that leaves a
                        # handful of coincidentally-matching episodes would otherwise
                        # starve the shuffle buffer forever with no error at all.
                        if (
                            self._episodes_checked >= 20
                            and self._skipped_length_mismatch > 0.5 * self._episodes_checked
                        ):
                            deltas = set(self._mismatch_deltas)
                            hint = (
                                f"every episode is off by a constant {deltas.pop()} -- the "
                                "index file and the loader disagree about what "
                                "`original_steps` counts"
                                if len(deltas) == 1
                                else f"the delta varies ({sorted(set(self._mismatch_deltas))[:5]}...) -- "
                                "the src_ep_idx <-> TFDS read order is broken"
                            )
                            raise RuntimeError(
                                f"AE dataset: {self._skipped_length_mismatch} of the "
                                f"{self._episodes_checked} episodes checked so far have a length "
                                f"mismatch against {self.wp_indices_path_for_log}; {hint}. "
                                "Refusing to train on a starved dataset."
                            )
                        continue

                yield from self._process_episode(
                    steps, wp_pairs, rc, ep_idx_for_sample=ep_idx,
                )
                del steps
                episodes_seen += 1
                if episodes_seen % 100 == 0:
                    gc.collect()

            # End of epoch: explicitly tear down the dataset object so the
            # interleave reader pool / iterator state is released before
            # we rebuild on the next iteration of the while loop.
            del dataset, builder
            gc.collect()
            logger.info(
                f"AE dataset epoch {epoch_count} complete (rank={rank}, "
                f"episodes_seen={episodes_seen}, yielded={self._yielded_samples}, "
                f"skipped[unknown_ep={self._skipped_unknown_episode}, "
                f"len_mismatch={self._skipped_length_mismatch}, "
                f"shift_out_of_range={self._skipped_out_of_range}]); "
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
                stream="ae_sample",
            )
            self._sample_shuffle_rng = np.random.default_rng(seed)

        buffer = []
        logged_fill = False
        for sample in self._raw_sample_iter():
            buffer.append(sample)
            if len(buffer) < self.shuffle_buffer_size:
                if not logged_fill and len(buffer) == 1:
                    logger.info(
                        f"AE shuffle buffer filling: target={self.shuffle_buffer_size} samples"
                    )
                continue
            if not logged_fill:
                logger.info(f"AE shuffle buffer full ({len(buffer)} samples), training starts")
                logged_fill = True
            idx = int(self._sample_shuffle_rng.integers(len(buffer)))
            yield buffer.pop(idx)


class WaypointAECollator:
    """Collates AE samples into batched tensors for the model."""

    def __init__(self, tokenizer_max_len: int = 64):
        self.tokenizer_max_len = tokenizer_max_len

        path_obj = None
        try:
            import sentencepiece

            import openpi.shared.download as download
            path_obj = download.maybe_download(
                "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
            )
            with path_obj.open("rb") as f:
                self._pg_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        except Exception:
            self._pg_tokenizer = None

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)

        all_images = {}
        all_image_masks = {}
        for key in batch[0]["images"]:
            imgs = np.stack([s["images"][key] for s in batch])  # (B, H, W, C)
            imgs = imgs.astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
            imgs = imgs.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            all_images[key] = torch.from_numpy(imgs)
            all_image_masks[key] = torch.ones(B, dtype=torch.bool)

        for model_key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            if model_key not in all_images:
                all_images[model_key] = torch.zeros(B, 3, 224, 224, dtype=torch.float32)
                all_image_masks[model_key] = torch.zeros(B, dtype=torch.bool)

        start_proprio = torch.from_numpy(np.stack([s["start_proprio"] for s in batch]))
        end_proprio = torch.from_numpy(np.stack([s["end_proprio"] for s in batch]))
        actions = torch.from_numpy(np.stack([s["actions"] for s in batch]))
        action_pad_mask = torch.from_numpy(np.stack([s["action_pad_mask"] for s in batch]))
        action_dim_mask = torch.from_numpy(np.stack([s["action_dim_mask"] for s in batch]))
        duration = torch.tensor([s["duration"] for s in batch], dtype=torch.float32)

        prompt_tokens = torch.zeros(B, self.tokenizer_max_len, dtype=torch.long)
        prompt_masks = torch.zeros(B, self.tokenizer_max_len, dtype=torch.bool)

        if self._pg_tokenizer is not None:
            for i, s in enumerate(batch):
                text = f"Task: {s['instruction'].strip().replace('_', ' ').lower()}, \n"
                tids = self._pg_tokenizer.encode(text, add_bos=True)
                length = min(len(tids), self.tokenizer_max_len)
                prompt_tokens[i, :length] = torch.tensor(tids[:length], dtype=torch.long)
                prompt_masks[i, :length] = True

        meta_ep_idx = [s.get("_ep_idx", -1) for s in batch]
        meta_wp_start = [s.get("_wp_start", -1) for s in batch]
        meta_wp_end = [s.get("_wp_end", -1) for s in batch]
        meta_shift = [s.get("_shift", 0) for s in batch]

        return {
            "images": all_images,
            "image_masks": all_image_masks,
            "start_proprio": start_proprio,
            "end_proprio": end_proprio,
            "actions": actions,
            "action_pad_mask": action_pad_mask,
            "action_dim_mask": action_dim_mask,
            "duration": duration,
            "prompt_tokens": prompt_tokens,
            "prompt_masks": prompt_masks,
            "_meta_ep_idx": meta_ep_idx,
            "_meta_wp_start": meta_wp_start,
            "_meta_wp_end": meta_wp_end,
            "_meta_shift": meta_shift,
        }

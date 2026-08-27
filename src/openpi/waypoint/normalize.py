"""Normalization utilities for waypoint VLA RLDS data.

Supports q99 (quantile) and normal (z-score) normalization,
with optional per-dimension masks.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def normalize_q99(
    values: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map [q01, q99] -> [-1, 1], clip outliers, zero constant dims.

    Ordering matters: the ``mask=False`` pass-through must be applied *last*.
    With the constant-dimension guard applied afterwards, an excluded dimension
    whose statistics happen to be degenerate (``q01 == q99``, e.g. a gripper that
    is >= 99% in one state across the statistics set) was silently overwritten
    with 0.0 -- which for a 0/1 gripper convention inverts half its labels.
    """
    eps = 1e-8
    normed = np.clip(2 * (values - q01) / (q99 - q01 + eps) - 1, -1, 1)
    degenerate = np.equal(q01, q99)
    normed = np.where(degenerate, 0.0, normed)
    if mask is not None:
        if np.any(degenerate & ~np.asarray(mask, dtype=bool)):
            bad = np.nonzero(degenerate & ~np.asarray(mask, dtype=bool))[0].tolist()
            logger.warning(
                "normalize_q99: dim(s) %s are excluded from normalization but have "
                "q01 == q99; passing raw values through (they are NOT zeroed)",
                bad,
            )
        normed = np.where(mask, normed, values)
    return normed.astype(np.float32)


def unnormalize_q99(
    values: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse of normalize_q99: [-1,1] -> [q01, q99]."""
    values_01 = 0.5 * (values + 1)  # [-1,1] -> [0,1]
    unnormed = values_01 * (q99 - q01) + q01
    if mask is not None:
        unnormed = np.where(mask, unnormed, values)
    return unnormed.astype(np.float32)


def normalize_normal(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Z-score normalization: (x - mean) / (std + eps)."""
    eps = 1e-8
    normed = (values - mean) / (std + eps)
    if mask is not None:
        normed = np.where(mask, normed, values)
    return normed.astype(np.float32)


def unnormalize_normal(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse of z-score normalization."""
    eps = 1e-8
    unnormed = values * (std + eps) + mean
    if mask is not None:
        unnormed = np.where(mask, unnormed, values)
    return unnormed.astype(np.float32)


def pad_to_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad the last dimension of ``arr`` to ``target_dim``.

    Raises ``ValueError`` when the array is already *wider* than the target:
    silently slicing ``[..., :target_dim]`` (the historical behaviour) would
    drop the trailing proprio/gripper dims of a wide robot layout -- e.g. a
    dual-arm 15-D state under a 14-D ``model_proprio_dim`` -- with no signal.
    """
    actual = arr.shape[-1]
    if actual == target_dim:
        return arr
    if actual > target_dim:
        raise ValueError(
            f"pad_to_dim: array width {actual} exceeds target_dim {target_dim}; "
            "refusing to truncate (raise model_proprio_dim / model_action_dim to "
            "at least the robot's proprio width)."
        )
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_dim - actual)]
    return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


def make_dim_mask(actual_dim: int, model_dim: int) -> np.ndarray:
    """Create boolean mask: True for real dims, False for padding."""
    return np.array(
        [True] * actual_dim + [False] * (model_dim - actual_dim),
        dtype=bool,
    )


def extract_proprio_from_obs(step_obs: dict, state_obs_keys: list[str]) -> np.ndarray:
    """Extract and concatenate proprio from an RLDS observation dict."""
    parts = []
    for key in state_obs_keys:
        val = step_obs[key]
        if hasattr(val, "numpy"):
            val = val.numpy()
        val = np.asarray(val, dtype=np.float32).flatten()
        parts.append(val)
    return np.concatenate(parts)


class NormalizationHelper:
    """Wraps normalization statistics and provides normalize/unnormalize methods.

    Supports two stat file layouts:
      1. Flat:   {"action": {"mean":..., "q01":..., ...}, "proprio": {...}}
      2. Nested: {"libero": {"action": {...}, "proprio": {...}}, ...}
    """

    def __init__(self, dataset_statistics: dict[str, Any], norm_type: str = "q99"):
        self.norm_type = norm_type
        stats = self._find_stats(dataset_statistics)

        self.action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
        self.action_std = np.array(stats["action"]["std"], dtype=np.float32)
        self.action_q01 = np.array(stats["action"]["q01"], dtype=np.float32)
        self.action_q99 = np.array(stats["action"]["q99"], dtype=np.float32)
        # Default norm mask: True for all dims (all normalized). Callers can override.
        n_action = len(self.action_mean)
        self.action_norm_mask = np.array(
            stats["action"].get("mask", [True] * n_action),
            dtype=bool,
        )

        self.proprio_mean = np.array(stats["proprio"]["mean"], dtype=np.float32)
        self.proprio_std = np.array(stats["proprio"]["std"], dtype=np.float32)
        self.proprio_q01 = np.array(stats["proprio"]["q01"], dtype=np.float32)
        self.proprio_q99 = np.array(stats["proprio"]["q99"], dtype=np.float32)

    @classmethod
    def for_robot(cls, dataset_statistics: dict[str, Any], norm_type: str, robot_config) -> "NormalizationHelper":
        """Build a helper whose ``action_norm_mask`` follows the robot config.

        Gripper action columns are stored as raw {0, 1} flags and are excluded
        from normalization by ``robot_config.action_norm_mask`` at training
        time (see ``WaypointAEDataset``).  Every consumer that *un*normalizes
        model actions must apply the same mask: with the default all-True mask
        a closed gripper (0.0) maps to 0.5 and rounds to "open" on any positive
        noise (found 2026-08-26 in the offline evaluator and the Rokae wrapper).
        """
        helper = cls(dataset_statistics, norm_type)
        mask = getattr(robot_config, "action_norm_mask", None)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != helper.action_norm_mask.shape:
                raise ValueError(
                    f"robot action_norm_mask has {mask.shape[0]} dims but the statistics "
                    f"describe {helper.action_norm_mask.shape[0]} action dims"
                )
            helper.action_norm_mask = mask
        return helper

    @staticmethod
    def _find_stats(dataset_statistics: dict) -> dict:
        # Layout 1: flat {"action": {...}, "proprio": {...}}
        if "action" in dataset_statistics and "proprio" in dataset_statistics:
            return dataset_statistics
        # Layout 2: nested {"dataset_name": {"action": {...}, "proprio": {...}}}
        for key, val in dataset_statistics.items():
            if key != "__total__" and isinstance(val, dict) and "action" in val:
                return val
        raise ValueError(
            "Cannot find action/proprio stats. Expected either flat "
            '{"action": {...}, "proprio": {...}} or nested {"name": {"action": ...}}'
        )

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.norm_type in ("q99", "bounds_q99"):
            return normalize_q99(actions, self.action_q01, self.action_q99, self.action_norm_mask)
        return normalize_normal(actions, self.action_mean, self.action_std, self.action_norm_mask)

    def unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.norm_type in ("q99", "bounds_q99"):
            return unnormalize_q99(actions, self.action_q01, self.action_q99, self.action_norm_mask)
        return unnormalize_normal(actions, self.action_mean, self.action_std, self.action_norm_mask)

    def normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        if self.norm_type in ("q99", "bounds_q99"):
            return normalize_q99(proprio, self.proprio_q01, self.proprio_q99)
        return normalize_normal(proprio, self.proprio_mean, self.proprio_std)

    def unnormalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        if self.norm_type in ("q99", "bounds_q99"):
            return unnormalize_q99(proprio, self.proprio_q01, self.proprio_q99)
        return unnormalize_normal(proprio, self.proprio_mean, self.proprio_std)


def load_dataset_statistics(path: str | Path) -> dict:
    """Load dataset statistics JSON from a file path."""
    path = Path(path)
    if path.is_dir():
        candidates = list(path.glob("dataset_statistics*.json"))
        if not candidates:
            raise FileNotFoundError(f"No dataset_statistics*.json found in {path}")
        path = candidates[0]
    with open(path) as f:
        return json.load(f)

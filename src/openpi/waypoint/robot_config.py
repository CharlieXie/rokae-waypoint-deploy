"""Robot configuration registry for waypoint VLA.

Defines per-robot configurations for cameras, action/proprio dimensions,
gripper normalization, and RLDS observation key mappings.
Supports LIBERO (single-arm Franka Panda), Galaxea R1 Lite (dual-arm) and
Rokae AR5-5 (dual-arm, real robot).
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Gripper normalization functions
# ---------------------------------------------------------------------------

def normalize_gripper_libero(actions: np.ndarray) -> np.ndarray:
    """LIBERO gripper: raw [-1,1] -> clip [0,1] -> invert (1-x).

    Result: 0=close, 1=open  (matches flow matching training convention).
    """
    actions = actions.copy()
    gripper = actions[:, -1]
    gripper = np.clip(gripper, 0, 1)
    actions[:, -1] = 1.0 - gripper
    return actions


def normalize_gripper_calvin(actions: np.ndarray) -> np.ndarray:
    """CALVIN gripper: raw [-1,+1] (+1=open, -1=close) -> clip [0,1].

    Result: 0=close, 1=open (matches flow matching convention).
    No inversion needed — CALVIN's +1=open already aligns with convention.
    """
    actions = actions.copy()
    gripper = actions[:, -1]
    gripper = np.clip(gripper, 0, 1)
    actions[:, -1] = gripper
    return actions


def normalize_gripper_rokae(actions: np.ndarray) -> np.ndarray:
    """Rokae (bi_rokae_robot) grippers: raw {0.0, 1.0} with 0=closed, 1=open.

    Verified on the shelf dataset (2026-08-25): both gripper columns are strictly
    binary and 6/6 video-inspected transitions show value 1 = fingers open,
    value 0 = fingers closed, i.e. the repository convention already -- so this
    is identity + clip, NO inversion (unlike LIBERO).  Indexes the RAW 30-dim
    action layout (left gripper col 14, right gripper col 29), applied before
    ``action_dim_indices`` selection exactly like ``normalize_gripper_r1_lite``.
    """
    actions = actions.copy()
    for col in ROKAE_GRIPPER_RAW_INDICES:
        actions[:, col] = np.clip(actions[:, col], 0, 1)
    return actions


def normalize_gripper_r1_lite(actions: np.ndarray) -> np.ndarray:
    """R1 Lite gripper: raw [0,100] -> [0,1] for both arms."""
    QPOS_DIM = 6
    actions = actions.copy()
    actions[:, QPOS_DIM] = np.clip(actions[:, QPOS_DIM] / 100.0, 0, 1)
    actions[:, 2 * QPOS_DIM + 1] = np.clip(actions[:, 2 * QPOS_DIM + 1] / 100.0, 0, 1)
    return actions


# ---------------------------------------------------------------------------
# Camera key mappings: canonical view name -> RLDS observation key
# ---------------------------------------------------------------------------

CAMERA_KEY_MAPS: dict[str, dict[str, str]] = {
    "rokae": {
        "head": "external",
        "wrist_left": "left_wrist",
        "wrist_right": "right_wrist",
    },
    "galaxea_r1_lite": {
        "head": "image_camera_head",
        "wrist_left": "image_camera_wrist_left",
        "wrist_right": "image_camera_wrist_right",
    },
    "libero": {
        "primary": "image",
        "wrist": "wrist_image",
    },
    "calvin": {
        "primary": "rgb_static",
        "wrist": "rgb_gripper",
    },
}

# Mapping from RLDS camera names to openpi model image keys
CAMERA_TO_MODEL_KEY: dict[str, dict[str, str]] = {
    "rokae": {
        "head": "base_0_rgb",
        "wrist_left": "left_wrist_0_rgb",
        "wrist_right": "right_wrist_0_rgb",
    },
    "galaxea_r1_lite": {
        "head": "base_0_rgb",
        "wrist_left": "left_wrist_0_rgb",
        "wrist_right": "right_wrist_0_rgb",
    },
    "libero": {
        "primary": "base_0_rgb",
        "wrist": "left_wrist_0_rgb",
    },
    "calvin": {
        "primary": "base_0_rgb",
        "wrist": "left_wrist_0_rgb",
    },
}


# ---------------------------------------------------------------------------
# Robot action/proprio dimension configs
# ---------------------------------------------------------------------------

# Rokae AR5-5 dual arm, LeRobot v3 export "bi_rokae_robot": 30 raw dims, per arm
# [joint_pos0..6, cart_pos0..5 (xyz+rpy), psi, gripper_pos]; left = 0..14, right = 15..29.
# Decision D1 (2026-08-25): the model's action space is joints + grippers only
# (16 dims); cart pose and psi are redundant re-encodings of the joints in this
# data (psi is even a near-bitwise copy of joint 2) and are never fed to the model.
ROKAE_RAW_DIM = 30
ROKAE_LEFT_JOINTS = list(range(0, 7))
ROKAE_RIGHT_JOINTS = list(range(15, 22))
ROKAE_LEFT_GRIPPER = 14
ROKAE_RIGHT_GRIPPER = 29
ROKAE_GRIPPER_RAW_INDICES = (ROKAE_LEFT_GRIPPER, ROKAE_RIGHT_GRIPPER)

ROBOT_ACTION_DIM_CONFIGS: dict[str, dict[str, list[int]]] = {
    "rokae": {
        "with_left_arm": ROKAE_LEFT_JOINTS + [ROKAE_LEFT_GRIPPER],
        "with_right_arm": ROKAE_RIGHT_JOINTS + [ROKAE_RIGHT_GRIPPER],
    },
    "galaxea_r1_lite": {
        "with_left_arm": list(range(0, 7)),
        "with_right_arm": list(range(7, 14)),
        "with_torso": list(range(14, 20)),
        "with_chassis": list(range(20, 26)),
    },
    "libero": {
        "with_arm": list(range(0, 7)),
    },
    "calvin": {
        "with_arm": list(range(0, 7)),
    },
}

# State observation keys in RLDS
ROBOT_STATE_KEYS: dict[str, dict[str, list[str]]] = {
    # The Rokae RLDS carries the full 30-dim state in one key regardless of which
    # arms are enabled; arm selection happens through the index lists below.
    "rokae": {
        "with_left_arm": ["state"],
        "with_right_arm": [],
    },
    "galaxea_r1_lite": {
        "with_left_arm": ["joint_position_arm_left", "gripper_state_left"],
        "with_right_arm": ["joint_position_arm_right", "gripper_state_right"],
        "with_torso": ["joint_position_torso"],
        "with_chassis": ["base_velocity"],
    },
    "libero": {
        "with_arm": ["state"],
    },
    "calvin": {
        "with_arm": ["state"],
    },
}


def get_action_dims(robot_type: str, robot_cfg: dict[str, bool]) -> list[int]:
    """Get filtered action dimension indices for a robot configuration."""
    if robot_type not in ROBOT_ACTION_DIM_CONFIGS:
        raise ValueError(f"Unknown robot type: {robot_type}")
    config = ROBOT_ACTION_DIM_CONFIGS[robot_type]
    return [i for key, idxs in config.items() if robot_cfg.get(key, True) for i in idxs]


def get_state_obs_keys(robot_type: str, robot_cfg: dict[str, bool]) -> list[str]:
    """Get state observation keys from RLDS for a robot configuration."""
    if robot_type not in ROBOT_STATE_KEYS:
        raise ValueError(f"Unknown robot type: {robot_type}")
    config = ROBOT_STATE_KEYS[robot_type]
    return [v for key, vs in config.items() if robot_cfg.get(key, True) for v in vs]


# ---------------------------------------------------------------------------
# RobotConfig dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobotConfig:
    """Complete configuration for a robot type in waypoint VLA."""
    robot_type: str

    # Actual dimensions before padding
    actual_action_dim: int
    actual_proprio_dim: int  # total raw state dims (e.g. 8 for LIBERO)

    # Continuous proprio dims (excluding gripper) for VLM tokenization
    continuous_proprio_dim: int = 6

    # Which raw-state indices are the continuous dims.  ``None`` means "the first
    # ``continuous_proprio_dim``", which is only correct when the gripper sits
    # *after* every continuous dim (true for LIBERO/CALVIN).  A dual-arm layout
    # such as [larm(6), lgrip, rarm(6), rgrip] interleaves them, so it must list
    # the indices explicitly or the second arm silently never reaches the planner.
    continuous_proprio_indices: list[int] | None = None

    # Gripper binarization settings for VLM waypoint tokens
    gripper_dim_index: int = 6  # index in raw state for gripper binarization
    gripper_threshold: float = 0.02  # > threshold = open
    # All gripper indices in the raw state.  The waypoint token layout currently
    # has exactly ONE gripper slot per waypoint, so a config with more than one
    # gripper must be rejected loudly rather than quietly dropping the others.
    gripper_dim_indices: list[int] | None = None

    # Indices into the raw RLDS action array
    action_dim_indices: list[int] = field(default_factory=list)

    # Observation keys for extracting proprio from RLDS
    state_obs_keys: list[str] = field(default_factory=list)

    # Camera views and their RLDS keys
    camera_views: list[str] = field(default_factory=list)
    camera_rlds_keys: dict[str, str] = field(default_factory=dict)
    camera_model_keys: dict[str, str] = field(default_factory=dict)

    # Gripper normalization function (for action processing)
    normalize_gripper: Callable[[np.ndarray], np.ndarray] = field(default=lambda x: x)

    # Normalization mask for actions (False = skip normalization, e.g. gripper)
    action_norm_mask: np.ndarray | None = None

    def binarize_gripper(self, state: np.ndarray) -> int:
        """Binarize gripper from raw state: 1=open, 0=close."""
        return int(state[self.gripper_dim_index] > self.gripper_threshold)

    def binarize_grippers(self, state: np.ndarray) -> list[int]:
        """Binarize every gripper in the raw state (1=open, 0=close)."""
        indices = self.gripper_dim_indices or [self.gripper_dim_index]
        return [int(state[i] > self.gripper_threshold) for i in indices]

    def num_grippers(self) -> int:
        return len(self.gripper_dim_indices or [self.gripper_dim_index])

    def select_continuous(self, state: np.ndarray) -> np.ndarray:
        """Pick the continuous proprio dims out of a raw state (or a [T, D] batch).

        The single place that knows whether the continuous dims are a contiguous
        prefix (LIBERO/CALVIN) or an explicit index list (a dual-arm layout that
        interleaves grippers).  Every consumer must go through this instead of
        slicing ``[:continuous_proprio_dim]``, or an interleaved layout silently
        normalizes a gripper value with a joint's quantiles.
        """
        arr = np.asarray(state)
        if self.continuous_proprio_indices is not None:
            return arr[..., self.continuous_proprio_indices].copy()
        return arr[..., : self.continuous_proprio_dim].copy()

    def split_proprio(self, state: np.ndarray) -> tuple[np.ndarray, int]:
        """Split raw state into continuous proprio and binary gripper.

        Returns:
            (continuous_proprio, gripper_binary): the continuous dims selected by
            ``continuous_proprio_indices`` (default: the leading
            ``continuous_proprio_dim`` dims), and a single 0/1 gripper.

        Raises:
            NotImplementedError: if the robot has more than one gripper.  The
                waypoint token layout has exactly one gripper slot per waypoint,
                so returning just the first gripper would silently drop the
                second arm's contact state from every waypoint.
        """
        if self.num_grippers() > 1:
            raise NotImplementedError(
                f"{self.robot_type} declares {self.num_grippers()} grippers "
                f"({self.gripper_dim_indices}), but the waypoint token layout has one "
                "gripper slot per waypoint, so one arm's contact state would be dropped "
                "from every waypoint. Either run single-arm (config: "
                "`robot_config_kwargs: {with_right_arm: false}`) or extend "
                "WaypointTokenizer to N gripper slots (tokens_per_waypoint, "
                "gripper_pos_in_wp, family_for_slot, decode_waypoints, both datasets and "
                "the block width all move together)."
            )
        continuous = self.select_continuous(state)
        gripper = self.binarize_gripper(state)
        return continuous, gripper

    def split_proprio_grippers(self, state: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Plural sibling of ``split_proprio``: continuous dims + EVERY gripper.

        Returns ``(continuous_proprio, [g_1, .., g_N])`` with one 0/1 per gripper
        in ``gripper_dim_indices`` order.  This is the accessor multi-gripper
        code paths (``n_gripper_slots > 1``) must use; ``split_proprio`` keeps its
        single-gripper contract and its loud rejection of dual-gripper configs so
        that every remaining single-slot consumer still fails closed.
        """
        return self.select_continuous(state), self.binarize_grippers(state)


def make_libero_config() -> RobotConfig:
    action_dims = list(range(7))
    # Gripper (dim 6) is already in [0,1] after normalize_gripper_libero → skip normalization
    norm_mask = np.array([True] * 6 + [False], dtype=bool)
    return RobotConfig(
        robot_type="libero",
        actual_action_dim=7,
        actual_proprio_dim=8,
        continuous_proprio_dim=6,  # dims 0-5 (EEF xyz + rot)
        gripper_dim_index=6,  # grip_qpos_L
        gripper_threshold=0.02,  # > 0.02 → open
        action_dim_indices=action_dims,
        state_obs_keys=["state"],
        camera_views=["primary", "wrist"],
        camera_rlds_keys=CAMERA_KEY_MAPS["libero"],
        camera_model_keys=CAMERA_TO_MODEL_KEY["libero"],
        normalize_gripper=normalize_gripper_libero,
        action_norm_mask=norm_mask,
    )


def make_calvin_config() -> RobotConfig:
    action_dims = list(range(7))
    # Gripper (dim 6) is already in [0,1] after normalize_gripper_calvin → skip normalization
    norm_mask = np.array([True] * 6 + [False], dtype=bool)
    return RobotConfig(
        robot_type="calvin",
        actual_action_dim=7,
        actual_proprio_dim=15,  # CALVIN robot_obs is 15D
        continuous_proprio_dim=6,  # dims 0-5 (TCP pos + euler)
        gripper_dim_index=6,  # gripper width (meters, 0~0.08)
        gripper_threshold=0.04,  # > 0.04m → open
        action_dim_indices=action_dims,
        state_obs_keys=["state"],
        camera_views=["primary", "wrist"],
        camera_rlds_keys=CAMERA_KEY_MAPS["calvin"],
        camera_model_keys=CAMERA_TO_MODEL_KEY["calvin"],
        normalize_gripper=normalize_gripper_calvin,
        action_norm_mask=norm_mask,
    )


def make_r1_lite_config(
    with_left_arm: bool = True,
    with_right_arm: bool = True,
    with_torso: bool = False,
    with_chassis: bool = False,
) -> RobotConfig:
    robot_cfg = {
        "with_left_arm": with_left_arm,
        "with_right_arm": with_right_arm,
        "with_torso": with_torso,
        "with_chassis": with_chassis,
    }
    action_dims = get_action_dims("galaxea_r1_lite", robot_cfg)
    state_keys = get_state_obs_keys("galaxea_r1_lite", robot_cfg)

    actual_action_dim = len(action_dims)
    gripper_indices = []
    if with_left_arm:
        gripper_indices.append(6)
    if with_right_arm:
        gripper_indices.append(13 if with_left_arm else 6)
    norm_mask = np.ones(actual_action_dim, dtype=bool)
    for gi in gripper_indices:
        if gi < actual_action_dim:
            norm_mask[gi] = False

    # The dual-arm state interleaves grippers with continuous dims, so both the
    # continuous index list and the gripper index list must be explicit: with the
    # dataclass defaults (continuous_proprio_dim=6, gripper_dim_index=6) a 14-dim
    # dual-arm state put ONLY the left arm into the waypoint and binarized the
    # left gripper as if it were a continuous dim.
    continuous_indices = [i for i in range(actual_action_dim) if i not in gripper_indices]

    return RobotConfig(
        robot_type="galaxea_r1_lite",
        actual_action_dim=actual_action_dim,
        actual_proprio_dim=actual_action_dim,
        continuous_proprio_dim=len(continuous_indices),
        continuous_proprio_indices=continuous_indices,
        gripper_dim_index=gripper_indices[0] if gripper_indices else 6,
        gripper_dim_indices=gripper_indices or None,
        gripper_threshold=0.02,
        action_dim_indices=action_dims,
        state_obs_keys=state_keys,
        camera_views=["head", "wrist_left", "wrist_right"],
        camera_rlds_keys=CAMERA_KEY_MAPS["galaxea_r1_lite"],
        camera_model_keys=CAMERA_TO_MODEL_KEY["galaxea_r1_lite"],
        normalize_gripper=normalize_gripper_r1_lite,
        action_norm_mask=norm_mask,
    )


def make_rokae_config(
    with_left_arm: bool = True,
    with_right_arm: bool = True,
) -> RobotConfig:
    """Rokae AR5-5 dual arm (LeRobot v3 ``bi_rokae_robot`` export converted to RLDS).

    Three index frames, kept apart on purpose (the R1-Lite comment above
    documents the bug class):

    * raw STATE / raw ACTION frame (30 dims, identical layout): used by
      ``continuous_proprio_indices``, ``gripper_dim_indices``,
      ``gripper_dim_index`` (state) and ``normalize_gripper`` (action);
    * filtered ACTION frame (``action_dim_indices`` applied, 16 dims dual-arm):
      used by ``action_norm_mask``.

    Dual-arm: continuous proprio = 14 joints ([0:7] + [15:22]), grippers at raw
    14 and 29, actions = joints + grippers = [0..6, 14, 15..21, 29].  Single-arm
    variants (``with_right_arm=False`` etc.) keep the raw 30-dim state and only
    shrink the index lists, so they run through the one-gripper code paths.
    """
    if not (with_left_arm or with_right_arm):
        raise ValueError("rokae: at least one arm must be enabled")
    robot_cfg = {"with_left_arm": with_left_arm, "with_right_arm": with_right_arm}
    action_dims = get_action_dims("rokae", robot_cfg)
    raw_grippers = []
    if with_left_arm:
        raw_grippers.append(ROKAE_LEFT_GRIPPER)
    if with_right_arm:
        raw_grippers.append(ROKAE_RIGHT_GRIPPER)
    continuous_indices = []
    if with_left_arm:
        continuous_indices += ROKAE_LEFT_JOINTS
    if with_right_arm:
        continuous_indices += ROKAE_RIGHT_JOINTS
    # action_norm_mask lives in the FILTERED action frame: False exactly at the
    # positions of the gripper raw columns after ``action_dim_indices`` selection.
    norm_mask = np.array([raw not in raw_grippers for raw in action_dims], dtype=bool)
    assert (~norm_mask).sum() == len(raw_grippers)
    return RobotConfig(
        robot_type="rokae",
        actual_action_dim=len(action_dims),
        actual_proprio_dim=ROKAE_RAW_DIM,
        continuous_proprio_dim=len(continuous_indices),
        continuous_proprio_indices=continuous_indices,
        gripper_dim_index=raw_grippers[0],
        gripper_dim_indices=raw_grippers,
        gripper_threshold=0.5,  # values are exactly {0, 1}; 1=open
        action_dim_indices=action_dims,
        state_obs_keys=["state"],
        camera_views=["head", "wrist_left", "wrist_right"],
        camera_rlds_keys=CAMERA_KEY_MAPS["rokae"],
        camera_model_keys=CAMERA_TO_MODEL_KEY["rokae"],
        normalize_gripper=normalize_gripper_rokae,
        action_norm_mask=norm_mask,
    )


def get_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    """Factory function to get a robot configuration."""
    if robot_type == "libero":
        return make_libero_config()
    elif robot_type == "calvin":
        return make_calvin_config()
    elif robot_type == "galaxea_r1_lite":
        return make_r1_lite_config(**kwargs)
    elif robot_type == "rokae":
        return make_rokae_config(**kwargs)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

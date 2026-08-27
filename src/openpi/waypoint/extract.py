"""Waypoint extraction (AWE) and waypoint-index migration -- pure-numpy core.

The scripts that touch RLDS live in ``scripts/``; everything here is importable
and unit-testable without TensorFlow, a GPU, or a dataset.

Provenance
----------
This is a port of the generator that actually produced the existing
``waypoint_indices.json`` files.  That generator lives in the open-source AWE
project (arXiv 2307.14326), not in this repository:

    waypoint_extraction/extract_waypoints_fast.py   (the DP)
    example/rlds_wp_extract.py                      (the RLDS driver)

That code is itself a numba-accelerated reimplementation of AWE's
``dp_waypoint_selection(..., pos_only=True)`` (arXiv 2307.14326,
``waypoint_extraction/extract_waypoints.py`` upstream).  ``dp_waypoint_selection``
below reproduces it **bit-for-bit** when ``max_duration is None``.  The parity
test (``tests/waypoint/test_extract.py``) asserts this against frozen reference
output in ``tests/waypoint/fixtures/``, so it runs without an AWE checkout; set
``AWE_REF_PATH`` to also re-verify the fixture against live AWE code.

The three things AWE actually specifies, and which an earlier version of this
file got wrong:

1. **The error is a point-to-line-SEGMENT projection distance**, not the
   distance to the point at the same *time fraction* along the chord.  Upstream:
   ``traj_reconstruction.point_line_distance`` clips ``t`` to ``[0, 1]`` and
   projects.  A trajectory that is perfectly collinear but traversed at a
   non-uniform speed has zero AWE error and must collapse to its endpoints; a
   time-aligned metric splits it.  (Note that AWE measures *rotation* by slerp at
   the matching time fraction -- that half IS time-aligned.  This module works on
   whatever coordinates the caller passes and uses the position metric for all of
   them, which is what the reference driver does for both LIBERO EEF pose and
   R1-Lite joint positions.)
2. **The DP minimises the number of waypoints subject to a hard error bound**,
   with the recurrence ``count[i] = 1 + count[k-1]`` over every ``k`` whose
   chord ``traj[k] -> traj[i]`` reconstructs frames ``k..i-1`` within ``eta``.
   Note the asymmetry: the chord starts at ``k`` while the *previous waypoint* is
   ``k-1``.  That off-by-one is in upstream AWE and in the reference; it is
   reproduced here deliberately, because the existing supervision was generated
   with it and "matches the shipped data" beats "is tidier".
3. The comparison is a strict ``<``, and ``eta <= 0`` means "every frame".

Two deliberate additions on top of the reference:

``max_duration``
    Segments longer than the Action Expert's action horizon cannot be executed:
    the planner would emit a duration the AE never trained on, and evaluation
    would clamp it.  Bounding segment length *inside the DP* makes the duration
    contract hold by construction.  The reference has no such cap, which is
    exactly why ``scripts/migrate_waypoint_indices.py`` had to exist.

``mandatory indices``
    First frame, last frame, and every gripper transition.  A gripper flip is a
    discrete contact event: interpolating across it is meaningless, and it is the
    one place where the "sparse waypoints" abstraction is known to be too weak.
    ``anchor_mode`` chooses how they are combined with the DP -- see
    ``extract_waypoints``.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------------


def segment_error(traj: np.ndarray, start: int, end: int, weights: np.ndarray | None = None) -> float:
    """Max point-to-**segment** distance over the frames the chord replaces.

    ``traj`` is ``[T, D]``.  Reproduces
    ``awe/waypoint_extraction/extract_waypoints_fast.py::_precompute_err_matrix``:
    the maximum, over frames ``start .. end - 1``, of the distance from that frame
    to its orthogonal projection onto the segment ``traj[start] -> traj[end]``
    (the projection parameter is clipped to ``[0, 1]``, so the distance is to the
    segment, not to the infinite line).

    Frame ``start`` itself contributes exactly 0 and is included only so the
    index range matches the reference.  Returns 0.0 when there is nothing between
    the endpoints.

    ``weights`` scales each coordinate before the distance is taken, which is a
    local extension (the reference has no weighting); it is applied to the
    difference vector, so ``weights=None`` is the reference behaviour.
    """
    if end - start < 2:
        return 0.0
    a = traj[start].astype(np.float64)
    b = traj[end].astype(np.float64)
    pts = traj[start:end].astype(np.float64)  # frames start .. end-1, as in the reference
    line = b - a
    denom = float(line @ line)
    if denom < 1e-30:
        delta = pts - a[None]
    else:
        t = ((pts - a[None]) @ line) / denom
        np.clip(t, 0.0, 1.0, out=t)
        delta = pts - (a[None] + t[:, None] * line[None])
    if weights is not None:
        delta = delta * np.asarray(weights, dtype=np.float64)[None, :]
    return float(np.max(np.linalg.norm(delta, axis=1)))


def _error_matrix(traj: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """``err[i, j]`` = ``segment_error(traj, i, j)``; 0 where ``j < i + 2``."""
    n = traj.shape[0]
    err = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 2, n):
            err[i, j] = segment_error(traj, i, j, weights)
    return err


# ---------------------------------------------------------------------------
# The AWE DP
# ---------------------------------------------------------------------------


def dp_waypoint_selection(
    traj: np.ndarray,
    eta: float,
    max_duration: int | None = None,
    weights: np.ndarray | None = None,
) -> list[int]:
    """AWE ``pos_only`` waypoint selection; returns indices **excluding** frame 0.

    Faithful to ``awe/waypoint_extraction/extract_waypoints_fast.py::
    dp_waypoint_selection_fast`` when ``max_duration is None``:

    * the last frame is always a waypoint;
    * ``eta <= 0`` returns every frame except 0;
    * feasibility is ``err[k, i] < eta`` (strict);
    * ties are broken toward the *smallest* ``k``, because the scan is ascending
      and the update test is strict.

    ``max_duration`` restricts ``k`` to ``k >= i - max_duration + 1``, i.e. it
    bounds the gap between the previous waypoint ``k - 1`` and ``i``.  Frame
    ``i - 1`` is always in range (its chord error is 0 by construction), so the
    DP always terminates with a finite cost for ``max_duration >= 1``.
    """
    traj = np.asarray(traj, dtype=np.float64)
    n = traj.shape[0]
    if n <= 1:
        return list(range(n))
    if eta <= 0:
        return list(range(1, n))
    if max_duration is not None and max_duration < 1:
        raise ValueError(f"max_duration must be >= 1, got {max_duration}")

    err = _error_matrix(traj, weights)

    inf = float("inf")
    count = [inf] * n
    came_from = [-1] * n
    count[0] = 0
    count[1] = 1
    came_from[1] = 1

    for i in range(2, n):
        k_lo = 1 if max_duration is None else max(1, i - max_duration + 1)
        for k in range(k_lo, i):
            if err[k, i] < eta:
                cand = 1 + count[k - 1]
                if cand < count[i]:
                    count[i] = cand
                    came_from[i] = k
                    if count[i] == 1:
                        break  # cannot do better than a single waypoint

    waypoints: list[int] = []
    i = n - 1
    while i > 0 and came_from[i] >= 0:
        waypoints.append(i)
        i = came_from[i] - 1
    waypoints.reverse()
    if n - 1 not in waypoints:
        waypoints.append(n - 1)  # the reference always forces the last frame
    return sorted(set(waypoints))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ExtractionResult:
    indices: list[int]
    num_frames: int
    max_error: float
    gaps: list[int]

    @property
    def compression(self) -> float:
        return self.num_frames / max(len(self.indices), 1)


def extract_waypoints(
    traj: np.ndarray,
    eta: float,
    max_duration: int | None = 32,
    mandatory: list[int] | None = None,
    weights: np.ndarray | None = None,
    anchor_mode: str = "union",
) -> ExtractionResult:
    """AWE waypoint extraction with an optional segment-length cap and keyframes.

    Args:
        traj: ``[T, D]`` trajectory in the space the tolerance is expressed in.
        eta: reconstruction tolerance (same units as ``traj``), compared with
            strict ``<`` exactly as the reference does.
        max_duration: hard cap on ``index[k+1] - index[k]``.  ``None`` reproduces
            the reference (no cap), which is what the shipped index files used.
        mandatory: extra frame indices that must be waypoints (gripper flips).
            The first and last frames are always included.
        weights: optional per-dimension weights for the distance (local
            extension; ``None`` is the reference behaviour).
        anchor_mode:
            ``"union"`` (default, **reference-faithful**) -- run one DP over the
            whole trajectory and union the mandatory frames in afterwards.  This
            is what produced the existing ``waypoint_indices.json``.

            Caveat, and it is a real one: a chord the DP accepted may span a
            mandatory frame, and the two sub-segments created by inserting it
            are then **never error-checked**.  Their error is *not* bounded by
            the parent's -- shortening a chord can move a frame's projection
            past the new endpoint, where the clipped distance is measured to
            that endpoint instead.  Concretely, for
            ``[(0,0), (5,0.1), (1,0), (10,0)]`` (an overshoot-and-return, with
            the anchor at index 2) the parent chord ``0->3`` has error 0.100
            while the child chord ``0->2`` has error 4.001 -- 40x worse.
            ``ExtractionResult.max_error`` is measured on the final index set,
            so it does report this; it just is not bounded by ``eta``.
            Use ``"split"`` if you need every executed chord checked.
            ``"split"`` -- treat the mandatory frames as hard boundaries and run
            the DP independently between consecutive anchors.  Produces a
            different (generally larger) index set; do not mix the two in one
            dataset.

    Returns:
        ``ExtractionResult`` with strictly increasing indices in ``[0, T-1]``,
        every gap in ``[1, max_duration]`` when a cap is given, and the achieved
        max error measured on the *final* index set.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"traj must be [T, D], got {traj.shape}")
    n_frames = traj.shape[0]
    if n_frames < 2:
        raise ValueError(f"need at least 2 frames, got {n_frames}")
    if max_duration is not None and max_duration < 1:
        raise ValueError(f"max_duration must be >= 1, got {max_duration}")
    if anchor_mode not in ("union", "split"):
        raise ValueError(f"anchor_mode must be 'union' or 'split', got {anchor_mode!r}")

    anchors = {0, n_frames - 1}
    for idx in mandatory or []:
        if not 0 <= idx <= n_frames - 1:
            raise ValueError(f"mandatory index {idx} out of range [0, {n_frames - 1}]")
        anchors.add(int(idx))

    if anchor_mode == "union":
        selected = dp_waypoint_selection(traj, eta, max_duration, weights)
        indices = sorted(anchors | set(selected))
    else:
        anchor_list = sorted(anchors)
        indices = [anchor_list[0]]
        for lo, hi in itertools.pairwise(anchor_list):
            sub = dp_waypoint_selection(traj[lo : hi + 1], eta, max_duration, weights)
            indices.extend(lo + s for s in sub if 0 < s < hi - lo)
            indices.append(hi)
        indices = sorted(set(indices))

    # Unioning anchors can only shrink gaps, so this is a no-op whenever the DP
    # ran with a cap.  It still matters for `max_duration=None` (the migration
    # path) and as a belt-and-braces guarantee for the returned contract.
    if max_duration is not None:
        indices = split_long_gaps(indices, max_duration)

    gaps = [b - a for a, b in itertools.pairwise(indices)]
    max_error = max(
        (segment_error(traj, a, b, weights) for a, b in itertools.pairwise(indices)),
        default=0.0,
    )
    return ExtractionResult(
        indices=indices, num_frames=n_frames, max_error=max_error, gaps=gaps
    )


# ---------------------------------------------------------------------------
# Gripper keyframes
# ---------------------------------------------------------------------------


def detect_gripper_changes(
    signal: np.ndarray,
    columns: list[int] | None = None,
    atol: float = 1.0,
) -> list[int]:
    """Frames where a *raw* gripper command changes, one anchor per event.

    Port of ``awe/example/rlds_wp_extract.py::detect_gripper_changes``, which is
    what tagged the contact events in the shipped index files.  ``signal`` is
    ``[T, D]`` (typically the RLDS ``action`` array); ``columns`` selects the
    gripper channels (LIBERO ``[6]``, binary +-1; R1-Lite ``[6, 13]``, 0..100).

    Consecutive changing frames are one event, and only the **first frame
    carrying the new value** (``i + 1``) is returned -- deliberately not both
    sides of the flip, so a slow 0..100 ramp does not become a run of waypoints.
    """
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    cols = list(range(arr.shape[1])) if columns is None else list(columns)
    out: set[int] = set()
    for c in cols:
        col = arr[:, c]
        if col.size < 2:
            continue
        changing = np.abs(np.diff(col)) > atol
        prev = False
        for i, now in enumerate(changing.tolist()):
            if now and not prev:
                out.add(i + 1)
            prev = now
    return sorted(out)


def gripper_transition_frames(gripper: np.ndarray, both_sides: bool = True) -> list[int]:
    """Frames where a **binarized** gripper signal changes.

    ``both_sides=True`` (default) keeps ``c`` and ``c + 1`` so the planner can
    express "arrive here, *then* close".  ``both_sides=False`` keeps only
    ``c + 1``, matching ``detect_gripper_changes`` and the shipped index files.

    For a binary signal the run-grouping in ``detect_gripper_changes`` is a
    no-op, so ``gripper_transition_frames(g, both_sides=False)`` and
    ``detect_gripper_changes(g, atol=0.5)`` agree.
    """
    g = np.asarray(gripper).astype(int).reshape(-1)
    changes = np.nonzero(g[1:] != g[:-1])[0]
    out: set[int] = set()
    for c in changes.tolist():
        if both_sides:
            out.add(int(c))
        out.add(int(c + 1))
    return sorted(out)


# ---------------------------------------------------------------------------
# Migration of an existing index set
# ---------------------------------------------------------------------------


def split_long_gaps(indices: list[int], max_duration: int) -> list[int]:
    """Insert **real** intermediate frame indices until every gap <= max_duration.

    Used by the ``max_duration=None`` extraction path and by the v1 -> v2
    migration.  Preserves every original index (so no waypoint the DP chose is
    lost) and never fabricates a synthetic state -- the inserted indices name
    actual frames of the original trajectory.
    """
    if max_duration < 1:
        raise ValueError(f"max_duration must be >= 1, got {max_duration}")
    if not indices:
        return []
    out = [int(indices[0])]
    for raw_end in indices[1:]:
        end = int(raw_end)
        if end <= out[-1]:
            raise ValueError(f"indices must be strictly increasing: {out[-1]} -> {end}")
        while end - out[-1] > max_duration:
            out.append(out[-1] + max_duration)
        if end != out[-1]:
            out.append(end)
    return out


@dataclasses.dataclass
class ContractViolation:
    episode: int
    kind: str
    detail: str


def validate_indices(
    indices: list[int],
    original_steps: int,
    max_duration: int,
    episode: int = -1,
) -> list[ContractViolation]:
    """Check the duration/index contract for one episode.

    The five things that must hold for both loaders to agree:
    strictly increasing, in range, at least two entries, every gap in
    ``[1, max_duration]``, and the last index inside the episode.
    """
    out: list[ContractViolation] = []
    if len(indices) < 2:
        out.append(ContractViolation(episode, "too_short", f"{len(indices)} indices"))
        return out
    if indices[0] < 0 or indices[-1] > original_steps - 1:
        out.append(
            ContractViolation(
                episode, "out_of_range",
                f"[{indices[0]}, {indices[-1]}] not inside [0, {original_steps - 1}]",
            )
        )
    for a, b in itertools.pairwise(indices):
        if b <= a:
            out.append(ContractViolation(episode, "not_increasing", f"{a} -> {b}"))
        elif b - a > max_duration:
            out.append(ContractViolation(episode, "gap_too_long", f"{a} -> {b} = {b - a}"))
    if len(set(indices)) != len(indices):
        out.append(ContractViolation(episode, "duplicate_index", "indices are not unique"))
    return out


def gap_histogram(gaps: list[int]) -> dict[str, float]:
    if not gaps:
        return {}
    arr = np.asarray(gaps, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }

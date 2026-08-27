"""Small distributed-loss helpers used by waypoint training.

PyTorch DDP averages gradients across ranks.  If every rank first divides its
loss by a different, data-dependent count, that average is an average of local
means rather than the mean over the pooled global batch.  ``global_mean_loss``
keeps the forward value and the gradient equal to the pooled global mean.

Only detached scalar statistics participate in the explicit collective.  The
autograd path stays local and is scaled by ``world_size`` to cancel DDP's
subsequent gradient average.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def distributed_world_size() -> int:
    """Return the active default process-group size, or one without DDP."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def global_mean_loss(local_sum: torch.Tensor, local_count: torch.Tensor | int) -> torch.Tensor:
    """Return a pooled global mean with the correct DDP gradient.

    Let rank ``r`` own differentiable numerator ``S_r`` and non-differentiable
    count ``C_r``.  The desired gradient is::

        sum_r grad(S_r) / sum_r C_r

    DDP divides reduced parameter gradients by ``W``.  Each rank therefore
    backpropagates through ``W * S_r / sum(C_r)``.  For logging, a detached
    all-reduce also computes ``sum(S_r) / sum(C_r)``; the zero-value surrogate
    term below gives that exact forward value while preserving the required
    local gradient.

    With no active multi-rank process group this is the historical local
    ``sum / count.clamp(min=1)`` formula.
    """
    if local_sum.ndim != 0:
        raise ValueError(f"local_sum must be scalar, got shape {tuple(local_sum.shape)}")

    count = torch.as_tensor(local_count, device=local_sum.device, dtype=local_sum.dtype)
    if count.numel() != 1:
        raise ValueError(f"local_count must be scalar, got shape {tuple(count.shape)}")
    count = count.reshape(())

    world_size = distributed_world_size()
    if world_size == 1:
        return local_sum / count.clamp(min=1)

    # Do not all-reduce an autograd-carrying numerator.  c10d's in-place
    # collective is not the gradient path we want here, and an autograd-aware
    # collective would be reduced a second time by DDP.
    global_stats = torch.stack((local_sum.detach(), count.detach()))
    dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
    global_sum, global_count = global_stats.unbind()
    denominator = global_count.clamp(min=1)

    backward_surrogate = local_sum * (world_size / denominator)
    global_mean = global_sum / denominator
    # Forward: global_mean.  Backward: d(backward_surrogate).
    return global_mean + (backward_surrogate - backward_surrogate.detach())


def require_unweighted_global_mean(
    family_weights: dict[str, float] | None,
    *,
    context: str,
) -> None:
    """Fail closed for grouped means not implemented by ``global_mean_loss``.

    A weighted sum of per-family means needs one global numerator and count per
    family.  Treating it as one global token mean would silently change its
    objective, so multi-rank training rejects that configuration for now.
    """
    if family_weights is not None and distributed_world_size() > 1:
        raise RuntimeError(
            f"{context}: planner_family_weights is not DDP-safe yet; it requires "
            "separate global numerator/count reductions for every token family. "
            "Use planner_family_weights: null for multi-rank training."
        )

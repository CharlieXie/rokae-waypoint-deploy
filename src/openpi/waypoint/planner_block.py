"""Waypoint-block autoregression: one whole waypoint per decoder call.

Motivation
----------
Token-AR spends one 2B-decoder call per *scalar* (6 pose bins, gripper,
duration) plus one per delimiter, so a 7-waypoint plan costs ~70 sequential
calls.  The semantic ordering constraint, however, lives *between* waypoints,
not between the fields of one waypoint: FASTer (arXiv 2512.04952) keeps AR
between blocks while predicting a whole block per forward and reports LIBERO
decode depth dropping from 21 calls to 3.

Evidence boundaries, so this is not oversold:

* FASTer's block-size guidance is **not** a clean formula.  ``3_vla.tex`` only
  defines ``N = J*B``; ``appendix.tex`` says B=8 for WBC/bimanual and B=7 for
  single-arm in prose, while its implementation table lists B=7 for *all* eight
  suites (including rows whose action-code width is 14 and 21), and the
  ablation remark ("strongest when the block size matches the action-dimension
  code length") was measured on yet another configuration.  One waypoint per
  block is chosen here because it is the natural semantic unit of *this*
  interface, not because a paper prescribes it.
* FASTer's BAR is not uniformly better: on LIBERO-Spatial its no-BAR variant
  scores higher, the rollout count behind the success numbers is not disclosed,
  and the headline latency figures come with hardware but no protocol.  Treat
  block AR as a latency change that must clear a *no-regression* gate, not as a
  free success-rate win.
* Block-wise decoding is **not** a free decode-time trick: FASTer trains with
  the block mask, so a token-AR checkpoint cannot be block-decoded.  Budget a
  training run.  (``planner_decode.generate_waypoints_jacobi`` is the
  no-retraining alternative.)

Mechanism
---------
The repository already contains the required attention primitive: openpi's
``make_att_2d_masks`` documents ``[[1 0 1 0 1 0 0 1 0 0]]`` as "causal attention
between 4 blocks -- tokens of a block can attend all previous blocks and all
tokens on the same block".  Token-AR sets ``ar_mask = 1`` on *every* waypoint
token; block-AR sets it to 1 only on each block's first slot.  Nothing else
about the mask machinery changes, and the *same* dataset samples are used.

    token_ar   ar = [1 1 1 1 1 1 1 1 1 1 | 1 1 ...]   strict causal
    block_ar   ar = [1 0 0 0 0 0 0 0 0 0 | 1 0 ...]   block causal

Two things have to move together (the invariant that G0.5's ``ar_helper.py``
enforces with special training labels and that we enforce by sharing one mask
builder between training and inference):

1. **Inputs are shifted by one block.**  Block ``j``'s slots are fed the token
   embeddings of block ``j-1``; block 0 is fed a learned query block.
2. **Labels are not shifted.**  The logits at block ``j``'s slots predict block
   ``j``'s own tokens -- *not* ``tokens[:, 1:]``.

Because block ``j``'s input only contains waypoint ``j-1``, there is no target
leakage even though attention inside the block is bidirectional.

Deliberate scope limits
-----------------------
* ``block_width == tokens_per_waypoint`` always.  The structural ``<wp>`` /
  ``<dur>`` slots are kept so the token layout, the dataset, the tokenizer and
  legacy checkpoints stay untouched; a 10-token forward costs essentially the
  same wall clock as a 1-token forward at this model size.
* The terminal signal stays "duration == 0", exactly as in token-AR, so the two
  arms are trained on byte-identical samples and the architecture is the only
  changed variable.  Splitting termination out into an explicit ``<stop>`` token
  is a *data-schema* change and is tracked separately.
* ActionCodec (arXiv 2602.15397) does *not* license the intra-block layout
  here.  An earlier revision of this docstring cited it as warning that
  mutually dependent fields must not be predicted as independent
  classifications; the paper's evidence runs the other way -- it credits
  *token independence* for parallel decoding matching AR and advocates a
  decoupled, independent tokenization.  Predicting one waypoint's fields as
  conditionally independent classifications (the intra-block bidirectional
  attention is over the *previous* block's embeddings, not over each other's
  outputs) is this project's own trade-off with no external paper backing;
  the no-regression gate above is the actual arbiter.  Retraction recorded
  in ``docs/10-waypoint-v2.md`` section 2, "证据边界".
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from openpi.waypoint.planner_decode import DURATION_FAMILY
from openpi.waypoint.planner_decode import NEG_INF
from openpi.waypoint.planner_decode import DecodeStats
from openpi.waypoint.planner_decode import WaypointDecodeConfig
from openpi.waypoint.planner_decode import _finalize_stats
from openpi.waypoint.planner_decode import _SectionTimer
from openpi.waypoint.planner_decode import compute_family_logits
from openpi.waypoint.planner_decode import decode_forward
from openpi.waypoint.planner_decode import embed_planner_prefix
from openpi.waypoint.planner_decode import prefill_planner_prefix
from openpi.waypoint.planner_decode import sample_from_family
from openpi.waypoint.tokenizer import PROPRIO_FAMILY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learned block-planner parameters
# ---------------------------------------------------------------------------


class WaypointBlockPlanner(nn.Module):
    """Query / slot / block embeddings for waypoint-block autoregression.

    The block-0 input follows the two published precedents, which agree on the
    input path: FASTer replicates its ``<BoBlk>`` control token ``B`` times to
    start the first block (``3_vla.tex``), and ActionCodec's released BAR variant
    expands one learned ``nn.Parameter`` BOS vector ``B`` times
    (``smolvla/bar.py``).  Here the base is the live ``sqrt(W)``-scaled ``<wp>``
    embedding (FASTer's choice, and already in-distribution for this checkpoint)
    plus a zero-initialised learned residual (ActionCodec's degree of freedom).

    Scale discipline (PaliGemma multiplies token embeddings by ``sqrt(W)``):

    * the query block's *base* is a live ``sqrt(W)``-scaled embedding of the
      ``<wp>`` token, so it starts in-distribution and follows the backbone (and
      its LoRA adapters) during training;
    * ``query_resid``, ``slot_embed`` and ``block_embed`` are zero-initialised
      residuals added *after* the ``sqrt(W)`` scaling -- never multiplied by a
      second ``sqrt(W)``.

    At initialisation the block-0 input is therefore exactly the (scaled)
    ``<wp>`` embedding repeated across slots, and every later block's input is
    exactly the token embedding the token-AR model would have seen.
    """

    def __init__(
        self,
        hidden_width: int,
        block_width: int,
        max_blocks: int,
        *,
        use_block_embed: bool = True,
    ):
        super().__init__()
        self.hidden_width = hidden_width
        self.block_width = block_width
        self.max_blocks = max_blocks
        self.query_resid = nn.Parameter(torch.zeros(block_width, hidden_width))
        self.slot_embed = nn.Embedding(block_width, hidden_width)
        nn.init.zeros_(self.slot_embed.weight)
        if use_block_embed:
            self.block_embed = nn.Embedding(max_blocks, hidden_width)
            nn.init.zeros_(self.block_embed.weight)
        else:
            self.block_embed = None

    # -- helpers -----------------------------------------------------------
    def _slot_ids(self, device) -> torch.Tensor:
        return torch.arange(self.block_width, device=device)

    def decorate(self, block_embs: torch.Tensor, block_index: torch.Tensor | int) -> torch.Tensor:
        """Add slot (and optional block-index) residuals to ``[..., K, W]`` inputs."""
        device = block_embs.device
        out = block_embs + self.slot_embed(self._slot_ids(device)).to(block_embs.dtype)
        if self.block_embed is not None:
            idx = (
                torch.tensor(block_index, device=device)
                if isinstance(block_index, int)
                else block_index.to(device)
            )
            resid = self.block_embed(idx.clamp(max=self.max_blocks - 1)).to(block_embs.dtype)
            out = out + resid.unsqueeze(-2)
        return out

    def query_block(
        self,
        embed_layer,
        wp_token_id: int,
        batch_size: int,
        device,
        cond_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Block-0 input ``[B, K, W]``.

        ``cond_tokens`` is the ``planner_block0_cond: current_state`` option: a
        ``[B, K]`` long tensor holding the *current* state encoded as one
        waypoint block (``WaypointTokenizer.encode_current_state_block``).  When
        given, block 0's base becomes those token embeddings instead of ``<wp>``
        repeated across the slots, so block 0 is conditioned through the same
        token families and slot positions as every later block.  The learned
        ``query_resid`` / ``slot_embed`` / ``block_embed[0]`` residuals are still
        added on top, exactly as in the unconditioned path.

        ``None`` (the default) reproduces the historical constant query block
        bit-for-bit.
        """
        if cond_tokens is None:
            base = embed_layer(torch.tensor([int(wp_token_id)], device=device))
            base = base * math.sqrt(base.shape[-1])
            block = base.expand(self.block_width, -1) + self.query_resid.to(base.dtype)
            block = self.decorate(block, 0)
            return block[None].expand(batch_size, -1, -1)

        if cond_tokens.dim() != 2 or cond_tokens.shape[1] != self.block_width:
            raise ValueError(
                f"block-0 condition must be [B, {self.block_width}], got "
                f"{tuple(cond_tokens.shape)}"
            )
        if cond_tokens.shape[0] != batch_size:
            raise ValueError(
                f"block-0 condition batch {cond_tokens.shape[0]} != batch {batch_size}"
            )
        base = embed_layer(cond_tokens.to(device))
        base = base * math.sqrt(base.shape[-1])
        block = base + self.query_resid.to(base.dtype)[None]
        return self.decorate(block, 0)


# ---------------------------------------------------------------------------
# Locating the waypoint region inside a tokenized sample
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BlockLayout:
    """Where the ``M x tokens_per_waypoint`` plan region sits in each row."""

    start: torch.Tensor  # [B] long, index of the first <wp>
    index: torch.Tensor  # [B, M, K] long, absolute token indices
    num_blocks: int
    block_width: int

    @property
    def flat_index(self) -> torch.Tensor:
        return self.index.reshape(self.index.shape[0], -1)


def locate_block_layout(
    tokens: torch.Tensor,
    wp_token_id: int,
    dur_token_id: int,
    num_blocks: int,
    block_width: int,
    dur_delim_pos: int,
) -> BlockLayout:
    """Find the plan region and verify the delimiter grid.

    The prefix text length varies per sample (the instruction is part of it), so
    the region start is per-row.  ``<wp>`` cannot occur in text or in any value
    family, which makes "first ``<wp>``" an unambiguous anchor.
    """
    if tokens.dim() != 2:
        raise ValueError(f"tokens must be [B, L], got {tuple(tokens.shape)}")
    b, length = tokens.shape
    is_wp = tokens == int(wp_token_id)
    if not bool(is_wp.any(dim=1).all()):
        missing = (~is_wp.any(dim=1)).nonzero().flatten().tolist()
        raise ValueError(f"no <wp> token found in rows {missing}; is this a waypoint VLM batch?")
    start = torch.argmax(is_wp.int(), dim=1)
    if int(start.max()) + num_blocks * block_width > length:
        raise ValueError(
            f"plan region overruns the token buffer: max start {int(start.max())} + "
            f"{num_blocks}x{block_width} > {length}; increase vlm_max_token_len"
        )
    offsets = (
        torch.arange(num_blocks, device=tokens.device)[:, None] * block_width
        + torch.arange(block_width, device=tokens.device)[None, :]
    )
    index = start[:, None, None] + offsets[None]
    region = tokens.gather(1, index.reshape(b, -1)).view(b, num_blocks, block_width)
    if not bool((region[:, :, 0] == int(wp_token_id)).all()):
        raise ValueError("waypoint block grid misaligned: slot 0 is not <wp> in every block")
    if not bool((region[:, :, dur_delim_pos] == int(dur_token_id)).all()):
        raise ValueError("waypoint block grid misaligned: <dur> delimiter not at the expected slot")
    return BlockLayout(start=start, index=index, num_blocks=num_blocks, block_width=block_width)


def build_block_ar_mask(
    ar_mask: torch.Tensor,
    layout: BlockLayout,
) -> torch.Tensor:
    """Turn a strict-causal waypoint ``ar_mask`` into a block-causal one.

    Only the plan region changes: ``1`` on each block's first slot, ``0`` on the
    rest.  Feeding the result to ``make_att_2d_masks`` yields "attend all
    previous blocks + bidirectional inside the current block", which is exactly
    the visibility the block decoder uses at inference time.
    """
    out = ar_mask.clone()
    pattern = torch.zeros(layout.block_width, dtype=out.dtype, device=out.device)
    pattern[0] = 1
    pattern = pattern.repeat(layout.num_blocks)[None].expand(out.shape[0], -1)
    return out.scatter(1, layout.flat_index, pattern)


def jitter_condition_proprio_tokens(
    region_ids: torch.Tensor,
    *,
    block_width: int,
    proprio_dim: int,
    prob: float,
    max_bins: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly shift the *conditioning* proprio bins of a plan region.

    ``region_ids`` is ``[B, num_blocks * block_width]`` -- the plan region's
    token ids.  Only the ``proprio_dim`` value slots of each block are touched;
    delimiters, the gripper slot and the duration slot are left alone.  The
    returned tensor is used **only** to build block ``j+1``'s *input*
    embeddings; targets are read from the untouched ``tokens`` buffer, so no
    label moves.

    Why this is the right invariance: ``docs/12-libero10-root-cause.md`` F2
    shows that shifting a conditioning waypoint by one bin (1/150 of the
    normalised range) collapses family-constrained top-1 on the *following*
    blocks from 1.000 to 0.105 -- the planner has learned an exact-token lookup.
    A plan is a sequence of demo targets; being one bin off in where the arm
    currently is should not change which targets come next, so the training
    signal wanted here is local *invariance*, and jittering the condition while
    holding the target fixed is exactly that signal.  Whether it improves
    rollout is an open hypothesis, not an established fix.

    Proprio token ids are ``PROPRIO_BASE - bin`` (descending in the semantic
    value), so a ``+delta`` bin shift is a ``-delta`` id shift; results are
    clamped into the family so a jittered id is always a legal proprio token.
    """
    if prob <= 0.0 or max_bins <= 0:
        return region_ids
    if proprio_dim <= 0:
        return region_ids

    b, total = region_ids.shape
    if total % block_width:
        raise ValueError(f"region width {total} is not a multiple of block_width {block_width}")
    num_blocks = total // block_width
    device = region_ids.device

    slot = torch.arange(block_width, device=device)
    is_proprio_slot = (slot >= 1) & (slot <= proprio_dim)
    slot_mask = is_proprio_slot.repeat(num_blocks)[None].expand(b, -1)

    draw = torch.rand(region_ids.shape, device=device, generator=generator)
    hit = slot_mask & (draw < float(prob))

    # Uniform non-zero offset in [-max_bins, max_bins] \ {0}.
    magnitude = torch.randint(
        1, int(max_bins) + 1, region_ids.shape, device=device, generator=generator
    )
    sign = torch.randint(0, 2, region_ids.shape, device=device, generator=generator) * 2 - 1
    delta = magnitude * sign

    # token = BASE - bin, so shifting the bin by +delta shifts the id by -delta.
    jittered = torch.where(hit, region_ids - delta, region_ids)
    return torch.where(
        slot_mask,
        jittered.clamp(PROPRIO_FAMILY.lo, PROPRIO_FAMILY.hi),
        region_ids,
    )


def build_block_shifted_embeddings(
    token_embs: torch.Tensor,
    layout: BlockLayout,
    planner: WaypointBlockPlanner,
    embed_layer,
    wp_token_id: int,
    *,
    block0_cond_tokens: torch.Tensor | None = None,
    condition_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace the plan region's inputs with block-shifted inputs.

    ``block j <- embeddings(block j-1)``, ``block 0 <- learned query block``.
    Everything outside the region (images, prefix text, header, footer, padding)
    is untouched.

    ``block0_cond_tokens`` ``[B, K]`` feeds block 0 a real ``waypoint -1`` block
    (see ``WaypointBlockPlanner.query_block``).  ``condition_token_ids``
    ``[B, num_blocks * K]`` overrides the ids whose embeddings become the
    *inputs* of the following blocks -- used for conditioning jitter.  Both
    default to ``None``, which reproduces the historical path exactly.
    """
    b, _, width = token_embs.shape
    flat = layout.flat_index
    gather_idx = flat[..., None].expand(b, flat.shape[1], width)
    if condition_token_ids is None:
        region = token_embs.gather(1, gather_idx)
    else:
        if condition_token_ids.shape != flat.shape:
            raise ValueError(
                f"condition_token_ids must be {tuple(flat.shape)}, got "
                f"{tuple(condition_token_ids.shape)}"
            )
        region = embed_layer(condition_token_ids) * math.sqrt(width)
        region = region.to(token_embs.dtype)
    region = region.view(b, layout.num_blocks, layout.block_width, width)

    query = planner.query_block(
        embed_layer, wp_token_id, b, token_embs.device, cond_tokens=block0_cond_tokens
    )
    query = query.to(region.dtype)[:, None]  # [B, 1, K, W], already carries block-0 residuals
    if layout.num_blocks > 1:
        block_ids = torch.arange(1, layout.num_blocks, device=token_embs.device)
        tail = planner.decorate(region[:, :-1], block_ids[None].expand(b, -1))
        shifted = torch.cat([query, tail], dim=1)
    else:
        shifted = query

    return token_embs.scatter(1, gather_idx, shifted.reshape(b, -1, width))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def block_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    family_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-entropy at block positions.

    ``family_weights`` switches from one global mean over every supervised
    logit (which implicitly weights pose 6x gripper) to a per-family weighted
    sum of per-family means.  Default ``None`` keeps the token-AR reduction so
    that block-vs-token comparisons change only the architecture.

    Returns ``(loss, per_token_loss, targets_used)``.
    """
    active_logits = logits[mask]
    active_targets = targets[mask]
    if active_targets.numel() == 0:
        zero = logits.sum() * 0.0
        empty = torch.empty(0, device=logits.device)
        return zero, empty, active_targets
    per_token = F.cross_entropy(active_logits, active_targets, reduction="none")
    if family_weights is None:
        return per_token.mean(), per_token, active_targets

    from openpi.waypoint.tokenizer import GRIPPER_FAMILY
    from openpi.waypoint.tokenizer import PROPRIO_FAMILY

    families = {
        "proprio": PROPRIO_FAMILY,
        "gripper": GRIPPER_FAMILY,
        "duration": DURATION_FAMILY,
    }
    total = per_token.sum() * 0.0
    used = torch.zeros_like(active_targets, dtype=torch.bool)
    for name, family in families.items():
        sel = (active_targets >= family.lo) & (active_targets <= family.hi)
        used |= sel
        if not bool(sel.any()):
            continue
        total = total + float(family_weights.get(name, 1.0)) * per_token[sel].mean()
    if bool((~used).any()):
        total = total + float(family_weights.get("other", 1.0)) * per_token[~used].mean()
    return total, per_token, active_targets


# ---------------------------------------------------------------------------
# Block inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_waypoints_block(
    model,
    images: dict[str, torch.Tensor],
    image_masks: dict[str, torch.Tensor],
    prompt_tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    wp_tokenizer,
    cfg: WaypointDecodeConfig,
    generator: torch.Generator | None = None,
    block0_cond_tokens: torch.Tensor | None = None,
) -> tuple[list[list[tuple]], DecodeStats]:
    """Block-wise AR planner inference: one decoder call per waypoint.

    Sequential 2B-decoder depth is ``1 + M`` (prefill + one call per waypoint)
    instead of ``len("Action: ") + M * tokens_per_waypoint``.

    Requires a checkpoint trained with ``planner_mode="block_ar"``; running it on
    a token-AR checkpoint is meaningless (the query block and the block-causal
    mask were never seen during training).

    Unlike G0.5's released BAR (``ar_helper.py`` loops ``for i in range(bsz)``
    into a per-sample state machine, because its mask builder only accepts
    batch-wide scalar ``kv_len``/``is_action_block``), this decodes the whole
    batch jointly: ``build_decode_mask`` takes a per-row ``new_key_valid``, so
    rows that have already stopped keep their KV slots invalid without
    degrading the other rows' intra-block visibility.
    """
    planner: WaypointBlockPlanner | None = getattr(model, "block_planner", None)
    if planner is None:
        raise RuntimeError(
            "block decoding requires model.block_planner; construct the model with "
            "planner_mode='block_ar'"
        )

    stats = DecodeStats(impl="block")
    wall0 = time.perf_counter()
    timer = _SectionTimer(cfg.profile, prompt_tokens.device)

    paligemma = model.paligemma_with_expert.paligemma
    embed_layer = paligemma.language_model.embed_tokens
    device = prompt_tokens.device
    batch = prompt_tokens.shape[0]
    tpw = wp_tokenizer.tokens_per_waypoint
    if planner.block_width != tpw:
        raise RuntimeError(
            f"block_planner.block_width={planner.block_width} != tokens_per_waypoint={tpw}; "
            "checkpoint and tokenizer layout disagree"
        )
    num_waypoints = wp_tokenizer.num_waypoints
    schedule = wp_tokenizer.slot_schedule()
    forced_slots: dict[int, int] = {}
    family_slots: dict[object, list[int]] = {}
    for k in range(tpw):
        forced_tok = wp_tokenizer.forced_token_for_slot(k)
        if forced_tok is not None:
            forced_slots[k] = int(forced_tok)
        else:
            family_slots.setdefault(schedule[k], []).append(k)

    # --- prefill: images + prompt (bidirectional) + "Action: " (causal) ---
    prefix_embs, prefix_pad = embed_planner_prefix(
        model, images, image_masks, prompt_tokens, prompt_mask, timer
    )
    header_ids = list(wp_tokenizer._pg_tokenizer.encode("Action: "))  # noqa: SLF001
    header = torch.tensor(header_ids, dtype=torch.long, device=device)[None].expand(batch, -1)
    header_embs = embed_layer(header) * math.sqrt(prefix_embs.shape[-1])
    prefix_embs = torch.cat([prefix_embs, header_embs.to(prefix_embs.dtype)], dim=1)
    prefix_pad = torch.cat(
        [prefix_pad, torch.ones(batch, len(header_ids), dtype=torch.bool, device=device)], dim=1
    )
    header_ar = torch.cat(
        [
            torch.zeros(batch, prefix_pad.shape[1] - len(header_ids), dtype=torch.int32, device=device),
            torch.ones(batch, len(header_ids), dtype=torch.int32, device=device),
        ],
        dim=1,
    )
    cache, _ = prefill_planner_prefix(
        model,
        prefix_embs,
        prefix_pad,
        respect_prefix_pad=True,  # new path: no legacy visibility to preserve
        timer=timer,
        prefix_ar=header_ar,
        stats=stats,
    )

    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    plans_tokens: list[list[int]] = [[] for _ in range(batch)]
    if block0_cond_tokens is not None:
        block0_cond_tokens = block0_cond_tokens.to(device=device, dtype=torch.long)
    block_input = planner.query_block(
        embed_layer,
        wp_tokenizer.wp_token_id,
        batch,
        device,
        cond_tokens=block0_cond_tokens,
    )
    block_input = block_input.to(prefix_embs.dtype)

    for block_idx in range(num_waypoints):
        active = ~finished
        hidden = decode_forward(
            model,
            cache,
            token_ids=None,
            new_key_valid=active[:, None].expand(batch, tpw).contiguous(),
            intra="full",
            inputs_embeds=block_input,
            timer=timer,
            stats=stats,
        )
        with timer.gpu("decode_lm_head_ms"):
            block_tokens = torch.zeros(batch, tpw, dtype=torch.long, device=device)
            for k, forced in forced_slots.items():
                block_tokens[:, k] = forced
            stats.generated_forced_tokens += len(forced_slots) * int(active.sum())
            # One projection per family, not per slot: the six pose slots share a
            # single [B*6, W] x [300, W]^T GEMM.
            for family, slots in family_slots.items():
                idx = torch.as_tensor(slots, device=device)
                logits = compute_family_logits(
                    embed_layer, hidden.index_select(1, idx).float(), family
                )
                if (
                    cfg.forbid_terminal_first_waypoint
                    and block_idx == 0
                    and family is DURATION_FAMILY
                ):
                    logits = logits.clone()
                    logits[..., family.count - 1] = NEG_INF  # duration 0 == terminal
                stats.lm_head_invocation_count += 1
                block_tokens[:, idx] = sample_from_family(
                    logits, family, cfg.temperature, generator
                )
                stats.generated_value_tokens += len(slots) * int(active.sum())

        with timer.cpu("decode_python_ms"):
            rows = block_tokens.tolist()
            for b in range(batch):
                if bool(active[b]):
                    plans_tokens[b].extend(int(t) for t in rows[b])

        durations = (DURATION_FAMILY.hi - block_tokens[:, wp_tokenizer.duration_pos_in_wp]).clamp(min=0)
        if cfg.early_stop:
            finished = finished | (active & (durations == 0))
            if bool(finished.all()):
                break
        if block_idx == num_waypoints - 1:
            break

        next_embs = embed_layer(block_tokens) * math.sqrt(hidden.shape[-1])
        block_input = planner.decorate(next_embs, block_idx + 1).to(prefix_embs.dtype)

    plans = [wp_tokenizer.decode_waypoints(seq) for seq in plans_tokens]
    stats.plan_len = max((len(p) for p in plans), default=0)
    _finalize_stats(stats, timer, wall0, device)
    return plans, stats

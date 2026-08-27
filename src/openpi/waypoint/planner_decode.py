"""Waypoint-planner decoding: shared machinery + the compact and Jacobi decoders.

The legacy planner (``PI0WaypointJoint.generate_waypoints``) emits one token per
2B-decoder call and projects every hidden state onto the full 257 152-row
PaliGemma vocabulary before masking it down to 300 / 2 / 34 legal candidates.
For the standard LIBERO/CALVIN layout (M=7 waypoints x 10 tokens) that is

    one prefill plus one decoder call per emitted token, with a full-vocabulary
    LM head on each; 14 of those positions are deterministic delimiters and the
    very last LM head is never consumed.

    The exact counts are pinned by ``tests/waypoint/test_compact_decode.py``
    (``transformer_forward_count == 1 + header + m * tpw`` and
    ``lm_head_invocation_count == transformer_forward_count``) -- read them there
    rather than from prose, because ``len("Action: ")`` depends on the PaliGemma
    tokenizer, which is fetched at runtime.

This module keeps the model, the tokenizer and the constrained-decoding
semantics exactly as they are and removes the three redundancies that need no
retraining:

``compact_logits``
    project onto one contiguous *token family* (``narrow`` view of the
    embedding matrix) instead of the full vocabulary.  Algebraically identical
    to the surviving entries of the legacy masked full-vocabulary logits.

``pack_forced_tokens``
    ``Action:``/``<wp>``/``<dur>`` are known before they are generated, so they
    are fed to the transformer together with their neighbour in a single
    multi-token causal forward instead of costing one decoder call each.  Also
    drops the legacy trailing forward + LM head whose output is discarded.

``early_stop``
    stop as soon as a duration decodes to 0.  The executor already ignores
    every waypoint after the first zero duration, so this is
    *execution-prefix* equivalent -- but it does change the raw returned plan
    length, which is why it is a separate flag.

``respect_prefix_pad``
    the legacy cached-decode step passes an all-zero (fully visible) attention
    mask, which lets generated tokens attend padded / missing-camera prefix KV
    slots.  Fixing that is a **behaviour change**, not an optimization, so it is
    gated separately and defaults to the legacy behaviour.

Everything here separates two coordinates that the legacy loop conflates:

    ``logical_len``    per-sample count of *valid* tokens -> RoPE position ids
    ``physical_len``   KV storage cursor, including padded prefix slots

They differ whenever the prefix is right-padded, a camera is missing, or a
batch row has already stopped.

References for the design (see docs/10-waypoint-v2.md for the full
evidence table): FASTer (arXiv 2512.04952) for block-wise decode depth,
G0.5 (arXiv 2608.11739) ``ar_helper.py`` for the block state machine, and
ActionCodec (arXiv 2602.15397) for the released block-AR reference
implementation.  (An earlier revision cited ActionCodec for "why
intra-waypoint fields must not be predicted independently" -- the paper
argues the opposite direction; retraction in docs/10-waypoint-v2.md
section 2, "证据边界".)
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import time

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.waypoint.tokenizer import DURATION_FAMILY
from openpi.waypoint.tokenizer import GRIPPER_FAMILY
from openpi.waypoint.tokenizer import PROPRIO_FAMILY
from openpi.waypoint.tokenizer import TokenFamily

logger = logging.getLogger(__name__)

# Same additive -inf constant the rest of the repo uses (bf16-safe).
NEG_INF = -2.3819763e38

DECODE_IMPLS = ("legacy", "compact", "block", "jacobi")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WaypointDecodeConfig:
    """Planner decoding knobs.

    The defaults reproduce the legacy behaviour bit-for-bit so that turning the
    feature on is always an explicit, separately measurable decision.
    """

    impl: str = "legacy"  # legacy | compact | block | jacobi
    compact_logits: bool = True  # family logits instead of full vocab (compact/block/jacobi)
    pack_forced_tokens: bool = True  # fold <wp>/<dur>/header into neighbouring forwards
    early_stop: bool = True  # stop at the first duration == 0
    respect_prefix_pad: bool = False  # False = legacy key visibility (see module docstring)
    # Constrained decoding currently allows duration 0 at *every* slot, including
    # the first waypoint.  A d1 == 0 plan executes zero actions and the evaluator
    # simply replans from the same observation, i.e. a possible no-progress loop.
    # Forbidding it is a behaviour change, hence a flag.
    forbid_terminal_first_waypoint: bool = False
    temperature: float = 0.0
    jacobi_max_iters: int = 0  # 0 -> tokens_per_plan (upper bound, always terminates)
    profile: bool = False

    @classmethod
    def from_dict(cls, cfg: dict | None) -> WaypointDecodeConfig:
        """Build from a (possibly nested) eval config mapping.

        Accepts both a flat ``waypoint_decode_*`` naming and a nested
        ``waypoint_decode: {...}`` block so existing YAMLs stay readable.
        """
        if not cfg:
            return cls()
        nested = cfg.get("waypoint_decode")
        if isinstance(nested, dict):
            src = dict(nested)
        else:
            src = {}
            prefix = "waypoint_decode_"
            for key, value in cfg.items():
                if key.startswith(prefix):
                    src[key[len(prefix) :]] = value
            if "waypoint_decode_impl" in cfg:
                src["impl"] = cfg["waypoint_decode_impl"]
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(src) - known
        if unknown:
            raise ValueError(f"Unknown waypoint_decode option(s): {sorted(unknown)}")
        out = cls(**{k: v for k, v in src.items() if k in known})
        if out.impl not in DECODE_IMPLS:
            raise ValueError(f"waypoint_decode.impl must be one of {DECODE_IMPLS}, got {out.impl!r}")
        return out


@dataclasses.dataclass
class DecodeStats:
    """Per-planning-call instrumentation.

    GPU sections are measured with CUDA events (recorded without
    synchronising, drained once at the end), Python bookkeeping with
    ``perf_counter``.  GPU and CPU numbers are reported separately and must not
    be added: they are different clocks over overlapping intervals.
    """

    impl: str = "legacy"
    planner_total_ms: float = 0.0  # wall clock, always measured
    image_encode_ms: float = 0.0
    prefix_prefill_ms: float = 0.0
    decode_transformer_ms: float = 0.0
    decode_lm_head_ms: float = 0.0
    decode_python_ms: float = 0.0
    transformer_forward_count: int = 0
    lm_head_invocation_count: int = 0
    generated_value_tokens: int = 0
    generated_forced_tokens: int = 0
    unused_final_logit_count: int = 0
    jacobi_iters: int = 0
    # Tri-state, jacobi only.  True  -> an iteration produced no change, so the
    # returned plan is a verified fixed point (== the greedy AR output).
    # None  -> jacobi_max_iters cut the loop short before any iteration could
    #          confirm a fixed point; the plan MAY still be the greedy output.
    # Never False: "not verified" is not the same claim as "verified wrong", and
    # the old `jacobi_converged=False` asserted the latter (see below).
    fixed_point_verified: bool | None = None
    plan_len: int = 0
    peak_allocated_mb: float = 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return dataclasses.asdict(self)


class _SectionTimer:
    """CUDA-event / perf_counter section timer that is a no-op when disabled."""

    def __init__(self, enabled: bool, device: torch.device | None = None):  # noqa: FBT001
        self.enabled = bool(enabled)
        self.cuda = self.enabled and torch.cuda.is_available() and (device is None or device.type == "cuda")
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.cpu_totals: dict[str, float] = {}

    @contextlib.contextmanager
    def gpu(self, name: str):
        if not self.cuda:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._events.append((name, start, end))

    @contextlib.contextmanager
    def cpu(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.cpu_totals[name] = self.cpu_totals.get(name, 0.0) + (time.perf_counter_ns() - t0) / 1e6

    def drain(self) -> dict[str, float]:
        out = dict(self.cpu_totals)
        if self.cuda and self._events:
            torch.cuda.synchronize()
            for name, start, end in self._events:
                out[name] = out.get(name, 0.0) + start.elapsed_time(end)
            self._events.clear()
        return out


# ---------------------------------------------------------------------------
# Family logits
# ---------------------------------------------------------------------------


def _embed_weight(embed_layer) -> torch.Tensor:
    base = getattr(embed_layer, "base_layer", embed_layer)
    return base.weight


def compute_family_logits(embed_layer, hidden: torch.Tensor, family: TokenFamily) -> torch.Tensor:
    """Logits over ``family``'s contiguous vocabulary rows only.

    Algebraically equal to ``compute_full_logits(hidden)[..., family.lo : family.hi + 1]``
    (bf16 GEMM kernel selection can still change the last bits, which is why the
    equivalence tests assert argmax equality plus an fp32 tolerance rather than
    bitwise identity).

    Supports both a plain ``nn.Embedding`` and ``LoRAEmbedding``; for the latter
    only the ``A`` factor is vocabulary-indexed, so the LoRA correction narrows
    exactly like the base weight.
    """
    # Delegate to the adapter's own slice implementation when it has one, so the
    # LoRA branch can never drift from `compute_logits`.
    if hasattr(embed_layer, "compute_logits_slice"):
        return embed_layer.compute_logits_slice(hidden, family.lo, family.count)
    weight = _embed_weight(embed_layer).narrow(0, family.lo, family.count)
    out_dtype = hidden.dtype
    h = hidden.to(weight.dtype) if hidden.dtype != weight.dtype else hidden
    return F.linear(h, weight).to(out_dtype)


def compute_full_logits(embed_layer, hidden: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary logits, matching ``PI0WaypointJoint._compute_lm_logits``."""
    if hasattr(embed_layer, "compute_logits"):
        return embed_layer.compute_logits(hidden)
    weight = embed_layer.weight
    out_dtype = hidden.dtype
    h = hidden.to(weight.dtype) if hidden.dtype != weight.dtype else hidden
    return F.linear(h, weight).to(out_dtype)


def sample_from_family(
    logits_local: torch.Tensor,
    family: TokenFamily,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one global token id per row from family-local logits.

    Accepts any leading shape ``[..., family.count]`` and returns ``[...]``, so
    several same-family slots can share one projection.

    ``temperature`` is applied *inside* the family, which is exactly what the
    legacy code does after adding ``-inf`` to every illegal entry (softmax is
    invariant to dropping ``-inf`` entries).
    """
    if logits_local.shape[-1] != family.count:
        raise ValueError(
            f"expected [..., {family.count}] logits for family {family.name}, "
            f"got {tuple(logits_local.shape)}"
        )
    if temperature > 0:
        flat = logits_local.reshape(-1, family.count).float() / temperature
        probs = F.softmax(flat, dim=-1)
        local = multinomial_with_generator(probs, generator)
        local = local.squeeze(-1).view(logits_local.shape[:-1])
    else:
        local = torch.argmax(logits_local, dim=-1)
    return local + family.lo


def multinomial_with_generator(
    probs: torch.Tensor, generator: torch.Generator | None
) -> torch.Tensor:
    """``torch.multinomial(probs, 1)`` that tolerates a generator on another device.

    ``torch.multinomial`` requires the generator and the input to share a device.
    The evaluator seeds one CPU ``planner_generator`` per trial while the logits
    live on the GPU, so draw on the generator's device and move the indices back
    -- that keeps a CPU-seeded planner stream reproducible on GPU.
    """
    if generator is not None and generator.device.type != probs.device.type:
        picked = torch.multinomial(probs.to(generator.device), 1, generator=generator)
        return picked.to(probs.device)
    return torch.multinomial(probs, 1, generator=generator)


def legacy_decode_budget(wp_tokenizer) -> int:
    """Exact token budget of a full legacy-decoded plan: header + M * tokens_per_waypoint.

    The legacy decoders (``PI0WaypointJoint._generate_waypoints_legacy`` and
    ``PI0WaypointVLM.generate_waypoints``) force-inject the ``"Action: "``
    header and then emit ``num_waypoints`` blocks of ``tokens_per_waypoint``
    tokens; they stop as soon as every batch row is complete, so this budget is
    both sufficient and tight.  It is the default for ``max_new_tokens=None``:
    the historical fixed default of 128 fits a P=6 plan (3 + 7*10 = 73) but
    silently truncates a P=14 layout (3 + 7*18 = 129) to six waypoints.
    """
    header = wp_tokenizer._pg_tokenizer.encode("Action: ")  # noqa: SLF001
    return len(header) + wp_tokenizer.max_waypoint_tokens


def legacy_constrained_next_token(
    last_logits: torch.Tensor,
    *,
    plan_lens: list[int],
    header_injected: list[bool],
    wp_tokenizer,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    forbid_terminal_first_waypoint: bool = False,
) -> torch.Tensor:
    """One step of the legacy token-at-a-time decoder: mask, then sample.

    Shared verbatim by ``PI0WaypointJoint._generate_waypoints_legacy`` and
    ``PI0WaypointVLM.generate_waypoints``, which used to carry two byte-identical
    copies of this block.  Three defects lived in that duplication: neither copy
    accepted the planner ``generator`` (so ``temperature > 0`` silently drew from
    the un-reseeded global RNG), and neither honoured
    ``forbid_terminal_first_waypoint`` even though ``WaypointDecodeConfig``
    accepts that combination without complaint.

    ``last_logits`` is ``[B, 1, V]`` and is masked **in place**, exactly as
    before, so greedy decoding stays bit-identical to the pre-refactor legacy
    path.  ``plan_lens[b]`` is how many plan tokens row ``b`` has already
    emitted (the injected ``"Action: "`` header does not count).

    Returns the ``[B, 1]`` sampled token ids; structural delimiters are still
    force-injected by the caller.
    """
    vocab_size = last_logits.shape[-1]
    device = last_logits.device
    tpw = wp_tokenizer.tokens_per_waypoint
    grip_pos = wp_tokenizer.gripper_pos_in_wp
    # Gripper slots occupy [grip_pos, dur_delim_pos): one slot per gripper.
    dur_delim_pos = wp_tokenizer.dur_delimiter_pos_in_wp
    dur_val_pos = wp_tokenizer.duration_pos_in_wp
    use_grip = wp_tokenizer.use_gripper_token

    for b in range(last_logits.shape[0]):
        if not header_injected[b]:
            continue
        emitted = plan_lens[b]
        pos_in_wp = emitted % tpw
        if 1 <= pos_in_wp <= wp_tokenizer.proprio_dim:
            family, forbid_id = PROPRIO_FAMILY, None
        elif use_grip and grip_pos <= pos_in_wp < dur_delim_pos:
            family, forbid_id = GRIPPER_FAMILY, None
        elif pos_in_wp == dur_val_pos:
            family = DURATION_FAMILY
            # duration 0 is the terminal marker (token = BASE = family.hi); a
            # first waypoint with d1 == 0 executes nothing and the evaluator
            # replans from the same observation, i.e. a no-progress loop.
            forbid_id = (
                DURATION_FAMILY.hi
                if (forbid_terminal_first_waypoint and emitted == dur_val_pos)
                else None
            )
        else:
            continue
        mask_val = torch.full((vocab_size,), float("-inf"), device=device)
        mask_val[family.lo : family.hi + 1] = 0.0
        if forbid_id is not None:
            mask_val[forbid_id] = float("-inf")
        last_logits[b, 0] += mask_val

    if temperature > 0:
        probs = F.softmax(last_logits[:, 0] / temperature, dim=-1)
        return multinomial_with_generator(probs, generator)
    return torch.argmax(last_logits[:, 0], dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Cache / mask bookkeeping
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DecodeCache:
    """KV cache plus the two length coordinates the legacy loop conflates."""

    past_kv: object
    key_valid: torch.Tensor  # [B, physical_len] bool: attendable cached slots
    logical_len: torch.Tensor  # [B] long: RoPE position of the next valid token
    physical_len: int  # KV storage cursor (padded prefix slots included)
    hidden_dtype: torch.dtype = torch.float32

    @property
    def batch_size(self) -> int:
        return self.key_valid.shape[0]

    def append(self, new_key_valid: torch.Tensor) -> None:
        """Register ``Q`` freshly written KV slots.

        ``physical_len`` always advances by ``Q`` (the cache write is
        rectangular over the batch); ``logical_len`` advances only by the number
        of *valid* new tokens in each row, so a finished row keeps its RoPE
        coordinates frozen.
        """
        if new_key_valid.dim() != 2:
            raise ValueError(f"new_key_valid must be [B, Q], got {tuple(new_key_valid.shape)}")
        self.key_valid = torch.cat([self.key_valid, new_key_valid], dim=1)
        self.logical_len = self.logical_len + new_key_valid.sum(dim=1).to(self.logical_len.dtype)
        self.physical_len += int(new_key_valid.shape[1])


def build_decode_mask(
    cache: DecodeCache,
    new_key_valid: torch.Tensor,
    *,
    intra: str = "causal",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive ``[B, 1, Q, physical_len + Q]`` mask for a packed decode step.

    ``intra="causal"``  token q sees new tokens <= q (packed token-AR).
    ``intra="full"``    every token in the block sees every token in the block
                        (block-wise AR: matches the training-time
                        ``make_att_2d_masks`` block pattern).

    A query row that would otherwise see nothing (a finished batch row whose own
    key is masked out) is given the first prefix key so SDPA cannot produce NaN;
    its output is discarded by the caller.
    """
    if intra not in ("causal", "full"):
        raise ValueError(f"intra must be 'causal' or 'full', got {intra!r}")
    b, q = new_key_valid.shape
    device = new_key_valid.device
    allowed_past = cache.key_valid[:, None, :].expand(b, q, cache.physical_len)
    if intra == "causal":
        tri = torch.tril(torch.ones(q, q, dtype=torch.bool, device=device))
    else:
        tri = torch.ones(q, q, dtype=torch.bool, device=device)
    allowed_self = tri[None].expand(b, q, q) & new_key_valid[:, None, :]
    allowed = torch.cat([allowed_past, allowed_self], dim=-1)
    dead = ~allowed.any(dim=-1)
    if bool(dead.any()):
        allowed = allowed.clone()
        allowed[..., 0] = allowed[..., 0] | dead
    return torch.where(allowed[:, None], 0.0, NEG_INF).to(dtype)


def build_decode_positions(cache: DecodeCache, new_key_valid: torch.Tensor) -> torch.Tensor:
    """Logical RoPE position ids ``[B, Q]`` for a packed decode step."""
    valid = new_key_valid.long()
    prior_valid = torch.cumsum(valid, dim=1) - valid
    return cache.logical_len[:, None] + prior_valid


# ---------------------------------------------------------------------------
# Prefix prefill
# ---------------------------------------------------------------------------


def embed_planner_prefix(
    model,
    images: dict[str, torch.Tensor],
    image_masks: dict[str, torch.Tensor],
    prompt_tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    timer: _SectionTimer | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Image + language prefix embeddings, identical to the legacy planner."""
    paligemma = model.paligemma_with_expert.paligemma
    b = prompt_tokens.shape[0]
    embs, pads = [], []

    ctx = timer.gpu("image_encode_ms") if timer is not None else contextlib.nullcontext()
    with ctx:
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            if key not in images:
                continue
            img = images[key]
            if img.shape[-1] == 3 and img.ndim == 4:
                img = img.permute(0, 3, 1, 2)
            img_emb = paligemma.model.get_image_features(img)
            embs.append(img_emb)
            pads.append(image_masks[key][:, None].expand(b, img_emb.shape[1]))

    token_embs = paligemma.language_model.embed_tokens(prompt_tokens)
    token_embs = token_embs * math.sqrt(token_embs.shape[-1])
    embs.append(token_embs)
    pads.append(prompt_mask)

    return torch.cat(embs, dim=1), torch.cat(pads, dim=1)


def prefill_planner_prefix(
    model,
    prefix_embs: torch.Tensor,
    prefix_pad: torch.Tensor,
    *,
    respect_prefix_pad: bool,
    timer: _SectionTimer | None = None,
    prefix_ar: torch.Tensor | None = None,
    stats: DecodeStats | None = None,
) -> tuple[DecodeCache, torch.Tensor]:
    """Run the prefix forward and return the cache + last hidden state.

    ``prefix_ar`` defaults to all-zeros (fully bidirectional prefix, as the
    legacy planner does).  The block planner passes ``1`` on the ``"Action: "``
    header tokens so the prefill reproduces the training-time visibility of that
    header exactly.

    ``key_valid`` starts as ``prefix_pad`` when ``respect_prefix_pad`` is set and
    as all-ones otherwise; the all-ones variant reproduces the legacy
    fully-visible cached-decode mask exactly.
    """
    paligemma = model.paligemma_with_expert.paligemma
    model_dtype = paligemma.language_model.embed_tokens.weight.dtype

    if prefix_ar is None:
        prefix_ar = torch.zeros_like(prefix_pad, dtype=torch.int32)
    att_2d = make_att_2d_masks(prefix_pad, prefix_ar)
    positions = torch.cumsum(prefix_pad, dim=1) - 1
    att_4d = torch.where(att_2d[:, None], 0.0, NEG_INF).to(model_dtype)

    if prefix_embs.dtype != model_dtype:
        prefix_embs = prefix_embs.to(model_dtype)

    ctx = timer.gpu("prefix_prefill_ms") if timer is not None else contextlib.nullcontext()
    with ctx:
        outputs = paligemma.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=att_4d,
            position_ids=positions,
            use_cache=True,
        )
    if stats is not None:
        # Counted like any other forward so the arms are directly comparable:
        # every implementation pays exactly one prefix prefill.
        stats.transformer_forward_count += 1

    physical_len = prefix_embs.shape[1]
    key_valid = prefix_pad.clone() if respect_prefix_pad else torch.ones_like(prefix_pad)
    cache = DecodeCache(
        past_kv=outputs.past_key_values,
        key_valid=key_valid,
        logical_len=prefix_pad.sum(dim=1).long(),
        physical_len=physical_len,
        hidden_dtype=outputs.last_hidden_state.dtype,
    )
    # The legacy planner takes `last_hidden_state[:, -1:]`, which is a *padding*
    # slot whenever the prompt is right-padded.  It happens to be harmless there
    # (the first emitted token is a forced header token, so that hidden state is
    # discarded), but returning it would be a footgun for any future caller:
    # gather each row's last valid position instead.
    hidden = outputs.last_hidden_state
    last_valid = (prefix_pad.sum(dim=1).long() - 1).clamp(min=0)
    index = last_valid[:, None, None].expand(-1, 1, hidden.shape[-1])
    return cache, hidden.gather(1, index)


def decode_forward(
    model,
    cache: DecodeCache,
    token_ids: torch.Tensor,
    new_key_valid: torch.Tensor,
    *,
    intra: str = "causal",
    inputs_embeds: torch.Tensor | None = None,
    timer: _SectionTimer | None = None,
    stats: DecodeStats | None = None,
    update_cache: bool = True,
) -> torch.Tensor:
    """One packed decoder forward over ``Q`` new positions.

    Args:
        token_ids: ``[B, Q]`` token ids, ignored when ``inputs_embeds`` is given.
        new_key_valid: ``[B, Q]`` whether each new KV slot may be attended later.
        intra: ``"causal"`` or ``"full"`` visibility inside the new block.
        inputs_embeds: pre-built ``[B, Q, W]`` embeddings (block planner).
        update_cache: ``False`` runs the forward against the cache without
            writing to it (Jacobi iterations re-decode the same positions).

    Returns:
        Hidden states ``[B, Q, W]``.
    """
    paligemma = model.paligemma_with_expert.paligemma
    model_dtype = paligemma.language_model.embed_tokens.weight.dtype

    if inputs_embeds is None:
        embs = paligemma.language_model.embed_tokens(token_ids)
        embs = embs * math.sqrt(embs.shape[-1])
    else:
        embs = inputs_embeds
    if embs.dtype != model_dtype:
        embs = embs.to(model_dtype)

    mask = build_decode_mask(cache, new_key_valid, intra=intra, dtype=model_dtype)
    positions = build_decode_positions(cache, new_key_valid)

    ctx = timer.gpu("decode_transformer_ms") if timer is not None else contextlib.nullcontext()
    with ctx:
        outputs = paligemma.language_model.forward(
            inputs_embeds=embs,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache.past_kv,
            use_cache=update_cache,
        )
    if stats is not None:
        stats.transformer_forward_count += 1
    if update_cache:
        cache.past_kv = outputs.past_key_values
        cache.append(new_key_valid)
    return outputs.last_hidden_state


# ---------------------------------------------------------------------------
# Compact token-AR decoder
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _CompactRunner:
    model: object
    wp_tokenizer: object
    cfg: WaypointDecodeConfig
    cache: DecodeCache
    stats: DecodeStats
    timer: _SectionTimer | None
    generator: torch.Generator | None = None

    def __post_init__(self):
        self.batch = self.cache.batch_size
        self.device = self.cache.key_valid.device
        self.finished = torch.zeros(self.batch, dtype=torch.bool, device=self.device)
        self.tokens: list[list[int]] = [[] for _ in range(self.batch)]
        self.embed_layer = self.model.paligemma_with_expert.paligemma.language_model.embed_tokens

    # -- token bookkeeping -------------------------------------------------
    def record(self, token_ids: torch.Tensor, *, forced: bool) -> None:
        """Append one token per *active* row to that row's output sequence."""
        ctx = self.timer.cpu("decode_python_ms") if self.timer is not None else contextlib.nullcontext()
        with ctx:
            ids = token_ids.tolist()
            for b in range(self.batch):
                if not bool(self.finished[b]):
                    self.tokens[b].append(int(ids[b]))
                    if forced:
                        self.stats.generated_forced_tokens += 1
                    else:
                        self.stats.generated_value_tokens += 1

    def broadcast(self, token_id: int) -> torch.Tensor:
        return torch.full((self.batch,), int(token_id), dtype=torch.long, device=self.device)

    def active_key_valid(self, q: int) -> torch.Tensor:
        """New KV slots are valid only for rows that have not stopped."""
        return (~self.finished)[:, None].expand(self.batch, q).contiguous()

    # -- model calls -------------------------------------------------------
    def feed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Feed ``[B, Q]`` tokens; return the hidden state of the last position."""
        q = token_ids.shape[1]
        hidden = decode_forward(
            self.model,
            self.cache,
            token_ids,
            self.active_key_valid(q),
            intra="causal",
            timer=self.timer,
            stats=self.stats,
        )
        return hidden[:, -1]

    def predict(
        self, hidden: torch.Tensor, family: TokenFamily, forbid_local: int | None = None
    ) -> torch.Tensor:
        ctx = self.timer.gpu("decode_lm_head_ms") if self.timer is not None else contextlib.nullcontext()
        with ctx:
            if self.cfg.compact_logits:
                logits = compute_family_logits(self.embed_layer, hidden.float(), family)
            else:
                full = compute_full_logits(self.embed_layer, hidden.float())
                logits = full[:, family.lo : family.hi + 1]
            if forbid_local is not None:
                logits = logits.clone()
                logits[:, forbid_local] = NEG_INF
        self.stats.lm_head_invocation_count += 1
        return sample_from_family(logits, family, self.cfg.temperature, self.generator)


@torch.no_grad()
def generate_waypoints_compact(
    model,
    images: dict[str, torch.Tensor],
    image_masks: dict[str, torch.Tensor],
    prompt_tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    wp_tokenizer,
    cfg: WaypointDecodeConfig,
    generator: torch.Generator | None = None,
) -> tuple[list[list[tuple]], DecodeStats]:
    """Constrained token-AR planner with packed delimiters and family logits.

    Emits exactly the same token stream as the legacy decoder under greedy
    decoding (see ``tests/waypoint/test_compact_decode.py``); the only
    intentional differences are the truncated plan under ``early_stop`` and the
    skipped RNG draws at forced positions under ``temperature > 0``.
    """
    stats = DecodeStats(impl="compact")
    wall0 = time.perf_counter()
    timer = _SectionTimer(cfg.profile, prompt_tokens.device)

    prefix_embs, prefix_pad = embed_planner_prefix(
        model, images, image_masks, prompt_tokens, prompt_mask, timer
    )
    cache, _prefill_hidden = prefill_planner_prefix(
        model, prefix_embs, prefix_pad, respect_prefix_pad=cfg.respect_prefix_pad,
        timer=timer, stats=stats,
    )
    # The legacy decoder computes an LM head on the prefill hidden state and
    # then discards it (the first emitted token is the forced "Action: " header).
    # Nothing to do here -- that is one of the removed redundancies.

    run = _CompactRunner(model, wp_tokenizer, cfg, cache, stats, timer, generator)
    tpw = wp_tokenizer.tokens_per_waypoint
    proprio_dim = wp_tokenizer.proprio_dim
    use_grip = wp_tokenizer.use_gripper_token
    num_waypoints = wp_tokenizer.num_waypoints

    header_ids = wp_tokenizer._pg_tokenizer.encode("Action: ")  # noqa: SLF001

    # --- header + first <wp> in one causal forward -----------------------
    lead = [*list(header_ids), wp_tokenizer.wp_token_id]
    if not cfg.pack_forced_tokens:
        # Unpacked reference schedule: one forward per structural token.
        hidden = None
        for tok in lead:
            hidden = run.feed(run.broadcast(tok)[:, None])
        run.record(run.broadcast(wp_tokenizer.wp_token_id), forced=True)
    else:
        lead_ids = torch.tensor(lead, dtype=torch.long, device=run.device)[None].expand(run.batch, -1)
        run.record(run.broadcast(wp_tokenizer.wp_token_id), forced=True)
        hidden = run.feed(lead_ids)

    for wp_idx in range(num_waypoints):
        # --- continuous pose slots (inherently sequential in token-AR) ----
        for _ in range(proprio_dim):
            tok = run.predict(hidden, PROPRIO_FAMILY)
            run.record(tok, forced=False)
            hidden = run.feed(tok[:, None])

        # --- gripper slot(s); the last one is packed with the <dur> delimiter
        dur_delim = run.broadcast(wp_tokenizer.dur_token_id)
        if use_grip:
            n_grip = getattr(wp_tokenizer, "n_gripper_slots", 1)
            for k in range(n_grip):
                grip = run.predict(hidden, GRIPPER_FAMILY)
                run.record(grip, forced=False)
                if k < n_grip - 1:
                    # Extra gripper slots are ordinary sequential predictions;
                    # only the final one shares a forward with <dur>.
                    hidden = run.feed(grip[:, None])
            run.record(dur_delim, forced=True)
            if cfg.pack_forced_tokens:
                hidden = run.feed(torch.stack([grip, dur_delim], dim=1))
            else:
                run.feed(grip[:, None])
                hidden = run.feed(dur_delim[:, None])
        else:
            run.record(dur_delim, forced=True)
            hidden = run.feed(dur_delim[:, None])

        # --- duration slot ------------------------------------------------
        # duration 0 is the terminal marker; token = DURATION_FAMILY.hi, i.e.
        # local index count-1 inside the narrowed slice.
        forbid_local = (
            DURATION_FAMILY.count - 1
            if (cfg.forbid_terminal_first_waypoint and wp_idx == 0)
            else None
        )
        dur_tok = run.predict(hidden, DURATION_FAMILY, forbid_local=forbid_local)
        run.record(dur_tok, forced=False)

        durations = (DURATION_FAMILY.hi - dur_tok).clamp(min=0)  # token = BASE - duration
        if cfg.early_stop:
            run.finished = run.finished | (durations == 0)
            if bool(run.finished.all()):
                break

        if wp_idx == num_waypoints - 1:
            # No further hidden state is needed: skip the legacy trailing
            # forward + LM head whose output is never consumed.
            break

        # --- next <wp>, packed with the duration token --------------------
        next_wp = run.broadcast(wp_tokenizer.wp_token_id)
        run.record(next_wp, forced=True)
        if cfg.pack_forced_tokens:
            hidden = run.feed(torch.stack([dur_tok, next_wp], dim=1))
        else:
            run.feed(dur_tok[:, None])
            hidden = run.feed(next_wp[:, None])

    plans = []
    for b in range(run.batch):
        seq = run.tokens[b]
        usable = (len(seq) // tpw) * tpw
        plans.append(wp_tokenizer.decode_waypoints(seq[:usable]))

    stats.plan_len = max((len(p) for p in plans), default=0)
    _finalize_stats(stats, timer, wall0, prompt_tokens.device)
    return plans, stats


# ---------------------------------------------------------------------------
# Jacobi (fixed-point parallel) decoder -- no retraining
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_waypoints_jacobi(
    model,
    images: dict[str, torch.Tensor],
    image_masks: dict[str, torch.Tensor],
    prompt_tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    wp_tokenizer,
    cfg: WaypointDecodeConfig,
    generator: torch.Generator | None = None,  # greedy only; kept for a uniform signature
) -> tuple[list[list[tuple]], DecodeStats]:
    """Greedy-equivalent parallel decode by Jacobi fixed-point iteration.

    All ``M * tokens_per_waypoint`` plan positions are guessed, then refreshed
    together each iteration.  Because the mask stays strictly causal inside the
    plan, position ``i`` is a deterministic function of positions ``< i``, so
    after iteration ``k`` at least the first ``k`` positions equal the
    token-by-token greedy result: the fixed point *is* the greedy AR output
    (PD-VLA, arXiv 2503.02310).  Only worthwhile with ``temperature == 0``.

    Cost: one ``N``-token forward per iteration (no cache writes) instead of
    ``N`` single-token forwards, so it pays off whenever the number of
    iterations to convergence is well below ``N``.
    """
    if cfg.temperature > 0:
        raise ValueError("jacobi decoding is only equivalence-safe for temperature == 0")

    stats = DecodeStats(impl="jacobi")
    wall0 = time.perf_counter()
    timer = _SectionTimer(cfg.profile, prompt_tokens.device)

    prefix_embs, prefix_pad = embed_planner_prefix(
        model, images, image_masks, prompt_tokens, prompt_mask, timer
    )
    cache, _ = prefill_planner_prefix(
        model, prefix_embs, prefix_pad, respect_prefix_pad=cfg.respect_prefix_pad,
        timer=timer, stats=stats,
    )

    device = prompt_tokens.device
    batch = prompt_tokens.shape[0]
    tpw = wp_tokenizer.tokens_per_waypoint
    num_waypoints = wp_tokenizer.num_waypoints
    header_ids = list(wp_tokenizer._pg_tokenizer.encode("Action: "))  # noqa: SLF001
    n_plan = num_waypoints * tpw
    n_total = len(header_ids) + n_plan

    embed_layer = model.paligemma_with_expert.paligemma.language_model.embed_tokens
    schedule = wp_tokenizer.slot_schedule()

    # Initial guess: header + delimiters are already final; every predicted slot
    # starts at its family's mid bin (any legal guess converges).
    seq = torch.zeros(batch, n_total, dtype=torch.long, device=device)
    for i, tok in enumerate(header_ids):
        seq[:, i] = tok
    forced = torch.zeros(n_total, dtype=torch.bool, device=device)
    forced[: len(header_ids)] = True
    family_positions: dict[TokenFamily, list[int]] = {}
    for j in range(num_waypoints):
        for k in range(tpw):
            pos = len(header_ids) + j * tpw + k
            forced_tok = wp_tokenizer.forced_token_for_slot(k)
            if forced_tok is not None:
                seq[:, pos] = forced_tok
                forced[pos] = True
            else:
                family = schedule[k]
                seq[:, pos] = family.lo + family.count // 2
                family_positions.setdefault(family, []).append(pos)

    new_key_valid = torch.ones(batch, n_total, dtype=torch.bool, device=device)
    # Positions whose family logits must exclude the terminal duration.  Applied
    # here too, so `forbid_terminal_first_waypoint` means the same thing under
    # every decoder -- otherwise a compact-vs-jacobi comparison would not be
    # comparing the same constrained-decoding rules.
    forbidden_first_duration = (
        len(header_ids) + wp_tokenizer.duration_pos_in_wp
        if cfg.forbid_terminal_first_waypoint
        else None
    )
    max_iters = cfg.jacobi_max_iters or n_total
    converged = False
    converged_iters = 0
    for it in range(max_iters):
        hidden = decode_forward(
            model,
            cache,
            seq,
            new_key_valid,
            intra="causal",
            timer=timer,
            stats=stats,
            update_cache=False,
        )
        # hidden[:, i] predicts position i+1, so the prediction for plan
        # position p comes from hidden[:, p-1].
        with timer.gpu("decode_lm_head_ms"):
            updated = seq.clone()
            # One projection per *family*, not per slot: all 42 pose positions
            # share a single [B*42, W] x [300, W]^T GEMM instead of 42 tiny ones.
            for family, positions in family_positions.items():
                src = torch.as_tensor([p - 1 for p in positions], device=device)
                logits = compute_family_logits(
                    embed_layer, hidden.index_select(1, src).float(), family
                )
                if forbidden_first_duration is not None and forbidden_first_duration in positions:
                    logits = logits.clone()
                    slot = positions.index(forbidden_first_duration)
                    logits[:, slot, family.count - 1] = NEG_INF  # duration 0 == terminal
                stats.lm_head_invocation_count += 1
                picked = sample_from_family(logits, family, 0.0)  # [B, len(positions)]
                updated[:, torch.as_tensor(positions, device=device)] = picked
        changed = bool((updated != seq).any())
        seq = updated
        converged_iters = it + 1
        if not changed:
            converged = True
            break

    stats.jacobi_iters = converged_iters
    stats.fixed_point_verified = True if converged else None
    if not converged:
        # The greedy-equivalence guarantee holds only at a *verified* fixed
        # point, and `jacobi_iters == jacobi_max_iters` cannot tell convergence
        # from truncation -- so the cap must be reported.  But it must not be
        # reported as "NOT greedy": when the cap lands on the iteration that
        # made the last change, the returned plan already IS the fixed point and
        # the old message was a false negative.  Verifying it costs one more
        # iteration, which is exactly what the cap exists to forbid.
        logger.warning(
            "jacobi decoding hit jacobi_max_iters=%d before any iteration confirmed a fixed "
            "point: greedy equivalence of the returned plan is UNVERIFIED (it may or may not "
            "be the greedy AR output; stats.fixed_point_verified=None)",
            max_iters,
        )
    plan_tokens = seq[:, len(header_ids) :]
    stats.generated_value_tokens = int((~forced[len(header_ids) :]).sum()) * batch
    stats.generated_forced_tokens = int(forced[len(header_ids) :].sum()) * batch

    plans = []
    for b in range(batch):
        seq_b = plan_tokens[b].tolist()
        if cfg.early_stop:
            seq_b = _truncate_after_first_stop(seq_b, wp_tokenizer)
        plans.append(wp_tokenizer.decode_waypoints(seq_b))
    stats.plan_len = max((len(p) for p in plans), default=0)
    _finalize_stats(stats, timer, wall0, device)
    return plans, stats


def _truncate_after_first_stop(token_ids: list[int], wp_tokenizer) -> list[int]:
    """Drop everything after the block whose duration token decodes to 0."""
    tpw = wp_tokenizer.tokens_per_waypoint
    dur_pos = wp_tokenizer.duration_pos_in_wp
    for start in range(0, len(token_ids) - tpw + 1, tpw):
        if wp_tokenizer.decode_duration(token_ids[start + dur_pos]) == 0:
            return token_ids[: start + tpw]
    return token_ids


def _finalize_stats(
    stats: DecodeStats, timer: _SectionTimer, wall0: float, device: torch.device
) -> None:
    stats.planner_total_ms = (time.perf_counter() - wall0) * 1000.0
    for name, value in timer.drain().items():
        if hasattr(stats, name):
            setattr(stats, name, getattr(stats, name) + value)
    if timer.enabled and device.type == "cuda" and torch.cuda.is_available():
        stats.peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)

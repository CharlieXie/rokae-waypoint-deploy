"""Waypoint tokenizer for VLM autoregressive waypoint prediction.

Maps continuous proprio values, discrete duration values, and binary gripper
state to PaliGemma vocabulary token IDs, following the same convention as
Pi0-FAST.

Token ID layout (mapped into PaliGemma vocab tail):
  - Proprio bins 0-299:  vocab_size - 1 - SKIP - i        (i=0..299)
  - Duration 0-33:       vocab_size - 1 - SKIP - 300 - d  (d=0..33)
  - <wp> delimiter:      vocab_size - 1 - SKIP - 300 - 34
  - <dur> delimiter:     vocab_size - 1 - SKIP - 300 - 35
  - <grip_open>:         vocab_size - 1 - SKIP - 300 - 36
  - <grip_close>:        vocab_size - 1 - SKIP - 300 - 37
"""

import dataclasses
import logging

import numpy as np

logger = logging.getLogger(__name__)

PALIGEMMA_VOCAB_SIZE = 257152
SKIP_TOKENS = 128  # Last 128 tokens reserved by PaliGemma (same as FAST)

PROPRIO_N_BINS = 300
DURATION_MAX = 33
DURATION_N_BINS = DURATION_MAX + 1  # 0..33 inclusive

# Duration code written into the *input-only* "waypoint -1" block that
# ``planner_block0_cond: current_state`` feeds to block 0.  The current state is
# not reached by any transition, so it has no duration; ``DURATION_MAX`` is used
# as a reserved marker rather than 0 because 0 already means "terminal" and
# conflating the two would push on exactly the representation defect recorded in
# ``docs/12-libero10-root-cause.md`` (F5, early-stop livelock).  Callers that
# enable the feature must verify no *supervised* duration target equals it;
# ``WaypointTokenizer.encode_current_state_block`` is input-only, so the code
# never appears in a label.
CURRENT_STATE_DURATION_CODE = DURATION_MAX

# Token ID offsets from vocab_size - 1 - SKIP_TOKENS
_PROPRIO_BASE = PALIGEMMA_VOCAB_SIZE - 1 - SKIP_TOKENS  # 257023
_DURATION_BASE = _PROPRIO_BASE - PROPRIO_N_BINS  # 256723
_WP_TOKEN_ID = _DURATION_BASE - DURATION_N_BINS  # 256689
_DUR_TOKEN_ID = _WP_TOKEN_ID - 1  # 256688
_GRIP_OPEN_ID = _DUR_TOKEN_ID - 1  # 256687
_GRIP_CLOSE_ID = _GRIP_OPEN_ID - 1  # 256686

# Total dedicated tokens: 300 + 34 + 2 + 2 = 338
# Token ID range: 256686 .. 257023


# ---------------------------------------------------------------------------
# Token families
# ---------------------------------------------------------------------------
# Every constrained-decoding slot draws from one *contiguous* range of vocab
# rows.  Exposing that range explicitly lets the decoder project the hidden
# state onto `count` embedding rows (via ``Tensor.narrow``) instead of onto the
# full 257 152-row vocabulary and then masking 99.99% of it away.
#
# Invariant relied upon by ``planner_decode``: within a family the token id is
# *descending* in the semantic value (``token = BASE - value``), so the
# local index inside the narrowed slice is ``token_id - family.lo`` and the
# semantic value is ``BASE - token_id``.  ``tests/waypoint/test_tokenizer.py``
# pins this down.


@dataclasses.dataclass(frozen=True)
class TokenFamily:
    """A contiguous range of PaliGemma vocabulary rows used by one slot type."""

    name: str
    lo: int  # lowest global token id in the family (inclusive)
    count: int  # number of contiguous vocabulary rows

    @property
    def hi(self) -> int:
        """Highest global token id in the family (inclusive)."""
        return self.lo + self.count - 1

    def contains(self, token_id: int) -> bool:
        return self.lo <= int(token_id) <= self.hi


PROPRIO_FAMILY = TokenFamily("proprio", _PROPRIO_BASE - PROPRIO_N_BINS + 1, PROPRIO_N_BINS)
GRIPPER_FAMILY = TokenFamily("gripper", _GRIP_CLOSE_ID, 2)
DURATION_FAMILY = TokenFamily("duration", _DURATION_BASE - DURATION_N_BINS + 1, DURATION_N_BINS)


class ProprioTokenizer:
    """Discretizes continuous proprio values in [-1,1] into PROPRIO_N_BINS tokens."""

    def __init__(self, n_bins: int = PROPRIO_N_BINS, min_val: float = -1.0, max_val: float = 1.0):
        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Encode continuous values to token IDs.

        Args:
            values: shape (...,) continuous values in [-1, 1].

        Returns:
            Token IDs array of same shape, dtype int64.
        """
        clipped = np.clip(values, self.min_val, self.max_val)
        bin_indices = np.digitize(clipped, self.bin_edges[1:-1])  # 0..n_bins-1
        token_ids = _PROPRIO_BASE - bin_indices
        return token_ids.astype(np.int64)

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        """Decode token IDs back to continuous bin-center values."""
        token_ids = np.asarray(token_ids)
        bin_indices = _PROPRIO_BASE - token_ids
        out_of_range = (bin_indices < 0) | (bin_indices >= self.n_bins)
        if np.any(out_of_range):
            bad_ids = token_ids[out_of_range].tolist()
            logger.warning(
                f"ProprioTokenizer.decode: {int(np.sum(out_of_range))}/{token_ids.size} "
                f"token(s) outside proprio range [{_PROPRIO_BASE - self.n_bins + 1}, "
                f"{_PROPRIO_BASE}]: {bad_ids[:5]}{'...' if len(bad_ids) > 5 else ''}"
            )
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        return self.bin_centers[bin_indices].astype(np.float32)


class WaypointTokenizer:
    """Tokenizes waypoint sequences for VLM autoregressive training.

    Combines ProprioTokenizer with duration tokens, gripper tokens, and
    structural delimiters.  Interfaces with PaliGemma's sentencepiece
    tokenizer for text prefix.

    Token layout per waypoint (use_gripper_token=True, proprio_dim=6):
        <wp> p1 p2 p3 p4 p5 p6 G <dur> d
        pos:  0  1  2  3  4  5  6 7   8  9
        G ∈ {grip_open, grip_close}

    With ``n_gripper_slots=N`` (dual-arm robots) the single ``G`` becomes ``N``
    consecutive gripper slots ``G_1 .. G_N`` right after the proprio run, each
    drawing from the same two-token GRIPPER_FAMILY (no vocabulary change; the
    arm identity is carried by the slot position):
        <wp> p1 .. pP G_1 .. G_N <dur> d
    ``n_gripper_slots=1`` reproduces today's layout, prompt string and token ids
    byte-for-byte, so existing checkpoints and tests are unaffected.
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        proprio_dim: int = 6,
        num_waypoints: int = 7,
        max_token_len: int = 256,
        use_gripper_token: bool = True,
        n_gripper_slots: int = 1,
    ):
        self.proprio_dim = proprio_dim  # continuous dims only (excludes gripper)
        self.num_waypoints = num_waypoints
        self.max_token_len = max_token_len
        self.use_gripper_token = use_gripper_token
        if use_gripper_token and int(n_gripper_slots) < 1:
            raise ValueError(f"n_gripper_slots must be >= 1 when use_gripper_token, got {n_gripper_slots}")
        # Number of gripper slots per waypoint block (0 when gripper tokens are off).
        self.n_gripper_slots = int(n_gripper_slots) if use_gripper_token else 0

        self.proprio_tokenizer = ProprioTokenizer()

        self.wp_token_id = _WP_TOKEN_ID
        self.dur_token_id = _DUR_TOKEN_ID
        self.grip_open_id = _GRIP_OPEN_ID
        self.grip_close_id = _GRIP_CLOSE_ID

        self.__pg_tokenizer = None

    # The PaliGemma SentencePiece model is downloaded from GCS on first use.
    # Keeping it lazy means the pure-token layout (families, slot positions,
    # block geometry) can be exercised in unit tests without network access.
    @property
    def _pg_tokenizer(self):
        if self.__pg_tokenizer is None:
            import sentencepiece

            import openpi.shared.download as download

            path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            with path.open("rb") as f:
                self.__pg_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        return self.__pg_tokenizer

    @property
    def tokens_per_waypoint(self) -> int:
        # <wp> + proprio + [grip x n_gripper_slots] + <dur> + duration
        return 1 + self.proprio_dim + self.n_gripper_slots + 1 + 1

    @property
    def max_waypoint_tokens(self) -> int:
        return self.num_waypoints * self.tokens_per_waypoint

    # --- Gripper token position (0-indexed within a waypoint block) ---

    @property
    def gripper_pos_in_wp(self) -> int:
        """Position of the (first) gripper token within a waypoint block."""
        return 1 + self.proprio_dim  # right after the last proprio token

    @property
    def gripper_slot_positions(self) -> list[int]:
        """Positions of every gripper slot within a waypoint block (may be empty)."""
        return [self.gripper_pos_in_wp + k for k in range(self.n_gripper_slots)]

    @property
    def dur_delimiter_pos_in_wp(self) -> int:
        """Position of the <dur> delimiter within a waypoint block."""
        return self.gripper_pos_in_wp + self.n_gripper_slots

    @property
    def duration_pos_in_wp(self) -> int:
        """Position of the duration value token within a waypoint block."""
        return self.dur_delimiter_pos_in_wp + 1

    # --- Slot schedule (single source of truth for every decoder) ---

    def forced_token_for_slot(self, pos_in_wp: int) -> int | None:
        """Return the deterministic token at ``pos_in_wp``, or None if predicted.

        Structural delimiters are never predicted: the legacy decoder computes
        logits for them and then overwrites the result, the compact/block
        decoders simply inject them.
        """
        if pos_in_wp == 0:
            return self.wp_token_id
        if pos_in_wp == self.dur_delimiter_pos_in_wp:
            return self.dur_token_id
        return None

    def family_for_slot(self, pos_in_wp: int) -> TokenFamily | None:
        """Return the token family a predicted slot draws from (None if forced)."""
        if 1 <= pos_in_wp <= self.proprio_dim:
            return PROPRIO_FAMILY
        if self.use_gripper_token and self.gripper_pos_in_wp <= pos_in_wp < self.dur_delimiter_pos_in_wp:
            return GRIPPER_FAMILY
        if pos_in_wp == self.duration_pos_in_wp:
            return DURATION_FAMILY
        return None

    def slot_schedule(self) -> list[TokenFamily | None]:
        """Per-slot families for one waypoint block; None entries are forced."""
        return [self.family_for_slot(k) for k in range(self.tokens_per_waypoint)]

    def encode_duration(self, duration: int) -> int:
        d = int(duration)
        if d < 0 or d > DURATION_MAX:
            # Silent clipping here corrupts the label (a 40-step segment becomes
            # a 33-step one while the endpoint still belongs to step 40), so the
            # callers are expected to filter first.  Warn loudly if they didn't.
            logger.warning(
                "encode_duration: duration=%d outside [0, %d] and will be clipped; "
                "the caller should have skipped this window instead",
                d,
                DURATION_MAX,
            )
            d = int(np.clip(d, 0, DURATION_MAX))
        return _DURATION_BASE - d

    def decode_duration(self, token_id: int) -> int:
        dur = int(_DURATION_BASE - token_id)
        if dur < 0 or dur > DURATION_MAX:
            logger.warning(
                f"decode_duration: token_id={token_id} decoded to duration={dur} "
                f"(valid: 0-{DURATION_MAX}), token outside duration range "
                f"[{_DURATION_BASE - DURATION_N_BINS + 1}, {_DURATION_BASE}]"
            )
        return dur

    def encode_gripper(self, gripper_open: bool) -> int:
        return _GRIP_OPEN_ID if gripper_open else _GRIP_CLOSE_ID

    def decode_gripper(self, token_id: int) -> float:
        """Decode gripper token to float: 1.0=open, 0.0=close."""
        if token_id == _GRIP_OPEN_ID:
            return 1.0
        elif token_id == _GRIP_CLOSE_ID:
            return 0.0
        else:
            logger.warning(f"decode_gripper: unexpected token_id={token_id}, defaulting to 0.0 (close)")
            return 0.0

    def gripper_vector(self, grippers) -> list[bool]:
        """Coerce a gripper value to exactly ``n_gripper_slots`` booleans.

        A scalar is accepted only for a single slot; a dual-arm tokenizer must be
        handed one value per gripper so that an arm can never be dropped silently
        (the failure mode the ``split_proprio`` guard exists for).
        """
        n = self.n_gripper_slots
        if n == 0:
            return []
        if np.isscalar(grippers) or np.ndim(grippers) == 0:
            if n != 1:
                raise ValueError(
                    f"tokenizer has {n} gripper slots but received a scalar gripper value; "
                    "pass one value per gripper"
                )
            return [bool(grippers)]
        vals = [bool(v) for v in np.asarray(grippers).reshape(-1).tolist()]
        if len(vals) != n:
            raise ValueError(f"expected {n} gripper values, got {len(vals)}: {vals}")
        return vals

    def encode_current_state_block(
        self,
        state_continuous_norm: np.ndarray,
        gripper_open,
    ) -> np.ndarray:
        """Encode the *current* state into one input-only waypoint block.

        Returns ``(tokens_per_waypoint,)`` int64 laid out exactly like a real
        waypoint block -- ``<wp> p1..pN [G] <dur> d`` -- so that block 0 can be
        conditioned with the same token families, slot positions and delimiters
        that condition blocks 1..M-1.  The duration slot carries
        ``CURRENT_STATE_DURATION_CODE``.

        This block is never a target: it only ever enters the model as an input
        embedding, so it adds no supervised token and cannot shift a label.
        """
        state = np.asarray(state_continuous_norm, dtype=np.float32).reshape(-1)
        if state.shape[0] < self.proprio_dim:
            padded = np.zeros(self.proprio_dim, dtype=np.float32)
            padded[: state.shape[0]] = state
            state = padded
        else:
            state = state[: self.proprio_dim]

        block = [self.wp_token_id]
        block.extend(int(t) for t in self.proprio_tokenizer.encode(state).flatten())
        for g in self.gripper_vector(gripper_open):
            block.append(self.encode_gripper(g))
        block.append(self.dur_token_id)
        block.append(self.encode_duration(CURRENT_STATE_DURATION_CODE))
        assert len(block) == self.tokens_per_waypoint, (
            f"current-state block width {len(block)} != tokens_per_waypoint "
            f"{self.tokens_per_waypoint}"
        )
        return np.asarray(block, dtype=np.int64)

    def is_proprio_token(self, token_id: int) -> bool:
        return (_PROPRIO_BASE - PROPRIO_N_BINS + 1) <= token_id <= _PROPRIO_BASE

    def is_duration_token(self, token_id: int) -> bool:
        return (_DURATION_BASE - DURATION_N_BINS + 1) <= token_id <= _DURATION_BASE

    def is_gripper_token(self, token_id: int) -> bool:
        return token_id in (_GRIP_OPEN_ID, _GRIP_CLOSE_ID)

    def prefix_text(self, prompt: str, state: np.ndarray, current_gripper=0) -> str:
        """The planner prompt exactly as training wrote it (the single source of
        truth for train/inference prefix parity).

        ``state``: continuous proprio (``proprio_dim``,) normalized to [-1, 1];
        ``current_gripper``: 0=close / 1=open, a length-N sequence for N grippers.
        """
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ").lower()
        discretized_state = np.digitize(
            np.clip(state, -1, 1),
            bins=np.linspace(-1, 1, PROPRIO_N_BINS + 1)[:-1],
        ) - 1
        state_str = " ".join(map(str, discretized_state.astype(int)))
        if self.use_gripper_token:
            # Compact wording (decision D5): one word per gripper, e.g.
            # "Gripper: open closed;" -- +1 sentencepiece token per extra gripper.
            grip_str = " ".join(
                "open" if g else "closed" for g in self.gripper_vector(current_gripper)
            )
            return f"Task: {cleaned_text}, State: {state_str}, Gripper: {grip_str};\n"
        return f"Task: {cleaned_text}, State: {state_str};\n"

    def encode_prefix(self, prompt: str, state: np.ndarray, current_gripper=0) -> list[int]:
        """Token ids of :meth:`prefix_text` (with BOS) -- what ``generate_waypoints``
        takes as ``prompt_tokens`` and what ``tokenize`` puts before the plan."""
        return self._pg_tokenizer.encode(self.prefix_text(prompt, state, current_gripper), add_bos=True)

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray,
        wp_proprios: np.ndarray | None,
        wp_durations: np.ndarray | None,
        current_gripper: int = 0,
        wp_grippers: np.ndarray | None = None,
        wp_pad_mask_proprio: np.ndarray | None = None,
        wp_pad_mask_duration: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a full VLM waypoint training sample.

        Args:
            prompt: Language instruction text.
            state: Current proprio state, shape (proprio_dim,) — continuous dims
                   only (no gripper), normalized to [-1,1]. Discretized into prefix.
            wp_proprios: Waypoint proprio values, shape (M, proprio_dim), continuous
                         dims only, normalized to [-1,1].  None during inference.
            wp_durations: Waypoint durations, shape (M,), integers 0..33.
                          None during inference.
            current_gripper: 0=close, 1=open.  Encoded in prefix text.  With
                         ``n_gripper_slots>1`` a length-N sequence (one per gripper).
            wp_grippers: (M,) or (M, N) int/float, 0=close, 1=open per waypoint
                         (and per gripper slot).  None during inference.
            wp_pad_mask_proprio: (M,) bool, True = this WP is padding (no loss).
            wp_pad_mask_duration: (M,) bool, True = this duration is padding (no loss).

        Returns:
            tokens: (max_token_len,) int array.
            token_mask: (max_token_len,) bool, True = valid (not sequence padding).
            ar_mask: (max_token_len,) int, 0 = bidirectional, 1 = causal.
            loss_mask: (max_token_len,) bool, True = compute CE loss on this token.
        """
        prefix_tokens = self.encode_prefix(prompt, state, current_gripper)

        if wp_proprios is not None:
            postfix_tokens, postfix_loss_mask = self._encode_waypoint_postfix(
                wp_proprios, wp_durations, wp_grippers,
                wp_pad_mask_proprio, wp_pad_mask_duration,
            )
        else:
            postfix_tokens = []
            postfix_loss_mask = []

        action_header = self._pg_tokenizer.encode("Action: ")
        action_footer = self._pg_tokenizer.encode("|", add_eos=True)

        full_postfix = action_header + postfix_tokens + action_footer
        full_postfix_loss = [False] * len(action_header) + postfix_loss_mask + [False] * len(action_footer)

        tokens = prefix_tokens + full_postfix
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(full_postfix)
        loss_mask = [False] * len(prefix_tokens) + full_postfix_loss

        tokens_len = len(tokens)
        if tokens_len < self.max_token_len:
            pad_len = self.max_token_len - tokens_len
            tokens = tokens + [0] * pad_len
            token_mask = token_mask + [False] * pad_len
            ar_mask = ar_mask + [0] * pad_len
            loss_mask = loss_mask + [False] * pad_len
        else:
            if tokens_len > self.max_token_len:
                logger.warning(
                    f"Token length ({tokens_len}) exceeds max ({self.max_token_len}), truncating."
                )
            tokens = tokens[: self.max_token_len]
            token_mask = token_mask[: self.max_token_len]
            ar_mask = ar_mask[: self.max_token_len]
            loss_mask = loss_mask[: self.max_token_len]

        return (
            np.asarray(tokens, dtype=np.int64),
            np.asarray(token_mask, dtype=bool),
            np.asarray(ar_mask, dtype=np.int32),
            np.asarray(loss_mask, dtype=bool),
        )

    def _encode_waypoint_postfix(
        self,
        wp_proprios: np.ndarray,
        wp_durations: np.ndarray,
        wp_grippers: np.ndarray | None,
        wp_pad_mask_proprio: np.ndarray | None,
        wp_pad_mask_duration: np.ndarray | None,
    ) -> tuple[list[int], list[bool]]:
        """Encode M waypoints into token IDs + per-token loss mask.

        Per-waypoint layout:
          <wp>  p1..pN  [G]  <dur>  d
        where G is the optional gripper token (grip_open / grip_close).
        """
        M = len(wp_proprios)
        tokens = []
        loss_mask = []

        for i in range(M):
            is_pad_proprio = wp_pad_mask_proprio is not None and wp_pad_mask_proprio[i]
            is_pad_duration = wp_pad_mask_duration is not None and wp_pad_mask_duration[i]

            # <wp> delimiter (forced, no loss)
            tokens.append(self.wp_token_id)
            loss_mask.append(False)

            # Continuous proprio tokens
            proprio_tids = self.proprio_tokenizer.encode(wp_proprios[i])
            for tid in proprio_tids.flatten():
                tokens.append(int(tid))
                loss_mask.append(not is_pad_proprio)

            # Gripper token(s), one per slot
            if self.use_gripper_token:
                if wp_grippers is not None:
                    grip_tids = [self.encode_gripper(g) for g in self.gripper_vector(wp_grippers[i])]
                else:
                    grip_tids = [self.grip_close_id] * self.n_gripper_slots  # default for padding
                for grip_tid in grip_tids:
                    tokens.append(grip_tid)
                    # Gripper shares pad mask with proprio
                    loss_mask.append(not is_pad_proprio)

            # <dur> delimiter (forced, no loss)
            tokens.append(self.dur_token_id)
            loss_mask.append(False)

            # Duration token
            dur_tid = self.encode_duration(int(wp_durations[i]))
            tokens.append(dur_tid)
            loss_mask.append(not is_pad_duration)

        return tokens, loss_mask

    def decode_waypoints(self, token_ids: list[int] | np.ndarray) -> list[tuple[np.ndarray, int]]:
        """Decode generated token sequence into list of (proprio, duration) tuples.

        Returns:
            List of (proprio, duration) tuples.
            If use_gripper_token: proprio is (proprio_dim + n_gripper_slots,) with
            the gripper values (1.0=open, 0.0=close) appended in slot order after
            the continuous dims (a single trailing value for one gripper).
            Otherwise: proprio is (proprio_dim,).
        """
        token_ids = list(token_ids)
        waypoints = []
        tpw = self.tokens_per_waypoint

        for start in range(0, len(token_ids), tpw):
            block = token_ids[start : start + tpw]
            if len(block) < tpw:
                break

            # Decode continuous proprio (positions 1..proprio_dim)
            proprio_tids = np.array(block[1 : 1 + self.proprio_dim])
            proprio_values = self.proprio_tokenizer.decode(proprio_tids)

            # Decode gripper(s) → append to proprio as the trailing dims
            for pos in self.gripper_slot_positions:
                proprio_values = np.append(proprio_values, self.decode_gripper(block[pos]))

            # Decode duration
            dur_tid = block[self.duration_pos_in_wp]
            duration = self.decode_duration(dur_tid)

            waypoints.append((proprio_values, duration))

        return waypoints

    def extract_waypoint_tokens_from_output(self, output_tokens: np.ndarray) -> list[int]:
        """Extract waypoint-relevant tokens from a full generated sequence.

        Looks for the 'Action:' header and '|' footer, returning only the
        waypoint tokens between them.
        """
        decoded = self._pg_tokenizer.decode(output_tokens.tolist())
        if "Action: " not in decoded:
            return []

        action_text = decoded.split("Action: ")[1].split("|")[0]
        wp_token_ids = self._pg_tokenizer.encode(action_text)
        return wp_token_ids

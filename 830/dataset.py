"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import math
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1

# Inter-event gap bucket boundaries, following a coarser human-time scale than
# the recency buckets above.  These buckets describe time between adjacent
# historical behaviors inside one sequence.
TIME_SPAN_BUCKET_BOUNDARIES = np.array([
    30, 60, 180, 300, 600, 900, 1800, 3600,
    10800, 21600, 43200, 86400, 172800, 345600, 604800,
], dtype=np.int64)

# Includes padding=0 plus all np.searchsorted buckets after +1.
NUM_TIME_SPAN_BUCKETS = len(TIME_SPAN_BUCKET_BOUNDARIES) + 2


# ─── EXPΩ Patch 1: per-event unified sincos 8d × 4 periods (UTC+8) ──────────
# Replaces Mega8's SeqFourierTimeEncoder. For every event timestamp produces
# (sin, cos) × (hour/24, dow/7, dom/31, month/12) → 8d/event. UTC+8 offset
# applied BEFORE deriving calendar fields (matches Mega9's hour_ids logic).
# Padding positions (ts == 0) yield all-zero rows.
PER_EVENT_UNIFIED_SINCOS_DIM = 8


def _compute_per_event_unified_sincos(ts_padded: np.ndarray) -> np.ndarray:
    """Per-event unified sin/cos × 4 periods (hour/dow/dom/month), UTC+8.

    Args:
        ts_padded: [B, L] int64 UTC Unix seconds (0 = padding).

    Returns:
        [B, L, 8] float32 sin/cos features. Padding rows are zeroed.
    """
    valid = ts_padded > 0
    safe_ts_utc = np.where(valid, ts_padded, 0).astype(np.int64)
    # Mega9-aligned UTC+8 shift; everything downstream is CST calendar.
    # Hard-coded literal because TZ_OFFSET_SEC is defined below this helper
    # block; both resolve to 8 * 3600 = 28800 seconds.
    safe_ts_local = safe_ts_utc + np.int64(8 * 3600)

    sec_of_day = safe_ts_local % 86400
    hour_of_day = (sec_of_day // 3600).astype(np.float32)

    days_since_epoch = safe_ts_local // 86400
    # 1970-01-01 = Thursday (=3 if Mon=0). Match the same dow convention as
    # the existing hour/weekday embedding (which uses % 7 alone, anchored on
    # epoch). Keep the (days + 4) % 7 pattern below for parity with the
    # 0..6 phase used in cyclical_features inside _embed_seq_domain.
    day_of_week = (days_since_epoch % 7).astype(np.float32)

    dt_days_local = safe_ts_local.astype("datetime64[s]").astype("datetime64[D]")
    years = dt_days_local.astype("datetime64[Y]")
    months = dt_days_local.astype("datetime64[M]")
    month_of_year = (
        months.astype(np.int64) - years.astype(np.int64) * 12
    ).astype(np.float32)
    day_of_month = (
        dt_days_local.astype(np.int64) - months.astype("datetime64[D]").astype(np.int64)
    ).astype(np.float32)

    hour_angle = (2.0 * np.pi * hour_of_day) / 24.0
    dow_angle = (2.0 * np.pi * day_of_week) / 7.0
    dom_angle = (2.0 * np.pi * day_of_month) / 31.0
    mon_angle = (2.0 * np.pi * month_of_year) / 12.0

    result = np.stack([
        np.sin(hour_angle), np.cos(hour_angle),
        np.sin(dow_angle),  np.cos(dow_angle),
        np.sin(dom_angle),  np.cos(dom_angle),
        np.sin(mon_angle),  np.cos(mon_angle),
    ], axis=-1).astype(np.float32)
    result[~valid] = 0.0
    return result


# ─── EXPΩ Patch 2: best2 session pseudo-fid helpers (renamed for uniqueness) ─
# Renamed so we do not collide with best2's SESSION_BREAK_SECONDS /
# SESSION_INDEX_BOUNDARIES symbols. Functional behaviour preserved.
_SESSION_GAP_THRESHOLD_SEC = 3600
_USER_SESSION_BUCKET_EDGES = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
_NUM_USER_SESSION_BUCKETS = len(_USER_SESSION_BUCKET_EDGES) + 1


def _quantize_user_session_index(session_idx: int) -> int:
    """Map raw user session count to a log-scale bucket id (0=pad)."""
    raw = int(np.searchsorted(_USER_SESSION_BUCKET_EDGES, session_idx, side="right"))
    return min(raw, _NUM_USER_SESSION_BUCKETS - 1)


def _quantize_session_relative_delta(delta_seconds: int) -> int:
    """Reuse BUCKET_BOUNDARIES for session elapsed / prev-gap quantization."""
    raw = int(np.searchsorted(BUCKET_BOUNDARIES, max(delta_seconds, 0), side="left"))
    raw = min(raw, len(BUCKET_BOUNDARIES) - 1)
    return raw + 1


def _compute_per_event_session_buckets(
    ts_padded: np.ndarray,
    break_seconds: int = _SESSION_GAP_THRESHOLD_SEC,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-event session bucket triples from [B, L] padded timestamps.

    Returns:
        ``(session_idx_bucket, session_elapsed_bucket, session_prev_gap_bucket)``
        each of shape ``[B, L]`` int64. Padding positions stay 0.
    """
    B, L = ts_padded.shape
    session_bucket = np.zeros((B, L), dtype=np.int64)
    elapsed_bucket = np.zeros((B, L), dtype=np.int64)
    prev_gap_bucket = np.zeros((B, L), dtype=np.int64)

    for b in range(B):
        ts = ts_padded[b]
        valid_idx = np.where(ts > 0)[0]
        if valid_idx.size == 0:
            continue

        sess_idx = 0
        sess_start_ts = int(ts[valid_idx[0]])
        for n, pos in enumerate(valid_idx):
            cur_ts = int(ts[pos])
            gap = 0
            if n > 0:
                prev_ts = int(ts[valid_idx[n - 1]])
                gap = abs(cur_ts - prev_ts)
                if gap > break_seconds:
                    sess_idx += 1
                    sess_start_ts = cur_ts

            session_bucket[b, pos] = _quantize_user_session_index(sess_idx)
            elapsed_bucket[b, pos] = _quantize_session_relative_delta(
                abs(cur_ts - sess_start_ts)
            )
            prev_gap_bucket[b, pos] = (
                _quantize_session_relative_delta(gap) if gap > 0 else 0
            )

    return session_bucket, elapsed_bucket, prev_gap_bucket


# v4 targeted sequence-length features. These are deliberately narrow:
# friend feedback points to domain_a_seq_38 and domain_d_seq_17 as strong
# sequence fields, so we expose their effective lengths and ratios only.
SEQ_LEN_STATS_FID = -4101
SEQ_LEN_STATS_DIM = 6
TARGET_SEQ_LEN_FIELDS = {
    'seq_a': 38,
    'seq_d': 17,
}

# Compact dense side-channel for industrial recency/match signals. These are
# intentionally small numeric features, not new sparse IDs.
# Ported verbatim from teammate's 0.8248 bundle (fromOutside/teammate-LB08248).
TIME_MATCH_LOCAL_TZ_OFFSET_SEC = 8 * 3600
SECONDS_PER_DAY = 24 * 3600
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY
SECONDS_PER_30D = 30 * SECONDS_PER_DAY
TIME_MATCH_CURRENT_DIM = 8
TIME_MATCH_DOMAIN_DIM = 6
TIME_MATCH_PAIR_DIM = 8
TIME_MATCH_LOG_DENOM = math.log1p(int(BUCKET_BOUNDARIES[-1]))

# Public discussion and our probes both point to seq_c:47 as the historical
# item-id field. The current item-id candidate appears as item_int fid 16.
TIME_MATCH_MANUAL_PAIRS: Tuple[Tuple[int, str, int, str], ...] = (
    (16, 'seq_c', 47, 'item16_seqc47'),
)

# ─── EXPΥ port from fromOutside/v29b_e6_sota (LB 0.82451) ─────────────────────
# SAFE_MATCH_V2 30-dim dense block + UTC+8 timezone alignment.
# (Mega3 = TIME_MATCH + UTC+8 + SAFE_MATCH only — no ACTSIG / fid 110 / tail_valid.)
PREMIUM_NULL_COL: str = "item_int_feats_83"
MATCH_PAIRS: List[Tuple[int, int]] = [
    (63, 9),  (64, 9),  (66, 85), (65, 10), (65, 5),
    (63, 83), (66, 8),  (64, 83), (66, 6),  (66, 12),
    (65, 84), (65, 6),
]
MATCH_DENSITY_PAIRS: List[Tuple[int, int]] = MATCH_PAIRS
SAFE_NULL_COUNT_COLS: Tuple[str, ...] = (
    "user_int_feats_101", "user_int_feats_102", "user_int_feats_103",
    "item_int_feats_83", "item_int_feats_84", "item_int_feats_85",
    "user_int_feats_109", "user_int_feats_100", "user_int_feats_99",
    "user_int_feats_86", "user_int_feats_96", "user_int_feats_60",
    "user_int_feats_94", "user_int_feats_108", "item_int_feats_11",
    "user_int_feats_92", "user_int_feats_91", "user_int_feats_104",
    "user_int_feats_95", "user_int_feats_107", "user_int_feats_54",
    "user_int_feats_105", "user_int_feats_97", "user_int_feats_80",
    "user_int_feats_82", "user_int_feats_93", "user_int_feats_58",
    "user_int_feats_59", "user_int_feats_106", "user_int_feats_15",
)
TZ_OFFSET_SEC = 8 * 3600  # alias for TIME_MATCH_LOCAL_TZ_OFFSET_SEC; v29b naming
SAFE_MATCH_TOTAL_DIM = 30
ACTSIG_HISTDIST_DIM = 4  # log1p(label_time - earliest_non_pad_ts) per seq domain

# ─── v11.5 port: synthetic timestamp bucket fid 110 (6 Beijing-day buckets) ──
# Sample-level timestamp discretized into 6 buckets: <Mar-01 / Mar 1 / Mar 2 /
# Mar 3 / Mar 4 / >=Mar-05. Adds 1 column to user_int (vocab=7, 0=padding).
TIMESTAMP_BUCKET_USER_FID = 110
TIMESTAMP_BUCKET_VOCAB_SIZE = 7
TIMESTAMP_BUCKET_BOUNDARIES_UTC = np.array([
    1772294400,  # 2026-03-01 00:00:00 UTC+8
    1772380800,  # 2026-03-02 00:00:00 UTC+8
    1772467200,  # 2026-03-03 00:00:00 UTC+8
    1772553600,  # 2026-03-04 00:00:00 UTC+8
    1772640000,  # 2026-03-05 00:00:00 UTC+8
], dtype=np.int64)


def bucketize_timestamp_utc8_day(timestamps: "npt.NDArray[np.int64]") -> "npt.NDArray[np.int64]":
    """Map Unix-second timestamps to 6 Beijing-day buckets (v11.5 verbatim)."""
    return (
        np.searchsorted(TIMESTAMP_BUCKET_BOUNDARIES_UTC, timestamps, side='right')
        .astype(np.int64, copy=False)
        + 1
    )


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        enable_time_match_features: bool = False,
        time_match_recent_k: int = 32,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.enable_time_match_features = bool(enable_time_match_features)
        self.time_match_recent_k = max(1, int(time_match_recent_k))
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_td = {}
        self._buf_seq_tg = {}
        self._buf_seq_hour = {}
        self._buf_seq_weekday = {}
        self._buf_seq_span = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_td[domain] = np.zeros((B, max_len), dtype=np.float32)
            self._buf_seq_tg[domain] = np.zeros((B, max_len), dtype=np.float32)
            self._buf_seq_hour[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_weekday[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_span[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset, fid))
            offset += dim

        # Fids whose dense values are raw statistics (not pre-trained embeddings)
        # and benefit from log1p compression.
        self._dense_log_fids: set = {62, 63, 64, 65, 66}

        # ─── EXPΥ Mega3: pre-resolve SAFE_MATCH plans (verbatim from v29b) ──
        self._premium_ci: int = self._col_idx.get(PREMIUM_NULL_COL, -1)
        u_off_by_fid: Dict[int, Tuple[int, int]] = {}
        for fid, off, length in self.user_int_schema.entries:
            u_off_by_fid[fid] = (off, length)
        i_off_by_fid: Dict[int, Tuple[int, int]] = {}
        for fid, off, length in self.item_int_schema.entries:
            i_off_by_fid[fid] = (off, length)
        self._match_plan: List[Tuple[int, int, int]] = []
        for u_fid, i_fid in MATCH_PAIRS:
            u_entry = u_off_by_fid.get(u_fid)
            i_entry = i_off_by_fid.get(i_fid)
            if u_entry is None or i_entry is None:
                logging.warning(
                    f"SAFE_MATCH: match pair (u_{u_fid}, i_{i_fid}) "
                    f"missing from schema — this match will always emit 0."
                )
                self._match_plan.append((-1, -1, -1))
            else:
                u_off, u_dim = u_entry
                i_off, _ = i_entry
                self._match_plan.append((u_off, u_dim, i_off))
        self._null_count_cols: List[int] = []
        for cname in SAFE_NULL_COUNT_COLS:
            ci = self._col_idx.get(cname, -1)
            if ci >= 0:
                self._null_count_cols.append(ci)
        logging.info(
            f"Mega3 SAFE_MATCH: premium_ci={self._premium_ci}, "
            f"match_valid={sum(1 for p in self._match_plan if p[0] >= 0)}/{len(self._match_plan)}, "
            f"null_count_cols={len(self._null_count_cols)}/{len(SAFE_NULL_COUNT_COLS)}, "
            f"safe_offset={self._safe_offset}, "
            f"user_dense_total={self.user_dense_schema.total_dim}")

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        self._target_seq_len_slots: Dict[str, Optional[int]] = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            target_slot = None
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
                if TARGET_SEQ_LEN_FIELDS.get(domain) == fid:
                    target_slot = slot
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)
            self._target_seq_len_slots[domain] = target_slot

        # ---- Time/match feature layout (ported from teammate 0.8248) ----
        self._init_time_match_feature_layout()

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")
        logging.info(
            f"TIME_MATCH_FEATURES enabled={self.enable_time_match_features} "
            f"dim={self.time_match_feature_dim} recent_k={self.time_match_recent_k} "
            f"pairs={','.join(pair['name'] for pair in self._time_match_pairs)}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)
        # ─── v11.5 Mega6: append synthetic fid 110 timestamp bucket ──────
        # 1 extra user_int column whose value is bucketize_timestamp_utc8_day(ts).
        if TIMESTAMP_BUCKET_USER_FID in self.user_int_schema._fid_to_entry:
            raise ValueError(
                f"schema.json already contains user_int fid {TIMESTAMP_BUCKET_USER_FID}; "
                f"this synthetic feature would collide with an existing column.")
        self.user_int_schema.add(TIMESTAMP_BUCKET_USER_FID, 1)
        self.user_int_vocab_sizes.append(TIMESTAMP_BUCKET_VOCAB_SIZE)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)
        self._base_user_dense_dim = self.user_dense_schema.total_dim
        self.user_dense_schema.add(SEQ_LEN_STATS_FID, SEQ_LEN_STATS_DIM)

        # ─── EXPΥ Mega3: SAFE_MATCH_V2 30 dims (pseudo fids -9001..-9030) ───
        # Layout (offsets relative to self._safe_offset):
        #   +0       has_premium (PREMIUM_NULL_COL is_null indicator)
        #   +1, +2   ts_hour sin/cos (UTC+8)
        #   +3..+14  12 binary U-I match flags (MATCH_PAIRS)
        #   +15..+26 12 match-density floats
        #   +27      n_null / 30
        #   +28, +29 weekday sin/cos (UTC+8)
        self._safe_offset: int = self.user_dense_schema.total_dim
        self._safe_premium_offset: int = self._safe_offset + 0
        self._safe_ts_hour_sin_offset: int = self._safe_offset + 1
        self._safe_ts_hour_cos_offset: int = self._safe_offset + 2
        self._safe_match_offset: int = self._safe_offset + 3
        self._safe_density_offset: int = self._safe_offset + 15
        self._safe_n_null_offset: int = self._safe_offset + 27
        self._safe_weekday_sin_offset: int = self._safe_offset + 28
        self._safe_weekday_cos_offset: int = self._safe_offset + 29
        for _pseudo_fid in range(-9001, -9001 - SAFE_MATCH_TOTAL_DIM, -1):
            self.user_dense_schema.add(_pseudo_fid, 1)

        # ─── EXPΩ Mega6: ACTSIG hist_dist 4 dims (pseudo fids -1001..-1004) ─
        # Slot d_idx -> log1p(label_time - earliest_non_pad_ts in domain[d_idx])
        # in seconds. Domain order follows self.seq_domains (sorted).
        # T2 already has SEQ_LEN_STATS_6d covering seq_a/seq_d so we skip
        # the redundant log1p(seq_len) 4 dims from v29b's full ACTSIG block.
        self._actsig_histdist_offset: int = self.user_dense_schema.total_dim
        for _pseudo_fid in range(-1001, -1001 - ACTSIG_HISTDIST_DIM, -1):
            self.user_dense_schema.add(_pseudo_fid, 1)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    # ─────────────────────── Time/Match Features (ported) ────────────────────
    # Below methods ported verbatim from teammate 0.8248 bundle
    # (fromOutside/teammate-LB08248/model_train/dataset.py lines 728-990).
    # DO NOT modify logic — they encode their training-time invariants.

    def _init_time_match_feature_layout(self) -> None:
        """Build the compact dense time/match feature layout."""
        self._time_match_pairs: List[Dict[str, Any]] = []
        self.time_match_feature_names: List[str] = []

        if not self.enable_time_match_features:
            self.time_match_feature_dim = 0
            return

        self.time_match_feature_names.extend([
            'cur_hour_sin',
            'cur_hour_cos',
            'cur_week_sin',
            'cur_week_cos',
            'cur_30d_sin',
            'cur_30d_cos',
            'cur_hour_frac',
            'cur_dow_frac',
        ])

        for domain in self.seq_domains:
            self.time_match_feature_names.extend([
                f'{domain}_len_frac',
                f'{domain}_no_history',
                f'{domain}_first_age_log',
                f'{domain}_recent_age_mean_log',
                f'{domain}_recent_age_min_log',
                f'{domain}_recent_age_max_log',
            ])

        seen_pairs = set()

        def _add_pair(item_fid: int, domain: str, seq_fid: int, name: str) -> None:
            key = (int(item_fid), domain, int(seq_fid))
            if key in seen_pairs:
                return
            if item_fid not in self.item_int_schema._fid_to_entry:
                return
            if domain not in self.sideinfo_fids:
                return
            if seq_fid not in self.sideinfo_fids[domain]:
                return
            item_offset, item_length = self.item_int_schema.get_offset_length(
                item_fid)
            seq_slot = self.sideinfo_fids[domain].index(seq_fid)
            self._time_match_pairs.append({
                'name': name,
                'item_fid': int(item_fid),
                'item_offset': int(item_offset),
                'item_length': int(item_length),
                'domain': domain,
                'seq_fid': int(seq_fid),
                'seq_slot': int(seq_slot),
            })
            seen_pairs.add(key)

        for item_fid, domain, seq_fid, name in TIME_MATCH_MANUAL_PAIRS:
            _add_pair(item_fid, domain, seq_fid, name)

        # Schema-safe generic fallback: if a candidate item fid also appears as
        # sequence side info, expose that exact-space match too.
        for item_fid, _, _ in self.item_int_schema.entries:
            if int(item_fid) >= 10000:
                continue
            for domain in self.seq_domains:
                if item_fid in self.sideinfo_fids[domain]:
                    _add_pair(item_fid, domain, item_fid,
                              f'item{item_fid}_{domain}_{item_fid}')

        for pair in self._time_match_pairs:
            prefix = pair['name']
            self.time_match_feature_names.extend([
                f'{prefix}_hit',
                f'{prefix}_recent_hit',
                f'{prefix}_count_log',
                f'{prefix}_recent_count_log',
                f'{prefix}_first_pos_score',
                f'{prefix}_recent_first_pos_score',
                f'{prefix}_match_age_min_log',
                f'{prefix}_recent_match_age_min_log',
            ])

        self.time_match_feature_dim = len(self.time_match_feature_names)

    @staticmethod
    def _log_norm_seconds(values: "npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
        clipped = np.clip(values, 0.0, float(BUCKET_BOUNDARIES[-1]))
        return (np.log1p(clipped) / TIME_MATCH_LOG_DENOM).astype(np.float32)

    def _current_time_features(
        self,
        timestamps: "npt.NDArray[np.int64]",
    ) -> "npt.NDArray[np.float32]":
        local_ts = timestamps.astype(np.int64, copy=False) + TIME_MATCH_LOCAL_TZ_OFFSET_SEC
        day_phase = (local_ts % SECONDS_PER_DAY).astype(np.float32) / float(SECONDS_PER_DAY)
        week_phase = (local_ts % SECONDS_PER_WEEK).astype(np.float32) / float(SECONDS_PER_WEEK)
        month_phase = (local_ts % SECONDS_PER_30D).astype(np.float32) / float(SECONDS_PER_30D)
        dow_frac = ((local_ts // SECONDS_PER_DAY) % 7).astype(np.float32) / 6.0
        return np.stack([
            np.sin(2.0 * np.pi * day_phase),
            np.cos(2.0 * np.pi * day_phase),
            np.sin(2.0 * np.pi * week_phase),
            np.cos(2.0 * np.pi * week_phase),
            np.sin(2.0 * np.pi * month_phase),
            np.cos(2.0 * np.pi * month_phase),
            day_phase,
            dow_frac,
        ], axis=1).astype(np.float32)

    def _domain_time_features(
        self,
        lengths: "npt.NDArray[np.int64]",
        time_diff: "npt.NDArray[np.int64]",
        ts_padded: "npt.NDArray[np.int64]",
        max_len: int,
    ) -> "npt.NDArray[np.float32]":
        B = lengths.shape[0]
        if max_len <= 0:
            return np.zeros((B, TIME_MATCH_DOMAIN_DIM), dtype=np.float32)

        idx = np.arange(max_len).reshape(1, -1)
        valid = (idx < lengths.reshape(-1, 1)) & (ts_padded > 0)
        recent_k = min(self.time_match_recent_k, max_len)
        recent_valid = valid[:, :recent_k]
        recent_diff = time_diff[:, :recent_k].astype(np.float32, copy=False)

        len_frac = (lengths.astype(np.float32) / float(max_len)).clip(0.0, 1.0)
        no_history = (lengths <= 0).astype(np.float32)
        first_valid = valid[:, 0] if max_len > 0 else np.zeros(B, dtype=bool)
        first_age = np.where(first_valid, time_diff[:, 0], 0).astype(np.float32)

        recent_count = recent_valid.sum(axis=1).astype(np.float32)
        denom = np.maximum(recent_count, 1.0)
        recent_sum = (recent_diff * recent_valid.astype(np.float32)).sum(axis=1)
        recent_mean = recent_sum / denom
        large = np.full_like(recent_diff, float(BUCKET_BOUNDARIES[-1]))
        recent_min = np.where(
            recent_count > 0,
            np.where(recent_valid, recent_diff, large).min(axis=1),
            0.0,
        )
        recent_max = np.where(
            recent_count > 0,
            np.where(recent_valid, recent_diff, 0.0).max(axis=1),
            0.0,
        )

        return np.stack([
            len_frac,
            no_history,
            self._log_norm_seconds(first_age),
            self._log_norm_seconds(recent_mean),
            self._log_norm_seconds(recent_min),
            self._log_norm_seconds(recent_max),
        ], axis=1).astype(np.float32)

    def _match_pair_features(
        self,
        pair: Dict[str, Any],
        item_int: "npt.NDArray[np.int64]",
        seq_values: "npt.NDArray[np.int64]",
        time_diff: Optional["npt.NDArray[np.int64]"],
        ts_padded: Optional["npt.NDArray[np.int64]"],
    ) -> "npt.NDArray[np.float32]":
        B, max_len = seq_values.shape
        cand = item_int[:, pair['item_offset']:pair['item_offset'] + pair['item_length']]
        cand_valid = cand > 0
        seq_valid = seq_values > 0
        if cand.shape[1] == 0 or max_len == 0:
            return np.zeros((B, TIME_MATCH_PAIR_DIM), dtype=np.float32)

        match = (
            (seq_values[:, None, :] == cand[:, :, None])
            & cand_valid[:, :, None]
            & seq_valid[:, None, :]
        )
        hit_pos = match.any(axis=1)
        full_count = hit_pos.sum(axis=1).astype(np.float32)
        hit = full_count > 0

        recent_k = min(self.time_match_recent_k, max_len)
        recent_hit_pos = hit_pos[:, :recent_k]
        recent_count = recent_hit_pos.sum(axis=1).astype(np.float32)
        recent_hit = recent_count > 0

        first_pos = np.argmax(hit_pos, axis=1).astype(np.float32)
        recent_first_pos = np.argmax(recent_hit_pos, axis=1).astype(np.float32)
        full_denom = float(max(max_len - 1, 1))
        recent_denom = float(max(recent_k - 1, 1))
        first_score = np.where(hit, 1.0 - first_pos / full_denom, 0.0)
        recent_first_score = np.where(
            recent_hit, 1.0 - recent_first_pos / recent_denom, 0.0)

        if time_diff is not None and ts_padded is not None:
            valid_time = ts_padded > 0
            diff_float = time_diff.astype(np.float32, copy=False)
            large = np.full_like(diff_float, float(BUCKET_BOUNDARIES[-1]))
            match_time = hit_pos & valid_time
            min_age = np.where(
                match_time.any(axis=1),
                np.where(match_time, diff_float, large).min(axis=1),
                0.0,
            )
            recent_match_time = match_time[:, :recent_k]
            recent_diff = diff_float[:, :recent_k]
            recent_large = large[:, :recent_k]
            recent_min_age = np.where(
                recent_match_time.any(axis=1),
                np.where(recent_match_time, recent_diff, recent_large).min(axis=1),
                0.0,
            )
        else:
            min_age = np.zeros(B, dtype=np.float32)
            recent_min_age = np.zeros(B, dtype=np.float32)

        return np.stack([
            hit.astype(np.float32),
            recent_hit.astype(np.float32),
            np.log1p(full_count) / math.log1p(max(max_len, 1)),
            np.log1p(recent_count) / math.log1p(max(recent_k, 1)),
            first_score.astype(np.float32),
            recent_first_score.astype(np.float32),
            self._log_norm_seconds(min_age.astype(np.float32)),
            self._log_norm_seconds(recent_min_age.astype(np.float32)),
        ], axis=1).astype(np.float32)

    def _build_time_match_features(
        self,
        timestamps: "npt.NDArray[np.int64]",
        item_int: "npt.NDArray[np.int64]",
        seq_time_diff: Dict[str, "npt.NDArray[np.int64]"],
        seq_ts_padded: Dict[str, "npt.NDArray[np.int64]"],
    ) -> "npt.NDArray[np.float32]":
        B = timestamps.shape[0]
        if not self.enable_time_match_features:
            return np.zeros((B, 0), dtype=np.float32)

        parts = [self._current_time_features(timestamps)]
        for domain in self.seq_domains:
            parts.append(self._domain_time_features(
                self._buf_seq_lens[domain][:B],
                seq_time_diff.get(domain, np.zeros((B, self._seq_maxlen[domain]), dtype=np.int64)),
                seq_ts_padded.get(domain, np.zeros((B, self._seq_maxlen[domain]), dtype=np.int64)),
                self._seq_maxlen[domain],
            ))

        for pair in self._time_match_pairs:
            domain = pair['domain']
            seq_values = self._buf_seq[domain][:B, pair['seq_slot'], :]
            parts.append(self._match_pair_features(
                pair,
                item_int,
                seq_values,
                seq_time_diff.get(domain),
                seq_ts_padded.get(domain),
            ))

        feats = np.concatenate(parts, axis=1).astype(np.float32)
        if feats.shape[1] != self.time_match_feature_dim:
            raise RuntimeError(
                f"time_match feature dim mismatch: got {feats.shape[1]}, "
                f"expected {self.time_match_feature_dim}")
        return feats

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded
        # ─── v11.5 Mega6: write synthetic fid 110 timestamp bucket ─────────
        ts_bucket_offset, _ = self.user_int_schema.get_offset_length(TIMESTAMP_BUCKET_USER_FID)
        user_int[:, ts_bucket_offset] = bucketize_timestamp_utc8_day(timestamps)

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset, fid in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            if fid in self._dense_log_fids:
                padded = np.sign(padded) * np.log1p(np.abs(padded))
            user_dense[:, offset:offset + dim] = padded

        # ─── EXPΥ Mega3: SAFE_MATCH_V2 30-dim fill (verbatim from v29b) ────
        # 1) has_premium
        if self._premium_ci >= 0:
            prem_col = batch.column(self._premium_ci)
            user_dense[:, self._safe_premium_offset] = (
                prem_col.is_valid().to_numpy(zero_copy_only=False).astype(np.float32)
            )
        else:
            user_dense[:, self._safe_premium_offset] = 0.0

        # 2) ts_hour sin/cos (UTC+8)
        two_pi = np.float32(2.0 * np.pi)
        ts_utc8 = timestamps + np.int64(TZ_OFFSET_SEC)
        ts_hour = (ts_utc8 // 3600 % 24).astype(np.float32)
        ts_ang = two_pi * ts_hour / np.float32(24.0)
        user_dense[:, self._safe_ts_hour_sin_offset] = np.sin(ts_ang)
        user_dense[:, self._safe_ts_hour_cos_offset] = np.cos(ts_ang)

        # 3) 12 binary U-I match + 12 density
        for pi, (u_off, u_dim, i_off) in enumerate(self._match_plan):
            if u_off < 0:
                continue
            u_slice = user_int[:, u_off:u_off + u_dim]
            i_val = item_int[:, i_off:i_off + 1]
            nonpad = (u_slice > 0)
            match = ((u_slice == i_val) & nonpad).any(axis=1)
            user_dense[:, self._safe_match_offset + pi] = match.astype(np.float32)
            if pi < len(MATCH_DENSITY_PAIRS):
                u_len = np.maximum(nonpad.sum(axis=1), 1).astype(np.float32)
                user_dense[:, self._safe_density_offset + pi] = (
                    match.astype(np.float32) / u_len
                )

        # 4) n_null_scalar
        if self._null_count_cols:
            null_counts = np.zeros(B, dtype=np.float32)
            for ci in self._null_count_cols:
                col = batch.column(ci)
                null_counts += col.is_null().to_numpy(
                    zero_copy_only=False
                ).astype(np.float32)
            user_dense[:, self._safe_n_null_offset] = (
                null_counts / np.float32(len(self._null_count_cols))
            )
        else:
            user_dense[:, self._safe_n_null_offset] = 0.0

        # 5) weekday sin/cos (UTC+8). Epoch 1970-01-01 00:00 UTC is Thursday;
        # (day + 3) % 7 stays correct after the +8h shift (Thu->Thu anchor).
        ts_day_utc8 = (ts_utc8 // 86400).astype(np.int64)
        ts_dow = ((ts_day_utc8 + 3) % 7).astype(np.float32)
        dow_ang = two_pi * ts_dow / np.float32(7.0)
        user_dense[:, self._safe_weekday_sin_offset] = np.sin(dow_ang)
        user_dense[:, self._safe_weekday_cos_offset] = np.cos(dow_ang)

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }
        target_seq_lens: Dict[str, "npt.NDArray[np.float32]"] = {}

        # ---- TIME_MATCH residual buffers (ported from teammate 0.8248) ----
        # Collected per-domain inside the loop, consumed by
        # ``_build_time_match_features`` after the loop.
        seq_time_diff: Dict[str, "npt.NDArray[np.int64]"] = {}
        seq_ts_padded: Dict[str, "npt.NDArray[np.int64]"] = {}

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for d_idx, domain in enumerate(self.seq_domains):  # d_idx for ACTSIG hist_dist
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            target_slot = self._target_seq_len_slots.get(domain)
            if target_slot is not None:
                target_seq_lens[domain] = (
                    out[:, target_slot, :] > 0
                ).sum(axis=1).astype(np.float32)

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            # ts_padded / time_diff are lifted out of the if-block so they can
            # also feed TIME_MATCH residual features below (ported from 0.8248).
            ts_padded = np.zeros((B, max_len), dtype=np.int64)
            time_diff = np.zeros((B, max_len), dtype=np.int64)
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)

                # ─── EXPΩ Mega6 ACTSIG hist_dist (verbatim from v29b) ──────
                # hist_dist = label_time - earliest_non_pad_ts (sec) per domain.
                # See v29b dataset.py:940-952 for the rationale around using
                # time_diff.max(axis=1) under padding (pad cells make time_diff
                # = timestamps which is the largest possible diff, so per-row
                # max yields earliest valid event; pure-padding rows give 0).
                hist_dist = time_diff.max(axis=1).astype(np.float32)
                user_dense[:, self._actsig_histdist_offset + d_idx] = np.log1p(hist_dist)

                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets
            else:
                # No ts_fid configured for this domain: hist_dist = 0.
                user_dense[:, self._actsig_histdist_offset + d_idx] = 0.0

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

            # Raw log-compressed time deltas for Temporal Attention Bias.
            time_delta = self._buf_seq_td[domain][:B]
            time_delta[:] = 0.0
            if ts_ci is not None:
                td_float = np.log1p(time_diff.astype(np.float32) / 3600.0)
                td_float[ts_padded == 0] = 0.0
                time_delta[:] = td_float
            result[f'{domain}_time_delta'] = torch.from_numpy(time_delta.copy())

            # Inter-event gap: time between adjacent behaviors in sequence.
            time_gap = self._buf_seq_tg[domain][:B]
            time_gap[:] = 0.0
            if ts_ci is not None:
                # ts_padded is in descending order, so gap[i] = ts[i] - ts[i+1]
                gaps = np.zeros((B, max_len), dtype=np.float32)
                gaps[:, :-1] = (ts_padded[:, :-1] - ts_padded[:, 1:]).astype(np.float32)
                gaps[gaps < 0] = 0.0
                gaps[ts_padded == 0] = 0.0
                # Also zero out the position right before padding starts
                for i in range(B):
                    l = int(lengths[i])
                    if l > 0 and l < max_len:
                        gaps[i, l - 1] = 0.0
                time_gap[:] = np.log1p(gaps / 60.0)  # convert to minutes + log compress
            result[f'{domain}_time_gap'] = torch.from_numpy(time_gap.copy())

            # Calendar-time ids from the event timestamp itself.  0 remains
            # padding; valid hours are 1..24 and valid weekdays are 1..7.
            time_hour = self._buf_seq_hour[domain][:B]
            time_weekday = self._buf_seq_weekday[domain][:B]
            time_hour[:] = 0
            time_weekday[:] = 0
            if ts_ci is not None:
                valid_ts = ts_padded > 0
                # ─── EXPΥ Mega3 UTC+8 shift (community +110~130bp) ────────
                # Test 是 Mon CST 7-9 = UTC 23 ~ Tue 01. 原 T2 用 UTC 算 hour
                # → train/test hour bucket 错位。+8h 让 hour/weekday 对齐 CST,
                # 跟 SAFE_MATCH ts_hour / weekday sin/cos 用同一个 offset。
                ts_padded_utc8 = ts_padded + np.int64(TZ_OFFSET_SEC)
                hour_ids = ((ts_padded_utc8 // 3600) % 24 + 1).astype(np.int64)
                weekday_ids = ((ts_padded_utc8 // 86400) % 7 + 1).astype(np.int64)
                hour_ids[~valid_ts] = 0
                weekday_ids[~valid_ts] = 0
                time_hour[:] = hour_ids
                time_weekday[:] = weekday_ids
            result[f'{domain}_time_hour'] = torch.from_numpy(time_hour.copy())
            result[f'{domain}_time_weekday'] = torch.from_numpy(time_weekday.copy())

            # Discrete inter-event gap bucket, complementary to the continuous
            # log-compressed time_gap above.
            time_span = self._buf_seq_span[domain][:B]
            time_span[:] = 0
            if ts_ci is not None:
                raw_span_buckets = np.searchsorted(
                    TIME_SPAN_BUCKET_BOUNDARIES,
                    gaps.astype(np.int64).ravel(),
                )
                span_ids = raw_span_buckets.reshape(B, max_len) + 1
                span_ids[gaps <= 0] = 0
                time_span[:] = span_ids
            result[f'{domain}_time_span_bucket'] = torch.from_numpy(time_span.copy())

            # ─── Mega8: per-seq-item raw Unix seconds for Fourier encoder ────
            # Reuses ts_padded already computed above. Pad positions stay 0
            # → model.py masks them out (* valid) before adding to token_emb.
            result[f'{domain}_seq_time_ts'] = torch.from_numpy(ts_padded.copy())

            # ─── EXPΩ Patch 1: per-event unified sincos 8d × 4 periods ───────
            # Replaces SeqFourierTimeEncoder. UTC+8 aligned (same as hour_ids).
            # Pad cells (ts == 0) are forced to zero. Always populated; the
            # model side toggles whether to consume it.
            event_sincos_8d = _compute_per_event_unified_sincos(ts_padded)
            result[f'{domain}_per_event_unified_sincos'] = torch.from_numpy(
                event_sincos_8d.copy())

            # ─── EXPΩ Patch 2: best2 session pseudo-fid bucket triples ───────
            # session_idx_bucket (0..NUM_USER_SESSION_BUCKETS-1),
            # session_elapsed_bucket / session_prev_gap_bucket (reuse
            # NUM_TIME_BUCKETS=65 vocab). Pad positions stay 0.
            sess_idx_b, sess_elapsed_b, sess_prev_gap_b = (
                _compute_per_event_session_buckets(ts_padded)
            )
            result[f'{domain}_session_idx_bucket'] = torch.from_numpy(
                sess_idx_b.copy())
            result[f'{domain}_session_elapsed_bucket'] = torch.from_numpy(
                sess_elapsed_b.copy())
            result[f'{domain}_session_prev_gap_bucket'] = torch.from_numpy(
                sess_prev_gap_b.copy())

            # Stash for TIME_MATCH residual features (ported from 0.8248).
            seq_time_diff[domain] = time_diff
            seq_ts_padded[domain] = ts_padded

        # ---- TIME_MATCH residual (ported verbatim from teammate 0.8248) ----
        time_match_feats = self._build_time_match_features(
            timestamps=timestamps,
            item_int=item_int,
            seq_time_diff=seq_time_diff,
            seq_ts_padded=seq_ts_padded,
        )
        result['time_match_feats'] = torch.from_numpy(time_match_feats.copy())

        len_a = target_seq_lens.get('seq_a', np.zeros(B, dtype=np.float32))
        len_d = target_seq_lens.get('seq_d', np.zeros(B, dtype=np.float32))
        max_a = float(max(1, self._seq_maxlen.get('seq_a', 1)))
        max_d = float(max(1, self._seq_maxlen.get('seq_d', 1)))
        log_a = np.log1p(len_a)
        log_d = np.log1p(len_d)
        seq_len_stats = np.stack([
            log_a,
            log_d,
            len_a / max_a,
            len_d / max_d,
            log_a - log_d,
            np.log((len_a + 1.0) / (len_d + 1.0)),
        ], axis=1).astype(np.float32)
        user_dense[:, self._base_user_dense_dim:self._base_user_dense_dim + SEQ_LEN_STATS_DIM] = seq_len_stats
        result['user_dense_feats'] = torch.from_numpy(user_dense.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    enable_time_match_features: bool = False,
    time_match_recent_k: int = 32,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups (in the file order returned by ``glob``).

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        enable_time_match_features=enable_time_match_features,
        time_match_recent_k=time_match_recent_k,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
        enable_time_match_features=enable_time_match_features,
        time_match_recent_k=time_match_recent_k,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset

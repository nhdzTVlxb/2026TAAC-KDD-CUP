"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    seq_time_deltas: dict   # {domain: tensor [B, L]} log1p(dt/3600)
    seq_time_gaps: dict     # {domain: tensor [B, L]} log1p(gap_minutes)
    seq_time_hours: dict    # {domain: tensor [B, L]} 0=padding, 1..24
    seq_time_weekdays: dict # {domain: tensor [B, L]} 0=padding, 1..7
    seq_time_span_buckets: dict  # {domain: tensor [B, L]} inter-event gap buckets
    seq_time_ts: dict       # {domain: tensor [B, L]} raw UTC Unix seconds (Mega8/Fourier)
    # ─── EXPΩ Patch 1: per-event unified sincos 8d (UTC+8) ──────────────────
    seq_per_event_unified_sincos: dict  # {domain: tensor [B, L, 8]} float32
    # ─── EXPΩ Patch 2: best2 session pseudo-fid bucket triples ──────────────
    seq_session_idx: dict          # {domain: tensor [B, L]} 0..9 log-bucketed session count
    seq_session_elapsed: dict      # {domain: tensor [B, L]} 0..NUM_TIME_BUCKETS elapsed-since-session-start
    seq_session_prev_gap: dict     # {domain: tensor [B, L]} 0..NUM_TIME_BUCKETS gap-from-previous-event
    time_match_feats: torch.Tensor  # (B, time_match_dim) dense residual feats (ported 0.8248)
    timestamp: torch.Tensor  # (B,) sample-level UTC Unix seconds (Mega9 Temporal Token)


class SeqFourierTimeEncoder(nn.Module):
    """3-period × 8-frequency Fourier time encoder (ported from teammate-0.825).

    Encodes per-seq-item Unix timestamps along 3 cyclical periods (day / week /
    year), each at 8 different frequencies, all UTC+8 aligned. Outputs a
    `d_model`-dim residual to add onto token embeddings.
    """

    def __init__(self, d_model: int, num_freqs: int = 8) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        total_dim = num_freqs * 2 * 3  # 3 periods × sin/cos × num_freqs = 48
        self.proj = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def _fourier_channel(self, ts: torch.Tensor, period_seconds: float, offset_seconds: float = 0) -> torch.Tensor:
        t = ((ts + offset_seconds) % period_seconds) / period_seconds
        return t

    def _fourier_encode(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(1, self.num_freqs + 1, device=t.device, dtype=t.dtype)
        angles = 2 * math.pi * t.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, ts_seq: torch.Tensor) -> torch.Tensor:
        # Hour of day (period = 1 day = 86400s), UTC+8 offset.
        t_hour = self._fourier_channel(ts_seq, period_seconds=86400.0, offset_seconds=8 * 3600.0)
        # Day of week (period = 7 days = 604800s), anchor offset 288000.
        t_dow = self._fourier_channel(ts_seq, period_seconds=7 * 86400.0, offset_seconds=288000.0)
        # Time of year (period ≈ 1 tropical year = 31556952s), UTC+8 offset.
        t_month = self._fourier_channel(ts_seq, period_seconds=31556952.0, offset_seconds=8 * 3600.0)

        fourier_hour = self._fourier_encode(t_hour)
        fourier_dow = self._fourier_encode(t_dow)
        fourier_month = self._fourier_encode(t_month)

        fourier_all = torch.cat([fourier_hour, fourier_dow, fourier_month], dim=-1)
        fourier_all = fourier_all.to(dtype=self.proj[0].weight.dtype)
        return self.proj(fourier_all)


class PerEventUnifiedSincosResidual(nn.Module):
    """EXPΩ Patch 1: per-event 8d unified sincos residual.

    Consumes the dataset-precomputed ``[B, L, 8]`` sin/cos features (hour/dow/
    dom/month, UTC+8 aligned) and projects them to ``d_model`` via
    ``Linear(8, d_model) → LayerNorm → SiLU → Linear(d_model, d_model)`` with
    the final Linear zero-initialized so the residual starts at zero
    (bit-identical to Mega9 base when no signal is wired in).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(8, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Zero-init the final Linear so the residual contributes exactly 0 at
        # step 0 — mirrors the TIME_MATCH / Temporal NS Token init pattern.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, sincos_8d: torch.Tensor) -> torch.Tensor:
        return self.proj(sincos_8d)


USER_DENSE_PAIR_START = 256
USER_DENSE_PAIR_END = 568


# EXPΩ Patch 2: Session pseudo-fid vocab size (mirrors dataset constant).
# 9 = pad(0) + 8 explicit session-index buckets defined in dataset.py's
# _USER_SESSION_BUCKET_EDGES. Kept here as a small literal so model.py does
# not have to import dataset (avoids circular pulls during state_dict load).
_NUM_USER_SESSION_BUCKETS_DEFAULT = 9


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class HashEmbedding(nn.Module):
    """Multi-hash embedding for ultra-high cardinality features.

    Uses multiple hash functions to reduce collision impact. IDs are hashed
    into fixed-size buckets via different primes, looked up independently,
    then averaged.  id=0 is treated as padding and always returns zeros.
    """

    HASH_PRIMES = [999983, 999979, 999961]

    def __init__(
        self,
        num_buckets: int = 100000,
        emb_dim: int = 64,
        num_hashes: int = 2,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.num_hashes = num_hashes
        self.embs = nn.ModuleList([
            nn.Embedding(num_buckets, emb_dim, padding_idx=0)
            for _ in range(num_hashes)
        ])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pad_mask = (ids == 0)
        result = self.embs[0](ids.abs() % (self.num_buckets - 1) + 1)
        for i in range(1, self.num_hashes):
            h = (ids.abs() * self.HASH_PRIMES[i - 1]) % (self.num_buckets - 1) + 1
            result = result + self.embs[i](h)
        result = result / self.num_hashes
        result[pad_mask] = 0.0
        return result


class DINTargetAttention(nn.Module):
    """DIN-style target-aware attention.

    Computes attention scores via MLP(concat(q, k, q-k, q*k)) and returns
    a weighted sum of key vectors.
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.attn_mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = keys.shape
        q = query.unsqueeze(1).expand(-1, L, -1)
        attn_input = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        scores = self.attn_mlp(attn_input).squeeze(-1)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        return (keys * weights.unsqueeze(-1)).sum(dim=1)


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        temporal_bias: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.
            temporal_bias: (B, num_heads, 1, Lk) or (B, num_heads, Lq, Lk),
                additive bias from TemporalBias.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        # When temporal_bias is present, use float mask to allow additive bias.
        use_float_mask = temporal_bias is not None
        sdpa_attn_mask = None

        if key_padding_mask is not None:
            if use_float_mask:
                # Float mask: 0 = attend, -inf = mask out
                float_mask = torch.zeros(B, 1, 1, Lk, device=query.device, dtype=query.dtype)
                float_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                sdpa_attn_mask = float_mask.expand(B, self.num_heads, Lq, Lk)
            else:
                sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
                sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            bool_attn = (attn_mask == 0)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if use_float_mask:
                float_attn = torch.zeros(B, self.num_heads, Lq, Lk, device=query.device, dtype=query.dtype)
                float_attn.masked_fill_(~bool_attn, float('-inf'))
                if sdpa_attn_mask is not None:
                    sdpa_attn_mask = sdpa_attn_mask + float_attn
                else:
                    sdpa_attn_mask = float_attn
            else:
                if sdpa_attn_mask is not None:
                    sdpa_attn_mask = sdpa_attn_mask & bool_attn
                else:
                    sdpa_attn_mask = bool_attn

        # Add temporal bias (additive, per-head decay)
        if temporal_bias is not None:
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask + temporal_bias.expand(B, self.num_heads, Lq, Lk)
            else:
                sdpa_attn_mask = temporal_bias.expand(B, self.num_heads, Lq, Lk)

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        temporal_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.
            temporal_bias: (B, num_heads, 1, Lk), additive time-decay bias.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            temporal_bias=temporal_bias,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(TargetEmb, F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model

        global_info_dim = (num_ns + 2) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        target_emb: torch.Tensor,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            target_emb: (B, D), pooled candidate item representation.
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(target_emb, NS_flat, seq_pooled_i)
            global_info = torch.cat([target_emb, ns_flat, seq_pooled], dim=-1)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope/time-delta and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask,
            time_delta).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask, kwargs.get('time_delta')


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None,
        temporal_bias_module: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.
            time_delta: (B, L), log1p(dt_hours) recency per sequence token.
            temporal_bias_module: Module that maps pairwise time distances to
                an additive attention bias.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask,
            time_delta).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        temporal_bias = None
        if temporal_bias_module is not None and time_delta is not None:
            pairwise_delta = pairwise_log_time_delta(time_delta, time_delta)
            temporal_bias = temporal_bias_module(pairwise_delta)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            temporal_bias=temporal_bias,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask, time_delta

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask, new_time_delta) so downstream can
    update the mask and aligned recency values.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        time_delta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
            top_k_time_delta: (B, top_k), gathered recency values when
                time_delta is provided.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        top_k_time_delta = None
        if time_delta is not None:
            top_k_time_delta = torch.gather(time_delta, dim=1, index=indices)
            top_k_time_delta = top_k_time_delta * (~new_padding_mask).to(time_delta.dtype)

        return top_k_tokens, new_padding_mask, position_indices, top_k_time_delta

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None,
        temporal_bias_module: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.
            time_delta: (B, L), log1p(dt_hours) recency per sequence token.
            temporal_bias_module: Module that maps pairwise time distances to
                an additive attention bias.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
            new_time_delta: (B, top_k) when the sequence is compressed,
                otherwise the input time_delta.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices, new_time_delta = self._gather_top_k(
                x, key_padding_mask, time_delta)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            temporal_bias = None
            if temporal_bias_module is not None and time_delta is not None and new_time_delta is not None:
                pairwise_delta = pairwise_log_time_delta(new_time_delta, time_delta)
                temporal_bias = temporal_bias_module(pairwise_delta)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
                temporal_bias=temporal_bias,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask
            new_time_delta = time_delta

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            temporal_bias = None
            if temporal_bias_module is not None and time_delta is not None:
                pairwise_delta = pairwise_log_time_delta(time_delta, time_delta)
                temporal_bias = temporal_bias_module(pairwise_delta)

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                temporal_bias=temporal_bias,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask, new_time_delta


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full',
        use_ns_self_attn: bool = False,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode
        )

        self.ns_self_attn = NSSelfAttention(d_model, num_heads, dropout) if use_ns_self_attn else None

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
        seq_time_deltas: Optional[List[torch.Tensor]] = None,
        query_temporal_bias: Optional[nn.Module] = None,
        seq_temporal_bias: Optional[nn.Module] = None,
    ) -> Tuple[list, torch.Tensor, list, list, Optional[List[torch.Tensor]]]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.
            seq_time_deltas: List of (B, L_i) log1p(dt_hours) tensors.
            query_temporal_bias: Module for query-to-sequence temporal decay.
            seq_temporal_bias: Module for sequence-evolution token-token decay.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            next_masks is a list of (B, L_i') updated padding masks, and
            next_time_deltas mirrors updated sequence lengths when provided.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        next_time_deltas = [] if seq_time_deltas is not None else None
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            td = seq_time_deltas[i] if seq_time_deltas is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
                time_delta=td,
                temporal_bias_module=seq_temporal_bias,
            )
            if len(result) == 3:
                next_seq_i, mask_i, next_td_i = result
            else:
                next_seq_i, mask_i = result
                next_td_i = td
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)
            if next_time_deltas is not None:
                next_time_deltas.append(next_td_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            tb = None
            if query_temporal_bias is not None and next_time_deltas is not None:
                tb = query_temporal_bias(next_time_deltas[i])
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs, temporal_bias=tb,
            )
            decoded_qs.append(decoded_q_i)

        # 2.5. NS Self-Attention (optional feature crossing)
        if self.ns_self_attn is not None:
            ns_tokens = self.ns_self_attn(ns_tokens)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks, next_time_deltas


# ═══════════════════════════════════════════════════════════════════════════════
# NS Self-Attention (Feature Crossing)
# ═══════════════════════════════════════════════════════════════════════════════


class NSSelfAttention(nn.Module):
    """Self-attention among NS tokens for feature crossing.

    NS tokens have no positional ordering, so standard multi-head attention
    (without RoPE) is used.  Pre-LN residual style, consistent with
    CrossAttention(ln_mode='pre').
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

    def forward(self, ns_tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            ns_tokens: (B, Nns, D)
        Returns:
            (B, Nns, D) with residual connection.
        """
        residual = ns_tokens
        x = self.norm(ns_tokens)
        x, _ = self.attn(x, x, x, need_weights=False)
        return residual + x


class TemporalBias(nn.Module):
    """Per-head learnable time-decay bias for attention scores.

    Given log1p-compressed time deltas, produces an additive bias. A 2D input
    (B, Lk) is treated as query-to-history recency and returns
    (B, num_heads, 1, Lk). A 3D input (B, Lq, Lk) is treated as pairwise
    token-token distance and returns (B, num_heads, Lq, Lk).
    Each head learns its own decay rate, enabling multi-scale temporal focus.
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_heads,), 0.1))

    def forward(self, time_delta: torch.Tensor) -> torch.Tensor:
        """Args:
            time_delta: (B, Lk) or (B, Lq, Lk) log1p(dt_hours).
        Returns:
            Additive attention bias with a per-head temporal decay.
        """
        if time_delta.dim() == 2:
            bias = -self.alpha.abs().view(1, -1, 1) * time_delta.unsqueeze(1)
            return bias.unsqueeze(2)
        if time_delta.dim() == 3:
            return -self.alpha.abs().view(1, -1, 1, 1) * time_delta.unsqueeze(1)
        raise ValueError(
            f"TemporalBias expects a 2D or 3D tensor, got shape {tuple(time_delta.shape)}"
        )


def pairwise_log_time_delta(
    q_time_delta: torch.Tensor,
    k_time_delta: torch.Tensor,
) -> torch.Tensor:
    """Builds log1p-compressed token-token time distances from recency values.

    q_time_delta and k_time_delta are log1p(current_ts - event_ts in hours).
    The pairwise event distance is recoverable because both are measured from
    the same current timestamp.
    """
    q_hours = torch.expm1(q_time_delta.clamp_min(0.0)).clamp_min(0.0)
    k_hours = torch.expm1(k_time_delta.clamp_min(0.0)).clamp_min(0.0)
    return torch.log1p((q_hours.unsqueeze(-1) - k_hours.unsqueeze(-2)).abs())


# ═══════════════════════════════════════════════════════════════════════════════
# SE-Net Feature Gating (FiBiNet-style)
# ═══════════════════════════════════════════════════════════════════════════════


class SENetGating(nn.Module):
    """Squeeze-and-Excitation gating for NS tokens.

    Learns per-token importance weights via a two-layer bottleneck MLP
    applied to the mean-pooled (squeezed) representation of each token.
    """

    def __init__(self, num_tokens: int, d_model: int, reduction: int = 2) -> None:
        super().__init__()
        mid = max(num_tokens // reduction, 4)
        self.gate = nn.Sequential(
            nn.Linear(num_tokens, mid),
            nn.ReLU(),
            nn.Linear(mid, num_tokens),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies SE gating.

        Args:
            x: (B, T, D) NS tokens.

        Returns:
            Gated tokens of shape (B, T, D).
        """
        w = x.mean(dim=-1)        # (B, T) — squeeze
        w = self.gate(w)           # (B, T) — excitation
        return x * w.unsqueeze(-1) # (B, T, D) — scale


# ═══════════════════════════════════════════════════════════════════════════════
# DCN-v2 Cross Network
# ═══════════════════════════════════════════════════════════════════════════════


class CrossNet(nn.Module):
    """DCN-v2 cross network with per-layer dropout and a learnable output gate.

    Each cross layer: x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
    After all layers: output = sigmoid(gate) * cross_out + (1 - sigmoid(gate)) * x_0
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
        low_rank: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.low_rank = low_rank

        if low_rank > 0:
            self.U = nn.ParameterList([
                nn.Parameter(torch.empty(in_features, low_rank))
                for _ in range(num_layers)
            ])
            self.V = nn.ParameterList([
                nn.Parameter(torch.empty(low_rank, in_features))
                for _ in range(num_layers)
            ])
            for u, v in zip(self.U, self.V):
                nn.init.xavier_normal_(u)
                nn.init.xavier_normal_(v)
        else:
            self.W = nn.ParameterList([
                nn.Parameter(torch.empty(in_features, in_features))
                for _ in range(num_layers)
            ])
            for w in self.W:
                nn.init.xavier_normal_(w)

        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(in_features))
            for _ in range(num_layers)
        ])

        self.drop = nn.Dropout(dropout) if dropout > 0 else None

        # Learnable gate initialized to -2 so sigmoid(-2)≈0.12,
        # meaning the model starts close to baseline and gradually
        # learns how much cross information to blend in.
        self.gate = nn.Parameter(torch.full((in_features,), -2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for i in range(self.num_layers):
            if self.low_rank > 0:
                vx = xl @ self.V[i].t()
                uvx = vx @ self.U[i].t()
            else:
                uvx = xl @ self.W[i].t()
            xl = x0 * (uvx + self.bias[i]) + xl
            if self.drop is not None:
                xl = self.drop(xl)
        g = torch.sigmoid(self.gate)
        return g * xl + (1 - g) * x0


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0,
                 hash_bucket_size: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Hash embeddings for high-cardinality features that were skipped
        self._hash_embs = nn.ModuleDict()
        if hash_bucket_size > 0:
            for i, (vs, offset, length) in enumerate(feature_specs):
                if self._emb_index[i] == -1 and int(vs) > 0:
                    self._hash_embs[str(i)] = HashEmbedding(hash_bucket_size, emb_dim)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    hash_key = str(fid_idx)
                    if hash_key in self._hash_embs:
                        emb_layer = self._hash_embs[hash_key]
                        if length == 1:
                            fid_emb = emb_layer(int_feats[:, offset].long())
                        else:
                            vals = int_feats[:, offset:offset + length].long()
                            emb_all = emb_layer(vals)
                            mask = (vals != 0).float().unsqueeze(-1)
                            count = mask.sum(dim=1).clamp(min=1)
                            fid_emb = (emb_all * mask).sum(dim=1) / count
                    else:
                        fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
        hash_bucket_size: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
            hash_bucket_size: Bucket count for hash embeddings on skipped features.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Hash embeddings for high-cardinality features that were skipped
        self._hash_embs = nn.ModuleDict()
        if hash_bucket_size > 0:
            for i, (vs, offset, length) in enumerate(feature_specs):
                if self._emb_index[i] == -1 and int(vs) > 0:
                    self._hash_embs[str(i)] = HashEmbedding(hash_bucket_size, emb_dim)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    hash_key = str(fid_idx)
                    if hash_key in self._hash_embs:
                        emb_layer = self._hash_embs[hash_key]
                        if length == 1:
                            fid_emb = emb_layer(int_feats[:, offset].long())
                        else:
                            vals = int_feats[:, offset:offset + length].long()
                            emb_all = emb_layer(vals)
                            mask = (vals != 0).float().unsqueeze(-1)
                            count = mask.sum(dim=1).clamp(min=1)
                            fid_emb = (emb_all * mask).sum(dim=1) / count
                    else:
                        fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        ids = int_feats[:, offset].long()
                        # ─── Mega7: High-Cardinality ID Mask Cold Start Exercise ────
                        # Ported verbatim from teammate-0.825 (model.py:1477-1481).
                        # vocab > 10000 → train-time random mask to padding.
                        if self.training and getattr(self, 'mask_prob', 0.0) > 0 and int(vs) > 10000:
                            rand_tensor = torch.rand_like(ids, dtype=torch.float)
                            id_mask = (rand_tensor < self.mask_prob) & (ids > 0)
                            ids = ids.clone()
                            ids[id_mask] = 0
                        fid_emb = emb_layer(ids)
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        # ─── Mega7: High-Cardinality ID Mask (multi-value path) ───
                        if self.training and getattr(self, 'mask_prob', 0.0) > 0 and int(vs) > 10000:
                            rand_tensor = torch.rand_like(vals, dtype=torch.float)
                            id_mask = (rand_tensor < self.mask_prob) & (vals > 0)
                            vals = vals.clone()
                            vals[id_mask] = 0
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class UserSparseDensePairResidual(nn.Module):
    """Builds a residual vector from paired user sparse ids and dense values."""

    def __init__(
        self,
        pair_specs: List[Tuple[int, int, int, int, int]],
        emb_dim: int,
        d_model: int,
    ) -> None:
        """Initializes paired sparse/dense residual features.

        Args:
            pair_specs: [(fid, vocab_size, int_offset, dense_offset, length), ...].
            emb_dim: Embedding dimension per fid.
            d_model: Output residual dimension.
        """
        super().__init__()
        self.pair_specs = pair_specs
        self.emb_dim = emb_dim
        self.embs = nn.ModuleList([
            nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
            for _, vs, _, _, _ in pair_specs
        ])
        self.proj = nn.Sequential(
            nn.Linear(len(pair_specs) * emb_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        user_int_feats: torch.Tensor,
        user_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        pooled = []
        for emb, (_, _, int_offset, dense_offset, length) in zip(self.embs, self.pair_specs):
            ids = user_int_feats[:, int_offset:int_offset + length].long()
            values = user_dense_feats[:, dense_offset:dense_offset + length]
            weights = torch.log1p(values.abs()) * values.sign()
            mask = (ids != 0).to(values.dtype)

            emb_all = emb(ids)
            weighted = emb_all * weights.unsqueeze(-1) * mask.unsqueeze(-1)
            count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled.append(weighted.sum(dim=1) / count)

        return F.silu(self.proj(torch.cat(pooled, dim=-1)))


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        num_time_span_buckets: int = 0,
        use_calendar_time: bool = True,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        # DCN-v2 cross network
        num_cross_layers: int = 0,
        cross_low_rank: int = 0,
        cross_dropout: float = 0.1,
        # SE-Net feature gating
        use_se_net: bool = False,
        # Hash embedding for ultra-high cardinality features
        hash_bucket_size: int = 0,
        # Target-aware attention
        use_target_attention: bool = False,
        # NS self-attention (feature crossing inside blocks)
        use_ns_self_attn: bool = False,
        # NS output fusion (fuse refined NS tokens into output)
        use_ns_output_fusion: bool = False,
        # Temporal Attention Bias (per-head time decay in cross-attention)
        use_temporal_bias: bool = False,
        # Inter-event time gap embedding
        use_time_gap: bool = False,
        user_sparse_dense_pair_specs: Optional[List[Tuple[int, int, int, int, int]]] = None,
        # TIME_MATCH residual (ported verbatim from teammate 0.8248 bundle)
        time_match_dim: int = 0,
        time_match_scale: float = 0.1,
        # ─── EXPΩ Patch 1: per-event unified sincos 8d × 4 periods ──────────
        use_per_event_unified_sincos: bool = True,
        # Gate the Mega8 SeqFourierTimeEncoder allocation. Default OFF in EXPΩ
        # because Patch 1 supersedes it; flipping this back to True turns
        # SeqFourier back on for ablation.
        use_seq_fourier_encoder: bool = False,
        # ─── EXPΩ Patch 2: best2 session pseudo-fid bucket triples ──────────
        use_session_pseudo_fids: bool = True,
        # ─── EXPΩ Patch 3: multi-token user_dense + per-token Temporal residual
        user_dense_token_count: int = 2,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.num_time_span_buckets = num_time_span_buckets
        self.use_calendar_time = use_calendar_time
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.hash_bucket_size = hash_bucket_size
        self.use_target_attention = use_target_attention
        self.use_temporal_bias = use_temporal_bias
        self.use_time_gap = use_time_gap
        self.time_match_dim = max(0, int(time_match_dim))
        self.time_match_scale_init = float(time_match_scale)
        # EXPΩ flag stashes — referenced by forward/predict + later init blocks.
        self.use_per_event_unified_sincos = bool(use_per_event_unified_sincos)
        self.use_seq_fourier_encoder = bool(use_seq_fourier_encoder)
        self.use_session_pseudo_fids = bool(use_session_pseudo_fids)
        self.user_dense_token_count = max(1, int(user_dense_token_count))
        self.user_dense_proj_dim = user_dense_dim
        if user_dense_dim >= USER_DENSE_PAIR_END:
            self.user_dense_proj_dim = (
                user_dense_dim - (USER_DENSE_PAIR_END - USER_DENSE_PAIR_START)
            )

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
                hash_bucket_size=hash_bucket_size,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
                hash_bucket_size=hash_bucket_size,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                hash_bucket_size=hash_bucket_size,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                hash_bucket_size=hash_bucket_size,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection (if available).
        # ─── EXPΩ Patch 3: multi-token user_dense ─────────────────────────────
        # Split the effective user_dense_proj_dim into N=user_dense_token_count
        # contiguous chunks; project each to d_model with its own Linear. The
        # first (N - remainder) chunks get base_size, the rest get base_size+1
        # so |chunk_sizes| sums to user_dense_proj_dim exactly.
        self.has_user_dense = user_dense_dim > 0
        self.user_sparse_dense_pair_residual = None
        if self.has_user_dense:
            base = self.user_dense_proj_dim // self.user_dense_token_count
            remainder = self.user_dense_proj_dim % self.user_dense_token_count
            chunk_sizes: List[int] = [
                base + (1 if i >= (self.user_dense_token_count - remainder) else 0)
                for i in range(self.user_dense_token_count)
            ]
            self.user_dense_chunk_sizes = chunk_sizes
            # ModuleList of N independent Linear(chunk_size_i -> d_model).
            # Mirrors Mega9's single-Linear allocation (no LayerNorm here;
            # _make_user_dense_token applies SiLU after the linear, same as
            # Mega9 base which had Linear+LayerNorm but used silu on output).
            self.user_dense_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(cs, d_model),
                    nn.LayerNorm(d_model),
                )
                for cs in chunk_sizes
            ])
            if user_sparse_dense_pair_specs:
                self.user_sparse_dense_pair_residual = UserSparseDensePairResidual(
                    pair_specs=user_sparse_dense_pair_specs,
                    emb_dim=emb_dim,
                    d_model=d_model,
                )
            logging.info(
                f"EXPΩ user_dense multi-token: count={self.user_dense_token_count}, "
                f"chunk_sizes={chunk_sizes}, total_dim={self.user_dense_proj_dim}")
        else:
            self.user_dense_chunk_sizes = []
            self.user_dense_proj = nn.ModuleList()

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Total NS token count. user_dense contributes user_dense_token_count
        # tokens (one per chunk projection) when has_user_dense, else 0.
        user_dense_ns_tokens = (
            self.user_dense_token_count if self.has_user_dense else 0
        )
        self.num_ns = (num_user_ns + user_dense_ns_tokens
                       + num_item_ns + (1 if self.has_item_dense else 0))

        # ================== SE-Net Feature Gating (optional) ==================
        self.use_se_net = use_se_net
        if use_se_net:
            self.se_gate = SENetGating(self.num_ns, d_model)
            logging.info(f"SENetGating enabled: num_tokens={self.num_ns}, d_model={d_model}")

        # ================== DCN-v2 Cross Network (optional) ==================
        self.num_cross_layers = num_cross_layers
        if num_cross_layers > 0:
            cross_dim = self.num_ns * d_model
            self.cross_net = CrossNet(
                in_features=cross_dim,
                num_layers=num_cross_layers,
                low_rank=cross_low_rank,
                dropout=cross_dropout,
            )
            self.cross_ln = nn.LayerNorm(d_model)
            logging.info(f"CrossNet: {num_cross_layers} layers, dim={cross_dim}, "
                         f"low_rank={cross_low_rank}, dropout={cross_dropout}")

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== Hash Embeddings for Sequences ==================
        self._seq_hash_embs = nn.ModuleDict()
        if hash_bucket_size > 0:
            for domain in self.seq_domains:
                hash_embs = nn.ModuleDict()
                vs_list = seq_vocab_sizes[domain]
                idx_map = self._seq_emb_index[domain]
                for i, vs in enumerate(vs_list):
                    if idx_map[i] == -1 and int(vs) > 0:
                        hash_embs[str(i)] = HashEmbedding(hash_bucket_size, emb_dim)
                if len(hash_embs) > 0:
                    self._seq_hash_embs[domain] = hash_embs
            n_hash = sum(len(v) for v in self._seq_hash_embs.values())
            if n_hash > 0:
                logging.info(f"HashEmbedding: {n_hash} features across "
                             f"{len(self._seq_hash_embs)} seq domains, "
                             f"bucket_size={hash_bucket_size}")

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== Calendar / Inter-event Time Feature Embeddings ==================
        # ─── EXPΩ Patch 2: best2 session pseudo-fid embeddings ──────────────
        # Always declare the attributes so downstream `is not None` checks
        # remain safe even when use_calendar_time is False. Real allocation
        # only happens when both flags are True.
        self.seq_session_idx_embedding = None
        self.seq_session_elapsed_embedding = None
        self.seq_session_prev_gap_embedding = None
        if use_calendar_time:
            self.hour_embedding = nn.Embedding(25, d_model, padding_idx=0)
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
            self.cyclical_time_proj = nn.Linear(4, d_model, bias=False)
            # Each event tagged with (a) session index (9 buckets incl. pad),
            # (b) elapsed-since-session-start bucket (65 = NUM_TIME_BUCKETS),
            # (c) gap from previous event bucket (65). Allocation gated by
            # use_session_pseudo_fids so we don't bloat the state dict when
            # the patch is toggled off via CLI.
            if self.use_session_pseudo_fids:
                self.seq_session_idx_embedding = nn.Embedding(
                    _NUM_USER_SESSION_BUCKETS_DEFAULT, d_model, padding_idx=0)
                self.seq_session_elapsed_embedding = nn.Embedding(
                    65, d_model, padding_idx=0)
                self.seq_session_prev_gap_embedding = nn.Embedding(
                    65, d_model, padding_idx=0)

        if use_calendar_time or num_time_span_buckets > 0:
            self.time_feature_norm = nn.LayerNorm(d_model)

        if num_time_span_buckets > 0:
            self.time_span_embedding = nn.Embedding(
                num_time_span_buckets, d_model, padding_idx=0)

        # ================== Temporal Attention Bias (optional) ==================
        if use_temporal_bias:
            self.temporal_bias = TemporalBias(num_heads)
            self.seq_temporal_bias = TemporalBias(num_heads)
        else:
            self.temporal_bias = None
            self.seq_temporal_bias = None

        # ================== Inter-event Time Gap Embedding (optional) ==================
        if use_time_gap:
            self.gap_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.SiLU(),
            )
        else:
            self.gap_proj = None

        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )

        # ================== Target-Aware Attention (optional) ==================
        if use_target_attention:
            self.item_repr_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            self.target_attns = nn.ModuleList([
                DINTargetAttention(d_model) for _ in self.seq_domains
            ])
            target_match_dim = self.num_sequences * (6 * d_model + 3)
            self.target_match_mlp = nn.Sequential(
                nn.Linear(target_match_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, d_model),
            )
            self.target_match_gate_logit = nn.Parameter(torch.full((1,), -3.0))
            logging.info(f"TargetSeqMatchResidual: DIN + mean/last/dot, "
                         f"{self.num_sequences} domains, dim={target_match_dim}")

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
                use_ns_self_attn=use_ns_self_attn,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # NS output fusion (optional)
        self.use_ns_output_fusion = use_ns_output_fusion
        if use_ns_output_fusion:
            self.ns_output_fusion = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            logging.info("NS output fusion enabled: pooled NS tokens fused into output")

        # ---- TIME_MATCH residual (ported verbatim from teammate 0.8248) ----
        if self.time_match_dim > 0:
            self.time_match_proj = nn.Sequential(
                nn.Linear(self.time_match_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, d_model),
            )
            self.time_match_scale = nn.Parameter(
                torch.tensor(self.time_match_scale_init, dtype=torch.float32))
        else:
            self.time_match_proj = None
            self.time_match_scale = None

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # Initialize parameters
        self._init_params()

        # ─── Mega7: ID Mask Cold Start Exercise (ported from teammate-0.825) ──
        # `mask_prob` starts at 0 (training kicks it up via the trainer
        # epoch schedule). It is applied to every NS tokenizer ID lookup and
        # to seq_id features inside _embed_seq_domain when vocab > 10000.
        self.mask_prob: float = 0.0
        # Sync the same handle to both tokenizers so the same probability
        # drives all high-cardinality ID lookups in one shot.
        if hasattr(self.user_ns_tokenizer, 'mask_prob') or True:
            self.user_ns_tokenizer.mask_prob = self.mask_prob
        if hasattr(self.item_ns_tokenizer, 'mask_prob') or True:
            self.item_ns_tokenizer.mask_prob = self.mask_prob

        # ─── Mega8: SeqFourierTimeEncoder (gated by --use_seq_fourier_encoder) ─
        # 3-period × 8-frequency Fourier encoding of per-seq-item Unix
        # timestamps. EXPΩ supersedes this with the per-event unified sincos
        # residual (Patch 1); set --use_seq_fourier_encoder back to True for
        # ablation. The class definition stays available for future toggling.
        if self.use_seq_fourier_encoder:
            self.fourier_time_encoder = SeqFourierTimeEncoder(d_model, num_freqs=8)
            logging.info(
                f"Mega8 SeqFourierTimeEncoder enabled (d_model={d_model}, num_freqs=8)")
        else:
            self.fourier_time_encoder = None
            logging.info("Mega8 SeqFourierTimeEncoder DISABLED (EXPΩ default)")

        # ─── EXPΩ Patch 1: per-event unified 8d sincos residual ──────────────
        # Allocated only when the patch is enabled. Module is zero-init so even
        # when wired in, step-0 contribution is exactly 0.
        if self.use_per_event_unified_sincos:
            self.per_event_unified_sincos = PerEventUnifiedSincosResidual(d_model)
            logging.info(
                f"EXPΩ PerEventUnifiedSincosResidual enabled (d_model={d_model})")
        else:
            self.per_event_unified_sincos = None

        # ─── Mega9 / EXPΩ Patch 3: per-token Temporal NS Token residual ──────
        # Sample-level (hour, dow, weekend) categorical embeddings × UTC+8, with
        # one projection head PER user_dense token (N heads total). Each head's
        # final Linear is zero-initialized so model starts bit-identical to
        # Mega9 base when N=1, and is structurally extended (extra zero residual
        # paths) when N>1. Learnable scales (one per head) start at 0.1.
        self.hour_emb    = nn.Embedding(25, emb_dim, padding_idx=0)
        self.dow_emb     = nn.Embedding(8,  emb_dim, padding_idx=0)
        self.weekend_emb = nn.Embedding(3,  emb_dim, padding_idx=0)
        self.temporal_proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * emb_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, d_model),
            )
            for _ in range(self.user_dense_token_count)
        ])
        self.temporal_residual_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            for _ in range(self.user_dense_token_count)
        ])
        logging.info(
            f"EXPΩ Temporal NS Token (per user_dense token) enabled: "
            f"heads={self.user_dense_token_count}, d_model={d_model}, "
            f"scale_init=0.1")

        # ---- TIME_MATCH proj final-layer zero-init (residual starts at 0) ----
        # Matches teammate 0.8248: identity at init, model learns the scale.
        if self.time_match_proj is not None:
            final = self.time_match_proj[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            logging.info(
                f"Time/match residual: enabled dim={self.time_match_dim} "
                f"scale_init={self.time_match_scale_init}")

        # ---- EXPΩ Temporal NS Token init (zero-init residual per head) -----
        # Xavier on hour/dow/weekend emb, zero padding row, and zero-init the
        # final Linear of EACH Temporal head so step-0 contribution is zero
        # (matches Mega9 base behaviour when N=1, structurally extends when
        # N>1 with all extra heads also at zero).
        for emb in [self.hour_emb, self.dow_emb, self.weekend_emb]:
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0
        for head in self.temporal_proj_heads:
            final_tp = head[-1]
            nn.init.zeros_(final_tp.weight)
            nn.init.zeros_(final_tp.bias)

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        # Init hash embeddings for sequences
        for domain_hash_embs in self._seq_hash_embs.values():
            for he in domain_hash_embs.values():
                for emb in he.embs:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
            # Init hash embeddings for NS tokenizers
            for he in tokenizer._hash_embs.values():
                for emb in he.embs:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0

        if self.user_sparse_dense_pair_residual is not None:
            for emb in self.user_sparse_dense_pair_residual.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

        if self.use_calendar_time:
            nn.init.xavier_normal_(self.hour_embedding.weight.data)
            self.hour_embedding.weight.data[0, :] = 0
            nn.init.xavier_normal_(self.weekday_embedding.weight.data)
            self.weekday_embedding.weight.data[0, :] = 0
            nn.init.xavier_normal_(self.cyclical_time_proj.weight.data)
            # EXPΩ Patch 2: init session pseudo-fid embeddings (if enabled).
            if self.seq_session_idx_embedding is not None:
                nn.init.xavier_normal_(self.seq_session_idx_embedding.weight.data)
                self.seq_session_idx_embedding.weight.data[0, :] = 0
            if self.seq_session_elapsed_embedding is not None:
                nn.init.xavier_normal_(self.seq_session_elapsed_embedding.weight.data)
                self.seq_session_elapsed_embedding.weight.data[0, :] = 0
            if self.seq_session_prev_gap_embedding is not None:
                nn.init.xavier_normal_(self.seq_session_prev_gap_embedding.weight.data)
                self.seq_session_prev_gap_embedding.weight.data[0, :] = 0

        if self.num_time_span_buckets > 0:
            nn.init.xavier_normal_(self.time_span_embedding.weight.data)
            self.time_span_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self,
        cardinality_threshold: int = 10000,
        reset_all: bool = False,
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.
            reset_all: If True, reinitialize every non-skipped sparse feature
                embedding regardless of threshold. Ported from teammate 0.8248
                for the ``--reset_sparse_each_epoch`` flag (KuaiShou MultiEpoch
                paper, arxiv 2305.19531).

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    continue
                emb = emb_list[real_idx]
                if reset_all or int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # Reinit hash embeddings for sequences (always high-cardinality)
        for domain_hash_embs in self._seq_hash_embs.values():
            for he in domain_hash_embs.values():
                for emb in he.embs:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if reset_all or int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

            # Reinit hash embeddings for NS tokenizers
            for he in tokenizer._hash_embs.values():
                for emb in he.embs:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1

        if self.user_sparse_dense_pair_residual is not None:
            for emb, (_, vs, _, _, _) in zip(
                self.user_sparse_dense_pair_residual.embs,
                self.user_sparse_dense_pair_residual.pair_specs,
            ):
                if reset_all or int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1
        if self.use_calendar_time:
            skip_count += 2
        if self.num_time_span_buckets > 0:
            skip_count += 1

        mode = "all non-skipped sparse" if reset_all else f"vocab>{cardinality_threshold}"
        logging.info(f"Re-initialized {reinit_count} sparse Embeddings "
                     f"({mode}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _make_user_dense_proj_input(self, user_dense_feats: torch.Tensor) -> torch.Tensor:
        """Drops fid62-66 dense values before the ordinary dense-token projection."""
        if user_dense_feats.shape[1] < USER_DENSE_PAIR_END:
            return user_dense_feats
        return torch.cat([
            user_dense_feats[:, :USER_DENSE_PAIR_START],
            user_dense_feats[:, USER_DENSE_PAIR_END:],
        ], dim=1)

    def _make_user_dense_tokens(
        self,
        user_int_feats: torch.Tensor,
        user_dense_feats: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Builds N user dense tokens by chunking the dense proj input.

        Returns a list of length ``user_dense_token_count``, each entry of
        shape ``(B, 1, d_model)``. The paired sparse/dense residual (added
        in Mega9 base for fid 62-66) is folded into the FIRST token only,
        preserving the residual's positional semantics.
        """
        dense_proj_input = self._make_user_dense_proj_input(user_dense_feats)
        tokens: List[torch.Tensor] = []
        cursor = 0
        for i, head in enumerate(self.user_dense_proj):
            cs = self.user_dense_chunk_sizes[i]
            chunk = dense_proj_input[:, cursor:cursor + cs]
            cursor += cs
            tok = F.silu(head(chunk))  # (B, d_model)
            tokens.append(tok)
        if self.user_sparse_dense_pair_residual is not None and tokens:
            tokens[0] = tokens[0] + self.user_sparse_dense_pair_residual(
                user_int_feats, user_dense_feats)
        return [t.unsqueeze(1) for t in tokens]

    def _make_item_repr(self, item_tokens: torch.Tensor) -> torch.Tensor:
        """Pools item tokens with mean/max and projects to d_model."""
        item_mean = item_tokens.mean(dim=1)
        item_max = item_tokens.max(dim=1).values
        return self.item_repr_mlp(torch.cat([item_mean, item_max], dim=-1))

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        hash_embs: Optional[nn.ModuleDict] = None,
        time_gap: Optional[torch.Tensor] = None,
        time_hour_ids: Optional[torch.Tensor] = None,
        time_weekday_ids: Optional[torch.Tensor] = None,
        time_span_bucket_ids: Optional[torch.Tensor] = None,
        time_ts: Optional[torch.Tensor] = None,  # Mega8: raw UTC seconds for Fourier encoding
        # ─── EXPΩ Patch 1: per-event 8d sincos (UTC+8) ─────────────────────
        per_event_unified_sincos: Optional[torch.Tensor] = None,
        # ─── EXPΩ Patch 2: best2 session pseudo-fid bucket ids ─────────────
        session_idx_ids: Optional[torch.Tensor] = None,
        session_elapsed_ids: Optional[torch.Tensor] = None,
        session_prev_gap_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                hash_key = str(i)
                if hash_embs is not None and hash_key in hash_embs:
                    e = hash_embs[hash_key](seq[:, i, :])
                    if is_id[i] and self.training:
                        e = self.seq_id_emb_dropout(e)
                    emb_list.append(e)
                else:
                    emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                current_seq = seq[:, i, :]
                # ─── Mega7: seq ID Mask Cold Start Exercise ─────────────────
                # Ported verbatim from teammate-0.825 (model.py:2337-2344).
                # 1. Dynamic ID mask drill (applies to ID tokens only).
                if self.training and is_id[i] and getattr(self, 'mask_prob', 0.0) > 0:
                    rand_tensor = torch.rand_like(current_seq, dtype=torch.float)
                    seq_id_mask = (rand_tensor < self.mask_prob) & (current_seq > 0)
                    masked_seq = current_seq.clone()
                    masked_seq[seq_id_mask] = 0
                    e = emb(masked_seq)  # lookup with masked-to-padding IDs
                else:
                    e = emb(current_seq)  # (B, L, emb_dim) normal lookup
                # 2. Regular embedding dropout (kept on top of mask).
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        # Add absolute calendar-time features from event timestamps.
        if self.use_calendar_time and time_hour_ids is not None and time_weekday_ids is not None:
            hour_emb = self.time_feature_norm(self.hour_embedding(time_hour_ids))
            weekday_emb = self.time_feature_norm(self.weekday_embedding(time_weekday_ids))

            valid = (time_hour_ids != 0).to(dtype=token_emb.dtype).unsqueeze(-1)
            hour_float = (time_hour_ids.clamp_min(1) - 1).to(dtype=token_emb.dtype)
            weekday_float = (time_weekday_ids.clamp_min(1) - 1).to(dtype=token_emb.dtype)
            cyclical_features = torch.stack([
                torch.sin(2 * math.pi * hour_float / 24.0),
                torch.cos(2 * math.pi * hour_float / 24.0),
                torch.sin(2 * math.pi * weekday_float / 7.0),
                torch.cos(2 * math.pi * weekday_float / 7.0),
            ], dim=-1)
            cyclical_emb = self.time_feature_norm(
                self.cyclical_time_proj(cyclical_features) * valid)
            token_emb = token_emb + hour_emb + weekday_emb + cyclical_emb

        # Add discrete inter-event gap bucket embedding.
        if self.num_time_span_buckets > 0 and time_span_bucket_ids is not None:
            token_emb = token_emb + self.time_feature_norm(
                self.time_span_embedding(time_span_bucket_ids))

        # Add inter-event gap embedding
        if self.gap_proj is not None and time_gap is not None:
            token_emb = token_emb + self.gap_proj(time_gap.unsqueeze(-1))

        # ─── Mega8: SeqFourierTimeEncoder per-seq-item residual ──────────────
        # Gated by --use_seq_fourier_encoder (EXPΩ default OFF — Patch 1
        # supersedes it). Pad cells (ts == 0) are zeroed out before adding.
        if (self.fourier_time_encoder is not None
                and self.use_seq_fourier_encoder
                and time_ts is not None):
            fourier_features = self.fourier_time_encoder(time_ts.to(torch.float64))
            valid = (time_ts > 0).to(dtype=token_emb.dtype).unsqueeze(-1)
            fourier_emb = fourier_features.to(token_emb.dtype) * valid
            token_emb = token_emb + fourier_emb

        # ─── EXPΩ Patch 1: per-event unified sincos 8d residual ──────────────
        # Replaces SeqFourier. Pad positions zeroed via the (ts > 0) mask we
        # derive from time_ts when available; otherwise the dataset has
        # already zeroed pad rows so the multiplier is a no-op.
        if (self.per_event_unified_sincos is not None
                and self.use_per_event_unified_sincos
                and per_event_unified_sincos is not None):
            sincos_emb = self.per_event_unified_sincos(
                per_event_unified_sincos.to(token_emb.dtype))
            if time_ts is not None:
                valid_mask = (time_ts > 0).to(dtype=token_emb.dtype).unsqueeze(-1)
                sincos_emb = sincos_emb * valid_mask
            token_emb = token_emb + sincos_emb

        # ─── EXPΩ Patch 2: best2 session pseudo-fid residuals ────────────────
        # Three additive residuals: session_idx / elapsed / prev_gap. Pad
        # positions zero via padding_idx=0. Mirrors EXPΡ-Session wiring.
        if (self.use_calendar_time and self.use_session_pseudo_fids
                and self.seq_session_idx_embedding is not None
                and session_idx_ids is not None):
            token_emb = token_emb + self.time_feature_norm(
                self.seq_session_idx_embedding(session_idx_ids))
        if (self.use_calendar_time and self.use_session_pseudo_fids
                and self.seq_session_elapsed_embedding is not None
                and session_elapsed_ids is not None):
            token_emb = token_emb + self.time_feature_norm(
                self.seq_session_elapsed_embedding(session_elapsed_ids))
        if (self.use_calendar_time and self.use_session_pseudo_fids
                and self.seq_session_prev_gap_embedding is not None
                and session_prev_gap_ids is not None):
            token_emb = token_emb + self.time_feature_norm(
                self.seq_session_prev_gap_embedding(session_prev_gap_ids))

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _seq_mean_last_pool(
        self,
        seq_tokens: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns valid mean, last valid token, and has-valid mask."""
        B, _, D = seq_tokens.shape
        valid = ~seq_mask
        valid_f = valid.to(dtype=seq_tokens.dtype).unsqueeze(-1)
        count = valid_f.sum(dim=1).clamp(min=1.0)
        seq_mean = (seq_tokens * valid_f).sum(dim=1) / count

        valid_len = valid.sum(dim=1)
        has_valid = valid_len > 0
        last_idx = (valid_len - 1).clamp(min=0)
        gather_idx = last_idx.view(B, 1, 1).expand(-1, 1, D)
        seq_last = seq_tokens.gather(dim=1, index=gather_idx).squeeze(1)
        seq_last = seq_last * has_valid.to(dtype=seq_tokens.dtype).unsqueeze(-1)
        return seq_mean, seq_last, has_valid

    def _apply_target_match_residual(
        self,
        output: torch.Tensor,
        item_repr: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
    ) -> torch.Tensor:
        """Adds a gated target-item/history match residual to final output."""
        match_parts = []
        for i in range(self.num_sequences):
            seq_mean, seq_last, has_valid = self._seq_mean_last_pool(
                seq_tokens_list[i], seq_masks_list[i])
            seq_attn = self.target_attns[i](
                item_repr, seq_tokens_list[i], seq_masks_list[i])
            seq_attn = seq_attn * has_valid.to(dtype=seq_attn.dtype).unsqueeze(-1)

            dot_scale = math.sqrt(self.d_model)
            dot_mean = (item_repr * seq_mean).sum(dim=-1, keepdim=True) / dot_scale
            dot_last = (item_repr * seq_last).sum(dim=-1, keepdim=True) / dot_scale
            dot_attn = (item_repr * seq_attn).sum(dim=-1, keepdim=True) / dot_scale
            match_parts.append(torch.cat([
                item_repr,
                seq_mean,
                seq_last,
                seq_attn,
                item_repr * seq_attn,
                item_repr - seq_attn,
                dot_mean,
                dot_last,
                dot_attn,
            ], dim=-1))

        match_feat = torch.cat(match_parts, dim=-1)
        residual = self.target_match_mlp(match_feat)
        gate = torch.sigmoid(self.target_match_gate_logit).to(dtype=output.dtype)
        return output + gate * residual

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True,
        time_delta_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Runs the multi-sequence block stack with dropout and output projection.

        Returns:
            A tuple (output, ns_out) where output is (B, D) and ns_out is
            (B, Nns, D) when NS output fusion is enabled, None otherwise.
        """
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list
        curr_time_deltas = time_delta_list

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks, curr_time_deltas = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
                seq_time_deltas=curr_time_deltas,
                query_temporal_bias=self.temporal_bias,
                seq_temporal_bias=self.seq_temporal_bias,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        ns_out = curr_ns if self.use_ns_output_fusion else None
        return output, ns_out

    def _apply_time_match_residual(
        self,
        output: torch.Tensor,
        time_match_feats: torch.Tensor,
    ) -> torch.Tensor:
        """TIME_MATCH residual head (ported verbatim from teammate 0.8248).

        Adds ``learnable_scale * MLP(time_match_feats)`` to ``output``. The
        final layer of ``time_match_proj`` is zero-initialized so the residual
        is exactly zero at the start of training.
        """
        if self.time_match_proj is None:
            return output
        if time_match_feats.numel() == 0:
            time_match_feats = output.new_zeros(
                output.shape[0], self.time_match_dim)
        time_match_feats = time_match_feats.float()
        return output + self.time_match_scale * self.time_match_proj(time_match_feats)

    def _build_temporal_cat(self, timestamp: torch.Tensor) -> torch.Tensor:
        """Compute the UTC+8 anchored (hour, dow, weekend) embedding cat.

        Shared input for all Temporal NS Token heads in EXPΩ Patch 3.
        """
        ts_local = timestamp.long() + 28800  # UTC+8 conversion (8h * 3600s)
        hour_idx = ((ts_local % 86400) // 3600 + 1).clamp(min=0, max=24).long()
        dow_idx = (((ts_local // 86400) + 4) % 7 + 1).clamp(min=0, max=7).long()
        weekend_idx = ((dow_idx >= 6).long() + 1).clamp(min=0, max=2)
        h_e = self.hour_emb(hour_idx)
        d_e = self.dow_emb(dow_idx)
        w_e = self.weekend_emb(weekend_idx)
        return torch.cat([h_e, d_e, w_e], dim=-1)  # (B, 3*emb_dim)

    def _apply_temporal_residual_per_token(
        self,
        user_dense_tokens: List[torch.Tensor],
        timestamp: torch.Tensor,
    ) -> List[torch.Tensor]:
        """EXPΩ Patch 3: per-token Temporal NS Token residual.

        For each user_dense token, add ``scale_i * head_i(temporal_cat)``.
        Each head's final Linear is zero-init so model starts bit-identical
        to Mega9 when N==1, and starts with zero-contribution extra residual
        paths when N>1.
        """
        if not user_dense_tokens:
            return user_dense_tokens
        temporal_cat = self._build_temporal_cat(timestamp)
        out: List[torch.Tensor] = []
        for i, tok in enumerate(user_dense_tokens):
            residual = self.temporal_proj_heads[i](temporal_cat)  # (B, d_model)
            scale = self.temporal_residual_scales[i]
            out.append(tok + scale * residual.unsqueeze(1))
        return out

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens: grouped projection
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)   # (B, num_user_groups, D)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)   # (B, num_item_groups, D)

        item_tokens = [item_ns]
        ns_parts = [user_ns]
        if self.has_user_dense:
            # EXPΩ Patch 3: list of N user_dense tokens (one per chunk).
            user_dense_toks = self._make_user_dense_tokens(
                inputs.user_int_feats, inputs.user_dense_feats)
            # EXPΩ Patch 3: per-token Temporal NS Token residual (zero-init).
            user_dense_toks = self._apply_temporal_residual_per_token(
                user_dense_toks, inputs.timestamp)
            # Append all N tokens at the user_dense position in the NS sequence.
            for tok in user_dense_toks:
                ns_parts.append(tok)
        # NS sequence layout (matches Mega9 base, just with N user_dense tokens):
        #   [user_ns, user_dense_0, ..., user_dense_{N-1}, item_ns, item_dense?]
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)  # (B, 1, D)
            item_tokens.append(item_dense_tok)
            ns_parts.append(item_dense_tok)

        item_tokens = torch.cat(item_tokens, dim=1)
        target_emb = self._make_item_repr(item_tokens) if self.use_target_attention else item_tokens.mean(dim=1)
        item_repr = target_emb if self.use_target_attention else None
        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

        if self.num_cross_layers > 0:
            B_ns = ns_tokens.shape[0]
            ns_flat = ns_tokens.view(B_ns, -1)
            ns_flat = self.cross_net(ns_flat)
            ns_tokens = self.cross_ln(ns_flat.view(B_ns, self.num_ns, self.d_model))

        if self.use_se_net:
            ns_tokens = self.se_gate(ns_tokens)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tg = inputs.seq_time_gaps[domain] if self.use_time_gap else None
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                hash_embs=self._seq_hash_embs[domain] if domain in self._seq_hash_embs else None,
                time_gap=tg,
                time_hour_ids=inputs.seq_time_hours[domain],
                time_weekday_ids=inputs.seq_time_weekdays[domain],
                time_span_bucket_ids=inputs.seq_time_span_buckets[domain],
                time_ts=inputs.seq_time_ts[domain],
                # EXPΩ Patch 1: per-event unified sincos 8d (UTC+8).
                per_event_unified_sincos=inputs.seq_per_event_unified_sincos.get(domain),
                # EXPΩ Patch 2: best2 session pseudo-fid bucket ids.
                session_idx_ids=inputs.seq_session_idx.get(domain),
                session_elapsed_ids=inputs.seq_session_elapsed.get(domain),
                session_prev_gap_ids=inputs.seq_session_prev_gap.get(domain))
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # 3. Collect time deltas for temporal attention bias
        time_delta_list = None
        if self.use_temporal_bias:
            time_delta_list = [inputs.seq_time_deltas[domain] for domain in self.seq_domains]

        # 5. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        q_tokens_list = self.query_generator(
            target_emb, ns_tokens, seq_tokens_list, seq_masks_list)

        # 6. Dropout + MultiSeqHyFormerBlock stack + output projection
        output, ns_out = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training,
            time_delta_list=time_delta_list,
        )

        # 6.5. NS output fusion (optional)
        if self.use_ns_output_fusion and ns_out is not None:
            ns_pooled = ns_out.mean(dim=1)  # (B, D)
            output = self.ns_output_fusion(torch.cat([output, ns_pooled], dim=-1))

        # 7. Target-item/history explicit match residual (parallel path)
        if self.use_target_attention:
            output = self._apply_target_match_residual(
                output, item_repr, seq_tokens_list, seq_masks_list)

        # 7.5. TIME_MATCH residual (ported verbatim from teammate 0.8248).
        output = self._apply_time_match_residual(
            output, inputs.time_match_feats)

        # 7. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        item_tokens = [item_ns]
        ns_parts = [user_ns]
        if self.has_user_dense:
            # EXPΩ Patch 3: list of N user_dense tokens + per-token Temporal.
            user_dense_toks = self._make_user_dense_tokens(
                inputs.user_int_feats, inputs.user_dense_feats)
            user_dense_toks = self._apply_temporal_residual_per_token(
                user_dense_toks, inputs.timestamp)
            for tok in user_dense_toks:
                ns_parts.append(tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            item_tokens.append(item_dense_tok)
            ns_parts.append(item_dense_tok)

        item_tokens = torch.cat(item_tokens, dim=1)
        target_emb = self._make_item_repr(item_tokens) if self.use_target_attention else item_tokens.mean(dim=1)
        item_repr = target_emb if self.use_target_attention else None
        ns_tokens = torch.cat(ns_parts, dim=1)

        if self.num_cross_layers > 0:
            B_ns = ns_tokens.shape[0]
            ns_flat = ns_tokens.view(B_ns, -1)
            ns_flat = self.cross_net(ns_flat)
            ns_tokens = self.cross_ln(ns_flat.view(B_ns, self.num_ns, self.d_model))

        if self.use_se_net:
            ns_tokens = self.se_gate(ns_tokens)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tg = inputs.seq_time_gaps[domain] if self.use_time_gap else None
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                hash_embs=self._seq_hash_embs[domain] if domain in self._seq_hash_embs else None,
                time_gap=tg,
                time_hour_ids=inputs.seq_time_hours[domain],
                time_weekday_ids=inputs.seq_time_weekdays[domain],
                time_span_bucket_ids=inputs.seq_time_span_buckets[domain],
                time_ts=inputs.seq_time_ts[domain],
                # EXPΩ Patch 1: per-event unified sincos 8d (UTC+8).
                per_event_unified_sincos=inputs.seq_per_event_unified_sincos.get(domain),
                # EXPΩ Patch 2: best2 session pseudo-fid bucket ids.
                session_idx_ids=inputs.seq_session_idx.get(domain),
                session_elapsed_ids=inputs.seq_session_elapsed.get(domain),
                session_prev_gap_ids=inputs.seq_session_prev_gap.get(domain))
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        time_delta_list = None
        if self.use_temporal_bias:
            time_delta_list = [inputs.seq_time_deltas[domain] for domain in self.seq_domains]

        q_tokens_list = self.query_generator(
            target_emb, ns_tokens, seq_tokens_list, seq_masks_list)

        output, ns_out = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False,
            time_delta_list=time_delta_list,
        )

        if self.use_ns_output_fusion and ns_out is not None:
            ns_pooled = ns_out.mean(dim=1)
            output = self.ns_output_fusion(torch.cat([output, ns_pooled], dim=-1))

        if self.use_target_attention:
            output = self._apply_target_match_residual(
                output, item_repr, seq_tokens_list, seq_masks_list)

        # TIME_MATCH residual (ported verbatim from teammate 0.8248).
        output = self._apply_time_match_residual(
            output, inputs.time_match_feats)

        logits = self.clsfier(output)
        return logits, output

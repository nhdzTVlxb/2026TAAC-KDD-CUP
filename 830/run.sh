#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- EXPΩ Mega9 + best2 fusion (3 surgical patches on Mega9 anchor LB 0.829708) ----
# Mega9 base: submissions/EXPΒ-Mega9-Mega8-TemporalToken-bf16-seed11-b128-20260520-1329/
#
# Patch 1: per-event unified sincos 8d × 4 periods (hour/dow/dom/month, UTC+8).
#          Replaces Mega8 SeqFourierTimeEncoder (gated OFF via
#          --use_seq_fourier_encoder; the class definition is kept in model.py
#          as a future-ablation toggle). The new PerEventUnifiedSincosResidual
#          is a 4-layer MLP with the final Linear zero-initialized so the
#          residual contributes exactly zero at step 0.
# Patch 2: best2 session pseudo-fid (3 per-event bucket signals: session
#          index / elapsed-since-session-start / gap-from-previous-event).
#          Constants renamed for uniqueness vs best2:
#            SESSION_BREAK_SECONDS         → _SESSION_GAP_THRESHOLD_SEC
#            SESSION_INDEX_BOUNDARIES      → _USER_SESSION_BUCKET_EDGES
#            NUM_SESSION_INDEX_BUCKETS     → _NUM_USER_SESSION_BUCKETS
#            _bucketize_session_index_scalar → _quantize_user_session_index
#            _time_bucketize_scalar_for_session → _quantize_session_relative_delta
#            _build_seq_session_features   → _compute_per_event_session_buckets
# Patch 3: multi-token user_dense N=2 + per-token Temporal NS Token residual.
#          Replaces Mega9 single user_dense_token. Each of the N=2 chunks gets
#          its own Linear projection (split evenly with first-N-1 = base,
#          last = base + remainder), and its own zero-init Temporal residual
#          head. Model starts bit-identical to Mega9 (Patch 3 residual is 0
#          at step 0, Patch 1 residual is 0 at step 0; only structural change
#          is the token count, balanced by item_ns_tokens 4 → 3).
#
# ---- Token math (must satisfy d_model % T == 0) -----------------------------
# d_model              = 64
# num_queries          = 2
# num_sequences        = 4 (seq_a / seq_b / seq_c / seq_d)
# user_ns_tokens       = 3 (unchanged vs Mega9)
# item_ns_tokens       = 3 (decreased from 4 to absorb +1 from
#                            user_dense_token_count=2). Chosen because it is
#                            the smallest structural change that keeps T as
#                            a valid divisor of d_model=64. (Reducing
#                            num_queries 2→1 would push T off any d_model
#                            divisor; trimming item_ns by 1 is cheapest.)
# user_dense_tokens    = 2 (Patch 3)
# has_item_dense       = False (item_dense_dim = 0 in our schema)
# num_ns = user_ns_tokens + user_dense_tokens + item_ns_tokens + has_item_dense
#        = 3 + 2 + 3 + 0
#        = 8
# T      = num_queries * num_sequences + num_ns
#        = 2 * 4 + 8
#        = 16
# Check: d_model(64) % T(16) = 0  ✓
#
# Mega9 base had T = 2*4 + (3 + 1 + 4 + 0) = 16. EXPΩ keeps T = 16 exactly by
# trading one item_ns token for the extra user_dense token.
#
# Mega9 carryover (active):
#   • Temporal NS Token residual (now per-user_dense-token, zero-init)
#   • SeqFourierTimeEncoder (DISABLED via --use_seq_fourier_encoder; replaced
#     by Patch 1 per-event unified sincos)
#   • ID Mask Cold Start Exercise (vocab>10K, epoch-warmup)
#   • TIME_MATCH 40d residual (teammate 0.8248)
#   • UTC+8 timezone alignment (v29b)
#   • SAFE_MATCH_V2 30d dense block (v29b)
#   • ACTSIG hist_dist 4d (v29b)
#   • fid 110 timestamp bucket (v11.5)
#   • EMA 0.9995, bce_pairwise loss with λ=0.05
#   • reset_sparse_each_epoch (teammate 0.8248)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 3 \
    --item_ns_tokens 3 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --hash_bucket_size 100000 \
    --use_target_attention \
    --num_workers 8 \
    --num_cross_layers 2 \
    --cross_low_rank 64 \
    --use_se_net \
    --use_ns_self_attn \
    --use_ns_output_fusion \
    --use_temporal_bias \
    --use_time_gap \
    --precision bf16 \
    --lr_schedule cosine \
    --warmup_steps 500 \
    --ema_decay 0.9995 \
    --label_smoothing 0.005 \
    --weight_decay 0.02 \
    --loss_type bce_pairwise \
    --pairwise_lambda 0.05 \
    --use_time_match_features \
    --time_match_recent_k 128 \
    --time_match_scale 0.1 \
    --reset_sparse_each_epoch \
    --use_per_event_unified_sincos \
    --use_session_pseudo_fids \
    --user_dense_token_count 2 \
    "$@"

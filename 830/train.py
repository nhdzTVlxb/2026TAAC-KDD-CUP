"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import (
    FeatureSchema,
    get_pcvr_data,
    NUM_TIME_BUCKETS,
    NUM_TIME_SPAN_BUCKETS,
)
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


USER_SPARSE_DENSE_PAIR_FIDS = [62, 63, 64, 65, 66]


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def build_user_sparse_dense_pair_specs(
    user_int_schema: FeatureSchema,
    user_dense_schema: FeatureSchema,
    user_int_vocab_sizes: List[int],
    fids: List[int] = USER_SPARSE_DENSE_PAIR_FIDS,
) -> List[Tuple[int, int, int, int, int]]:
    """Build paired sparse-id/float-value specs from FeatureSchema offsets."""
    specs: List[Tuple[int, int, int, int, int]] = []
    for fid in fids:
        int_offset, int_length = user_int_schema.get_offset_length(fid)
        dense_offset, dense_length = user_dense_schema.get_offset_length(fid)
        if int_length != dense_length:
            raise ValueError(
                f"Paired user fid {fid} has mismatched lengths: "
                f"user_int={int_length}, user_dense={dense_length}"
            )
        vs = max(user_int_vocab_sizes[int_offset:int_offset + int_length])
        specs.append((fid, vs, int_offset, dense_offset, int_length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')
    parser.add_argument('--precision', type=str, default='auto',
                        choices=['auto', 'fp32', 'bf16'],
                        help='Training precision: auto enables bf16 autocast on '
                             'supported CUDA GPUs, bf16 requires bf16 autocast '
                             'with fp32 fallback when unsupported, fp32 disables '
                             'autocast')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--use_calendar_time', action='store_true', default=True,
                        help='Enable hour/weekday embedding and cyclic sin-cos calendar time features')
    parser.add_argument('--no_calendar_time', dest='use_calendar_time', action='store_false',
                        help='Disable calendar-time sequence features')
    parser.add_argument('--use_time_span_buckets', action='store_true', default=True,
                        help='Enable discrete inter-event time-span bucket embeddings')
    parser.add_argument('--no_time_span_buckets', dest='use_time_span_buckets', action='store_false',
                        help='Disable inter-event time-span bucket embeddings')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce',
                        choices=['bce', 'focal', 'bce_pairwise'],
                        help='Loss type: bce = BCEWithLogits, '
                             'focal = Focal Loss, '
                             'bce_pairwise = BCE + batch pairwise ranking loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--pairwise_lambda', type=float, default=0.05,
                        help='Weight for batch pairwise ranking loss '
                             '(effective only when --loss_type=bce_pairwise)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')
    parser.add_argument('--reset_sparse_each_epoch', action='store_true', default=False,
                        help='Reset every sparse Embedding (regardless of cardinality) '
                             'after each epoch. Ported from teammate 0.8248 bundle; '
                             'mirrors the KuaiShou MultiEpoch trick (arxiv 2305.19531).')

    # TIME_MATCH dense residual features (ported from teammate 0.8248).
    parser.add_argument('--use_time_match_features', action='store_true', default=False,
                        help='Enable the compact dense time/match residual feature '
                             'channel (cyclical current time + per-domain recency + '
                             '(item_fid, seq_fid) match signals). Ported verbatim from '
                             'teammate 0.8248 bundle.')
    parser.add_argument('--time_match_recent_k', type=int, default=32,
                        help='Newest-first window size for the recency / match '
                             'aggregations inside the TIME_MATCH feature channel.')
    parser.add_argument('--time_match_scale', type=float, default=0.1,
                        help='Initial value of the learnable scale applied to the '
                             'TIME_MATCH residual (model adapts it during training).')

    # ─── EXPΩ Patch 1: per-event unified sincos 8d (UTC+8) ────────────────────
    parser.add_argument('--use_per_event_unified_sincos', action='store_true',
                        default=True,
                        help='Enable per-event unified sincos 8d residual '
                             '(hour/dow/dom/month × sin/cos, UTC+8 aligned). '
                             'EXPΩ default ON; supersedes SeqFourierTimeEncoder.')
    parser.add_argument('--no_per_event_unified_sincos',
                        dest='use_per_event_unified_sincos',
                        action='store_false',
                        help='Disable per-event unified sincos residual.')
    parser.add_argument('--use_seq_fourier_encoder', action='store_true',
                        default=False,
                        help='Re-enable Mega8 SeqFourierTimeEncoder allocation + '
                             'forward call (EXPΩ default OFF, kept as toggle '
                             'for future ablation).')

    # ─── EXPΩ Patch 2: best2 session pseudo-fid bucket triples ─────────────────
    parser.add_argument('--use_session_pseudo_fids', action='store_true',
                        default=True,
                        help='Enable per-event session bucket triples (session '
                             'index / elapsed-since-session-start / gap-from-'
                             'previous-event) as additive residuals on seq '
                             'token embeddings. Ported from EXPΡ-Session.')
    parser.add_argument('--no_session_pseudo_fids',
                        dest='use_session_pseudo_fids',
                        action='store_false',
                        help='Disable session pseudo-fid residuals.')

    # ─── EXPΩ Patch 3: multi-token user_dense + per-token Temporal residual ────
    parser.add_argument('--user_dense_token_count', type=int, default=2,
                        help='Number of user_dense NS tokens (1 = Mega9 base '
                             'behaviour, 2 = EXPΩ default). Increasing this '
                             'expands the NS sequence length; T must remain '
                             'd_model-divisible.')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    # DCN-v2 cross network.
    parser.add_argument('--num_cross_layers', type=int, default=0,
                        help='Number of DCN-v2 cross layers applied to NS tokens '
                             'before query generation (0 = disabled)')
    parser.add_argument('--cross_low_rank', type=int, default=0,
                        help='Low-rank factorization rank for cross layer W matrix '
                             '(0 = full rank)')
    parser.add_argument('--cross_dropout', type=float, default=0.1,
                        help='Dropout rate inside cross layers (default 0.1)')

    # SE-Net feature gating.
    parser.add_argument('--use_se_net', action='store_true', default=False,
                        help='Enable SE-Net feature gating on NS tokens')

    # Hash embedding for ultra-high cardinality features.
    parser.add_argument('--hash_bucket_size', type=int, default=100000,
                        help='Bucket count for hash embeddings on features whose '
                             'vocab exceeds emb_skip_threshold (0 = disabled, '
                             'those features fall back to zero vectors)')

    # Target-aware attention.
    parser.add_argument('--use_target_attention', action='store_true', default=True,
                        help='Enable DIN-style target-aware attention + item-conditioned '
                             'query generation (default on)')
    parser.add_argument('--no_target_attention', dest='use_target_attention',
                        action='store_false',
                        help='Disable target-aware attention')

    # NS self-attention (feature crossing inside HyFormerBlocks).
    parser.add_argument('--use_ns_self_attn', action='store_true', default=False,
                        help='Enable self-attention among NS tokens within each '
                             'HyFormerBlock for feature crossing')
    parser.add_argument('--no_ns_self_attn', dest='use_ns_self_attn',
                        action='store_false',
                        help='Disable NS self-attention (default)')

    # NS output fusion.
    parser.add_argument('--use_ns_output_fusion', action='store_true', default=False,
                        help='Fuse final NS tokens into output via mean-pool + '
                             'Linear fusion layer')
    parser.add_argument('--no_ns_output_fusion', dest='use_ns_output_fusion',
                        action='store_false',
                        help='Disable NS output fusion (default)')

    # Temporal Attention Bias.
    parser.add_argument('--use_temporal_bias', action='store_true', default=False,
                        help='Enable per-head temporal decay bias in cross-attention')
    parser.add_argument('--no_temporal_bias', dest='use_temporal_bias',
                        action='store_false',
                        help='Disable temporal attention bias (default)')

    # Inter-event time gap.
    parser.add_argument('--use_time_gap', action='store_true', default=False,
                        help='Enable inter-event time gap embedding in sequences')
    parser.add_argument('--no_time_gap', dest='use_time_gap',
                        action='store_false',
                        help='Disable inter-event time gap (default)')

    # Training tricks.
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of linear LR warmup steps (0 = no warmup)')
    parser.add_argument('--lr_schedule', type=str, default='none',
                        choices=['none', 'cosine'],
                        help='LR schedule: none = constant, cosine = warmup + cosine decay')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA decay rate for model weights (0 = disabled, '
                             'typical: 0.999 or 0.9999)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing epsilon (0 = disabled, typical: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='AdamW weight decay for dense parameters')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        enable_time_match_features=args.use_time_match_features,
        time_match_recent_k=args.time_match_recent_k,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "num_time_span_buckets": NUM_TIME_SPAN_BUCKETS if args.use_time_span_buckets else 0,
        "use_calendar_time": args.use_calendar_time,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "num_cross_layers": args.num_cross_layers,
        "cross_low_rank": args.cross_low_rank,
        "cross_dropout": args.cross_dropout,
        "use_se_net": args.use_se_net,
        "hash_bucket_size": args.hash_bucket_size,
        "use_target_attention": args.use_target_attention,
        "use_ns_self_attn": args.use_ns_self_attn,
        "use_ns_output_fusion": args.use_ns_output_fusion,
        "use_temporal_bias": args.use_temporal_bias,
        "use_time_gap": args.use_time_gap,
        "user_sparse_dense_pair_specs": build_user_sparse_dense_pair_specs(
            pcvr_dataset.user_int_schema,
            pcvr_dataset.user_dense_schema,
            pcvr_dataset.user_int_vocab_sizes,
        ),
        # TIME_MATCH residual (ported verbatim from teammate 0.8248)
        "time_match_dim": (
            pcvr_dataset.time_match_feature_dim
            if args.use_time_match_features else 0
        ),
        "time_match_scale": args.time_match_scale,
        # ─── EXPΩ patches ─────────────────────────────────────────────────────
        "use_per_event_unified_sincos": args.use_per_event_unified_sincos,
        "use_seq_fourier_encoder": args.use_seq_fourier_encoder,
        "use_session_pseudo_fids": args.use_session_pseudo_fids,
        "user_dense_token_count": args.user_dense_token_count,
    }

    # Persist derived TIME_MATCH dim into args so train_config.json captures
    # it for infer.py (the dim is computed from schema, not user-supplied).
    args.time_match_dim = int(model_args.get("time_match_dim", 0))
    # Persist EXPΩ patch flags too (infer.py reads these via train_config.json).
    args.use_per_event_unified_sincos = bool(
        model_args.get("use_per_event_unified_sincos", True))
    args.use_seq_fourier_encoder = bool(
        model_args.get("use_seq_fourier_encoder", False))
    args.use_session_pseudo_fids = bool(
        model_args.get("use_session_pseudo_fids", True))
    args.user_dense_token_count = int(
        model_args.get("user_dense_token_count", 2))

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        reset_sparse_each_epoch=args.reset_sparse_each_epoch,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        ema_decay=args.ema_decay,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        pairwise_lambda=args.pairwise_lambda,
        precision=args.precision,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
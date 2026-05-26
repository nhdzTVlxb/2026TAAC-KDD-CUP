"""PCVRHyFormer inference script.

Model construction mirrors train.py:
- rebuild model from schema.json + ns_groups.json + train_config.json
- resolve model hyperparameters from train_config.json first
- fallback only when train_config.json is missing or incomplete

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory containing model.pt / train_config.json / schema.json.
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for predictions.json.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHyFormer, ModelInput


try:
    from dataset import NUM_TIME_SPAN_BUCKETS
except ImportError:
    NUM_TIME_SPAN_BUCKETS = 0


USER_SPARSE_DENSE_PAIR_FIDS = [62, 63, 64, 65, 66]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Fallback values are used only when train_config.json is missing or incomplete.
# Ideally, inference should always use the train_config.json saved with checkpoint.
_FALLBACK_MODEL_CFG = {
    # Basic model config
    "d_model": 64,
    "emb_dim": 64,
    "num_queries": 1,
    "num_hyformer_blocks": 2,
    "num_heads": 4,
    "seq_encoder_type": "transformer",
    "hidden_mult": 4,
    "dropout_rate": 0.01,
    "seq_top_k": 50,
    "seq_causal": False,
    "action_num": 1,

    # Time features
    "num_time_buckets": NUM_TIME_BUCKETS,
    "num_time_span_buckets": NUM_TIME_SPAN_BUCKETS,
    "use_calendar_time": True,

    # RankMixer / RoPE / embedding
    "rank_mixer_mode": "full",
    "use_rope": False,
    "rope_base": 10000.0,
    "emb_skip_threshold": 0,
    "seq_id_threshold": 10000,

    # NS tokenizer
    "ns_tokenizer_type": "rankmixer",
    "user_ns_tokens": 0,
    "item_ns_tokens": 0,

    # DCN-v2 cross network
    "num_cross_layers": 0,
    "cross_low_rank": 0,
    "cross_dropout": 0.1,

    # SE-Net
    "use_se_net": False,

    # Hash embedding
    "hash_bucket_size": 100000,

    # Target-aware attention / NS / temporal switches
    "use_target_attention": True,
    "use_ns_self_attn": False,
    "use_ns_output_fusion": False,
    "use_temporal_bias": False,
    "use_time_gap": False,

    # TIME_MATCH residual (ported from teammate 0.8248). When the trained
    # model did not enable it, time_match_dim=0 disables the projection.
    "time_match_dim": 0,
    "time_match_scale": 0.1,

    # ─── EXPΩ patch defaults (mirror PCVRHyFormer constructor) ──────────────
    "use_per_event_unified_sincos": True,
    "use_seq_fourier_encoder": False,
    "use_session_pseudo_fids": True,
    "user_dense_token_count": 2,
}

_FALLBACK_SEQ_MAX_LENS = "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16

_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs = [(vocab_size, offset, length), ...]."""
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
    """Build paired sparse-id / dense-value specs.

    Each spec:
        (fid, vocab_size, int_offset, dense_offset, length)
    """
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


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like 'seq_a:256,seq_b:256,...' into a dict."""
    seq_max_lens: Dict[str, int] = {}

    if not sml_str:
        return seq_max_lens

    for pair in sml_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split(":")
        seq_max_lens[k.strip()] = int(v.strip())

    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load train_config.json from checkpoint directory."""
    train_config_path = os.path.join(model_dir, "train_config.json")

    if os.path.exists(train_config_path):
        with open(train_config_path, "r") as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg

    logging.warning(
        f"train_config.json not found in {model_dir}, "
        "falling back to hardcoded defaults. "
        "Shape mismatch may occur if training used non-default hyperparameters."
    )
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from train_config.

    Special handling:
    - num_time_buckets can be derived from use_time_buckets.
    - num_time_span_buckets can be derived from use_time_span_buckets.
    """
    cfg: Dict[str, Any] = {}

    for key in _MODEL_CFG_KEYS:
        if key == "num_time_buckets":
            if "num_time_buckets" in train_config:
                cfg[key] = train_config["num_time_buckets"]
            elif "use_time_buckets" in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config["use_time_buckets"] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    "train_config missing both 'num_time_buckets' and "
                    f"'use_time_buckets', using fallback = {cfg[key]}"
                )
            continue

        if key == "num_time_span_buckets":
            if "num_time_span_buckets" in train_config:
                cfg[key] = train_config["num_time_span_buckets"]
            elif "use_time_span_buckets" in train_config:
                cfg[key] = NUM_TIME_SPAN_BUCKETS if train_config["use_time_span_buckets"] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    "train_config missing both 'num_time_span_buckets' and "
                    f"'use_time_span_buckets', using fallback = {cfg[key]}"
                )
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}"
            )

    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = "cpu",
) -> PCVRHyFormer:
    """Construct PCVRHyFormer from dataset schema + train_config."""
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]

    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")

        with open(ns_groups_json, "r") as f:
            ns_groups_cfg = json.load(f)

        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }

        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg["user_ns_groups"].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg["item_ns_groups"].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                "present in schema.json. The ns_groups.json and schema.json "
                "must come from the same training run."
            ) from exc
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema,
        dataset.user_int_vocab_sizes,
    )
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema,
        dataset.item_int_vocab_sizes,
    )

    user_sparse_dense_pair_specs = build_user_sparse_dense_pair_specs(
        dataset.user_int_schema,
        dataset.user_dense_schema,
        dataset.user_int_vocab_sizes,
    )

    logging.info(f"Using user_sparse_dense_pair_specs: {user_sparse_dense_pair_specs}")
    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")

    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        user_sparse_dense_pair_specs=user_sparse_dense_pair_specs,
        **model_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load state_dict."""
    state_dict = torch.load(ckpt_path, map_location=device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by infer.py does NOT match the checkpoint. "
            "Check train_config.json, schema.json, ns_groups.json, and model.py."
        )
        raise e


def get_ckpt_path() -> Optional[str]:
    """Locate model.pt or the first *.pt file inside MODEL_OUTPUT_PATH."""
    model_dir = os.environ.get("MODEL_OUTPUT_PATH")

    if not model_dir:
        return None

    preferred = os.path.join(model_dir, "model.pt")
    if os.path.exists(preferred):
        return preferred

    for item in os.listdir(model_dir):
        if item.endswith(".pt"):
            return os.path.join(model_dir, item)

    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ModelInput, handling dynamic seq domains."""
    device_batch: Dict[str, Any] = {}

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch["_seq_domains"]

    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    seq_time_deltas: Dict[str, torch.Tensor] = {}
    seq_time_gaps: Dict[str, torch.Tensor] = {}
    seq_time_hours: Dict[str, torch.Tensor] = {}
    seq_time_weekdays: Dict[str, torch.Tensor] = {}
    seq_time_span_buckets: Dict[str, torch.Tensor] = {}
    seq_time_ts: Dict[str, torch.Tensor] = {}    # Mega8 Fourier raw ts per seq item
    # EXPΩ Patch 1: per-event unified sincos 8d.
    seq_per_event_unified_sincos: Dict[str, torch.Tensor] = {}
    # EXPΩ Patch 2: best2 session pseudo-fid bucket triples.
    seq_session_idx: Dict[str, torch.Tensor] = {}
    seq_session_elapsed: Dict[str, torch.Tensor] = {}
    seq_session_prev_gap: Dict[str, torch.Tensor] = {}

    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f"{domain}_len"]

        B, _, L = device_batch[domain].shape

        zero_long = torch.zeros(B, L, dtype=torch.long, device=device)
        zero_float = torch.zeros(B, L, dtype=torch.float, device=device)
        zero_sincos = torch.zeros(B, L, 8, dtype=torch.float, device=device)

        seq_time_buckets[domain] = device_batch.get(
            f"{domain}_time_bucket",
            zero_long,
        )
        seq_time_deltas[domain] = device_batch.get(
            f"{domain}_time_delta",
            zero_float,
        )
        seq_time_gaps[domain] = device_batch.get(
            f"{domain}_time_gap",
            zero_float,
        )
        seq_time_hours[domain] = device_batch.get(
            f"{domain}_time_hour",
            zero_long,
        )
        seq_time_weekdays[domain] = device_batch.get(
            f"{domain}_time_weekday",
            zero_long,
        )
        seq_time_span_buckets[domain] = device_batch.get(
            f"{domain}_time_span_bucket",
            zero_long,
        )
        seq_time_ts[domain] = device_batch.get(
            f"{domain}_seq_time_ts",
            zero_long,
        )
        seq_per_event_unified_sincos[domain] = device_batch.get(
            f"{domain}_per_event_unified_sincos",
            zero_sincos,
        )
        seq_session_idx[domain] = device_batch.get(
            f"{domain}_session_idx_bucket",
            zero_long,
        )
        seq_session_elapsed[domain] = device_batch.get(
            f"{domain}_session_elapsed_bucket",
            zero_long,
        )
        seq_session_prev_gap[domain] = device_batch.get(
            f"{domain}_session_prev_gap_bucket",
            zero_long,
        )

    return ModelInput(
        user_int_feats=device_batch["user_int_feats"],
        item_int_feats=device_batch["item_int_feats"],
        user_dense_feats=device_batch["user_dense_feats"],
        item_dense_feats=device_batch["item_dense_feats"],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        seq_time_deltas=seq_time_deltas,
        seq_time_gaps=seq_time_gaps,
        seq_time_hours=seq_time_hours,
        seq_time_weekdays=seq_time_weekdays,
        seq_time_span_buckets=seq_time_span_buckets,
        seq_time_ts=seq_time_ts,
        seq_per_event_unified_sincos=seq_per_event_unified_sincos,
        seq_session_idx=seq_session_idx,
        seq_session_elapsed=seq_session_elapsed,
        seq_session_prev_gap=seq_session_prev_gap,
        time_match_feats=device_batch.get(
            "time_match_feats",
            torch.zeros(
                device_batch["user_int_feats"].shape[0], 0,
                dtype=torch.float, device=device,
            ),
        ),
        timestamp=device_batch.get(
            "timestamp",
            torch.zeros(
                device_batch["user_int_feats"].shape[0],
                dtype=torch.long, device=device,
            ),
        ),
    )


def main() -> None:
    model_dir = os.environ.get("MODEL_OUTPUT_PATH")
    data_dir = os.environ.get("EVAL_DATA_PATH")
    result_dir = os.environ.get("EVAL_RESULT_PATH")

    if not model_dir:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    if not data_dir:
        raise ValueError("EVAL_DATA_PATH is not set")
    if not result_dir:
        raise ValueError("EVAL_RESULT_PATH is not set")

    os.makedirs(result_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Prefer checkpoint schema to exactly match training.
    schema_path = os.path.join(model_dir, "schema.json")
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, "schema.json")
    logging.info(f"Using schema: {schema_path}")

    train_config = load_train_config(model_dir)

    sml_str = train_config.get("seq_max_lens", _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    batch_size = int(train_config.get("batch_size", _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get("num_workers", _FALLBACK_NUM_WORKERS))

    # Restore TIME_MATCH dataset settings from the train config (ported from 0.8248).
    use_time_match = bool(train_config.get("use_time_match_features", False))
    time_match_recent_k = int(train_config.get("time_match_recent_k", 32))
    logging.info(
        f"TIME_MATCH: enabled={use_time_match} recent_k={time_match_recent_k}"
    )

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
        enable_time_match_features=use_time_match,
        time_match_recent_k=time_match_recent_k,
    )

    logging.info(f"Total test samples: {test_dataset.num_rows}")

    model_cfg = resolve_model_cfg(train_config)

    ns_groups_json = train_config.get("ns_groups_json", None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        dataset=test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: "
            f"{os.listdir(model_dir) if os.path.isdir(model_dir) else 'N/A'}"
        )

    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    loader_kwargs: Dict[str, Any] = {
        "dataset": test_dataset,
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    test_loader = DataLoader(**loader_kwargs)

    all_probs: List[float] = []
    all_user_ids: List[Any] = []

    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get("user_id", [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"Processed about {(batch_idx + 1) * batch_size} samples")

    logging.info(f"Inference complete: {len(all_probs)} predictions")

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    output_path = os.path.join(result_dir, "predictions.json")
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()

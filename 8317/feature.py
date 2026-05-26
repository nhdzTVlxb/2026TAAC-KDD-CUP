"""Deterministic feature builder for the v33 clean experiment.

The script reads raw parquet files, appends five row-level features, and writes
an updated schema.json.  It intentionally does not use labels or future data.

Added features:
    user_int_feats_201: local hour, values 0..23
    user_int_feats_202: local day-of-week bucket, values 0..6
    user_int_feats_203: hour/day cross bucket, values 0..167
    user_dense_feats_901: log1p(length(domain_a_seq_38))
    user_dense_feats_902: length(domain_d_seq_17) / (length(domain_a_seq_38)+1)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LOCAL_TIME_OFFSET_SECONDS = 8 * 3600
SECONDS_PER_DAY = 24 * 3600
DAYS_PER_WEEK = 7

HOUR_FID = 201
DOW_FID = 202
HOUR_DOW_FID = 203
SEQ_A_LEN_FID = 901
SEQ_D_OVER_A_FID = 902

SEQ_A_FIELD = "domain_a_seq_38"
SEQ_D_FIELD = "domain_d_seq_17"


def read_parquet(path: Path) -> pa.Table:
    """Load one parquet file as a PyArrow table."""
    return pq.read_table(path)


def compute_local_hour_and_weekday(timestamp_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Convert UTC unix seconds into local hour and day-of-week ids."""
    local_ts = timestamp_series + LOCAL_TIME_OFFSET_SECONDS
    hour_id = (local_ts % SECONDS_PER_DAY) // 3600
    weekday_id = (local_ts // SECONDS_PER_DAY + 4) % DAYS_PER_WEEK
    return hour_id.astype(np.int64), weekday_id.astype(np.int64)


def make_hour_weekday_cross(hour_id: pd.Series, weekday_id: pd.Series) -> pd.Series:
    """Combine hour and weekday into one compact categorical bucket."""
    return (hour_id * DAYS_PER_WEEK + weekday_id).astype(np.int64)


def append_time_int_features(table: pa.Table) -> pa.Table:
    """Append 201/202/203 time-derived sparse features."""
    timestamps = table.column("timestamp").to_pandas()
    hour_id, weekday_id = compute_local_hour_and_weekday(timestamps)
    hour_weekday_id = make_hour_weekday_cross(hour_id, weekday_id)

    table = table.append_column(
        f"user_int_feats_{HOUR_FID}",
        pa.array(hour_id, type=pa.int64()),
    )
    table = table.append_column(
        f"user_int_feats_{DOW_FID}",
        pa.array(weekday_id, type=pa.int64()),
    )
    table = table.append_column(
        f"user_int_feats_{HOUR_DOW_FID}",
        pa.array(hour_weekday_id, type=pa.int64()),
    )
    return table


def sequence_lengths(table: pa.Table, column_name: str) -> pd.Series:
    """Return sequence lengths; missing columns become all-zero lengths."""
    if column_name not in table.column_names:
        return pd.Series(np.zeros(len(table), dtype=np.int64))
    seq_col = table.column(column_name).to_pandas()
    return seq_col.apply(lambda values: len(values) if values is not None else 0)


def to_float_list_array(values: Iterable[float]) -> pa.Array:
    """Convert scalar floats into list<float32> rows expected by user_dense."""
    return pa.array([[float(x)] for x in values], type=pa.list_(pa.float32()))


def append_activity_dense_features(table: pa.Table) -> pa.Table:
    """Append 901/902 sequence-activity dense features."""
    seq_a_len = sequence_lengths(table, SEQ_A_FIELD)
    seq_d_len = sequence_lengths(table, SEQ_D_FIELD)

    log_seq_a_len = np.log1p(seq_a_len).astype(float)
    seq_d_over_a = (seq_d_len / (seq_a_len + 1.0)).astype(float)

    table = table.append_column(
        f"user_dense_feats_{SEQ_A_LEN_FID}",
        to_float_list_array(log_seq_a_len),
    )
    table = table.append_column(
        f"user_dense_feats_{SEQ_D_OVER_A_FID}",
        to_float_list_array(seq_d_over_a),
    )
    return table


def transform_table(table: pa.Table, add_activity_features: bool = True) -> pa.Table:
    """Apply all deterministic row-level feature transformations."""
    table = append_time_int_features(table)
    if add_activity_features:
        table = append_activity_dense_features(table)
    return table


def process_parquet_file(input_file: Path, output_file: Path, add_activity_features: bool) -> None:
    """Transform one parquet file and write it to the output directory."""
    print(f">>> 写入特征文件: {output_file.name}")
    table = read_parquet(input_file)
    table = transform_table(table, add_activity_features=add_activity_features)
    pq.write_table(table, output_file)


def update_schema(input_dir: Path, output_dir: Path, add_activity_features: bool) -> None:
    """Copy and extend schema.json so training reads the new feature columns."""
    print(">>> 更新 schema.json")
    schema_path = input_dir / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"schema.json not found: {schema_path}")

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    schema["user_int"].extend([
        [HOUR_FID, 24, 1],
        [DOW_FID, 7, 1],
        [HOUR_DOW_FID, 168, 1],
    ])
    if add_activity_features:
        schema["user_dense"].extend([
            [SEQ_A_LEN_FID, 1],
            [SEQ_D_OVER_A_FID, 1],
        ])

    with (output_dir / "schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


def iter_parquet_files(input_dir: Path) -> Iterable[Path]:
    """Yield parquet files in deterministic sorted order."""
    return sorted(input_dir.glob("*.parquet"))


def rebuild_output_dir(output_dir: Path) -> None:
    """Create a clean output directory for processed parquet files."""
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic v33 row-level features.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--enable_dense_901_902",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to add user_dense_feats_901/902. 1=enabled, 0=disabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_activity_features = bool(args.enable_dense_901_902)

    print(
        ">>> activity dense features 901/902: "
        f"{'enabled' if add_activity_features else 'disabled'}"
    )
    rebuild_output_dir(args.output_dir)

    parquet_files = list(iter_parquet_files(args.input_dir))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.input_dir}")

    for input_file in parquet_files:
        output_file = args.output_dir / input_file.name
        process_parquet_file(input_file, output_file, add_activity_features)

    update_schema(args.input_dir, args.output_dir, add_activity_features)
    print(">>> 特征工程完成：已生成 processed parquet + schema.json")


if __name__ == "__main__":
    main()

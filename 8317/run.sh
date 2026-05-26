#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

INPUT_DATA_DIR="${TRAIN_DATA_PATH:-${SCRIPT_DIR}/data}"
PROCESSED_DATA_DIR="${SCRIPT_DIR}/processed_data"
ENABLE_DENSE_901_902="${ENABLE_DENSE_901_902:-1}"
EXCLUDE_USER_DENSE_87="${EXCLUDE_USER_DENSE_87:-1}"
EXCLUDED_USER_DENSE_FIDS=""
if [ "$EXCLUDE_USER_DENSE_87" = "1" ]; then
    EXCLUDED_USER_DENSE_FIDS="87"
fi

export TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-${SCRIPT_DIR}/ckpt}"
export TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${SCRIPT_DIR}/logs}"
export TRAIN_TF_EVENTS_PATH="${TRAIN_TF_EVENTS_PATH:-${SCRIPT_DIR}/tf_events}"

print_stage() {
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

print_stage "0. 清理 processed_data 特征缓存"
rm -rf "${PROCESSED_DATA_DIR}"

print_stage "1. 构建 v36 行级特征: 201/202/203 + 901/902 开关=${ENABLE_DENSE_901_902}"
python3 "${SCRIPT_DIR}/feature.py" \
    --input_dir "${INPUT_DATA_DIR}" \
    --output_dir "${PROCESSED_DATA_DIR}" \
    --enable_dense_901_902 "${ENABLE_DENSE_901_902}"

print_stage "2. 启动 v36 训练: v33 主体 + EMA0.9995 + 两个轻量正则微调"
export TRAIN_DATA_PATH="${PROCESSED_DATA_DIR}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --data_dir "${PROCESSED_DATA_DIR}" \
    --schema_path "${PROCESSED_DATA_DIR}/schema.json" \
    --ckpt_dir "${TRAIN_CKPT_PATH}" \
    --log_dir "${TRAIN_LOG_PATH}" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 3 \
    --item_ns_tokens 3 \
    --user_dense_pair_mode new \
    --user_dense_tokenizer_type aligned \
    --aligned_user_dense_fids 62,63,64,65,66 \
    --independent_user_dense_fids 61,87 \
    --enable_user_dense_smooth_norm \
    --user_dense_smooth_norm_fids 62,63,64,65,66 \
    --excluded_user_dense_fids "${EXCLUDED_USER_DENSE_FIDS}" \
    --num_queries 2 \
    --seq_interest_ratios 1.0,0.7,0.1 \
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
    --dropout_rate 0.015 \
    --precision bf16 \
    --lr_schedule cosine \
    --warmup_steps 500 \
    --ema_decay 0.9995 \
    --label_smoothing 0.005 \
    --weight_decay 0.02 \
    --loss_type bce_pairwise \
    --pairwise_lambda 0.05 \
    "$@"

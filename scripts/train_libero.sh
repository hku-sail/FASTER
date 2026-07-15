#!/bin/bash
# 本地 LIBERO 微调启动脚本
# 用法: bash scripts/train_libero.sh [config] [exp_name] [batch_size] [fsdp_devices] [额外参数]
# 示例: bash scripts/train_libero.sh pi05_libero run1 32 2
#       bash scripts/train_libero.sh pi05_libero run1 32 2 --resume
#       bash scripts/train_libero.sh pi05_libero run1 32 1 --no-wandb-enabled

set -e

export PATH="/root/.local/bin:$PATH"

CONFIG=${1:-pi05_libero}
EXP_NAME=${2:-run1}
BATCH_SIZE=${3:-256}
FSDP_DEVICES=${4:-2}
shift 4 2>/dev/null || true  # 剩余参数透传给 train.py

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENPI_DATA_HOME=/home/xhy/data/cache/openpi
export HF_LEROBOT_HOME=/home/xhy/data
export HF_DATASETS_CACHE=/home/xhy/data/cache/huggingface/datasets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_ENTITY=haoyuxiong

LOG_DIR="logs/${CONFIG}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Config:     $CONFIG"
echo "Exp:        $EXP_NAME"
echo "BatchSize:  $BATCH_SIZE"
echo "FSDPDevs:   $FSDP_DEVICES"
echo "Log:        $LOG_FILE"

uv run scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" \
    --project-name=openpi_libero \
    --fsdp-devices="$FSDP_DEVICES" \
    --batch-size="$BATCH_SIZE" \
    "$@" 2>&1 | tee "$LOG_FILE"

#!/bin/bash
# 本地 LIBERO 归一化统计预计算
# 用法: bash scripts/compute_norm_stats_libero.sh [pi05_libero|pi05_faster_libero]

set -e

CONFIG=${1:-pi05_libero}

export HF_LEROBOT_HOME=/home/xhy/data
export HF_DATASETS_CACHE=/home/xhy/data/cache/huggingface/datasets

echo "Config:  $CONFIG"
echo "LeRobot: $HF_LEROBOT_HOME"

uv run scripts/compute_norm_stats.py --config-name "$CONFIG"

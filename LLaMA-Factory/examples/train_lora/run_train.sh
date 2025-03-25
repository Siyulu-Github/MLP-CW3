#!/bin/bash

# 自动选择空闲端口（29000~29999）
function get_free_port() {
  while :; do
    PORT=$(shuf -i 29000-29999 -n 1)
    ss -tuln | grep -q ":$PORT " || break
  done
  echo $PORT
}

# 获取空闲端口
MASTER_PORT=$(get_free_port)
echo "[INFO] Using master port: $MASTER_PORT"

# 设置可选环境变量（建议开启）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 启动训练
torchrun \
  --nproc_per_node=4 \
  --master_port=$MASTER_PORT \
  -m llamafactory.cli train examples/train_lora/qwen2vl_lora_sft.yaml

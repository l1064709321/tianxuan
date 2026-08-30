#!/usr/bin/env bash
# 天玄训练启动 — 多神经协同 (MoE + Titans + STDP/STDA)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SANDBOX="$PROJECT_ROOT/sandbox"

cd "$SANDBOX"
rm -f logs/.train-paused

nohup node dist/ml/train.js \
  --tokens 150000 \
  --epochs 4 \
  --lr 0.002 \
  --ckpt 50 \
  --out ck-full-multi \
  --mamba 1 \
  --attn 1 \
  --cnn 1 \
  --moe 1 \
  --moetopk 2 \
  --titans 1 \
  --stdp 1 \
  --stda 1 \
  > logs/train-full.log 2>&1 &

echo "训练已启动(MoE多神经协同+Titans在线学习), PID=$!, 日志: logs/train-full.log"

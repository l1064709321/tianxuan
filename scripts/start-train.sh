#!/usr/bin/env bash
cd /home/aidlux/tianxuan/sandbox
rm -f logs/.train-paused
nohup node dist/ml/train.js \
  --tokens 150000 \
  --epochs 2 \
  --lr 0.002 \
  --ckpt 50 \
  --out ck-full-multi \
  --mamba 1 \
  --attn 0 \
  --stdp 1 \
  --stda 1 \
  > logs/train-full.log 2>&1 &
echo "训练已启动, PID=$!, 日志: logs/train-full.log"

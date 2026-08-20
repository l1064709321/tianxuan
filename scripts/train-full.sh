#!/usr/bin/env bash
# ============================================================
# 天玄完整训练脚本(可复现)
# 用法: bash scripts/train-full.sh [续训]
# 首次训练: bash scripts/train-full.sh
# 续训:     bash scripts/train-full.sh resume
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CK="sandbox/ck-full-multi"
LOG_DIR="sandbox/logs"

# 确保沙箱编译产物存在
if [ ! -d "sandbox/dist/ml" ]; then
  echo "[*] 首次运行, 编译 TypeScript → sandbox/dist ..."
  rm -rf sandbox/dist
  npx tsc --outDir sandbox/dist
fi

# 确保语料存在
if [ ! -f "data/corpus.txt" ] || [ "$(wc -c < data/corpus.txt)" -lt 100000 ]; then
  echo "[*] 生成训练语料(500万字符)..."
  node -e "
    const { buildCorpus } = require('./sandbox/dist/ml/data');
    const fs = require('fs');
    fs.mkdirSync('data', { recursive: true });
    const b = buildCorpus({ seed: 7, tokens: 5000000, rawPerFile: 30000 });
    fs.writeFileSync('data/corpus.txt', b.text, 'utf-8');
    console.log('语料: ' + b.text.length + ' 字符');
  "
fi

mkdir -p "$LOG_DIR"

if [ "${1:-}" = "resume" ] && [ -f "$CK/meta.json" ]; then
  echo "[*] 续训模式: 从 $CK 检查点继续"
  cd sandbox
  exec node ../sandbox/dist/ml/train.js \
    --resume ck-full-multi \
    --epochs 2 --lr 0.002 --ckpt 50 --out ck-full-multi
else
  echo "[*] 从头训练"
  cd sandbox
  exec node ../sandbox/dist/ml/train.js \
    --tokens 5000000 --epochs 2 --lr 0.002 \
    --hidden 256 --emb 64 --bptt 32 \
    --attn 1 --mamba 1 --cnn 1 \
    --ckpt 50 --out ck-full-multi
fi

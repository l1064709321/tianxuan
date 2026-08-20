#!/usr/bin/env bash
# 天玄沙箱验证: 所有环节(编译/训练/生成/实验)在 sandbox/ 内完成
# 编译产物 → sandbox/dist; 数据产物 → sandbox/data|sandbox/checkpoints; 不触碰项目根 dist/ 与 data/
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== [1/3] 编译检查 + 沙箱编译产物 =="
npx tsc --noEmit
rm -rf sandbox/dist
npx tsc --outDir sandbox/dist

echo "== [2/4] 数值梯度检查(GRU/SSM/Mamba/CNN) =="
( cd sandbox && node gradcheck.js && node gradcheck-ssm.js && node gradcheck-mamba-full.js && node gradcheck-cnn.js )

echo "== [3/4] 沙箱冒烟训练 =="
rm -rf sandbox/checkpoints sandbox/data sandbox/corpus.txt
mkdir -p sandbox
( cd sandbox && node ../sandbox/dist/ml/train.js \
    --tokens 1500 --rawperfile 0 --epochs 1 \
    --bptt 16 --hidden 96 --emb 32 --ctx 8 \
    --out checkpoints )

echo "== [4/4] 沙箱生成冒烟 =="
( cd sandbox && node ../sandbox/dist/ml/generate.js \
    --checkpoint checkpoints --prompt 小明 --len 40 --budget 3 )

echo "== 沙箱验证通过: 全部产物仅限 sandbox/ =="
find sandbox -maxdepth 2 -type d | sort

#!/usr/bin/env bash
# ============================================================
# 天玄 TianXuan · 一键复现脚本(VM / 任意 Linux)
# 用法: git clone <repo> && cd 天玄 && bash setup.sh
# 前置: Node.js ≥ 18 (推荐 22), npm ≥ 9, curl
# ============================================================
set -euo pipefail

echo "========================================="
echo "  天玄 TianXuan · 一键复现"
echo "========================================="

# --- 1. 环境检测 ---
echo "[1/6] 环境检测..."
if ! command -v node &>/dev/null; then
  echo "  [!] 未检测到 Node.js, 请先安装 Node.js ≥ 18"
  echo "      推荐: curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash - && sudo apt install -y nodejs"
  exit 1
fi

NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VER" -lt 18 ]; then
  echo "  [!] Node.js 版本过低 ($(node -v)), 需要 ≥ 18"
  exit 1
fi
echo "  Node.js $(node -v) ✓"
echo "  npm $(npm -v) ✓"

# --- 2. 安装依赖 ---
echo "[2/6] 安装 npm 依赖..."
npm install --no-audit --no-fund 2>&1 | tail -3
echo "  依赖安装完成 ✓"

# --- 3. 编译沙箱产物 ---
echo "[3/6] 编译 TypeScript → sandbox/dist ..."
rm -rf sandbox/dist
npx tsc --outDir sandbox/dist
echo "  编译完成 ✓"

# --- 4. 拉取公版中文语料 ---
echo "[4/6] 拉取公版中文语料(data/raw)..."
if [ -d "data/raw" ] && [ "$(ls data/raw/*.txt 2>/dev/null | wc -l)" -ge 5 ]; then
  echo "  语料已存在 ($(ls data/raw/*.txt | wc -l) 本书), 跳过下载"
else
  bash scripts/fetch-zh-classics.sh data/raw
fi
echo "  语料就绪 ✓"

# --- 5. 生成训练语料(data/corpus.txt) ---
echo "[5/6] 生成训练语料..."
if [ -f "data/corpus.txt" ] && [ "$(wc -c < data/corpus.txt)" -gt 100000 ]; then
  echo "  语料已存在 ($(wc -c < data/corpus.txt) 字符), 跳过"
else
  # 用与原始训练一致的参数: 500万字符
  node -e "
    const { buildCorpus } = require('./sandbox/dist/ml/data');
    const fs = require('fs');
    fs.mkdirSync('data', { recursive: true });
    const b = buildCorpus({ seed: 7, tokens: 5000000, rawPerFile: 30000 });
    fs.writeFileSync('data/corpus.txt', b.text, 'utf-8');
    console.log('  语料生成: ' + b.text.length + ' 字符');
  "
fi
echo "  语料就绪 ✓"

# --- 6. 拉取/验证现有检查点 ---
echo "[6/6] 检查模型检查点..."
CK="sandbox/ck-full-multi"
if [ -f "$CK/meta.json" ]; then
  LAST=$(python3 -c "import json; m=json.load(open('$CK/meta.json')); print(m.get('lastBatch',0))" 2>/dev/null || echo "?")
  PARAM=$(python3 -c "import json; m=json.load(open('$CK/meta.json')); print(m.get('paramCount',0))" 2>/dev/null || echo "?")
  echo "  检查点已存在: lastBatch=$LAST, 参数=$PARAM"
  echo "  续训命令: bash scripts/bootstrap-resume.sh"
else
  echo "  无检查点, 将从头训练"
  echo "  训练命令: cd sandbox && node ../sandbox/dist/ml/train.js --tokens 5000000 --epochs 2 --lr 0.002 --ckpt 50 --out ck-full-multi"
fi

echo ""
echo "========================================="
echo "  复现完成！"
echo "========================================="
echo ""
echo "常用命令:"
echo "  # 训练(从头)"
echo "  cd sandbox && node ../sandbox/dist/ml/train.js --tokens 5000000 --epochs 2 --out ck-full-multi"
echo ""
echo "  # 训练(续训, 从检查点)"
echo "  cd sandbox && node ../sandbox/dist/ml/train.js --resume ck-full-multi --epochs 2 --out ck-full-multi"
echo ""
echo "  # 生成"
echo "  cd sandbox && node ../sandbox/dist/ml/generate.js --checkpoint ck-full-multi --prompt '却说玄德' --len 120"
echo ""
echo "  # 建向量库"
echo "  cd sandbox && node ../sandbox/dist/ml/ingest.js --corpus ../data/corpus.txt --checkpoint ck-full-multi"
echo ""
echo "  # Web 服务"
echo "  cd sandbox && node ../sandbox/dist/server/index.js"
echo ""
echo "  # 沙箱验证(完整自检)"
echo "  bash scripts/sandbox-verify.sh"
echo ""
echo "  # 每日检查"
echo "  bash scripts/daily-check.sh"

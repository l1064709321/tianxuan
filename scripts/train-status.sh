#!/usr/bin/env bash
# 天玄训练实时状态: 解析后台训练日志 + 检查进程/检查点
LOG="sandbox/logs/train-full.log"
CK="sandbox/ck-full-multi"

echo "== 训练进程 =="
if pgrep -f 'dist/ml/train.js' > /dev/null; then
  echo "  RUNNING (pid: $(pgrep -f 'dist/ml/train.js' | head -1))"
else
  echo "  STOPPED"
fi

echo "== 最近进度(每 40 批刷新一次) =="
tail -40 "$LOG" | grep -E '批 [0-9]+/' | tail -1 \
  | sed -E 's/\[epoch ([0-9]+)\/([0-9]+)\] 批 ([0-9]+)\/([0-9]+) loss ([0-9.]+) \(([0-9.]+) batch\/s\)/  epoch \1\/\2 | batch \3\/\4 | loss \5 | \6 batch\/s/' \
  || echo "  (尚无进度输出)"

echo "== 检查点 =="
if [ -f "$CK/meta.json" ]; then
  node -e 'const m=require("./'+$CK+'/meta.json"); console.log("  已保存: epoch "+m.epochs+" | loss 见训练日志 | config hidden="+m.config.hidden+" attn="+(m.config.attn?1:0)+" mamba="+(m.config.mamba?1:0)+" cnn="+(m.config.cnn?1:0))'
else
  echo "  尚未保存(第一轮跑完自动保存)"
fi
echo "== 日志末尾 =="
tail -2 "$LOG"

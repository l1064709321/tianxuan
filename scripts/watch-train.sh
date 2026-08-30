#!/usr/bin/env bash
# 天玄训练看门狗 — 自动重启模式(从检查点续训,不丢进度)
# 每 WATCH_INTERVAL 秒巡检; 训练被杀/冻结时自动重启。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SANDBOX="$PROJECT_ROOT/sandbox"

LOG="$SANDBOX/logs/train-full.log"
EV="$SANDBOX/logs/train-events.log"
ST="$SANDBOX/logs/train-watch.json"
CK="$SANDBOX/ck-full-multi"
INTERVAL="${WATCH_INTERVAL:-120}"
AGE_LIMIT="${AGE_LIMIT:-1800}"

# 从检查点读取续训参数(永不从头训练)
read_ckpt_params() {
  if [ -f "$CK/meta.json" ]; then
    python3 - "$CK" << 'PYEOF'
import json, sys
m = json.load(open(sys.argv[1] + "/meta.json"))
cfg = m["config"]
args = "--resume %s --epochs 5 --lr 0.002 --ckpt 50 --out %s" % (sys.argv[1], sys.argv[1])
args += " --hidden %d --emb %d --bptt %d --ctx %d" % (cfg["hidden"], cfg["emb"], cfg["bptt"], cfg["ctx"])
args += " --attn %d --mamba %d --cnn %d" % (int(cfg.get("attn")), int(cfg.get("mamba")), int(cfg.get("cnn")))
args += " --moe 1 --moetopk %d --moegatehidden %d --moen %d --moelbweight %s" % (cfg.get("moeTopK",2), cfg.get("moeGateHidden",32), cfg.get("moeNExperts",10), cfg.get("moeLoadBalanceWeight",0.01))
args += " --titans %d --prediction 1 --replay 1 --dopamine 1" % (int(cfg.get("onlineTitans")),)
args += " --spikethreshold %s --warmupsteps 20" % cfg.get("spikeThreshold",0.3)
print(args)
PYEOF
  else
    echo "--tokens 500000 --epochs 3 --lr 0.002 --ckpt 50 --out $CK --hidden 64 --emb 32 --bptt 32 --ctx 8 --attn 1 --mamba 0 --cnn 1 --moe 1 --moetopk 2 --moegatehidden 32 --moen 10 --moelbweight 0.01 --titans 1 --prediction 1 --replay 1 --dopamine 1 --spikethreshold 0.3 --warmupsteps 20"
  fi
}

TRAIN_PARAMS=$(read_ckpt_params)
mkdir -p "$(dirname "$EV")" "$(dirname "$ST")"
touch "$EV"
echo "[watchdog] 检查点参数: $TRAIN_PARAMS"
echo "[watchdog] 巡检间隔=${INTERVAL}s, 冻结阈值=${AGE_LIMIT}s"

while true; do
  pid=$(pgrep -f '^node.*dist/ml/train\.js' | head -1)
  done=0; grep -q '训练完成' "$LOG" 2>/dev/null && done=1

  ts="$(date -Is)"
  last="$(grep -E '批 [0-9]+/' "$LOG" 2>/dev/null | tail -1)"
  ep=""; ba=""; lu=""; lo=""; sp=""
  if [[ "$last" =~ \[epoch\ +([0-9]+)/([0-9]+)\]\ 批\ ([0-9]+)/([0-9]+)\ loss\ ([0-9.]+)\ \(([0-9.]+)\ batch/s\) ]]; then
    ep="${BASH_REMATCH[1]}"; ba="${BASH_REMATCH[3]}"; lu="${BASH_REMATCH[4]}"; lo="${BASH_REMATCH[5]}"; sp="${BASH_REMATCH[6]}"
  fi

  if [ "$done" = "1" ]; then
    echo "$ts EVENT training-done" >> "$EV"
    echo "{\"ts\":\"$ts\",\"running\":0,\"done\":1,\"batch\":\"$ba\",\"stepTotal\":\"$lu\",\"loss\":\"$lo\",\"speed\":\"$sp\"}" > "$ST"
    exit 0
  fi

  if [ -z "$pid" ]; then
    echo "$ts EVENT auto-restart (进程消失, 自动重启)" >> "$EV"
    setsid bash -c "cd $SANDBOX && node dist/ml/train.js $TRAIN_PARAMS" >> "$LOG" 2>&1 < /dev/null &
    sleep 5
    newpid=$(pgrep -f '^node.*dist/ml/train\.js' | head -1)
    if [ -n "$newpid" ]; then
      echo "$ts EVENT restart-ok PID=$newpid" >> "$EV"
    else
      echo "$ts EVENT restart-failed" >> "$EV"
    fi
    echo "{\"ts\":\"$ts\",\"running\":0,\"restarting\":1,\"batch\":\"$ba\",\"stepTotal\":\"$lu\",\"loss\":\"$lo\",\"speed\":\"$sp\",\"checkpointSaved\":1,\"done\":0}" > "$ST"
  else
    mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    now=$(date +%s)
    age=$(( now - mtime ))
    if [ "$age" -gt "$AGE_LIMIT" ]; then
      kill -9 "$pid" 2>/dev/null
      sleep 2
      echo "$ts EVENT auto-restart (冻结 ${age}s, 自动重启)" >> "$EV"
      setsid bash -c "cd $SANDBOX && node dist/ml/train.js $TRAIN_PARAMS" >> "$LOG" 2>&1 < /dev/null &
      sleep 5
      newpid=$(pgrep -f '^node.*dist/ml/train\.js' | head -1)
      echo "$ts EVENT restart-after-freeze PID=$newpid" >> "$EV"
      echo "{\"ts\":\"$ts\",\"running\":0,\"frozen\":1,\"frozen_sec\":$age,\"batch\":\"$ba\",\"stepTotal\":\"$lu\",\"loss\":\"$lo\",\"speed\":\"$sp\",\"checkpointSaved\":1,\"done\":0}" > "$ST"
    else
      echo "{\"ts\":\"$ts\",\"running\":1,\"epoch\":\"$ep\",\"batch\":\"$ba\",\"stepTotal\":\"$lu\",\"loss\":\"$lo\",\"speed\":\"$sp\",\"checkpointSaved\":1,\"done\":0}" > "$ST"
    fi
  fi
  sleep "$INTERVAL"
done

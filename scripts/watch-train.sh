#!/usr/bin/env bash
# 天玄训练看门狗(暂停模式):
#   每 WATCH_INTERVAL 秒巡检; 训练被系统冻结/杀死时 → 自动挂暂停标志并退出(不自动续跑)。
#   恢复方式: rm -f /root/天玄/sandbox/logs/.train-paused && bash scripts/bootstrap-resume.sh
#   判定冻结: 训练进程存在但日志超过 AGE_LIMIT 秒未写入。
LOG="/root/天玄/sandbox/logs/train-full.log"
EV="/root/天玄/sandbox/logs/train-events.log"
ST="/root/天玄/sandbox/logs/train-watch.json"
CK="/root/天玄/sandbox/ck-full-multi"
FLAG="/root/天玄/sandbox/logs/.train-paused"
INTERVAL="${WATCH_INTERVAL:-300}"
AGE_LIMIT="${AGE_LIMIT:-3600}"

touch "$EV"

while true; do
  pid=$(pgrep -f '^node .*dist/ml/train\.js' | grep -v "$$" | head -1)
  done=0; grep -q '训练完成' "$LOG" 2>/dev/null && done=1

  ts="$(date -Is)"
  last="$(grep -E '批 [0-9]+/' "$LOG" 2>/dev/null | grep -v '已保存' | tail -1)"
  ep=""; ba=""; lu=""; lo=""; sp=""
  if [[ "$last" =~ \[epoch[[:space:]]+([0-9]+)/([0-9]+)\][[:space:]]+批[[:space:]]+([0-9]+)/([0-9]+)[[:space:]]+loss[[:space:]]+([0-9.]+)[[:space:]]+\(([0-9.]+)[[:space:]]+batch/s\) ]]; then
    ep="${BASH_REMATCH[1]}"; ba="${BASH_REMATCH[3]}"; lu="${BASH_REMATCH[4]}"; lo="${BASH_REMATCH[5]}"; sp="${BASH_REMATCH[6]}"
  fi

  if [ "$done" = "1" ]; then
    echo "$ts EVENT training-done" >> "$EV"
    echo "{\"ts\":\"$ts\",\"running\":0,\"done\":1,\"batch\":\"$ba\",\"stepTotal\":\"$lu\",\"loss\":\"$lo\",\"speed\":\"$sp\"}" > "$ST"
    exit 0
  fi

  if [ -z "$pid" ]; then
    touch "$FLAG"
    echo "$ts EVENT auto-pause (进程被杀/消失, 自动暂停, 等手动恢复)" >> "$EV"
    echo "{\"ts\":\"$ts\",\"running\":0,\"paused\":1,\"reason\":\"dead\",\"batch\":\"$ba\",\"stepTotal\":\"$lu\",\"loss\":\"$lo\",\"speed\":\"$sp\",\"checkpointSaved\":1,\"done\":0}" > "$ST"
    exit 0
  else
    mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    now=$(date +%s)
    age=$(( now - mtime ))
    if [ "$age" -gt "$AGE_LIMIT" ]; then
      kill -9 "$pid" 2>/dev/null
      sleep 2
      touch "$FLAG"
      echo "$ts EVENT auto-pause (冻结: 日志停 ${age}s, 自动暂停, 等手动恢复)" >> "$EV"
      echo "{\"ts\":\"$ts\",\"running\":0,\"paused\":1,\"reason\":\"frozen\",\"frozen_sec\":$age,\"batch\":\"$ba\",\"stepTotal\":\"$lu\",\"loss\":\"$lo\",\"speed\":\"$sp\",\"checkpointSaved\":1,\"done\":0}" > "$ST"
      exit 0
    else
      echo "{\"ts\":\"$ts\",\"running\":1,\"epoch\":\"$ep\",\"batch\":\"$ba\",\"stepTotal\":\"$lu\",\"loss\":\"$lo\",\"speed\":\"$sp\",\"checkpointSaved\":1,\"done\":0}" > "$ST"
    fi
  fi
  sleep "$INTERVAL"
done

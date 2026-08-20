#!/usr/bin/env bash
# 天玄启动钩子(幂等): 环境重启后, 只要本环境有任何终端/命令被触发,
# 若训练未在跑且有检查点 → 立即从断点续训并拉起看门狗。
# 训练已在跑或无事可做 → 秒退, 不干扰。已挂 ~/.profile 与 ~/.bashrc。
set -u

ROOT="/root/天玄"
LOG="$ROOT/sandbox/logs/train-full.log"
EV="$ROOT/sandbox/logs/train-events.log"
CK="$ROOT/sandbox/ck-full-multi"
LOCK="$ROOT/sandbox/logs/.bootstrap.lockdir"

# 用户暂停标志: 存在则不开机自动续跑(用户主动暂停, 需手动恢复)
[ -f "$ROOT/sandbox/logs/.train-paused" ] && exit 0
# 训练已在跑 → 什么都不做
pgrep -f 'dist/ml/train.js' >/dev/null 2>&1 && exit 0
# 没有检查点 → 无训练可续
[ -f "$CK/meta.json" ] || exit 0

# 互斥锁(防并发重复拉起); 残留锁且持有者已死则清除
if [ -d "$LOCK" ]; then
  oldpid=$(cat "$LOCK/pid" 2>/dev/null || echo 0)
  if ! kill -0 "$oldpid" 2>/dev/null; then
    rm -rf "$LOCK"
  else
    exit 0
  fi
fi
mkdir "$LOCK" 2>/dev/null || exit 0
echo $$ > "$LOCK/pid"

touch "$EV"
echo "$(date -Is) EVENT bootstrap-resume (启动钩子触达, 自动续跑)" >> "$EV"

# 1) 拉起训练(从检查点续跑)
setsid bash -c "cd $ROOT/sandbox && exec node ../sandbox/dist/ml/train.js --resume ck-full-multi --epochs 2 --lr 0.002 --ckpt 50 --out ck-full-multi" >> "$LOG" 2>&1 < /dev/null &

# 2) 确保看门狗在跑
pgrep -f 'watch-train.sh' >/dev/null 2>&1 || setsid bash "$ROOT/scripts/watch-train.sh" >/dev/null 2>&1 < /dev/null &

sleep 1
rm -rf "$LOCK"
exit 0

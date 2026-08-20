#!/usr/bin/env bash
# 天玄每日一致性检查: 对照 AGENTS.md 宪章, 机器可查项
set -uo pipefail
cd "$(dirname "$0")/.."
PASS=0; FAIL=0
ok(){ PASS=$((PASS+1)); echo "  [PASS] $1"; }
bad(){ FAIL=$((FAIL+1)); echo "  [FAIL] $1"; }

echo "== 天玄宪章每日检查 =="

# 1. 宪章文件存在
[ -f AGENTS.md ] && ok "AGENTS.md 宪章存在" || bad "AGENTS.md 缺失"

# 2. 根目录 dist/ 不应存在(编译产物只在 sandbox/dist)
[ ! -d dist ] && ok "根目录无 dist/(产物只在 sandbox)" || bad "根目录存在 dist/, 应删除"

# 3. 沙箱存在
[ -d sandbox ] && ok "sandbox/ 存在" || bad "sandbox/ 缺失"

# 4. 生产 data/ 未被验证污染: data/checkpoints 应只有 meta.json + model.json
if [ -d data/checkpoints ]; then
  cnt=$(ls data/checkpoints | grep -vc '^meta.json$\|^model.json$' || true)
  [ "$cnt" -eq 0 ] && ok "data/checkpoints 无污染产物" || bad "data/checkpoints 有 $cnt 个非预期文件"
fi

# 5. Task 带任务类型路由(多神经协同前提)
grep -q 'type: TaskType' src/station/types.ts && ok "任务类型路由已定义" || bad "Task.type 缺失"

# 6. 神经链按类型过滤
grep -q 'computeUnits(type: TaskType)' src/station/registry.ts && ok "神经链按类型路由" || bad "registry.computeUnits(type) 缺失"

# 7. 共享 z 走黑板(一次采样: z:logits)
grep -q 'z:logits:${task.id}' src/ml/engine.ts && ok "中间状态 z:logits 走黑板" || bad "z 黑板写入缺失"

# 8. stub 只允许在 demo.ts
stub=$(grep -rn '0.92\|0.45' src --include='*.ts' | grep -v 'src/station/demo.ts' | grep -v 'cascadeThreshold' | grep -v 'Math\|0.4 +\|0.85\|0.6\|maxConf' || true)
[ -z "$stub" ] && ok "真实路径无写死置信度 stub" || { echo "  [FAIL] 发现疑似 stub: $stub"; FAIL=$((FAIL+1)); }

# 9. 沙箱验证脚本存在
[ -f scripts/sandbox-verify.sh ] && ok "沙箱验证入口存在" || bad "scripts/sandbox-verify.sh 缺失"

# 10. CNN 训练开关存在(2026-08-12 修复: 此前无 --cnn, 消融无法复现)
grep -q 'const cnn = num("cnn", 0) === 1;' src/ml/train.ts && ok "CNN 训练开关 --cnn 存在" || bad "train.ts 缺少 --cnn 开关"

# 11. CNN 梯度检查脚本存在(防 CNN 反向静默回归)
[ -f sandbox/gradcheck-cnn.js ] && ok "CNN 梯度检查脚本存在" || bad "sandbox/gradcheck-cnn.js 缺失"

# 12. CNN 证据已量化(防回到 result=0/待消融)
grep -q 'result: 49.3' src/station/registry.ts && grep -q 'result: 49.3' src/ml/engine.ts && ok "CNN 证据已量化(49.3)" || bad "CNN 注册表证据缺失/回退"

# 13. README 结构清单新鲜(防 README 过期问题复发)
grep -q 'mamba.ts' README.md && grep -q 'titans.ts' README.md && grep -q 'probe.ts' README.md && ok "README 含多神经结构清单" || bad "README 结构清单过期"

# 14. 批级续训存在(防冻结后重跑开头, 2026-08-14 用户拍板方案)
grep -q lastBatch src/ml/train.ts && ok "批级续训 lastBatch 存在" || bad "train.ts 缺少批级续训"

# 15. 检查点原子写(防 kill 中断写坏 model.json)
grep -q model.json.tmp src/ml/train.ts && ok "检查点原子写存在" || bad "train.ts 缺少原子检查点写"

echo ""
echo "结果: $PASS 通过 / $FAIL 失败"
[ "$FAIL" -eq 0 ]

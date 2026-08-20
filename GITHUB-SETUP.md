# GitHub 仓库部署指南

## 快速复现（从 GitHub clone）

```bash
git clone <repo-url>
cd 天玄

# 一键部署
bash setup.sh

# 或手动：
npm install
npx tsc --outDir dist
```

## 已包含的训练好的模型

`checkpoints/` 目录包含训练好的模型检查点（786K 参数，epoch 2/2，验证 loss 3.093，top-1 41.8%）。

### 直接使用（无需训练）

```bash
# 生成文本
node dist/ml/generate.js --checkpoint checkpoints --prompt "却说玄德" --len 120

# Web 服务
node dist/server/index.js  # http://localhost:3800
```

### 从头训练（可选）

```bash
# 安装依赖 + 编译
npm install
npx tsc --outDir dist

# 拉取语料（11本公版中文经典）
bash scripts/fetch-zh-classics.sh data/raw

# 训练
bash scripts/train-full.sh
```

## 依赖

- Node.js ≥ 18（推荐 22）
- npm ≥ 9
- curl（语料下载）
- python3（可选，用于检查点读取）

## 文件大小

- `checkpoints/model.json`: 16MB（GitHub 单文件上限 100MB）
- `checkpoints/meta.json`: 4.5KB
- 仓库总大小：< 20MB（不含 node_modules）

## 注意事项

1. 训练好的模型是**可复现的**：相同的种子、语料、参数，训练结果一致
2. 语料是公版的（Project Gutenberg），可以重新下载
3. 检查点是标准 JSON，不依赖任何原生库，可移植到任何平台
4. 如需更新模型，训练后将新检查点复制到 `checkpoints/` 并 push

## 跨平台训练看门狗 (Windows / macOS / Linux)

看门狗用 Node.js 实现，任何平台都能跑，无需 bash/Git Bash。

```bash
# 看门狗模式: 每 5 分钟巡检, 训练被杀/冻结 → 自动暂停
npm run watchdog

# 单次巡检(配合 Windows 计划任务 / cron)
npm run watchdog:once

# 恢复训练: 清暂停标志 + 拉起训练 + 自动带上看门狗
npm run watchdog:resume
```

环境变量(可选):
- `WATCH_INTERVAL` 巡检间隔秒数(默认 300)
- `AGE_LIMIT` 冻结判定阈值秒数(默认 3600, 即 1 小时)
- `TIANXUAN_ROOT` 项目根路径(默认当前目录)

Windows 示例(计划任务):
```powershell
# 每 5 分钟跑一次单次巡检
schtasks /create /tn "TianXuanWatchdog" /sc minute /mo 5 /tr "node C:\tianxuan\sandbox\dist\ml\watchdog.js --once"
```

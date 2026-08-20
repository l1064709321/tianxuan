# 天玄 TianXuan

多神经协同字符级小语言模型 · 纯 TypeScript 数值内核(零原生依赖,CPU 可训可推) · 全局工作站(注册中心 + 黑板 z + MoD 式分级执行器) · 真实公版中文语料。

多神经协同不是 demo: 感知神经 → 中央神经 → 输出神经 经**共享工作空间 z(黑板)**协同, 神经间不得直接互调; 注册中心每个神经带证据门槛, 无量化证据不允许 enabled 进主线。

## 依赖

- Node.js >= 18 (推荐 22)
- npm >= 9
- curl (语料下载)

## 从 GitHub 部署 (任何平台: Windows / macOS / Linux)

```bash
git clone https://github.com/l1064709321/tianxuan.git
cd tianxuan

# 一键部署(安装依赖 + 编译 + 下载语料)
bash setup.sh
```

或手动:
```bash
npm install
npx tsc --outDir dist
bash scripts/fetch-zh-classics.sh data/raw   # 下载 11 部公版中文经典
```

`checkpoints/` 目录已包含训练好的模型(786K 参数, epoch 2/2, loss 3.093), 无需重新训练即可生成。

## 启动

```bash
# CLI 生成(使用已训练的模型)
node sandbox/dist/ml/generate.js --checkpoint checkpoints --prompt "却说玄德" --len 120

# Web 服务
node sandbox/dist/server/index.js            # http://localhost:3800

# 训练
bash scripts/train-full.sh                   # 从头训练
bash scripts/train-full.sh resume            # 从检查点续训
```

常用训练选项:
```bash
npm run train -- --tokens 500000 --epochs 5 --out data/checkpoints
npm run train -- --tokens 30000 --epochs 2 --hidden 96 --emb 32 --bptt 16 --mamba 1
```

**注意**: 所有验证/训练/生成/实验一律在 `sandbox/` 内完成(产物 `sandbox/dist` / `sandbox/checkpoints`), 生产 `data/` 与根目录 `dist/` 永不触碰(见「沙箱纪律」)。

## 跨平台训练看门狗 (Windows / macOS / Linux)

看门狗用 Node.js 实现, 任何平台都能跑, 无需 bash/Git Bash。

```bash
# 看门狗模式: 每 5 分钟巡检, 训练被杀/冻结 → 自动暂停(等你手动恢复)
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

Windows 计划任务示例:
```powershell
schtasks /create /tn "TianXuanWatchdog" /sc minute /mo 5 /tr "node C:\tianxuan\sandbox\dist\ml\watchdog.js --once"
```

## 实测基线(2026-08, 同机 CPU, 评估集口径不同不可直接横向比较)

| 模型 | 语料 | 参数 | 验证 loss | top-1 | 评估口径 |
|---|---|---:|---:|---:|---|
| MLP(旧对照) | 15 万字符 x 3 轮 | 30.5 万 | 3.10 | 47.7% | 评估以合成语料为主(易) |
| 双层 GRU | 15 万字符 x 4 轮 | 76.2 万 | 1.177 | 54.5% | 同上(易) |
| 双层 GRU | 30 万真实字符 x 4 轮 | 76.2 万 | 3.437 | 36.0% | 100% 古典小说, 更难 |
| 双层 GRU(生产当前) | 30 万真实字符 x 8 轮 | 76.2 万 | 3.413 | 35.7% | 续训 4 轮; 最佳 epoch7: 3.361/36.4% |
| Mamba 快动力学(沙箱) | 事件语料 30k x 2ep | -- | -- | 47.5% | GRU 基线 49.1%, 同配置持平 |

## 多神经协同现状(按证据门槛: 全部 pending, 未 enabled 进主线)

- **神经链路由**: 任务类型 text/speech/vision/world → 该领域优势神经链(借鉴 GPT-5 实时路由器逻辑); 链结构 = 感知神经 → 中央神经 → 输出神经, 只经黑板 z, 禁止直接互调。
- **已进代码的真神经**(全部过沙箱梯度检查 relErr=0.0000):
  - GRU(`src/ml/gru.ts`): 感知 L1 + 快动力学 L2, 重置门/循环梯度/Attention 历史反向均已修复。
  - Mamba(`src/ml/mamba.ts`): Delta/B/C 输入依赖的真选择性 SSM, `--mamba 1` 时替代 GRU cell2。
  - 稀疏 Attention(`src/ml/gru.ts` attn 路径): 层内调制 gate(慢语义), 点火 = gate。
  - Titans 在线神经记忆(`src/ml/titans.ts`): 持久记忆(动量)+ 深记忆槽(相似度检索), 生成期在线写入 L0 直出。
  - CNN(`src/ml/gru.ts` cnn 路径): 1D 卷积 n-gram 局部感知(待消融)。
- **证据门槛**: 注册表(`src/station/registry.ts`)每个系统带 `evidence`(metric/baseline/result/status); 文本链神经 status=pending(机制正确、不损害基线, 但暂无「修复某指标缺口」的量化证据) → 不 enabled; SNN/STDP/STDA/MoE 等无 CPU 文本世界收益场景 → evidence=fail, 保持货架。
- **协同验证**(Ignition Index 思路, `src/station/probe.ts`): 实测 attention 点火方差 0.012 > SSM 0.004, 符合论文「transformer 点火、SSM 近线性」; 预算转向让路测试 12 任务中 9 次正确跳过(审计记 `cold-central`)。

## 结构

- `src/ml/corpus.ts` — 文本世界 v1: 中文合成事件时序语料(临时可复现)
- `src/ml/data.ts` — 数据管线: 读取 `data/raw/*.txt`(Gutenberg 头尾自动剥离 + CJK 清洗), 真实语料优先, 合成兜底填预算
- `src/ml/gru.ts` — 字符级双层 GRU 序列模型: 截断 BPTT + Adam + 交叉熵; 附稀疏 Attention 层与 CNN 局部感知层; `embedAvg` 提供向量库嵌入
- `src/ml/mamba.ts` — 真选择性 SSM 层(Delta/B/C 输入依赖扫描), `--mamba` 开关
- `src/ml/titans.ts` — Titans 在线神经记忆(持久动量记忆 + 深记忆槽检索)
- `src/ml/tokenizer.ts` — 字符级分词器
- `src/ml/vectorstore.ts` — 纯 TS 向量库(余弦+持久化), 用模型嵌入(embedAvg)建库
- `src/ml/ingest.ts` — 向量库摄入: 语料切块 → 嵌入 → 写入 `data/vectorstore`
- `src/ml/engine.ts` — 推理引擎: 按 checkpoint 配置注册真实神经单元(GRU/Mamba/Attention/CNN)到工作站, 走黑板 z 协同 + Titans L0 记忆
- `src/ml/train.ts` / `src/ml/generate.ts` — 训练与 CLI 生成(真实走工作站分级路由)
- `src/ml/watchdog.ts` — 跨平台训练看门狗(Windows/macOS/Linux)
- `src/station/` — 全局工作站: 注册中心(`registry.ts`, 带证据门槛) + 黑板 z(`blackboard.ts`) + MoD 式分级执行器(`executor.ts`, 预算内扩深 + 按点火预算转向) + 协同探针(`probe.ts`) + 审计(`audit.ts`); `demo.ts` 是唯一允许 stub 的演示文件
- `src/server/` + `public/` — Express API 与 Web 产品壳

## 数据与记忆

- 真实数据: 内置 11 部全本公版古典名著于 `data/raw/`(Project Gutenberg 源, UTF-8):
  三国志演义 / 水浒传 / 红楼梦 / 西游记 / 儒林外史 / 聊斋志异 / 史记 / 东周列国志 / 隋唐演义 / 封神演义 / 世说新语,
  清洗后合计约 **548 万字符**; 拉取脚本 `scripts/fetch-zh-classics.sh`(可复现, 带重试与完整性校验)
- 向量库: `npm run ingest` 后, `GET /api/retrieve?q=关羽` 可检索语料片段; 每轮生成的新内容会**测试时记忆**回写向量库
  (`ingest --corpus data/corpus.txt` 可直接灌入指定语料, 保证与训练语料严格一致)
- L0 记忆: 生成时命中库内上下文片段, 零算力直出; Titans 在线记忆按上下文嵌入检索槽位, 权重 > 0.5 才直出
- 训练控制: `--tokens 500000 --epochs 5 --rawperfile 160000 --maxvocab 380 --emb 64 --hidden 256 --bptt 32`; 词表末位保留 UNK 槽(低频字统一入槽, 推理时禁出 UNK)
- 续训: `npm run train -- --resume data/checkpoints --epochs 4 --lr 0.001` 从检查点继续训练
  (要求 `data/corpus.txt` 与检查点 `corpusTokens` 一致; checkpoint 每轮自动保存)

## 沙箱纪律(宪章硬性要求)

- 所有验证/训练/生成/实验一律在 `sandbox/` 内: 编译产物 `sandbox/dist`, 数据产物 `sandbox/data|checkpoints`。
- 正式模型进沙箱跑 = 只读复制(`data/checkpoints` → `sandbox/`), 生产文件永不修改。
- 入口: `bash scripts/sandbox-verify.sh`(编译 + 冒烟训练 + 冒烟生成); 每日一致性检查 `bash scripts/daily-check.sh`(当前 15/15 过)。

## 说明

- 语料预算: 默认 15 万字符属于「超小训练量」; 按 Chinchilla(约 20 token/参数), 10M 模型约需 2 亿 token, 把真实文本放 `data/raw` 并调大 --tokens 即可
- 全量语料训练参考: `npm run train -- --tokens 5000000 --rawperfile 600000 --epochs 2`
  (548 万字符 x 2 轮, 本机 CPU 约每轮 7-10 小时, 检查点每轮保存可续)
- 官方依据(只借逻辑, 不抄代码): GPT-5 实时路由 / Anthropic 全局工作空间 / The Ignition Index / Titans / MoD / Worldscape-MoE, 详见 `PLAN.md`
- tfjs-node 原生绑定损坏不影响本项目: 数值内核纯 TS, 换机器无需重新编译

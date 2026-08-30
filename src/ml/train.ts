import * as fs from "fs";
import * as path from "path";
import { buildCorpus, chunkText } from "./data";
import { WorldCorpus, AssocCorpus, RecallCorpus } from "./corpus";
import { CharTokenizer } from "./tokenizer";
import { CharMultiNeuro, CharMultiNeuroConfig } from "./multineuro";
import { CharTransformer, CharTransformerConfig } from "./transformer";
import { MAX_PARAMS } from "./model";
import { mulberry32 } from "./rng";
import { STDP } from "./stdp";
import { STDA } from "./stda";
import { DataCleaner } from "./data_cleaner";
import { initLogger, TrainLogger, LogLevel } from "./logger";

function ts(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function flag(name: string): string | null {
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    if (args[i] === `--${name}`) return args[i + 1] ?? null;
    if (args[i].startsWith(`--${name}=`)) return args[i].slice(name.length + 3);
  }
  return null;
}

function num(name: string, def: number): number {
  const raw = flag(name);
  if (raw === null) return def;
  const n = Number(raw);
  return Number.isFinite(n) ? n : def;
}

function main(): number {
  const seed = num("seed", 7);
  const tokens = num("tokens", 150_000);
  const epochs = Math.max(1, num("epochs", 4));
  const lr = num("lr", 0.002);
  const ctx = Math.max(2, Math.min(64, num("ctx", 8)));
  const emb = Math.max(16, num("emb", 64));
  const hidden = Math.max(64, num("hidden", 256));
  const bptt = Math.max(8, Math.min(128, num("bptt", 32)));
  const maxVocab = Math.max(64, num("maxvocab", 380));
  const rawPerFile = num("rawperfile", 30_000);
  const attn = num("attn", 1) === 1;
  const attnBias = num("attnbias", 0) === 1;
  const mamba = num("mamba", 0) === 1;
  const cnn = num("cnn", 0) === 1;
  const stdp = num("stdp", 0) === 1;
  const stda = num("stda", 0) === 1;
  const stdpRate = num("stdprate", 0.001);
  const stdaRate = num("stdarate", 0.0005);
  const moe = num("moe", 0) === 1;
  const moeTopK = num("moetopk", 2);
  const moeGateHidden = num("moegatehidden", 32);
  const moeNExperts = num("moen", 50);
  const moeLoadBalanceWeight = num("moelbweight", 0.01);
  const titans = num("titans", 1) === 1;
  const prediction = num("prediction", 1) === 1;
  const predLossWeight = num("predlossweight", 0.01);
  const replay = num("replay", 1) === 1;
  const replayProb = num("replayprob", 0.1);
  const replayBatchSize = Math.max(1, num("replaybatchsize", 8));
  const dopamine = num("dopamine", 1) === 1;
  const dopamineTau = num("dopaminetau", 0.95);
  const spikeThreshold = num("spikethreshold", 0.3);
  const moeSpike = num("moespike", 0) === 1;
  const moeSpikeTemp = num("moespiketemp", 0.5);
  const moeSpikeTh = num("moespiketh", 0.15);
  // Transformer 骨干开关 (默认 GRU, --transformer 1 切换到 CharTransformer)
  const transformer = num("transformer", 0) === 1;
  const tNLayer = num("nlayers", 4);    // Transformer 层数
  const tNHead = num("nhead", 4);       // Transformer 头数
  const ckptEvery = Math.max(10, num("ckpt", 10));  // 每10批保存，防崩溃丢失
  const warmupSteps = num("warmupsteps", 20);  // Adam bias correction 预热步数
  const outDir = flag("out") ?? "data/checkpoints";
  const resumeDir = flag("resume");
  const cleanData = num("cleandata", 1) === 1;  // 是否启用数据清洗
  const anomalyThreshold = num("anomalythresh", 0.5);  // 异常阈值
  const logLevel = num("loglevel", 0) as LogLevel;  // 日志级别: 0=INFO, 1=WARN, 2=ERROR, 3=DEBUG, 4=DIAG
  const logFile = flag("log");  // 可选: 日志文件路径

  // 初始化日志系统
  const log = initLogger({ logLevel });
  if (logFile) {
    log.info(`日志文件: ${path.resolve(logFile)}`);
  }

  console.log("天玄 TianXuan · 多神经协同模型" + (moe ? " + MoE" : "") + (titans ? " + Titans" : "") + (stdp ? " + STDP" : "") + (stda ? " + STDA" : "") + (prediction ? " + 预测误差" : "") + (replay ? " + 重放" : "") + (dopamine ? " + 多巴胺" : "") + (spikeThreshold > 0 ? " + 脉冲门控" : "") + (moeSpike ? " + MoE脉冲" : ""));

  let text: string;
  let tokenizer = new CharTokenizer();
  let baseEpochs = 0;
  let startBatch = 0;

  if (resumeDir !== null) {
    const meta = JSON.parse(fs.readFileSync(path.join(resumeDir, "meta.json"), "utf-8")) as {
      config: CharMultiNeuroConfig;
      corpusTokens: number;
      epochs: number;
      vocab: string[];
      lastBatch?: number;
      t?: number;
      stdp?: boolean;
      stda?: boolean;
      stdpRate?: number;
      stdaRate?: number;
      moe?: boolean;
      moeTopK?: number;
      moeGateHidden?: number;
      moeLoadBalanceWeight?: number;
      titans?: boolean;
      titansWriteLog?: Array<{ slot: number; h1: number[] }>;
    };
    if (!fs.existsSync("data/corpus.txt")) {
      console.error("未找到 data/corpus.txt: 续训要求与原训练完全相同的语料");
      return 1;
    }
    text = fs.readFileSync("data/corpus.txt", "utf-8");
    if (text.length !== meta.corpusTokens) {
      console.error(`语料不一致: data/corpus.txt 为 ${text.length} 字符,原检查点为 ${meta.corpusTokens} 字符`);
      return 1;
    }
    tokenizer.load(meta.vocab);
    const cfg: CharMultiNeuroConfig = {
      ...meta.config,
      moeTopK: meta.moeTopK ?? moeTopK,
      moeGateHidden: meta.moeGateHidden ?? moeGateHidden,
      moeNExperts: meta.config.moeNExperts ?? moeNExperts,
      moeLoadBalanceWeight: meta.moeLoadBalanceWeight ?? moeLoadBalanceWeight,
      onlineTitans: meta.titans !== undefined ? meta.titans : titans,
      predictionHead: prediction,
      predLossWeight: predLossWeight,
      spikeThreshold: spikeThreshold > 0 ? spikeThreshold : undefined,
      replayCapacity: replay ? 4096 : undefined,
      replayProb: replay ? replayProb : undefined,
      replayBatchSize: replay ? replayBatchSize : undefined,
      dopamineTau: dopamine ? dopamineTau : undefined,
      moeSpikeTemperature: moeSpike ? moeSpikeTemp : undefined,
      moeSpikeThreshold: moeSpike ? moeSpikeTh : undefined,
    };
    const model = new CharMultiNeuro(cfg, seed);
    const params = JSON.parse(fs.readFileSync(path.join(resumeDir, "model.json"), "utf-8")) as number[];
    model.load(params);
    const trainedSteps = meta.t ?? 0;
    model.setSteps(trainedSteps);
    // [FIX] 同步 backbone Adam 步数: 否则 backbone 用 t=0 导致 bias_correction≈10x,
    // 与 MoE heads(t=211) 产生 lr 失配, backbone 剧烈震荡拖垮训练
    model.model.setSteps(trainedSteps);
    baseEpochs = meta.epochs ?? 0;
    startBatch = meta.lastBatch ?? 0;
    if (meta.titansWriteLog && (model as any).titans) {
      for (const rec of meta.titansWriteLog) {
        const h1 = new Float64Array(rec.h1);
        (model as any).titans.write(h1);
      }
      console.log(`续训: 恢复Titans记忆 ${meta.titansWriteLog.length} 条记录`);
    }
    console.log(`续训: 载入 ${resumeDir}(已训 ${baseEpochs} 轮, 本轮已训 ${startBatch} 批)→ 再训 ${epochs} 轮`);

    const useStdp = stdp || (meta.stdp ?? false);
    const useStda = stda || (meta.stda ?? false);
    const stdpRule = useStdp ? new STDP({ rateLTP: meta.stdpRate ?? stdpRate }) : null;
    const stdaRule = useStda ? new STDA(meta.stdaRate ?? stdaRate) : null;
    console.log(`STDP=${useStdp ? "on(rate="+stdpRate+")" : "off"} STDA=${useStda ? "on(rate="+stdaRate+")" : "off"}`);

    const ids = tokenizer.encode(text);
    const paramsCount = model.paramCount();
    if (paramsCount > MAX_PARAMS) {
      console.error(`参数超限: ${paramsCount.toLocaleString()} > ${MAX_PARAMS.toLocaleString()}`);
      return 1;
    }
    console.log(`词表: ${tokenizer.vocabSize} | 隐层: ${cfg.hidden} | BPTT: ${cfg.bptt} | 神经: attn=${cfg.attn ? 1 : 0} mamba=${cfg.mamba ? 1 : 0} cnn=${cfg.cnn ? 1 : 0} | MoE=${moe ? "on" : "off"} Titans=${cfg.onlineTitans ? "on" : "off"} | STDP=${useStdp ? "on" : "off"} STDA=${useStda ? "on" : "off"} | 预测=${prediction ? "on" : "off"} 重放=${replay ? "on" : "off"} 多巴胺=${dopamine ? "on" : "off"} | 参数: ${paramsCount.toLocaleString()}`);

    const len = cfg.bptt + 1;
    const seqs: Array<{ ids: number[] }> = [];
    for (let i = 0; i + len <= ids.length; i += len) {
      seqs.push({ ids: ids.slice(i, i + len) });
    }
    const trainCount = Math.floor(seqs.length * 0.95);
    const trainSeqs = seqs.slice(0, trainCount);
    const valSeqs = seqs.slice(trainCount);
    const batch = 16;
    const stepCount = Math.floor(trainSeqs.length / batch);
    const order = new Uint32Array(trainSeqs.length);
    for (let i = 0; i < order.length; i++) order[i] = i;
    const rng = mulberry32(seed + 2);
    const t0 = Date.now();

    const totalEpochs = baseEpochs + epochs;
    let currentBatch = startBatch; // 跨 epoch 维护批号，防止数据重复
    for (let ep = 1; ep <= epochs; ep++) {
      const fromBatch = ep === 1 ? currentBatch : 0;
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }
      for (let s = fromBatch; s < stepCount; s++) {
        const batchSeqs: Array<{ ids: number[] }> = [];
        for (let b = 0; b < batch; b++) batchSeqs.push(trainSeqs[order[s * batch + b]]);
        // Adam warmup: 前 warmupSteps 步线性提升 lr，避免 bias_correction 初期有效 lr 过小
        // [FIXED] 续训时从 startBatch 开始计数，避免 warmup 立即完成
        const totalStep = (baseEpochs * stepCount + startBatch) + (ep - 1) * stepCount + s;
        const warmupFactor = Math.min(1, (totalStep + 1) / Math.max(1, warmupSteps));
        const effectiveLr = lr * warmupFactor;
        const loss = model.trainStepBatch(batchSeqs, effectiveLr);

        // STDP 更新 (per-batch: trainStepBatch 内部已记录全部 timestep 的 h1/z1 历史,
        // 此处对整批一次性 apply，与 snn_trainer.ts 的 per-timestep apply 行为不同但等价)
       // 【已禁用】STDP/STDA 与 BPTT 目标冲突
       // STDP 优化 pre/post 相关性，BPTT 优化预测准确性
       // 两者同时修改 g 导致梯度冲突，训练不稳定
       // 如需使用，请改为后处理模式（训练完成后微调）
       /*
       if (stdpRule) {
         const lastH1 = model.model._stdpH1Hist[model.model._stdpH1Hist.length - 1];
         const lastPre = model.model._stdpPreHist[model.model._stdpPreHist.length - 1];
         if (lastH1 && lastPre) {
           stdpRule.record(lastH1, lastPre);
           stdpRule.apply(model.model.cell1, emb);
         }
         model.model.clearStdHist();
       }
       if (stdaRule && model.model._stdaLastH2) {
         stdaRule.update(model.model._stdaLastH2);
         stdaRule.apply(model.model.getGroups());
       }
       */

        // 训练日志
        log.logTrainStep(
          baseEpochs + ep,
          totalEpochs,
          s + 1,
          stepCount,
          loss,
          {
            gradNorm: Math.sqrt(model.model.getGroups().reduce((s, g) => s + g.g.reduce((a, b) => a + b * b, 0), 0)),
            stdpActive: useStdp,
            dopamine: model.getDopamineLevel(),
            replayOccupancy: model.getReplayOccupancy(),
          },
        );

        if (s % 40 === 0 || s === stepCount - 1) {
          const sps = (s + 1) / ((Date.now() - t0) / 1000);
          const moeSummary = model.moe.summary();
          console.log(`[${ts()}] [epoch ${baseEpochs + ep}/${totalEpochs}] 批 ${s + 1}/${stepCount} loss ${loss.toFixed(3)} (${sps.toFixed(1)} batch/s) | ${moeSummary}`);
        }
        currentBatch = s + 1; // 跟踪当前批号，防止跨 epoch 重复
        if (s > 0 && (s % ckptEvery === 0 || s === stepCount - 1)) {
          saveCheckpoint(outDir, model, tokenizer, paramsCount, text.length, baseEpochs + ep - 1, rawPerFile, currentBatch, {
            stdp: useStdp, stda: useStda, stdpRate, stdaRate,
            moe: true, moeTopK: cfg.moeTopK, moeGateHidden: cfg.moeGateHidden,
            moeLoadBalanceWeight: cfg.moeLoadBalanceWeight,
            titans: cfg.onlineTitans,
            prediction, replay, dopamine, spikeThreshold,
            moeSpike, moeSpikeTemp, moeSpikeTh,
          });
          console.log(`  [${ts()}] [周期检查点] epoch ${baseEpochs + ep} 批 ${s}/${stepCount} 已保存 → ${outDir}`);
        }
      }
      const ev = evaluate(model, valSeqs, tokenizer);
      log.logValidation(baseEpochs + ep, ev.loss, {
        top1: ev.acc,
        recall: ev.recall ?? undefined,
        perplexity: Math.exp(ev.loss),
      });

      const stdaMeta = stdaRule ? ` | STDA阈值=${stdaRule.threshold.toFixed(4)} 活性=${stdaRule.activity.toFixed(4)}` : "";
      const titansOcc = model.titans.occupancy().toFixed(3);
      const dmLevel = model.getDopamineLevel().toFixed(4);
      const replayOcc = model.getReplayOccupancy().toFixed(3);
      console.log(`[${ts()}] epoch ${baseEpochs + ep} 完成 → 验证损失 ${ev.loss.toFixed(3)} | top-1 准确率 ${(ev.acc * 100).toFixed(1)}%` +
        (ev.recall !== null ? ` | 就后召回 ${(ev.recall * 100).toFixed(1)}%` : "") +
        ` | Titans${titansOcc} 重放${replayOcc} 多巴胺${dmLevel}` + stdaMeta);
      saveCheckpoint(outDir, model, tokenizer, paramsCount, text.length, baseEpochs + ep, rawPerFile, 0, {
        stdp: useStdp, stda: useStda, stdpRate, stdaRate,
        moe: true, moeTopK: cfg.moeTopK, moeGateHidden: cfg.moeGateHidden,
        moeLoadBalanceWeight: cfg.moeLoadBalanceWeight,
        titans: cfg.onlineTitans,
        prediction, replay, dopamine, spikeThreshold,
        moeSpike, moeSpikeTemp, moeSpikeTh,
      });
    }
    console.log(`训练完成(共 ${totalEpochs} 轮),检查点已保存到 ` + path.resolve(outDir));
    return 0;
  }

  // ── 新训练 ──────────────────────────────────────────────────
  const corpusKind = flag("corpus") ?? "event";
  let bundle: { text: string; sources: string[] };
  if (corpusKind === "assoc") {
    const t = new AssocCorpus({ seed }).generate(tokens);
    bundle = { text: t, sources: ["assoc-recall"] };
  } else if (corpusKind === "recall") {
    const t = new RecallCorpus({ seed }).generate(tokens);
    bundle = { text: t, sources: ["recall-v2"] };
  } else {
    bundle = buildCorpus({ seed, tokens, rawPerFile });
  }
  text = bundle.text;
  console.log(`语料预算: ${tokens.toLocaleString()} 字符 | 来源: ${bundle.sources.join(", ")}`);

  // 数据清洗 (防御投毒攻击)
  if (cleanData) {
    const cleaner = new DataCleaner();
    const chunks = chunkText(text, 128, 64);  // 更大窗口
    let cleanCount = 0;
    let poisonCount = 0;

    for (const chunk of chunks) {
      const result = cleaner.detect(chunk);
      if (!result.isAnomalous) {
        cleanCount++;
      } else {
        poisonCount++;
      }
    }

    const totalChecked = cleanCount + poisonCount;
    console.log(`数据清洗: 正常 ${cleanCount} 块 | 异常 ${poisonCount} 块 (${totalChecked > 0 ? (poisonCount / totalChecked * 100).toFixed(1) : 0}%)`);
    if (poisonCount > 0) {
      console.log(`  ⚠️ 检测到异常数据，建议检查数据来源`);
    }
  }

  fs.mkdirSync(outDir, { recursive: true });
  fs.mkdirSync("data", { recursive: true });
  fs.writeFileSync("data/corpus.txt", text, "utf-8");
  tokenizer.fitTopN(text, maxVocab);

  // ── 模型选择: GRU(MultiNeuro) vs Transformer ──────────────
  let paramsCount = 0;
  let stepCount = 0;
  let currentBatch = 0;
  const ids = tokenizer.encode(text);

  if (transformer) {
    // ── CharTransformer 路径 ────────────────────────────────
    const tCfg: CharTransformerConfig = {
      vocabSize: tokenizer.vocabSize,
      emb: Math.max(64, emb),        // Transformer 需要更大 emb
      nLayer: tNLayer,
      nHead: tNHead,
      ctx,
      bptt,
    };
    const model = new CharTransformer(tCfg, seed + 1);
    paramsCount = model.paramCount();
    console.log(`[Transformer] 词表: ${tokenizer.vocabSize} | emb: ${tCfg.emb} | 层: ${tCfg.nLayer} | 头: ${tCfg.nHead} | BPTT: ${tCfg.bptt} | 参数: ${paramsCount.toLocaleString()}`);

    const len = bptt + 1;
    const seqs: Array<{ ids: number[] }> = [];
    for (let i = 0; i + len <= ids.length; i += len) {
      seqs.push({ ids: ids.slice(i, i + len) });
    }
    const trainCount = Math.floor(seqs.length * 0.95);
    const trainSeqs = seqs.slice(0, trainCount);
    const valSeqs = seqs.slice(trainCount);
    const batch = 8;  // Transformer 参数多，用更小的 batch
    stepCount = Math.floor(trainSeqs.length / batch);
    const order = new Uint32Array(trainSeqs.length);
    for (let i = 0; i < order.length; i++) order[i] = i;
    const rng = mulberry32(seed + 2);
    const t0 = Date.now();

    for (let ep = 1; ep <= epochs; ep++) {
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }
      for (let s = 0; s < stepCount; s++) {
        const batchSeqs: Array<{ ids: number[] }> = [];
        for (let b = 0; b < batch; b++) batchSeqs.push(trainSeqs[order[s * batch + b]]);
        const warmupFactor = Math.min(1, (ep * stepCount + s + 1) / Math.max(1, warmupSteps));
        const effectiveLr = lr * warmupFactor;
        const { loss } = model.trainStepBatch(batchSeqs, effectiveLr);

        if (s % 20 === 0) {
          const sps = (s + 1) / ((Date.now() - t0) / 1000);
          console.log(`[${ts()}] [epoch ${ep}/${epochs}] 批 ${s + 1}/${stepCount} loss ${loss.toFixed(3)} (${sps.toFixed(1)} batch/s) | attnGate=${model.lastAttnGate.toFixed(3)}`);
        }
        currentBatch = s + 1;
        if (s > 0 && (s % ckptEvery === 0 || s === stepCount - 1)) {
          saveTransformerCheckpoint(outDir, model as any, tokenizer, paramsCount, text.length, ep, rawPerFile, currentBatch, { transformer: true, nLayer: tNLayer, nHead: tNHead });
          console.log(`  [${ts()}] [周期检查点] epoch ${ep} 批 ${s}/${stepCount} 已保存 → ${outDir}`);
        }
      }
      const ev = evaluateTransformer(model as any, valSeqs, tokenizer);
      console.log(`[${ts()}] epoch ${ep} 完成 → 验证损失 ${ev.loss.toFixed(3)} | top-1 准确率 ${(ev.acc * 100).toFixed(1)}% | attnGate=${model.lastAttnGate.toFixed(3)}`);
      saveTransformerCheckpoint(outDir, model as any, tokenizer, paramsCount, text.length, ep, rawPerFile, 0, { transformer: true, nLayer: tNLayer, nHead: tNHead });
    }
    console.log(`训练完成(共 ${epochs} 轮, Transformer),检查点已保存到 ` + path.resolve(outDir));
    return 0;
  }

  // ── CharMultiNeuro (GRU 骨干) 路径 ────────────────────────
  const multiCfg: CharMultiNeuroConfig = {
    vocabSize: tokenizer.vocabSize,
    emb,
    hidden,
    ctx,
    bptt,
    attn,
    mamba,
    cnn,
    moeTopK,
    moeGateHidden,
    moeNExperts,
    moeLoadBalanceWeight,
    onlineTitans: titans,
    predictionHead: prediction,
    predLossWeight: predLossWeight,
    spikeThreshold: spikeThreshold > 0 ? spikeThreshold : undefined,
    replayCapacity: replay ? 4096 : undefined,
    replayProb: replay ? replayProb : undefined,
    replayBatchSize: replay ? replayBatchSize : undefined,
    dopamineTau: dopamine ? dopamineTau : undefined,
    moeSpikeTemperature: moeSpike ? moeSpikeTemp : undefined,
    moeSpikeThreshold: moeSpike ? moeSpikeTh : undefined,
  };

  const model = new CharMultiNeuro(multiCfg, seed + 1);
  model.model.fixedAttnBias = attnBias;

  const useStdp = stdp;
  const useStda = stda;
  const stdpRule = useStdp ? new STDP({ rateLTP: stdpRate }) : null;
  const stdaRule = useStda ? new STDA(stdaRate) : null;
  console.log(`STDP=${useStdp ? "on(rate="+stdpRate+")" : "off"} STDA=${useStda ? "on(rate="+stdaRate+")" : "off"}`);

  paramsCount = model.paramCount();
  if (paramsCount > MAX_PARAMS) {
    console.error(`参数超限: ${paramsCount.toLocaleString()} > ${MAX_PARAMS.toLocaleString()}`);
    return 1;
  }
  console.log(`词表: ${tokenizer.vocabSize} | 隐层: ${hidden} | BPTT: ${bptt} | 神经: attn=${attn ? 1 : 0} mamba=${mamba ? 1 : 0} cnn=${cnn ? 1 : 0} | MoE=${moe ? "on(TopK="+moeTopK+")" : "off"} Titans=${titans ? "on" : "off"} | 预测=${prediction ? "on" : "off"} 重放=${replay ? "on" : "off"} 多巴胺=${dopamine ? "on" : "off"} | 脉冲门控=${spikeThreshold > 0 ? "on(thr="+spikeThreshold+")" : "off"} | 参数: ${paramsCount.toLocaleString()}`);

  const len = bptt + 1;
  const seqs: Array<{ ids: number[] }> = [];
  for (let i = 0; i + len <= ids.length; i += len) {
    seqs.push({ ids: ids.slice(i, i + len) });
  }
  const trainCount = Math.floor(seqs.length * 0.95);
  const trainSeqs = seqs.slice(0, trainCount);
  const valSeqs = seqs.slice(trainCount);
  const batch = 16;
  stepCount = Math.floor(trainSeqs.length / batch);
  const order = new Uint32Array(trainSeqs.length);
  for (let i = 0; i < order.length; i++) order[i] = i;
  const rng = mulberry32(seed + 2);
  const t0 = Date.now();
  currentBatch = 0; // 跟踪当前批号，防止跨 epoch 重复

  for (let ep = 1; ep <= epochs; ep++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
    for (let s = 0; s < stepCount; s++) {
      const batchSeqs: Array<{ ids: number[] }> = [];
      for (let b = 0; b < batch; b++) batchSeqs.push(trainSeqs[order[s * batch + b]]);
      // Adam warmup: 前 warmupSteps 步线性提升 lr，避免 bias_correction 初期有效 lr 过小
      const warmupFactor = Math.min(1, (ep * stepCount + s + 1) / Math.max(1, warmupSteps));
      const effectiveLr = lr * warmupFactor;
      const loss = model.trainStepBatch(batchSeqs, effectiveLr);

      if (stdpRule) {
        const lastH1 = model.model._stdpH1Hist[model.model._stdpH1Hist.length - 1];
        const lastPre = model.model._stdpPreHist[model.model._stdpPreHist.length - 1];
        if (lastH1 && lastPre) {
          stdpRule.record(lastH1, lastPre);
          stdpRule.apply(model.model.cell1, emb);
        }
        model.model.clearStdHist();
      }
      if (stdaRule && model.model._stdaLastH2) {
        stdaRule.update(model.model._stdaLastH2);
        stdaRule.apply(model.model.getGroups());
      }

      if (s % 40 === 0) {
        const sps = (s + 1) / ((Date.now() - t0) / 1000);
        const moeSummary = model.moe.summary();
        console.log(`[${ts()}] [epoch ${ep}/${epochs}] 批 ${s + 1}/${stepCount} loss ${loss.toFixed(3)} (${sps.toFixed(1)} batch/s) | ${moeSummary}`);
      }
      currentBatch = s + 1; // 跟踪当前批号
      if (s > 0 && (s % ckptEvery === 0 || s === stepCount - 1)) {
        saveCheckpoint(outDir, model, tokenizer, paramsCount, text.length, ep, rawPerFile, currentBatch, {
          stdp: useStdp, stda: useStda, stdpRate, stdaRate,
          moe: true, moeTopK, moeGateHidden, moeLoadBalanceWeight,
          titans,
          prediction, replay, dopamine, spikeThreshold,
          moeSpike, moeSpikeTemp, moeSpikeTh,
        });
        console.log(`  [${ts()}] [周期检查点] epoch ${ep} 批 ${s}/${stepCount} 已保存 → ${outDir}`);
      }
    }
    const ev = evaluate(model, valSeqs, tokenizer);
    const stdaMeta = stdaRule ? ` | STDA阈值=${stdaRule.threshold.toFixed(4)} 活性=${stdaRule.activity.toFixed(4)}` : "";
    const titansOcc = model.titans.occupancy().toFixed(3);
    const dmLevel = model.getDopamineLevel().toFixed(4);
    const replayOcc = model.getReplayOccupancy().toFixed(3);
    console.log(`[${ts()}] epoch ${ep} 完成 → 验证损失 ${ev.loss.toFixed(3)} | top-1 准确率 ${(ev.acc * 100).toFixed(1)}%` +
      (ev.recall !== null ? ` | 就后召回 ${(ev.recall * 100).toFixed(1)}%` : "") +
      ` | Titans${titansOcc} 重放${replayOcc} 多巴胺${dmLevel}` + stdaMeta);
    saveCheckpoint(outDir, model, tokenizer, paramsCount, text.length, ep, rawPerFile, 0, {
      stdp: useStdp, stda: useStda, stdpRate, stdaRate,
      moe: true, moeTopK, moeGateHidden, moeLoadBalanceWeight,
      titans,
      prediction, replay, dopamine, spikeThreshold,
      moeSpike, moeSpikeTemp, moeSpikeTh,
    });
  }
  console.log(`训练完成(共 ${epochs} 轮, GRU骨干),检查点已保存到 ` + path.resolve(outDir));
  return 0;
}

function evaluate(model: any, valSeqs: Array<{ ids: number[] }>, tokenizer: CharTokenizer): { loss: number; acc: number; recall: number | null } {
  const v = model.cfg.vocabSize;
  const chars: number[] = [];
  for (const seq of valSeqs) for (const id of seq.ids) if (id < v) chars.push(id);
  if (chars.length < 32) return { loss: 0, acc: 0, recall: null };
  const maxChars = Math.min(chars.length - 1, 3000);
  const input = chars.slice(0, maxChars);
  const fw = model.forwardSeq(input);
  let loss = 0;
  let acc = 0;
  let recall = 0;
  let recallN = 0;
  const jiuId = tokenizer.idOf("\u5c31");
  const n = fw.length;
  for (let t = 0; t < n; t++) {
    const y = chars[t + 1];
    const pr = model.logitsToProbs(fw[t]);
    loss -= Math.log(pr[y] + 1e-12);
    let best = 0;
    for (let i = 1; i < pr.length; i++) if (pr[i] > pr[best]) best = i;
    if (best === y) acc += 1;
    if (jiuId !== undefined && chars[t] === jiuId && recallN < 400) {
      recallN += 1;
      if (best === y) recall += 1;
    }
  }
  return { loss: loss / n, acc: acc / n, recall: recallN > 0 ? recall / recallN : null };
}

function saveCheckpoint(
  dir: string,
  model: CharMultiNeuro,
  tokenizer: CharTokenizer,
  params: number,
  corpusTokens: number,
  epochs: number,
  rawPerFile: number,
  lastBatch = 0,
  extraMeta?: Record<string, unknown>,
): void {
  const meta = {
    config: model.cfg,
    paramCount: params,
    corpusTokens,
    epochs,
    rawPerFile,
    lastBatch,
    t: model.getSteps(),
    vocab: tokenizer.charset(),
    trainedAt: new Date().toISOString(),
    multiNeuro: true,
    moeTopK: model.cfg.moeTopK,
    moeGateHidden: model.cfg.moeGateHidden,
    moeNExperts: model.cfg.moeNExperts,
    moeLoadBalanceWeight: model.cfg.moeLoadBalanceWeight,
    onlineTitans: model.cfg.onlineTitans,
    ...(extraMeta ?? {}),
  };
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "model.json.tmp"), JSON.stringify(model.save()), "utf-8");
  fs.renameSync(path.join(dir, "model.json.tmp"), path.join(dir, "model.json"));
  fs.writeFileSync(path.join(dir, "meta.json.tmp"), JSON.stringify(meta, null, 2), "utf-8");
  fs.renameSync(path.join(dir, "meta.json.tmp"), path.join(dir, "meta.json"));
  // 保存世界模型 (如果存在)
  if ((model as any).worldModel) {
    try {
      (model as any).worldModel.save(path.join(dir, "..", "world_model"));
    } catch { /* 世界模型保存失败不影响主训练 */ }
  }
}

/** Transformer 检查点保存 */
function saveTransformerCheckpoint(
  dir: string,
  model: CharTransformer,
  tokenizer: CharTokenizer,
  params: number,
  corpusTokens: number,
  epochs: number,
  rawPerFile: number,
  lastBatch = 0,
  extraMeta?: Record<string, unknown>,
): void {
  const meta = {
    config: model.cfg,
    paramCount: params,
    corpusTokens,
    epochs,
    rawPerFile,
    lastBatch,
    t: (model as any).t,
    vocab: tokenizer.charset(),
    trainedAt: new Date().toISOString(),
    multiNeuro: false,
    transformer: true,
    ...(extraMeta ?? {}),
  };
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "model.json.tmp"), JSON.stringify(model.save()), "utf-8");
  fs.renameSync(path.join(dir, "model.json.tmp"), path.join(dir, "model.json"));
  fs.writeFileSync(path.join(dir, "meta.json.tmp"), JSON.stringify(meta, null, 2), "utf-8");
  fs.renameSync(path.join(dir, "meta.json.tmp"), path.join(dir, "meta.json"));
}

/** Transformer 专用评估 */
function evaluateTransformer(model: CharTransformer, valSeqs: Array<{ ids: number[] }>, tokenizer: CharTokenizer): { loss: number; acc: number; recall: number | null } {
  const v = model.cfg.vocabSize;
  const chars: number[] = [];
  for (const seq of valSeqs) for (const id of seq.ids) if (id < v) chars.push(id);
  if (chars.length < 32) return { loss: 0, acc: 0, recall: null };
  const maxChars = Math.min(chars.length - 1, 3000);
  const input = chars.slice(0, maxChars);

  // Transformer 需要逐步前向 (无 forwardSeq)
  let totalLoss = 0;
  let acc = 0;
  let recall = 0;
  let recallN = 0;
  const jiuId = tokenizer.idOf("\u5c31");
  const state = model.newState();

  for (let t = 0; t < input.length; t++) {
    const logits = model.step(input[t], state);
    const y = chars[t + 1];
    const pr = model.logitsToProbs(logits);
    totalLoss -= Math.log(pr[y] + 1e-12);
    let best = 0;
    for (let i = 1; i < pr.length; i++) if (pr[i] > pr[best]) best = i;
    if (best === y) acc += 1;
    if (jiuId !== undefined && chars[t] === jiuId && recallN < 400) {
      recallN += 1;
      if (best === y) recall += 1;
    }
  }
  const n = input.length;
  return { loss: totalLoss / n, acc: acc / n, recall: recallN > 0 ? recall / recallN : null };
}

process.exitCode = main();

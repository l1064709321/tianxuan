/**
 * SNN 训练入口 — 真正的脉冲神经网络在线学习
 *
 * ## 使用方式
 *
 * ```bash
 * # 基础训练
 * node dist/ml/snn/train.js --tokens 10000 --epochs 5
 *
 * # 启用 STDP
 * node dist/ml/snn/train.js --tokens 10000 --epochs 5 --stdp 1
 *
 * # 调整参数
 * node dist/ml/snn/train.js --tokens 10000 --epochs 5 --stdprate 0.001 --vthresh 1.5
 * ```
 */
import * as fs from "fs";
import * as path from "path";
import { buildCorpus } from "../data";
import { CharTokenizer } from "../tokenizer";
import { SpikingGRU } from "./spiking_gru";
import { SNNOnlineLearner } from "./snn_trainer";
import { mulberry32 } from "../rng";

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

function bool(name: string, def: boolean): boolean {
  const raw = flag(name);
  if (raw === null) return def;
  return raw === "1" || raw === "true";
}

function main(): number {
  // 解析参数
  const seed = num("seed", 42);
  const tokens = num("tokens", 10000);
  const epochs = Math.max(1, num("epochs", 5));
  const lr = num("lr", 0.002);
  const emb = Math.max(16, num("emb", 32));
  const hidden = Math.max(32, num("hidden", 128));
  const bptt = Math.max(8, Math.min(64, num("bptt", 32)));
  const ctx = Math.max(4, Math.min(32, num("ctx", 16)));
  const maxVocab = Math.max(64, num("maxvocab", 380));
  const rawPerFile = num("rawperfile", 30000);
  const outDir = flag("out") ?? "data/checkpoints";

  // SNN 参数
  const stdpRate = num("stdprate", 0.001);
  const vThresh = num("vthresh", 1.5);
  const tauMem = num("taumem", 10.0);
  const rInput = num("rinput", 2.0);
  const useStdp = bool("stdp", true);

  console.log(`\n${"=".repeat(60)}`);
  console.log("天玄 TianXuan · 脉冲神经网络 (SNN)");
  console.log(`${"=".repeat(60)}\n`);
  console.log(`配置:`);
  console.log(`  种子: ${seed}`);
  console.log(`  词表: ${maxVocab}`);
  console.log(`  嵌入: ${emb}`);
  console.log(`  隐藏: ${hidden}`);
  console.log(`  BPTT: ${bptt}`);
  console.log(`  上下文: ${ctx}`);
  console.log(`  学习率: ${lr}`);
  console.log(`  训练 token: ${tokens.toLocaleString()}`);
  console.log(`\nSNN 参数:`);
  console.log(`  STDP 学习率: ${stdpRate}`);
  console.log(`  启用 STDP: ${useStdp}`);
  console.log(`  LIF vThresh: ${vThresh}`);
  console.log(`  LIF tauMem: ${tauMem}`);
  console.log(`  LIF rInput: ${rInput}`);
  console.log();

  // 构建语料
  const bundle = buildCorpus({ seed, tokens, rawPerFile });
  const text = bundle.text;
  console.log(`语料预算: ${tokens.toLocaleString()} 字符 | 来源: ${bundle.sources.join(", ")}`);

  // 分词
  const tokenizer = new CharTokenizer();
  tokenizer.fitTopN(text, maxVocab);
  console.log(`词表大小: ${tokenizer.vocabSize}`);

  // 创建 SNN 模型
  const model = new SpikingGRU({
    inputSize: emb,
    hiddenSize: hidden,
    vocabSize: tokenizer.vocabSize,
    emb: emb,
    bptt: bptt,
    ctx: ctx,
    stdpRate: stdpRate,
    useStdp: useStdp,
    lif: {
      vThresh: vThresh,
      tauMem: tauMem,
      rInput: rInput,
    },
  });

  console.log(`参数数量: ${model.paramCount().toLocaleString()}`);
  console.log(`LIF 神经元数: ${hidden}`);
  console.log();

  // 创建训练器
  const learner = new SNNOnlineLearner(model, {
    stdpRateLTP: stdpRate,
    stdpRateLTD: stdpRate * 0.5,
  });

  // 准备数据
  const ids = tokenizer.encode(text);
  const len = bptt + 1;
  const seqs: Array<{ ids: number[] }> = [];
  for (let i = 0; i + len <= ids.length; i += len) {
    seqs.push({ ids: ids.slice(i, i + len) });
  }

  const trainCount = Math.floor(seqs.length * 0.95);
  const trainSeqs = seqs.slice(0, trainCount);
  const valSeqs = seqs.slice(trainCount);

  console.log(`训练样本: ${trainSeqs.length}, 验证样本: ${valSeqs.length}`);
  console.log();

  // 训练循环
  const t0 = Date.now();
  const order = new Uint32Array(trainSeqs.length);
  for (let i = 0; i < order.length; i++) order[i] = i;
  const rng = mulberry32(seed + 2);

  for (let ep = 1; ep <= epochs; ep++) {
    // Shuffle
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }

    let epochLoss = 0;
    let steps = 0;

    for (let s = 0; s < trainSeqs.length; s++) {
      const batchSeqs = [trainSeqs[order[s]]];
      const loss = learner.trainBatch(batchSeqs, lr);
      epochLoss += loss;
      steps++;

      if (s % 50 === 0) {
        const sps = steps / ((Date.now() - t0) / 1000);
        const m = learner.getMetrics();
        console.log(`[${ts()}] [epoch ${ep}/${epochs}] 批 ${s}/${trainSeqs.length} loss ${loss.toFixed(3)} (${sps.toFixed(1)} batch/s)`);
        console.log(`  稀疏度: ${m.sparsity.toFixed(4)} 近期发放率: ${m.recentFiringRate.toFixed(4)}`);
        console.log(`  多巴胺: ${m.dopamine.toFixed(4)} 重放占用: ${m.replayOccupancy.toFixed(3)}`);
      }
    }

    const avgLoss = epochLoss / steps;

    // 验证
    let valLoss = 0;
    let valChars = 0;
    for (const seq of valSeqs.slice(0, 10)) {
      const vl = learner.trainStepOnline(seq.ids, lr);
      valLoss += vl * (seq.ids.length - 1);
      valChars += seq.ids.length - 1;
    }
    valLoss /= Math.max(valChars, 1);

    const m = learner.getMetrics();
    console.log(`[${ts()}] epoch ${ep} 完成 → 训练损失 ${avgLoss.toFixed(3)} | 验证损失 ${valLoss.toFixed(3)}`);
    console.log(`  最终稀疏度: ${m.sparsity.toFixed(4)} | 总发放次数: ${model.totalSpikeCount}`);
    console.log();
  }

  console.log(`训练完成 (共 ${epochs} 轮),检查点已保存到 ${path.resolve(outDir)}`);

  // 保存检查点
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "snn_model.json"), JSON.stringify(model.save()), "utf-8");
  const meta = {
    config: model.cfg,
    paramCount: model.paramCount(),
    trainedAt: new Date().toISOString(),
    totalSpikes: model.totalSpikeCount,
    finalSparsity: model.sparsity(),
  };
  fs.writeFileSync(path.join(outDir, "snn_meta.json"), JSON.stringify(meta, null, 2), "utf-8");

  console.log(`\n检查点已保存:`);
  console.log(`  ${path.join(outDir, "snn_model.json")}`);
  console.log(`  ${path.join(outDir, "snn_meta.json")}`);

  return 0;
}

const code = main();
process.exitCode = code;

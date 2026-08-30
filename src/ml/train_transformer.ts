/**
 * train_transformer.ts — Transformer 训练入口
 *
 * 用法:
 *   npm run train:transformer          # 默认参数训练
 *   npm run train:transformer -- --tokens 5000000 --epochs 2
 */
import * as fs from "fs";
import * as path from "path";
import { buildCorpus } from "./data";
import { CharTransformer } from "./transformer";
import { CharTokenizer } from "./tokenizer";

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

function ts(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function main(): number {
  const seed = num("seed", 7);
  const tokens = num("tokens", 150_000);
  const epochs = Math.max(1, num("epochs", 4));
  const lr = num("lr", 0.001);
  const ctx = Math.max(16, Math.min(128, num("ctx", 64)));
  const emb = Math.max(128, num("emb", 256));
  const nLayer = Math.max(2, Math.min(12, num("nlayers", 6)));
  const nHead = Math.max(2, Math.min(8, num("nhead", 4)));
  const bptt = Math.max(8, Math.min(128, num("bptt", 32)));
  const maxVocab = Math.max(64, num("maxvocab", 380));
  const rawPerFile = num("rawperfile", 30_000);
  const ckptEvery = Math.max(10, num("ckpt", 10));
  const outDir = flag("out") ?? "data/checkpoints";

  console.log(`天玄 TianXuan · Transformer 语言模型`);
  console.log(`语料预算: ${tokens.toLocaleString()} 字符`);

  // 构建语料
  const bundle = buildCorpus({ tokens, seed, rawPerFile });
  const text = bundle.text;
  console.log(`语料来源: ${bundle.sources.join(", ")}`);

  // 分词器
  const tokenizer = new CharTokenizer();
  tokenizer.fitTopN(text, maxVocab);
  console.log(`词表: ${tokenizer.vocabSize}`);

  // 模型配置
  const modelCfg = {
    vocabSize: tokenizer.vocabSize,
    emb,
    nLayer,
    nHead,
    ctx,
    bptt,
  };

  // 创建模型
  const model = new CharTransformer(modelCfg, seed);
  const paramsCount = model.paramCount();
  console.log(`参数: ${paramsCount.toLocaleString()}`);
  console.log(`配置: emb=${emb} nLayer=${nLayer} nHead=${nHead} ctx=${ctx} bptt=${bptt}`);

  // 准备训练数据
  const ids = tokenizer.encode(text);
  const len = bptt + 1;
  const seqs: Array<{ ids: number[] }> = [];
  for (let i = 0; i + len <= ids.length; i += len) {
    seqs.push({ ids: ids.slice(i, i + len) });
  }
  const trainCount = Math.floor(seqs.length * 0.95);
  const trainSeqs = seqs.slice(0, trainCount);
  const valSeqs = seqs.slice(trainCount);
  const batch = 16;
  const stepCount = Math.floor(trainSeqs.length / batch);

  console.log(`训练集: ${trainSeqs.length} 序列 | 验证集: ${valSeqs.length} 序列`);
  console.log(`每轮 ${stepCount} 批, 共 ${epochs} 轮`);

  // 训练循环
  const order = new Uint32Array(trainSeqs.length);
  for (let i = 0; i < order.length; i++) order[i] = i;
  const rng = require("./rng").mulberry32(seed + 2);
  const t0 = Date.now();

  fs.mkdirSync(outDir, { recursive: true });

  for (let ep = 1; ep <= epochs; ep++) {
    // 打乱数据
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }

    for (let s = 0; s < stepCount; s++) {
      const batchSeqs: Array<{ ids: number[] }> = [];
      for (let b = 0; b < batch; b++) {
        batchSeqs.push(trainSeqs[order[s * batch + b]]);
      }

      const result = model.trainStepBatch(batchSeqs, lr);
      const loss = result.loss;

      if (s % 40 === 0 || s === stepCount - 1) {
        const sps = (s + 1) / ((Date.now() - t0) / 1000);
        console.log(`[${ts()}] [epoch ${ep}/${epochs}] 批 ${s + 1}/${stepCount} loss ${loss.toFixed(3)} (${sps.toFixed(1)} batch/s)`);
      }

      // 周期检查点
      if (s > 0 && s % ckptEvery === 0) {
        saveCheckpoint(outDir, model, tokenizer, paramsCount, text.length, ep, s);
      }
    }

    // 验证
    const ev = evaluate(model, valSeqs, tokenizer);
    console.log(`[${ts()}] epoch ${ep} 完成 → 验证损失 ${ev.loss.toFixed(3)} | top-1 准确率 ${(ev.acc * 100).toFixed(1)}% | perplexity ${Math.exp(ev.loss).toFixed(2)}`);

    // 保存检查点
    saveCheckpoint(outDir, model, tokenizer, paramsCount, text.length, ep, stepCount);
  }

  console.log(`训练完成(共 ${epochs} 轮),检查点已保存到 ${path.resolve(outDir)}`);
  return 0;
}

function evaluate(model: CharTransformer, valSeqs: Array<{ ids: number[] }>, tokenizer: CharTokenizer): { loss: number; acc: number } {
  const v = model.cfg.vocabSize;
  const chars: number[] = [];
  for (const seq of valSeqs) for (const id of seq.ids) if (id < v) chars.push(id);
  if (chars.length < 32) return { loss: 0, acc: 0 };

  const maxChars = Math.min(chars.length - 1, 2000);
  const input = chars.slice(0, maxChars);
  const states = []; let st = model.newState(); for (const id of input) { const logits = model.step(id, st); states.push(st.h.slice()); } const fw = states;
  let loss = 0;
  let acc = 0;

  for (let t = 0; t < fw.length; t++) {
    const y = chars[t + 1];
    const pr = model.logitsToProbs(fw[t]);
    loss -= Math.log(pr[y] + 1e-12);
    let best = 0;
    for (let i = 1; i < pr.length; i++) if (pr[i] > pr[best]) best = i;
    if (best === y) acc += 1;
  }

  return { loss: loss / fw.length, acc: acc / fw.length };
}

function saveCheckpoint(
  dir: string,
  model: CharTransformer,
  tokenizer: CharTokenizer,
  params: number,
  corpusTokens: number,
  epochs: number,
  lastBatch: number,
): void {
  const meta = {
    config: model.cfg,
    paramCount: params,
    corpusTokens,
    epochs,
    vocab: tokenizer.charset(),
    trainedAt: new Date().toISOString(),
    transformer: true,
    lastBatch,
    t: model["t"],
  };
  fs.writeFileSync(path.join(dir, "meta.json"), JSON.stringify(meta, null, 2), "utf-8");
  fs.writeFileSync(path.join(dir, "model.json.tmp"), JSON.stringify(model.save()), "utf-8");
  fs.renameSync(path.join(dir, "model.json.tmp"), path.join(dir, "model.json"));
  console.log(`  检查点已保存 → ${dir}`);
}

main();
process.exitCode = 0;

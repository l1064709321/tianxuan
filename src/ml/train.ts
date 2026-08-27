import * as fs from "fs";
import * as path from "path";
import { buildCorpus } from "./data";
import { WorldCorpus, AssocCorpus, RecallCorpus } from "./corpus";
import { CharTokenizer } from "./tokenizer";
import { CharGRU, CharGRUConfig } from "./gru";
import { MAX_PARAMS } from "./model";
import { mulberry32 } from "./rng";
import { STDP } from "./stdp";
import { STDA } from "./stda";

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
  const stdpRate = num("stdprate", 0.01);
  const stdaRate = num("stdarate", 0.005);
  const ckptEvery = Math.max(10, num("ckpt", 50));
  const outDir = flag("out") ?? "data/checkpoints";
  const resumeDir = flag("resume");

  console.log("天玄 TianXuan · 序列模型(双层 GRU)训练");

  let text: string;
  let tokenizer = new CharTokenizer();
  let cfg: CharGRUConfig;
  let model: CharGRU;
  let baseEpochs = 0;
  let startBatch = 0;

  if (resumeDir !== null) {
    const meta = JSON.parse(fs.readFileSync(path.join(resumeDir, "meta.json"), "utf-8")) as {
      config: CharGRUConfig;
      corpusTokens: number;
      epochs: number;
      vocab: string[];
      lastBatch?: number;
      t?: number;
      stdp?: boolean;
      stda?: boolean;
      stdpRate?: number;
      stdaRate?: number;
    };
    if (!fs.existsSync("data/corpus.txt")) {
      console.error("未找到 data/corpus.txt: 续训要求与原训练完全相同的语料,请先用原参数完整跑一次训练");
      return 1;
    }
    text = fs.readFileSync("data/corpus.txt", "utf-8");
    if (text.length !== meta.corpusTokens) {
      console.error(`语料不一致: data/corpus.txt 为 ${text.length} 字符,原检查点为 ${meta.corpusTokens} 字符;请用原参数重跑训练生成一致语料`);
      return 1;
    }
    tokenizer.load(meta.vocab);
    cfg = meta.config;
    model = new CharGRU(cfg, 1);
    const params = JSON.parse(fs.readFileSync(path.join(resumeDir, "model.json"), "utf-8")) as number[];
    model.load(params);
    baseEpochs = meta.epochs ?? 0;
    startBatch = meta.lastBatch ?? 0;
    const trainedSteps = meta.t ?? 0;
    model.setSteps(trainedSteps);
    console.log(`续训: 载入 ${resumeDir}(已训 ${baseEpochs} 轮, 本轮已训 ${startBatch} 批)→ 再训 ${epochs} 轮 | 语料 ${text.length.toLocaleString()} 字符(data/corpus.txt)`);
  } else {
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
    fs.mkdirSync(outDir, { recursive: true });
    fs.mkdirSync("data", { recursive: true });
    fs.writeFileSync("data/corpus.txt", text, "utf-8");
    tokenizer.fitTopN(text, maxVocab);
    cfg = { vocabSize: tokenizer.vocabSize, emb, hidden, ctx, bptt, attn, mamba, cnn };
    model = new CharGRU(cfg, seed + 1);
    model.fixedAttnBias = attnBias;
  }

  // 初始化 STDP / STDA 规则
  const useStdp = stdp || (resumeDir !== null && fs.existsSync(path.join(resumeDir, "meta.json")) && ((JSON.parse(fs.readFileSync(path.join(resumeDir, "meta.json"), "utf-8") as any)).stdp ?? false));
  const useStda = stda || (resumeDir !== null && fs.existsSync(path.join(resumeDir, "meta.json")) && ((JSON.parse(fs.readFileSync(path.join(resumeDir, "meta.json"), "utf-8") as any)).stda ?? false));
  const stdpRule = useStdp ? new STDP(stdpRate) : null;
  const stdaRule = useStda ? new STDA(stdaRate) : null;
  console.log(`STDP=${useStdp ? "on(rate="+stdpRate+")" : "off"} STDA=${useStda ? "on(rate="+stdaRate+")" : "off"}`);

  const ids = tokenizer.encode(text);
  const params = model.paramCount();
  if (params > MAX_PARAMS) {
    console.error(`参数超限: ${params.toLocaleString()} > ${MAX_PARAMS.toLocaleString()},请调小模型`);
    return 1;
  }
  console.log(`词表: ${tokenizer.vocabSize} | 隐层: ${cfg.hidden} | BPTT: ${cfg.bptt} | 神经: attn=${cfg.attn ? 1 : 0} mamba=${cfg.mamba ? 1 : 0} cnn=${cfg.cnn ? 1 : 0} | 参数: ${params.toLocaleString()} (上限 ${MAX_PARAMS.toLocaleString()})`);

  const len = cfg.bptt + 1;
  const seqs: Array<{ ids: number[] }> = [];
  for (let i = 0; i + len <= ids.length; i += len) {
    const chunk = ids.slice(i, i + len);
    seqs.push({ ids: chunk });
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
  for (let ep = 1; ep <= epochs; ep++) {
    const fromBatch = ep === 1 ? startBatch : 0;
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
    for (let s = fromBatch; s < stepCount; s++) {
      const batchSeqs: Array<{ ids: number[] }> = [];
      for (let b = 0; b < batch; b++) batchSeqs.push(trainSeqs[order[s * batch + b]]);
      const loss = model.trainStepBatch(batchSeqs, lr);
      // STDP/STDA 后处理
      if (stdpRule) {
        stdpRule.apply(model.getGroups());
        model.clearStdHist();
      }
      if (stdaRule && model._stdaLastH2) {
        stdaRule.update(model._stdaLastH2);
      }
      if (s % 40 === 0) {
        const sps = (s + 1) / ((Date.now() - t0) / 1000);
        console.log(`[${ts()}] [epoch ${baseEpochs + ep}/${totalEpochs}] 批 ${s + 1}/${stepCount} loss ${loss.toFixed(3)} (${sps.toFixed(1)} batch/s)`);
      }
  // 周期检查点: 每 ckptEvery 批存一次, 进程被冻结/回收时最多丢 ckptEvery 批, 不整轮丢失
      if (s > 0 && s % ckptEvery === 0) {
        saveCheckpoint(outDir, model, tokenizer, params, text.length, baseEpochs + ep - 1, rawPerFile, s + 1, {
          stdp: useStdp, stda: useStda, stdpRate, stdaRate
        });
        console.log(`  [${ts()}] [周期检查点] epoch ${baseEpochs + ep} 批 ${s}/${stepCount} 已保存 → ${outDir}`);
      }
    }
    const ev = evaluate(model, valSeqs, tokenizer);
    const stdaMeta = stdaRule ? { threshold: stdaRule.threshold.toFixed(4), activity: stdaRule.activity.toFixed(4) } : {};
    console.log(`[${ts()}] epoch ${baseEpochs + ep} 完成 → 验证损失 ${ev.loss.toFixed(3)} | top-1 准确率 ${(ev.acc * 100).toFixed(1)}%` + (ev.recall !== null ? ` | 就后召回 ${(ev.recall * 100).toFixed(1)}%` : "") + (stdaRule ? ` | STDA阈值=${stdaRule.threshold.toFixed(4)} 活性=${stdaRule.activity.toFixed(4)}` : ""));
    saveCheckpoint(outDir, model, tokenizer, params, text.length, baseEpochs + ep, rawPerFile, 0, {
      stdp: useStdp, stda: useStda, stdpRate, stdaRate
    });
  }
  console.log(`训练完成(共 ${totalEpochs} 轮),检查点已保存到 ` + path.resolve(outDir));
  return 0;
}

function evaluate(model: CharGRU, valSeqs: Array<{ ids: number[] }>, tokenizer: CharTokenizer): { loss: number; acc: number; recall: number | null } {
  const v = model.cfg.vocabSize;
  const chars: number[] = [];
  for (const seq of valSeqs) for (const id of seq.ids) if (id < v) chars.push(id);
  if (chars.length < 32) return { loss: 0, acc: 0, recall: null };
  const maxChars = Math.min(chars.length - 1, 3000);
  const input = chars.slice(0, maxChars);
  const fw = model.forwardSeq(input, model.cfg.cnn ? 4 : model.cfg.attn ? 3 : 2);
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
    // 就后召回探针: fw[t] 预测 chars[t+1]; chars[t]===就 时目标即"就后动词"(assoc 语料才有效)
    if (jiuId !== undefined && chars[t] === jiuId && recallN < 400) {
      recallN += 1;
      if (best === y) recall += 1;
    }
  }
  return { loss: loss / n, acc: acc / n, recall: recallN > 0 ? recall / recallN : null };
}

function saveCheckpoint(
  dir: string,
  model: CharGRU,
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
    t: model.steps,
    vocab: tokenizer.charset(),
    trainedAt: new Date().toISOString(),
    ...(extraMeta ?? {}),
  };
  fs.mkdirSync(dir, { recursive: true });
  // 原子写: 先写 .tmp 再 rename, 进程被 kill 时不会留下半截 model.json
  fs.writeFileSync(path.join(dir, "model.json.tmp"), JSON.stringify(model.save()), "utf-8");
  fs.renameSync(path.join(dir, "model.json.tmp"), path.join(dir, "model.json"));
  fs.writeFileSync(path.join(dir, "meta.json.tmp"), JSON.stringify(meta, null, 2), "utf-8");
  fs.renameSync(path.join(dir, "meta.json.tmp"), path.join(dir, "meta.json"));
}

process.exitCode = main();

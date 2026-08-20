import * as fs from "fs";
import * as path from "path";
import express from "express";
import { Engine } from "../ml/engine";

const CHECKPOINT_DIR = process.env.CHECKPOINT ?? path.join(process.cwd(), "data", "checkpoints");
const PORT = Number(process.env.PORT ?? 3800);

let engine: Engine | null = null;
try {
  engine = Engine.load(CHECKPOINT_DIR);
  console.log(`模型已加载: ${engine.model.paramCount().toLocaleString()} 参数, 词表 ${engine.tokenizer.vocabSize}`);
} catch (err) {
  console.error(`未找到可用模型(${(err as Error).message}).请先运行: npm run train`);
}

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "..", "..", "public")));

app.get("/api/health", (_req, res) => {
  if (!engine) {
    res.status(503).json({ ready: false, message: "未训练模型,请先运行 npm run train" });
    return;
  }
  res.json({
    ready: true,
    params: engine.model.paramCount(),
    vocabSize: engine.tokenizer.vocabSize,
    corpusTokens: engine.meta.corpusTokens,
    epochs: engine.meta.epochs,
    trainedAt: engine.meta.trainedAt,
    storeSize: engine.retrieveCount(),
  });
});

app.get("/api/retrieve", (req, res) => {
  if (!engine) {
    res.status(503).json({ error: "未训练模型" });
    return;
  }
  const q = String(req.query.q ?? "").slice(0, 64);
  if (q.length === 0) {
    res.status(400).json({ error: "缺少查询词 q" });
    return;
  }
  res.json({ hits: engine.retrieve(q, 5) });
});

app.post("/api/generate", async (req, res) => {
  if (!engine) {
    res.status(503).json({ error: "未训练模型,请先运行 npm run train" });
    return;
  }
  const body = (req.body ?? {}) as {
    prompt?: string;
    maxLen?: number;
    budget?: number;
    temperature?: number;
    topK?: number;
    seed?: number;
  };
  const prompt = typeof body.prompt === "string" ? body.prompt.slice(0, 128) : "小明";
  const maxLen = Math.min(800, Math.max(1, Math.floor(body.maxLen ?? 120)));
  const budget = Math.min(4, Math.max(1, Math.floor(body.budget ?? 3)));
  const temperature = Math.min(1.5, Math.max(0.1, body.temperature ?? 0.85));
  const topK = Math.min(30, Math.max(1, Math.floor(body.topK ?? 5)));
  const t0 = Date.now();
  try {
    const result = await engine.generate({
      prompt,
      maxLen,
      budget,
      temperature,
      topK,
      seed: body.seed,
    });
    res.json({ ...result, ms: Date.now() - t0 });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

app.listen(PORT, () => {
  console.log(`天玄工作站已上线: http://localhost:${PORT}`);
});

import * as fs from "fs";
import * as path from "path";
import express from "express";

const API_KEY = process.env.API_KEY ?? "";
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN ?? "*";
const CHECKPOINT_BASE = process.env.CHECKPOINT_BASE ?? path.join(process.cwd(), "data");
const CHECKPOINT_DIR = process.env.CHECKPOINT ?? path.join(CHECKPOINT_BASE, "checkpoints");
const PORT = Number(process.env.PORT ?? 3800);

// ── 路径校验: CHECKPOINT_DIR 必须位于 CHECKPOINT_BASE 之下 ──────────────────
function validateCheckpointDir(dir: string): string {
  const resolved = path.resolve(dir);
  const base = path.resolve(CHECKPOINT_BASE);
  if (!resolved.startsWith(base + path.sep) && resolved !== base) {
    throw new Error("CHECKPOINT_DIR 不允许跳出 CHECKPOINT_BASE: " + resolved);
  }
  return resolved;
}
const RESOLVED_CHECKPOINT_DIR = validateCheckpointDir(CHECKPOINT_DIR);

let engine: import("../ml/engine").Engine | null = null;

// ── CORS ─────────────────────────────────────────────────────────────────────
function applyCors(req: express.Request, res: express.Response, next: () => void): void {
  res.setHeader("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, X-API-Key");
  if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }
  next();
}

// ── 简单 API Key 鉴权 (空字符串 = 不启用) ───────────────────────────────────
function authMiddleware(req: express.Request, res: express.Response, next: () => void): void {
  if (!API_KEY) return next();
  const key = (req.headers["x-api-key"] as string | undefined) ?? req.query.api_key as string | undefined;
  if (key !== API_KEY) {
    res.status(401).json({ error: "unauthorized" });
    return;
  }
  next();
}

// ── 速率限制: 每 IP 每窗口最多 N 次请求 ──────────────────────────────────────
const RATE_LIMIT = Number(process.env.RATE_LIMIT ?? 60);       // 每分钟最大请求数
const RATE_WINDOW = 60_000;                                    // 窗口(ms)
const rateStores = new Map<string, { count: number; reset: number }>();

function rateLimitMiddleware(req: express.Request, res: express.Response, next: () => void): void {
  if (RATE_LIMIT <= 0) return next();
  const ip = req.headers["x-forwarded-for"]?.toString().split(",")[0]?.trim()
    ?? req.socket.remoteAddress ?? "unknown";
  const now = Date.now();
  let entry = rateStores.get(ip);
  if (!entry || now > entry.reset) {
    entry = { count: 0, reset: now + RATE_WINDOW };
    rateStores.set(ip, entry);
  }
  entry.count++;
  if (entry.count > RATE_LIMIT) {
    res.status(429).json({ error: "rate limit exceeded, retry after " + Math.ceil((entry.reset - now) / 1000) + "s" });
    return;
  }
  next();
}

// ── 启动服务 ─────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  const { Engine } = await import("../ml/engine") as { Engine: typeof import("../ml/engine").Engine };
  try {
    engine = Engine.load(RESOLVED_CHECKPOINT_DIR);
    console.log(`模型已加载: ${engine.totalParamCount().toLocaleString()} 参数, 词表 ${engine.tokenizer.vocabSize}`);
  } catch (err) {
    console.error(`未找到可用模型(${(err as Error).message}).请先运行: npm run train`);
  }

  const app = express();
  app.use(applyCors);
  app.use(rateLimitMiddleware);
  app.use(authMiddleware);
  app.use(express.json({ limit: "16kb" }));
  app.use(express.static(path.join(__dirname, "..", "..", "public")));

  // ── /api/health ────────────────────────────────────────────────────────────
  app.get("/api/health", (_req, res) => {
    if (!engine) {
      res.status(503).json({ ready: false, message: "未训练模型,请先运行 npm run train" });
      return;
    }
    res.json({
      ready: true,
      params: engine.totalParamCount(),
      vocabSize: engine.tokenizer.vocabSize,
      corpusTokens: engine.meta.corpusTokens,
      epochs: engine.meta.epochs,
      trainedAt: engine.meta.trainedAt,
      storeSize: engine.retrieveCount(),
    });
  });

  // ── /api/retrieve ──────────────────────────────────────────────────────────
  app.get("/api/retrieve", (req, res) => {
    if (!engine) { res.status(503).json({ error: "未训练模型" }); return; }
    const q = String(req.query.q ?? "").slice(0, 64);
    const k = Math.min(10, Math.max(1, Number(req.query.k ?? 4)));
    const category = String(req.query.category ?? "");
    if (q.length === 0) { res.status(400).json({ error: "缺少查询词 q" }); return; }
    const hits = engine.retrieve(q, k);
    const cnHits = engine.retrieveCN(q, k, category || undefined);
    const mathHits = engine.retrieveMath(q, k, category || undefined);
    res.json({ textHits: hits, cnHits, mathHits });
  });

  // ── /api/retrieve/math ─────────────────────────────────────────────────────
  app.get("/api/retrieve/math", (req, res) => {
    if (!engine) { res.status(503).json({ error: "未训练模型" }); return; }
    const q = String(req.query.q ?? "").slice(0, 64);
    const k = Math.min(10, Math.max(1, Number(req.query.k ?? 6)));
    const category = String(req.query.category ?? "");
    if (q.length === 0) { res.status(400).json({ error: "缺少查询词 q" }); return; }
    const hits = engine.retrieveMath(q, k, category || undefined);
    res.json({ hits, stats: engine.mathVectorStore?.stats() ?? null });
  });

  // ── /api/retrieve/cn ───────────────────────────────────────────────────────
  app.get("/api/retrieve/cn", (req, res) => {
    if (!engine) { res.status(503).json({ error: "未训练模型" }); return; }
    const q = String(req.query.q ?? "").slice(0, 64);
    const k = Math.min(10, Math.max(1, Number(req.query.k ?? 4)));
    const category = String(req.query.category ?? "");
    if (q.length === 0) { res.status(400).json({ error: "缺少查询词 q" }); return; }
    const hits = engine.retrieveCN(q, k, category || undefined);
    res.json({ hits, stats: engine.cnVectorStore?.stats() ?? null });
  });

  // ── /api/reason ────────────────────────────────────────────────────────────
  app.post("/api/reason", (req, res) => {
    if (!engine) { res.status(503).json({ error: "未训练模型" }); return; }
    const q = String(req.body?.query ?? req.query?.q ?? "").slice(0, 200);
    if (q.length === 0) { res.status(400).json({ error: "缺少 query 参数" }); return; }
    const t0 = Date.now();
    const result = engine.reason(q);
    res.json({ ...result, ms: Date.now() - t0 });
  });

  // ── /api/generate ──────────────────────────────────────────────────────────
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
    if (API_KEY) console.log("  [安全] API Key 鉴权已启用 (X-API-Key header)");
    else console.log("  [警告] 未设置 API_KEY, 所有接口无鉴权");
    console.log(`  [安全] 速率限制: ${RATE_LIMIT} 次/分钟 per IP`);
    console.log(`  [安全] CORS 允许来源: ${ALLOWED_ORIGIN}`);
  });
}

main().catch(console.error);

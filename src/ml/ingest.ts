/**
 * 向量库摄入工具: 从语料/数学库构建 HNSW 索引
 *
 * 用法:
 *   npm run ingest              # 从 data/corpus.txt 构建中文向量库
 *   npm run ingest -- --math 1  # 额外构建数学向量库
 *   npm run ingest -- --cn 1    # 构建 CNVectorStore (带 HNSW + N-gram)
 *   npm run ingest -- --file data/raw/红楼梦.txt
 */
import * as fs from "fs";
import * as path from "path";
import { buildCorpus, chunkText, loadRealFiles } from "./data";
import { CharGRU } from "./gru";
import { CharTokenizer } from "./tokenizer";
import { VectorStore } from "./vectorstore";
import { CNVectorStore } from "./cn_vectorstore";
import { MathVectorStore } from "./math_vectorstore";
import { MathCorpus } from "./math_corpus";

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
  const checkpointDir = flag("checkpoint") ?? "data/checkpoints";
  const cnStoreDir = flag("out") ?? "data/cn_vectorstore";
  const mathStoreDir = flag("mathout") ?? "data/math_vectorstore";
  const textStoreDir = flag("textout") ?? "data/vectorstore";
  const tokens = num("tokens", 150_000);
  const seed = num("seed", 7);
  const rawPerFile = num("rawperfile", 30_000);
  const chunkSize = num("chunk", 64);
  const corpusFile = flag("corpus");
  const buildMath = flag("math") === "1";
  const buildCN = flag("cn") === "1" || flag("math") === "1"; // 数学也需要 CN 库
  const mathCount = num("mathcount", 200); // 生成多少条数学知识

  // 加载模型
  const meta = JSON.parse(fs.readFileSync(path.join(checkpointDir, "meta.json"), "utf-8")) as {
    config: { vocabSize: number; emb: number; hidden: number; ctx: number; bptt: number };
    vocab: string[];
  };
  const params = JSON.parse(fs.readFileSync(path.join(checkpointDir, "model.json"), "utf-8")) as number[];
  const model = new CharGRU(meta.config, 1);
  model.load(params);
  const tokenizer = new CharTokenizer();
  tokenizer.load(meta.vocab);

  const embedText = (text: string): number[] => {
    const ids = tokenizer.encode(text);
    return model.embedAvg(ids);
  };

  // ── 1. 中文向量库 (基础版 VectorStore + HNSW) ─────────────
  let text: string;
  let sources: string[];
  if (corpusFile !== null) {
    text = fs.readFileSync(corpusFile, "utf-8");
    sources = [corpusFile];
  } else {
    const bundle = buildCorpus({ tokens, seed, rawPerFile });
    text = bundle.text;
    sources = bundle.sources;
  }

  const store = new VectorStore();
  const chunks = chunkText(text, chunkSize, chunkSize / 2);
  for (const chunk of chunks) {
    const ids = tokenizer.encode(chunk);
    store.add(chunk, embedText(chunk));
  }
  store.save(textStoreDir);
  console.log(`[中文向量库] 写入 ${store.size} 块 → ${textStoreDir} | 语料来源: ${sources.join(", ")}`);

  // 检索样例
  const probe = store.search(embedText("刘备"), 3);
  for (const hit of probe) {
    console.log(`  检索样例 score=${hit.score.toFixed(3)}: ${hit.text.slice(0, 50)}...`);
  }

  // ── 2. 中文语义向量库 (CNVectorStore + HNSW + N-gram) ──────
  if (buildCN) {
    const cnStore = new CNVectorStore(meta.config.emb);
    const categories = ["古典小说", "历史", "事件"];
    let catIdx = 0;
    for (const chunk of chunks) {
      const cat = categories[catIdx % categories.length];
      cnStore.add(chunk, embedText(chunk), cat, sources[catIdx % sources.length]);
      catIdx++;
    }
    cnStore.save(cnStoreDir);
    const stats = cnStore.stats();
    console.log(`[CN语义向量库] 写入 ${stats.total} 块 | 类别分布: ${JSON.stringify(stats.byCategory)}`);

    // 中文检索测试
    const cnProbe = cnStore.search(embedText("诸葛亮"), 3, "古典小说");
    for (const hit of cnProbe) {
      console.log(`  CN检索 score=${hit.score.toFixed(3)} [${hit.category}] ${hit.text?.slice(0, 40)}...`);
    }
  }

  // ── 3. 数学向量库 (MathCorpus + MathVectorStore) ───────────
  if (buildMath) {
    const mathCorpus = new MathCorpus(seed);
    const mathEntries = mathCorpus.generate(mathCount);

    // 用模型向量化数学表达式+描述
    const mathStore = new MathVectorStore();
    for (const entry of mathEntries) {
      const vec = embedText(`${entry.expression}。${entry.description}`);
      mathStore.add(entry, (_t) => vec);
    }
    mathStore.save(mathStoreDir);
    const mStats = mathStore.stats();
    console.log(`[数学向量库] 写入 ${mStats.total} 条 | 类别: ${JSON.stringify(mStats.byCategory)} | 平均难度: ${mStats.avgDifficulty.toFixed(1)}`);

    // 数学检索测试
    const msResults = mathStore.searchSemantic(embedText("勾股定理"), 5);
    for (const r of msResults) {
      console.log(`  数学检索 [${r.category}] ${r.expr} | ${r.description.slice(0, 30)}... (score=${r.score.toFixed(3)})`);
    }

    // 结构搜索测试
    const structResults = mathStore.searchStructural("a²+b²=c²", 3);
    for (const r of structResults) {
      console.log(`  结构检索 ${r.expr} ↔ ${r.description.slice(0, 30)}...`);
    }

    // 关键词搜索测试
    const kwResults = mathStore.searchKeyword("勾股定理", 3);
    for (const r of kwResults) {
      console.log(`  关键词检索 ${r.expr} | ${r.keywords.join(", ")}`);
    }

    // 综合搜索测试
    const hybrid = mathStore.search("勾股定理 直角三角形", embedText("勾股定理"), 5);
    for (const r of hybrid) {
      console.log(`  综合检索 [${r.matchType}] ${r.expr} | ${r.description.slice(0, 30)}`);
    }
  }

  console.log("\n摄入完成。");
  return 0;
}

process.exitCode = main();

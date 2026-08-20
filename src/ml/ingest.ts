import * as fs from "fs";
import * as path from "path";
import { buildCorpus, chunkText } from "./data";
import { CharGRU } from "./gru";
import { CharTokenizer } from "./tokenizer";
import { VectorStore } from "./vectorstore";

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

/** 摄入: 用已训练模型的字符嵌入,把语料切块向量化,写入向量库 */
function main(): number {
  const checkpointDir = flag("checkpoint") ?? "data/checkpoints";
  const storeDir = flag("out") ?? "data/vectorstore";
  const tokens = num("tokens", 150_000);
  const seed = num("seed", 7);
  const rawPerFile = num("rawperfile", 30_000);
  const chunkSize = num("chunk", 64);
  const corpusFile = flag("corpus");

  const meta = JSON.parse(fs.readFileSync(path.join(checkpointDir, "meta.json"), "utf-8")) as {
    config: { vocabSize: number; emb: number; hidden: number; ctx: number; bptt: number };
    vocab: string[];
  };
  const params = JSON.parse(fs.readFileSync(path.join(checkpointDir, "model.json"), "utf-8")) as number[];
  const model = new CharGRU(meta.config, 1);
  model.load(params);
  const tokenizer = new CharTokenizer();
  tokenizer.load(meta.vocab);

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
  for (const chunk of chunkText(text, chunkSize, chunkSize / 2)) {
    const ids = tokenizer.encode(chunk);
    store.add(chunk, model.embedAvg(ids));
  }
  store.save(storeDir);
  console.log(`向量库已写入 ${storeDir}: ${store.size} 块 | 语料来源: ${sources.join(", ")} | 总字符 ${text.length}`);
  const probe = store.search(model.embedAvg(tokenizer.encode("关羽")), 3);
  for (const hit of probe) {
    console.log(`  检索样例 score=${hit.score.toFixed(3)}: ${hit.text.slice(0, 40)}...`);
  }
  return 0;
}

process.exitCode = main();

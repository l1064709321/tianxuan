import * as path from "path";
import { Engine } from "./engine";

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

async function main(): Promise<number> {
  const checkpointDir = flag("checkpoint") ?? "data/checkpoints";
  const prompt = flag("prompt") ?? "小明";
  const maxLen = num("len", 120);
  const budget = Math.min(4, Math.max(1, num("budget", 3)));
  const temperature = num("temp", 0.85);
  const topK = Math.min(30, Math.max(1, num("topk", 5)));
  const seed = flag("seed") === null ? undefined : num("seed", 0);

  const engine = Engine.load(path.resolve(checkpointDir));
  const res = await engine.generate({ prompt, maxLen, budget, temperature, topK, seed });
  console.log("── 天玄生成 ──");
  console.log(res.text);
  console.log("");
  const byDepth = new Map<number, number>();
  let memHits = 0;
  for (const r of res.routes) {
    byDepth.set(r.depth, (byDepth.get(r.depth) ?? 0) + 1);
    if (r.memoryHit) memHits += 1;
  }
  const summary = [...byDepth.entries()].map(([d, c]) => `L${d}:${c}`).join(" ");
  console.log(`路由统计: ${summary} | 记忆直出 ${memHits} | 平均置信 ${(res.routes.reduce((s, r) => s + r.confidence, 0) / Math.max(res.routes.length, 1)).toFixed(2)}`);
  return 0;
}

main()
  .then((code) => {
    process.exitCode = code;
  })
  .catch((err: unknown) => {
    console.error(err);
    process.exitCode = 1;
  });

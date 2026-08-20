import * as fs from "fs";
import * as path from "path";
import { WorldCorpus } from "./corpus";

export interface DataSource {
  name: string;
  text: string;
}

/** 去掉 Project Gutenberg 头尾声明,保留正文 */
export function stripGutenberg(text: string): string {
  const startMark = /START OF (THE|THIS) PROJECT GUTENBERG EBOOK/i;
  const endMark = /END OF (THE|THIS) PROJECT GUTENBERG EBOOK/i;
  const lines = text.split(/\r?\n/);
  let started = false;
  const body: string[] = [];
  for (const line of lines) {
    if (!started && startMark.test(line)) {
      started = true;
      continue;
    }
    if (started && endMark.test(line)) break;
    if (started) body.push(line);
  }
  return body.join("\n").trim();
}

/** 只保留 CJK 汉字与中文标点(丢弃 ASCII 与内嵌 Gutenberg 贡献者注记) */
export function cleanCJK(text: string): string {
  return text.replace(/[^\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF\u2014\u2018-\u201D\u2026]/g, "");
}

/** 读取 data/raw 下的真实语料(*.txt) */
export function loadRealFiles(dir = "data/raw", maxPerFile = 40_000): DataSource[] {
  if (!fs.existsSync(dir)) return [];
  const out: DataSource[] = [];
  for (const file of fs.readdirSync(dir).filter((f) => f.endsWith(".txt"))) {
    const raw = fs.readFileSync(path.join(dir, file), "utf-8");
    const cleaned = cleanCJK(stripGutenberg(raw).replace(/\s+/g, ""));
    if (cleaned.length === 0) continue;
    out.push({ name: file, text: cleaned.slice(0, maxPerFile) });
  }
  return out;
}

export interface CorpusBuildOptions {
  /** 总字符预算 */
  tokens?: number;
  /** 合成语料(前 seed; 真实语料优先填满, 剩余用合成) */
  seed?: number;
  rawDir?: string;
  /** 每本真实书最多引入的字符数 */
  rawPerFile?: number;
}

export interface CorpusBundle {
  text: string;
  sources: string[];
}

/** 混合语料: 真实公版文本优先, 合成事件文本兜底补满预算 */
export function buildCorpus(options: CorpusBuildOptions = {}): CorpusBundle {
  const tokens = options.tokens ?? 150_000;
  const real = loadRealFiles(options.rawDir ?? "data/raw", options.rawPerFile ?? 30_000);
  const sources: string[] = [];
  let text = "";
  for (const r of real) {
    text += r.text;
    sources.push(r.name);
  }
  if (text.length >= tokens) {
    return { text: text.slice(0, tokens), sources };
  }
  const synth = new WorldCorpus({ seed: options.seed ?? 7, tokens: tokens - text.length }).generate(tokens - text.length);
  text += synth;
  sources.push("synthetic");
  return { text: cleanCJK(text).slice(0, tokens), sources };
}

/** 切块: 返回带来源号的有序片段 */
export function chunkText(text: string, size = 64, overlap = 32): string[] {
  const chunks: string[] = [];
  for (let i = 0; i + size <= text.length; i += size - overlap) {
    chunks.push(text.slice(i, i + size));
  }
  if (chunks.length === 0 && text.length > 0) chunks.push(text);
  return chunks;
}

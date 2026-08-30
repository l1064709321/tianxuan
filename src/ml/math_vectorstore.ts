/**
 * MathVectorStore — 数学表达式向量检索库
 *
 * 设计思路:
 * 1. 数学表达式不仅是文本,还有结构(运算符/变量/常量)
 * 2. 用 CharGRU 提取语义向量 + N-gram 哈希结构特征
 * 3. HNSW 索引加速检索
 * 4. 支持按公式类型(等式/不等式/定义/定理)分类检索
 * 5. 支持模糊匹配(相似结构不同数值)
 */
import * as fs from "fs";
import * as path from "path";
import { HNSWIndex } from "./hnsw";
import { MathEntry } from "./math_corpus";

export interface MathStoreEntry {
  id: string;
  expr: string;           // 原始表达式
  description: string;    // 中文描述
  category: string;       // 算术/代数/几何/数论/微积分/概率/三角/线代
  difficulty: number;
  keywords: string[];
  /** CharGRU embedAvg 语义向量 */
  semVec: Float64Array;
  /** N-gram 结构哈希 (用于快速结构相似匹配) */
  structHash: number;
  /** 表达式 token 化序列 */
  tokens: number[];
}

export interface MathSearchResult {
  id: string;
  expr: string;
  description: string;
  category: string;
  difficulty: number;
  keywords: string[];
  score: number;
  matchType: "semantic" | "structural" | "keyword";
}

function hashStruct(expr: string): number {
  // 提取运算符模式作为结构指纹
  let h = 0;
  const ops = "+-×÷=≠≤≥()²³√π".split("");
  for (const ch of expr) {
    if (ops.includes(ch)) {
      h = (h * 31 + ch.codePointAt(0)!) | 0;
    }
  }
  return h;
}

/** 从表达式提取 token 序列 (数字→连续, 运算符→单独, 字母→单独) */
function tokenizeExpr(expr: string): number[] {
  const tokens: number[] = [];
  let numBuf = "";
  for (const ch of expr) {
    if (/\d/.test(ch) || ch === ".") {
      numBuf += ch;
    } else {
      if (numBuf) { tokens.push(0xFF); numBuf = ""; } // 数字占位
      tokens.push(ch.codePointAt(0)! & 0xFF);
    }
  }
  if (numBuf) tokens.push(0xFF);
  return tokens;
}

/**
 * 数学向量库: 语义检索 + 结构匹配 + 关键词过滤
 */
export class MathVectorStore {
  private entries: MathStoreEntry[] = [];
  private hnsw: HNSWIndex | null = null;
  private keywordIndex: Map<string, number[]> = new Map();
  private _dirty = false;

  get size(): number { return this.entries.length; }
  get isEmpty(): boolean { return this.entries.length === 0; }

  /** 从 MathEntry 批量导入 */
  addBatch(entries: MathEntry[], embedFn: (text: string) => number[]): void {
    for (const e of entries) {
      this.add(e, embedFn);
    }
  }

  /** 添加一条数学知识 */
  add(entry: MathEntry, embedFn: (text: string) => number[]): void {
    const id = `math${this.entries.length}`;
    const semVec = new Float64Array(embedFn(`${entry.expression}。${entry.description}`));
    const tokenSeq = tokenizeExpr(entry.expression);
    const storeEntry: MathStoreEntry = {
      id,
      expr: entry.expression,
      description: entry.description,
      category: entry.category,
      difficulty: entry.difficulty,
      keywords: entry.keywords,
      semVec,
      structHash: hashStruct(entry.expression),
      tokens: tokenSeq,
    };
    this.entries.push(storeEntry);
    // 更新关键词索引
    for (const kw of entry.keywords) {
      if (!this.keywordIndex.has(kw)) this.keywordIndex.set(kw, []);
      this.keywordIndex.get(kw)!.push(this.entries.length - 1);
    }
    this._dirty = true;
  }

  /**
   * 语义搜索: 用 CharGRU 嵌入查询文本, 返回最相似的数学知识
   */
  searchSemantic(queryVec: number[], k = 4, category?: string): MathSearchResult[] {
    if (this.entries.length === 0) return [];
    if (this._dirty) this._buildIndex();
    const filtered = category
      ? this.entries.filter(e => e.category === category)
      : this.entries;
    if (filtered.length === 0) return [];

    const results = this.hnsw!.search(new Float64Array(queryVec), k);
    return results.map(r => {
      const e = this.entries.find(en => en.id === r.id);
      return e ? { ...this._toResult(e, r.score, "semantic") } : null;
    }).filter(Boolean) as MathSearchResult[];
  }

  /**
   * 结构搜索: 查找表达式结构相似的数学知识 (如 a²+b²=c² 匹配 x²+y²=r²)
   */
  searchStructural(queryExpr: string, k = 4, category?: string): MathSearchResult[] {
    if (this.entries.length === 0) return [];
    const qHash = hashStruct(queryExpr);
    const filtered = category
      ? this.entries.filter(e => e.category === category)
      : this.entries;
    if (filtered.length === 0) return [];

    const scored = filtered.map(e => ({
      e,
      score: 1.0 - Math.abs(e.structHash - qHash) / (Number.MAX_SAFE_INTEGER),
    })).sort((a, b) => b.score - a.score).slice(0, k);

    return scored.map(s => this._toResult(s.e, s.score, "structural"));
  }

  /**
   * 关键词搜索: 按关键词标签检索
   */
  searchKeyword(keyword: string, k = 4, category?: string): MathSearchResult[] {
    const ids = this.keywordIndex.get(keyword) ?? [];
    const filtered = category
      ? ids.filter(i => this.entries[i].category === category)
      : ids;
    return filtered.slice(0, k).map(i => this._toResult(this.entries[i], 1.0, "keyword"));
  }

  /** 综合搜索: 语义 + 结构 + 关键词 融合排序 */
  search(query: string, queryVec: number[], k = 6, category?: string): MathSearchResult[] {
    const semResults = this.searchSemantic(queryVec, k, category);
    const structResults = this.searchStructural(query, k, category);
    // 关键词提取
    const foundKw = this.entries.flatMap(e => e.keywords).filter(kw => query.includes(kw));
    const kwResults: MathSearchResult[] = [...new Set(foundKw)].flatMap(kw =>
      this.searchKeyword(kw, k, category)
    );

    // 融合去重
    const merged = new Map<string, MathSearchResult>();
    for (const r of [...semResults, ...structResults, ...kwResults]) {
      const existing = merged.get(r.id);
      if (!existing || r.score > existing.score) {
        merged.set(r.id, r);
      }
    }
    return [...merged.values()].sort((a, b) => b.score - a.score).slice(0, k);
  }

  /** 按类别列出 */
  byCategory(category: string): MathStoreEntry[] {
    return this.entries.filter(e => e.category === category);
  }

  /** 统计 */
  stats(): { total: number; byCategory: Record<string, number>; avgDifficulty: number } {
    const byCat: Record<string, number> = {};
    let totalDiff = 0;
    for (const e of this.entries) {
      byCat[e.category] = (byCat[e.category] ?? 0) + 1;
      totalDiff += e.difficulty;
    }
    return {
      total: this.entries.length,
      byCategory: byCat,
      avgDifficulty: this.entries.length > 0 ? totalDiff / this.entries.length : 0,
    };
  }

  save(dir: string): void {
    fs.mkdirSync(dir, { recursive: true });
    const data = {
      entries: this.entries.map(e => ({
        ...e,
        semVec: Array.from(e.semVec),
      })),
      keywordIndex: Object.fromEntries(
        [...this.keywordIndex.entries()].map(([k, v]) => [k, v])
      ),
    };
    fs.writeFileSync(path.join(dir, "math_store.json"), JSON.stringify(data), "utf-8");
    if (this.hnsw) this.hnsw.save(dir);
  }

  static load(dir: string): MathVectorStore | null {
    const file = path.join(dir, "math_store.json");
    if (!fs.existsSync(file)) return null;
    const data = JSON.parse(fs.readFileSync(file, "utf-8")) as {
      entries: Array<{ id: string; expr: string; description: string; category: string;
        difficulty: number; keywords: string[]; semVec: number[]; structHash: number; tokens: number[] }>;
      keywordIndex: Record<string, number[]>;
    };
    const store = new MathVectorStore();
    store.entries = data.entries.map(e => ({ ...e, semVec: new Float64Array(e.semVec) }));
    store.keywordIndex = new Map(Object.entries(data.keywordIndex));
    store.hnsw = HNSWIndex.load(dir);
    store._dirty = !store.hnsw; // HNSW 加载失败则标记脏, 下次搜索时重建
    return store;
  }

  private _buildIndex(): void {
    this.hnsw = new HNSWIndex({ M: 16, maxLevel: 6, efConstruction: 80, efSearch: 40 });
    for (const e of this.entries) {
      this.hnsw.add({ id: e.id, text: `${e.expr}。${e.description}`, vec: e.semVec });
    }
    this._dirty = false;
  }

  private _toResult(e: MathStoreEntry, score: number, matchType: MathSearchResult["matchType"]): MathSearchResult {
    return { ...e, score, matchType };
  }
}

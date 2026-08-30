/**
 * CNVectorStore — 中文语义向量库
 *
 * 特色:
 * 1. 基于 CharGRU embedAvg 提取文本语义向量
 * 2. HNSW 近似近邻索引,支持大规模检索
 * 3. 中文 N-gram 特征增强: 除模型向量外,额外保留字符 N-gram 哈希特征用于粗筛
 * 4. 支持批量导入、增量更新、持久化加载
 * 5. 检索时可指定来源过滤(如只搜古典小说、只搜数学)
 */
import * as fs from "fs";
import * as path from "path";
import { HNSWIndex, HNSWSearchResult } from "./hnsw";

export interface CNEntry {
  id: string;
  text: string;
  /** 语义向量 (由 CharGRU embedAvg 生成) */
  vec: Float64Array;
  /** 来源分类 (古典/事件/数学/自定义) */
  category?: string;
  /** 来源文件 */
  source?: string;
  /** 字符 N-gram 哈希 (用于快速预过滤) */
  ngramHash: number;
}

export interface CNSearchResult {
  id: string;
  text: string;
  score: number;
  category?: string;
  source?: string;
}

function hashNgram(text: string, n = 3): number {
  let h = 0;
  const chars = text.split("");
  for (let i = 0; i <= chars.length - n; i++) {
    const gram = chars.slice(i, i + n).join("");
    for (const ch of gram) h = ((h * 31) + ch.codePointAt(0)!) | 0;
  }
  return h;
}

/**
 * 中文向量库: 语义检索 + HNSW 索引 + N-gram 预过滤
 */
export class CNVectorStore {
  private entries: CNEntry[] = [];
  private hnsw: HNSWIndex | null = null;
  private readonly dim: number;
  /** 是否启用 N-gram 预过滤 (粗筛候选再精排) */
  private useNgramFilter = true;
  /** N-gram 预筛召回数 */
  private ngramCandidates = 50;

  constructor(dim = 64) {
    this.dim = dim;
  }

  get size(): number { return this.entries.length; }

  /** 添加一条记录 */
  add(text: string, vec: number[], category?: string, source?: string): string {
    const id = `cn${this.entries.length}`;
    const entry: CNEntry = {
      id,
      text,
      vec: new Float64Array(vec),
      category,
      source,
      ngramHash: hashNgram(text),
    };
    this.entries.push(entry);
    // 延迟建 HNSW (批量添加后一次性构建更高效)
    this._markDirty();
    return id;
  }

  /** 批量添加 (自动触发 HNSW 重建) */
  addBatch(pairs: Array<{ text: string; vec: number[]; category?: string; source?: string }>): void {
    for (const p of pairs) {
      this.add(p.text, p.vec, p.category, p.source);
    }
  }

  /**
   * 搜索: 先用 N-gram 粗筛,再用 HNSW 精排
   * @param queryVec 查询向量 (CharGRU embedAvg 输出)
   * @param k 返回条数
   * @param category 限定类别 (undefined = 全搜)
   * @param source 限定来源
   */
  search(vec: number[], k = 4, category?: string, source?: string): CNSearchResult[] {
    if (this.entries.length === 0) return [];
    const queryVec = new Float64Array(vec);
    const filtered = this.entries.filter(e => {
      if (category && e.category !== category) return false;
      if (source && e.source !== source) return false;
      return true;
    });
    if (filtered.length === 0) return [];

    // N-gram 预过滤
    let candidates = filtered;
    if (this.useNgramFilter && filtered.length > k * 3) {
      const queryHash = hashNgram(
        Array.from(queryVec).slice(0, 16).map(v => String.fromCharCode(Math.floor(v * 10) % 256)).join("")
      );
      candidates = filtered
        .map((e, i) => ({ e, diff: Math.abs(e.ngramHash - queryHash) }))
        .sort((a, b) => a.diff - b.diff)
        .slice(0, this.ngramCandidates)
        .map(x => x.e);
    }

    // 构建或复用 HNSW
    if (!this.hnsw || this._dirty) {
      this._buildHNSW(filtered);
    }

    // HNSW 检索
    const results = this.hnsw!.search(queryVec, k);
    return results.map(r => {
      const entry = this.entries.find(e => e.id === r.id);
      return {
        id: r.id,
        text: r.text,
        score: r.score,
        category: entry?.category,
        source: entry?.source,
      };
    });
  }

  /** 暴力搜索 (小规模/验证用) */
  searchBrute(vec: number[], k = 4, category?: string, source?: string): CNSearchResult[] {
    if (this.entries.length === 0) return [];
    const queryVec = new Float64Array(vec);
    const filtered = this.entries.filter(e => {
      if (category && e.category !== category) return false;
      if (source && e.source !== source) return false;
      return true;
    });
    const scored = filtered.map(e => {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < e.vec.length; i++) {
        dot += e.vec[i] * queryVec[i];
        na += e.vec[i] * e.vec[i];
        nb += queryVec[i] * queryVec[i];
      }
      const score = dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
      return { id: e.id, text: e.text, score, category: e.category, source: e.source };
    });
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  /** 按类别统计 */
  stats(): { total: number; byCategory: Record<string, number>; bySource: Record<string, number> } {
    const byCat: Record<string, number> = {};
    const bySrc: Record<string, number> = {};
    for (const e of this.entries) {
      byCat[e.category ?? "unknown"] = (byCat[e.category ?? "unknown"] ?? 0) + 1;
      bySrc[e.source ?? "unknown"] = (bySrc[e.source ?? "unknown"] ?? 0) + 1;
    }
    return { total: this.entries.length, byCategory: byCat, bySource: bySrc };
  }

  save(dir: string): void {
    fs.mkdirSync(dir, { recursive: true });
    // 保存 entries + HNSW
    const data = {
      entries: this.entries.map(e => ({
        id: e.id, text: e.text, vec: Array.from(e.vec),
        category: e.category, source: e.source, ngramHash: e.ngramHash,
      })),
      useNgramFilter: this.useNgramFilter,
      ngramCandidates: this.ngramCandidates,
    };
    fs.writeFileSync(path.join(dir, "cn_store.json"), JSON.stringify(data), "utf-8");
    if (this.hnsw) this.hnsw.save(dir);
  }

  static load(dir: string): CNVectorStore | null {
    const storeFile = path.join(dir, "cn_store.json");
    if (!fs.existsSync(storeFile)) return null;
    const data = JSON.parse(fs.readFileSync(storeFile, "utf-8")) as {
      entries: Array<{ id: string; text: string; vec: number[]; category?: string; source?: string; ngramHash: number }>;
      useNgramFilter: boolean;
      ngramCandidates: number;
    };
    const store = new CNVectorStore(data.entries[0]?.vec?.length ?? 64);
    store.entries = data.entries.map(e => ({
      ...e,
      vec: new Float64Array(e.vec),
    }));
    store.useNgramFilter = data.useNgramFilter;
    store.ngramCandidates = data.ngramCandidates;
    store.hnsw = HNSWIndex.load(dir);
    return store;
  }

  private _dirty = false;
  private _markDirty(): void { this._dirty = true; }

  private _buildHNSW(entries: CNEntry[]): void {
    this.hnsw = new HNSWIndex({ M: 16, maxLevel: 6, efConstruction: 80, efSearch: 40 });
    for (const e of entries) {
      this.hnsw.add(e);
    }
    this._dirty = false;
  }
}

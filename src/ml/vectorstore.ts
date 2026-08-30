import * as fs from "fs";
import * as path from "path";
import { HNSWIndex } from "./hnsw";

export interface StoreEntry {
  id: string;
  text: string;
  vec: number[];
}

/**
 * 向量库: HNSW 索引 + 余弦检索 + JSON 持久化
 * 支持大规模(数万条)近似近邻搜索, O(log N) 查询 vs O(N) 暴力搜索
 */
export class VectorStore {
  private entries: StoreEntry[] = [];
  private hnsw: HNSWIndex | null = null;
  private dim = 0;
  private _dirty = false;

  get size(): number {
    return this.entries.length;
  }

  add(text: string, vec: number[]): string {
    const id = `v${this.entries.length}`;
    this.entries.push({ id, text, vec });
    if (this.entries.length === 1) this.dim = vec.length;
    this._dirty = true;
    return id;
  }

  /** 批量添加,构建后自动建 HNSW */
  addBatch(pairs: Array<{ text: string; vec: number[] }>): void {
    for (const p of pairs) this.add(p.text, p.vec);
  }

  cosine(a: number[], b: number[]): number {
    let dot = 0, na = 0, nb = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    const denom = Math.sqrt(Math.max(na, 1e-12)) * Math.sqrt(Math.max(nb, 1e-12));
    return denom === 0 ? 0 : dot / denom;
  }

  /** HNSW 近似近邻搜索 (O(log N)) */
  search(vec: number[], k = 4): Array<{ id: string; text: string; score: number }> {
    if (this.entries.length === 0 || vec.length !== this.dim) return [];
    if (this._dirty || !this.hnsw) this._buildHNSW();
    if (!this.hnsw) return [];
    const results = this.hnsw.search(new Float64Array(vec), k);
    return results;
  }

  /** 暴力搜索 (小规模/验证用) */
  searchBrute(vec: number[], k = 4): Array<{ id: string; text: string; score: number }> {
    if (this.entries.length === 0 || vec.length !== this.dim) return [];
    const scored = this.entries.map(e => ({
      id: e.id, text: e.text, score: this.cosine(e.vec, vec),
    })).sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  save(dir: string): void {
    fs.mkdirSync(dir, { recursive: true });
    const data = { dim: this.dim, entries: this.entries };
    fs.writeFileSync(path.join(dir, "store.json"), JSON.stringify(data), "utf-8");
    if (this.hnsw) this.hnsw.save(dir);
  }

  static load(dir: string): VectorStore | null {
    const file = path.join(dir, "store.json");
    if (!fs.existsSync(file)) return null;
    const data = JSON.parse(fs.readFileSync(file, "utf-8")) as { dim: number; entries: StoreEntry[] };
    const store = new VectorStore();
    store.entries = data.entries;
    store.dim = data.dim;
    store.hnsw = HNSWIndex.load(dir);
    store._dirty = false;
    return store;
  }

  private _buildHNSW(): void {
    this.hnsw = new HNSWIndex({ M: 16, maxLevel: 6, efConstruction: 80, efSearch: 40 });
    for (const e of this.entries) {
      this.hnsw.add({ id: e.id, text: e.text, vec: new Float64Array(e.vec) });
    }
    this._dirty = false;
  }
}

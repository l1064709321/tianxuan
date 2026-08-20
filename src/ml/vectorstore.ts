import * as fs from "fs";
import * as path from "path";

export interface StoreEntry {
  id: string;
  text: string;
  vec: number[];
}

/**
 * 纯 TS 向量库: 余弦检索 + JSON 持久化
 * 后续可替换为 HNSW/IVF,先顾内存规模(几千条级别)
 */
export class VectorStore {
  private entries: StoreEntry[] = [];

  get size(): number {
    return this.entries.length;
  }

  add(text: string, vec: number[]): string {
    const id = `v${this.entries.length}`;
    this.entries.push({ id, text, vec });
    return id;
  }

  cosine(a: number[], b: number[]): number {
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    const denom = Math.sqrt(Math.max(na, 1e-12)) * Math.sqrt(Math.max(nb, 1e-12));
    return denom === 0 ? 0 : dot / denom;
  }

  search(vec: number[], k = 4): Array<{ id: string; text: string; score: number }> {
    return this.entries
      .map((e) => ({ id: e.id, text: e.text, score: this.cosine(e.vec, vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  save(dir: string): void {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, "store.json"), JSON.stringify(this.entries), "utf-8");
  }

  static load(dir: string): VectorStore | null {
    const file = path.join(dir, "store.json");
    if (!fs.existsSync(file)) return null;
    const store = new VectorStore();
    store.entries = JSON.parse(fs.readFileSync(file, "utf-8")) as StoreEntry[];
    return store;
  }
}

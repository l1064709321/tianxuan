/**
 * HNSW (Hierarchical Navigable Small World) — 近似最近邻搜索索引
 *
 * 纯 TypeScript 实现,零原生依赖,CPU 可训可搜。
 * 核心思想:多层图结构,高层稀疏导航快速接近目标区域,低层密集检索精确结果。
 *
 * 与暴力搜索对比:
 *   暴力: O(N·D) 每次查询,N=条目数,D=向量维度
 *   HNSW: O(log N · D) 平均查询,构建 O(N·log N · D)
 */
import * as fs from "fs";
import * as path from "path";

export interface HNSWEntry {
  id: string;
  text: string;
  vec: Float64Array;
}

export interface HNSWSearchResult {
  id: string;
  text: string;
  score: number;
}

interface GraphNode {
  entry: HNSWEntry;
  neighbors: Map<number, number[]>; // layer → [nodeId, ...]
}

/** 余弦相似度 */
function cosineSim(a: Float64Array, b: Float64Array): number {
  let dot = 0, na = 0, nb = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

/** KNN: 从候选堆中返回 top-k */
function knnFromCandidates(cands: Array<{ id: number; score: number }>, k: number): number[] {
  cands.sort((a, b) => b.score - a.score);
  return cands.slice(0, k).map(c => c.id);
}

export class HNSWIndex {
  private nodes: GraphNode[] = [];
  private entryById: Map<string, HNSWEntry> = new Map();
  private entryByIdx: Map<number, HNSWEntry> = new Map();
  /** 入口点 (每层一个) */
  private entryPoints: number[] = [];
  /** 图最大层数 */
  private readonly maxLevel: number;
  /** 每节点最大连接数 */
  private readonly M: number;
  /** 构建时采样候选数 */
  private readonly efConstruction: number;
  /** 查询时搜索宽度 */
  private readonly efSearch: number;
  private totalEntries = 0;
  private dim = 0;

  constructor(opts: { M?: number; maxLevel?: number; efConstruction?: number; efSearch?: number } = {}) {
    this.M = opts.M ?? 16;
    this.maxLevel = opts.maxLevel ?? 8;
    this.efConstruction = opts.efConstruction ?? 100;
    this.efSearch = opts.efSearch ?? 50;
  }

  get size(): number { return this.totalEntries; }
  get isEmpty(): boolean { return this.totalEntries === 0; }

  /** 加入一条记录,返回内部 nodeId */
  add(entry: HNSWEntry): number {
    const idx = this.totalEntries++;
    if (this.totalEntries === 1) {
      this.dim = entry.vec.length;
    }
    const node: GraphNode = { entry, neighbors: new Map() };
    for (let l = 0; l < this.maxLevel; l++) node.neighbors.set(l, []);
    this.nodes.push(node);
    this.entryById.set(entry.id, entry);
    this.entryByIdx.set(idx, entry);

    if (this.totalEntries === 1) {
      this.entryPoints = [0];
      return 0;
    }

    // 随机分配层 (几何分布)
    const level = Math.floor(-Math.log(Math.random()) * this.maxLevel / Math.log(2));
    const clampedLevel = Math.min(level, this.maxLevel - 1);

    // 从高层入口逐层下降搜索 (取候选列表中最优的作为入口)
    let searchEp = this.entryPoints[this.entryPoints.length - 1] ?? 0;
    for (let l = this.entryPoints.length - 1; l > clampedLevel; l--) {
      const layerResult = this._searchLayer(searchEp, idx, l, 1);
      searchEp = layerResult[0];
    }

    // 在目标层及以下是建边
    for (let l = Math.min(clampedLevel, this.entryPoints.length - 1); l >= 0; l--) {
      const neighbors = this._searchLayer(searchEp, idx, l, this.efConstruction);
      // 为新建节点选 M 个最近邻
      const best = neighbors.slice(0, this.M);
      node.neighbors.set(l, best);
      // 反向更新: 每个已有邻居也限制连接数
      for (const nbId of best) {
        const nbNode = this.nodes[nbId];
        const nbNeighbors = nbNode.neighbors.get(l) ?? [];
        if (nbNeighbors.length < this.M) {
          nbNeighbors.push(idx);
          nbNode.neighbors.set(l, nbNeighbors);
        } else {
          // 替换最远的
          const allCombined = [...nbNeighbors, idx];
          allCombined.sort((a, ca) => {
            const sa = cosineSim(this.nodes[a].entry.vec, this.nodes[idx].entry.vec);
            const sc = cosineSim(this.nodes[ca].entry.vec, this.nodes[idx].entry.vec);
            return sc - sa;
          });
          nbNode.neighbors.set(l, allCombined.slice(0, this.M));
        }
      }
    }

    // 更新入口点
    if (clampedLevel >= this.entryPoints.length) {
      while (this.entryPoints.length <= clampedLevel) this.entryPoints.push(0);
      this.entryPoints[clampedLevel] = idx;
    }

    return idx;
  }

  /** 搜索最近邻 */
  search(vec: Float64Array, k = 4): HNSWSearchResult[] {
    if (this.nodes.length === 0 || vec.length !== this.dim) return [];
    let ep = this.entryPoints[this.entryPoints.length - 1] ?? 0;
    for (let l = this.entryPoints.length - 1; l >= 0; l--) {
      const layerResult = this._searchLayer(ep, -1, l, 1);
      ep = layerResult[0];
    }
    // 在底层做 efSearch 宽度的搜索
    const cands: Array<{ id: number; score: number }> = [];
    const visited = new Set<number>();
    const queue: number[] = [ep];
    visited.add(ep);
    const initialScores = new Map<number, number>();
    initialScores.set(ep, cosineSim(this.nodes[ep].entry.vec, vec));

    let head = 0;
    while (head < queue.length && cands.length < this.efSearch) {
      const curr = queue[head++];
      const currScore = initialScores.get(curr) ?? 0;
      cands.push({ id: curr, score: currScore });
      const nbList = this.nodes[curr].neighbors.get(0) ?? [];
      for (const nbId of nbList) {
        if (!visited.has(nbId)) {
          visited.add(nbId);
          const score = cosineSim(this.nodes[nbId].entry.vec, vec);
          initialScores.set(nbId, score);
          queue.push(nbId);
        }
      }
    }
    cands.sort((a, b) => b.score - a.score);
    return cands.slice(0, k).map(c => ({
      id: this.nodes[c.id].entry.id,
      text: this.nodes[c.id].entry.text,
      score: c.score,
    }));
  }

  private _searchLayer(inputId: number, targetId: number, level: number, ef: number): number[] {
    const node = this.nodes[inputId];
    const nbList = node.neighbors.get(level) ?? [];
    const entryVec = targetId >= 0 ? this.nodes[targetId].entry.vec : null;
    if (entryVec === null) return [inputId];

    // 贪婪搜索 + 候选池 (ef 宽度)
    const candidates: Array<{ id: number; score: number }> = [{ id: inputId, score: cosineSim(this.nodes[inputId].entry.vec, entryVec) }];
    const visited = new Set<number>([inputId]);
    let head = 0;
    while (head < candidates.length && candidates.length < ef) {
      const curr = candidates[head++];
      const currNode = this.nodes[curr.id];
      const currNb = currNode.neighbors.get(level) ?? [];
      for (const nbId of currNb) {
        if (!visited.has(nbId)) {
          visited.add(nbId);
          const sc = cosineSim(this.nodes[nbId].entry.vec, entryVec);
          candidates.push({ id: nbId, score: sc });
        }
      }
    }
    candidates.sort((a, b) => b.score - a.score);
    return candidates.slice(0, ef).map(c => c.id);
  }

  save(dir: string): void {
    fs.mkdirSync(dir, { recursive: true });
    const data = {
      dim: this.dim,
      totalEntries: this.totalEntries,
      entryPoints: this.entryPoints,
      nodes: this.nodes.map((n, i) => ({
        id: n.entry.id,
        text: n.entry.text,
        vec: Array.from(n.entry.vec),
        neighbors: Object.fromEntries(
          [...n.neighbors.entries()].map(([l, ids]) => [String(l), ids])
        ),
      })),
    };
    fs.writeFileSync(path.join(dir, "hnsw.json"), JSON.stringify(data), "utf-8");
  }

  static load(dir: string): HNSWIndex | null {
    const file = path.join(dir, "hnsw.json");
    if (!fs.existsSync(file)) return null;
    const data = JSON.parse(fs.readFileSync(file, "utf-8")) as {
      dim: number; totalEntries: number; entryPoints: number[];
      nodes: Array<{ id: string; text: string; vec: number[]; neighbors: Record<string, number[]> }>;
    };
    const idx = new HNSWIndex();
    idx.dim = data.dim;
    idx.totalEntries = data.totalEntries;
    idx.entryPoints = data.entryPoints;

    for (const nd of data.nodes) {
      const entry: HNSWEntry = { id: nd.id, text: nd.text, vec: new Float64Array(nd.vec) };
      idx.nodes.push({ entry, neighbors: new Map<number, number[]>() });
      const nodeId = idx.nodes.length - 1;
      idx.entryById.set(nd.id, entry);
      idx.entryByIdx.set(nodeId, entry);
    }
    // 重建图
    for (let i = 0; i < idx.nodes.length; i++) {
      const nd = data.nodes[i];
      const neighbors = new Map<number, number[]>();
      for (const [lStr, ids] of Object.entries(nd.neighbors)) {
        neighbors.set(parseInt(lStr), ids);
      }
      idx.nodes[i].neighbors = neighbors;
    }
    return idx;
  }
}

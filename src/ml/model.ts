import { makeGaussian, mulberry32 } from "./rng";

export interface CharMLPConfig {
  vocabSize: number;
  ctx: number;
  emb: number;
  h1: number;
  h2: number;
}

export const MAX_PARAMS = 10_000_000;

export interface ForwardResult {
  logits: Float64Array;
  xv: Float64Array;
  pre1: Float64Array;
  h1: Float64Array;
  pre2: Float64Array | null;
  h2: Float64Array | null;
}

function tanh(x: number): number {
  if (x > 20) return 1;
  if (x < -20) return -1;
  const e = Math.exp(2 * x);
  return (e - 1) / (e + 1);
}

/** 参数组: 参数 + 梯度 + Adam 动量 */
export class Group {
  readonly p: Float64Array;
  readonly g: Float64Array;
  private readonly m: Float64Array;
  private readonly v: Float64Array;

  constructor(size: number, init: (i: number) => number) {
    this.p = new Float64Array(size);
    this.g = new Float64Array(size);
    this.m = new Float64Array(size);
    this.v = new Float64Array(size);
    for (let i = 0; i < size; i++) this.p[i] = init(i);
  }

  zeroGrad(): void {
    this.g.fill(0);
  }

  /** Adam 更新: lr 由调用方给定 */
  adam(lr: number, t: number): void {
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    for (let i = 0; i < this.p.length; i++) {
      const grad = this.g[i];
      this.m[i] = b1 * this.m[i] + (1 - b1) * grad;
      this.v[i] = b2 * this.v[i] + (1 - b2) * grad * grad;
      const mh = this.m[i] / (1 - Math.pow(b1, t));
      const vh = this.v[i] / (1 - Math.pow(b2, t));
      this.p[i] -= lr * mh / (Math.sqrt(vh) + eps);
    }
  }
}

/**
 * 天玄核心模型 v0: 字符级 MLP 语言模型(带浅/深两个路径)
 * depth=1: emb → h1 → head(浅,快)
 * depth=2: 再叠加 h2 → head2(深,更准)
 * 全部纯 TypeScript 数值内核(零原生依赖),CPU 可训可推
 */
export class CharMLP {
  readonly cfg: CharMLPConfig;
  private xdim: number;
  private groups: Group[];
  private emb: Group;
  private w1: Group;
  private b1: Group;
  private w2: Group;
  private b2: Group;
  private out1: Group;
  private out2: Group;
  private t = 0;

  constructor(cfg: CharMLPConfig, seed = 42) {
    this.cfg = cfg;
    this.xdim = cfg.ctx * cfg.emb;
    const gaussian = makeGaussian(mulberry32(seed));
    const scale = (fanIn: number) => 1 / Math.sqrt(fanIn);
    this.emb = new Group(cfg.vocabSize * cfg.emb, () => gaussian() * 0.1);
    this.w1 = new Group(cfg.h1 * this.xdim, () => gaussian() * scale(this.xdim));
    this.b1 = new Group(cfg.h1, () => 0);
    this.w2 = new Group(cfg.h2 * cfg.h1, () => gaussian() * scale(cfg.h1));
    this.b2 = new Group(cfg.h2, () => 0);
    this.out1 = new Group(cfg.vocabSize * cfg.h1, () => gaussian() * 0.1);
    this.out2 = new Group(cfg.vocabSize * cfg.h2, () => gaussian() * 0.1);
    this.groups = [this.emb, this.w1, this.b1, this.w2, this.b2, this.out1, this.out2];
  }

  paramCount(): number {
    return this.groups.reduce((s, g) => s + g.p.length, 0);
  }

  /** 前向: 输入 ctx 个字符 id,返回 logits */
  forward(ids: Int32Array | number[], depth: 1 | 2): ForwardResult {
    const { vocabSize: v, ctx, emb, h1, h2 } = this.cfg;
    const xdim = this.xdim;
    const xv = new Float64Array(xdim);
    for (let pos = 0; pos < ctx; pos++) {
      const id = ids[pos] % v;
      const src = this.emb.p.subarray(id * emb, (id + 1) * emb);
      xv.set(src, pos * emb);
    }
    const pre1 = new Float64Array(h1);
    const h1a = new Float64Array(h1);
    for (let i = 0; i < h1; i++) {
      let acc = this.b1.p[i];
      const row = i * xdim;
      for (let j = 0; j < xdim; j++) acc += this.w1.p[row + j] * xv[j];
      pre1[i] = acc;
      h1a[i] = tanh(acc);
    }
    const logits = new Float64Array(v);
    for (let k = 0; k < v; k++) {
      let acc = 0;
      const row = k * h1;
      for (let i = 0; i < h1; i++) acc += this.out1.p[row + i] * h1a[i];
      logits[k] = acc;
    }
    let pre2: Float64Array | null = null;
    let h2a: Float64Array | null = null;
    if (depth === 2) {
      pre2 = new Float64Array(h2);
      h2a = new Float64Array(h2);
      for (let j = 0; j < h2; j++) {
        let acc = this.b2.p[j];
        const row = j * h1;
        for (let i = 0; i < h1; i++) acc += this.w2.p[row + i] * h1a[i];
        pre2[j] = acc;
        h2a[j] = tanh(acc);
      }
      for (let k = 0; k < v; k++) {
        let acc = 0;
        const row = k * h2;
        for (let j = 0; j < h2; j++) acc += this.out2.p[row + j] * h2a[j];
        logits[k] += acc;
      }
    }
    return { logits, xv, pre1, h1: h1a, pre2, h2: h2a };
  }

  /** 一个 batch 的训练步骤(纯随机梯度,带样本平均) */
  trainStep(samples: Array<{ x: number[]; y: number }>, lr: number): number {
    const { vocabSize: v, ctx, emb, h1, h2 } = this.cfg;
    this.t += 1;
    for (const g of this.groups) g.zeroGrad();
    let totalLoss = 0;
    for (const s of samples) {
      const fw = this.forward(s.x, 2);
      const logits = fw.logits;
      let max = -Infinity;
      for (let k = 0; k < v; k++) if (logits[k] > max) max = logits[k];
      let sum = 0;
      const pr = new Float64Array(v);
      for (let k = 0; k < v; k++) {
        pr[k] = Math.exp(logits[k] - max);
        sum += pr[k];
      }
      for (let k = 0; k < v; k++) pr[k] /= sum;
      totalLoss -= Math.log(pr[s.y] + 1e-12);

      const d = new Float64Array(v);
      for (let k = 0; k < v; k++) d[k] = pr[k];
      d[s.y] -= 1;

      const dh2Raw = new Float64Array(h2);
      for (let k = 0; k < v; k++) {
        const row = k * h1;
        for (let i = 0; i < h1; i++) this.out1.g[row + i] += d[k] * fw.h1[i];
      }
      const dh1 = new Float64Array(h1);
      for (let k = 0; k < v; k++) {
        const row = k * h1;
        for (let i = 0; i < h1; i++) dh1[i] += d[k] * this.out1.p[row + i];
      }
      if (fw.h2 && fw.pre2) {
        for (let k = 0; k < v; k++) {
          const row = k * h2;
          for (let j = 0; j < h2; j++) {
            this.out2.g[row + j] += d[k] * fw.h2[j];
            dh2Raw[j] += d[k] * this.out2.p[row + j];
          }
        }
        for (let j = 0; j < h2; j++) {
          const dpre2 = dh2Raw[j] * (1 - fw.h2[j] * fw.h2[j]);
          const row = j * h1;
          for (let i = 0; i < h1; i++) {
            this.w2.g[row + i] += dpre2 * fw.h1[i];
            dh1[i] += dpre2 * this.w2.p[row + i];
          }
          this.b2.g[j] += dpre2;
        }
      }
      const dpre1 = new Float64Array(h1);
      for (let i = 0; i < h1; i++) dpre1[i] = dh1[i] * (1 - fw.h1[i] * fw.h1[i]);
      for (let i = 0; i < h1; i++) {
        const row = i * this.xdim;
        for (let j = 0; j < this.xdim; j++) this.w1.g[row + j] += dpre1[i] * fw.xv[j];
        this.b1.g[i] += dpre1[i];
      }
      const xdim = this.xdim;
      for (let pos = 0; pos < ctx; pos++) {
        const id = s.x[pos] % v;
        for (let e = 0; e < emb; e++) {
          let acc = 0;
          for (let i = 0; i < h1; i++) acc += this.w1.p[i * xdim + pos * emb + e] * dpre1[i];
          this.emb.g[id * emb + e] += acc;
        }
      }
    }
    const n = samples.length;
    for (const g of this.groups) {
      for (let i = 0; i < g.g.length; i++) g.g[i] /= n;
      g.adam(lr, this.t);
    }
    return totalLoss / n;
  }

  save(): number[] {
    const out: number[] = [];
    for (const g of this.groups) for (let i = 0; i < g.p.length; i++) out.push(g.p[i]);
    return out;
  }

  load(params: number[]): void {
    let off = 0;
    for (const g of this.groups) {
      for (let i = 0; i < g.p.length; i++) g.p[i] = params[off + i];
      off += g.p.length;
    }
  }

  logitsToProbs(logits: Float64Array): Float64Array {
    let max = -Infinity;
    for (let k = 0; k < logits.length; k++) if (logits[k] > max) max = logits[k];
    let sum = 0;
    const pr = new Float64Array(logits.length);
    for (let k = 0; k < logits.length; k++) {
      pr[k] = Math.exp(logits[k] - max);
      sum += pr[k];
    }
    for (let k = 0; k < pr.length; k++) pr[k] /= sum;
    return pr;
  }

  /** 熵归一置信度: 1 - H/log(V), 0..1 */
  confidence(logits: Float64Array): number {
    const pr = this.logitsToProbs(logits);
    let h = 0;
    for (let k = 0; k < pr.length; k++) if (pr[k] > 0) h -= pr[k] * Math.log(pr[k]);
    const maxH = Math.log(pr.length);
    return Math.max(0, Math.min(1, 1 - h / maxH));
  }

  /** 字符嵌入均值: 供向量库做语义检索 */
  embedAvg(ids: number[]): number[] {
    const { vocabSize: v, emb, ctx } = this.cfg;
    const out = new Float64Array(emb);
    let count = 0;
    for (let pos = 0; pos < ids.length; pos++) {
      const id = ids[pos] % v;
      const row = this.emb.p.subarray(id * emb, (id + 1) * emb);
      for (let e = 0; e < emb; e++) out[e] += row[e];
      count += 1;
    }
    if (count > 0) for (let e = 0; e < emb; e++) out[e] /= count;
    return Array.from(out);
  }
}

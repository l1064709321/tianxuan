import { Group } from "./model";

export interface TitansConfig {
  dim: number;
  /** 深记忆槽位数 */
  slots: number;
}

export interface TitansCache {
  query: Float64Array<ArrayBufferLike>;
  keys: Float64Array<ArrayBufferLike>;
  values: Float64Array<ArrayBufferLike>;
  weights: Float64Array<ArrayBufferLike>;
}

/**
 * Titans 在线神经记忆(逻辑自研实现, 不抄代码):
 * - 持久记忆 P(D): 按「记忆即梯度」动量更新, 每一大步把梯度信息写入持久槽
 * - 深记忆槽 D(keys/values): 按相似度检索的短期神经记忆
 * - 读: out = P + softmax(q·K)·V  (检索加权)
 * 用途: L0 零算力直出通道 + 生成期在线写入(不需要预训练)。
 */
export class TitansMemory {
  readonly cfg: TitansConfig;
  private persistent: Float64Array;
  private keys: Float64Array;
  private values: Float64Array;
  private t = 0;

  constructor(cfg: TitansConfig) {
    this.cfg = cfg;
    this.persistent = new Float64Array(cfg.dim);
    this.keys = new Float64Array(cfg.slots * cfg.dim);
    this.values = new Float64Array(cfg.slots * cfg.dim);
  }

  /** 检索: 返回记忆输出 + 命中槽位计数 */
  read(query: Float64Array<ArrayBufferLike>, cache?: TitansCache): { out: Float64Array; hitSlots: number } {
    const { dim: D, slots: S } = this.cfg;
    const out = new Float64Array(D);
    for (let i = 0; i < D; i++) out[i] = this.persistent[i];
    const q = new Float64Array(D);
    let qn = 0;
    for (let i = 0; i < D; i++) qn += query[i] * query[i];
    qn = Math.sqrt(qn) + 1e-8;
    for (let i = 0; i < D; i++) q[i] = query[i] / qn;
    const keys = new Float64Array(D);
    const values = new Float64Array(D);
    let scores = 0;
    const scale = Math.sqrt(D);
    for (let s = 0; s < S; s++) {
      let acc = 0;
      for (let i = 0; i < D; i++) acc += q[i] * this.keys[s * D + i];
      keys[s] = acc / scale;
      values[s] = 0;
    }
    // softmax 权重
    let max = -Infinity;
    for (let s = 0; s < S; s++) if (keys[s] > max) max = keys[s];
    let sum = 0;
    for (let s = 0; s < S; s++) {
      keys[s] = Math.exp(keys[s] - max);
      sum += keys[s];
    }
    let hitSlots = 0;
    for (let s = 0; s < S; s++) {
      const w = keys[s] / sum;
      if (w > 0.05) hitSlots += 1;
      scores += w;
      for (let i = 0; i < D; i++) out[i] += w * this.values[s * D + i];
    }
    if (cache) {
      cache.query = q;
      cache.keys = keys;
      cache.values = values;
      cache.weights = keys;
    }
    return { out, hitSlots };
  }

  /** 在线写入: 持久记忆按动量吸收 + 深记忆槽 FIFO 覆盖, 返回写入槽位 */
  write(x: Float64Array<ArrayBufferLike>): number {
    const { dim: D, slots: S } = this.cfg;
    this.t += 1;
    const alpha = 0.05;
    for (let i = 0; i < D; i++) {
      this.persistent[i] = (1 - alpha) * this.persistent[i] + alpha * x[i];
    }
    const slot = this.t % S;
    for (let i = 0; i < D; i++) {
      this.keys[slot * D + i] = x[i];
      this.values[slot * D + i] = x[i];
    }
    return slot;
  }

  /** 复现确定性(测试用) */
  state(): { persistent: Float64Array; keys: Float64Array; values: Float64Array } {
    return { persistent: this.persistent.slice(), keys: this.keys.slice(), values: this.values.slice() };
  }

  /** 记忆活跃度探针: 最近写入的非零槽占比(0..1) */
  occupancy(): number {
    const { dim: D, slots: S } = this.cfg;
    let nz = 0;
    for (let s = 0; s < S; s++) {
      let acc = 0;
      for (let i = 0; i < D; i++) acc += Math.abs(this.values[s * D + i]);
      if (acc > 1e-9) nz += 1;
    }
    return nz / S;
  }
}

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
 * - 深记忆槽 D(keys/values/chars): 按相似度检索的短期神经记忆
 * - 读: out = P + softmax(q·K)·V  (检索加权), chars 存储对应字符
 * - 写: 每个 timestep 记录 h1 → 持久记忆吸收 + 槽位 FIFO
 *
 * 真正的在线学习: 训练期和推理期都在写入 Titan, 跨批次记忆自适应
 */
export class TitansMemory {
  readonly cfg: TitansConfig;
  private persistent: Float64Array;
  private keys: Float64Array;
  private values: Float64Array;
  private chars: string[];  // 每个槽位对应的输出字符
  private t = 0;

  constructor(cfg: TitansConfig) {
    this.cfg = cfg;
    this.persistent = new Float64Array(cfg.dim);
    this.keys = new Float64Array(cfg.slots * cfg.dim);
    this.values = new Float64Array(cfg.slots * cfg.dim);
    this.chars = new Array(cfg.slots).fill("");
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
    const scoreBuf = new Float64Array(S);
    const scale = Math.sqrt(D);
    for (let s = 0; s < S; s++) {
      let acc = 0;
      for (let i = 0; i < D; i++) acc += q[i] * this.keys[s * D + i];
      scoreBuf[s] = acc / scale;
    }
    let max = -Infinity;
    for (let s = 0; s < S; s++) if (scoreBuf[s] > max) max = scoreBuf[s];
    let sum = 0;
    const w = new Float64Array(S);
    for (let s = 0; s < S; s++) {
      w[s] = Math.exp(scoreBuf[s] - max);
      sum += w[s];
    }
    let hitSlots = 0;
    for (let s = 0; s < S; s++) {
      const pw = w[s] / sum;
      if (pw > 0.05) hitSlots += 1;
      for (let i = 0; i < D; i++) out[i] += pw * this.values[s * D + i];
    }
    if (cache) {
      cache.query = q;
      cache.keys = scoreBuf;
      cache.values = new Float64Array(D);
      cache.weights = w;
    }
    return { out, hitSlots };
  }

  /**
   * 在线写入: 持久记忆按动量吸收 + 深记忆槽 FIFO 覆盖
   * 训练阶段也调用此方法 → 真正的跨批在线学习
   * @param x 输入向量
   * @param char 对应的输出字符 (用于 L0 直出)
   * @returns 写入的槽位 ID
   */
  write(x: Float64Array<ArrayBufferLike>, char: string = ""): number {
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
    this.chars[slot] = char;  // 记录对应字符
    return slot;
  }

  /** 获取指定槽位的字符 */
  getChar(slot: number): string {
    return this.chars[slot] ?? "";
  }

  state(): { persistent: Float64Array; keys: Float64Array; values: Float64Array } {
    return { persistent: this.persistent.slice(), keys: this.keys.slice(), values: this.values.slice() };
  }

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

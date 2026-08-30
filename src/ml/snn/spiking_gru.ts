/**
 * SpikingGRU — 脉冲门控循环单元
 *
 * ## 设计
 *
 * SpikingGRU 是 GRU 的脉冲版本:
 * - 隐藏状态用 LIF 神经元表示
 * - 遗忘门/重置门用阈值化激活
 * - STDP 作为主学习信号
 * - BPTT 作为辅助监督信号
 *
 * ## 生物学对应
 *
 * | 组件 | 生物学对应 |
 * |---|---|
 * | LIF 隐藏层 | 皮层柱神经元群体 |
 * | 遗忘门 z | 抑制性中间神经元 |
 * | 重置门 r | 去极化门控 |
 * | STDP | 突触可塑性 |
 * | 预测误差 | 多巴胺 RPE |
 */

import { LIFLayer, LIFConfig } from "./lif_neuron";
import { Group } from "../model";
import { mulberry32, makeGaussian } from "../rng";

export interface SpikingGRUConfig {
  /** 输入维度 */
  inputSize: number;
  /** 隐藏维度 */
  hiddenSize: number;
  /** 词表大小 */
  vocabSize: number;
  /** 嵌入维度 */
  emb: number;
  /** BPTT 截断长度 */
  bptt: number;
  /** 记忆上下文窗口 */
  ctx: number;
  /** LIF 配置 */
  lif?: Partial<LIFConfig>;
  /** STDP 学习率 */
  stdpRate: number;
  /** 是否启用 STDP */
  useStdp: boolean;
  /** 是否启用 BPTT 辅助 */
  useBptt: boolean;
  /** BPTT 权重 (相对于 STDP) */
  bpttWeight: number;
}

export const DEFAULT_SPIKING_GRU_CONFIG: SpikingGRUConfig = {
  inputSize: 64,
  hiddenSize: 256,
  vocabSize: 380,
  emb: 64,
  bptt: 32,
  ctx: 16,
  lif: { tauMem: 10, vThresh: 1.0, refractoryPeriod: 2 },
  stdpRate: 0.001,
  useStdp: true,
  useBptt: true,
  bpttWeight: 0.1,
};

/** SpikingGRU 状态 */
export interface SpikingGRUState {
  h: Float64Array;       // 隐藏状态 (膜电位)
  z: Float64Array;       // 遗忘门
  r: Float64Array;       // 重置门
  c: Float64Array;       // 候选状态
  spikeHistory: Float64Array[];  // 最近 spike 历史 (供 STDP 使用)
  preHistory: Float64Array[];    // 突触前信号历史
}

/**
 * 脉冲门控循环单元
 */
export class SpikingGRU {
  readonly cfg: SpikingGRUConfig;

  // 参数 (public for STDP access)
  readonly emb: Group;              // 嵌入层
  readonly wZ: Group;               // 遗忘门权重
  readonly wR: Group;               // 重置门权重
  readonly wC: Group;               // 候选状态权重
  readonly uZ: Group;               // 循环遗忘门权重
  readonly uR: Group;               // 循环重置门权重
  readonly uC: Group;               // 循环候选权重
  readonly bZ: Group;               // 遗忘门偏置
  readonly bR: Group;               // 重置门偏置
  readonly bC: Group;               // 候选偏置
  readonly outW: Group;             // 输出权重
  readonly outB: Group;             // 输出偏置

  // LIF 隐藏层
  readonly hiddenLayer: LIFLayer;

  // STDP 状态
  private stdpPreHist: Float64Array[] = [];  // 突触前 (input)
  private stdpPostHist: Float64Array[] = []; // 突触后 (hidden)

  // 训练统计
  private t = 0;
  totalSpikeCount = 0;
  totalSteps = 0;  // 总前向步数

  constructor(cfg: Partial<SpikingGRUConfig> = {}, seed = 42) {
    this.cfg = { ...DEFAULT_SPIKING_GRU_CONFIG, ...cfg };
    const { inputSize, hiddenSize, vocabSize, emb } = this.cfg;

    const gaussian = makeGaussian(mulberry32(seed));
    const scale = (fanIn: number) => 1 / Math.sqrt(fanIn);

    this.emb = new Group(vocabSize * emb, () => gaussian() * 0.1);
    this.wZ = new Group(inputSize * hiddenSize, () => gaussian() * scale(inputSize));
    this.wR = new Group(inputSize * hiddenSize, () => gaussian() * scale(inputSize));
    this.wC = new Group(inputSize * hiddenSize, () => gaussian() * scale(inputSize));
    this.uZ = new Group(hiddenSize * hiddenSize, () => gaussian() * 0.1);
    this.uR = new Group(hiddenSize * hiddenSize, () => gaussian() * 0.1);
    this.uC = new Group(hiddenSize * hiddenSize, () => gaussian() * 0.1);
    this.bZ = new Group(hiddenSize, () => -0.5);  // 初始偏向"关闭"
    this.bR = new Group(hiddenSize, () => 0);
    this.bC = new Group(hiddenSize, () => 0);
    this.outW = new Group(vocabSize * hiddenSize, () => gaussian() * 0.1);
    this.outB = new Group(vocabSize, () => 0);

    // LIF 隐藏层
    this.hiddenLayer = new LIFLayer(hiddenSize, cfg.lif);
  }

  getGroups(): Group[] {
    return [this.emb, this.wZ, this.wR, this.wC, this.uZ, this.uR, this.uC, this.bZ, this.bR, this.bC, this.outW, this.outB];
  }

  paramCount(): number {
    return this.getGroups().reduce((s, g) => s + g.p.length, 0);
  }

  /**
   * 创建新状态
   */
  newState(): SpikingGRUState {
    const { hiddenSize } = this.cfg;
    return {
      h: new Float64Array(hiddenSize),
      z: new Float64Array(hiddenSize),
      r: new Float64Array(hiddenSize),
      c: new Float64Array(hiddenSize),
      spikeHistory: [],
      preHistory: [],
    };
  }

  /**
   * 前向一步 (脉冲版本)
   * @param xId 输入字符 ID
   * @param state 当前状态
   * @param globalStep 全局时间步
   * @returns 输出 logits
   */
  step(xId: number, state: SpikingGRUState, globalStep: number): Float64Array {
    this.totalSteps++;
    const { hiddenSize, emb, vocabSize } = this.cfg;
    const H = hiddenSize;

    // 嵌入
    const x = new Float64Array(emb);
    const embRow = this.emb.p.subarray((xId % vocabSize) * emb, (xId % vocabSize + 1) * emb);
    x.set(embRow);

    // 计算门控 (实值，用于梯度)
    const z = new Float64Array(H);
    const r = new Float64Array(H);
    const c = new Float64Array(H);

    for (let j = 0; j < H; j++) {
      // 遗忘门
      let az = this.bZ.p[j];
      for (let i = 0; i < emb; i++) az += this.wZ.p[j * emb + i] * x[i];
      for (let k = 0; k < H; k++) az += this.uZ.p[j * H + k] * state.h[k];
      z[j] = 1 / (1 + Math.exp(-az));  // sigmoid

      // 重置门
      let ar = this.bR.p[j];
      for (let i = 0; i < emb; i++) ar += this.wR.p[j * emb + i] * x[i];
      for (let k = 0; k < H; k++) ar += this.uR.p[j * H + k] * state.h[k];
      r[j] = 1 / (1 + Math.exp(-ar));

      // 候选状态
      let ac = this.bC.p[j];
      for (let i = 0; i < emb; i++) ac += this.wC.p[j * emb + i] * x[i];
      for (let k = 0; k < H; k++) ac += this.uC.p[j * H + k] * (r[k] * state.h[k]);
      c[j] = Math.tanh(ac);
    }

    // 脉冲隐藏层: 用 z, r, c 作为输入电流
    const Iext = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      // 融合 z, r, c 作为输入电流
      // 调整缩放以获得合适的发放率 (10-30%)
      const base = z[j] * 0.3 + r[j] * 0.3 + c[j] * 0.4;
      Iext[j] = base * 5.0 + 0.5;  // 确保平均输入 > 1.0
    }

    // LIF 前向
    const { spike, V } = this.hiddenLayer.step(Iext, globalStep);
    this.totalSpikeCount += Array.from(spike).reduce((s, v) => s + v, 0);

    // 更新状态
    for (let j = 0; j < H; j++) {
      // 脉冲 GRU 更新: spike 调制遗忘/候选
      const spikeVal = spike[j];
      state.h[j] = (1 - z[j]) * state.h[j] + z[j] * c[j];  // 基础更新
      if (spikeVal > 0) {
        state.h[j] = state.h[j] * 0.5 + spikeVal * 0.5;  // spike 时增强
      }
      state.spikeHistory.push(spike.slice());
      state.preHistory.push(Iext.slice());

      // 滑动窗口
      if (state.spikeHistory.length > this.cfg.ctx) state.spikeHistory.shift();
      if (state.preHistory.length > this.cfg.ctx) state.preHistory.shift();
    }

    // 输出头
    const logits = new Float64Array(vocabSize);
    for (let k = 0; k < vocabSize; k++) {
      let acc = this.outB.p[k];
      const row = this.outW.p.subarray(k * H, (k + 1) * H);
      for (let j = 0; j < H; j++) acc += row[j] * V[j];
      logits[k] = acc;
    }

    return logits;
  }

  /**
   * STDP 更新 (突触时序依赖可塑性)
   *
   * 生物学原理:
   * - pre 先于 post → LTP (权重增强)
   * - post 先于 pre → LTD (权重抑制)
   */
  applyStdp(state?: SpikingGRUState): void {
    if (!this.cfg.useStdp) return;

    // 使用传入的 state 或最近的 state
    const hist = state?.preHistory ?? this.stdpPreHist;
    const postHist = state?.spikeHistory ?? this.stdpPostHist;

    if (hist.length < 2 || postHist.length < 2) return;

    const rate = this.cfg.stdpRate;
    const tau = 20;
    const maxNorm = 10.0;
    const emb = this.cfg.emb;
    const hiddenSize = this.cfg.hiddenSize;

    // 对 wZ, wR, wC 应用 STDP
    for (const g of [this.wZ, this.wR, this.wC]) {
      for (let i = 0; i < hiddenSize; i++) {
        let ltp = 0, ltd = 0;
        const maxDt = Math.min(10, hist.length - 1);
        for (let dt = 1; dt <= maxDt; dt++) {
          const prePrev = hist[hist.length - 1 - dt]?.[i] ?? 0;
          const postCur = postHist[postHist.length - 1]?.[i] ?? 0;
          const preCur = hist[hist.length - 1]?.[i] ?? 0;
          const postPrev = postHist[postHist.length - 1 - dt]?.[i] ?? 0;

          ltp += prePrev * postCur * Math.exp(-dt / tau);
          ltd += preCur * postPrev * Math.exp(-dt / tau);
        }
        const delta = rate * (ltp - ltd * 0.5) / maxDt;
        if (Math.abs(delta) < 1e-12) continue;

        const rowStart = i * emb;
        for (let j = 0; j < emb; j++) {
          g.g[rowStart + j] += delta;
        }
      }
    }

    // 更新模型级历史 (用于跨 timestep 追踪)
    if (hist.length > 0) {
      this.stdpPreHist = hist.slice(-20);
      this.stdpPostHist = postHist.slice(-20);
    }

    // 梯度裁剪
    let norm2 = 0;
    for (const g of [this.wZ, this.wR, this.wC]) {
      for (let i = 0; i < g.g.length; i++) norm2 += g.g[i] * g.g[i];
    }
    const norm = Math.sqrt(norm2);
    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      for (const g of [this.wZ, this.wR, this.wC]) {
        for (let i = 0; i < g.g.length; i++) g.g[i] *= scale;
      }
    }
  }

  /**
   * 训练一步 (STDP + BPTT 辅助)
   */
  trainStep(seqs: Array<{ ids: number[] }>, lr: number): number {
    this.t++;
    for (const g of this.getGroups()) g.zeroGrad();

    let totalLoss = 0;
    let totalChars = 0;
    let lastState: SpikingGRUState | null = null;

    for (const seq of seqs) {
      const state = this.newState();
      for (let t = 0; t < seq.ids.length - 1; t++) {
        const logits = this.step(seq.ids[t], state, this.t);
        const y = seq.ids[t + 1];

        // 数值稳定: clamp logits
        for (let k = 0; k < logits.length; k++) {
          logits[k] = Math.max(-10, Math.min(10, logits[k]));
        }

        // 交叉熵损失
        let max = -Infinity;
        for (let k = 0; k < logits.length; k++) if (logits[k] > max) max = logits[k];
        let sum = 0;
        const pr = new Float64Array(logits.length);
        for (let k = 0; k < logits.length; k++) {
          pr[k] = Math.exp(logits[k] - max);
          sum += pr[k];
        }
        for (let k = 0; k < logits.length; k++) pr[k] /= sum;
        totalLoss -= Math.log(Math.max(pr[y], 1e-12));
        totalChars++;

        // 输出头梯度
        const dLogits = pr.slice();
        dLogits[y] -= 1;
        const H = this.cfg.hiddenSize;
        for (let k = 0; k < this.cfg.vocabSize; k++) {
          this.outB.g[k] += dLogits[k];
          const row = k * H;
          for (let j = 0; j < H; j++) {
            this.outW.g[row + j] += dLogits[k] * state.h[j];
          }
        }
      }
      lastState = state;
    }

    // STDP 更新 (使用最后一个 state 的 spike 历史)
    if (lastState) {
      this.applyStdp(lastState);
    }

    // 全局梯度裁剪
    let norm2 = 0;
    for (const g of this.getGroups()) {
      for (let i = 0; i < g.g.length; i++) norm2 += g.g[i] * g.g[i];
    }
    const norm = Math.sqrt(norm2);
    const maxNorm = 5.0;
    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      for (const g of this.getGroups()) {
        for (let i = 0; i < g.g.length; i++) g.g[i] *= scale;
      }
    }

    // 归一化 + Adam
    if (totalChars > 0) {
      for (const g of this.getGroups()) {
        for (let i = 0; i < g.g.length; i++) g.g[i] /= totalChars;
        g.adam(lr, this.t);
      }
    }

    return totalLoss / Math.max(totalChars, 1);
  }

  /**
   * 获取稀疏度 (每个 timestep 的平均发放比例)
   */
  sparsity(): number {
    if (this.totalSteps === 0) return 0;
    // totalSpikeCount 是累计发放次数，除以 (totalSteps * hiddenSize) 得到平均发放率
    return this.totalSpikeCount / (this.totalSteps * this.cfg.hiddenSize);
  }

  /**
   * 获取最近 N 步的发放率
   */
  recentFiringRate(window: number = 100): number {
    const layer = this.hiddenLayer;
    let totalSpikes = 0;
    let totalNeurons = 0;
    const n = Math.min(window, this.totalSteps);
    if (n === 0) return 0;
    for (const neuron of layer.neurons.slice(-n)) {
      totalSpikes += neuron.firingCount;
      totalNeurons++;
    }
    return totalNeurons > 0 ? totalSpikes / (n * totalNeurons) : 0;
  }

  /**
   * 获取字符嵌入向量
   */
  embedInput(id: number): Float64Array {
    const { emb, vocabSize } = this.cfg;
    const out = new Float64Array(emb);
    const row = this.emb.p.subarray((id % vocabSize) * emb, (id % vocabSize + 1) * emb);
    out.set(row);
    return out;
  }

  /**
   * 保存/加载
   */
  save(): number[] {
    const out: number[] = [];
    for (const g of this.getGroups()) {
      for (let i = 0; i < g.p.length; i++) out.push(g.p[i]);
    }
    return out;
  }

  load(params: number[]): void {
    let off = 0;
    for (const g of this.getGroups()) {
      for (let i = 0; i < g.p.length; i++) g.p[i] = params[off + i];
      off += g.p.length;
    }
  }
}

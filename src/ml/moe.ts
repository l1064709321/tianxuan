/**
 * MoE (Mixture of Experts) — 混合专家路由层
 *
 * 架构: 输入嵌入 → 两层门控网络 → softmax 路由权重 → Top-K 稀疏选择
 * 与 CharGRU 串行架构的区别:
 * - 串行: 所有神经按固定顺序执行, 算力固定
 * - MoE: 动态路由, 简单任务只用浅层专家, 复杂任务招募更多深度
 */
import { Group } from "./model";
import { mulberry32, makeGaussian } from "./rng";

export interface MoEConfig {
  /** 专家数量 */
  nExperts: number;
  /** Top-K 稀疏路由: 每步只激活 K 个专家 */
  topK: number;
  /** 门控网络隐层维度 */
  gateHidden: number;
  /** 输入维度 (通常为 emb dim) */
  inputSize: number;
  /** 负载均衡辅助损失权重 */
  loadBalanceWeight: number;
  /** 脉冲近似温度: 越高越尖锐(接近硬选择), 0=禁用 */
  spikeTemperature?: number;
  /** 脉冲近似阈值: softmax prob < threshold 的专家被抑制 */
  spikeThreshold?: number;
}

export interface MoERoute {
  /** 被激活的专家索引 (Top-K) */
  expertIndices: number[];
  /** 各专家权重 (softmax after top-k masking) */
  weights: number[];
  /** 原始门控分数 (softmax前) */
  logits: number[];
  /** 负载均衡辅助损失 */
  loadBalanceLoss: number;
}

/**
 * MoE 核心: 门控网络 + 专家选择 + 负载均衡
 * 门控网络的梯度通过 BPTT 传播, 真正学习「何时用哪个神经」
 */
export class MixtureOfExperts {
  readonly cfg: MoEConfig;
  readonly nExperts: number;
  readonly topK: number;
  readonly loadBalanceWeight: number;

  /** 门控网络: emb → gateHidden → nExperts */
  private gateW1: Group;
  private gateB1: Group;
  private gateW2: Group;
  private gateB2: Group;

  /** 专家路由历史 (用于负载均衡计算) */
  private expertCounts: number[];
  private totalSteps = 0;

  /** 协同探针: 最近一次路由熵 */
  lastRouteEntropy = 0;
  /** 协同探针: 最近一次负载均衡辅助损失 */
  lastLoadBalanceLoss = 0;

  constructor(cfg: MoEConfig, seed = 42) {
    this.cfg = cfg;
    this.nExperts = cfg.nExperts;
    this.topK = Math.min(cfg.topK, cfg.nExperts);
    this.loadBalanceWeight = cfg.loadBalanceWeight;

    const gaussian = makeGaussian(mulberry32(seed));
    const scale1 = 1 / Math.sqrt(cfg.inputSize);
    const scale2 = 1 / Math.sqrt(cfg.gateHidden);

    // gateW1: gateHidden × inputSize (输入→隐层)
    this.gateW1 = new Group(cfg.gateHidden * cfg.inputSize, () => gaussian() * scale1);
    this.gateB1 = new Group(cfg.gateHidden, () => 0);
    // gateW2: nExperts × gateHidden (隐层→输出)
    this.gateW2 = new Group(cfg.nExperts * cfg.gateHidden, () => gaussian() * scale2);
    this.gateB2 = new Group(cfg.nExperts, () => 0);

    this.expertCounts = new Array(cfg.nExperts).fill(0);
  }

  getGroups(): Group[] {
    return [this.gateW1, this.gateB1, this.gateW2, this.gateB2];
  }

  paramCount(): number {
    return this.getGroups().reduce((s, g) => s + g.p.length, 0);
  }

  /** 门控网络前向: input(emb dim) → softmax(nExperts) */
  private gateForward(input: Float64Array): { scores: Float64Array; hidden: Float64Array } {
    const { nExperts, gateHidden, inputSize } = this.cfg;
    const hidden = new Float64Array(gateHidden);
    for (let j = 0; j < gateHidden; j++) {
      let acc = this.gateB1.p[j];
      const row = this.gateW1.p.subarray(j * inputSize, (j + 1) * inputSize);
      for (let i = 0; i < inputSize; i++) acc += row[i] * input[i];
      hidden[j] = Math.max(0, acc);
    }
    const scores = new Float64Array(nExperts);
    for (let k = 0; k < nExperts; k++) {
      let acc = this.gateB2.p[k];
      const row = this.gateW2.p.subarray(k * gateHidden, (k + 1) * gateHidden);
      for (let j = 0; j < gateHidden; j++) acc += row[j] * hidden[j];
      scores[k] = acc;
    }
    return { scores, hidden };
  }

  /** Softmax + Top-K 稀疏路由 + 脉冲近似 */
  route(input: Float64Array): MoERoute {
    const { nExperts, topK } = this;
    this.totalSteps += 1;
    const { scores } = this.gateForward(input);

    // 脉冲近似: 温度缩放使分布更尖锐, 阈值抑制弱专家
    const temp = this.cfg.spikeTemperature ?? 1.0;
    const spikeTh = this.cfg.spikeThreshold ?? 0.0;
    let max = -Infinity;
    for (let i = 0; i < nExperts; i++) if (scores[i] > max) max = scores[i];
    let sum = 0;
    const probs = new Float64Array(nExperts);
    for (let i = 0; i < nExperts; i++) {
      probs[i] = Math.exp((scores[i] - max) / Math.max(temp, 1e-6));
      sum += probs[i];
    }
    for (let i = 0; i < nExperts; i++) probs[i] /= sum;
    // 阈值抑制: prob < threshold → 0, 重新归一化
    if (spikeTh > 0) {
      for (let i = 0; i < nExperts; i++) {
        if (probs[i] < spikeTh) probs[i] = 0;
      }
      let ps = 0;
      for (let i = 0; i < nExperts; i++) ps += probs[i];
      if (ps > 0) for (let i = 0; i < nExperts; i++) probs[i] /= ps;
      else {
        // 全部被抑制 → 强制 top-1
        let best = 0;
        for (let i = 1; i < nExperts; i++) if (scores[i] > scores[best]) best = i;
        probs.fill(0);
        probs[best] = 1;
      }
    }

    const indexed = Array.from({ length: nExperts }, (_, i) => i);
    indexed.sort((a, b) => probs[b] - probs[a]);
    const expertIndices = indexed.slice(0, topK);

    const weights = new Float64Array(nExperts);
    let kSum = 0;
    for (const idx of expertIndices) {
      weights[idx] = probs[idx];
      kSum += probs[idx];
    }
    for (const idx of expertIndices) weights[idx] /= kSum;

    let entropy = 0;
    for (let i = 0; i < nExperts; i++) {
      if (probs[i] > 0) entropy -= probs[i] * Math.log(probs[i]);
    }
    this.lastRouteEntropy = entropy;

    this.expertCounts = this.expertCounts.map((c, i) => c + (expertIndices.includes(i) ? 1 : 0));
    const frac = this.expertCounts.map(c => c / this.totalSteps);
    let lbLoss = 0;
    for (let i = 0; i < nExperts; i++) lbLoss += frac[i] * frac[i];
    lbLoss = nExperts * lbLoss - 1;
    this.lastLoadBalanceLoss = lbLoss;

    return {
      expertIndices,
      weights: Array.from(weights),
      logits: Array.from(scores),
      loadBalanceLoss: lbLoss,
    };
  }

  resetCounts(): void {
    this.expertCounts = new Array(this.nExperts).fill(0);
    this.totalSteps = 0;
  }

  summary(): string {
    const avgCount = this.totalSteps > 0
      ? this.expertCounts.map(c => (c / this.totalSteps * 100).toFixed(1) + "%")
      : Array(this.nExperts).fill("0%");
    return `MoE: topK=${this.topK}, nExperts=${this.nExperts}, ` +
           `entropy=${this.lastRouteEntropy.toFixed(4)}, ` +
           `lbLoss=${this.lastLoadBalanceLoss.toFixed(6)}, ` +
           `spikeTh=${this.cfg.spikeThreshold ?? 0}, temp=${this.cfg.spikeTemperature ?? 1}, ` +
           `distribution=[${avgCount.join(", ")}]`;
  }
}

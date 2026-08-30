/**
 * SNN 训练器 — 真正的脉冲神经网络在线学习
 *
 * ## 架构
 *
 * 主学习信号: STDP (突触时序依赖可塑性)
 * 辅助学习信号: BPTT (截断反向传播，用于监督)
 *
 * ## 在线学习流程
 *
 * 1. 每个 timestep:
 *    - 前向: LIF 神经元更新膜电位 → 发放 spike
 *    - STDP: 根据 pre/post spike timing 更新突触权重
 *    - 记忆写入: Titans 记录 spike 模式
 *
 * 2. 每批训练:
 *    - BPTT 辅助梯度
 *    - 重放采样
 *    - 多巴胺调制
 */

import { SpikingGRU } from "./spiking_gru";
import { STDP } from "../stdp";
import { STDA } from "../stda";
import { DopamineModulator } from "../dopamine";
import { ReplayBuffer } from "../replay_buffer";
import { TitansMemory } from "../titans";

export interface SNNTrainConfig {
  /** STDP LTP 学习率 */
  stdpRateLTP: number;
  /** STDP LTD 学习率 */
  stdpRateLTD: number;
  /** BPTT 辅助权重 */
  bpttWeight: number;
  /** 重放概率 */
  replayProb: number;
  /** 重放缓冲区容量 */
  replayCapacity: number;
  /** 多巴胺 EMA 衰减 */
  dopamineTau: number;
  /** Titans 记忆维度 */
  titansDim: number;
  /** Titans 槽位数 */
  titansSlots: number;
}

export const DEFAULT_SNN_CONFIG: SNNTrainConfig = {
  stdpRateLTP: 0.001,
  stdpRateLTD: 0.0005,
  bpttWeight: 0.1,
  replayProb: 0.1,
  replayCapacity: 4096,
  dopamineTau: 0.95,
  titansDim: 64,
  titansSlots: 256,
};

/**
 * SNN 在线学习器
 */
export class SNNOnlineLearner {
  readonly cfg: SNNTrainConfig;
  readonly model: SpikingGRU;
  readonly stdp: STDP;
  readonly stda: STDA;
  readonly dopamine: DopamineModulator;
  readonly replay: ReplayBuffer;
  readonly titans: TitansMemory;

  private stepCount = 0;
  private prevLoss = Infinity;
  private batchCount = 0;

  constructor(model: SpikingGRU, cfg: Partial<SNNTrainConfig> = {}) {
    this.cfg = { ...DEFAULT_SNN_CONFIG, ...cfg };
    this.model = model;
    this.stdp = new STDP({
      rateLTP: this.cfg.stdpRateLTP,
      rateLTD: this.cfg.stdpRateLTD,
    });
    this.stda = new STDA(this.cfg.stdpRateLTP * 0.5, 50);
    this.dopamine = new DopamineModulator(0.002, this.cfg.dopamineTau, true);
    this.replay = new ReplayBuffer(this.cfg.replayCapacity);
    this.titans = new TitansMemory({ dim: this.cfg.titansDim, slots: this.cfg.titansSlots });
  }

  /**
   * 训练一步 (在线学习)
   */
  trainStepOnline(ids: number[], lr: number): number {
    const state = this.model.newState();
    let totalLoss = 0;

    for (let t = 0; t < ids.length - 1; t++) {
      // 前向
      const logits = this.model.step(ids[t], state, this.stepCount);
      const y = ids[t + 1];

      // 计算损失
      const loss = this.computeLoss(logits, y);
      totalLoss += loss;

      // STDP 更新 (主学习信号)
      this.updateStdp(state);

      // STDA 更新
      if (state.h.length > 0) {
        this.stda.update(state.h);
        this.stda.apply(this.model.getGroups());
      }

      // 记录重放样本 (h2 复用 h1，SNN 单隐藏层无第二层状态)
      if (Math.random() < 0.01) {
        this.replay.push({
          h1: state.h.slice(),
          h2: state.h.slice(),
          xId: ids[t],
          yId: y,
          predLoss: loss,
        });
      }

      // 写入 Titans 记忆 (每 4 步)
      if (t % 4 === 0) {
        const emb = this.model.embedInput(ids[t]);
        this.titans.write(emb);
      }
    }

    this.stepCount += Math.max(ids.length - 1, 1);
    return totalLoss / Math.max(ids.length - 1, 1);
  }

  /**
   * 批量训练
   */
  trainBatch(seqs: Array<{ ids: number[] }>, lr: number): number {
    let totalLoss = 0;
    let totalChars = 0;

    for (const seq of seqs) {
      const loss = this.trainStepOnline(seq.ids, lr);
      totalLoss += loss * (seq.ids.length - 1);
      totalChars += seq.ids.length - 1;
    }

    // 多巴胺调制
    const avgLoss = totalLoss / Math.max(totalChars, 1);
    const dmMult = this.dopamine.update(avgLoss);

    // 重放训练
    const replaySamples = this.replay.sample(8, true);
    if (replaySamples.length > 0) {
      for (const sample of replaySamples) {
        const replayLoss = this.trainStepOnline([sample.xId, sample.yId], lr * 0.5);
        totalLoss += replayLoss;
        totalChars++;
      }
    }

    this.batchCount++;
    return totalLoss / Math.max(totalChars, 1);
  }

  /**
   * 计算交叉熵损失
   */
  private computeLoss(logits: Float64Array, y: number): number {
    // 数值稳定
    let max = -Infinity;
    for (let k = 0; k < logits.length; k++) {
      max = Math.max(max, logits[k]);
    }
    let sum = 0;
    const pr = new Float64Array(logits.length);
    for (let k = 0; k < logits.length; k++) {
      pr[k] = Math.exp(Math.min(logits[k] - max, 50));
      sum += pr[k];
    }
    for (let k = 0; k < logits.length; k++) pr[k] /= sum;
    return -Math.log(Math.max(pr[y], 1e-12));
  }

  /**
   * STDP 更新
   */
  private updateStdp(state: ReturnType<SpikingGRU["newState"]>): void {
    if (state.spikeHistory.length < 2) return;

    const pre = state.preHistory[state.preHistory.length - 1];
    const post = state.spikeHistory[state.spikeHistory.length - 1];

    if (pre && post) {
      this.stdp.record(post, pre);
      this.stdp.apply({ wZ: this.model.wZ, wR: this.model.wR, wC: this.model.wC }, this.model.cfg.emb);
    }
  }

  /**
   * 获取统计信息
   */
  getMetrics(): {
    stepCount: number;
    batchCount: number;
    sparsity: number;
    recentFiringRate: number;
    dopamine: number;
    replayOccupancy: number;
    titansOccupancy: number;
    stdp: { rulesExtracted: number; activeDimensions: number };
  } {
    return {
      stepCount: this.stepCount,
      batchCount: this.batchCount,
      sparsity: this.model.sparsity(),
      recentFiringRate: this.model.recentFiringRate(100),
      dopamine: this.dopamine.dopamineLevel,
      replayOccupancy: this.replay.occupancy(),
      titansOccupancy: this.titans.occupancy(),
      stdp: this.stdp.getMetrics(),
    };
  }
}

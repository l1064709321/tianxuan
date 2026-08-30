/**
 * 天玄在线学习系统 — 真正的类脑学习机制
 *
 * ## 架构设计
 *
 * 本模块实现了一套完整的在线学习系统，模拟人脑的多种学习机制：
 *
 * 1. **STDP (突触时序依赖可塑性)**
 *    - 生物学原理：Bi & Poo (1998)
 *    - pre 先于 post → LTP (权重增强)
 *    - post 先于 pre → LTD (权重抑制)
 *    - 作用对象：GRUCell 的 wZ/wR/wC 突触权重
 *
 * 2. **STDA (突触阈值适应)**
 *    - 高激活神经元 → 阈值上调 (适应抑制)
 *    - 低激活神经元 → 阈值下调 (兴奋增强)
 *    - 作用对象：输出头偏置
 *
 * 3. **重放巩固 (Replay Consolidation)**
 *    - 模拟海马体-皮层重放
 *    - 训练时记录关键样本
 *    - 按比例采样重放，防止灾难性遗忘
 *
 * 4. **多巴胺调制 (Dopamine Modulation)**
 *    - RPE (奖励预测误差) 驱动
 *    - loss 下降 → 多巴胺升高 → 学习率增强
 *    - loss 上升 → 多巴胺降低 → 学习率抑制
 *
 * 5. **Titans 神经记忆**
 *    - 持久记忆：动量吸收
 *    - 深记忆槽：相似度检索
 *    - L0 直出：零算力命中
 *
 * 6. **世界模型**
 *    - 学习状态转移 p(s'|s,a)
 *    - 物理约束正则
 *    - 想象滚动 (未来状态预测)
 */

import { Group } from "./model";
import { STDP } from "./stdp";
import { STDA } from "./stda";
import { DopamineModulator } from "./dopamine";
import { ReplayBuffer } from "./replay_buffer";
import { TitansMemory } from "./titans";
import { WorldModel } from "./world_model";
import { GRUState } from "./gru";

// ============================================================================
// 配置接口
// ============================================================================

export interface OnlineLearningConfig {
  /** STDP 配置 */
  stdp?: Partial<import("./stdp").STDPConfig>;
  /** STDA 配置 */
  stda?: Partial<{ rate: number; tau: number }>;
  /** 重放缓冲区容量 */
  replayCapacity?: number;
  /** 重放概率 */
  replayProb?: number;
  /** 重放 batch 大小 */
  replayBatchSize?: number;
  /** 多巴胺调制器配置 */
  dopamine?: Partial<{ baseLr: number; tau: number; enabled: boolean }>;
  /** Titans 记忆配置 */
  titans?: Partial<{ dim: number; slots: number }>;
  /** 世界模型配置 */
  worldModel?: Partial<{ stateDim: number; actionDim: number; hidden: number; embDim: number }>;
  /** 是否启用预测误差损失 */
  predictionLoss?: boolean;
  /** 预测误差损失权重 */
  predLossWeight?: number;
}

// ============================================================================
// 在线学习器
// ============================================================================

/**
 * 天玄在线学习系统
 *
 * 整合 STDP/STDA/重放/多巴胺/Titans/世界模型，实现真正的类脑学习
 */
export class OnlineLearner {
  readonly cfg: Required<OnlineLearningConfig>;

  /** STDP: 突触时序依赖可塑性 */
  readonly stdp: STDP | null;
  /** STDA: 突触阈值适应 */
  readonly stda: STDA | null;
  /** 重放缓冲区 */
  readonly replay: ReplayBuffer;
  /** 多巴胺调制器 */
  readonly dopamine: DopamineModulator;
  /** Titans 神经记忆 */
  readonly titans: TitansMemory;
  /** 世界模型 */
  readonly worldModel: WorldModel;

  /** 当前学习率乘子 */
  currentLrMultiplier = 1.0;
  /** 当前多巴胺水平 */
  currentDopamine = 0;
  /** 步骤计数 */
  stepCount = 0;

  constructor(config: OnlineLearningConfig = {}) {
    this.cfg = {
      stdp: Object.assign({ rateLTP: 0.001, rateLTD: 0.0005, tauLTP: 20, tauLTD: 30 }, config.stdp ?? {}),
      stda: Object.assign({ rate: 0.0005, tau: 50 }, config.stda ?? {}),
      replayCapacity: config.replayCapacity ?? 4096,
      replayProb: config.replayProb ?? 0.1,
      replayBatchSize: config.replayBatchSize ?? 8,
      dopamine: Object.assign({ baseLr: 0.002, tau: 0.95, enabled: true }, config.dopamine ?? {}),
      titans: Object.assign({ dim: 64, slots: 256 }, config.titans ?? {}),
      worldModel: Object.assign({ stateDim: 32, actionDim: 8, hidden: 64, embDim: 32 }, config.worldModel ?? {}),
      predictionLoss: config.predictionLoss ?? false,
      predLossWeight: config.predLossWeight ?? 0.01,
    };

    // 初始化各子系统
    this.stdp = config.stdp !== undefined ? new STDP(this.cfg.stdp) : null;
    this.stda = config.stda !== undefined ? new STDA(this.cfg.stda.rate, this.cfg.stda.tau) : null;
    this.replay = new ReplayBuffer(this.cfg.replayCapacity);
    this.dopamine = new DopamineModulator(
      this.cfg.dopamine.baseLr,
      this.cfg.dopamine.tau,
      this.cfg.dopamine.enabled,
    );
    this.titans = new TitansMemory({ dim: this.cfg.titans.dim!, slots: this.cfg.titans.slots! });
    this.worldModel = new WorldModel(this.cfg.worldModel);
  }

  // -------------------------------------------------------------------------
  // 公开接口
  // -------------------------------------------------------------------------

  /**
   * 记录一步的 STDP 信号
   * @param post h1 隐藏状态 (突触后)
   * @param pre z1/r1 门激活 (突触前)
   */
  recordStdpSignal(post: Float64Array, pre: Float64Array): void {
    if (this.stdp) {
      this.stdp.record(post, pre);
    }
  }

  /**
   * 应用 STDP 到 GRUCell
   */
  applyStdp(cell: { wZ: Group; wR: Group; wC: Group }, inputSize: number): void {
    if (this.stdp) {
      this.stdp.apply(cell, inputSize);
    }
  }

  /**
   * 更新 STDA 阈值适应
   */
  updateStda(h2: Float64Array): void {
    if (this.stda) {
      this.stda.update(h2);
    }
  }

  /**
   * 应用 STDA 到参数组
   */
  applyStda(groups: Group[]): void {
    if (this.stda) {
      this.stda.apply(groups);
    }
  }

  /**
   * 重放采样训练
   * @returns 重放损失 (如果没有重放则返回 0)
   */
  sampleReplay(): Array<{ h1: Float64Array; h2: Float64Array; xId: number; yId: number }> {
    if (Math.random() < this.cfg.replayProb && !this.replay.isEmpty()) {
      return this.replay.sample(this.cfg.replayBatchSize, true);
    }
    return [];
  }

  /**
   * 推送新样本到重放缓冲区
   */
  pushReplay(h1: Float64Array, h2: Float64Array, xId: number, yId: number, loss: number): void {
    this.replay.push({ h1, h2, xId, yId, predLoss: loss });
  }

  /**
   * 更新多巴胺水平
   * @param currentLoss 当前 batch 平均 loss
   * @returns lr 乘子
   */
  updateDopamine(currentLoss: number): number {
    this.currentDopamine = this.dopamine.update(currentLoss);
    this.currentLrMultiplier = this.dopamine.currentMult;
    return this.currentLrMultiplier;
  }

  /**
   * 写入 Titans 记忆
   * @param h1 h1 隐藏状态
   * @returns 写入的槽位 ID
   */
  writeTitans(h1: Float64Array): number {
    return this.titans.write(h1);
  }

  /**
   * 读取 Titans 记忆
   */
  readTitans(query: Float64Array): { out: Float64Array; hitSlots: number } {
    return this.titans.read(query);
  }

  /**
   * 提取 STDP 知识规律
   */
  extractKnowledge(dim: number): import("./stdp").LearnedRule[] {
    if (this.stdp) {
      return this.stdp.extractKnowledge(dim);
    }
    return [];
  }

  /**
   * 清空 STDP/STDA 历史
   */
  clearHist(): void {
    if (this.stdp) this.stdp.clear();
    if (this.stda) this.stda.clear();
  }

  /**
   * 获取学习统计信息
   */
  getMetrics(): {
    stdp: { stepCount: number; rulesExtracted: number; activeDimensions: number };
    stda: { threshold: number; activity: number };
    replay: { occupancy: number; size: number };
    dopamine: { level: number; multiplier: number };
    titans: { occupancy: number };
    worldModel: { stepCount: number };
  } {
    return {
      stdp: this.stdp ? this.stdp.getMetrics() : { stepCount: 0, rulesExtracted: 0, activeDimensions: 0 },
      stda: this.stda ? { threshold: this.stda.threshold, activity: this.stda.activity } : { threshold: 0, activity: 0 },
      replay: { occupancy: this.replay.occupancy(), size: this.replay.size() },
      dopamine: { level: this.currentDopamine, multiplier: this.currentLrMultiplier },
      titans: { occupancy: this.titans.occupancy() },
      worldModel: { stepCount: 0 },
    };
  }

  /**
   * 保存学习状态
   */
  save(): Record<string, unknown> {
    return {
      stepCount: this.stepCount,
      currentDopamine: this.currentDopamine,
      currentLrMultiplier: this.currentLrMultiplier,
      titansState: this.titans.state(),
    };
  }

  /**
   * 加载学习状态
   */
  load(state: Record<string, unknown>): void {
    this.stepCount = state.stepCount as number || 0;
    this.currentDopamine = state.currentDopamine as number || 0;
    this.currentLrMultiplier = state.currentLrMultiplier as number || 1.0;
    if (state.titansState) {
      const s = state.titansState as { persistent: number[]; keys: number[]; values: number[] };
      // 注意：TitansMemory.state() 返回 Float64Array，这里需要做类型转换
    }
  }
}

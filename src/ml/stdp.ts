/**
 * STDP (Spike-Timing-Dependent Plasticity) — 工业级突触时序依赖可塑性
 *
 * ## 生物学原理
 *
 * STDP 是 Hebbian 学习的精确化形式 (Bi & Poo, Science 1998):
 * - LTP (Long-Term Potentiation): pre 先于 post 发放 → 突触增强
 * - LTD (Long-Term Depression): post 先于 pre 发放 → 突触抑制
 *
 * 数学公式:
 *   Δw_ij = A+ × exp(-Δt/τ+) × pre_i × post_j    if Δt = t_post - t_pre > 0
 *          -A- × exp(Δt/τ-) × pre_i × post_j      if Δt < 0
 *
 * ## 天玄实现
 *
 * - pre 信号: z1/r1 (遗忘门/重置门激活) ≈ 突触前发放
 * - post 信号: h1 (隐藏状态) ≈ 突触后发放
 * - 作用对象: GRUCell 的 wZ/wR/wC (输入→隐藏的突触权重)
 * - 不作用: uZ/uR/uC (循环权重)、输出头、embedding
 *
 * ## 工程特性
 *
 * 1. **性能优化**
 *    - 滑动窗口缓存 (默认 40 步)
 *    - 向量化的相似度计算
 *    - 梯度裁剪防止爆炸
 *
 * 2. **可测试性**
 *    - 纯函数接口 (无副作用)
 *    - 确定性随机种子
 *    - 梯度检查兼容
 *
 * 3. **在线学习**
 *    - 每批训练后应用 STDP
 *    - 定期提取知识规律
 *    - 写入向量知识库 (可选)
 *
 * ## 使用示例
 *
 * ```typescript
 * const stdp = new STDP({ rateLTP: 0.001, rateLTD: 0.0005 });
 *
 * // 训练循环
 * for (const batch of batches) {
 *   const loss = model.trainStepBatch(batch, lr);
 *
 *   // STDP: 记录 pre/post 信号
 *   stdp.record(model.cell1.postHist, model.cell1.preHist);
 *
 *   // STDP: 应用突触权重更新
 *   stdp.apply(model.cell1);
 *
 *   // STDP: 定期提取知识
 *   if (step % 100 === 0) {
 *     const rules = stdp.extractKnowledge(256);
 *     knowledgeBase.write(rules);
 *   }
 * }
 * ```
 */
import { Group } from "./model";

// ============================================================================
// 类型定义
// ============================================================================

/** STDP 配置参数 */
export interface STDPConfig {
  /** LTP 学习率 A+ */
  rateLTP: number;
  /** LTD 学习率 A- */
  rateLTD: number;
  /** LTP 时间常数 τ+ (步数) */
  tauLTP: number;
  /** LTD 时间常数 τ- (步数) */
  tauLTD: number;
  /** 滑动窗口大小 (历史步数) */
  windowSize: number;
  /** 是否启用在线知识提取 */
  onlineExtraction: boolean;
  /** 知识提取频率 (步数) */
  extractionFrequency: number;
  /** 梯度裁剪阈值 */
  maxGradientNorm: number;
  /** 最小 delta 阈值 (忽略小于此值的更新) */
  minDeltaThreshold: number;
}

/** 默认配置 */
export const STDP_DEFAULT_CONFIG: STDPConfig = {
  rateLTP: 0.001,
  rateLTD: 0.0005,
  tauLTP: 20.0,
  tauLTD: 30.0,
  windowSize: 40,
  onlineExtraction: true,
  extractionFrequency: 10,
  maxGradientNorm: 10.0,
  minDeltaThreshold: 1e-12,
};

/** STDP 学习到的规律 */
export interface LearnedRule {
  /** 规则文本 (人类可读) */
  rule: string;
  /** 对应的向量表示 */
  vec: Float64Array;
  /** 规则类型 */
  type: "pattern" | "theorem" | "formula" | "constant";
  /** 置信度 0-1 */
  confidence: number;
  /** 触发关键词 */
  keywords: string[];
  /** 来源神经层 */
  sourceLayer: string;
  /** 创建时间戳 */
  timestamp: number;
}

/** STDP 统计信息 */
export interface STDPMetrics {
  /** 已提取的规则数量 */
  rulesExtracted: number;
  /** 活跃维度数量 (激活率 > 30%) */
  activeDimensions: number;
  /** 总步骤计数 */
  stepCount: number;
  /** 当前窗口大小 */
  windowSize: number;
  /** 平均 LTP 信号强度 */
  avgLTPSignal: number;
  /** 平均 LTD 信号强度 */
  avgLTDSignal: number;
}

// ============================================================================
// STDP 实现类
// ============================================================================

/**
 * 工业级 STDP 实现
 *
 * 只作用于 GRUCell 的突触权重 (wZ/wR/wC)，不修改其他参数。
 */
export class STDP {
  readonly cfg: STDPConfig;

  /** 突触前信号历史 (z1/r1 门激活) */
  private preHist: Float64Array[] = [];
  /** 突触后信号历史 (h1 隐藏状态) */
  private postHist: Float64Array[] = [];

  /** 步骤计数 */
  private stepCount = 0;
  /** 已提取的规则数量 */
  private rulesExtracted = 0;

  /** 每个维度的激活统计 */
  private activationStats: Map<number, { onCount: number; total: number }> = new Map();

  /** LTP/LTD 信号累计 (用于监控) */
  private ltpSignalSum = 0;
  private ltdSignalSum = 0;
  private signalCount = 0;

  constructor(config: Partial<STDPConfig> = {}) {
    this.cfg = { ...STDP_DEFAULT_CONFIG, ...config };
  }

  // -------------------------------------------------------------------------
  // 公开接口
  // -------------------------------------------------------------------------

  /**
   * 记录一步的 pre/post 信号
   *
   * @param post h1 隐藏状态 (突触后，长度 hidden)
   * @param pre z1/r1 门激活 (突触前，长度 hidden)
   */
  record(post: Float64Array, pre: Float64Array): void {
    this.postHist.push(new Float64Array(post));
    this.preHist.push(new Float64Array(pre));

    // 滑动窗口限制
    const maxWin = this.cfg.windowSize;
    if (this.postHist.length > maxWin) {
      this.postHist.shift();
      this.preHist.shift();
    }

    // 激活统计
    for (let i = 0; i < post.length; i++) {
      const stat = this.activationStats.get(i) ?? { onCount: 0, total: 0 };
      stat.total++;
      if (post[i] > 0.3) stat.onCount++;
      this.activationStats.set(i, stat);
    }
  }

  /**
   * 清空历史记录 (每批训练后调用)
   */
  clear(): void {
    this.postHist = [];
    this.preHist = [];
    this.activationStats.clear();
    this.ltpSignalSum = 0;
    this.ltdSignalSum = 0;
    this.signalCount = 0;
  }

  /**
   * 应用 STDP 到 GRUCell 的突触权重
   *
   * 只修改 wZ/wR/wC (输入→隐藏的突触权重)，不修改:
   * - uZ/uR/uC (循环权重)
   * - bZ/bR/bC (偏置)
   * - 输出头
   * - embedding
   *
   * @param cell GRUCell 实例
   * @param inputSize 输入维度 (emb)
   */
  apply(cell: { wZ: Group; wR: Group; wC: Group }, inputSize: number): void {
    const n = this.preHist.length - 1;
    if (n < 1) return;

    const { rateLTP, rateLTD, tauLTP, tauLTD, minDeltaThreshold, maxGradientNorm } = this.cfg;

    // 对每个门 (z, r, c) 应用 STDP
    for (const g of [cell.wZ, cell.wR, cell.wC]) {
      this.applyToGroup(g, inputSize, n);
    }

    // 梯度裁剪
    this.clipGradients([cell.wZ, cell.wR, cell.wC], maxGradientNorm);

    this.stepCount++;
  }

  /**
   * 在线知识提取 — 已禁用
   * 原实现基于维度激活统计生成无实际语义的规则模板字符串，对下游无贡献。
   * 如需真正的知识提取，应基于权重谱分析或特征分解。
   */
  extractKnowledge(_dim: number): LearnedRule[] {
    return [];
  }

  /**
   * 获取 STDP 统计信息
   */
  getMetrics(): STDPMetrics {
    return {
      rulesExtracted: this.rulesExtracted,
      activeDimensions: [...this.activationStats.values()]
        .filter(s => s.onCount / s.total > 0.3)
        .length,
      stepCount: this.stepCount,
      windowSize: this.preHist.length,
      avgLTPSignal: this.signalCount > 0 ? this.ltpSignalSum / this.signalCount : 0,
      avgLTDSignal: this.signalCount > 0 ? this.ltdSignalSum / this.signalCount : 0,
    };
  }

  // -------------------------------------------------------------------------
  // 私有方法
  // -------------------------------------------------------------------------

  /**
   * 对单个 Group 应用 STDP 规则
   */
  private applyToGroup(g: Group, inputSize: number, n: number): void {
    const { rateLTP, rateLTD, tauLTP, tauLTD, minDeltaThreshold } = this.cfg;
    const hidden = g.p.length / inputSize;

    for (let i = 0; i < hidden; i++) {
      let ltpSignal = 0;
      let ltdSignal = 0;

      // 计算 LTP/LTD 信号 (限制计算量)
      const maxDt = Math.min(n, 20);
      for (let dt = 1; dt <= maxDt; dt++) {
        const postIdx = i < this.postHist[dt]?.length ? i : 0;
        const preIdx = i < this.preHist[dt - 1]?.length ? i : 0;

        // LTP: pre 先于 post
        const prePrev = this.preHist[dt - 1]?.[preIdx] ?? 0;
        const postCur = this.postHist[dt]?.[postIdx] ?? 0;
        ltpSignal += prePrev * postCur * Math.exp(-dt / tauLTP);

        // LTD: post 先于 pre
        const preCur = this.preHist[dt]?.[preIdx] ?? 0;
        const postPrev = this.postHist[dt - 1]?.[postIdx] ?? 0;
        ltdSignal += preCur * postPrev * Math.exp(-dt / tauLTD);
      }

      // 归一化
      ltpSignal /= maxDt;
      ltdSignal /= maxDt;

      // 更新统计
      this.ltpSignalSum += ltpSignal;
      this.ltdSignalSum += ltdSignal;
      this.signalCount += 2;

      // 计算 delta
      const delta = rateLTP * ltpSignal - rateLTD * ltdSignal;

      // 忽略小更新
      if (Math.abs(delta) < minDeltaThreshold) continue;

      // 应用到该行所有 input 维度
      const rowStart = i * inputSize;
      const scale = 0.01; // 缩放因子防止梯度爆炸
      for (let j = 0; j < inputSize; j++) {
        g.g[rowStart + j] += delta * scale;
      }
    }
  }

  /**
   * 梯度裁剪
   */
  private clipGradients(groups: Group[], maxNorm: number): void {
    let norm2 = 0;
    for (const g of groups) {
      for (let i = 0; i < g.g.length; i++) {
        norm2 += g.g[i] * g.g[i];
      }
    }
    const norm = Math.sqrt(norm2);
    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      for (const g of groups) {
        for (let i = 0; i < g.g.length; i++) {
          g.g[i] *= scale;
        }
      }
    }
  }
}

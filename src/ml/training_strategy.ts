/**
 * 训练策略优化 — 学习率调度、梯度裁剪、混合精度
 *
 * ## 设计原则
 *
 * 1. **学习率调度**：Warmup + Cosine Decay + 重启
 * 2. **梯度裁剪**：全局梯度范数裁剪 + 逐层裁剪
 * 3. **混合精度**：FP16/BF16 训练加速（简化版）
 * 4. **早停机制**：验证集监控防止过拟合
 */

export interface TrainingSchedule {
  /** 基础学习率 */
  baseLr: number;
  /** Warmup 步数 */
  warmupSteps: number;
  /** 总训练步数 */
  totalSteps: number;
  /** Cosine 重启周期 */
  cycleLength?: number;
  /** 最小学习率 */
  minLr: number;
  /** 梯度裁剪阈值 */
  maxGradNorm: number;
  /** 早停耐心值 */
  patience: number;
  /** 早停阈值改善 */
  minDelta: number;
}

export const DEFAULT_SCHEDULE: TrainingSchedule = {
  baseLr: 0.002,
  warmupSteps: 100,
  totalSteps: 10000,
  minLr: 0.0001,
  maxGradNorm: 1.0,
  patience: 5,
  minDelta: 0.001,
};

export interface TrainingMetrics {
  step: number;
  loss: number;
  gradNorm: number;
  lr: number;
  epoch: number;
}

/**
 * 学习率调度器
 */
export class LearningRateScheduler {
  readonly cfg: TrainingSchedule;
  private step = 0;
  private bestLoss = Infinity;
  private patienceCounter = 0;

  constructor(cfg: Partial<TrainingSchedule> = {}) {
    this.cfg = { ...DEFAULT_SCHEDULE, ...cfg };
  }

  /**
   * 计算当前学习率 (Warmup + Cosine Decay)
   */
  getLr(): number {
    this.step++;
    const { warmupSteps, totalSteps, baseLr, minLr, cycleLength } = this.cfg;

    // Warmup 阶段
    if (this.step <= warmupSteps) {
      return baseLr * (this.step / warmupSteps);
    }

    // Cosine Decay with Warm Restarts
    const progress = (this.step - warmupSteps) / (totalSteps - warmupSteps);
    let lr = minLr + 0.5 * (baseLr - minLr) * (1 + Math.cos(Math.PI * progress));

    // Warm Restarts
    if (cycleLength) {
      const cyclePos = (this.step - warmupSteps) % cycleLength;
      const cycleProgress = cyclePos / cycleLength;
      lr = minLr + 0.5 * (baseLr - minLr) * (1 + Math.cos(Math.PI * cycleProgress));
    }

    return lr;
  }

  /**
   * 更新调度器状态
   */
  update(loss: number): { lr: number; shouldStop: boolean } {
    const lr = this.getLr();

    // 早停检查
    if (loss < this.bestLoss - this.cfg.minDelta) {
      this.bestLoss = loss;
      this.patienceCounter = 0;
    } else {
      this.patienceCounter++;
    }

    const shouldStop = this.patienceCounter >= this.cfg.patience;

    return { lr, shouldStop };
  }

  /**
   * 重置早停计数器
   */
  resetPatience(): void {
    this.patienceCounter = 0;
    this.bestLoss = Infinity;
  }

  getProgress(): number {
    return Math.min(1, this.step / this.cfg.totalSteps);
  }

  getEpoch(): number {
    return Math.floor(this.step / this.cfg.warmupSteps);
  }
}

/**
 * 梯度裁剪器
 */
export class GradientClipper {
  private maxNorm: number;
  private clipHistory: number[] = [];
  private maxHistory = 100;

  constructor(maxNorm: number = 1.0) {
    this.maxNorm = maxNorm;
  }

  /**
   * 全局梯度裁剪
   * @returns 裁剪前的梯度范数
   */
  clip(groups: { p: Float64Array; g: Float64Array }[]): number {
    // 计算全局梯度范数
    let norm2 = 0;
    for (const g of groups) {
      for (let i = 0; i < g.g.length; i++) {
        norm2 += g.g[i] * g.g[i];
      }
    }
    const norm = Math.sqrt(norm2);

    // 记录历史
    this.clipHistory.push(norm);
    if (this.clipHistory.length > this.maxHistory) this.clipHistory.shift();

    // 裁剪
    if (norm > this.maxNorm) {
      const scale = this.maxNorm / norm;
      for (const g of groups) {
        for (let i = 0; i < g.g.length; i++) {
          g.g[i] *= scale;
        }
      }
    }

    return norm;
  }

  /**
   * 逐层梯度裁剪
   */
  clipPerLayer(groups: { name: string; p: Float64Array; g: Float64Array }[]): void {
    const layerNorm = 10.0; // 每层最大范数
    for (const g of groups) {
      let norm2 = 0;
      for (let i = 0; i < g.g.length; i++) norm2 += g.g[i] * g.g[i];
      const norm = Math.sqrt(norm2);
      if (norm > layerNorm) {
        const scale = layerNorm / norm;
        for (let i = 0; i < g.g.length; i++) g.g[i] *= scale;
      }
    }
  }

  getAvgClipNorm(): number {
    if (this.clipHistory.length === 0) return 0;
    return this.clipHistory.reduce((a, b) => a + b, 0) / this.clipHistory.length;
  }
}

/**
 * 混合精度训练器 (简化版)
 *
 * 注意：TypeScript 原生不支持 FP16，这里用 FP32 模拟
 * 实际部署时可切换到 tfjs-fp16 或 WebGPU
 */
export class MixedPrecisionTrainer {
  private scale = 1.0;
  private freezeThreshold = 2.0; // 梯度溢出阈值

  /**
   * 缩放梯度防止下溢
   */
  scaleGradients(groups: { g: Float64Array }[]): void {
    for (const g of groups) {
      for (let i = 0; i < g.g.length; i++) {
        g.g[i] *= this.scale;
      }
    }
  }

  /**
   * 取消缩放（在更新参数前）
   */
  unscaleGradients(groups: { g: Float64Array }[]): void {
    const invScale = 1 / this.scale;
    for (const g of groups) {
      for (let i = 0; i < g.g.length; i++) {
        g.g[i] *= invScale;
      }
    }
  }

  /**
   * 检测梯度溢出并动态调整 scale
   */
  adjustScale(groups: { g: Float64Array }[]): boolean {
    let maxGrad = 0;
    for (const g of groups) {
      for (let i = 0; i < g.g.length; i++) {
        maxGrad = Math.max(maxGrad, Math.abs(g.g[i]));
      }
    }

    if (maxGrad > this.freezeThreshold * 1e8) {
      // 溢出：缩小 scale
      this.scale /= 2;
      return false; // 跳过本次更新
    }

    if (maxGrad < this.freezeThreshold) {
      // 正常：增大 scale
      this.scale = Math.min(65504, this.scale * 2);
    }

    return true;
  }

  getScale(): number {
    return this.scale;
  }
}

/**
 * 训练策略管理器
 *
 * 整合学习率调度、梯度裁剪、混合精度
 */
export class TrainingStrategy {
  readonly scheduler: LearningRateScheduler;
  readonly clipper: GradientClipper;
  readonly mixedPrecision: MixedPrecisionTrainer;

  constructor(cfg: Partial<TrainingSchedule> = {}) {
    this.scheduler = new LearningRateScheduler(cfg);
    this.clipper = new GradientClipper(cfg.maxGradNorm);
    this.mixedPrecision = new MixedPrecisionTrainer();
  }

  /**
   * 执行一步训练
   */
  step(
    groups: { p: Float64Array; g: Float64Array }[],
    loss: number,
    adamStep: number
  ): { lr: number; gradNorm: number; skipped: boolean } {
    // 1. 梯度裁剪
    const gradNorm = this.clipper.clip(groups);

    // 2. 混合精度
    const canUpdate = this.mixedPrecision.adjustScale(groups);
    if (!canUpdate) {
      return { lr: this.scheduler.getLr(), gradNorm, skipped: true };
    }
    this.mixedPrecision.unscaleGradients(groups);

    // 3. Adam 更新（由调用方执行）
    const lr = this.scheduler.getLr();

    // 4. 更新调度器
    this.scheduler.update(loss);

    return { lr, gradNorm, skipped: false };
  }

  /**
   * 生成训练报告
   */
  report(): string {
    return [
      `== 训练策略报告 ==`,
      `当前学习率: ${this.scheduler.getLr().toExponential(4)}`,
      `进度: ${(this.scheduler.getProgress() * 100).toFixed(1)}%`,
      `平均梯度范数: ${this.clipper.getAvgClipNorm().toFixed(6)}`,
      `混合精度 scale: ${this.mixedPrecision.getScale()}`,
    ].join('\n');
  }
}

/**
 * 天玄训练日志系统 — 实时诊断输出
 *
 * 日志级别:
 * - INFO: 训练进度、损失值
 * - WARN: 异常检测、梯度爆炸
 * - DEBUG: 详细内部状态 (STDP、发放、黑板等)
 * - DIAG: 诊断信息 (调试用)
 */

export enum LogLevel {
  INFO = 0,
  WARN = 1,
  ERROR = 2,
  DEBUG = 3,
  DIAG = 4,
}

export interface LogEntry {
  ts: number;
  level: LogLevel;
  category: string;
  message: string;
  data?: Record<string, unknown>;
}

/** 日志格式化输出 */
function formatLog(entry: LogEntry): string {
  const time = new Date(entry.ts).toISOString().slice(11, 23);
  const levelStr = ['INFO', 'WARN', 'ERROR', 'DEBUG', 'DIAG'][entry.level];
  const catPad = entry.category.padEnd(12);
  return `[${time}] ${levelStr.padEnd(5)} | ${catPad} | ${entry.message}`;
}

/**
 * 训练日志器
 */
export class TrainLogger {
  private entries: LogEntry[] = [];
  private maxEntries = 10000;
  private logLevel: LogLevel;
  private startTime = Date.now();
  private lastLogTime = Date.now();

  // 训练统计
  private totalSteps = 0;
  private totalChars = 0;
  private totalLoss = 0;
  private lastLoss = 0;

  // 诊断数据
  private spikeHistory: number[] = [];
  private gradNormHistory: number[] = [];
  private lossHistory: number[] = [];

  constructor(options: { logLevel?: LogLevel } = {}) {
    this.logLevel = options.logLevel ?? LogLevel.INFO;
  }

  /**
   * 记录训练步骤
   */
  logTrainStep(
    epoch: number,
    totalEpochs: number,
    step: number,
    totalSteps: number,
    loss: number,
    diagnostics: {
      sparsity?: number;
      gradNorm?: number;
      stdpActive?: boolean;
      dopamine?: number;
      replayOccupancy?: number;
      titansOccupancy?: number;
    } = {},
  ): void {
    this.totalSteps++;
    this.lastLoss = loss;
    this.lossHistory.push(loss);

    if (diagnostics.sparsity !== undefined) {
      this.spikeHistory.push(diagnostics.sparsity);
    }
    if (diagnostics.gradNorm !== undefined) {
      this.gradNormHistory.push(diagnostics.gradNorm);
    }

    // INFO 级别: 打印进度
    if (this.totalSteps % 10 === 0 || step === totalSteps - 1) {
      const elapsed = (Date.now() - this.startTime) / 1000;
      const sps = this.totalSteps / elapsed;
      const avgLoss = this.lossHistory.slice(-100).reduce((a, b) => a + b, 0) / Math.min(this.lossHistory.length, 100);

      this.info(
        `epoch ${epoch}/${totalEpochs} step ${step}/${totalSteps}`,
        {
          loss: loss.toFixed(4),
          avgLoss: avgLoss.toFixed(4),
          sps: sps.toFixed(1),
          elapsed: `${elapsed.toFixed(1)}s`,
        },
      );
    }

    // DEBUG 级别: 打印诊断信息
    if (this.logLevel >= LogLevel.DEBUG) {
      const diag: Record<string, unknown> = {};
      if (diagnostics.sparsity !== undefined) diag.sparsity = diagnostics.sparsity.toFixed(4);
      if (diagnostics.gradNorm !== undefined) diag.gradNorm = diagnostics.gradNorm.toFixed(4);
      if (diagnostics.stdpActive !== undefined) diag.stdpActive = diagnostics.stdpActive;
      if (diagnostics.dopamine !== undefined) diag.dopamine = diagnostics.dopamine.toFixed(4);
      if (diagnostics.replayOccupancy !== undefined) diag.replayOcc = diagnostics.replayOccupancy.toFixed(3);
      if (diagnostics.titansOccupancy !== undefined) diag.titansOcc = diagnostics.titansOccupancy.toFixed(3);

      this.debug(`step ${step}`, diag);
    }

    // WARN: 检测异常
    if (isNaN(loss) || loss > 20) {
      this.warn(`loss 异常: ${loss.toFixed(4)}`, { loss, step });
    }
    if (diagnostics.gradNorm !== undefined && diagnostics.gradNorm > 10) {
      this.warn(`梯度爆炸: norm=${diagnostics.gradNorm.toFixed(2)}`, { gradNorm: diagnostics.gradNorm });
    }
  }

  /**
   * 记录验证结果
   */
  logValidation(
    epoch: number,
    valLoss: number,
    metrics: {
      top1?: number;
      recall?: number;
      perplexity?: number;
    } = {},
  ): void {
    const parts = [`val_loss=${valLoss.toFixed(4)}`];
    if (metrics.top1 !== undefined) parts.push(`top1=${(metrics.top1 * 100).toFixed(1)}%`);
    if (metrics.recall !== undefined) parts.push(`recall=${(metrics.recall * 100).toFixed(1)}%`);
    if (metrics.perplexity !== undefined) parts.push(`ppl=${metrics.perplexity.toFixed(2)}`);

    this.info(`epoch ${epoch} validation`, { parts: parts.join(' | ') });
  }

  /**
   * 记录 STDP 活动
   */
  logStdp(
    step: number,
    stats: {
      ltpSignal?: number;
      ltdSignal?: number;
      weightChanges?: number;
      activeNeurons?: number;
      totalNeurons?: number;
    },
  ): void {
    if (this.logLevel < LogLevel.DEBUG) return;

    const diag: Record<string, unknown> = {};
    if (stats.ltpSignal !== undefined) diag.ltp = stats.ltpSignal.toFixed(4);
    if (stats.ltdSignal !== undefined) diag.ltd = stats.ltdSignal.toFixed(4);
    if (stats.weightChanges !== undefined) diag.changes = stats.weightChanges;
    if (stats.activeNeurons !== undefined && stats.totalNeurons !== undefined) {
      diag.activePct = `${(stats.activeNeurons / stats.totalNeurons * 100).toFixed(1)}%`;
    }

    this.debug(`STDP step=${step}`, diag);
  }

  /**
   * 记录数据清洗结果
   */
  logDataCleaning(
    total: number,
    clean: number,
    poisoned: number,
    details?: Array<{ chunkId: string; score: number; reason: string }>,
  ): void {
    const ratio = poisoned / Math.max(total, 1);
    this.info(`data cleaning`, {
      total,
      clean,
      poisoned,
      ratio: `${(ratio * 100).toFixed(1)}%`,
    });

    if (ratio > 0.1 && this.logLevel >= LogLevel.WARN) {
      this.warn(`检测到 ${poisoned} 块异常数据 (${(ratio * 100).toFixed(1)}%)，建议审查数据来源`);
    }

    if (this.logLevel >= LogLevel.DIAG && details) {
      for (const d of details.slice(0, 5)) {
        this.diag(`poison_chunk_${d.chunkId}`, { score: d.score, reason: d.reason });
      }
    }
  }

  /**
   * 记录黑板状态
   */
  logBlackboard(key: string, value: unknown): void {
    if (this.logLevel < LogLevel.DIAG) return;
    this.diag(`blackboard[${key}]`, { value });
  }

  /**
   * 记录协同探针数据
   */
  logProbe(unitId: string, ignition: number, taskType: string): void {
    if (this.logLevel < LogLevel.DEBUG) return;
    this.debug(`probe[${unitId}]`, {
      ignition: ignition.toFixed(4),
      type: taskType,
    });
  }

  // ── 便捷方法 ────────────────────────────────────────────────

  info(message: string, data?: Record<string, unknown>): void {
    this._log(LogLevel.INFO, 'main', message, data);
  }

  warn(message: string, data?: Record<string, unknown>): void {
    this._log(LogLevel.WARN, 'warn', message, data);
  }

  error(message: string, data?: Record<string, unknown>): void {
    this._log(LogLevel.ERROR, 'error', message, data);
  }

  debug(message: string, data?: Record<string, unknown>): void {
    this._log(LogLevel.DEBUG, 'debug', message, data);
  }

  diag(message: string, data?: Record<string, unknown>): void {
    this._log(LogLevel.DIAG, 'diag', message, data);
  }

  private _log(level: LogLevel, category: string, message: string, data?: Record<string, unknown>): void {
    if (level < this.logLevel) return;

    const entry: LogEntry = {
      ts: Date.now(),
      level,
      category,
      message,
      data,
    };

    this.entries.push(entry);
    if (this.entries.length > this.maxEntries) {
      this.entries.shift();
    }

    // 实时输出
    const now = Date.now();
    if (now - this.lastLogTime > 1000 || level >= LogLevel.WARN) {
      console.log(formatLog(entry));
      this.lastLogTime = now;
    }
  }

  /**
   * 获取最近的日志条目
   */
  recent(count: number = 50): LogEntry[] {
    return this.entries.slice(-count);
  }

  /**
   * 生成报告
   */
  report(): string {
    const info = this.entries.filter(e => e.level === LogLevel.INFO).length;
    const warn = this.entries.filter(e => e.level === LogLevel.WARN).length;
    const error = this.entries.filter(e => e.level === LogLevel.ERROR).length;
    const debug = this.entries.filter(e => e.level === LogLevel.DEBUG).length;

    const avgLoss = this.lossHistory.length > 0
      ? (this.lossHistory.reduce((a, b) => a + b, 0) / this.lossHistory.length).toFixed(4)
      : 'N/A';

    const avgSpike = this.spikeHistory.length > 0
      ? (this.spikeHistory.reduce((a, b) => a + b, 0) / this.spikeHistory.length).toFixed(4)
      : 'N/A';

    return [
      `== 训练日志报告 ==`,
      `总步骤: ${this.totalSteps}`,
      `平均损失: ${avgLoss}`,
      `平均稀疏度: ${avgSpike}`,
      `日志条目: ${this.entries.length} (INFO:${info} WARN:${warn} ERROR:${error} DEBUG:${debug})`,
    ].join('\n');
  }

  /**
   * 导出日志为 JSON
   */
  export(): LogEntry[] {
    return [...this.entries];
  }
}

// ── 全局日志实例 ──────────────────────────────────────────────
export let logger: TrainLogger;

export function initLogger(options: { logLevel?: LogLevel } = {}): TrainLogger {
  logger = new TrainLogger(options);
  return logger;
}

/**
 * 协同探针(自研, 借鉴 Ignition Index 思路, 不抄代码):
 * 记录每个计算单元每次执行的点火值(activity), 输出:
 * - ignitionAvg: 滑窗平均点火(慢语义神经是否在"点火")
 * - ignitionVar: 点火抖动(全或无 vs 渐进)
 * 执行器据此做预算转向: 中央神经长期不点火 → 后续任务跳过该深度。
 */
export interface IgnitionSample {
  seq: number;
  taskId: string;
  unitId: string;
  kind: string;
  ignition: number;
  ts: number;
}

export class CollaborationProbe {
  private samples: IgnitionSample[] = [];
  private maxSamples = 4096;

  record(taskId: string, unitId: string, kind: string, ignition: number): void {
    this.samples.push({ seq: this.samples.length + 1, taskId, unitId, kind, ignition, ts: Date.now() });
    if (this.samples.length > this.maxSamples) this.samples.splice(0, this.samples.length - this.maxSamples);
  }

  /** 某单元累计样本数(暖机判断用) */
  sampleCount(unitId: string): number {
    let n = 0;
    for (const s of this.samples) if (s.unitId === unitId) n += 1;
    return n;
  }

  /** 某单元最近 window 条样本的平均点火(0..1) */
  ignitionAvg(unitId: string, window = 32): number {
    let sum = 0;
    let n = 0;
    for (let i = this.samples.length - 1; i >= 0 && n < window; i--) {
      if (this.samples[i].unitId !== unitId) continue;
      sum += this.samples[i].ignition;
      n += 1;
    }
    return n > 0 ? sum / n : 0;
  }

  /** 某单元最近 window 条样本的点火方差(全或无 → 高方差) */
  ignitionVar(unitId: string, window = 32): number {
    let n = 0;
    let sum = 0;
    for (let i = this.samples.length - 1; i >= 0 && n < window; i--) {
      if (this.samples[i].unitId !== unitId) continue;
      sum += this.samples[i].ignition;
      n += 1;
    }
    if (n === 0) return 0;
    const mean = sum / n;
    let v = 0;
    let m = 0;
    for (let i = this.samples.length - 1; i >= 0 && m < window; i--) {
      if (this.samples[i].unitId !== unitId) continue;
      v += (this.samples[i].ignition - mean) ** 2;
      m += 1;
    }
    return v / n;
  }

  /** 全部样本(审计回放用) */
  all(): IgnitionSample[] {
    return [...this.samples];
  }
}

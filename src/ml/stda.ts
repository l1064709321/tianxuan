/**
 * STDA (Spike-Timing-Dependent Adaptation) — 兴奋性/阈值适应规则
 *
 * 人脑机制: 神经元持续高激活 → 上调阈值(抑制)降低增益
 *           神经元长时间静默 → 下调阈值(兴奋)提高增益
 *
 * 实现: 直接修改输出头参数的基线, 形成等效的阈值滑动
 */
import { Group } from "./model";

export class STDA {
  readonly rate: number;
  readonly tau: number;

  private actSum = 0;
  private actCount = 0;
  private actAvg = 0.5;
  private thresholdShift = 0;

  constructor(rate = 0.0005, tau = 50.0) {
    this.rate = rate;
    this.tau = tau;
  }

  update(h2: Float64Array): void {
    let act = 0;
    const n = h2.length;
    for (let i = 0; i < n; i++) act += Math.abs(h2[i]);
    act /= n;
    this.actCount++;
    const alpha = 1.0 / (this.tau * 0.01 + 1.0);
    this.actAvg = (1 - alpha) * this.actAvg + alpha * act;
    const target = 0.5;
    this.thresholdShift += this.rate * (target - this.actAvg);
    this.thresholdShift = Math.max(-0.3, Math.min(0.3, this.thresholdShift));
  }

  clear(): void {
    this.actSum = 0;
    this.actCount = 0;
    this.actAvg = 0.5;
    this.thresholdShift = 0;
  }

  /**
   * 将阈值偏移直接作用于输出头的偏置:
   * 激活偏高时 → 偏置下移(抑制输出)
   * 激活偏低时 → 偏置上移(兴奋输出)
   */
  apply(groups: Group[]): void {
    // 只修改最后两个组(输出头偏置)
    const applyGroups = groups.slice(-2);
    for (const g of applyGroups) {
      for (let i = 0; i < g.g.length; i++) {
        g.g[i] *= (1.0 + this.thresholdShift);
      }
    }
  }

  get threshold(): number {
    return this.thresholdShift;
  }

  get activity(): number {
    return this.actAvg;
  }
}

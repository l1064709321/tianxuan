/**
 * STDA (Spike-Timing-Dependent Adaptation) — 兴奋性/阈值适应规则
 *
 * 原理: 神经元连续激活后自动上调"发放阈值",降低后续增益;
 *   长时间静默后自动下调阈值。用 h2 激活强度的滑动平均估算静默/活跃程度,
 *   动态调节输出头权重衰减率,实现等效的阈值适应。
 *
 * 与路由熵正则的互补:
 *   路由熵控制注意力选择的稀疏度;STDA 控制单个神经元的活性基线,
 *   两者作用于不同层级,可同时启用。
 */

import { Group } from "./model";

export class STDA {
  readonly rate: number;
  readonly tau: number;

  private actSum = 0;
  private actCount = 0;
  private actAvg = 0.5;
  private thresholdShift = 0;

  constructor(rate = 0.005, tau = 50.0) {
    this.rate = rate;
    this.tau = tau;
  }

  /** 每批训练结束后更新活性统计 */
  update(h2: Float64Array): void {
    let act = 0;
    const n = h2.length;
    for (let i = 0; i < n; i++) act += Math.abs(h2[i]);
    act /= n;
    // 指数移动平均
    this.actCount++;
    const alpha = 1.0 / (this.tau * 0.01 + 1.0);
    this.actAvg = (1 - alpha) * this.actAvg + alpha * act;
    // 阈值适应: 激活高 → 阈值上调(抑制); 激活低 → 阈值下调(兴奋)
    const target = 0.5;
    this.thresholdShift += this.rate * (target - this.actAvg);
    this.thresholdShift = Math.max(-0.5, Math.min(0.5, this.thresholdShift));
  }

  clear(): void {
    this.actSum = 0;
    this.actCount = 0;
    this.actAvg = 0.5;
    this.thresholdShift = 0;
  }

  /**
   * 对输出头参数施加阈值适应后的额外衰减。
   * 阈值上调时减弱输出权重(抑制),阈值下调时轻微增强(兴奋)。
   */
  apply(groups: Group[]): void {
    const factor = 1.0 + this.thresholdShift;
    for (let gi = 0; gi < groups.length; gi++) {
      const g = groups[gi];
      // 只修改输出层附近参数,避免干扰底层嵌入
      if (gi >= groups.length - 2) {
        for (let i = 0; i < g.g.length; i++) {
          g.g[i] *= factor;
        }
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

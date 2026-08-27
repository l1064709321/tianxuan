/**
 * STDP (Spike-Timing-Dependent Plasticity) — 突触权重可塑性规则
 *
 * 原理: 如果 presynaptic spike 在 postsynaptic spike 之前出现(△t > 0),
 *   权重加强;反之则减弱。用连续近似版本(非离散 spike),
 *   基于相邻时间步 h1 激活度的指数相关性实现。
 *
 * 与端到端训练的兼容性:
 *   STDP 作为附加局部正则项叠加到主梯度上,不替换 BPTT 反向传播。
 *   强度由 stdpRate 控制,默认 0.01,可调至 0。
 */

import { Group } from "./model";

export class STDP {
  readonly rate: number;
  readonly tau: number;

  private h1Hist: Float64Array[] = [];
  private preHist: Float64Array[] = [];

  constructor(rate = 0.01, tau = 20.0) {
    this.rate = rate;
    this.tau = tau;
  }

  /** 每批训练结束后记录 h1 和 pre 历史,供下一次 backward 时使用 */
  record(h1: Float64Array, pre: Float64Array): void {
    this.h1Hist.push(h1);
    this.preHist.push(pre);
    const maxWin = Math.max(20, Math.round(this.tau * 2));
    if (this.h1Hist.length > maxWin) {
      this.h1Hist.shift();
      this.preHist.shift();
    }
  }

  clear(): void {
    this.h1Hist = [];
    this.preHist = [];
  }

  /**
   * 对给定参数组加上 STDP 局部修正梯度。
   * 遍历窗口内相邻帧对 (i, i+1),计算 △h1 = h1[i+1] - h1[i],
   * 按 e^{-|△t|/tau} 加权后累加到每组梯度。
   */
  apply(groups: Group[]): void {
    if (this.h1Hist.length < 2) return;
    const n = this.h1Hist.length - 1;
    for (let k = 0; k < n; k++) {
      const dt = k + 1;
      const w = Math.exp(-dt / this.tau);
      const hCur = this.h1Hist[k + 1];
      const hPrev = this.h1Hist[k];
      const preCur = this.preHist[k + 1];
      const prePrev = this.preHist[k];
      // △h1 和 △pre 作为相关性信号
      for (let i = 0; i < hCur.length; i++) {
        const dh = (hCur[i] - hPrev[i]) * w;
        const dp = (preCur[i] - prePrev[i]) * w;
        for (const g of groups) {
          // 均匀分配到参数中,作为局部 Hebbian 相关项
          for (let j = 0; j < g.g.length; j++) {
            // 按组内位置分布: 这里用简单均匀采样替代复杂投影
            if ((j + i) % 3 === 0) {
              g.g[j] += this.rate * dh * 0.01;
            }
            if ((j + i + 1) % 3 === 0) {
              g.g[j] += this.rate * dp * 0.01;
            }
          }
        }
      }
    }
  }
}

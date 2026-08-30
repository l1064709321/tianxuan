/**
 * DopamineModulator — 多巴胺样奖赏预测误差调制器 (RPE-based)
 *
 * 人脑机制:
 *   多巴胺神经元在"意外奖赏"时发放增加, 在"预期落空"时发放减少。
 *   此处用预测误差 (loss前向 - loss前一步) 作为 RPE 信号:
 *   - RPE > 0 (比预期差): 多巴胺下降 → 降低学习率 (抑制不必要的更新)
 *   - RPE < 0 (比预期好): 多巴胺上升 → 提高学习率 (巩固有效突触)
 *   输出: lr_multiplier ∈ [0.1, 3.0], 平滑 EMA 轨迹
 *
 * 信用分配: 对每个参数按梯度范数比例缩放, 模拟多巴胺的"全域调制但差异化响应"
 */
export class DopamineModulator {
  /** EMA 平滑的 RPE 轨迹 */
  private emaRpe = 0;
  /** 基础学习率 */
  readonly baseLr: number;
  /** 当前有效 lr 乘子 */
  currentMult = 1.0;
  /** 滑窗大小 */
  private windowSize = 50;
  private recentLosses: number[] = [];
  /** 是否启用 */
  readonly enabled: boolean;
  /** EMA 衰减系数 */
  readonly tau: number;

  constructor(baseLr = 0.002, tau = 0.95, enabled = true) {
    this.baseLr = baseLr;
    this.tau = tau;
    this.enabled = enabled;
  }

  /** 更新: 传入当前 batch 平均 loss, 返回 lr 乘子 */
  update(loss: number): number {
    if (!this.enabled) { this.currentMult = 1.0; return 1.0; }
    this.recentLosses.push(loss);
    if (this.recentLosses.length > this.windowSize) this.recentLosses.shift();

    const rpe = this.recentLosses.length >= 2
      ? this.recentLosses[this.recentLosses.length - 1] - this.recentLosses[this.recentLosses.length - 2]
      : 0;

    // EMA 更新: RPE 负值 (loss 下降 = 惊喜) → 多巴胺升高
    const dopamine = -rpe * 2.0; // 放大信号
    this.emaRpe = this.tau * this.emaRpe + (1 - this.tau) * dopamine;

    // 映射到 lr 乘子: exp(emaRpe) 钳制到 [0.1, 3.0]
    this.currentMult = Math.max(0.1, Math.min(3.0, Math.exp(this.emaRpe)));
    return this.currentMult;
  }

  /** 对参数梯度施加多巴胺调制 (差异化信用分配) */
  modulateGradients(groups: { p: Float64Array; g: Float64Array }[]): void {
    if (!this.enabled) return;
    // 计算每组梯度范数作为"重要性"信号
    const norms = groups.map(g => {
      let s = 0;
      for (let i = 0; i < g.g.length; i++) s += g.g[i] * g.g[i];
      return Math.sqrt(s) + 1e-12;
    });
    const maxNorm = Math.max(...norms);
    for (let gi = 0; gi < groups.length; gi++) {
      const importance = norms[gi] / maxNorm;
      // 高重要性参数 → 更强多巴胺响应, 低重要性参数 → 微弱响应
      const factor = this.currentMult * (0.5 + 0.5 * importance);
      for (let i = 0; i < groups[gi].g.length; i++) {
        groups[gi].g[i] *= factor;
      }
    }
  }

  get dopamineLevel(): number { return this.emaRpe; }
  get emaLoss(): number {
    if (this.recentLosses.length === 0) return 0;
    const last = this.recentLosses[this.recentLosses.length - 1];
    return last;
  }
}

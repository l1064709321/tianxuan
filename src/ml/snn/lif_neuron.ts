/**
 * LIF (Leaky Integrate-and-Fire) 脉冲神经元 — 工业级实现
 *
 * ## 生物学原理
 *
 * LIF 是最简单的脉冲神经元模型 (Laing & Chow, 2002):
 *   τ_m × dV/dt = -(V - V_rest) + R_m × I_syn
 *
 * 发放条件: V >= V_thresh → 产生 spike → V = V_reset → 参考期
 *
 * ## 天玄实现
 *
 * - 膜电位 V (Float64)
 * - 泄漏项: -V/τ_m
 * - 输入电流: I_syn (来自突触权重 × pre spike)
 * - 阈值发放: V >= V_thresh → spike
 * - 参考期: spike 后 refractoryPeriod 步不发放
 * - 随机阈值: 每个神经元略有不同 (模拟生物异质性)
 */

export interface LIFConfig {
  /** 膜电位时间常数 (步数) */
  tauMem: number;
  /** 发放阈值 */
  vThresh: number;
  /** 重置电位 */
  vReset: number;
  /** 参考期长度 (步数) */
  refractoryPeriod: number;
  /** 输入电阻 (默认 1.0) */
  rInput: number;
  /** 静息电位 (默认 0.0) */
  vRest: number;
  /** 阈值噪声标准差 (模拟生物异质性) */
  threshNoise: number;
}

export const LIF_DEFAULT_CONFIG: LIFConfig = {
  tauMem: 10.0,       // 较慢的时间常数
  vThresh: 1.0,       // 标准阈值
  vReset: 0.0,
  refractoryPeriod: 2,
  rInput: 4.0,        // 适中增益
  vRest: -0.3,        // 静息电位
  threshNoise: 0.08,  // 异质性
};

/** LIF 神经元状态 */
export interface LIFState {
  /** 当前膜电位 */
  V: number;
  /** 剩余参考期步数 */
  refractory: number;
  /** 有效阈值 (带噪声) */
  vThreshEffective: number;
  /** 上一 timestep 的输入电流 */
  Ilast: number;
}

/** LIF 神经元输出 */
export interface LIFOutput {
  /** 是否发放 spike */
  spike: boolean;
  /** 膜电位 (发放前) */
  V: number;
  /** 发放时间戳 (全局步数) */
  spikeTime?: number;
}

/**
 * 单个 LIF 脉冲神经元
 */
export class LIFNeuron {
  readonly cfg: LIFConfig;
  state: LIFState;
  /** 全局发放计数 */
  firingCount = 0;
  /** 最近一次发放时间 */
  lastSpikeTime = -1;

  constructor(cfg: Partial<LIFConfig> = {}) {
    this.cfg = { ...LIF_DEFAULT_CONFIG, ...cfg };
    this.state = {
      V: this.cfg.vRest,
      refractory: 0,
      vThreshEffective: this.cfg.vThresh + (Math.random() - 0.5) * this.cfg.threshNoise,
      Ilast: 0,
    };
  }

  /**
   * 前向一步
   * @param Iext 外部输入电流
   * @param globalStep 全局时间步
   * @returns 输出 (是否发放 spike)
   */
  step(Iext: number, globalStep: number): LIFOutput {
    const { V, refractory, vThreshEffective, Ilast } = this.state;
    const { tauMem, vReset, rInput, vRest } = this.cfg;

    // 参考期递减
    const newRefractory = Math.max(0, refractory - 1);

    let spike = false;
    let newV = V;
    let I_syn = 0;

    if (newRefractory > 0) {
      // 参考期内: 膜电位保持重置值
      newV = vReset;
    } else {
      // Leaky integrate: dV/dt = (-V + I*R + V_rest) / tau
      I_syn = Iext * rInput;
      const dV = ((-V + I_syn + vRest) / tauMem) * 1.0;  // dt=1
      newV = V + dV;

      // 阈值发放
      if (newV >= vThreshEffective) {
        spike = true;
        this.firingCount++;
        this.lastSpikeTime = globalStep;
        newV = vReset;  // 发放后重置
      }
    }

    this.state = {
      V: newV,
      refractory: newRefractory,
      vThreshEffective,
      Ilast: I_syn,
    };

    return { spike, V: newV, spikeTime: spike ? globalStep : undefined };
  }

  /**
   * 重置状态
   */
  reset(): void {
    this.state = {
      V: this.cfg.vRest,
      refractory: 0,
      vThreshEffective: this.cfg.vThresh + (Math.random() - 0.5) * this.cfg.threshNoise,
      Ilast: 0,
    };
    this.firingCount = 0;
    this.lastSpikeTime = -1;
  }

  /**
   * 获取发放率 (最近 N 步)
   */
  firingRate(window: number = 100): number {
    if (this.lastSpikeTime < 0) return 0;
    const recent = this.lastSpikeTime;
    return Math.min(1, recent / window);
  }
}

/**
 * LIF 神经元层
 */
export class LIFLayer {
  readonly n: number;
  readonly neurons: LIFNeuron[];
  readonly cfg: LIFConfig;

  constructor(n: number, cfg: Partial<LIFConfig> = {}) {
    this.cfg = { ...LIF_DEFAULT_CONFIG, ...cfg };
    this.n = n;
    this.neurons = Array.from({ length: n }, () => new LIFNeuron(this.cfg));
  }

  /**
   * 批量前向
   * @param Iext 输入电流 [batch, n]
   * @param globalStep 全局时间步
   * @returns 输出 spike [batch, n]
   */
  step(Iext: Float64Array, globalStep: number): { spike: Float64Array; V: Float64Array } {
    const spike = new Float64Array(this.n);
    const V = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) {
      const out = this.neurons[i].step(Iext[i], globalStep);
      spike[i] = out.spike ? 1.0 : 0.0;
      V[i] = out.V;
    }
    return { spike, V };
  }

  /**
   * 重置所有神经元
   */
  reset(): void {
    for (const n of this.neurons) n.reset();
  }

  /**
   * 获取统计信息
   */
  getStats(): { firingRate: number; avgV: number; activeCount: number } {
    let totalSpike = 0;
    let totalV = 0;
    for (const n of this.neurons) {
      totalV += n.state.V;
      if (n.firingCount > 0) totalSpike++;
    }
    return {
      firingRate: totalSpike / this.n,
      avgV: totalV / this.n,
      activeCount: totalSpike,
    };
  }
}

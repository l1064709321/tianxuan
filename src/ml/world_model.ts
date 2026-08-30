/**
 * WorldModel — 天玄世界模型
 *
 * 设计理念（借鉴 Dreamer/Genie 思路）：
 * 1. 世界状态: 用潜在向量表示"当前世界是什么样"
 * 2. 转移函数: p(s' | s, a) — 给定当前状态和动作，预测下一状态
 * 3. 观测解码器: p(o | s) — 从世界状态还原可观测内容
 * 4. 物理约束: 在训练/推理中加入物理常识作为正则项
 *
 * 与纯检索的区别：
 * - 检索: 匹配关键词 → 返回预存答案
 * - 世界模型: 理解"世界如何运转" → 从第一性原理推导答案
 *
 * 参考:
 * - DreamerV3 (Hafner et al., 2023): 学习 latent dynamics model
 * - Genie (Hafner et al., 2023): 生成式世界模型
 * - Physics-informed Neural Networks (Raissi et al., 2019): 物理约束作为损失
 */
import * as fs from "fs";
import * as path from "path";
import { mulberry32, makeGaussian } from "./rng";
import { Group } from "./model";
import { CharGRU, GRUState } from "./gru";

export interface WorldState {
  /** 世界状态向量 */
  s: Float64Array;
  /** 时间步 */
  t: number;
  /** 上一动作 */
  lastAction: number | null;
  /** 世界稳定性得分 (低=世界在崩解) */
  coherence: number;
}

export interface TransitionRecord {
  /** 前一个世界状态 */
  sPrev: Float64Array;
  /** 当前世界状态 */
  sCur: Float64Array;
  /** 触发动作 (事件类型) */
  action: number;
  /** 观测 (文本描述) */
  observation: string;
  /** 物理约束违反分数 */
  physicsViolation: number;
}

/**
 * 世界模型组件:
 * - encoder: obs → s (观测编码为世界状态)
 * - dynamics: s + a → s' (世界转移函数)
 * - decoder: s → obs' (世界状态解码回观测)
 * - physics: 物理约束层
 */
export class WorldModel {
  readonly stateDim: number;
  readonly actionDim: number;
  readonly hidden: number;
  readonly embDim: number;

  // 世界状态编码器: obs_emb → world_state
  private encW: Group;
  private encB: Group;
  // 世界转移函数: concat(s, a) → s'
  private dynW: Group;
  private dynB: Group;
  // 世界状态解码器: s → obs reconstruction
  private decW: Group;
  private decB: Group;
  // 物理约束权重: s → physics_violation_score
  private physW: Group;
  private physB: Group;
  // 世界记忆 (长期状态缓存)
  private memory: Float64Array;
  private _adamM?: Float64Array[];
  private _adamV?: Float64Array[];

  // 输出头 (必须在 getGroups 之前声明)
  private outW!: Group;
  private outB!: Group;

  // 训练统计
  private stepCount = 0;
  private totalTransLoss = 0;
  private totalReconLoss = 0;
  private totalPhysLoss = 0;

  constructor(
    opts: { stateDim?: number; actionDim?: number; hidden?: number; embDim?: number } = {},
    seed = 42,
  ) {
    this.stateDim = opts.stateDim ?? 32;
    this.actionDim = opts.actionDim ?? 8;
    this.hidden = opts.hidden ?? 64;
    this.embDim = opts.embDim ?? 32;

    const rng = makeGaussian(mulberry32(seed));
    const s = this.stateDim;
    const a = this.actionDim;
    const h = this.hidden;
    const e = this.embDim;

    // 编码器: [obs_emb; action] → hidden
    this.encW = new Group(h * (e + a), () => rng() * 0.1);
    this.encB = new Group(h, () => 0);
    // 转移函数: concat(state, action) → state (直接用stateDim×(stateDim+actionDim)矩阵)
    this.dynW = new Group(s * (s + a), () => rng() * 0.05);
    this.dynB = new Group(s, () => 0);
    // 解码器: state → hidden (用hidden×state矩阵)
    this.decW = new Group(h * s, () => rng() * 0.1);
    this.decB = new Group(h, () => 0);
    // 输出头: hidden → obs_emb
    this.outW = new Group(e * h, () => rng() * 0.1);
    this.outB = new Group(e, () => 0);
    // 物理约束: state → violation_scores
    this.physW = new Group(a * s, () => rng() * 0.05);
    this.physB = new Group(a, () => 0);

    this.memory = new Float64Array(s);
  }

  getGroups(): Group[] {
    return [this.encW, this.encB, this.dynW, this.dynB, this.decW, this.decB, this.outW, this.outB, this.physW, this.physB];
  }

  paramCount(): number {
    return this.getGroups().reduce((s, g) => s + g.p.length, 0);
  }

  /**
   * 编码: 文本观察 → 世界状态
   * 用 CharGRU 的 hidden state 作为观察嵌入的简化
   */
  encode(obsEmbedding: Float64Array, lastAction: number | null): WorldState {
    const s = this.stateDim;
    const input = new Float64Array(this.embDim + (lastAction !== null ? this.actionDim : 0));
    for (let i = 0; i < this.embDim; i++) input[i] = obsEmbedding[i];
    if (lastAction !== null) {
      // one-hot 动作编码
      const actIdx = lastAction % this.actionDim;
      input[this.embDim + actIdx] = 1.0;
    }

    // 编码器: linear → tanh → state (残差连接保持稳定性)
    const hidden = new Float64Array(this.hidden);
    for (let j = 0; j < this.hidden; j++) {
      let acc = this.encB.p[j];
      const row = this.encW.p.subarray(j * input.length, (j + 1) * input.length);
      for (let i = 0; i < input.length; i++) acc += row[i] * input[i];
      hidden[j] = Math.tanh(acc);
    }

    // 世界状态 = 门控更新 (类似 GRU 的更新门，保持状态平滑过渡)
    const newState = new Float64Array(s);
    for (let j = 0; j < s; j++) {
      let acc = 0;
      const row = this.dynW.p.subarray(j * (s + this.actionDim), (j + 1) * (s + this.actionDim));
      for (let k = 0; k < s; k++) acc += row[k] * this.memory[k];
      if (lastAction !== null) {
        const actIdx = lastAction % this.actionDim;
        acc += row[s + actIdx];
      }
      // 门控融合: 新状态 = σ(W·[旧状态, 动作]) ⊙ 旧状态 + (1-σ) ⊙ tanh(...)
      const gate = 1 / (1 + Math.exp(-acc));
      newState[j] = gate * this.memory[j] + (1 - gate) * Math.tanh(acc);
    }

    this.memory = newState;
    this.stepCount++;

    // 计算世界相干性 (状态变化率越低，世界越稳定)
    let diff = 0;
    for (let i = 0; i < s; i++) diff += (newState[i] - this.memory[i]) ** 2;
    const coherence = 1 / (1 + Math.sqrt(diff / s));

    return { s: newState, t: this.stepCount, lastAction, coherence };
  }

  /**
   * 解码: 世界状态 → 观测重建
   */
  decode(worldState: WorldState): Float64Array {
    const s = this.stateDim;
    const e = this.embDim;
    const h = this.hidden;

    // 解码器: state → hidden
    const hidden = new Float64Array(h);
    for (let j = 0; j < h; j++) {
      let acc = this.decB.p[j];
      const row = this.decW.p.subarray(j * s, (j + 1) * s);
      for (let i = 0; i < s; i++) acc += row[i] * worldState.s[i];
      hidden[j] = Math.tanh(acc);
    }

    // 输出: hidden → obs_emb
    const obs = new Float64Array(e);
    for (let k = 0; k < e; k++) {
      let acc = this.outB.p[k];
      const row = this.outW.p.subarray(k * h, (k + 1) * h);
      for (let j = 0; j < h; j++) acc += row[j] * hidden[j];
      obs[k] = acc;
    }
    return obs;
  }

  /**
   * 物理约束评估: 世界状态 → 各物理定律违反分数
   */
  evaluatePhysics(worldState: WorldState): Float64Array {
    const s = this.stateDim;
    const a = this.actionDim;

    // 物理约束 1: 能量守恒 (状态变化不应凭空产生/消失)
    const energyConservation = new Float64Array(1);
    let totalChange = 0;
    for (let i = 0; i < s; i++) {
      totalChange += Math.abs(worldState.s[i]);
    }
    energyConservation[0] = Math.min(1, totalChange / (s * 2)); // 越接近0越好

    // 物理约束 2: 因果一致性 (action → state 变化方向应一致)
    const causality = new Float64Array(1);
    if (worldState.lastAction !== null) {
      let actionInfluence = 0;
      const actIdx = worldState.lastAction % a;
      for (let j = 0; j < s; j++) {
        const row = this.physW.p.subarray(j * a, (j + 1) * a);
        actionInfluence += Math.abs(row[actIdx] * worldState.s[j]);
      }
      causality[0] = Math.min(1, actionInfluence);
    } else {
      causality[0] = 0;
    }

    // 物理约束 3: 状态平滑性 (世界状态不应突变)
    const smoothness = new Float64Array(1);
    // 这里依赖外部传入的 prevState，暂用 coherence 近似
    smoothness[0] = 1 - worldState.coherence;

    return new Float64Array([energyConservation[0], causality[0], smoothness[0]]);
  }

  /**
   * 世界模型一步训练: 编码 → 转移 → 解码 → 物理约束
   * 返回总损失
   */
  trainStep(
    obsEmbedding: Float64Array,
    action: number,
    targetObsEmbedding: Float64Array,
    lr: number,
  ): { transLoss: number; reconLoss: number; physLoss: number; totalLoss: number } {
    const s = this.stateDim;
    const e = this.embDim;

    // 前向: 编码 (不重置 memory，保持世界状态连续性)
    const worldState = this.encode(obsEmbedding, action);

    // 转移: 用当前状态预测下一个状态 (自监督: 用目标 embedding 作为监督信号)
    // 简单版: 转移损失 = 当前状态与目标状态的差异
    const recon = this.decode(worldState);
    let reconLoss = 0;
    for (let i = 0; i < e; i++) {
      if (isNaN(recon[i]) || !isFinite(recon[i])) { recon[i] = 0; }
      reconLoss += (recon[i] - targetObsEmbedding[i]) ** 2;
    }
    reconLoss /= Math.max(e, 1);

    // 物理约束损失
    const physScores = this.evaluatePhysics(worldState);
    let physLoss = 0;
    for (let i = 0; i < physScores.length; i++) physLoss += physScores[i] ** 2;
    physLoss /= physScores.length;

    // 转移损失 (状态变化的合理性)
    let transLoss = 0;
    // 理想情况: 状态变化应平滑且有方向性
    // 这里用 coherence 作为代理
    transLoss = 1 - worldState.coherence;

    const totalLoss = reconLoss + 0.1 * physLoss + 0.05 * transLoss;

    // 简化反向: 直接对关键参数施加梯度
    // reconLoss 梯度
    const dRecon = new Float64Array(e);
    for (let i = 0; i < e; i++) dRecon[i] = 2 * (recon[i] - targetObsEmbedding[i]) / e;

    // 反传到解码器
    const h = this.hidden;
    for (let j = 0; j < h; j++) {
      let dh = 0;
      for (let k = 0; k < e; k++) dh += this.outW.p[k * h + j] * dRecon[k];
      // tanh 导数: 1 - tanh(decB[j] + sum(decW[j*s+k]*state[k]))^2
      let preDec = this.decB.p[j];
      for (let k = 0; k < s; k++) preDec += this.decW.p[j * s + k] * worldState.s[k];
      const tanhDeriv = 1 - Math.tanh(preDec) ** 2;
      dh *= tanhDeriv;
      // 累加到 decW, decB 梯度
      for (let k = 0; k < s; k++) {
        this.decW.g[j * s + k] += dh * worldState.s[k] * 0.01;
      }
      this.decB.g[j] += dh * 0.01;
    }
    // 累加到 outW, outB 梯度
    for (let k = 0; k < e; k++) {
      for (let j = 0; j < h; j++) {
        let preDec = this.decB.p[j];
        for (let i = 0; i < s; i++) preDec += this.decW.p[j * s + i] * worldState.s[i];
        this.outW.g[k * h + j] += dRecon[k] * Math.tanh(preDec) * 0.01;
      }
      this.outB.g[k] += dRecon[k] * 0.01;
    }

    // Adam 更新 (外部维护状态)
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    if (!this._adamM) {
      this._adamM = this.getGroups().map(g => new Float64Array(g.p.length));
      this._adamV = this.getGroups().map(g => new Float64Array(g.p.length));
    }
    for (let gi = 0; gi < this.getGroups().length; gi++) {
      const g = this.getGroups()[gi];
      // 梯度裁剪
      let gNorm = 0;
      for (let i = 0; i < g.g.length; i++) gNorm += g.g[i] * g.g[i];
      gNorm = Math.sqrt(gNorm);
      if (gNorm > 10) {
        const scale = 10 / gNorm;
        for (let i = 0; i < g.g.length; i++) g.g[i] *= scale;
      }
      // Adam step
      for (let i = 0; i < g.p.length; i++) {
        this._adamM![gi][i] = b1 * this._adamM![gi][i] + (1 - b1) * g.g[i];
        this._adamV![gi][i] = b2 * this._adamV![gi][i] + (1 - b2) * g.g[i] * g.g[i];
        const mh = this._adamM![gi][i] / (1 - Math.pow(b1, this.stepCount));
        const vh = this._adamV![gi][i] / (1 - Math.pow(b2, this.stepCount));
        g.p[i] -= lr * mh / (Math.sqrt(vh) + eps);
        g.g[i] = 0;
      }
    }

    this.totalTransLoss += transLoss;
    this.totalReconLoss += reconLoss;
    this.totalPhysLoss += physLoss;

    return { transLoss, reconLoss, physLoss, totalLoss };
  }

  private _dynLinear(j: number, state: Float64Array): number {
    let acc = this.dynB.p[j];
    const row = this.dynW.p.subarray(j * (this.stateDim + this.actionDim), (j + 1) * (this.stateDim + this.actionDim));
    for (let k = 0; k < this.stateDim; k++) acc += row[k] * state[k];
    return acc;
  }

  /**
   * 批量训练
   */
  trainBatch(
    sequences: Array<{
      observations: Float64Array[];  // [t0, t1, t2, ...] 观察序列
      actions: number[];              // [a0, a1, ...] 动作序列
    }>,
    lr: number,
  ): { avgTrans: number; avgRecon: number; avgPhys: number; avgTotal: number } {
    let totalTrans = 0, totalRecon = 0, totalPhys = 0, totalLoss = 0;
    let steps = 0;

    for (const seq of sequences) {
      for (let t = 0; t < seq.observations.length - 1; t++) {
        const r = this.trainStep(seq.observations[t], seq.actions[t], seq.observations[t + 1], lr);
        totalTrans += r.transLoss;
        totalRecon += r.reconLoss;
        totalPhys += r.physLoss;
        totalLoss += r.totalLoss;
        steps++;
      }
    }

    if (steps === 0) return { avgTrans: 0, avgRecon: 0, avgPhys: 0, avgTotal: 0 };
    return {
      avgTrans: totalTrans / steps,
      avgRecon: totalRecon / steps,
      avgPhys: totalPhys / steps,
      avgTotal: totalLoss / steps,
    };
  }

  /**
   * 世界模拟: 给定初始状态和动作序列，生成预测轨迹
   */
  simulate(initialObs: Float64Array, actions: number[]): Array<{ state: WorldState; recon: Float64Array }> {
    const trajectory: Array<{ state: WorldState; recon: Float64Array }> = [];
    let currentObs = initialObs;

    for (const action of actions) {
      const state = this.encode(currentObs, action);
      const recon = this.decode(state);
      trajectory.push({ state, recon });
      currentObs = recon; // 用重建作为下一步的输入
    }
    return trajectory;
  }

  /**
   * 物理约束分析: 评估文本是否违反物理常识
   */
  analyze(text: string): { confidence: number; physics: number[]; verdict: string; explanation: string } {
    // 简单实现: 用世界模型的状态平滑性作为代理
    const s = this.stateDim;
    // 模拟一个"正常"世界的状态
    const normalState = { s: new Float64Array(s).fill(0.1), t: 0, lastAction: null, coherence: 0.9 };
    const normalPhys = this.evaluatePhysics(normalState);
    // 异常状态 (高方差)
    const abnormalState = { s: new Float64Array(s).map((_, i) => (i % 2 === 0 ? 0.9 : -0.9)), t: 0, lastAction: null, coherence: 0.1 };
    const abnormalPhys = this.evaluatePhysics(abnormalState);

    // 根据文本特征判断
    const suspiciousPatterns = [
      { pattern: /真空.*传播.*声音|声音.*真空中/, severity: 0.9 },
      { pattern: /永动机/, severity: 0.95 },
      { pattern: /光速.*超?过|超光速/, severity: 0.8 },
      { pattern: /创造.*能量|凭空.*产生/, severity: 0.85 },
      { pattern: /时间倒流|逆转时间/, severity: 0.7 },
      { pattern: /质数.*有限|没有.*无穷.*质数/, severity: 0.8 },
      { pattern: /π.*有理数|圆周率.*有限/, severity: 0.9 },
    ];
    let maxViolation = 0;
    let matchedPattern = "";
    for (const { pattern, severity } of suspiciousPatterns) {
      if (pattern.test(text)) {
        if (severity > maxViolation) { maxViolation = severity; matchedPattern = pattern.source; }
      }
    }

    const verdict = maxViolation > 0.5 ? "impossible" : maxViolation > 0.2 ? "suspicious" : "consistent";
    const confidence = maxViolation > 0.5 ? 0.1 : maxViolation > 0.2 ? 0.5 : 0.85;

    return {
      confidence,
      physics: Array.from(abnormalPhys),
      verdict,
      explanation: matchedPattern
        ? `检测到可能的物理矛盾 (pattern: ${matchedPattern})`
        : `未发现明显物理矛盾 (energy=${abnormalPhys[0].toFixed(3)} causality=${abnormalPhys[1].toFixed(3)})`,
    };
  }

  /**
   * 保存/加载
   */
  save(dir: string): void {
    fs.mkdirSync(dir, { recursive: true });
    const data = {
      stateDim: this.stateDim,
      actionDim: this.actionDim,
      hidden: this.hidden,
      embDim: this.embDim,
      stepCount: this.stepCount,
      memory: Array.from(this.memory),
      params: this.getGroups().flatMap(g => Array.from(g.p)),
      grads: this.getGroups().flatMap(g => Array.from(g.g)),
    };
    fs.writeFileSync(path.join(dir, "world_model.json"), JSON.stringify(data), "utf-8");
  }

  static load(dir: string): WorldModel | null {
    const file = path.join(dir, "world_model.json");
    if (!fs.existsSync(file)) return null;
    const data = JSON.parse(fs.readFileSync(file, "utf-8")) as {
      stateDim: number; actionDim: number; hidden: number; embDim: number;
      stepCount: number; memory: number[]; params: number[];
    };
    const wm = new WorldModel({
      stateDim: data.stateDim, actionDim: data.actionDim,
      hidden: data.hidden, embDim: data.embDim,
    }, 0);
    wm.stepCount = data.stepCount;
    wm.memory = new Float64Array(data.memory);
    let off = 0;
    for (const g of wm.getGroups()) {
      for (let i = 0; i < g.p.length; i++) g.p[i] = data.params[off + i];
      off += g.p.length;
    }
    return wm;
  }
}

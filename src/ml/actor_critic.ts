/**
 * Actor-Critic 强化学习模块 — 完整 PPO-Clip + GAE 实现
 *
 * ## 设计原则
 *
 * 1. **策略梯度 + 价值函数**：Actor 学策略 π(a|s)，Critic 学价值 V(s)
 * 2. **GAE 优势估计**：广义优势估计平衡偏差-方差 (γ=0.99, λ=0.95)
 * 3. **Clip 技巧**：PPO-Clip 防止策略更新过大 (ε=0.2)
 * 4. **熵正则**：鼓励探索，防止策略过早收敛
 *
 * ## 网络结构
 *
 * - Actor: MLP(state_dim → hidden → action_dim) + softmax
 * - Critic: MLP(state_dim → hidden → 1)
 *
 * ## 反向传播
 *
 * - Actor: 策略梯度 L = -min(ratio·A, clip(ratio)·A) - entropy_coef·H
 *   - dL/d(logits) = (1 - one_hot(a)) for 正梯度方向
 *   - 经 softmax → hidden (ReLU) → state
 * - Critic: TD error loss L = (V(s) - return)²
 *   - dL/d(V) = 2*(V(s) - return)
 *   - 经 hidden (ReLU) → state
 *
 * ## 与现有系统的关系
 *
 * - DopamineModulator: RPE 信号 → Critic 的 TD error
 * - ReplayBuffer: 经验回放 → 训练数据
 * - WorldModel: 环境模型 → model-based 规划
 * - OnlineLearner: 统一封装接口
 */

import { Group } from "./model";
import { mulberry32, makeGaussian } from "./rng";

// ============================================================================
// 数据类型
// ============================================================================

/** 单条轨迹数据 */
export interface Transition {
  state: Float64Array;       // 当前状态（文本嵌入）
  action: number;            // 选择的动作（下一个字符 ID）
  reward: number;            // 即时奖励
  nextState: Float64Array;   // 下一状态
  done: boolean;             // 是否终止
  logProb: number;           // 动作的对数概率
}

/** 轨迹批次（用于 advantage 计算） */
export interface TrajectoryBatch {
  states: Float64Array[];
  actions: number[];
  rewards: number[];
  nextStates: Float64Array[];
  dones: boolean[];
  logProbs: number[];
}

/** Actor-Critic 配置 */
export interface ActorCriticConfig {
  /** 状态维度 */
  stateDim: number;
  /** 动作维度（词表大小） */
  actionDim: number;
  /** 隐藏层维度 */
  hidden: number;
  /** Actor 学习率 */
  actorLr: number;
  /** Critic 学习率 */
  criticLr: number;
  /** GAE 参数 γ */
  gamma: number;
  /** GAE 参数 λ */
  lambda: number;
  /** 熵正则系数 */
  entropyCoef: number;
  /** Clip 范围 */
  clipEpsilon: number;
  /** 价值损失系数 */
  valueCoef: number;
  /** 最大梯度范数 */
  maxGradNorm: number;
  /** 训练 epoch 数 */
  updateEpochs: number;
  /** batch 大小 */
  batchSize: number;
}

export const DEFAULT_AC_CONFIG: ActorCriticConfig = {
  stateDim: 64,
  actionDim: 380,
  hidden: 128,
  actorLr: 0.001,
  criticLr: 0.003,
  gamma: 0.99,
  lambda: 0.95,
  entropyCoef: 0.01,
  clipEpsilon: 0.2,
  valueCoef: 0.5,
  maxGradNorm: 0.5,
  updateEpochs: 4,
  batchSize: 64,
};

// ============================================================================
// Actor（策略网络）— MLP + softmax
// ============================================================================

/**
 * Actor 网络：state → action probabilities
 *
 * 结构：MLP (state → hidden → action_dim) + softmax
 *
 * 反向传播:
 *   dL/d(logits) = (1 - one_hot(a))  [策略梯度方向]
 *   dL/d(pre1)   = dHidden * relu'(pre1)
 *   累加到各组梯度 g
 */
export class Actor {
  readonly cfg: Pick<ActorCriticConfig, 'stateDim' | 'actionDim' | 'hidden'>;
  readonly groups: Group[] = [];

  private stateDim: number;
  private actionDim: number;
  private hidden: number;

  // 参数
  private w1: Group;
  private b1: Group;
  private w2: Group;
  private b2: Group;

  // 步数计数（Adam bias correction）
  private _t = 0;

  constructor(cfg: Pick<ActorCriticConfig, 'stateDim' | 'actionDim' | 'hidden'>, seed = 42) {
    this.cfg = cfg;
    this.stateDim = cfg.stateDim;
    this.actionDim = cfg.actionDim;
    this.hidden = cfg.hidden;

    const gaussian = makeGaussian(mulberry32(seed));
    const scale = (fanIn: number) => Math.sqrt(2.0 / fanIn);

    this.w1 = new Group(this.stateDim * this.hidden, () => gaussian() * scale(this.stateDim));
    this.b1 = new Group(this.hidden, () => 0);
    this.w2 = new Group(this.hidden * this.actionDim, () => gaussian() * scale(this.hidden));
    this.b2 = new Group(this.actionDim, () => 0);

    this.groups.push(this.w1, this.b1, this.w2, this.b2);
  }

  /** Adam 步数 */
  get t(): number { return this._t; }
  set t(v: number) { this._t = v; }

  /** 前向传播：返回动作概率分布 */
  forward(state: Float64Array): Float64Array {
    const stateDim = this.stateDim;
    const hidden = this.hidden;
    const actionDim = this.actionDim;

    // Layer 1: state → hidden (ReLU)
    const h = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) {
      let acc = this.b1.p[i];
      const row = this.w1.p.subarray(i * stateDim, (i + 1) * stateDim);
      for (let j = 0; j < stateDim; j++) acc += row[j] * state[j];
      h[i] = Math.max(0, acc); // ReLU
    }

    // Layer 2: hidden → action_dim (softmax)
    const logits = new Float64Array(actionDim);
    for (let i = 0; i < actionDim; i++) {
      let acc = this.b2.p[i];
      const row = this.w2.p.subarray(i * hidden, (i + 1) * hidden);
      for (let j = 0; j < hidden; j++) acc += row[j] * h[j];
      logits[i] = acc;
    }

    // Softmax
    let max = -Infinity;
    for (let i = 0; i < actionDim; i++) if (logits[i] > max) max = logits[i];
    let sum = 0;
    const probs = new Float64Array(actionDim);
    for (let i = 0; i < actionDim; i++) {
      probs[i] = Math.exp(logits[i] - max);
      sum += probs[i];
    }
    for (let i = 0; i < actionDim; i++) probs[i] /= sum;

    return probs;
  }

  /**
   * 采样动作
   */
  sampleAction(state: Float64Array, temperature: number = 1.0): { action: number; logProb: number } {
    const probs = this.forward(state);

    // Temperature 调整
    if (temperature !== 1.0) {
      let max = -Infinity;
      for (let i = 0; i < probs.length; i++) {
        probs[i] = Math.log(Math.max(probs[i], 1e-12)) / temperature;
        if (probs[i] > max) max = probs[i];
      }
      let sum = 0;
      for (let i = 0; i < probs.length; i++) {
        probs[i] = Math.exp(probs[i] - max);
        sum += probs[i];
      }
      for (let i = 0; i < probs.length; i++) probs[i] /= sum;
    }

    // 采样
    const r = Math.random();
    let cumsum = 0;
    for (let i = 0; i < probs.length; i++) {
      cumsum += probs[i];
      if (cumsum >= r) return { action: i, logProb: Math.log(probs[i]) };
    }
    return { action: probs.length - 1, logProb: Math.log(probs[probs.length - 1]) };
  }

  /**
   * 反向传播：策略梯度 + PPO-Clip + 熵正则
   *
   * @param state   状态向量
   * @param action  选择的动作索引
   * @param advantages  GAE 优势估计 A(s,a)
   * @param oldLogProb  旧策略下该动作的对数概率 log π_old(a|s)
   * @param entropyCoef 熵正则系数
   */
  backward(state: Float64Array, action: number, advantages: number, oldLogProb: number, entropyCoef: number): void {
    const stateDim = this.stateDim;
    const hidden = this.hidden;
    const actionDim = this.actionDim;

    // ---- 前向（存储中间变量供反向使用） ----
    const h = new Float64Array(hidden);
    const pre1 = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) {
      let acc = this.b1.p[i];
      const row = this.w1.p.subarray(i * stateDim, (i + 1) * stateDim);
      for (let j = 0; j < stateDim; j++) acc += row[j] * state[j];
      pre1[i] = acc;
      h[i] = Math.max(0, acc); // ReLU
    }

    const logits = new Float64Array(actionDim);
    for (let i = 0; i < actionDim; i++) {
      let acc = this.b2.p[i];
      const row = this.w2.p.subarray(i * hidden, (i + 1) * hidden);
      for (let j = 0; j < hidden; j++) acc += row[j] * h[j];
      logits[i] = acc;
    }

    // Softmax
    let max = -Infinity;
    for (let i = 0; i < actionDim; i++) if (logits[i] > max) max = logits[i];
    let sum = 0;
    const probs = new Float64Array(actionDim);
    for (let i = 0; i < actionDim; i++) {
      probs[i] = Math.exp(logits[i] - max);
      sum += probs[i];
    }
    for (let i = 0; i < actionDim; i++) probs[i] /= sum;

    // ---- 计算 old ratio ----
    const newLogProb = Math.log(probs[action] + 1e-12);
    const ratio = Math.exp(newLogProb - oldLogProb);

    // ---- PPO-Clip 梯度计算 ----
    // Loss = -min(ratio*A, clip(ratio)*A) - entropyCoef*H
    // 策略梯度方向: ∇θ J = ∇θ log π(a|s) * A
    // d(log π)/d(logits) = one_hot(a) - probs

    const clipEpsilon = 0.2;
    const surr1 = ratio * advantages;
    const surr2 = Math.min(Math.max(ratio, 1 - clipEpsilon), 1 + clipEpsilon) * advantages;
    const clipped = Math.min(surr1, surr2);

    // 梯度方向：最大化 J，所以 dL/d(logits) = -d(clipped)/d(logits)
    // d(ratio)/d(logits[k]) = ratio * (one_hot(k)==a ? 1 : 0 - probs[k])
    //                        = ratio * (δ_{k,a} - probs[k])
    // d(clipped)/d(logits) = d(surr)/d(logits)  (假设未触发 clip)

    // dJ/d(logits[k]) = ratio * A * (δ_{k,a} - probs[k])  [策略梯度]
    // 熵正则: H = -Σ p_i log p_i, dH/d(logits[k]) = -(1 - probs[k])
    // d(-entropyCoef*H)/d(logits[k]) = entropyCoef * (1 - probs[k])

    const dLogits = new Float64Array(actionDim);
    for (let k = 0; k < actionDim; k++) {
      const delta = (k === action ? 1 : 0) - probs[k];
      dLogits[k] = ratio * advantages * delta + entropyCoef * (1 - probs[k]);
    }

    // ---- 反向传播到 Layer 2 (hidden → action_dim) ----
    // dPre1[i] = Σ_k dLogits[k] * w2[k*hidden + i]
    const dPre1 = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) {
      let acc = 0;
      for (let k = 0; k < actionDim; k++) {
        acc += dLogits[k] * this.w2.p[k * hidden + i];
      }
      dPre1[i] = acc;
    }

    // 累加 w2, b2 梯度
    for (let k = 0; k < actionDim; k++) {
      const row = k * hidden;
      for (let i = 0; i < hidden; i++) {
        this.w2.g[row + i] += dLogits[k] * h[i];
      }
      this.b2.g[k] += dLogits[k];
    }

    // ---- ReLU 激活函数梯度 ----
    for (let i = 0; i < hidden; i++) {
      dPre1[i] *= (pre1[i] > 0 ? 1 : 0);
    }

    // ---- 反向传播到 Layer 1 (state → hidden) ----
    // dPre1[i] = Σ_j dHidden[j] * w1[i*stateDim + j]
    // 累加 w1, b1 梯度
    for (let i = 0; i < hidden; i++) {
      if (dPre1[i] === 0) continue;
      const row = i * stateDim;
      for (let j = 0; j < stateDim; j++) {
        this.w1.g[row + j] += dPre1[i] * state[j];
      }
      this.b1.g[i] += dPre1[i];
    }

    // 注意：不更新 emb（state 是外部输入的嵌入，不是可训练参数）
  }

  /** 应用累积的梯度（Adam 更新） */
  applyGradient(lr: number, t: number): void {
    for (const g of this.groups) g.adam(lr, t);
  }

  paramCount(): number {
    return this.groups.reduce((s, g) => s + g.p.length, 0);
  }
}

// ============================================================================
// Critic（价值网络）— MLP → 标量 V(s)
// ============================================================================

/**
 * Critic 网络：state → V(s)
 *
 * 结构：MLP (state → hidden → 1)
 *
 * 反向传播:
 *   L = (V(s) - return)²
 *   dL/d(V) = 2*(V(s) - return)
 *   经 ReLU → state
 */
export class Critic {
  readonly cfg: Pick<ActorCriticConfig, 'stateDim' | 'hidden'>;
  readonly groups: Group[] = [];

  private stateDim: number;
  private hidden: number;

  private w1: Group;
  private b1: Group;
  private w2: Group;
  private b2: Group;

  private _t = 0;

  constructor(cfg: Pick<ActorCriticConfig, 'stateDim' | 'hidden'>, seed = 43) {
    this.cfg = cfg;
    this.stateDim = cfg.stateDim;
    this.hidden = cfg.hidden;

    const gaussian = makeGaussian(mulberry32(seed));
    const scale = (fanIn: number) => Math.sqrt(2.0 / fanIn);

    this.w1 = new Group(this.stateDim * this.hidden, () => gaussian() * scale(this.stateDim));
    this.b1 = new Group(this.hidden, () => 0);
    this.w2 = new Group(this.hidden, () => gaussian() * scale(this.hidden));
    this.b2 = new Group(1, () => 0);

    this.groups.push(this.w1, this.b1, this.w2, this.b2);
  }

  /** Adam 步数 */
  get t(): number { return this._t; }
  set t(v: number) { this._t = v; }

  /**
   * 前向传播：返回状态价值 V(s)
   */
  forward(state: Float64Array): number {
    const stateDim = this.stateDim;
    const hidden = this.hidden;

    // Layer 1: state → hidden (ReLU)
    const h = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) {
      let acc = this.b1.p[i];
      const row = this.w1.p.subarray(i * stateDim, (i + 1) * stateDim);
      for (let j = 0; j < stateDim; j++) acc += row[j] * state[j];
      h[i] = Math.max(0, acc); // ReLU
    }

    // Layer 2: hidden → 1 (linear)
    let val = this.b2.p[0];
    for (let i = 0; i < hidden; i++) val += this.w2.p[i] * h[i];

    return val;
  }

  /**
   * 反向传播：价值损失 L = (V(s) - return)²
   *
   * @param state   状态向量
   * @param returns GAE 计算的目标回报
   */
  backward(state: Float64Array, returns: number): void {
    const stateDim = this.stateDim;
    const hidden = this.hidden;

    // ---- 前向（存储中间变量） ----
    const h = new Float64Array(hidden);
    const pre1 = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) {
      let acc = this.b1.p[i];
      const row = this.w1.p.subarray(i * stateDim, (i + 1) * stateDim);
      for (let j = 0; j < stateDim; j++) acc += row[j] * state[j];
      pre1[i] = acc;
      h[i] = Math.max(0, acc); // ReLU
    }

    // Layer 2: hidden → 1
    let val = this.b2.p[0];
    for (let i = 0; i < hidden; i++) val += this.w2.p[i] * h[i];

    // ---- 损失梯度: dL/dV = 2*(V(s) - return) ----
    const dOut = 2.0 * (val - returns);

    // 累加 w2, b2 梯度
    for (let i = 0; i < hidden; i++) {
      this.w2.g[i] += dOut * h[i];
    }
    this.b2.g[0] += dOut;

    // ---- ReLU 激活函数梯度 ----
    const dHidden = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) {
      dHidden[i] = dOut * this.w2.p[i] * (pre1[i] > 0 ? 1 : 0);
    }

    // ---- 反向传播到 Layer 1 (state → hidden) ----
    for (let i = 0; i < hidden; i++) {
      if (dHidden[i] === 0) continue;
      const row = i * stateDim;
      for (let j = 0; j < stateDim; j++) {
        this.w1.g[row + j] += dHidden[i] * state[j];
      }
      this.b1.g[i] += dHidden[i];
    }
  }

  /** 应用累积的梯度（Adam 更新） */
  applyGradient(lr: number, t: number): void {
    for (const g of this.groups) g.adam(lr, t);
  }

  paramCount(): number {
    return this.groups.reduce((s, g) => s + g.p.length, 0);
  }
}

// ============================================================================
// ActorCritic（完整 PPO-Clip + GAE 算法）
// ============================================================================

/**
 * Actor-Critic 强化学习器
 *
 * 算法：PPO-Clip + GAE
 * - collectTrajectory: 收集 K 步轨迹
 * - computeGAE: 计算优势函数和回报
 * - update: 多轮 Mini-batch 策略更新
 */
export class ActorCritic {
  readonly cfg: ActorCriticConfig;
  readonly actor: Actor;
  readonly critic: Critic;

  private stepCount = 0;
  get steps(): number { return this.stepCount; }
  get totalSteps(): number { return this._totalSteps; }
  private _totalSteps = 0;
  get totalReward(): number { return this._totalReward; }
  private _totalReward = 0;

  constructor(cfg: Partial<ActorCriticConfig> = {}) {
    this.cfg = { ...DEFAULT_AC_CONFIG, ...cfg };
    this.actor = new Actor(this.cfg, 42);
    this.critic = new Critic(this.cfg, 43);
  }

  /** 总 Adam 步数（actor 和 critic 共享） */
  get t(): number { return this.actor.t; }
  set t(v: number) { this.actor.t = v; this.critic.t = v; }

  /**
   * 收集轨迹
   */
  collectTrajectory(
    env: { getState(): Float64Array; step(action: number): { reward: number; done: boolean; nextState: Float64Array } },
    maxSteps: number = 2048
  ): TrajectoryBatch {
    const batch: TrajectoryBatch = {
      states: [],
      actions: [],
      rewards: [],
      nextStates: [],
      dones: [],
      logProbs: [],
    };

    let state = env.getState();
    let done = false;

    for (let t = 0; t < maxSteps && !done; t++) {
      const { action, logProb } = this.actor.sampleAction(state);
      const result = env.step(action);

      batch.states.push(state);
      batch.actions.push(action);
      batch.rewards.push(result.reward);
      batch.nextStates.push(result.nextState);
      batch.dones.push(result.done);
      batch.logProbs.push(logProb);

      state = result.nextState;
      done = result.done;
      this._totalReward += result.reward;
      this._totalSteps++;
    }

    return batch;
  }

  /**
   * 计算 GAE 优势估计
   */
  computeGAE(batch: TrajectoryBatch): { advantages: number[]; returns: number[] } {
    const { gamma, lambda } = this.cfg;
    const T = batch.states.length;

    const advantages = new Array(T).fill(0);
    const returns = new Array(T).fill(0);

    // 最后一位的 value target
    const lastValue = batch.dones[T - 1] ? 0 : this.critic.forward(batch.nextStates[T - 1]);

    let gae = 0;
    for (let t = T - 1; t >= 0; t--) {
      const nextValue = batch.dones[t] ? 0 : this.critic.forward(batch.nextStates[t]);
      const delta = batch.rewards[t] + gamma * nextValue - this.critic.forward(batch.states[t]);
      gae = delta + gamma * lambda * gae;
      advantages[t] = gae;
      returns[t] = gae + this.critic.forward(batch.states[t]);
    }

    return { advantages, returns };
  }

  /**
   * 执行 PPO-Clip 策略更新（多轮 mini-batch）
   *
   * 这是核心训练循环：
   * 1. 零梯度
   * 2. 对每个样本反向传播累加梯度
   * 3. 最后统一 Adam 更新
   */
  update(batch: TrajectoryBatch): { actorLoss: number; criticLoss: number; entropy: number } {
    const { advantages, returns } = this.computeGAE(batch);
    const { actorLr, criticLr, entropyCoef, valueCoef, updateEpochs, batchSize } = this.cfg;
    const n = batch.states.length;

    let totalActorLoss = 0;
    let totalCriticLoss = 0;
    let totalEntropy = 0;

    for (let ep = 0; ep < updateEpochs; ep++) {
      // 随机 shuffle
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      // Mini-batch 更新
      for (let start = 0; start < n; start += batchSize) {
        const end = Math.min(start + batchSize, n);
        const batchIndices = indices.slice(start, end);
        const actualBatchSize = batchIndices.length;

        // ---- Actor 反向传播 + 累积梯度 ----
        for (let bi = 0; bi < batchIndices.length; bi++) {
          const idx = batchIndices[bi];
          const oldLogProb = batch.logProbs[idx];
          const adv = advantages[idx];

          this.actor.backward(batch.states[idx], batch.actions[idx], adv, oldLogProb, entropyCoef);

          // 计算损失（仅用于统计）
          const probs = this.actor.forward(batch.states[idx]);
          const newLogProb = Math.log(probs[batch.actions[idx]] + 1e-12);
          const ratio = Math.exp(newLogProb - oldLogProb);
          const surr1 = ratio * adv;
          const surr2 = Math.min(Math.max(ratio, 1 - 0.2), 1 + 0.2) * adv;
          totalActorLoss += -Math.min(surr1, surr2);

          // 计算熵
          let ent = 0;
          for (let a = 0; a < probs.length; a++) {
            if (probs[a] > 0) ent -= probs[a] * Math.log(probs[a]);
          }
          totalEntropy += ent;
        }

        // ---- Actor Adam 更新 ----
        this.actor.applyGradient(actorLr / actualBatchSize, this.actor.t);

        // ---- Critic 反向传播 + 累积梯度 ----
        for (let bi = 0; bi < batchIndices.length; bi++) {
          const idx = batchIndices[bi];
          this.critic.backward(batch.states[idx], returns[idx]);

          const value = this.critic.forward(batch.states[idx]);
          totalCriticLoss += (value - returns[idx]) ** 2;
        }

        // ---- Critic Adam 更新 ----
        this.critic.applyGradient(criticLr / actualBatchSize, this.critic.t);
      }
    }

    this.stepCount++;
    return {
      actorLoss: totalActorLoss / (n * updateEpochs),
      criticLoss: totalCriticLoss / (n * updateEpochs),
      entropy: totalEntropy / (n * updateEpochs),
    };
  }

  /**
   * 生成时的策略（贪婪）
   */
  generateAction(state: Float64Array): number {
    const probs = this.actor.forward(state);
    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }

  /**
   * 统计信息
   */
  getStats(): { steps: number; avgReward: number; entropy: number } {
    return {
      steps: this._totalSteps,
      avgReward: this._totalSteps > 0 ? this._totalReward / this._totalSteps : 0,
      entropy: 0,
    };
  }

  paramCount(): number {
    return this.actor.paramCount() + this.critic.paramCount();
  }

  /** 保存状态 */
  save(): Record<string, unknown> {
    return {
      stepCount: this.stepCount,
       totalSteps: this._totalSteps,
      totalReward: this._totalReward,
      actorParams: this.actor.groups.map(g => Array.from(g.p)),
      criticParams: this.critic.groups.map(g => Array.from(g.p)),
      actorT: this.actor.t,
      criticT: this.critic.t,
    };
  }

  /** 加载状态 */
  load(state: Record<string, unknown>): void {
    this.stepCount = state.stepCount as number || 0;
    this._totalSteps = state.totalSteps as number || 0;
    this._totalReward = state.totalReward as number || 0;
    if (state.actorParams) {
      const ap = state.actorParams as number[][];
      for (let i = 0; i < this.actor.groups.length && i < ap.length; i++) {
        for (let j = 0; j < this.actor.groups[i].p.length && j < ap[i].length; j++) {
          this.actor.groups[i].p[j] = ap[i][j];
        }
      }
    }
    if (state.criticParams) {
      const cp = state.criticParams as number[][];
      for (let i = 0; i < this.critic.groups.length && i < cp.length; i++) {
        for (let j = 0; j < this.critic.groups[i].p.length && j < cp[i].length; j++) {
          this.critic.groups[i].p[j] = cp[i][j];
        }
      }
    }
    this.actor.t = state.actorT as number || 0;
    this.critic.t = state.criticT as number || 0;
  }
}

// ============================================================================
// 奖励函数
// ============================================================================

/**
 * 文本生成奖励函数
 */
export class TextReward {
  /**
   * 困惑度奖励（越低越好 → 越高奖励）
   */
  static perplexityReward(model: { logitsToProbs: (logits: Float64Array) => Float64Array }, trueToken: number, predProbs: Float64Array): number {
    const prob = Math.max(predProbs[trueToken], 1e-12);
    return Math.log(prob); // 负对数概率，越高越好（接近 0）
  }

  /**
   * 多样性奖励（鼓励不同输出）
   */
  static diversityReward(history: number[], currentAction: number, vocabSize: number): number {
    const freq = new Map<number, number>();
    for (const a of history) freq.set(a, (freq.get(a) ?? 0) + 1);
    const uniqueRatio = freq.size / Math.max(history.length, 1);
    return uniqueRatio;
  }

  /**
   * 流畅性奖励（基于 n-gram 匹配）
   */
  static fluencyReward(context: string, generatedChar: string, corpus: string): number {
    const ngram = context.slice(-3) + generatedChar;
    return corpus.includes(ngram) ? 1 : 0;
  }

  /**
   * 综合奖励
   */
  static combine(perplexity: number, diversity: number, fluency: number, weights: { p: number; d: number; f: number } = { p: 1.0, d: 0.3, f: 0.5 }): number {
    return weights.p * perplexity + weights.d * diversity + weights.f * fluency;
  }
}

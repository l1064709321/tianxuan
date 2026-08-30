import { Group } from "./model";
import { CharGRU, CharGRUConfig, GRUState } from "./gru";
import { MixtureOfExperts } from "./moe";
import { TitansMemory } from "./titans";
import { ReplayBuffer } from "./replay_buffer";
import { DopamineModulator } from "./dopamine";
import { WorldModel } from "./world_model";

export interface CharMultiNeuroConfig extends CharGRUConfig {
  moeTopK?: number;
  moeGateHidden?: number;
  moeLoadBalanceWeight?: number;
  /** MoE 专家数量 (默认50) */
  moeNExperts?: number;
  onlineTitans?: boolean;
  /** 重放缓冲区大小 */
  replayCapacity?: number;
  /** 每批从中采样的概率 */
  replayProb?: number;
  /** 重放 batch 大小(采样数) */
  replayBatchSize?: number;
  /** 多巴胺调制器 EMA 衰减 */
  dopamineTau?: number;
  /** 是否启用预测误差损失 */
  predictionLoss?: boolean;
  /** 预测误差损失权重 */
  predLossWeight?: number;
  /** MoE 脉冲近似温度 */
  moeSpikeTemperature?: number;
  /** MoE 脉冲近似阈值 */
  moeSpikeThreshold?: number;
}

export interface TitansWriteRecord {
  slot: number;
  h1: Float64Array;
}

/**
 * CharMultiNeuro — 端到端可微多神经协同 + 真正在线学习
 *
 * 人脑机制集成:
 * 1. 统一计算图: CharGRU backbone + MoE路由 + N个独立专家头 + Titans记忆偏差
 * 2. 预测误差(内在动机): h1 → pred_h2, MSE loss 加权, 模拟大脑预测编码
 * 3. 重放缓冲区: 训练时记录过渡样本, 按比例采样重放, 模拟海马体-皮层巩固
 * 4. 多巴胺调制: RPE-based lr 自适应, 模拟奖赏预测误差调控突触可塑性
 * 5. 脉冲近似门控: GRU z/r gate + MoE gate 使用阈值化, 模拟神经元发放
 * 6. STDP/STDA 直接修改参数梯度
 *
 * 神经链结构:
 *   感知神经 (GRU L1) → 中央神经 (Mamba/GRU L2 + Attention L3 + CNN L4)
 *   → 输出神经 (MoE路由 + 专家头)
 *   所有神经经共享工作空间 z(黑板)协同, 神经间不直接互调
 */
export class CharMultiNeuro {
  readonly cfg: CharMultiNeuroConfig;
  readonly model: CharGRU;
  readonly moe: MixtureOfExperts;
  readonly titans: TitansMemory;
  readonly replay: ReplayBuffer;
  readonly dopamine: DopamineModulator;
  /** 世界模型: 学习世界状态转移 p(s'|s,a) */
  readonly worldModel: WorldModel;

  private heads: Group[];
  private nExperts: number;

  private t = 0;
  private titansWriteLog: TitansWriteRecord[] = [];
  private prevLoss = Infinity;

  constructor(cfg: CharMultiNeuroConfig, seed = 42) {
    this.cfg = cfg;
    this.model = new CharGRU(cfg, seed);
    this.nExperts = cfg.moeNExperts ?? 50;
    this.moe = new MixtureOfExperts({
      nExperts: this.nExperts,
      topK: cfg.moeTopK ?? 2,
      gateHidden: cfg.moeGateHidden ?? 32,
      inputSize: cfg.emb,
      loadBalanceWeight: cfg.moeLoadBalanceWeight ?? 0.01,
      spikeTemperature: cfg.moeSpikeTemperature,
      spikeThreshold: cfg.moeSpikeThreshold,
    }, seed + 1);
    this.titans = new TitansMemory({ dim: cfg.emb, slots: 256 });
    this.replay = new ReplayBuffer(cfg.replayCapacity ?? 4096);
    this.dopamine = new DopamineModulator(
      0.002,
      cfg.dopamineTau ?? 0.95,
      !!(cfg.predictionHead ?? cfg.predictionLoss),
    );
    this.worldModel = new WorldModel({
      stateDim: Math.max(16, Math.floor(cfg.hidden / 4)),
      actionDim: 8,
      hidden: Math.max(32, Math.floor(cfg.hidden / 2)),
      embDim: cfg.emb,
    }, seed + 2);

    const V = cfg.vocabSize;
    const H = cfg.hidden;
    const rand = () => (Math.random() - 0.5) * 0.1;
    this.heads = Array.from({ length: this.nExperts }, () => new Group(V * H, rand));
  }

  getGroups(): Group[] {
    return [
      ...this.model.getGroups(),
      ...this.moe.getGroups(),
      ...this.heads,
    ];
  }

  paramCount(): number {
    return this.getGroups().reduce((s, g) => s + g.p.length, 0);
  }

  save(): number[] {
    const out: number[] = [];
    for (const g of this.getGroups()) {
      for (let i = 0; i < g.p.length; i++) out.push(g.p[i]);
    }
    return out;
  }

  load(params: number[]): void {
    let off = 0;
    for (const g of this.getGroups()) {
      for (let i = 0; i < g.p.length; i++) g.p[i] = params[off + i];
      off += g.p.length;
    }
  }

  setSteps(t: number): void { this.t = Math.max(1, Math.floor(t)); }
  getSteps(): number { return this.t; }

  /** 评估用: 整段前向, 返回每个位置的最终logits */
  forwardSeq(ids: number[]): Float64Array[] {
    const state = this.model.newState();
    const out: Float64Array[] = [];
    for (let i = 0; i < ids.length; i++) {
      out.push(this.step(ids[i], state, false));
    }
    return out;
  }

  /**
   * 单步前向(评估用): CharGRU(depth=4) + MoE路由 + Titans记忆偏差
   *
   * 【MoD 串行共享】depth=1/2/3/4 是串行递进的，不是独立重算：
   * - depth=1: cell1 → h1
   * - depth=2: h1 → cell2 → h2
   * - depth=3: h2 → attn → h3
   * - depth=4: h3 → cnn → output
   *
   * 每个深度复用前一层的输出，避免重复计算（符合 MoD 论文原意）。
   */
  step(xId: number, state: GRUState, onlineWrite = false): Float64Array {
    const { hidden: H, emb: E, vocabSize: V } = this.cfg;

    const snapH1 = state.h1.slice() as Float64Array;
    const snapH2 = state.h2.slice() as Float64Array;
    const snapHist = state.hist.map(h => h.slice() as Float64Array<ArrayBuffer>);
    const snapIds = [...state.ids];

    // 串行深度前向: depth=1 → depth=2 → depth=3 → depth=4
    // 每层复用前一层输出，不重复计算
    const s1: GRUState = { h1: snapH1.slice(), h2: snapH2.slice(), hist: snapHist, ids: snapIds };
    this.model.step(xId, s1, 1);
    const h1Expert = s1.h1;

    const s2: GRUState = { h1: s1.h1.slice(), h2: s1.h2.slice(), hist: s1.hist, ids: s1.ids };
    this.model.step(xId, s2, 2);
    const h2Expert = s2.h2;

    const s3: GRUState = { h1: s2.h1.slice(), h2: s2.h2.slice(), hist: s2.hist, ids: s2.ids };
    this.model.step(xId, s3, 3);
    const h3Expert = s3.h2;

    const s4: GRUState = { h1: s3.h1.slice(), h2: s3.h2.slice(), hist: s3.hist, ids: s3.ids };
    this.model.step(xId, s4, 4);
    const h4Expert = s4.h2;

    // 按层分配专家: 前4个用不同深度, 其余均匀分配
    const hiddenSources: Float64Array[] = [];
    for (let i = 0; i < this.nExperts; i++) {
      if (i === 0) hiddenSources.push(h1Expert);
      else if (i === 1) hiddenSources.push(h2Expert);
      else if (i === 2) hiddenSources.push(h3Expert);
      else if (i === 3) hiddenSources.push(h4Expert);
      else hiddenSources.push(h2Expert); // 其余用L2输出
    }

    // 批量计算所有专家 logits
    const expertLogits: Float64Array[] = [];
    for (let i = 0; i < this.nExperts; i++) {
      expertLogits.push(this.expertHeadLogits(this.heads[i], hiddenSources[i]));
    }

    const embVec = this.model.embedInput(xId);
    const route = this.moe.route(embVec);

    // MoE 路由聚合
    const combined = new Float64Array(V);
    for (let ei = 0; ei < route.expertIndices.length; ei++) {
      const idx = route.expertIndices[ei];
      const w = route.weights[ei];
      const el = expertLogits[idx];
      for (let k = 0; k < V; k++) combined[k] += w * el[k];
    }

    // Titans 记忆偏差
    if (this.cfg.onlineTitans) {
      const q = Float64Array.from(snapH1);
      const mem = this.titans.read(q);
      if (mem.hitSlots > 0) {
        const scale = 0.05;
        // 用第一个专家头做记忆偏差 (L1感知)
        const H = this.cfg.hidden;
        for (let k = 0; k < V; k++) {
          const row = k * H;
          for (let j = 0; j < H; j++) {
            combined[k] += this.heads[0].p[row + j] * mem.out[j] * scale;
          }
        }
      }
    }

    if (this.cfg.onlineTitans && onlineWrite) {
      const slot = this.titans.write(snapH1);
      this.titansWriteLog.push({ slot, h1: Float64Array.from(snapH1) });
    }

    return combined;
  }

  private expertHeadLogits(head: Group, hidden: Float64Array): Float64Array {
    const V = this.cfg.vocabSize;
    const H = this.cfg.hidden;
    const logits = new Float64Array(V);
    for (let k = 0; k < V; k++) {
      let acc = 0;
      const row = head.p.subarray(k * H, (k + 1) * H);
      for (let j = 0; j < H; j++) acc += row[j] * hidden[j];
      logits[k] = acc;
    }
    return logits;
  }

  /**
   * 训练一批序列: 端到端可微多神经协同 + 预测误差 + 重放 + 多巴胺 + STDP正则
   */
  trainStepBatch(seqs: Array<{ ids: number[] }>, lr: number): number {
    const { hidden: H, emb: E, vocabSize: V, bptt } = this.cfg;
    this.t += 1;
    for (const g of this.getGroups()) g.zeroGrad();

    let totalLoss = 0;
    let totalChars = 0;

    // ── Phase 1: CharGRU backbone BPTT (含预测误差) ──────────────
     const { loss: backboneLoss, totalChars: backboneChars } = this.model.trainStepBatch(seqs, lr);
     // [FIXED] 将 backbone 语言建模损失纳入 totalLoss，否则多巴胺调制看不到 backbone 学习进度
     totalLoss += backboneLoss * backboneChars;

    // ── Phase 1.5: STDP 正则项 (不直接改梯度，只加正则化损失) ────
    // STDP 正则项: -λ * Σ pre_i * post_i (鼓励突触前/后相关性)
    // 不与 BPTT 梯度冲突，仅作为辅助正则
    const stdpRegWeight = this.cfg.spikeThreshold ?? 0;
    let stdpRegLoss = 0;
    if (stdpRegWeight > 0 && this.model._stdpH1Hist.length > 0) {
      const h1Hist = this.model._stdpH1Hist;
      const preHist = this.model._stdpPreHist;
      const win = Math.min(h1Hist.length, preHist.length, 20);
      for (let w = 0; w < win; w++) {
        const post = h1Hist[h1Hist.length - win + w];
        const pre = preHist[preHist.length - win + w];
        if (post && pre) {
          let correl = 0;
          for (let i = 0; i < post.length; i++) correl += pre[i] * post[i];
          stdpRegLoss -= correl / post.length;  // 负相关 → 鼓励正相关
        }
      }
      stdpRegLoss *= stdpRegWeight * 0.01;
      totalLoss += stdpRegLoss * totalChars;  // 按比例缩放
    }

    // ── Phase 2: 收集专家 hidden states, 计算 MoE 路由 loss ──────
    const allSnapH1: Float64Array[][] = [];
    const allSnapH2: Float64Array[][] = [];
    const allSnapH3: Float64Array[][] = [];
    const allSnapH4: Float64Array[][] = [];
    const allLogits: Float64Array[][] = [];
    const allSeqIds: number[][] = [];

    for (const seq of seqs) {
      const T = seq.ids.length - 1;
      let state = this.model.newState();
      const seqSnapH1: Float64Array[] = [];
      const seqSnapH2: Float64Array[] = [];
      const seqSnapH3: Float64Array[] = [];
      const seqSnapH4: Float64Array[] = [];
      const seqLogits: Float64Array[] = [];
      const seqIds: number[] = [];

      for (let t = 0; t < T; t++) {
        seqSnapH1.push(state.h1.slice() as Float64Array);
        seqSnapH2.push(state.h2.slice() as Float64Array);

        const snapState: GRUState = {
          h1: state.h1.slice() as Float64Array,
          h2: state.h2.slice() as Float64Array,
          hist: state.hist.map(h => h.slice() as Float64Array<ArrayBuffer>),
          ids: [...state.ids],
        };
        const logits = this.step(seq.ids[t], snapState, false);
        // 收集各深度 hidden states
        seqSnapH3.push(snapState.h2.slice() as Float64Array);
        seqSnapH4.push(snapState.h2.slice() as Float64Array);
        seqLogits.push(logits);
        seqIds.push(seq.ids[t]);

        let max = -Infinity;
        for (let k = 0; k < V; k++) if (logits[k] > max) max = logits[k];
        let sum = 0;
        const pr = new Float64Array(V);
        // Logit clamp: MoE 多专家 logits 叠加可能超出 safe exp 范围(-100~100)
        for (let k = 0; k < V; k++) {
          const clamped = Math.max(-100, Math.min(100, logits[k]));
          pr[k] = Math.exp(clamped - max);
          sum += pr[k];
        }
        for (let k = 0; k < V; k++) pr[k] /= sum;
        totalLoss -= Math.log(pr[seq.ids[t + 1]] + 1e-12);
        totalChars += 1;

        this.replay.push({
          h1: seqSnapH1[seqSnapH1.length - 1],
          h2: seqSnapH2[seqSnapH2.length - 1],
          xId: seq.ids[t],
          yId: seq.ids[t + 1],
          predLoss: backboneLoss,
          nextPredLoss: 0,
        });
      }

      allSnapH1.push(seqSnapH1);
      allSnapH2.push(seqSnapH2);
      allSnapH3.push(seqSnapH3);
      allSnapH4.push(seqSnapH4);
      allLogits.push(seqLogits);
      allSeqIds.push(seqIds);
    }

    // ── Phase 3: 端到端反向 (MoE路由 + 专家头梯度) ───────────────
    for (let si = 0; si < seqs.length; si++) {
      const seq = seqs[si];
      const T = seq.ids.length - 1;
      const snapH1 = allSnapH1[si];
      const snapH2 = allSnapH2[si];
      const snapH3 = allSnapH3[si];
      const snapH4 = allSnapH4[si];

      for (let t = 0; t < T; t++) {
        const logits = allLogits[si][t];
        // Logit clamp: 与 Phase 2 保持一致，防止存储的原始未clamp logits 在反向时溢出
        const clampedLogits = new Float64Array(logits.length);
        for (let k = 0; k < logits.length; k++) clampedLogits[k] = Math.max(-100, Math.min(100, logits[k]));
        const dLogits = new Float64Array(V);
        let max = -Infinity;
        for (let k = 0; k < V; k++) if (clampedLogits[k] > max) max = clampedLogits[k];
        let sum = 0;
        const pr = new Float64Array(V);
        for (let k = 0; k < V; k++) { pr[k] = Math.exp(clampedLogits[k] - max); sum += pr[k]; }
        for (let k = 0; k < V; k++) dLogits[k] = pr[k] / sum;
        dLogits[seq.ids[t + 1]] -= 1;

        const embVec = this.model.embedInput(seq.ids[t]);
        const route = this.moe.route(embVec);

        // 按层分配专家 hidden (与 step() 推理一致: Expert0=h1, Expert1=h2, Expert2=h3, Expert3=h4, 其余=h2)
        const hiddenSources: Float64Array[] = [];
        for (let i = 0; i < this.nExperts; i++) {
          if (i === 0) hiddenSources.push(snapH1[t]);
          else if (i === 1) hiddenSources.push(snapH2[t]);
          else if (i === 2) hiddenSources.push(snapH3[t]);
          else if (i === 3) hiddenSources.push(snapH4[t]);
          else hiddenSources.push(snapH2[t]);
        }

        for (let ei = 0; ei < route.expertIndices.length; ei++) {
          const idx = route.expertIndices[ei];
          const w = route.weights[ei];
          const head = this.heads[idx];
          const h = hiddenSources[idx];
          for (let k = 0; k < V; k++) {
            const row = k * H;
            for (let j = 0; j < H; j++) {
              head.g[row + j] += w * dLogits[k] * h[j];
            }
          }
        }

        const lbW = this.cfg.moeLoadBalanceWeight ?? 0.01;
        if (lbW > 0) {
          for (const g of this.moe.getGroups()) {
            for (let i = 0; i < g.g.length; i++) g.g[i] += lbW * 0.001 * g.p[i];
          }
        }
      }
    }

    // ── Phase 4: 重放采样训练 (模拟海马体巩固) ──────────────────
    const replayProb = this.cfg.replayProb ?? 0.1;
    const replaySize = this.cfg.replayBatchSize ?? 8;
    if (Math.random() < replayProb && !this.replay.isEmpty()) {
      const samples = this.replay.sample(replaySize, true);
      if (samples.length > 0) {
        for (const sample of samples) {
          const snapState: GRUState = {
            h1: sample.h1.slice(),
            h2: sample.h2.slice(),
            hist: [],
            ids: [sample.xId],
          };
          const repLogits = this.step(sample.xId, snapState, false);
          let max = -Infinity;
          for (let k = 0; k < V; k++) if (repLogits[k] > max) max = repLogits[k];
          let sum = 0;
          const pr = new Float64Array(V);
          for (let k = 0; k < V; k++) { pr[k] = Math.exp(repLogits[k] - max); sum += pr[k]; }
          for (let k = 0; k < V; k++) pr[k] /= sum;
          totalLoss -= Math.log(pr[sample.yId] + 1e-12);
          totalChars += 1;

          const dRepLogits = new Float64Array(V);
          for (let k = 0; k < V; k++) dRepLogits[k] = pr[k];
          dRepLogits[sample.yId] -= 1;

          const embVec = this.model.embedInput(sample.xId);
          const route = this.moe.route(embVec);
          const rHiddenSources: Float64Array[] = [];
          for (let i = 0; i < this.nExperts; i++) {
            rHiddenSources.push(i === 0 ? sample.h1 : sample.h2);
          }
          for (let ei = 0; ei < route.expertIndices.length; ei++) {
            const idx = route.expertIndices[ei];
            const w = route.weights[ei];
            const head = this.heads[idx];
            const h = rHiddenSources[idx];
            for (let k = 0; k < V; k++) {
              const row = k * H;
              for (let j = 0; j < H; j++) {
                head.g[row + j] += w * dRepLogits[k] * h[j];
              }
            }
          }
        }
      }
    }

    // ── Phase 5: Titans 在线写入 ──────────────────────────────────
    if (this.cfg.onlineTitans) {
      for (const seq of seqs) {
        let state = this.model.newState();
        for (let t = 0; t < seq.ids.length - 1; t++) {
          const snapH1 = state.h1.slice() as Float64Array;
          const slot = this.titans.write(snapH1);
          this.titansWriteLog.push({ slot, h1: snapH1 });
          const snapState: GRUState = {
            h1: state.h1.slice() as Float64Array,
            h2: state.h2.slice() as Float64Array,
            hist: state.hist.map(h => h.slice() as Float64Array<ArrayBuffer>),
            ids: [...state.ids],
          };
          this.model.step(seq.ids[t], state, 4);
        }
      }
      this.titansWriteLog = [];
    }

    // ── Phase 5.5: 世界模型训练 ──────────────────────────────────
    const wm = this.worldModel;
    const wmSeq: Array<{ observations: Float64Array[]; actions: number[] }> = [];
    for (const seq of seqs) {
      const obs: Float64Array[] = [];
      const acts: number[] = [];
      let state = this.model.newState();
      for (let t = 0; t < seq.ids.length - 1; t++) {
        obs.push(state.h1.slice() as Float64Array);
        acts.push(seq.ids[t] % wm.actionDim);
        const snap: GRUState = { h1: state.h1.slice() as Float64Array, h2: state.h2.slice() as Float64Array, hist: [], ids: [...state.ids] };
        this.model.step(seq.ids[t], state, 1);
      }
      if (obs.length >= 2) wmSeq.push({ observations: obs, actions: acts });
    }
    if (wmSeq.length > 0) wm.trainBatch(wmSeq, lr * 0.1);

    // ── Phase 6: 归一化 + 裁剪 + 多巴胺调制 + Adam ────────────────
    // 使用 max(backboneChars, totalChars) 确保 backbone 和 MoE 梯度在同一尺度归一化
    const normDivisor = Math.max(backboneChars, totalChars);
    if (normDivisor > 0) {
      for (const g of [...this.heads, ...this.moe.getGroups()]) {
        for (let i = 0; i < g.g.length; i++) g.g[i] /= normDivisor;
      }
    }

    let norm2 = 0;
    for (const g of [...this.heads, ...this.moe.getGroups()]) {
      for (let i = 0; i < g.g.length; i++) norm2 += g.g[i] * g.g[i];
    }
    const norm = Math.sqrt(norm2);
    // PyTorch clip_grad_norm_ 文本模型标准: maxNorm=1.0
    // 参考: HuggingFace Transformers max_grad_norm=1.0, Google BERT clip_norm=1.0
    // 之前用5.0仍过高, norm频繁达到上限导致有效lr被过度压缩
    const maxNorm = 1.0;
    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      for (const g of [...this.heads, ...this.moe.getGroups()]) {
        for (let i = 0; i < g.g.length; i++) g.g[i] *= scale;
      }
    }

    const currentLoss = totalLoss / Math.max(normDivisor, 1);
    const dmMult = this.dopamine.update(currentLoss);
    const expertLr = lr * 0.5 * dmMult;

    const dmGroups = [...this.heads, ...this.moe.getGroups()];
    this.dopamine.modulateGradients(dmGroups);

    for (const g of this.heads) {
      g.adam(expertLr, this.t);
    }
    for (const g of this.moe.getGroups()) {
      g.adam(expertLr, this.t);
    }

    this.prevLoss = currentLoss;
    return currentLoss;
  }

  embedAvg(ids: number[]): number[] {
    return this.model.embedAvg(ids);
  }

  logitsToProbs(logits: Float64Array): Float64Array {
    return this.model.logitsToProbs(logits);
  }

  confidence(logits: Float64Array): number {
    return this.model.confidence(logits);
  }

  getTitansWriteLog(): TitansWriteRecord[] {
    return this.titansWriteLog;
  }

  /** 获取当前多巴胺水平 (用于日志) */
  getDopamineLevel(): number { return this.dopamine.dopamineLevel; }
  getReplayOccupancy(): number { return this.replay.occupancy(); }
  getPrevLoss(): number { return this.prevLoss; }
  /** 返回专家数量 */
  getNExperts(): number { return this.nExperts; }
}

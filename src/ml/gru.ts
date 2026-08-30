import { Group, MAX_PARAMS } from "./model";
import { SelectiveSSM, SSMCache } from "./mamba";
import { makeGaussian, mulberry32 } from "./rng";

export interface CharGRUConfig {
  vocabSize: number;
  emb: number;
  hidden: number;
  /** 记忆/上下文窗口宽度(推理与向量检索用) */
  ctx: number;
  /** BPTT 截断长度 */
  bptt: number;
  /** 稀疏 Attention 层(depth 3): 跨位置语义,独立参数参与训练 */
  attn?: boolean;
  /** CNN 层(depth 4): 1D 卷积局部特征,独立预测头与 GRU 输出融合 */
  cnn?: boolean;
  /** Mamba 真选择性层(depth 2): 替代第二层 GRU,快动力学线性状态流 */
  mamba?: boolean;
  /** 脉冲近似门控: 用阈值函数替代纯 sigmoid, h > threshold 才"发放" */
  spikeThreshold?: number;
  /** 下一步预测头: h1 → pred_h2, 预测误差作为内在动机信号 */
  predictionHead?: boolean;
  /** 预测误差损失权重 */
  predLossWeight?: number;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function tanh(x: number): number {
  if (x > 20) return 1;
  if (x < -20) return -1;
  const e = Math.exp(2 * x);
  return (e - 1) / (e + 1);
}

/** 单个 GRU 单元(线性状态流,纯 TS) */
export class GRUCell {
  readonly hidden: number;
  private inputSize: number;
  readonly wZ: Group;
  readonly wR: Group;
  readonly wC: Group;
  readonly uZ: Group;
  readonly uR: Group;
  readonly uC: Group;
  readonly bZ: Group;
  readonly bR: Group;
  readonly bC: Group;

  constructor(inputSize: number, hidden: number, rng: () => number, scale = 1) {
    this.inputSize = inputSize;
    this.hidden = hidden;
    const g = () => rng() * scale;
    this.wZ = new Group(inputSize * hidden, g);
    this.wR = new Group(inputSize * hidden, g);
    this.wC = new Group(inputSize * hidden, g);
    this.uZ = new Group(hidden * hidden, () => rng() * 0.1);
    this.uR = new Group(hidden * hidden, () => rng() * 0.1);
    this.uC = new Group(hidden * hidden, () => rng() * 0.1);
    this.bZ = new Group(hidden, () => 0);
    this.bR = new Group(hidden, () => 0);
    this.bC = new Group(hidden, () => 0);
  }

  groups(): Group[] {
    return [this.wZ, this.wR, this.wC, this.uZ, this.uR, this.uC, this.bZ, this.bR, this.bC];
  }

  /** 前向一步: x(长度 inputSize) + hPrev → 返回 { z, r, c, h } */
  forward(x: Float64Array, hPrev: Float64Array, cache: { z: Float64Array; r: Float64Array; c: Float64Array; h: Float64Array }): void {
    const H = this.hidden;
    const inp = this.inputSize;
    const pz = cache.z;
    const pr = cache.r;
    const pc = cache.c;
    const h = cache.h;
    // 标准 GRU: 先算 z/r(仅依赖 hPrev),再算候选 c = tanh(Wc·x + Uc·(r⊙hPrev))
    for (let j = 0; j < H; j++) {
      let az = this.bZ.p[j];
      let ar = this.bR.p[j];
      const wz = this.wZ.p.subarray(j * inp, (j + 1) * inp);
      const wr = this.wR.p.subarray(j * inp, (j + 1) * inp);
      const uz = this.uZ.p.subarray(j * H, (j + 1) * H);
      const ur = this.uR.p.subarray(j * H, (j + 1) * H);
      for (let i = 0; i < inp; i++) {
        az += wz[i] * x[i];
        ar += wr[i] * x[i];
      }
      for (let k = 0; k < H; k++) {
        az += uz[k] * hPrev[k];
        ar += ur[k] * hPrev[k];
      }
      pz[j] = sigmoid(az);
      pr[j] = sigmoid(ar);
    }
    for (let j = 0; j < H; j++) {
      let ac = this.bC.p[j];
      const wc = this.wC.p.subarray(j * inp, (j + 1) * inp);
      const uc = this.uC.p.subarray(j * H, (j + 1) * H);
      for (let i = 0; i < inp; i++) ac += wc[i] * x[i];
      for (let k = 0; k < H; k++) ac += uc[k] * (pr[k] * hPrev[k]);
      pc[j] = tanh(ac);
      h[j] = (1 - pz[j]) * hPrev[j] + pz[j] * pc[j];
    }
  }

  /** 反向一步(就地累加梯度) */
  backward(
    dH: Float64Array,
    x: Float64Array,
    hPrev: Float64Array,
    z: Float64Array,
    r: Float64Array,
    c: Float64Array,
    h: Float64Array,
    dx: Float64Array,
    dHPrev: Float64Array,
  ): void {
    const H = this.hidden;
    const inp = this.inputSize;
    const dz = new Float64Array(H);
    const dr = new Float64Array(H);
    const dc = new Float64Array(H);
    const dpreZ = new Float64Array(H);
    const dpreR = new Float64Array(H);
    const dpreC = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      dz[j] = dH[j] * (c[j] - hPrev[j]);
      dr[j] = 0;
      dc[j] = dH[j] * z[j];
    }
    // c 路径先算 a = r⊙hPrev 的梯度
    for (let j = 0; j < H; j++) {
      const dPre = dc[j] * (1 - c[j] * c[j]);
      dpreC[j] = dPre;
      const rowU = this.uC.p.subarray(j * H, (j + 1) * H);
      for (let k = 0; k < H; k++) {
        const grad = dPre * (r[k] * hPrev[k]);
        this.uC.g[j * H + k] += grad;
        dr[k] += dPre * this.uC.p[j * H + k] * hPrev[k];
        dHPrev[k] += dPre * this.uC.p[j * H + k] * r[k];
      }
      const rowW = this.wC.p.subarray(j * inp, (j + 1) * inp);
      for (let i = 0; i < inp; i++) this.wC.g[j * inp + i] += dPre * x[i];
      this.bC.g[j] += dPre;
    }
    for (let j = 0; j < H; j++) {
      const dPre = dz[j] * z[j] * (1 - z[j]);
      dpreZ[j] = dPre;
      const rowU = this.uZ.p.subarray(j * H, (j + 1) * H);
      const rowW = this.wZ.p.subarray(j * inp, (j + 1) * inp);
      for (let k = 0; k < H; k++) {
        this.uZ.g[j * H + k] += dPre * hPrev[k];
        dHPrev[k] += dPre * rowU[k];
      }
      for (let i = 0; i < inp; i++) this.wZ.g[j * inp + i] += dPre * x[i];
      this.bZ.g[j] += dPre;
    }
    for (let j = 0; j < H; j++) {
      const dPreR = dr[j] * r[j] * (1 - r[j]);
      dpreR[j] = dPreR;
      const rowU = this.uR.p.subarray(j * H, (j + 1) * H);
      const rowW = this.wR.p.subarray(j * inp, (j + 1) * inp);
      for (let k = 0; k < H; k++) {
        this.uR.g[j * H + k] += dPreR * hPrev[k];
        dHPrev[k] += dPreR * rowU[k];
      }
      for (let i = 0; i < inp; i++) this.wR.g[j * inp + i] += dPreR * x[i];
      this.bR.g[j] += dPreR;
    }
    // dH 直接路径 (1-z)
    for (let j = 0; j < H; j++) dHPrev[j] += dH[j] * (1 - z[j]);
    for (let i = 0; i < inp; i++) {
      let acc = 0;
      for (let j = 0; j < H; j++) {
        acc += this.wZ.p[j * inp + i] * dpreZ[j];
        acc += this.wR.p[j * inp + i] * dpreR[j];
        acc += this.wC.p[j * inp + i] * dpreC[j];
      }
      dx[i] += acc;
    }
  }
}

export interface GRUState {
  h1: Float64Array;
  h2: Float64Array;
  /** 最近 ctx 个 h2 历史(Attention 层跨位置交互用) */
  hist: Array<Float64Array<ArrayBufferLike>>;
  /** 最近 ctx 个输入字符 id(CNN 层局部窗口用) */
  ids: number[];
}

export interface AttnCache {
  K: Array<Float64Array<ArrayBufferLike>>;
  V: Array<Float64Array<ArrayBufferLike>>;
  /** RMS 归一化后的历史输入(权重梯度用,非投影) */
  Hn: Array<Float64Array<ArrayBufferLike>>;
  w: Float64Array;
  q: Float64Array;
  /** 门控残差: gate ∈ (0,1), read = h2 + gate·context */
  gate: number;
  gatePre: number;
  /** RMS 归一化后的当前输入(gate 权重梯度用) */
  qSrc: Float64Array<ArrayBufferLike>;
  /** 归一化前的当前 h1(RMS 雅可比反向用) */
  h1Raw: Float64Array<ArrayBufferLike>;
}

export interface CnnCache {
  ids: number[];
  embWin: Array<Float64Array<ArrayBufferLike>>;
  P: number;
  pre: Array<Float64Array<ArrayBufferLike>>;
  out: Array<Float64Array<ArrayBufferLike>>;
  feat: Float64Array<ArrayBufferLike>;
}

/**
 * 天玄序列模型 v1: 双层 GRU 字符语言模型(纯 TS)
 * - 深度 1: 第一层 GRU(状态流) + 输出头(浅/快)
 * - 深度 2: 叠加第二层 GRU(更深的语义状态)(预算内扩深)
 * - 训练: 截断 BPTT + Adam + 交叉熵
 */
export class CharGRU {
  readonly cfg: CharGRUConfig;
  private emb: Group;
  readonly cell1: GRUCell;
  private cell2: GRUCell;
  readonly ssm: SelectiveSSM | null;
  private attnQ: Group | null;
  private attnK: Group | null;
  private attnV: Group | null;
  private attnGW: Group | null;
  private attnGB: Group | null;
  /** [实验D] 诊断: 固定位置衰减分数 */
  fixedAttnBias = false;
  /** 诊断/协同监控: 收集每步 gate(注意力活跃度) */
  debugCollectGates: number[] | null = null;
  /** 协同探针: 最近一次 attention gate(0..1, 慢语义点火) */
  lastAttnGate = 0.5;
  /** 协同探针: 最近一次 CNN 激活均值(局部感知点火) */
  lastCnnAct = 0;
  /** 诊断/协同监控: 收集每步注意力权重分布(点火位置探针) */
  debugCollectAttn: Array<{ w: Float64Array }> | null = null;
  private cnnW: Group | null;
  private cnnB: Group | null;
  private outC: Group | null;
  private bC: Group | null;
  private out: Group;
  private bOut: Group;
  /** 下一步预测头: h1 → pred_h2 (内在动机/预测误差) */
  private predW: Group | null;
  private predB: Group | null;
  /** 脉冲近似门控阈值(>threshold 才发放) */
  private spikeThresh: number;
  private groups: Group[];
  private t = 0;

  /** Adam 累计更新步数(供检查点保存/恢复, 保持 bias correction 连续) */
  get steps(): number {
    return this.t;
  }

  setSteps(t: number): void {
    this.t = Math.max(1, Math.floor(t));
  }

  /** STDP/STDA 用的当前批次 h1 历史(最近 bptt 步) */
  _stdpH1Hist: Float64Array[] = [];
  /** STDP/STDA 用的当前批次 pre 历史(最近 bptt 步) */
  _stdpPreHist: Float64Array[] = [];
  /** STDA 用的最近 h2 向量(供外部统计) */
  _stdaLastH2: Float64Array | null = null;

  /** 供 STDP 等外部规则访问参数组(避免直接暴露私有字段) */
  getGroups(): Group[] {
    return this.groups;
  }

  /** 每批训练后清空 STDP/STDA 历史记录 */
  clearStdHist(): void {
    this._stdpH1Hist = [];
    this._stdpPreHist = [];
    this._stdaLastH2 = null;
  }

  constructor(cfg: CharGRUConfig, seed = 42) {
    this.cfg = cfg;
    const gaussian = makeGaussian(mulberry32(seed));
    this.emb = new Group(cfg.vocabSize * cfg.emb, () => gaussian() * 0.1);
    this.cell1 = new GRUCell(cfg.emb, cfg.hidden, gaussian, 1 / Math.sqrt(cfg.emb));
    this.cell2 = new GRUCell(cfg.hidden, cfg.hidden, gaussian, 1 / Math.sqrt(cfg.hidden));
    this.ssm = null;
    this.attnQ = null;
    this.attnK = null;
    this.attnV = null;
    this.attnGW = null;
    this.attnGB = null;
    this.cnnW = null;
    this.cnnB = null;
    this.outC = null;
    this.bC = null;
    if (cfg.attn) {
      const scale = 1 / Math.sqrt(cfg.hidden);
      this.attnQ = new Group(cfg.hidden * cfg.hidden, () => gaussian() * scale);
      this.attnK = new Group(cfg.hidden * cfg.hidden, () => gaussian() * scale);
      this.attnV = new Group(cfg.hidden * cfg.hidden, () => gaussian() * scale);
      this.attnGW = new Group(cfg.hidden, () => 0);
      this.attnGB = new Group(1, () => 0);
    }
    if (cfg.cnn) {
      const k = 3;
      const scale = 1 / Math.sqrt(cfg.emb * k);
      this.cnnW = new Group(k * cfg.emb * cfg.hidden, () => gaussian() * scale);
      this.cnnB = new Group(cfg.hidden, () => 0);
      this.outC = new Group(cfg.vocabSize * cfg.hidden, () => gaussian() * 0.1);
      this.bC = new Group(cfg.vocabSize, () => 0);
    }
    this.out = new Group(cfg.vocabSize * cfg.hidden, () => gaussian() * 0.1);
    this.bOut = new Group(cfg.vocabSize, () => 0);
    this.spikeThresh = cfg.spikeThreshold ?? 0.3;
    if (cfg.predictionHead) {
      const scale = 1 / Math.sqrt(cfg.hidden);
      this.predW = new Group(cfg.hidden * cfg.hidden, () => gaussian() * scale);
      this.predB = new Group(cfg.hidden, () => 0);
    } else {
      this.predW = null;
      this.predB = null;
    }
    if (cfg.mamba) {
      const ss = 1 / Math.sqrt(cfg.hidden * 2);
      this.ssm = new SelectiveSSM({ input: cfg.hidden, dState: 16 }, (i) => gaussian() * ss);
    }
    this.groups = [
      this.emb,
      ...this.cell1.groups(),
      ...(this.ssm ? this.ssm.groupsAll() : this.cell2.groups()),
      ...(cfg.attn ? [this.attnQ!, this.attnK!, this.attnV!, this.attnGW!, this.attnGB!] : []),
      ...(cfg.cnn ? [this.cnnW!, this.cnnB!, this.outC!, this.bC!] : []),
      ...(this.predW ? [this.predW!, this.predB!] : []),
      this.out,
      this.bOut,
    ];
  }

  paramCount(): number {
    return this.groups.reduce((s, g) => s + g.p.length, 0);
  }

  newState(): GRUState {
    // SSM 模式下 h2 是展平状态向量 (length = hidden * dState)，GRU 模式下是 hidden 向量。
    // 调用方访问 h2 时需确认 ssm 是否启用。
    return { h1: new Float64Array(this.cfg.hidden), h2: this.ssm ? this.ssm.newState() : new Float64Array(this.cfg.hidden), hist: [], ids: [] };
  }

  private embRow(id: number, out: Float64Array): void {
    const { vocabSize: v, emb } = this.cfg;
    out.set(this.emb.p.subarray((id % v) * emb, ((id % v) + 1) * emb));
  }

  /** 公开版: 获取字符嵌入向量(供CharMultiNeuro等外部类使用) */
  embedInput(id: number): Float64Array {
    const out = new Float64Array(this.cfg.emb);
    this.embRow(id, out);
    return out;
  }

  /** 推理一步(推进 state,可传副本做"窥视") */
  step(xId: number, state: GRUState, depth: 1 | 2 | 3 | 4): Float64Array {
    const { hidden: H, emb: E, vocabSize: V } = this.cfg;
    state.ids.push(xId);
    if (state.ids.length > this.cfg.ctx) state.ids.shift();
    const x = new Float64Array(E);
    this.embRow(xId, x);
    const z = new Float64Array(H);
    const r = new Float64Array(H);
    const c = new Float64Array(H);
    const h = new Float64Array(H);
    this.cell1.forward(x, state.h1, { z, r, c, h });
    state.h1.set(h);
    let read: Float64Array<ArrayBufferLike> = h;
    if (depth >= 2) {
      let x2 = h;
      if (depth >= 3 && this.attnQ) {
        const cache: AttnCache = { K: [], V: [], Hn: [], w: new Float64Array(0), q: new Float64Array(0), gate: 0, gatePre: 0, qSrc: new Float64Array(0), h1Raw: new Float64Array(0) };
        const attnOut = this.attnForward(h, state.hist, cache);
        x2 = new Float64Array(H);
        for (let i = 0; i < H; i++) x2[i] = h[i] + attnOut[i];
      }
      if (this.ssm) {
        const sc = this.ssm.newCache();
        this.ssm.forward(x2, state.h2, sc);
        state.h2.set(sc.h);
        read = sc.y;
      } else {
        const z2 = new Float64Array(H);
        const r2 = new Float64Array(H);
        const c2 = new Float64Array(H);
        const h2 = new Float64Array(H);
        this.cell2.forward(x2, state.h2, { z: z2, r: r2, c: c2, h: h2 });
        state.h2.set(h2);
        read = h2;
      }
      if (this.attnQ) {
        state.hist.push(read as Float64Array<ArrayBuffer>);
        if (state.hist.length > this.cfg.ctx) state.hist.shift();
      }
    }
    const logits = new Float64Array(V);
    for (let k = 0; k < V; k++) {
      let acc = this.bOut.p[k];
      const row = this.out.p.subarray(k * H, (k + 1) * H);
      for (let j = 0; j < H; j++) acc += row[j] * read[j];
      logits[k] = acc;
    }
    if (depth >= 4 && this.cnnW) {
      const cnn = this.cnnForward(state.ids);
      for (let k = 0; k < V; k++) logits[k] += cnn.logits[k];
    }
    return logits;
  }

  /** 评估用: 从零状态整段前向,返回每个位置的 logits */
  forwardSeq(ids: number[], depth: 2 | 3 | 4): Float64Array[] {
    const state = this.newState();
    const out: Float64Array[] = [];
    for (const id of ids) out.push(this.step(id, state, depth));
    return out;
  }

  /** 学习型自注意力前向(窗口 = hist,残差输出) */
  private attnForward(h1Cur: Float64Array<ArrayBufferLike>, hist: Array<Float64Array<ArrayBufferLike>>, cache: AttnCache): Float64Array<ArrayBufferLike> {
    const H = this.cfg.hidden;
    const L = hist.length;
    const norm = (v: Float64Array): Float64Array<ArrayBuffer> => {
      let sq = 0;
      for (let i = 0; i < v.length; i++) sq += v[i] * v[i];
      const rms = Math.sqrt(sq / v.length) + 1e-8;
      const out = new Float64Array(v.length);
      for (let i = 0; i < v.length; i++) out[i] = v[i] / rms;
      return out;
    };
    const qSrc = norm(h1Cur);
    const histN = hist.map((h) => norm(h));
    const q = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      let acc = 0;
      const row = this.attnQ!.p.subarray(j * H, (j + 1) * H);
      for (let i = 0; i < H; i++) acc += row[i] * qSrc[i];
      q[j] = acc;
    }
    const K: Array<Float64Array<ArrayBuffer>> = [];
    const V: Array<Float64Array<ArrayBuffer>> = [];
    const scores = new Float64Array(L);
    const sqrtH = Math.sqrt(H);
    for (let si = 0; si < L; si++) {
      const k = new Float64Array(H);
      const v = new Float64Array(H);
      const hs = histN[si];
      for (let j = 0; j < H; j++) {
        let ak = 0;
        let av = 0;
        const rk = this.attnK!.p.subarray(j * H, (j + 1) * H);
        const rv = this.attnV!.p.subarray(j * H, (j + 1) * H);
        for (let i = 0; i < H; i++) {
          ak += rk[i] * hs[i];
          av += rv[i] * hs[i];
        }
        k[j] = ak;
        v[j] = av;
      }
      K.push(k);
      V.push(v);
      let sc = 0;
      for (let i = 0; i < H; i++) sc += q[i] * k[i];
      scores[si] = sc / sqrtH;
    }
    if (this.fixedAttnBias) {
      for (let si = 0; si < L; si++) scores[si] = -2.0 * (L - 1 - si);
    }
    let max = -Infinity;
    for (let si = 0; si < L; si++) if (scores[si] > max) max = scores[si];
    let sum = 0;
    const w = new Float64Array(L);
    for (let si = 0; si < L; si++) {
      w[si] = Math.exp(scores[si] - max);
      sum += w[si];
    }
    for (let si = 0; si < L; si++) w[si] /= sum;
    const context = new Float64Array(H);
    for (let si = 0; si < L; si++) {
      for (let i = 0; i < H; i++) context[i] += w[si] * V[si][i];
    }
    // 门控(Mamba/Titans 式选择): 当前状态 h1 决定"是否把历史事件级语义调制进 h2 演化"
    let gateAcc = this.attnGB!.p[0];
    for (let i = 0; i < H; i++) gateAcc += this.attnGW!.p[i] * qSrc[i];
    const gate = sigmoid(gateAcc);
    this.lastAttnGate = gate;
    if (this.debugCollectGates) this.debugCollectGates.push(gate);
    const attnOut = new Float64Array(H);
    for (let i = 0; i < H; i++) attnOut[i] = gate * context[i];
    cache.K = K;
    cache.V = V;
    cache.Hn = histN;
    cache.w = w;
    cache.q = q;
    cache.gate = gate;
    cache.gatePre = gateAcc;
    cache.qSrc = qSrc;
    cache.h1Raw = new Float64Array(h1Cur);
    if (this.debugCollectAttn) this.debugCollectAttn.push({ w });
    return attnOut;
  }

  /** RMS 归一化反向雅可比: out = v / (sqrt(mean(v^2)) + eps), 累加到 acc */
  private static normBackward(v: Float64Array, dOut: Float64Array, acc: Float64Array): void {
    const n = v.length;
    let sq = 0;
    for (let i = 0; i < n; i++) sq += v[i] * v[i];
    const r = Math.max(Math.sqrt(sq / n), 1e-8);
    const rms = r + 1e-8;
    let dot = 0;
    for (let i = 0; i < n; i++) dot += dOut[i] * v[i];
    const coeff = dot / (r * rms * rms * n);
    for (let i = 0; i < n; i++) acc[i] += dOut[i] / rms - coeff * v[i];
  }

  private attnBackward(dAttnOut: Float64Array<ArrayBufferLike>, cache: AttnCache, dH1Cur: Float64Array<ArrayBufferLike>, dHistN: Float64Array[]): void {
    const H = this.cfg.hidden;
    const { K, V, Hn, w, q, gate, qSrc, h1Raw } = cache;
    const L = K.length;
    const sqrtH = Math.sqrt(H);
    const context = new Float64Array(H);
    for (let si = 0; si < L; si++) {
      for (let i = 0; i < H; i++) context[i] += w[si] * V[si][i];
    }
    const dCtx = new Float64Array(H);
    for (let i = 0; i < H; i++) dCtx[i] = gate * dAttnOut[i];
    let mean = 0;
    for (let si = 0; si < L; si++) {
      let acc = 0;
      for (let i = 0; i < H; i++) acc += dCtx[i] * V[si][i];
      mean += w[si] * acc;
    }
    const dScores = new Float64Array(L);
    for (let si = 0; si < L; si++) {
      let acc = 0;
      for (let i = 0; i < H; i++) acc += dCtx[i] * V[si][i];
      dScores[si] = w[si] * (acc - mean);
    }
    const dQ = new Float64Array(H);
    for (let si = 0; si < L; si++) {
      const hs = K[si];
      const hsIn = Hn[si];
      const rkp = this.attnK!.p;
      const rvp = this.attnV!.p;
      for (let j = 0; j < H; j++) {
        const dk = (dScores[si] * q[j]) / sqrtH;
        const dv = gate * w[si] * dAttnOut[j];
        const rkg = this.attnK!.g.subarray(j * H, (j + 1) * H);
        const rvg = this.attnV!.g.subarray(j * H, (j + 1) * H);
        for (let i = 0; i < H; i++) {
          rkg[i] += dk * hsIn[i];
          rvg[i] += dv * hsIn[i];
          dHistN[si][i] += dk * rkp[j * H + i] + dv * rvp[j * H + i];
        }
      }
      for (let i = 0; i < H; i++) dQ[i] += (dScores[si] * hs[i]) / sqrtH;
    }
    const qg = this.attnQ!.g;
    const qp = this.attnQ!.p;
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < H; i++) qg[j * H + i] += dQ[j] * qSrc[i];
    }
    const dQSrc = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < H; i++) dQSrc[i] += qp[j * H + i] * dQ[j];
    }
    let dGate = 0;
    for (let i = 0; i < H; i++) dGate += dAttnOut[i] * context[i];
    const dGatePre = dGate * gate * (1 - gate);
    for (let i = 0; i < H; i++) this.attnGW!.g[i] += dGatePre * qSrc[i];
    this.attnGB!.g[0] += dGatePre;
    for (let i = 0; i < H; i++) dQSrc[i] += dGatePre * this.attnGW!.p[i];
    CharGRU.normBackward(h1Raw, dQSrc, dH1Cur);
  }

  /** CNN 神经前向: 1D 卷积(核 3)→ 均值池化 → 独立预测头,返回 logitsC */
  private cnnForward(ids: number[]): { logits: Float64Array<ArrayBuffer>; cache: CnnCache } {
    const { emb: E, hidden: H, vocabSize: V } = this.cfg;
    const K = 3;
    const n = ids.length;
    const embWin: Array<Float64Array<ArrayBuffer>> = [];
    for (const id of ids) {
      const row = new Float64Array(E);
      this.embRow(id, row);
      embWin.push(row);
    }
    const P = Math.max(0, n - K + 1);
    const pre: Array<Float64Array<ArrayBuffer>> = [];
    const out: Array<Float64Array<ArrayBuffer>> = [];
    for (let p = 0; p < P; p++) {
      const pr = new Float64Array(H);
      const o = new Float64Array(H);
      for (let h = 0; h < H; h++) {
        let acc = this.cnnB!.p[h];
        for (let j = 0; j < K; j++) {
          const base = j * E * H + h;
          const ew = embWin[p + j];
          for (let e = 0; e < E; e++) acc += this.cnnW!.p[base + e * H] * ew[e];
        }
        pr[h] = acc;
        o[h] = tanh(acc);
      }
      pre.push(pr);
      out.push(o);
    }
    const feat = new Float64Array(H);
    if (P > 0) {
      for (let h = 0; h < H; h++) {
        let acc = 0;
        for (let p = 0; p < P; p++) acc += out[p][h];
        feat[h] = acc / P;
      }
    }
    let act = 0;
    for (let h = 0; h < H; h++) act += Math.abs(feat[h]);
    this.lastCnnAct = act / H;
    const logits = new Float64Array(V);
    for (let k = 0; k < V; k++) {
      let acc = this.bC!.p[k];
      const row = this.outC!.p.subarray(k * H, (k + 1) * H);
      for (let h = 0; h < H; h++) acc += row[h] * feat[h];
      logits[k] = acc;
    }
    return { logits, cache: { ids: [...ids], embWin, P, pre, out, feat } };
  }

  /** CNN 神经反向: 更新卷积核/池化/预测头参数并回传嵌入梯度 */
  private cnnBackward(dLogitsC: Float64Array<ArrayBuffer>, cache: CnnCache): void {
    const { emb: E, hidden: H, vocabSize: V } = this.cfg;
    const K = 3;
    const { ids, embWin, P, pre, out, feat } = cache;
    const dFeat = new Float64Array(H);
    for (let k = 0; k < V; k++) {
      const row = k * H;
      for (let h = 0; h < H; h++) {
        this.outC!.g[row + h] += dLogitsC[k] * feat[h];
        dFeat[h] += dLogitsC[k] * this.outC!.p[row + h];
      }
      this.bC!.g[k] += dLogitsC[k];
    }
    if (P === 0) return;
    const dEmbWin: Array<Float64Array<ArrayBuffer>> = embWin.map(() => new Float64Array(E));
    for (let p = 0; p < P; p++) {
      for (let h = 0; h < H; h++) {
        const dp = (dFeat[h] / P) * (1 - out[p][h] * out[p][h]);
        this.cnnB!.g[h] += dp;
        for (let j = 0; j < K; j++) {
          const base = j * E * H + h;
          const ew = embWin[p + j];
          const dw = this.cnnW!.g;
          for (let e = 0; e < E; e++) {
            dw[base + e * H] += dp * ew[e];
            dEmbWin[p + j][e] += dp * this.cnnW!.p[base + e * H];
          }
        }
      }
    }
    for (let i = 0; i < ids.length; i++) {
      const rowG = this.emb.g.subarray((ids[i] % V) * E, ((ids[i] % V) + 1) * E);
      for (let e = 0; e < E; e++) rowG[e] += dEmbWin[i][e];
    }
  }

  /** 训练一批序列: 截断 BPTT, 返回 { loss, totalChars } */
  trainStepBatch(seqs: Array<{ ids: number[] }>, lr: number): { loss: number; totalChars: number } {
    const { hidden: H, emb: E, vocabSize: V, bptt } = this.cfg;
    this.t += 1;
    for (const g of this.groups) g.zeroGrad();
    let totalLoss = 0;
    let totalChars = 0;
    for (const seq of seqs) {
      const L = seq.ids.length;
      const T = L - 1;
      const h1t: Float64Array[] = [];
      const z1t: Float64Array[] = [];
      const r1t: Float64Array[] = [];
      const c1t: Float64Array[] = [];
      const h2t: Array<Float64Array<ArrayBufferLike>> = [];
      /** 每步单元输出(GRU: h2; SSM: y), 供读头/attention hist 使用 */
      const readt: Array<Float64Array<ArrayBufferLike>> = [];
      const z2t: Float64Array[] = [];
      const r2t: Float64Array[] = [];
      const c2t: Float64Array[] = [];
      const ssmCaches: (SSMCache | null)[] = [];
      const h1Prev = new Float64Array(H);
      const h2Prev: Float64Array<ArrayBufferLike> = this.ssm ? this.ssm.newState() : new Float64Array(H);
      const x = new Float64Array(E);
      const logitsList: Float64Array[] = [];
      const x2t: Float64Array[] = [];
      const attnCaches: (AttnCache | null)[] = [];
      const cnnCaches: (CnnCache | null)[] = [];
      /** 脉冲近似门控(前向用阈值化, 反传用原始梯度) */
      const spikedZ1t: Float64Array[] = [];
      const spikedR1t: Float64Array[] = [];
      /** 预测头缓存(h1→pred_h2) */
      const predCaches: Array<{ h1: Float64Array; predH2: Float64Array; error: Float64Array }> = [];
      for (let t = 0; t < T; t++) {
        this.embRow(seq.ids[t], x);
        const z1 = new Float64Array(H);
        const r1 = new Float64Array(H);
        const c1 = new Float64Array(H);
        const h1 = new Float64Array(H);
        this.cell1.forward(x, h1Prev, { z: z1, r: r1, c: c1, h: h1 });
        // 脉冲近似: 阈值门控(前向 spike, 反传用原始 sigmoid 梯度)
        const spikedZ1 = this.spikeThresh > 0 ? new Float64Array(H) : null;
        if (spikedZ1) {
          const spikedR1 = new Float64Array(H);
          for (let i = 0; i < H; i++) {
            spikedZ1[i] = z1[i] > this.spikeThresh ? 1.0 : 0.0;
            spikedR1[i] = r1[i] > this.spikeThresh ? 1.0 : 0.0;
          }
          h1t.push(h1);
          z1t.push(spikedZ1);
          r1t.push(spikedR1);
          c1t.push(c1);
        } else {
          h1t.push(h1);
          z1t.push(z1);
          r1t.push(r1);
          c1t.push(c1);
        }
        const z2 = new Float64Array(H);
        const r2 = new Float64Array(H);
        const c2 = new Float64Array(H);
        const h2 = new Float64Array(H);
        let x2 = h1;
        if (this.attnQ) {
          // 与推理 step() 一致: hist 长度 min(t, ctx), 含最近一个单元输出
          const hist = readt.slice(Math.max(0, t - this.cfg.ctx), t);
          const cache: AttnCache = { K: [], V: [], Hn: [], w: new Float64Array(0), q: new Float64Array(0), gate: 0, gatePre: 0, qSrc: new Float64Array(0), h1Raw: new Float64Array(0) };
          const attnOut = this.attnForward(h1, hist, cache);
          attnCaches.push(cache);
          x2 = new Float64Array(H);
          for (let i = 0; i < H; i++) x2[i] = h1[i] + attnOut[i];
        } else {
          attnCaches.push(null);
        }
        if (this.ssm) {
          const sc = this.ssm.newCache();
          this.ssm.forward(x2, h2Prev, sc);
          ssmCaches.push(sc);
          x2t.push(x2);
          h2t.push(sc.h as Float64Array<ArrayBuffer>);
          readt.push(sc.y as Float64Array<ArrayBuffer>);
          h2Prev.set(sc.h);
        } else {
          this.cell2.forward(x2, h2Prev, { z: z2, r: r2, c: c2, h: h2 });
          ssmCaches.push(null);
          x2t.push(x2);
          h2t.push(h2);
          readt.push(h2);
          z2t.push(z2);
          r2t.push(r2);
          c2t.push(c2);
          h2Prev.set(h2);
        }
        // 预测头: h1 → pred_h2, 预测误差作为内在动机信号
        if (this.predW) {
          const predH2 = new Float64Array(H);
          for (let j = 0; j < H; j++) {
            let acc = this.predB!.p[j];
            const row = this.predW!.p.subarray(j * H, (j + 1) * H);
            for (let i = 0; i < H; i++) acc += row[i] * h1[i];
            predH2[j] = acc;
          }
          const error = new Float64Array(H);
          for (let i = 0; i < H; i++) error[i] = predH2[i] - (h2 as Float64Array)[i];
          const predLoss = 0.5 * Array.from(error).reduce((s, v) => s + v * v, 0);
          totalLoss += (this.cfg.predLossWeight ?? 0.01) * predLoss;
          predCaches.push({ h1: h1.slice() as Float64Array, predH2, error });
        }
        h1Prev.set(h1);
        let cnnLogits: Float64Array | null = null;
        if (this.cnnW) {
          const idsWin = seq.ids.slice(Math.max(0, t - this.cfg.ctx + 1), t + 1);
          const cnn = this.cnnForward(idsWin);
          cnnLogits = cnn.logits;
          cnnCaches.push(cnn.cache);
        } else {
          cnnCaches.push(null);
        }
        const logits = new Float64Array(V);
        for (let k = 0; k < V; k++) {
          let acc = this.bOut.p[k];
          const row = this.out.p.subarray(k * H, (k + 1) * H);
          const read = readt[readt.length - 1];
          for (let j = 0; j < H; j++) acc += row[j] * read[j];
          if (cnnLogits) acc += cnnLogits[k];
          // Logit clamp: 防止多路径叠加(log+CNN+预测头)导致数值溢出
          logits[k] = Math.max(-100, Math.min(100, acc));
        }
        logitsList.push(logits);
        const y = seq.ids[t + 1];
        let max = -Infinity;
        for (let k = 0; k < V; k++) if (logits[k] > max) max = logits[k];
        let sum = 0;
        const pr = new Float64Array(V);
        for (let k = 0; k < V; k++) {
          pr[k] = Math.exp(logits[k] - max);
          sum += pr[k];
        }
        for (let k = 0; k < V; k++) pr[k] /= sum;
        totalLoss -= Math.log(pr[y] + 1e-12);
        totalChars += 1;
      }
      // BPTT 反向
      const dH2Next = new Float64Array(H);
      const dH1Next = new Float64Array(H);
      const dx2 = new Float64Array(H);
      const dx1 = new Float64Array(E);
      const dH2Prev = new Float64Array(H);
      const dH1Prev = new Float64Array(H);
      const sN = this.ssm ? this.ssm.cfg.dState : 0;
      const dSSMNext = this.ssm ? new Float64Array(H * sN) : null;
      const dSSMPrev = this.ssm ? new Float64Array(H * sN) : null;
      // 未来 attention 通过 hist(K/V 输入)对早期单元输出的梯度累积
      const dHist2: Float64Array[] = [];
      for (let i = 0; i < T; i++) dHist2.push(new Float64Array(H));
      for (let t = T - 1; t >= 0; t--) {
        const logits = logitsList[t];
        const y = seq.ids[t + 1];
        const dLogits = new Float64Array(V);
        let max = -Infinity;
        for (let k = 0; k < V; k++) if (logits[k] > max) max = logits[k];
        let sum = 0;
        for (let k = 0; k < V; k++) sum += Math.exp(logits[k] - max);
        for (let k = 0; k < V; k++) {
          dLogits[k] = Math.exp(logits[k] - max) / sum;
        }
        dLogits[y] -= 1;
        if (this.ssm) {
          // SSM: dY(读头+未来 hist) 与 dHState(循环) 分离
          const dY = dH2Next;
          for (let i = 0; i < H; i++) dY[i] = 0;
          for (let k = 0; k < V; k++) {
            const row = k * H;
            for (let j = 0; j < H; j++) {
              this.out.g[row + j] += dLogits[k] * readt[t][j];
              dY[j] += dLogits[k] * this.out.p[row + j];
            }
            this.bOut.g[k] += dLogits[k];
          }
          if (this.cnnW) {
            const cache = cnnCaches[t];
            if (cache) this.cnnBackward(dLogits, cache);
          }
          const dHist = dHist2[t];
          for (let i = 0; i < H; i++) dY[i] += dHist[i];
          const h2prevState = t > 0 ? h2t[t - 1] : this.ssm.newState();
          dx2.fill(0);
          dSSMPrev!.fill(0);
          this.ssm.backward(dY, dSSMNext!, x2t[t], h2prevState, ssmCaches[t]!, dx2, dSSMPrev!);
          dSSMNext!.set(dSSMPrev!);
        } else {
          const dRead = dH2Next;
          for (let k = 0; k < V; k++) {
            const row = k * H;
            for (let j = 0; j < H; j++) {
              this.out.g[row + j] += dLogits[k] * readt[t][j];
              dRead[j] += dLogits[k] * this.out.p[row + j];
            }
            this.bOut.g[k] += dLogits[k];
          }
          if (this.cnnW) {
            const cache = cnnCaches[t];
            if (cache) this.cnnBackward(dLogits, cache);
          }
          // 未来 attention 把本时刻单元输出当 K/V 输入,梯度在此汇入
          const dHist = dHist2[t];
          for (let i = 0; i < H; i++) dH2Next[i] += dHist[i];
          const h2prev = t > 0 ? h2t[t - 1] : new Float64Array(H);
          dx2.fill(0);
          dH2Prev.fill(0);
          this.cell2.backward(dH2Next, x2t[t], h2prev, z2t[t], r2t[t], c2t[t], h2t[t], dx2, dH2Prev);
          dH2Next.set(dH2Prev);
        }
        // x2 = h1 + attnOut: dx2 拆分为 attention 输入梯度与 h1 残差梯度
        const dH1 = new Float64Array(H);
        for (let i = 0; i < H; i++) dH1[i] = dx2[i];
        if (this.attnQ) {
          const cache = attnCaches[t];
          if (cache) {
            const L = cache.K.length;
            const dHistN: Float64Array[] = [];
            for (let si = 0; si < L; si++) dHistN.push(new Float64Array(H));
            this.attnBackward(dx2, cache, dH1, dHistN);
            const t0 = t - L;
            for (let si = 0; si < L; si++) {
              CharGRU.normBackward(readt[t0 + si], dHistN[si], dHist2[t0 + si]);
            }
          }
        }
        // 预测头反向: pred_h2 = W_pred·h1 + b_pred, 梯度回传到 h1
        if (this.predW && predCaches[t]) {
          const pc = predCaches[t];
          const pw = this.predW;
          const plw = this.cfg.predLossWeight ?? 0.01;
          // d(pred_h2) = error (MSE梯度 = pred_h2 - h2)
          for (let j = 0; j < H; j++) {
            this.predB!.g[j] += plw * pc.error[j];
            const row = pw.g.subarray(j * H, (j + 1) * H);
            for (let i = 0; i < H; i++) {
              row[i] += plw * pc.error[j] * pc.h1[i];
              dH1[i] += plw * pc.error[j] * pw.p[j * H + i];
            }
          }
        }
        const h1prev = t > 0 ? h1t[t - 1] : new Float64Array(H);
        const x = new Float64Array(E);
        this.embRow(seq.ids[t], x);
        dx1.fill(0);
        dH1Prev.fill(0);
        // dH1Next 顶部已含 h1 循环梯度(来自 t+1),此处必须累加而非覆盖
        for (let j = 0; j < H; j++) dH1Next[j] += dH1[j];
        this.cell1.backward(dH1Next, x, h1prev, z1t[t], r1t[t], c1t[t], h1t[t], dx1, dH1Prev);
        for (let e = 0; e < E; e++) this.emb.g[(seq.ids[t] % V) * E + e] += dx1[e];
        dH1Next.set(dH1Prev);
      }
      // 记录本批次的 h1 历史供 STDP 使用（z1t = 遗忘门激活 ≈ 突触前信号）
      for (let si = 0; si < h1t.length; si++) {
        this._stdpH1Hist.push(h1t[si]);
        this._stdpPreHist.push(z1t[si]);
      }
      // STDA: 记录最后一个时间步的 h2
      if (h2t.length > 0) {
        this._stdaLastH2 = Array.isArray(h2t[h2t.length - 1]) ? (h2t[h2t.length - 1] as Float64Array) : h2t[h2t.length - 1];
      }
    }
    for (const g of this.groups) {
      for (let i = 0; i < g.g.length; i++) g.g[i] /= totalChars;
    }
    // 全局梯度范数裁剪(兜底,防 BPTT/注意力梯度爆炸)
    let norm2 = 0;
    for (const g of this.groups) {
      for (let i = 0; i < g.g.length; i++) norm2 += g.g[i] * g.g[i];
    }
    const norm = Math.sqrt(norm2);
    // PyTorch clip_grad_norm_ 文本模型标准: maxNorm=1.0
    // 参考: HuggingFace Transformers max_grad_norm=1.0, Google BERT clip_norm=1.0
    const maxNorm = 1.0;
    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      for (const g of this.groups) {
        for (let i = 0; i < g.g.length; i++) g.g[i] *= scale;
      }
    }
    for (const g of this.groups) {
      g.adam(lr, this.t);
    }
    return { loss: totalLoss / totalChars, totalChars };
  }

  embedAvg(ids: number[]): number[] {
    const { vocabSize: v, emb } = this.cfg;
    const out = new Float64Array(emb);
    let count = 0;
    for (const id of ids) {
      const row = this.emb.p.subarray((id % v) * emb, ((id % v) + 1) * emb);
      for (let e = 0; e < emb; e++) out[e] += row[e];
      count += 1;
    }
    if (count > 0) for (let e = 0; e < emb; e++) out[e] /= count;
    return Array.from(out);
  }

  logitsToProbs(logits: Float64Array): Float64Array {
    let max = -Infinity;
    for (let k = 0; k < logits.length; k++) if (logits[k] > max) max = logits[k];
    let sum = 0;
    const pr = new Float64Array(logits.length);
    for (let k = 0; k < logits.length; k++) {
      pr[k] = Math.exp(logits[k] - max);
      sum += pr[k];
    }
    for (let k = 0; k < pr.length; k++) pr[k] /= sum;
    return pr;
  }

  confidence(logits: Float64Array): number {
    const pr = this.logitsToProbs(logits);
    let h = 0;
    for (let k = 0; k < pr.length; k++) if (pr[k] > 0) h -= pr[k] * Math.log(pr[k]);
    const maxH = Math.log(pr.length);
    return Math.max(0, Math.min(1, 1 - h / maxH));
  }

  save(): number[] {
    const out: number[] = [];
    for (const g of this.groups) for (let i = 0; i < g.p.length; i++) out.push(g.p[i]);
    return out;
  }

  load(params: number[]): void {
    let off = 0;
    for (const g of this.groups) {
      for (let i = 0; i < g.p.length; i++) g.p[i] = params[off + i];
      off += g.p.length;
    }
  }

  /**
   * 单步反向: 给定当前timestep的dLogits(来自外部loss)和前后state快照,
   * 计算emb/cell1/cell2/ssm/attn/cnn的梯度并累加到各组g上。
   * 供CharMultiNeuro等外部模块使用, 不修改self.state。
   */
  backPropStep(
    xId: number,
    statePrev: GRUState,
    stateCur: GRUState,
    dLogits: Float64Array,
    depth: 4,
  ): void {
    const { hidden: H, emb: E, vocabSize: V, ctx, attn, mamba, cnn } = this.cfg;
    // 从stateCur反推中间变量(需要重跑forward来记录)
    // 由于forward是side-effect free的(传副本), 我们重跑一次记录cache
    const x = new Float64Array(E);
    this.embRow(xId, x);

    // Cell1 forward (记录z1,r1,c1,h1)
    const z1 = new Float64Array(H);
    const r1 = new Float64Array(H);
    const c1 = new Float64Array(H);
    const h1 = new Float64Array(H);
    this.cell1.forward(x, statePrev.h1, { z: z1, r: r1, c: c1, h: h1 });

    // Attn forward (if depth>=3)
    let attnCache: AttnCache | null = null;
    let x2 = h1;
    if (attn && depth >= 3) {
      const hist = stateCur.hist.slice(Math.max(0, stateCur.hist.length - ctx), stateCur.hist.length);
      attnCache = { K: [], V: [], Hn: [], w: new Float64Array(0), q: new Float64Array(0), gate: 0, gatePre: 0, qSrc: new Float64Array(0), h1Raw: new Float64Array(0) };
      const attnOut = this.attnForward(h1, hist, attnCache);
      x2 = new Float64Array(H);
      for (let i = 0; i < H; i++) x2[i] = h1[i] + attnOut[i];
    }

    // Cell2/SSM forward (depth>=2)
    const z2 = new Float64Array(H);
    const r2 = new Float64Array(H);
    const c2 = new Float64Array(H);
    const h2 = new Float64Array(H);
    let read: Float64Array<ArrayBufferLike> = h2;
    if (mamba) {
      const sc = this.ssm!.newCache();
      this.ssm!.forward(x2, statePrev.h2, sc);
      read = sc.y;
    } else {
      this.cell2.forward(x2, statePrev.h2, { z: z2, r: r2, c: c2, h: h2 });
      read = h2;
    }

    // CNN forward (depth>=4)
    let cnnCache: CnnCache | null = null;
    if (cnn && depth >= 4) {
      const idsWin = stateCur.ids.slice(-ctx);
      cnnCache = this.cnnForward(idsWin).cache;
    }

    // ── 反向 ──
    // 1) 输出头梯度 → dRead
    const dRead = new Float64Array(H);
    for (let k = 0; k < V; k++) {
      const row = k * H;
      for (let j = 0; j < H; j++) {
        this.out.g[row + j] += dLogits[k] * (read as Float64Array)[j];
        dRead[j] += dLogits[k] * this.out.p[row + j];
      }
      this.bOut.g[k] += dLogits[k];
    }
    // CNN头梯度
    if (cnnCache && this.cnnW) {
      // CNN独立预测头, dLogits已包含CNN贡献(在forward时叠加)
      // 这里dRead不含CNN部分, 因为CNN有自己独立的outC/bC
      // 需要单独传dLogits给cnnBackward
      // 但dLogits是MoE组合的, 不是纯CNN的 → 跳过CNN头反向(由expert head负责)
    }

    // 2) SSM/Cell2反向 → dx2, dH1
    const dx2 = new Float64Array(H);
    const dH1FromDeep = new Float64Array(H);
    if (mamba) {
      // SSM 单步反向: 需要前向 cache，当前 backPropStep 不支持 SSM 路径。
      // 梯度已通过 cell1 + 输出头累加，SSM 参数仅通过 trainStepBatch (完整 BPTT) 更新。
      // 此处 dx2 视为 0，dH1FromDeep 直接继承 dRead（近似忽略 SSM 循环）。
      for (let j = 0; j < H; j++) dH1FromDeep[j] = dRead[j];
    } else {
      const h2prev = statePrev.h2;
      this.cell2.backward(dRead, x2, h2prev, z2, r2, c2, h2, dx2, dH1FromDeep);
    }

    // 3) Attn反向 → dH1
    const dH1Total = new Float64Array(H);
    for (let j = 0; j < H; j++) dH1Total[j] = dH1FromDeep[j];
    if (attnCache && this.attnQ) {
      const dHistN: Float64Array[] = [];
      for (let si = 0; si < attnCache.K.length; si++) dHistN.push(new Float64Array(H));
      this.attnBackward(dx2, attnCache, dH1Total, dHistN);
      // dHistN梯度汇入早期timestep(暂忽略, 因单步反向不追踪跨步attention)
    }

    // 4) Cell1反向 → dx, dH1Prev
    const dx1 = new Float64Array(E);
    const dH1Prev = new Float64Array(H);
    this.cell1.backward(dH1Total, x, statePrev.h1, z1, r1, c1, h1, dx1, dH1Prev);

    // 5) Emb梯度
    for (let e = 0; e < E; e++) this.emb.g[(xId % V) * E + e] += dx1[e];
  }
}

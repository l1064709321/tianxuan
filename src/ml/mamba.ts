import { Group } from "./model";

export interface SSMConfig {
  /** 通道数(= 隐层) */
  input: number;
  /** 每通道状态维 */
  dState: number;
}

export interface SSMCache {
  pre: Float64Array<ArrayBufferLike>;
  delta: Float64Array<ArrayBufferLike>;
  b: Float64Array<ArrayBufferLike>;
  c: Float64Array<ArrayBufferLike>;
  aexp: Float64Array<ArrayBufferLike>;
  h: Float64Array<ArrayBufferLike>;
  /** 单元输出 y[d] = Σ_n c[n]·h[d,n] */
  y: Float64Array<ArrayBufferLike>;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function softplus(x: number): number {
  if (x > 30) return x;
  if (x < -30) return Math.exp(x);
  return Math.log1p(Math.exp(x));
}

/**
 * 选择性状态空间模型(S6/Mamba 逻辑自研实现, 不抄代码):
 *   Δ_t = softplus(Wd·x_t + bd)          (D, 输入决定遗忘速率 = 选择性)
 *   B_t = Wb·x_t (N), C_t = Wc·x_t (N)   (输入依赖的读写投影)
 *   h_t[d,n] = exp(Δ_t[d]·A[n])·h_{t-1}[d,n] + Δ_t[d]·B_t[n]·x_t[d]
 *   y_t[d] = Σ_n C_t[n]·h_t[d,n]
 * 快动力学: token 级线性状态流, 与 Attention(慢语义)互补。
 */
export class SelectiveSSM {
  readonly cfg: SSMConfig;
  readonly wd: Group;
  readonly bd: Group;
  readonly wb: Group;
  readonly wc: Group;
  readonly a: Group;
  private readonly groups: Group[];
  /** 协同探针: 最近一步 Δ 均值(选择遗忘强度) */
  lastDeltaMean = 0;

  constructor(cfg: SSMConfig, init: (i: number) => number) {
    this.cfg = cfg;
    const D = cfg.input;
    const N = cfg.dState;
    this.wd = new Group(D * D, init);
    this.bd = new Group(D, () => 0);
    this.wb = new Group(D * N, () => init(1) * 0.1);
    this.wc = new Group(D * N, () => init(1) * 0.1);
    this.a = new Group(N, (i) => -0.1 - Math.abs(init(i)) * 0.9);
    this.groups = [this.wd, this.bd, this.wb, this.wc, this.a];
  }

  groupsAll(): Group[] {
    return this.groups;
  }

  /** 单步前向: x(D) + hPrev(D*N) → h, 缓存供反向 */
  forward(x: Float64Array<ArrayBufferLike>, hPrev: Float64Array<ArrayBufferLike>, cache: SSMCache): void {
    const { input: D, dState: N } = this.cfg;
    const pre = cache.pre;
    const delta = cache.delta;
    const b = cache.b;
    const c = cache.c;
    const aexp = cache.aexp;
    const h = cache.h;
    for (let d = 0; d < D; d++) {
      let acc = this.bd.p[d];
      const row = this.wd.p.subarray(d * D, (d + 1) * D);
      for (let i = 0; i < D; i++) acc += row[i] * x[i];
      pre[d] = acc;
      delta[d] = softplus(acc) + 1e-6;
    }
    for (let n = 0; n < N; n++) {
      const a = Math.min(this.a.p[n], 0);
      let bb = 0;
      let cc = 0;
      const rb = this.wb.p.subarray(n * D, (n + 1) * D);
      const rc = this.wc.p.subarray(n * D, (n + 1) * D);
      for (let i = 0; i < D; i++) {
        bb += rb[i] * x[i];
        cc += rc[i] * x[i];
      }
      b[n] = bb;
      c[n] = cc;
      for (let d = 0; d < D; d++) {
        aexp[d * N + n] = Math.exp(delta[d] * a);
        h[d * N + n] = aexp[d * N + n] * hPrev[d * N + n] + delta[d] * bb * x[d];
      }
    }
    for (let d = 0; d < D; d++) {
      let acc = 0;
      for (let n = 0; n < N; n++) acc += c[n] * h[d * N + n];
      cache.y[d] = acc;
    }
    let dm = 0;
    for (let d = 0; d < D; d++) dm += delta[d];
    this.lastDeltaMean = dm / D;
  }

  /**
   * 单步反向: dY(∂L/∂输出 y_t) + dHState(∂L/∂状态 h_t, 来自循环下游)
   * → dx, dHPrev(∂L/∂h_{t-1}), 参数梯度就地累加。
   */
  backward(dY: Float64Array<ArrayBufferLike>, dHState: Float64Array<ArrayBufferLike>, x: Float64Array<ArrayBufferLike>, hPrev: Float64Array<ArrayBufferLike>, cache: SSMCache, dx: Float64Array<ArrayBufferLike>, dHPrev: Float64Array<ArrayBufferLike>): void {
    const { input: D, dState: N } = this.cfg;
    const { pre, delta, b, c, aexp, h } = cache;
    // 总状态梯度: y[d]=Σ_n c[n]·h[d,n] → dH[d,n] = dY[d]·c[n] + dHState[d,n]
    const dH = new Float64Array(D * N);
    for (let d = 0; d < D; d++) {
      for (let n = 0; n < N; n++) dH[d * N + n] = dY[d] * c[n] + dHState[d * N + n];
    }
    const dPre = new Float64Array(D);
    const db = new Float64Array(N);
    const dc = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      for (let d = 0; d < D; d++) dc[n] += dY[d] * h[d * N + n];
    }
    // Δ 路径(选择性遗忘): Δ 影响 exp(Δ·A)·hPrev 与 Δ·B·x
    for (let d = 0; d < D; d++) {
      const sg = sigmoid(pre[d]);
      let dDelta = 0;
      for (let n = 0; n < N; n++) {
        const a = Math.min(this.a.p[n], 0);
        const dh = dH[d * N + n];
        dDelta += dh * (a * aexp[d * N + n] * hPrev[d * N + n] + b[n] * x[d]);
      }
      const dPreVal = dDelta * sg;
      dPre[d] = dPreVal;
      const rowG = this.wd.g.subarray(d * D, (d + 1) * D);
      for (let i = 0; i < D; i++) rowG[i] += dPreVal * x[i];
      this.bd.g[d] += dPreVal;
    }
    for (let n = 0; n < N; n++) {
      const a = Math.min(this.a.p[n], 0);
      let dA = 0;
      for (let d = 0; d < D; d++) {
        const dh = dH[d * N + n];
        dA += dh * delta[d] * aexp[d * N + n] * hPrev[d * N + n];
        db[n] += dh * delta[d] * x[d];
      }
      this.a.g[n] += dA * (this.a.p[n] < 0 ? 1 : 0);
      const rbG = this.wb.g.subarray(n * D, (n + 1) * D);
      const rcG = this.wc.g.subarray(n * D, (n + 1) * D);
      for (let i = 0; i < D; i++) {
        rbG[i] += db[n] * x[i];
        rcG[i] += dc[n] * x[i];
      }
    }
    // 状态回传: h_t → h_{t-1}(exp(ΔA)) 与 x(Δ·B 直连 + Wd/Wb/Wc 投影)
    dx.fill(0);
    for (let d = 0; d < D; d++) {
      for (let n = 0; n < N; n++) {
        const dh = dH[d * N + n];
        dHPrev[d * N + n] += aexp[d * N + n] * dh;
        dx[d] += dh * delta[d] * b[n];
      }
    }
    for (let n = 0; n < N; n++) {
      const rb = this.wb.p.subarray(n * D, (n + 1) * D);
      const rc = this.wc.p.subarray(n * D, (n + 1) * D);
      for (let i = 0; i < D; i++) {
        dx[i] += db[n] * rb[i] + dc[n] * rc[i];
      }
    }
    for (let d = 0; d < D; d++) {
      const row = this.wd.p.subarray(d * D, (d + 1) * D);
      for (let i = 0; i < D; i++) dx[i] += dPre[d] * row[i];
    }
  }

  newState(): Float64Array {
    return new Float64Array(this.cfg.input * this.cfg.dState);
  }

  newCache(): SSMCache {
    const { input: D, dState: N } = this.cfg;
    return { pre: new Float64Array(D), delta: new Float64Array(D), b: new Float64Array(N), c: new Float64Array(N), aexp: new Float64Array(D * N), h: new Float64Array(D * N), y: new Float64Array(D) };
  }
}

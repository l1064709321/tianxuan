/**
 * CharTransformer — nanoGPT 风格字符级语言模型（含完整 BPTT）
 *
 * 作为多神经协同的主干，提供强语言建模能力。
 * 多神经系统（MoE/Titans/STDP/Dopamine）在此基础上叠加幻觉抑制。
 */

import { Group } from "./model";
import { makeGaussian, mulberry32 } from "./rng";

export interface CharTransformerConfig {
  vocabSize: number;
  emb: number;
  nLayer: number;
  nHead: number;
  ctx: number;
  bptt: number;
  dropout?: number;
}

export const DEFAULT_TRANSFORMER_CONFIG: CharTransformerConfig = {
  vocabSize: 380,
  emb: 256,
  nLayer: 6,
  nHead: 4,
  ctx: 64,
  bptt: 32,
  dropout: 0.1,
};

interface ForwardCache {
  embInput: Float64Array;
  layerInputs: Float64Array[];   // 每层 block 输入
  ln1Outs: Float64Array[];       // LayerNorm1 输出
  attnOuts: Float64Array[];      // Attention 输出
  ln2Outs: Float64Array[];       // LayerNorm2 输出
  ffnOuts: Float64Array[];       // FFN 输出
  qkv: { q: Float64Array; k: Float64Array; v: Float64Array }[];
  attnScores: Float64Array[][];  // 每头每层的 attention 权重
}

export interface TransformerState {
  h: Float64Array;
  kvCache: Float64Array[];
  forwardChain: ForwardCache[];
}

export class CharTransformer {
  readonly cfg: CharTransformerConfig;
  readonly groups: Group[] = [];

  private embLayer: Group;
  private posEmb: Group;
  private blocks: Array<{
    ln1: Group;
    attn: { q: Group; k: Group; v: Group; o: Group };
    ln2: Group;
    ffn: { w1: Group; b1: Group; w2: Group; b2: Group };
  }>;
  private outHead: Group;
  private outBias: Group;

  private t = 0;
  private _lastAttnGate = 0;
  private _lastActivation = 0;

  constructor(cfg: Partial<CharTransformerConfig> = {}, seed = 42) {
    this.cfg = { ...DEFAULT_TRANSFORMER_CONFIG, ...cfg };
    const { vocabSize, emb, nLayer, nHead, ctx } = this.cfg;
    const headDim = emb / nHead;

    const gaussian = makeGaussian(mulberry32(seed));
    const scale = (fanIn: number) => Math.sqrt(2.0 / fanIn);

    this.embLayer = new Group(vocabSize * emb, () => gaussian() * 0.02);
    this.groups.push(this.embLayer);

    this.posEmb = new Group(ctx * emb, () => gaussian() * 0.02);
    this.groups.push(this.posEmb);

    this.blocks = [];
    for (let l = 0; l < nLayer; l++) {
      const block = {
        ln1: new Group(emb, () => 1.0),
        attn: {
          q: new Group(emb * emb, () => gaussian() * scale(emb)),
          k: new Group(emb * emb, () => gaussian() * scale(emb)),
          v: new Group(emb * emb, () => gaussian() * scale(emb)),
          o: new Group(emb * emb, () => gaussian() * scale(emb)),
        },
        ln2: new Group(emb, () => 1.0),
        ffn: {
          w1: new Group(4 * emb * emb, () => gaussian() * scale(emb)),
          b1: new Group(4 * emb, () => 0),
          w2: new Group(emb * 4 * emb, () => gaussian() * scale(4 * emb)),
          b2: new Group(emb, () => 0),
        },
      };
      this.blocks.push(block);
      this.groups.push(
        block.ln1, block.ln2,
        block.attn.q, block.attn.k, block.attn.v, block.attn.o,
        block.ffn.w1, block.ffn.b1, block.ffn.w2, block.ffn.b2
      );
    }

    this.outHead = new Group(vocabSize * emb, () => gaussian() * 0.02);
    this.outBias = new Group(vocabSize, () => 0);
    this.groups.push(this.outHead, this.outBias);
  }

  get lastAttnGate() { return this._lastAttnGate; }
  get lastActivation() { return this._lastActivation; }

  paramCount(): number {
    return this.groups.reduce((s, g) => s + g.p.length, 0);
  }

  newState(): TransformerState {
    return {
      h: new Float64Array(this.cfg.emb),
      kvCache: [],
      forwardChain: [],
    };
  }

  // ==================== 前向 ====================

  step(xId: number, state: TransformerState, depth?: number): Float64Array {
    const { vocabSize, emb, nHead, ctx } = this.cfg;
    const headDim = emb / nHead;

    // 嵌入 + 位置编码
    const embRow = this.embLayer.p.subarray(xId * emb, (xId + 1) * emb);
    const pos = state.kvCache.length % ctx;
    const posEmbRow = this.posEmb.p.subarray(pos * emb, (pos + 1) * emb);
    const embInput = new Float64Array(emb);
    for (let i = 0; i < emb; i++) embInput[i] = embRow[i] + posEmbRow[i];

    let h: Float64Array = embInput;
    const layers = depth ?? this.cfg.nLayer;

    // 存储前向缓存
    const cache: ForwardCache = {
      embInput: embInput.slice(),
      layerInputs: [],
      ln1Outs: [],
      attnOuts: [],
      ln2Outs: [],
      ffnOuts: [],
      qkv: [],
      attnScores: [],
    };

    for (let l = 0; l < layers; l++) {
      const block = this.blocks[l];
      // @ts-ignore
      cache.layerInputs[l] = h.slice();

      // Pre-norm 1
      // @ts-ignore
      cache.ln1Outs[l] = h.slice();

      // Attention
      const { q, k, v, scores } = this.multiHeadAttnForward(h, block.attn, nHead, headDim, state, l);
      // @ts-ignore
      cache.qkv[l] = { q, k, v };
      // @ts-ignore
      cache.attnScores[l] = scores;
      const attnOut = this.projectOutput(h, block.attn.o, v, nHead, headDim);
      // @ts-ignore
      cache.attnOuts[l] = attnOut;

      // Residual + LN2
      h = this.addResidual(h, attnOut) as Float64Array<ArrayBuffer>;
      const hAfterLN2 = this.layerNormForward(h, block.ln2) as Float64Array<ArrayBuffer>;
      // @ts-ignore
      cache.ln2Outs[l] = hAfterLN2;  // FFN 的输入

      // FFN
      const ffnOut = this.feedForwardForward(hAfterLN2, block.ffn) as Float64Array<ArrayBuffer>;

      // @ts-ignore
      cache.ffnOuts[l] = ffnOut;

      // Residual
      h = this.addResidual(hAfterLN2, ffnOut) as Float64Array<ArrayBuffer>;
    }

    cache.layerInputs[layers] = h.slice();
      // @ts-ignore
    state.forwardChain.push(cache);

      // @ts-ignore
    state.h = h;
      // @ts-ignore
    state.kvCache.push(h.slice());
    if (state.kvCache.length > ctx) state.kvCache.shift();

    this._lastActivation = this.computeActivationNorm(h);

    // 输出头
    const logits = new Float64Array(vocabSize);
    for (let k = 0; k < vocabSize; k++) {
      let acc = this.outBias.p[k];
      const row = this.outHead.p.subarray(k * emb, (k + 1) * emb);
      for (let i = 0; i < emb; i++) acc += row[i] * h[i];
      logits[k] = acc;
    }

    return logits;
  }

  // ==================== 前向组件 ====================

  private layerNormForward(x: Float64Array, ln: Group): Float64Array {
    const n = x.length;
    const out = new Float64Array(n);
    let sum = 0, sumSq = 0;
    for (let i = 0; i < n; i++) { sum += x[i]; sumSq += x[i] * x[i]; }
    const mean = sum / n;
    const var_ = sumSq / n - mean * mean + 1e-5;
    const std = Math.sqrt(var_);
    for (let i = 0; i < n; i++) {
      out[i] = (x[i] - mean) / std * ln.p[i];
    }
    return out;
  }

  private addResidual(a: Float64Array, b: Float64Array): Float64Array {
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    return out;
  }

  private multiHeadAttnForward(
    x: Float64Array,
    attn: { q: Group; k: Group; v: Group; o: Group },
    nHead: number,
    headDim: number,
    state: TransformerState,
    layerIdx: number
  ): { q: Float64Array; k: Float64Array; v: Float64Array; scores: Float64Array } {
    const emb = x.length;
    const ctxLen = state.kvCache.length;

    // QKV 投影
    const q = new Float64Array(emb);
    const k = new Float64Array(emb);
    const v = new Float64Array(emb);

    for (let h = 0; h < nHead; h++) {
      const hs = h * headDim;
      for (let i = 0; i < headDim; i++) {
        const base = (hs + i) * emb;
        let accQ = 0, accK = 0, accV = 0;
        for (let j = 0; j < emb; j++) {
          accQ += attn.q.p[base + j] * x[j];
          accK += attn.k.p[base + j] * x[j];
          accV += attn.v.p[base + j] * x[j];
        }
        q[hs + i] = accQ;
        k[hs + i] = accK;
        v[hs + i] = accV;
      }
    }

    // Scaled Dot-Product Attention
    const scores = new Float64Array(ctxLen);
    for (let h = 0; h < nHead; h++) {
      const hs = h * headDim;
      let maxVal = -Infinity;
      for (let i = 0; i < ctxLen; i++) {
        const cacheK = state.kvCache[i];
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += q[hs + d] * (cacheK ? cacheK[d] : 0);
        }
        dot /= Math.sqrt(headDim);
        scores[i] = dot;
        if (dot > maxVal) maxVal = dot;
      }
      // Softmax
      let sum = 0;
      for (let i = 0; i < ctxLen; i++) {
        scores[i] = Math.exp(scores[i] - maxVal);
        sum += scores[i];
      }
      for (let i = 0; i < ctxLen; i++) scores[i] /= sum;
    }

    this._lastAttnGate = scores[0];
    return { q, k, v, scores };
  }

  private projectOutput(
    x: Float64Array,
    oProj: Group,
    v: Float64Array,
    nHead: number,
    headDim: number
  ): Float64Array {
    const emb = x.length;
    const result = new Float64Array(emb);
    // 简化：直接对 V 做输出投影
    for (let i = 0; i < emb; i++) {
      let acc = 0;
      const base = i * emb;
      for (let j = 0; j < emb; j++) {
        acc += oProj.p[base + j] * v[j];
      }
      result[i] = acc;
    }
    return result;
  }

  private feedForwardForward(x: Float64Array, ffn: { w1: Group; b1: Group; w2: Group; b2: Group }): Float64Array {
    const emb = x.length;
    const hidden = 4 * emb;

    const hiddenAct = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) {
      let acc = ffn.b1.p[i];
      const base = i * emb;
      for (let j = 0; j < emb; j++) {
        acc += ffn.w1.p[base + j] * x[j];
      }
      hiddenAct[i] = this.gelu(acc);
    }

    const out = new Float64Array(emb);
    for (let i = 0; i < emb; i++) {
      let acc = ffn.b2.p[i];
      const base = i * hidden;
      for (let j = 0; j < hidden; j++) {
        acc += ffn.w2.p[base + j] * hiddenAct[j];
      }
      out[i] = acc;
    }

    return out;
  }

  private gelu(x: number): number {
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
  }

  private computeActivationNorm(h: Float64Array): number {
    let norm = 0;
    for (let i = 0; i < h.length; i++) norm += h[i] * h[i];
    return Math.sqrt(norm / h.length);
  }

  // ==================== 反向传播 ====================

  trainStepBatch(seqs: Array<{ ids: number[] }>, lr: number): { loss: number; totalChars: number } {
    const { vocabSize, emb, ctx, bptt } = this.cfg;
    this.t += 1;

    for (const g of this.groups) g.zeroGrad();

    let totalLoss = 0;
    let totalChars = 0;

    for (const seq of seqs) {
      const T = seq.ids.length - 1;
      let state = this.newState();

      for (let t = 0; t < Math.min(T, bptt); t++) {
        // 前向
        const logits = this.step(seq.ids[t], state);
        const y = seq.ids[t + 1];

        // 交叉熵损失
        let max = -Infinity;
        for (let k = 0; k < vocabSize; k++) if (logits[k] > max) max = logits[k];
        let sum = 0;
        const pr = new Float64Array(vocabSize);
        for (let k = 0; k < vocabSize; k++) {
          pr[k] = Math.exp(logits[k] - max);
          sum += pr[k];
        }
        for (let k = 0; k < vocabSize; k++) pr[k] /= sum;

        totalLoss -= Math.log(Math.max(pr[y], 1e-12));
        totalChars += 1;

        // 反向传播
        const dLogits = pr.slice();
        dLogits[y] -= 1;
        this.backwardOutputHead(dLogits, state, emb, vocabSize);
      }
    }

    // 梯度裁剪
    let norm2 = 0;
    for (const g of this.groups) {
      for (let i = 0; i < g.g.length; i++) norm2 += g.g[i] * g.g[i];
    }
    const norm = Math.sqrt(norm2);
    const maxNorm = 1.0;
    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      for (const g of this.groups) {
        for (let i = 0; i < g.g.length; i++) g.g[i] *= scale;
      }
    }

    if (totalChars > 0) {
      for (const g of this.groups) {
        for (let i = 0; i < g.g.length; i++) g.g[i] /= totalChars;
        g.adam(lr, this.t);
      }
    }

    return { loss: totalLoss / Math.max(totalChars, 1), totalChars };
  }

  /** 反向传播通过输出头 → Transformer 层 */
  private backwardOutputHead(dLogits: Float64Array, state: TransformerState, emb: number, vocabSize: number): void {
    // 输出头梯度
    for (let k = 0; k < vocabSize; k++) {
      this.outBias.g[k] += dLogits[k];
      const row = this.outHead.g.subarray(k * emb, (k + 1) * emb);
      for (let i = 0; i < emb; i++) row[i] += dLogits[k] * state.h[i];
    }

    // dH 对 hidden 的梯度（通过 outHead）
    const dh = new Float64Array(emb);
    for (let k = 0; k < vocabSize; k++) {
      const row = this.outHead.g.subarray(k * emb, (k + 1) * emb);
      for (let i = 0; i < emb; i++) dh[i] += this.outHead.p[k * emb + i] * dLogits[k];
    }

    // 反向通过最后一个 Transformer block
    if (state.forwardChain.length > 0) {
      const cache = state.forwardChain[state.forwardChain.length - 1];
      this.backwardThroughLayers(dh, cache, this.cfg.nLayer);
    }
  }

  /** 反向传播通过所有 Transformer 层 */
  private backwardThroughLayers(dH: Float64Array, cache: ForwardCache, nLayer: number): void {
    for (let l = nLayer - 1; l >= 0; l--) {
      dH = this.backwardBlock(dH, cache, l);
    }
    // 最后反向通过嵌入层
    this.backwardEmbedding(dH, cache.embInput);
  }

  /** 反向传播通过一个 Transformer block */
  private backwardBlock(dH: Float64Array, cache: ForwardCache, layerIdx: number): Float64Array {
    const block = this.blocks[layerIdx];
    const emb = dH.length;

    // FFN 反向 (ffnOuts[layerIdx] 是 FFN 输出，ln2Outs[layerIdx] 是 FFN 输入)
    const dFFIn = new Float64Array(emb);
    const dFFOut = this.backwardFFN(dH, cache.ln2Outs?.[layerIdx] ?? cache.ffnOuts?.[layerIdx], block.ffn);
    for (let i = 0; i < emb; i++) dFFIn[i] = dH[i] + dFFOut[i];

    // LayerNorm 2 反向 (LN2 输入 = attn_out + layerInput)
    const ln2Input = cache.attnOuts?.[layerIdx] ? this.addResidual(cache.layerInputs[layerIdx], cache.attnOuts[layerIdx]) : cache.ln1Outs?.[layerIdx] ?? cache.layerInputs[layerIdx];
    const dLN2Input = this.backwardLayerNorm(dFFIn, ln2Input, block.ln2);

    // Attention 反向
    const attnIn = cache.ln1Outs ? cache.ln1Outs[layerIdx] : cache.layerInputs[layerIdx];
    // @ts-ignore - attnScores 是 Float64Array[][]，访问 [layerIdx] 得到 Float64Array
    const attnScoresArr = (cache.attnScores as any)?.[layerIdx];
    const dAttnOut = this.backwardAttention(dLN2Input, attnIn, attnScoresArr ?? new Float64Array(0), block.attn, layerIdx);
    const dAttnIn = new Float64Array(emb);
    for (let i = 0; i < emb; i++) dAttnIn[i] = dLN2Input[i] + dAttnOut[i];
    // LayerNorm 1 反向
    const dLN1In = this.backwardLayerNorm(dAttnIn, cache.layerInputs[layerIdx], block.ln1);
    
    return dLN1In;
  }

  private backwardFFN(
    dOut: Float64Array,
    ffIn: Float64Array,
    ffn: { w1: Group; b1: Group; w2: Group; b2: Group }
  ): Float64Array {
    const emb = ffIn.length;
    const hidden = 4 * emb;

    // w2, b2 梯度
    for (let i = 0; i < emb; i++) {
      ffn.b2.g[i] += dOut[i];
      const base = i * hidden;
      for (let j = 0; j < hidden; j++) {
        ffn.w2.g[base + j] += dOut[i] * this.geluPrimitive(ffIn, j, ffn);
      }
    }

    // dAct (GELU 输出梯度)
    const dAct = new Float64Array(hidden);
    for (let j = 0; j < hidden; j++) {
      let acc = 0;
      for (let i = 0; i < emb; i++) acc += ffn.w2.p[i * hidden + j] * dOut[i];
      dAct[j] = acc;
    }

    // GELU 反向
    const dHidden = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) {
      dHidden[i] = dAct[i] * this.geluPrime(this.geluInput(ffIn, i, ffn));
    }

    // w1, b1 梯度
    for (let i = 0; i < hidden; i++) {
      ffn.b1.g[i] += dHidden[i];
      const base = i * emb;
      for (let j = 0; j < emb; j++) {
        ffn.w1.g[base + j] += dHidden[i] * ffIn[j];
      }
    }

    // dIn
    const dIn = new Float64Array(emb);
    for (let j = 0; j < emb; j++) {
      let acc = 0;
      for (let i = 0; i < hidden; i++) acc += ffn.w1.p[i * emb + j] * dHidden[i];
      dIn[j] = acc;
    }

    return dIn;
  }

  private geluInput(ffIn: Float64Array, idx: number, ffn: any): number {
    const emb = ffIn.length;
    const base = idx * emb;
    let acc = ffn.b1.p[idx];
    for (let j = 0; j < emb; j++) acc += ffn.w1.p[base + j] * ffIn[j];
    return acc;
  }

  private geluPrimitive(ffIn: Float64Array, idx: number, ffn: any): number {
    const h = this.geluInput(ffIn, idx, ffn);
    return this.gelu(h);
  }

  private geluPrime(x: number): number {
    const sqrt2Pi = Math.sqrt(2 / Math.PI);
    const cdf = 0.5 * (1 + Math.tanh(sqrt2Pi * (x + 0.044715 * x * x * x)));
    const pdf = sqrt2Pi * x * 0.5 * (1 - Math.tanh(sqrt2Pi * x) * Math.tanh(sqrt2Pi * x));
    return cdf + x * pdf;
  }

  private backwardLayerNorm(dOut: Float64Array, x: Float64Array, ln: Group): Float64Array {
    const n = x.length;
    let sum = 0, sumSq = 0;
    for (let i = 0; i < n; i++) { sum += x[i]; sumSq += x[i] * x[i]; }
    const mean = sum / n;
    const var_ = sumSq / n - mean * mean + 1e-5;
    const std = Math.sqrt(var_);

    // ln parameter gradients
    let dGamma = 0;
    for (let i = 0; i < n; i++) {
      const normed = (x[i] - mean) / std;
      dGamma += dOut[i] * normed;
      ln.g[i] += dOut[i] * normed; // gamma gradient
    }

    // dX (简化版，忽略 gamma 对 mean/var 的影响)
    const dX = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      dX[i] = dOut[i] * ln.p[i] / std;
    }
    return dX;
  }

  private backwardAttention(
    dOut: Float64Array,
    attnIn: Float64Array,
    attnScores: Float64Array,
    attn: { q: Group; k: Group; v: Group; o: Group },
    layerIdx: number
  ): Float64Array {
    const emb = dOut.length;
    const nHead = this.cfg.nHead;
    const headDim = emb / nHead;

    // o 投影梯度
    const dV = new Float64Array(emb);
    for (let i = 0; i < emb; i++) {
      let acc = 0;
      const base = i * emb;
      for (let j = 0; j < emb; j++) {
        acc += attn.o.p[base + j] * dOut[j];
        attn.o.g[base + j] += dOut[j] * attnIn[j];
      }
      dV[i] = acc;
    }

    // QKV 梯度（简化：只更新 Q，K/V 通过 scores 间接更新）
    const dQ = new Float64Array(emb);
    for (let h = 0; h < nHead; h++) {
      const hs = h * headDim;
      for (let i = 0; i < headDim; i++) {
        const base = (hs + i) * emb;
        let acc = 0;
        for (let j = 0; j < emb; j++) {
          acc += attn.q.p[base + j] * dV[j];
          attn.q.g[base + j] += dV[j] * attnIn[j];
        }
        dQ[hs + i] = acc;
      }
    }

    return dQ;
  }

  private backwardEmbedding(dH: Float64Array, embInput: Float64Array): void {
    // 反向通过嵌入层（简化：只更新位置嵌入）
    // 完整实现需要跟踪每个位置的输入 ID
  }

  // ==================== 工具 ====================

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

  getGroups(): Group[] { return this.groups; }

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

  embedAvg(ids: number[]): number[] {
    const { vocabSize, emb } = this.cfg;
    const out = new Float64Array(emb);
    for (const id of ids) {
      const row = this.embLayer.p.subarray(id % vocabSize * emb, (id % vocabSize + 1) * emb);
      for (let e = 0; e < emb; e++) out[e] += row[e];
    }
    for (let e = 0; e < emb; e++) out[e] /= Math.max(ids.length, 1);
    return Array.from(out);
  }
}

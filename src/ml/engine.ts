import * as fs from "fs";
import * as path from "path";
import { CharGRU, CharGRUConfig, GRUState } from "./gru";
import { CharTokenizer } from "./tokenizer";
import { mulberry32 } from "./rng";
import { createStation } from "../station";
import { Task, TaskResult } from "../station/types";
import { VectorStore } from "./vectorstore";
import { TitansMemory } from "./titans";
import { chunkText } from "./data";

export interface CheckpointMeta {
  config: CharGRUConfig;
  paramCount: number;
  corpusTokens: number;
  epochs: number;
  vocab: string[];
  trainedAt: string;
}

function sampleToken(probs: Float64Array, topK: number, temperature: number, rand: () => number): number {
  const v = probs.length;
  const scaled = new Float64Array(v);
  let sum = 0;
  for (let k = 0; k < v; k++) {
    scaled[k] = Math.exp(Math.log(probs[k] + 1e-12) / Math.max(temperature, 1e-3));
    sum += scaled[k];
  }
  const k = Math.min(topK, v);
  const idxs = Array.from({ length: v }, (_, i) => i).sort((a, b) => scaled[b] - scaled[a]).slice(0, k);
  let mass = 0;
  for (const i of idxs) mass += scaled[i];
  let r = rand() * mass;
  for (const i of idxs) {
    r -= scaled[i];
    if (r <= 0) return i;
  }
  return idxs[idxs.length - 1];
}

export interface GenerateOptions {
  prompt: string;
  maxLen: number;
  budget: number;
  temperature: number;
  topK: number;
  seed?: number;
}

export interface CharRoute {
  char: string;
  depth: number;
  confidence: number;
  units: string[];
  memoryHit: boolean;
  /** 记忆来源: 向量库 / 重复上下文 */
  memorySource?: "vector" | "repeat";
}

export interface GenerateResult {
  text: string;
  routes: CharRoute[];
  budget: number;
}

/** 测试时记忆块标记: 写入向量库的生成内容前缀,检索时跳过,防止自举循环 */
const MEMORY_TAG = "\uE000";

/**
 * 推理引擎: 模型 + 字符分词 + 工作站(分级路由 + Titans L0 记忆)
 * L0 记忆缓存直出(零算力) → L1 浅路径 → L2 深路径(预算内扩深)
 */
export class Engine {
  readonly model: CharGRU;
  readonly tokenizer: CharTokenizer;
  readonly meta: CheckpointMeta;
  private station = createStation();
  private habit: () => number;
  private memory = new Map<string, string>();
  private ctxWin: number[] = [];
  private store: VectorStore | null;
  private recentChars: string[] = [];
  private state: GRUState;
  /** 在线神经记忆(Titans 逻辑): 上下文嵌入 → 下一字, 生成期持续写入 */
  private titansMem: TitansMemory;
  /** 记忆槽 → 存储的下一字符(读取用) */
  private titansChars = new Map<number, string>();

  private constructor(model: CharGRU, tokenizer: CharTokenizer, meta: CheckpointMeta) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.meta = meta;
    this.habit = mulberry32(Date.now() % 2147483647);
    this.state = model.newState();
    this.store = VectorStore.load("data/vectorstore");
    if (this.store) console.log(`向量库已加载: ${this.store.size} 块`);
    this.titansMem = new TitansMemory({ dim: meta.config.emb, slots: 256 });
    this.registerUnits();
  }

  /** 检索: 把查询文本向量化,返回库内 top-k 片段 */
  retrieve(q: string, k = 4): Array<{ text: string; score: number }> {
    if (!this.store) return [];
    const ids = this.tokenizer.encode(q);
    return this.store.search(this.model.embedAvg(ids), k);
  }

  retrieveCount(): number {
    return this.store?.size ?? 0;
  }

  /** 测试时记忆: 把生成内容切块写入向量库(可持久化) */
  remember(text: string): void {
    if (!this.store) return;
    let added = 0;
    for (const chunk of chunkText(text, 64, 32)) {
      if (chunk.length < 8) continue;
      const ids = this.tokenizer.encode(chunk);
      this.store.add(MEMORY_TAG + chunk, this.model.embedAvg(ids));
      added += 1;
    }
    this.store.save("data/vectorstore");
    if (added > 0) console.log(`测试时记忆: 新写入 ${added} 块,向量库现有 ${this.store.size} 块`);
  }

  /** 采样前降权: 近 16 个已出字符中重复≥2 次的禁出 + UNK 槽禁出(防 12 字级自举循环) */
  private penalized(probs: Float64Array): Float64Array {
    const out = probs.slice();
    out[out.length - 1] = 0;
    const freq = new Map<string, number>();
    for (const ch of this.recentChars) freq.set(ch, (freq.get(ch) ?? 0) + 1);
    for (const [ch, count] of freq) {
      if (count < 2) continue;
      const id = this.tokenizer.idOf(ch);
      if (id !== undefined && id < out.length) out[id] = 0;
    }
    let sum = 0;
    for (let i = 0; i < out.length; i++) sum += out[i];
    for (let i = 0; i < out.length; i++) out[i] /= sum;
    return out;
  }

  private propose(ch: string): void {
    this.recentChars.push(ch);
    if (this.recentChars.length > 16) this.recentChars.shift();
  }

  /** 窥视: 用当前状态副本推一步,不推进真实状态 */
  private peekLogits(depth: 1 | 2 | 3 | 4): Float64Array {
    const lastId = this.ctxWin.length > 0 ? this.ctxWin[this.ctxWin.length - 1] : 0;
    const copy: GRUState = {
      h1: this.state.h1.slice(),
      h2: this.state.h2.slice(),
      hist: this.state.hist.map((h) => h.slice()),
      ids: [...this.state.ids],
    };
    return this.model.step(lastId, copy, depth);
  }

  /** 推进: 真实状态吸收新字符 */
  private advance(chId: number): void {
    this.model.step(chId, this.state, 4);
  }

  private recallAt(ctxIds: number[]): { char: string; source: "vector" | "repeat" } | null {
    if (this.store) {
      const q = this.tokenizer.decode(ctxIds);
      const hits = this.store.search(this.model.embedAvg(ctxIds), 8);
      for (const hit of hits) {
        if (hit.text.startsWith(MEMORY_TAG)) continue;
        const at = hit.text.indexOf(q);
        if (at >= 0 && at + q.length < hit.text.length) {
          const ch = hit.text[at + q.length];
          if (this.isDegenerate(ch)) return null;
          return { char: ch, source: "vector" };
        }
      }
    }
    const cached = this.memory.get(ctxIds.join(","));
    if (cached !== undefined && !this.isDegenerate(cached)) {
      return { char: cached, source: "repeat" };
    }
    // Titans 在线神经记忆: 上下文嵌入检索 → 槽位存储的下一字符
    const q = Float64Array.from(this.model.embedAvg(ctxIds));
    const cache = { query: new Float64Array(0), keys: new Float64Array(0), values: new Float64Array(0), weights: new Float64Array(0) };
    const mem = this.titansMem.read(q, cache);
    if (mem.hitSlots > 0 && cache.weights.length > 0) {
      let best = 0;
      for (let i = 1; i < cache.weights.length; i++) if (cache.weights[i] > cache.weights[best]) best = i;
      if (cache.weights[best] > 0.5) {
        const ch = this.titansChars.get(best);
        if (ch !== undefined && !this.isDegenerate(ch)) {
          this.station.blackboard.write("z:act:titans", mem.hitSlots / this.titans.cfg.slots);
          return { char: ch, source: "repeat" };
        }
      }
    }
    return null;
  }

  /** 循环防护: 单字符重复 ≥3 次或最近 6 字符整段重复,判定为退化输出 */
  private isDegenerate(ch: string): boolean {
    if (this.recentChars.filter((c) => c === ch).length >= 3) return true;
    const n = this.recentChars.length;
    if (n >= 12) {
      const a = this.recentChars.slice(-6).join("");
      const b = this.recentChars.slice(-12, -6).join("");
      if (a === b) return true;
    }
    return false;
  }

  static load(checkpointDir: string): Engine {
    const meta = JSON.parse(fs.readFileSync(path.join(checkpointDir, "meta.json"), "utf-8")) as CheckpointMeta;
    const params = JSON.parse(fs.readFileSync(path.join(checkpointDir, "model.json"), "utf-8")) as number[];
    const model = new CharGRU(meta.config, 1);
    if (model.paramCount() !== params.length) {
      throw new Error(`checkpoint 参数数量与配置不匹配: ${model.paramCount()} != ${params.length}`);
    }
    model.load(params);
    const tokenizer = new CharTokenizer();
    tokenizer.load(meta.vocab);
    return new Engine(model, tokenizer, meta);
  }

  private registerUnits(): void {
    const reg = this.station.registry;
    const engine = this;
    // 感知神经: 第 1 层 GRU(emb → 状态流), 神经链入口
    reg.register({
      kind: "gru", name: "GRU L1", role: "感知: 字符嵌入 → 状态流", stage: "phase1", enabled: true, status: "idle",
      roleTag: "perception",
      justification: "基线状态流; 无此层无任何输出",
      compute: {
        unitId: "gru:shallow", depth: 1, cost: 1, chains: ["text"],
        forward: async (task, state) => {
          const logits = engine.peekLogits(1);
          state.data[`z:logits:${task.id}`] = logits;
          state.data["z:act:gru:shallow"] = engine.model.confidence(logits);
          return engine.model.confidence(logits);
        },
      },
    });
    if (engine.model.cfg.mamba) {
      // 中央神经(快动力学): 真选择性 SSM(depth 2), 点火 = Δ 均值
      reg.register({
        kind: "mamba", name: "Mamba", role: "快动力学: 选择性 SSM 状态流", stage: "phase1", enabled: true, status: "idle",
        roleTag: "central",
        justification: "对照基线: 线性状态建模 vs GRU(同配置)", evidence: { metric: "事件语料 top-1(30k×2ep)", baseline: 49.1, result: 47.5, status: "pending", note: "gradcheck=0.0000; 与基线持平, 缺口证据待补" },
        compute: {
          unitId: "mamba:ssm", depth: 2, cost: 2, chains: ["text"],
          forward: async (task, state) => {
            const logits = engine.peekLogits(2);
            state.data[`z:logits:${task.id}`] = logits;
            state.data["z:act:mamba:ssm"] = Math.min(1, engine.model.ssm!.lastDeltaMean / 1.5);
            return engine.model.confidence(logits);
          },
        },
      });
    } else {
      // 中央神经(快动力学): GRU 第 2 层
      reg.register({
        kind: "gru", name: "GRU L2", role: "快动力学: 第二层 GRU 状态流", stage: "phase1", enabled: true, status: "idle",
        roleTag: "central",
        justification: "共享底座深路径",
        compute: {
          unitId: "gru:deep", depth: 2, cost: 2, chains: ["text"],
          forward: async (task, state) => {
            const logits = engine.peekLogits(2);
            state.data[`z:logits:${task.id}`] = logits;
            state.data["z:act:gru:deep"] = engine.model.confidence(logits);
            return engine.model.confidence(logits);
          },
        },
      });
    }
    if (engine.model.cfg.attn) {
      // 中央神经(慢语义): 学习型自注意力(depth 3), 点火 = gate
      reg.register({
        kind: "sparse-attention", name: "稀疏 Attention", role: "慢语义: 学习型自注意力", stage: "phase2", enabled: true, status: "idle",
        roleTag: "central",
        justification: "跨位置信息混合,独立于循环路径;真实计算单元", evidence: { metric: "事件语料 top-1(30k×2ep)", baseline: 49.1, result: 47.8, status: "pending", note: "机制正确+gate 选择性; 任务级证据未达成→不 enabled" },
        compute: {
          unitId: "attn:layer3", depth: 3, cost: 3, chains: ["text"],
          forward: async (task, state) => {
            const logits = engine.peekLogits(3);
            state.data[`z:logits:${task.id}`] = logits;
            state.data["z:act:attn:layer3"] = engine.model.lastAttnGate;
            return engine.model.confidence(logits);
          },
        },
      });
    }
    if (engine.model.cfg.cnn) {
      // 感知神经(局部): 1D 卷积 n-gram(depth 4), 点火 = 特征激活均值
      reg.register({
        kind: "cnn", name: "CNN", role: "局部感知: 1D 卷积 n-gram 特征", stage: "phase1", enabled: true, status: "idle",
        roleTag: "perception",
        justification: "局部字符模式特征,独立预测头与循环/注意力输出融合", evidence: { metric: "事件语料 top-1(30k×4ep 同配置消融)", baseline: 47.5, result: 49.3, status: "pass", note: "gradcheck-cnn=0.0000; 修复 top-1 缺口 +1.8pt; 就后召回探针 9.7→6.5(小规模代价, 大语料复查)" },
        compute: {
          unitId: "cnn:layer4", depth: 4, cost: 1, chains: ["text"],
          forward: async (task, state) => {
            const logits = engine.peekLogits(4);
            state.data[`z:logits:${task.id}`] = logits;
            state.data["z:act:cnn:layer4"] = engine.model.lastCnnAct;
            return engine.model.confidence(logits);
          },
        },
      });
    }
  }

  /** 单次生成选项(由 generate 设置,单元读取) */
  private options: { temperature: number; topK: number } = { temperature: 0.85, topK: 5 };

  async generate(opts: GenerateOptions): Promise<GenerateResult> {
    const { prompt, maxLen, budget } = opts;
    this.options = { temperature: opts.temperature, topK: opts.topK };
    if (opts.seed !== undefined) this.habit = mulberry32(opts.seed);
    // 关键: prompt 必须先推进真实状态,模型才能"看到"完整上下文
    this.state = this.model.newState();
    for (const id of this.tokenizer.encode(prompt)) this.advance(id);
    this.ctxWin = this.tokenizer.encode(prompt).slice(-this.model.cfg.ctx);
    const routes: CharRoute[] = [];
    let text = prompt;
    for (let i = 0; i < maxLen; i++) {
      const ctx = this.ctxWin.slice();
      const task: Task = { id: `m${i}`, type: "text", input: ctx, complexity: Math.min(1, 0.4 + (i / Math.max(maxLen, 1)) * 0.4), budget };
      const recalled = this.recallAt(ctx);
      if (recalled) this.station.blackboard.write(`mem:${task.id}`, recalled.char);
      const res: TaskResult = await this.station.executor.execute(task);
      let ch = String(res.output ?? "");
      if (res.depth > 0) {
        const zLogits = this.station.blackboard.read(`z:logits:${task.id}`) as Float64Array | undefined;
        if (zLogits) {
          const probs = this.model.logitsToProbs(zLogits);
          ch = this.tokenizer.charOf(sampleToken(this.penalized(probs), this.options.topK, this.options.temperature, this.habit));
        }
      }
      routes.push({
        char: ch,
        depth: res.depth,
        confidence: res.confidence,
        units: res.units,
        memoryHit: res.memoryHit,
        memorySource: recalled?.source,
      });
      this.propose(ch);
      text += ch;
      const id = this.tokenizer.idOf(ch);
      if (id !== undefined) {
        this.advance(id);
        this.ctxWin.push(id);
        this.ctxWin = this.ctxWin.slice(-this.model.cfg.ctx);
      }
      if (!res.memoryHit) this.memory.set(ctx.join(","), ch);
      if (this.memory.size > 600) {
        const first = this.memory.keys().next().value;
        if (first !== undefined) this.memory.delete(first);
      }
      // 在线写入 Titans 神经记忆(上下文嵌入 → 本步输出字符)
      if (id !== undefined) {
        const slot = this.titansMem.write(Float64Array.from(this.model.embedAvg(ctx)));
        this.titansChars.set(slot, ch);
      }
    }
    this.remember(text.substring(prompt.length));
    return { text, routes, budget };
  }

  get registry() {
    return this.station.registry;
  }

  get blackboard() {
    return this.station.blackboard;
  }

  get audit() {
    return this.station.audit;
  }

  get probe() {
    return this.station.probe;
  }

  get titans() {
    return this.titansMem;
  }
}

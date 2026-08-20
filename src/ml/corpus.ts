import { mulberry32 } from "./rng";

export interface CorpusOptions {
  seed?: number;
  /** 目标生成的字符数(文本世界预算口径) */
  tokens?: number;
}

/**
 * 文本世界 v1: 程序生成的"中文事件时序语料"
 * 人物在场景里做一连串事件,句号分隔,可复现、可标注
 */
export class WorldCorpus {
  private rng: () => number;
  private names = ["小明", "阿宝", "小刚", "小美", "老王", "阿芳"];
  private places = ["屋", "院", "街", "厨", "园", "室"];
  private objects = ["门", "灯", "茶", "石", "球", "书", "笔", "杯", "桌", "蕉", "玩具", "熊"];
  private verbs = ["开", "关", "拿", "放", "吃", "读", "看", "摸", "敲", "洗", "扔", "捡"];
  private feels = ["好", "气", "累", "乐", "急"];

  constructor(options: CorpusOptions = {}) {
    this.rng = mulberry32(options.seed ?? 7);
  }

  private pick<T>(arr: T[]): T {
    return arr[Math.floor(this.rng() * arr.length)];
  }

  private sentence(): string {
    const n = this.pick(this.names);
    const v1 = this.pick(this.verbs);
    const o1 = this.pick(this.objects);
    const kind = Math.floor(this.rng() * 4);
    if (kind === 0) return `${n}${v1}${o1}。`;
    if (kind === 1) return `${n}在${this.pick(this.places)}里${v1}${o1}。`;
    if (kind === 2) {
      const v2 = this.pick(this.verbs);
      const o2 = this.pick(this.objects);
      return `如果${n}${v1}${o1},${n}就${v2}${o2}。`;
    }
    const v2 = this.pick(this.verbs);
    return `${n}${v1}${o1},又${v2}${this.pick(this.objects)},${n}觉得${this.pick(this.feels)}。`;
  }

  /** 一段事件: 5~9 句 */
  paragraph(): string {
    let out = "";
    const count = 5 + Math.floor(this.rng() * 5);
    for (let i = 0; i < count; i++) out += this.sentence();
    return out + "\n";
  }

  /** 生成目标字符数的全文 */
  generate(tokens?: number): string {
    const target = tokens ?? 150_000;
    let text = "";
    while (text.length < target) text += this.paragraph();
    return text.slice(0, target);
  }
}

export function buildCorpus(options: CorpusOptions = {}): string {
  return new WorldCorpus(options).generate(options.tokens);
}

/**
 * 关联回忆语料(Attention 证据任务): 
 * "N1 V1 O1, N2 V2 O2, ..., N1就V1O1。"
 * 句尾动词/宾语由句首主语决定, 中间夹 1~3 个干扰三元组(主语互不相同),
 * 必须跨位置检索才能预测 → 给 attention 的"它擅长的任务"(证据门槛用)。
 */
export class AssocCorpus {
  private rng: () => number;
  private names = ["小明", "阿宝", "小刚", "小美", "老王", "阿芳"];
  private verbs = ["开", "关", "拿", "放", "吃", "读", "看", "摸"];
  private objects = ["门", "灯", "茶", "石", "球", "书", "笔", "杯"];

  constructor(options: CorpusOptions = {}) {
    this.rng = mulberry32(options.seed ?? 7);
  }

  private pick<T>(arr: T[]): T {
    return arr[Math.floor(this.rng() * arr.length)];
  }

  private sentence(): string {
    const k = 1 + Math.floor(this.rng() * 3); // 干扰三元组 1~3
    const pool = [...this.names];
    const n1 = pool.splice(Math.floor(this.rng() * pool.length), 1)[0];
    const triples: Array<{ n: string; v: string; o: string }> = [{ n: n1, v: this.pick(this.verbs), o: this.pick(this.objects) }];
    for (let i = 0; i < k; i++) {
      const n = pool.splice(Math.floor(this.rng() * pool.length), 1)[0];
      triples.push({ n, v: this.pick(this.verbs), o: this.pick(this.objects) });
    }
    let out = "";
    for (const t of triples) out += `${t.n}${t.v}${t.o},`;
    out += `${n1}就${triples[0].v}${triples[0].o}。`;
    return out;
  }

  paragraph(): string {
    let out = "";
    const count = 4 + Math.floor(this.rng() * 4);
    for (let i = 0; i < count; i++) out += this.sentence();
    return out + "\n";
  }

  generate(tokens?: number): string {
    const target = tokens ?? 150_000;
    let text = "";
    while (text.length < target) text += this.paragraph();
    return text.slice(0, target);
  }
}

/**
 * 强关联回忆语料(Attention 证据任务 v2):
 * 单字人名(甲乙丙丁戊己); 句中所有干扰三元组的动词 ≠ 目标动词(只出现一次, 在句首);
 * 句尾 "X就V O。" 必须跨位置按主语 X 召回 → 最近动词启发式=0%, 信号干净。
 */
export class RecallCorpus {
  private rng: () => number;
  private names = ["甲", "乙", "丙", "丁", "戊", "己"];
  private verbs = ["开", "关", "拿", "放", "吃", "读", "看", "摸"];
  private objects = ["门", "灯", "茶", "石", "球", "书", "笔", "杯"];

  constructor(options: CorpusOptions = {}) {
    this.rng = mulberry32(options.seed ?? 7);
  }

  private pick<T>(arr: T[]): T {
    return arr[Math.floor(this.rng() * arr.length)];
  }

  private sentence(): string {
    const dist = 2 + Math.floor(this.rng() * 3); // 干扰三元组 2~4
    const pool = [...this.names];
    const x = pool.splice(Math.floor(this.rng() * pool.length), 1)[0];
    const v = this.pick(this.verbs);
    const o = this.pick(this.objects);
    let out = `${x}${v}${o},`;
    for (let i = 0; i < dist; i++) {
      const n = pool.splice(Math.floor(this.rng() * pool.length), 1)[0];
      let v2 = this.pick(this.verbs);
      while (v2 === v) v2 = this.pick(this.verbs); // 干扰动词 ≠ 目标动词
      out += `${n}${v2}${this.pick(this.objects)},`;
    }
    out += `${x}就${v}${o}。`;
    return out;
  }

  paragraph(): string {
    let out = "";
    const count = 4 + Math.floor(this.rng() * 4);
    for (let i = 0; i < count; i++) out += this.sentence();
    return out + "\n";
  }

  generate(tokens?: number): string {
    const target = tokens ?? 150_000;
    let text = "";
    while (text.length < target) text += this.paragraph();
    return text.slice(0, target);
  }
}

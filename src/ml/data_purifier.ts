/**
 * 数据净化管道 V2 — 从清洗到结构化知识提取
 *
 * ## 设计原则
 *
 * 1. **多层过滤**：字符级 → 序列级 → 语料级
 * 2. **信息密度评估**：保留高信息量样本，丢弃冗余/噪声
 * 3. **结构化提取**：从文本中提取可复用的知识单元
 * 4. **在线适应**：根据训练反馈动态调整过滤阈值
 *
 * ## 与 V3 的区别
 *
 * | 维度 | V3 (旧) | V2 (新) |
 * |---|---|---|
 * | 过滤层级 | 单级异常检测 | 多级流水线 |
 * | 信息评估 | 无 | 字符熵 + N-gram 多样性 |
 * | 结构化输出 | 无 | 知识单元提取 |
 * | 在线适应 | 固定阈值 | EMA 动态调整 |
 */

export interface PurifyConfig {
  /** 字符利用率下限 (unique/total) */
  minUniqueRatio: number;
  /** 字符熵上限 (bits/char) */
  maxEntropy: number;
  /** Bigram 重复率上限 */
  maxBigramRepeat: number;
  /** 周期性检测阈值 */
  periodicThreshold: number;
  /** 最小文本长度 */
  minLen: number;
  /** 最大文本长度 */
  maxLen: number;
  /** 信息密度阈值 (用于排序) */
  minInfoDensity: number;
}

export const DEFAULT_PURIFY_CONFIG: PurifyConfig = {
  minUniqueRatio: 0.3,
  maxEntropy: 8.0,
  // 放宽到 0.6: 古典中文有大量合法重复 bigram (如"臣谨按"、"帝曰"、"话说")
  // 原始 0.8 阈值会误杀正常文本; 噪声检测主要靠熵和周期性
  maxBigramRepeat: 0.6,
  periodicThreshold: 0.95,
  minLen: 8,
  maxLen: 512,
  minInfoDensity: 0.1,
};

export interface PurifyResult {
  text: string;
  score: number;          // 综合质量分 0-1
  isKept: boolean;
  details: Array<{
    method: string;
    score: number;
    threshold: number;
    pass: boolean;
  }>;
  /** 提取的知识单元 (如有) */
  knowledge?: KnowledgeUnit[];
}

export interface KnowledgeUnit {
  type: 'pattern' | 'phrase' | 'entity' | 'relation';
  content: string;
  confidence: number;
  sourceText: string;
  position: number;
}

/**
 * 数据净化器
 */
export class DataPurifier {
  readonly cfg: PurifyConfig;
  private scoreHistory: number[] = [];
  private maxHistory = 100;

  constructor(cfg: Partial<PurifyConfig> = {}) {
    this.cfg = { ...DEFAULT_PURIFY_CONFIG, ...cfg };
  }

  /**
   * 净化单条文本
   */
  purify(text: string): PurifyResult {
    const details: PurifyResult['details'] = [];
    let score = 1.0;

    // 长度过滤
    if (text.length < this.cfg.minLen || text.length > this.cfg.maxLen) {
      return {
        text,
        score: 0,
        isKept: false,
        details: [{ method: '长度过滤', score: 0, threshold: this.cfg.minLen, pass: false }],
      };
    }

    // --- 层级 1: 字符级统计 ---
    const charStats = this.analyzeChars(text);

    // 字符利用率
    const uniqueRatio = charStats.uniqueChars / Math.max(text.length, 1);
    const uniquePass = uniqueRatio >= this.cfg.minUniqueRatio;
    details.push({ method: '字符利用率', score: uniqueRatio, threshold: this.cfg.minUniqueRatio, pass: uniquePass });
    if (!uniquePass) score *= 0.5;

    // 字符熵
    const entropyScore = charStats.entropy > this.cfg.maxEntropy
      ? (charStats.entropy - this.cfg.maxEntropy) / this.cfg.maxEntropy
      : 0;
    details.push({ method: '字符熵', score: entropyScore, threshold: 1, pass: entropyScore < 0.5 });
    if (entropyScore > 0.5) score *= 0.3;

    // Bigram 重复率
    const bigramRepeat = this.calcBigramRepeat(text);
    const bigramPass = bigramRepeat <= this.cfg.maxBigramRepeat;
    details.push({ method: 'Bigram重复率', score: bigramRepeat, threshold: this.cfg.maxBigramRepeat, pass: bigramPass });
    if (!bigramPass) score *= 0.4;

    // 周期性检测
    if (text.length >= 40) {
      const periodicity = this.detectPeriodicity(text);
      details.push({ method: '周期性', score: periodicity, threshold: this.cfg.periodicThreshold, pass: periodicity < this.cfg.periodicThreshold });
      if (periodicity > this.cfg.periodicThreshold) score *= 0.2;
    }

    // --- 层级 2: 信息密度评估 ---
    const infoDensity = this.calcInfoDensity(text);
    details.push({ method: '信息密度', score: infoDensity, threshold: this.cfg.minInfoDensity, pass: infoDensity >= this.cfg.minInfoDensity });
    if (infoDensity < this.cfg.minInfoDensity) score *= 0.5;

    // --- 层级 3: 知识单元提取 ---
    const knowledge = this.extractKnowledge(text, charStats);

    const isKept = score >= this.cfg.minInfoDensity && details.every(d => d.pass || d.score < d.threshold * 0.8);

    this.scoreHistory.push(score);
    if (this.scoreHistory.length > this.maxHistory) this.scoreHistory.shift();

    return { text, score, isKept, details, knowledge: knowledge.length > 0 ? knowledge : undefined };
  }

  /**
   * 批量净化 + 排序
   */
  purifyBatch(texts: string[], topK?: number): PurifyResult[] {
    const results = texts.map(t => this.purify(t));
    // 按质量分排序
    results.sort((a, b) => b.score - a.score);
    // 只保留通过过滤的
    const kept = results.filter(r => r.isKept);
    // 如果要求 topK，只返回前 K 个
    return topK ? kept.slice(0, topK) : kept;
  }

  /**
   * 动态调整阈值 (基于历史分数 EMA)
   */
  adaptThresholds(alpha: number = 0.05): void {
    if (this.scoreHistory.length < 10) return;
    const ema = this.scoreHistory.reduce((a, b) => a + b, 0) / this.scoreHistory.length;
    // 如果平均分数太低，放宽阈值
    if (ema < 0.3) {
      this.cfg.minUniqueRatio = Math.max(0.2, this.cfg.minUniqueRatio - 0.05);
      this.cfg.maxEntropy = Math.min(9.0, this.cfg.maxEntropy + 0.5);
    } else if (ema > 0.7) {
      // 分数高，可以收紧
      this.cfg.minUniqueRatio = Math.min(0.5, this.cfg.minUniqueRatio + 0.02);
      this.cfg.maxEntropy = Math.max(7.0, this.cfg.maxEntropy - 0.2);
    }
  }

  // ==================== 内部方法 ====================

  private analyzeChars(text: string): { uniqueChars: number; entropy: number; freq: Map<string, number> } {
    const freq = new Map<string, number>();
    for (const ch of text) freq.set(ch, (freq.get(ch) ?? 0) + 1);

    const uniqueChars = freq.size;
    const totalChars = text.length;

    let entropy = 0;
    for (const [, count] of freq) {
      const p = count / totalChars;
      if (p > 0) entropy -= p * Math.log2(p);
    }

    return { uniqueChars, entropy, freq };
  }

  private calcBigramRepeat(text: string): number {
    if (text.length < 2) return 0;
    const bigrams = new Map<string, number>();
    for (let i = 0; i < text.length - 1; i++) {
      const bg = text.slice(i, i + 2);
      bigrams.set(bg, (bigrams.get(bg) ?? 0) + 1);
    }
    const total = text.length - 1;
    const repeated = [...bigrams.values()].filter(c => c > 1).reduce((a, b) => a + b, 0);
    return repeated / total;
  }

  private detectPeriodicity(text: string): number {
    const n = text.length;
    const maxPeriod = Math.min(Math.floor(n / 4), 50);
    let maxMatch = 0;

    for (let period = 2; period <= maxPeriod; period++) {
      let matches = 0;
      for (let i = period; i < n; i++) {
        if (text[i] === text[i % period]) matches++;
      }
      const ratio = matches / (n - period);
      maxMatch = Math.max(maxMatch, ratio);
    }

    return maxMatch;
  }

  private calcInfoDensity(text: string): number {
    const chars = new Set(text.split('')).size;
    const len = text.length;
    // 字符多样性 / 长度
    const diversity = chars / Math.max(len, 1);
    // Bigram 多样性
    const bigrams = new Set();
    for (let i = 0; i < len - 1; i++) bigrams.add(text.slice(i, i + 2));
    const bigramDiv = bigrams.size / Math.max(len - 1, 1);
    // 综合
    return 0.5 * diversity + 0.5 * bigramDiv;
  }

  private extractKnowledge(text: string, stats: { freq: Map<string, number> }): KnowledgeUnit[] {
    const units: KnowledgeUnit[] = [];

    // 提取高频字符序列 (潜在的模式)
    const bigrams = new Map<string, number>();
    for (let i = 0; i < text.length - 1; i++) {
      const bg = text.slice(i, i + 2);
      bigrams.set(bg, (bigrams.get(bg) ?? 0) + 1);
    }

    // 提取 Top-5 高频 bigram
    const sorted = [...bigrams.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
    for (const [bg, count] of sorted) {
      if (count >= 3) {
        units.push({
          type: 'pattern',
          content: bg,
          confidence: Math.min(1, count / 10),
          sourceText: text,
          position: text.indexOf(bg),
        });
      }
    }

    return units;
  }

  getAverageScore(): number {
    if (this.scoreHistory.length === 0) return 1;
    return this.scoreHistory.reduce((a, b) => a + b, 0) / this.scoreHistory.length;
  }

  report(results: PurifyResult[]): string {
    const total = results.length;
    const kept = results.filter(r => r.isKept).length;
    const avgScore = results.reduce((s, r) => s + r.score, 0) / Math.max(total, 1);
    return [
      `== 数据净化报告 ==`,
      `总样本: ${total}`,
      `保留: ${kept} (${(kept / Math.max(total, 1) * 100).toFixed(1)}%)`,
      `平均质量分: ${avgScore.toFixed(4)}`,
      avgScore > 0.5 ? '✅ 数据质量良好' : '⚠️ 数据质量偏低，建议调整阈值',
    ].join('\n');
  }
}

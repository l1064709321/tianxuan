/**
 * 数据清洗器 V3 — 简化且可靠的异常检测
 *
 * 核心指标:
 * 1. 字符熵: 随机噪声熵高，正常文本熵适中
 * 2. 重复字符比例: 重复攻击会有高重复率
 * 3. 字符种类数: 随机噪声种类多
 */

export interface AnomalyResult {
  id: string;
  anomalyScore: number;
  isAnomalous: boolean;
  details: Array<{ method: string; score: number; threshold: number; pass: boolean }>;
  trustScore: number;
}

/**
 * 计算字符串统计
 */
function analyzeText(text: string): {
  uniqueChars: number;
  totalChars: number;
  repeatRatio: number;
  entropy: number;
  freq: Map<string, number>;
} {
  const freq = new Map<string, number>();
  for (const ch of text) {
    freq.set(ch, (freq.get(ch) ?? 0) + 1);
  }

  const uniqueChars = freq.size;
  const totalChars = text.length;
  const repeatRatio = 1 - uniqueChars / Math.max(totalChars, 1);

  // 计算熵
  let entropy = 0;
  for (const [, count] of freq) {
    const p = count / totalChars;
    if (p > 0) entropy -= p * Math.log2(p);
  }

  return { uniqueChars, totalChars, repeatRatio, entropy, freq };
}

/**
 * 数据清洗器 V3
 */
export class DataCleaner {
  private history: Array<{ uniqueRatio: number; repeatRatio: number; entropy: number }> = [];
  private maxHistory = 50;

  detect(text: string, context?: { id?: string }): AnomalyResult {
    const details: AnomalyResult["details"] = [];
    let maxScore = 0;

    if (text.length < 4) {
      return {
        id: context?.id ?? "unknown",
        anomalyScore: 0,
        isAnomalous: false,
        details: [{ method: "过短文本", score: 0, threshold: 0, pass: true }],
        trustScore: 1.0,
      };
    }

    const stats = analyzeText(text);
    const { uniqueChars, totalChars, repeatRatio, entropy } = stats;

    // ── 指标 1: 重复字符比例 ────────────────────────────────────
    // 正常中文: repeatRatio ≈ 0.1-0.4 (有些字符会重复)
    // 重复攻击: repeatRatio → 1.0 (所有字符相同)
    // 随机噪声: repeatRatio → 0 (几乎不重复)
    const repeatScore = repeatRatio > 0.8 ? (repeatRatio - 0.8) / 0.2 : 0;
    details.push({
      method: "重复字符比例",
      score: repeatScore,
      threshold: 0.5,
      pass: repeatScore < 0.5,
    });
    maxScore = Math.max(maxScore, repeatScore);

    // ── 指标 2: 字符种类比例 ────────────────────────────────────
    // 正常中文: unique/total ≈ 0.5-0.9
    // 随机噪声: unique/total → 1.0 (几乎每个字符都不同)
    const uniqueRatio = uniqueChars / totalChars;
    const uniqueScore = uniqueRatio > 0.9 ? (uniqueRatio - 0.9) / 0.1 : 0;
    details.push({
      method: "字符种类比例",
      score: uniqueScore,
      threshold: 0.5,
      pass: uniqueScore < 0.5,
    });
    maxScore = Math.max(maxScore, uniqueScore);

    // ── 指标 3: 字符熵 ──────────────────────────────────────────
    // 正常中文: 熵 ≈ 4-7 bits/char
    // 随机噪声: 熵 → 8+ bits/char
    const entropyScore = entropy > 7.5 ? Math.min(1, (entropy - 7.5) / 2) : 0;
    details.push({
      method: "字符熵",
      score: entropyScore,
      threshold: 0.5,
      pass: entropyScore < 0.5,
    });
    maxScore = Math.max(maxScore, entropyScore);

    // ── 指标 4: 周期性检测 (仅长文本，周期上限 100 防 O(N²)) ─────────────────
    if (totalChars >= 40) {
      let periodicScore = 0;
      const maxPeriod = Math.min(Math.floor(totalChars / 4), 100);  // 限制最大周期
      for (let period = 2; period <= maxPeriod; period++) {
        let matches = 0;
        for (let i = period; i < totalChars; i++) {
          if (text[i] === text[i % period]) matches++;
        }
        const matchRatio = matches / (totalChars - period);
        if (matchRatio > 0.95) {  // 非常严格
          periodicScore = Math.max(periodicScore, matchRatio);
        }
      }
      if (periodicScore > 0) {
        details.push({
          method: "周期性模式",
          score: periodicScore,
          threshold: 0.8,
          pass: periodicScore < 0.8,
        });
        maxScore = Math.max(maxScore, periodicScore);
      }
    }

    // 更新历史
    this.history.push({ uniqueRatio, repeatRatio, entropy });
    if (this.history.length > this.maxHistory) this.history.shift();

    // ── 综合判定 ───────────────────────────────────────────────
    const anomalyScore = maxScore;
    const isAnomalous = anomalyScore > 0.5;

    return {
      id: context?.id ?? `chunk_${this.history.length}`,
      anomalyScore,
      isAnomalous,
      details,
      trustScore: 1 - anomalyScore,
    };
  }

  clean(texts: string[], options?: { maxAnomalyScore?: number }): string[] {
    const maxScore = options?.maxAnomalyScore ?? 0.5;
    return texts.filter((t, i) => !this.detect(t, { id: `chunk_${i}` }).isAnomalous);
  }

  report(results: AnomalyResult[]): string {
    const total = results.length;
    const anomalous = results.filter(r => r.isAnomalous).length;
    const avgTrust = results.reduce((s, r) => s + r.trustScore, 0) / Math.max(total, 1);
    return [
      `== 数据清洗报告 ==`,
      `总样本: ${total}`,
      `异常: ${anomalous} (${(anomalous / Math.max(total, 1) * 100).toFixed(1)}%)`,
      `平均信任分: ${avgTrust.toFixed(4)}`,
      anomalous > total * 0.1 ? "⚠️ 检测到大量异常数据" : "✅ 数据质量良好",
    ].join("\n");
  }
}

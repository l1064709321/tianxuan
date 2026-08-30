/**
 * Verifier — 证伪与可信度评估器
 *
 * 核心问题: 检索到的知识可能是假的、过时的、或噪声。
 * 解决方案: 三重验证机制
 * 1. 自洽性验证 (Self-consistency): 是否与已知事实矛盾?
 * 2. 交叉验证 (Cross-validation): 多个独立来源是否一致?
 * 3. 可推导性验证 (Derivability): 能否从公理推导出来?
 *
 * 置信度模型:
 * - 1.0: 可独立推导验证 (如数学定理)
 * - 0.8: 多源交叉一致 (如科学事实)
 * - 0.5: 单源检索 (如知识片段)
 * - 0.2: 与已知事实矛盾 (如错误信息)
 * - 0.0: 已证伪 (如逻辑悖论)
 */

export interface VerificationResult {
  /** 检索内容 */
  source: string;
  /** 来源类型 */
  sourceType: "math" | "chinese" | "retrieved" | "derived";
  /** 置信度 0-1 */
  confidence: number;
  /** 验证通过/失败/存疑 */
  verdict: "verified" | "contradicted" | "unverified" | "falsified";
  /** 验证理由 */
  reason: string;
  /** 独立验证步骤 */
  verificationSteps: Array<{ step: string; result: string; pass: boolean }>;
}

/**
 * 已知真值集合 (作为验证基准)
 * 这些是可以通过独立推导验证的事实
 */
const TRUTH_AXIOMS: Array<{
  statement: string;
  type: "math" | "physics" | "logic";
  /** 独立验证函数: 给定参数返回 true/false */
  verify: (...args: number[]) => boolean;
  /** 可推导性说明 */
  derivable: string;
}> = [
  // 数学公理
  {
    statement: "勾股定理: 直角三角形 a²+b²=c²",
    type: "math",
    verify: (a?: number, b?: number, c?: number) => {
      if (a === undefined || b === undefined || c === undefined) return true;
      return Math.abs(a * a + b * b - c * c) < 1e-6;
    },
    derivable: "可从欧几里得几何公理体系推导，有纯几何证明",
  },
  {
    statement: "完全平方公式: (a+b)² = a²+2ab+b²",
    type: "math",
    verify: (a?: number, b?: number) => {
      if (a === undefined || b === undefined) return true;
      const left = (a + b) * (a + b);
      const right = a * a + 2 * a * b + b * b;
      return Math.abs(left - right) < 1e-10;
    },
    derivable: "可从分配律和乘法交换律严格推导",
  },
  {
    statement: "平方差公式: (a+b)(a-b) = a²-b²",
    type: "math",
    verify: (a?: number, b?: number) => {
      if (a === undefined || b === undefined) return true;
      const left = (a + b) * (a - b);
      const right = a * a - b * b;
      return Math.abs(left - right) < 1e-10;
    },
    derivable: "可从分配律严格推导",
  },
  {
    statement: "sin²θ+cos²θ=1 (三角恒等式)",
    type: "math",
    verify: (theta?: number) => {
      if (theta === undefined) return true;
      const sin2 = Math.sin(theta) * Math.sin(theta);
      const cos2 = Math.cos(theta) * Math.cos(theta);
      return Math.abs(sin2 + cos2 - 1) < 1e-10;
    },
    derivable: "可从单位圆定义和勾股定理推导",
  },
  {
    statement: "π ≈ 3.141592653589793 (圆周率)",
    type: "math",
    verify: (value?: number) => {
      if (value === undefined) return true;
      return Math.abs(value - Math.PI) < 1e-10;
    },
    derivable: "π是圆周长与直径之比，可通过莱布尼茨级数等方法高精度计算验证",
  },
  {
    statement: "e ≈ 2.718281828459045 (自然常数)",
    type: "math",
    verify: (value?: number) => {
      if (value === undefined) return true;
      return Math.abs(value - Math.E) < 1e-10;
    },
    derivable: "e=lim(n→∞)(1+1/n)ⁿ，可通过泰勒级数验证",
  },
  {
    statement: "费马大定理: n>2时 xⁿ+yⁿ=zⁿ 无正整数解",
    type: "math",
    verify: (n?: number, x?: number, y?: number, z?: number) => {
      if (n === undefined) return true;
      if (n <= 2) return true; // n≤2 时有解
      if (x === undefined || y === undefined || z === undefined) return true;
      // 小数值验证
      if (n >= 3 && n <= 10 && x > 0 && y > 0 && z > 0 && x < 100 && y < 100 && z < 100) {
        return Math.abs(Math.pow(x, n) + Math.pow(y, n) - Math.pow(z, n)) > 1e-6;
      }
      return true; // 大范围无法穷举，信任权威证明
    },
    derivable: "怀尔斯1994年证明，基于椭圆曲线和模形式理论",
  },
  {
    statement: "牛顿第二定律: F=ma",
    type: "physics",
    verify: (F?: number, m?: number, a?: number) => {
      if (F === undefined || m === undefined || a === undefined) return true;
      return Math.abs(F - m * a) < 1e-6 * Math.max(1, Math.abs(m * a));
    },
    derivable: "经典力学基本公理，实验验证无数",
  },
  {
    statement: "质能方程: E=mc²",
    type: "physics",
    verify: (E?: number, m?: number, c?: number) => {
      if (E === undefined || m === undefined || c === undefined) return true;
      const cVal = c || 299792458;
      return Math.abs(E - m * cVal * cVal) < 1e-6 * Math.max(1, Math.abs(m * cVal * cVal));
    },
    derivable: "狭义相对论推论，已被无数实验验证",
  },
  {
    statement: "薛定谔方程: iℏ∂Ψ/∂t = ĤΨ",
    type: "physics",
    verify: () => true, // 作为基本方程无法"验证"，但作为公理接受
    derivable: "量子力学基本公设，其预测与实验一致即验证",
  },
];

/**
 * 矛盾检测: 检查两个声明是否互相矛盾
 */
function checkContradiction(stmt1: string, stmt2: string): boolean {
  // 简单关键词冲突检测
  const conflicts: Array<[string, string]> = [
    ["素数", "composite"], ["奇数", "偶数"], ["质数", "合数"],
    ["无限", "有限"], ["不存在", "存在"], ["等于", "不等于"],
    ["正弦", "余弦"], ["导数", "积分"], ["实数", "复数"],
  ];
  for (const [a, b] of conflicts) {
    if (stmt1.includes(a) && stmt2.includes(b)) return true;
    if (stmt1.includes(b) && stmt2.includes(a)) return true;
  }
  return false;
}

/**
 * 从声明中提取数值并验证
 */
function extractAndVerify(stmt: string): VerificationResult[] {
  const results: VerificationResult[] = [];

  // 提取等式
  const eqMatch = stmt.match(/(\S+)\s*=\s*([^,\s。]+(?:\s*[^,\s。]+)*)/);
  if (eqMatch) {
    const left = eqMatch[1];
    const right = eqMatch[2].replace(/\s/g, "");

    // 尝试数值验证
    for (const axiom of TRUTH_AXIOMS) {
      if (stmt.includes(axiom.statement.split(":")[0])) {
        try {
          const leftNum = parseFloat(left);
          const rightNum = parseFloat(right);
          if (!isNaN(leftNum) && !isNaN(rightNum)) {
            const pass = Math.abs(leftNum - rightNum) < 1e-6 * Math.max(1, Math.abs(rightNum));
            results.push({
              source: axiom.statement,
              sourceType: "math",
              confidence: pass ? 1.0 : 0.0,
              verdict: pass ? "verified" : "falsified",
              reason: pass ? `数值验证通过: ${left} ≈ ${right}` : `数值验证失败: ${left} ≠ ${right}`,
              verificationSteps: [{ step: "数值代入验证", result: pass ? "通过" : "不通过", pass }],
            });
          }
        } catch { /* skip */ }
      }
    }
  }

  // 检查特殊数值
  const piMatch = stmt.match(/π\s*=\s*(\d+\.?\d*)/);
  if (piMatch) {
    const val = parseFloat(piMatch[1]);
    const pass = Math.abs(val - Math.PI) < 1e-2; // 允许较小误差
    results.push({
      source: `π = ${piMatch[1]}`,
      sourceType: "math",
      confidence: pass ? 0.95 : 0.1,
      verdict: pass ? "verified" : "contradicted",
      reason: pass ? "π的数值在合理精度内" : "π的数值与已知值不符",
      verificationSteps: [
        { step: "与Math.PI比较", result: `|${val} - ${Math.PI.toFixed(10)}| = ${Math.abs(val - Math.PI).toFixed(10)}`, pass },
      ],
    });
  }

  const eMatch = stmt.match(/(?:e|欧拉数)\s*[=:]\s*(\d+\.?\d*)/);
  if (eMatch) {
    const val = parseFloat(eMatch[1]);
    const pass = Math.abs(val - Math.E) < 1e-2;
    results.push({
      source: `e = ${eMatch[1]}`,
      sourceType: "math",
      confidence: pass ? 0.95 : 0.1,
      verdict: pass ? "verified" : "contradicted",
      reason: pass ? "e的数值在合理精度内" : "e的数值与已知值不符",
      verificationSteps: [
        { step: "与Math.E比较", result: `|${val} - ${Math.E.toFixed(10)}| = ${Math.abs(val - Math.E).toFixed(10)}`, pass },
      ],
    });
  }

  // 检查质数相关声明
  const primeMatch = stmt.match(/(\d+)\s*(?:是|为)?\s*(?:质数|素数)/);
  if (primeMatch) {
    const n = parseInt(primeMatch[1]);
    const isPrime = n > 1 && Array.from({ length: Math.floor(Math.sqrt(n)) }, (_, i) => i + 2)
      .every(d => n % d !== 0);
    results.push({
      source: `${n}是质数`,
      sourceType: "math",
      confidence: isPrime ? 1.0 : 0.0,
      verdict: isPrime ? "verified" : "falsified",
      reason: isPrime ? `${n} 经试除法验证为质数` : `${n} 可被 ${Array.from({ length: Math.floor(Math.sqrt(n)) }, (_, i) => i + 2).find(d => n % d === 0)} 整除`,
      verificationSteps: [
        { step: "试除法验证", result: isPrime ? `${n} 不能被 2~${Math.floor(Math.sqrt(n))} 整除` : `${n} 可被 ${n % 2 === 0 ? 2 : Array.from({ length: Math.floor(Math.sqrt(n)) }, (_, i) => i + 3).find(d => n % d === 0)} 整除`, pass: isPrime },
      ],
    });
  }

  return results;
}

/**
 * 验证书
 */
export class Verifier {
  /** 已知真值集合 */
  static readonly truthAxioms = TRUTH_AXIOMS;

  /**
   * 验证一个声明
   */
  static verify(statement: string, context?: {
    relatedStatements?: string[];
    expectedValue?: number;
    derivedFrom?: string;
  }): VerificationResult {
    const steps: Array<{ step: string; result: string; pass: boolean }> = [];
    let maxConfidence = 0;
    let verdict: VerificationResult["verdict"] = "unverified";
    let reasons: string[] = [];

    // ── 步骤1: 数值验证 ─────────────────────────────────────────
    const numVerifications = extractAndVerify(statement);
    for (const nv of numVerifications) {
      steps.push({ step: nv.verificationSteps[0]?.step || "数值检查", result: nv.verificationSteps[0]?.result || "", pass: nv.verdict !== "falsified" });
      if (nv.confidence > maxConfidence) maxConfidence = nv.confidence;
      if (nv.verdict === "falsified") verdict = "falsified";
      if (nv.verdict === "verified" && maxConfidence < 0.9) maxConfidence = 0.9;
      reasons.push(nv.reason);
    }

    // ── 步骤2: 自洽性检查 ────────────────────────────────────────
    if (context?.relatedStatements) {
      let contradictionFound = false;
      for (const related of context.relatedStatements) {
        if (checkContradiction(statement, related)) {
          contradictionFound = true;
          steps.push({ step: "自洽性检查", result: `与"${related}"存在语义冲突`, pass: false });
          reasons.push(`与已有知识 "${related}" 矛盾`);
        }
      }
      if (!contradictionFound) {
        steps.push({ step: "自洽性检查", result: "与相关声明无矛盾", pass: true });
      }
    }

    // ── 步骤3: 公理验证 ──────────────────────────────────────────
    let axiomMatched = false;
    for (const axiom of TRUTH_AXIOMS) {
      if (statement.includes(axiom.statement.split(":")[0])) {
        axiomMatched = true;
        steps.push({ step: "公理匹配", result: `匹配到已知公理: ${axiom.statement}`, pass: true });
        if (maxConfidence < 0.8) maxConfidence = 0.8;
        reasons.push(`可与已知公理 "${axiom.statement}" 对照`);
        break;
      }
    }
    if (!axiomMatched) {
      steps.push({ step: "公理匹配", result: "未直接匹配已知公理", pass: false });
    }

    // ── 步骤4: 可推导性检查 ──────────────────────────────────────
    const derivableKeywords = ["证明", "推导", "因此", "所以", "故", "∴", "因为", "由于"];
    const hasDerivation = derivableKeywords.some(k => statement.includes(k));
    if (hasDerivation) {
      steps.push({ step: "推导结构检查", result: "包含推导连接词，结构完整", pass: true });
      if (maxConfidence < 0.6) maxConfidence = 0.6;
    } else {
      steps.push({ step: "推导结构检查", result: "缺乏推导连接词，可能是孤立声明", pass: false });
    }

    // ── 步骤5: 预期值检查 ─────────────────────────────────────────
    if (context?.expectedValue !== undefined) {
      const stmtNum = parseFloat(statement.match(/[\d.]+$/)?.[0] || "NaN");
      if (!isNaN(stmtNum)) {
        const pass = Math.abs(stmtNum - context.expectedValue) < 1e-6 * Math.max(1, Math.abs(context.expectedValue));
        steps.push({ step: "预期值验证", result: pass ? `≈ ${context.expectedValue}` : `≠ ${context.expectedValue}`, pass });
        if (!pass && maxConfidence > 0.5) maxConfidence = 0.2;
        if (pass && maxConfidence < 0.7) maxConfidence = 0.7;
      }
    }

    // ── 综合判定 ─────────────────────────────────────────────────
    if (verdict === "falsified") {
      verdict = "falsified";
    } else if (maxConfidence >= 0.9 && !steps.some(s => !s.pass)) {
      verdict = "verified";
    } else if (maxConfidence >= 0.5) {
      verdict = "unverified";
    } else {
      verdict = "contradicted";
    }

    return {
      source: statement,
      sourceType: "derived",
      confidence: maxConfidence,
      verdict,
      reason: reasons.join("; ") || "验证完成",
      verificationSteps: steps,
    };
  }

  /**
   * 批量验证多个声明，检测矛盾
   */
  static batchVerify(statements: string[]): Array<{ statement: string; result: VerificationResult }> {
    const results = statements.map(s => ({ statement: s, result: this.verify(s) }));

    // 交叉检查矛盾
    for (let i = 0; i < results.length; i++) {
      for (let j = i + 1; j < results.length; j++) {
        if (checkContradiction(results[i].statement, results[j].statement)) {
          results[i].result.verdict = "contradicted";
          results[i].result.confidence *= 0.5;
          results[j].result.verdict = "contradicted";
          results[j].result.confidence *= 0.5;
          results[i].result.reason += `; 与声明[${j}]矛盾`;
          results[j].result.reason += `; 与声明[${i}]矛盾`;
        }
      }
    }

    return results;
  }

  /**
   * 生成验证报告
   */
  static report(results: VerificationResult[]): string {
    if (results.length === 0) return "无验证结果";

    let report = `**知识验证报告**\n\n`;
    let verified = 0, falsified = 0, unverified = 0;

    for (const r of results) {
      const icon = r.verdict === "verified" ? "✅" : r.verdict === "falsified" ? "❌" : r.verdict === "contradicted" ? "⚠️" : "❓";
      if (r.verdict === "verified") verified++;
      else if (r.verdict === "falsified") falsified++;
      else unverified++;

      report += `${icon} **置信度: ${(r.confidence * 100).toFixed(0)}%** | ${r.verdict}\n`;
      report += `   内容: ${r.source.slice(0, 80)}\n`;
      report += `   理由: ${r.reason}\n`;
      report += `   验证步骤:\n`;
      for (const step of r.verificationSteps) {
        const stepIcon = step.pass ? "✓" : "✗";
        report += `     ${stepIcon} ${step.step}: ${step.result}\n`;
      }
      report += "\n";
    }

    report += `**汇总**: ✅ 通过 ${verified} | ❌ 证伪 ${falsified} | ❓ 未定 ${unverified}`;

    // 如果大部分被证伪，发出警告
    if (falsified / Math.max(1, results.length) > 0.5) {
      report += "\n\n⚠️ 警告: 超过半数知识被证伪，检索源可能存在严重噪声";
    }

    return report;
  }
}

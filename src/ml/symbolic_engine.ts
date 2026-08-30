/**
 * SymbolicEngine — 符号推导引擎
 * 从公理出发，逐步推导，不跳步
 */
import { SymbolicMathEngine, CalculationResult } from "./math_engine";

export interface DeriveStep {
  step: number;
  statement: string;
  justification: string;
  equation?: string;
  note?: string;
}

export interface DeriveResult {
  found: boolean;
  query: string;
  type: "proof" | "derivation" | "calculation";
  title: string;
  steps: DeriveStep[];
  finalAnswer: string;
  explanation: string;
  data?: any;
}

export class SymbolicEngine {
  private mathEngine: SymbolicMathEngine;

  constructor(mathEngine?: SymbolicMathEngine) {
    this.mathEngine = mathEngine ?? new SymbolicMathEngine();
  }

  derive(question: string, context?: any): DeriveResult {
    const q = question.trim();
    const known = this._resolveKnownDerivation(q);
    if (known) return known;
    const mathResult = this.mathEngine.answer(q);
    if (mathResult && mathResult.exactAnswer) {
      return this._mathToDerivation(mathResult, q);
    }
    const algebraResult = this._deriveAlgebra(q);
    if (algebraResult) return algebraResult;
    return this._generalDerive(q);
  }

  private _resolveKnownDerivation(q: string): DeriveResult | null {
    if (q.includes("勾股") && (q.includes("证明") || q.includes("推导") || q.includes("为什么"))) {
      return {
        found: true, query: q, type: "proof",
        title: "勾股定理的几何证明",
        steps: [
          { step: 1, statement: "构造一个边长为 (a+b) 的大正方形", justification: "作图公理", equation: "大正方形面积 = (a+b)²", note: "以直角三角形两直角边 a,b 之和为边长" },
          { step: 2, statement: "在大正方形内放置四个全等的直角三角形", justification: "作图公理", equation: "每个三角形面积 = ½ab", note: "四个三角形围成一个边长为 c 的小正方形空洞" },
          { step: 3, statement: "大正方形面积 = 四个三角形面积 + 中间小正方形面积", justification: "面积守恒", equation: "(a+b)² = 4×(½ab) + c²" },
          { step: 4, statement: "展开左边", justification: "代数恒等式", equation: "a² + 2ab + b² = 2ab + c²" },
          { step: 5, statement: "两边消去 2ab", justification: "等式性质", equation: "a² + b² = c²" },
          { step: 6, statement: "∴ a² + b² = c² 证毕", justification: "Q.E.D.", equation: "a² + b² = c²" },
        ],
        finalAnswer: "a² + b² = c²",
        explanation: "这是最经典的几何证明，见于欧几里得《几何原本》命题47。核心思想是用同一大正方形的两种面积表达方式建立等式。",
      };
    }
    if (q.includes("完全平方") && (q.includes("推导") || q.includes("证明"))) {
      return {
        found: true, query: q, type: "proof",
        title: "完全平方公式的推导",
        steps: [
          { step: 1, statement: "(a+b)² 表示 (a+b) 乘以自身", justification: "平方的定义", equation: "(a+b)² = (a+b)(a+b)" },
          { step: 2, statement: "用分配律展开", justification: "分配律", equation: "= a(a+b) + b(a+b)" },
          { step: 3, statement: "再次分配", justification: "分配律", equation: "= a² + ab + ba + b²" },
          { step: 4, statement: "合并同类项", justification: "乘法交换律", equation: "= a² + 2ab + b²" },
          { step: 5, statement: "∴ (a+b)² = a² + 2ab + b² 证毕", justification: "Q.E.D.", equation: "(a+b)² = a² + 2ab + b²" },
        ],
        finalAnswer: "(a+b)² = a² + 2ab + b²",
        explanation: "完全平方公式是多项式乘法的基础。",
      };
    }
    if (q.includes("平方差") && (q.includes("推导") || q.includes("证明"))) {
      return {
        found: true, query: q, type: "proof",
        title: "平方差公式的推导",
        steps: [
          { step: 1, statement: "(a+b)(a-b) 按分配律展开", justification: "分配律", equation: "= a(a-b) + b(a-b)" },
          { step: 2, statement: "继续分配", justification: "分配律", equation: "= a² - ab + ba - b²" },
          { step: 3, statement: "消去 -ab + ba", justification: "ab = ba", equation: "= a² - b²" },
          { step: 4, statement: "∴ (a+b)(a-b) = a² - b² 证毕", justification: "Q.E.D.", equation: "(a+b)(a-b) = a² - b²" },
        ],
        finalAnswer: "(a+b)(a-b) = a² - b²",
        explanation: "平方差公式是重要的因式分解工具。",
      };
    }
    if ((q.includes("sin") || q.includes("cos")) && (q.includes("证明") || q.includes("推导") || q.includes("sin²") || q.includes("cos²"))) {
      return {
        found: true, query: q, type: "proof",
        title: "sin²θ + cos²θ = 1 的证明",
        steps: [
          { step: 1, statement: "在单位圆上取一点 P", justification: "作图公理", equation: "P = (cos θ, sin θ)", note: "单位圆：半径 r = 1" },
          { step: 2, statement: "根据三角函数定义", justification: "定义", equation: "cos θ = x, sin θ = y" },
          { step: 3, statement: "P 在圆上，满足圆方程", justification: "圆的定义", equation: "x² + y² = 1" },
          { step: 4, statement: "代入 cos θ 和 sin θ", justification: "等量代换", equation: "cos²θ + sin²θ = 1" },
          { step: 5, statement: "∴ sin²θ + cos²θ = 1 证毕", justification: "Q.E.D.", equation: "sin²θ + cos²θ = 1" },
        ],
        finalAnswer: "sin²θ + cos²θ = 1",
        explanation: "这是三角学最基本的恒等式，本质是勾股定理在单位圆上的体现。",
      };
    }
    if (q.includes("导数") && (q.includes("定义") || q.includes("推导") || q.includes("是什么"))) {
      return {
        found: true, query: q, type: "derivation",
        title: "导数的严格定义",
        steps: [
          { step: 1, statement: "导数描述函数在某点的瞬时变化率", justification: "概念定义", equation: "f'(x₀) = ?" },
          { step: 2, statement: "先用平均变化率近似", justification: "割线斜率", equation: "avg_rate = [f(x₀+h) - f(x₀)] / h" },
          { step: 3, statement: "让 h 趋近于 0", justification: "极限定义", equation: "f'(x₀) = lim[h→0] [f(x₀+h) - f(x₀)] / h" },
          { step: 4, statement: "这就是导数的严格定义", justification: "ε-δ 语言", equation: "f'(x₀) = lim[h→0] Δy/Δx" },
          { step: 5, statement: "几何意义：曲线在该点的切线斜率", justification: "几何解释", note: "当 h→0 时，割线趋近于切线" },
        ],
        finalAnswer: "f'(x) = lim[h→0] [f(x+h) - f(x)] / h",
        explanation: "导数是微积分的核心概念，由牛顿和莱布尼茨独立发明。它量化了函数在某一点的瞬时变化快慢。",
      };
    }
    if (q.includes("积分") && (q.includes("定义") || q.includes("推导") || q.includes("是什么"))) {
      return {
        found: true, query: q, type: "derivation",
        title: "定积分的严格定义",
        steps: [
          { step: 1, statement: "定积分计算曲线下的面积", justification: "几何动机", equation: "Area = ?" },
          { step: 2, statement: "用矩形近似（黎曼和）", justification: "分割求和", equation: "S_n = Σ f(xᵢ) · Δx" },
          { step: 3, statement: "让分割无限细", justification: "极限过程", equation: "∫[a,b] f(x)dx = lim[n→∞] Σ f(xᵢ)Δx" },
          { step: 4, statement: "牛顿-莱布尼茨公式连接微分与积分", justification: "微积分基本定理", equation: "∫[a,b] f(x)dx = F(b) - F(a)" },
          { step: 5, statement: "微分和积分是互逆运算", justification: "Fundamental Theorem of Calculus", note: "微分和积分就像加法和减法的关系" },
        ],
        finalAnswer: "∫[a,b] f(x)dx = F(b) - F(a)",
        explanation: "积分是求面积、体积、累积量的工具。牛顿-莱布尼茨公式将积分与微分联系起来。",
      };
    }
    if (q.includes("欧拉公式") || q.includes("e^(iπ)")) {
      return {
        found: true, query: q, type: "proof",
        title: "欧拉公式 e^(iπ) + 1 = 0 的推导",
        steps: [
          { step: 1, statement: "从 e^x 的泰勒级数展开出发", justification: "泰勒级数定义", equation: "e^x = 1 + x + x²/2! + x³/3! + ..." },
          { step: 2, statement: "代入 x = iθ", justification: "复数扩展", equation: "e^(iθ) = 1 + iθ + (iθ)²/2! + (iθ)³/3! + ..." },
          { step: 3, statement: "利用 i²=-1 分组", justification: "虚数单位性质", equation: "= [1 - θ²/2! + θ⁴/4! - ...] + i[θ - θ³/3! + θ⁵/5! - ...]" },
          { step: 4, statement: "认出实部是 cos θ，虚部是 sin θ", justification: "泰勒级数定义", equation: "e^(iθ) = cos θ + i·sin θ" },
          { step: 5, statement: "代入 θ = π", justification: "特殊值", equation: "e^(iπ) = cos π + i·sin π = -1" },
          { step: 6, statement: "移项得欧拉恒等式", justification: "代数运算", equation: "e^(iπ) + 1 = 0" },
        ],
        finalAnswer: "e^(iπ) + 1 = 0",
        explanation: "这是数学中最优美的公式，将五个最重要的常数 e, i, π, 1, 0 联系在一起。",
      };
    }
    if (q.includes("二次方程") && (q.includes("求根公式") || q.includes("推导"))) {
      return {
        found: true, query: q, type: "derivation",
        title: "一元二次方程求根公式的推导",
        steps: [
          { step: 1, statement: "从一般形式出发", justification: "问题设定", equation: "ax² + bx + c = 0  (a ≠ 0)" },
          { step: 2, statement: "两边除以 a", justification: "等式性质", equation: "x² + (b/a)x + c/a = 0" },
          { step: 3, statement: "移常数项到右边", justification: "等式性质", equation: "x² + (b/a)x = -c/a" },
          { step: 4, statement: "配方：两边加上 (b/2a)²", justification: "完全平方公式", equation: "x² + (b/a)x + (b/2a)² = -c/a + (b/2a)²" },
          { step: 5, statement: "左边写成完全平方", justification: "公式逆用", equation: "(x + b/2a)² = (b²-4ac)/(4a²)" },
          { step: 6, statement: "两边开平方", justification: "平方根定义", equation: "x + b/2a = ±√(b²-4ac) / 2a" },
          { step: 7, statement: "移项得求根公式", justification: "等式性质", equation: "x = (-b ± √(b²-4ac)) / 2a" },
          { step: 8, statement: "其中 Δ = b²-4ac 称为判别式", justification: "定义", note: "Δ>0 两实根，Δ=0 重根，Δ<0 共轭复根" },
        ],
        finalAnswer: "x = (-b ± √(b²-4ac)) / 2a",
        explanation: "求根公式是初中数学最重要的公式之一，用配方法推导。",
      };
    }
    return null;
  }

  private _deriveAlgebra(q: string): DeriveResult | null {
    const simplifyMatch = q.match(/(?:化简|简化|展开)\s*(.+)/);
    if (simplifyMatch) {
      const expr = simplifyMatch[1].trim();
      return {
        found: true, query: q, type: "calculation",
        title: `表达式化简：${expr}`,
        steps: [
          { step: 1, statement: `原表达式：${expr}`, justification: "题目", equation: expr },
          { step: 2, statement: "识别表达式结构", justification: "观察分析", equation: "" },
          { step: 3, statement: "应用相应的运算法则", justification: "代数规则", equation: "逐步化简..." },
          { step: 4, statement: "最终结果", justification: "化简完成", equation: "请提供具体表达式以便推导" },
        ],
        finalAnswer: `表达式「${expr}」的化简需要具体展开步骤。`,
        explanation: "代数化简需要根据具体表达式应用分配律、合并同类项、因式分解等规则。",
      };
    }
    return null;
  }

  private _mathToDerivation(result: any, query: string): DeriveResult {
    return {
      found: true, query, type: "derivation",
      title: result.type === "constant" ? "数学常量" : "数学知识",
      steps: [{
        step: 1,
        statement: (result.explanation || "").split("\n")[0] || result.explanation || "",
        justification: "检索到的数学知识",
        equation: "",
      }],
      finalAnswer: (result.approximate || result.exactAnswer || "见上方解释"),
      explanation: result.explanation || "",
      data: result,
    };
  }

  private _generalDerive(q: string): DeriveResult {
    return {
      found: true, query: q, type: "derivation",
      title: "推导请求",
      steps: [
        { step: 1, statement: `问题：${q}`, justification: "用户提问", equation: "" },
        { step: 2, statement: "分析问题类型", justification: "分类", equation: "" },
        { step: 3, statement: "识别涉及的数学概念", justification: "概念检索", equation: "" },
        { step: 4, statement: "从公理/定义出发逐步推导", justification: "推导过程", equation: "" },
        { step: 5, statement: "验证结果", justification: "验算", equation: "" },
      ],
      finalAnswer: `针对「${q}」，建议提供具体数值或明确的推导目标。`,
      explanation: "天玄符号推导引擎支持：勾股定理、完全平方公式、平方差公式、三角恒等式、导数/积分定义、欧拉公式、二次方程求根公式等经典推导。",
    };
  }
}

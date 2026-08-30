/**
 * SymbolicMathEngine — 符号数学推理引擎
 *
 * 不追求数值近似, 追求概念理解:
 *  - 常量: 定义 + 性质 + 已知展开式 (pi, e, phi, sqrt2 等)
 *  - 定理: 陈述 + 适用条件 + 推广/逆命题
 *  - 输出不是单一数字, 而是"理解包"
 */

export interface ConstantInfo {
  name: string;
  symbol: string;
  definition: string;
  approximation: string;
  properties: string[];
  appearances: string[];
  related: string[];
}

export interface TheoremInfo {
  name: string;
  symbol: string;
  statement: string;
  conditions: string[];
  proof_idea: string;
  generalizations: string[];
  counterexamples: string[];
  applications: string[];
  related: string[];
}

export interface CalculationResult {
  question: string;
  exactAnswer: string;
  approximate?: string;
  derivation: string[];
  explanation: string;
  relatedInfo?: { type: "constant" | "theorem"; key: string };
}

const CONSTANT_DB: Record<string, ConstantInfo> = {
  pi: {
    name: "圆周率",
    symbol: "π",
    definition: "圆的周长与直径之比, pi = C/d = 2*pi*r/r = 2A/r^2, 其中 A 为圆面积",
    approximation: "3.14159265358979323846264338327950288419716939937510...",
    properties: [
      "无理数: 不能表示为两个整数之比 (Lambert, 1768)",
      "超越数: 不是任何整系数多项式方程的根 (Lindemann, 1882)",
      "因此圆方问题尺规作图不可能",
      "欧拉恒等式: e^(i*pi) + 1 = 0, 连接五个基本常数",
      "出现在概率中: 布丰投针问题 P(交叉) = 2l/(pi*d)",
      "傅里叶变换: integral_{-inf}^{inf} e^(-ix*xi) dxi = 2*pi*delta(x)",
    ],
    appearances: [
      "圆面积 S = pi*r^2, 圆周长 C = 2*pi*r",
      "球体积 V = (4/3)*pi*r^3, 球表面积 S = 4*pi*r^2",
      "正态分布密度: f(x) = (1/(sigma*sqrt(2*pi))) * exp(-(x-mu)^2/(2*sigma^2))",
      "欧拉公式: e^(i*theta) = cos(theta) + i*sin(theta)",
      "黎曼 zeta 函数: zeta(2) = pi^2/6 (巴塞尔问题)",
    ],
    related: ["e", "i", "phi", "欧拉恒等式", "傅里叶分析", "复分析"],
  },
  e: {
    name: "自然常数",
    symbol: "e",
    definition: "e = lim(n->inf)(1+1/n)^n = sum(n=0..inf) 1/n! ≈ 2.71828..., 是指数函数 d/dx[e^x]=e^x 的底",
    approximation: "2.71828182845904523536028747135266249775724709369995...",
    properties: [
      "无理数 (Euler, 1737)",
      "超越数 (Hermite, 1873)",
      "e^x 是唯一满足 f'=f, f(0)=1 的函数",
      "ln(e) = 1, log_e 是以 e 为底的对数",
      "e^(i*pi) = -1 (欧拉恒等式)",
    ],
    appearances: [
      "指数增长/衰减: N(t) = N_0 * e^(rt)",
      "复利极限: lim(n->inf)(1+r/n)^(nt) = e^(rt)",
      "正态分布, 泊松分布, 指数分布",
      "双曲函数: cosh(x) = (e^x + e^(-x))/2",
    ],
    related: ["pi", "i", "欧拉恒等式", "微积分", "微分方程"],
  },
  phi: {
    name: "黄金比例",
    symbol: "phi",
    definition: "phi = (1+sqrt(5))/2, 满足 phi^2 = phi+1, 即 phi/1 = 1/(phi-1)",
    approximation: "1.61803398874989484820458683436563811772030917980576...",
    properties: [
      "无理数 (古希腊已知)",
      "phi = 1 + 1/phi, 连分数 [1;1,1,1,...] 无限循环",
      "phi^2 = phi+1, phi^3 = 2*phi+1",
      "1(phi) = phi - 1 ≈ 0.618",
      "与斐波那契数列关系: F_n / F_(n-1) -> phi",
    ],
    appearances: [
      "正五边形对角线/边长 = phi",
      "斐波那契螺旋, 向日葵种子排列",
      "黄金矩形, 美学中的经典比例",
    ],
    related: ["sqrt5", "斐波那契数列", "连分数", "正五边形"],
  },
  sqrt2: {
    name: "根号2",
    symbol: "sqrt(2)",
    definition: "sqrt(2) 是满足 x^2 = 2 的正实数, 即单位正方形对角线长度",
    approximation: "1.41421356237309504880168872420969807856967187537694...",
    properties: [
      "无理数 (毕达哥拉斯学派, 希帕索斯发现)",
      "最早被证明的无理数之一",
      "sqrt(2) 不是任何整系数二次方程的有理根",
    ],
    appearances: [
      "勾股定理特例: 直角边为1的等腰直角三角形斜边",
      "单位正方形对角线长",
    ],
    related: ["勾股定理", "无理数", "毕达哥拉斯"],
  },
};

const THEOREM_DB: Record<string, TheoremInfo> = {
  pythagoras: {
    name: "勾股定理",
    symbol: "a^2 + b^2 = c^2",
    statement: "在欧几里得平面中, 任意直角三角形的两条直角边的平方和等于斜边的平方",
    conditions: ["三角形是直角三角形", "处于欧几里得平面 (平坦空间)", "二维或更高维欧氏空间"],
    proof_idea: "经典证法: 相似三角形法、面积拼接法 (赵爽弦图)、向量法 (|a+b|^2=|a|^2+|b|^2 当 a perp b)。现代观点: 勾股定理等价于内积空间的范数平行四边形恒等式",
    generalizations: [
      "余弦定理: c^2 = a^2+b^2-2ab*cos(C) (一般三角形的推广)",
      "勾股定理在黎曼流形中不再成立 (测地线偏离)",
      "高维: 勾股定理对正交向量组的范数仍然成立",
    ],
    counterexamples: [
      "球面三角形: 三面角之和 > 180度, 勾股定理不适用 (非欧几何)",
      "双曲几何: 直角三角形斜边 > sqrt(a^2+b^2)",
      "曼哈顿距离: d = |x1-x2| + |y1-y2|, 勾股定理形式不成立",
    ],
    applications: [
      "计算两点间欧氏距离: d = sqrt((x2-x1)^2+(y2-y1)^2)",
      "向量内积: a.b = |a||b|cos(theta), 勾股定理是 theta=90度的特例",
      "Parseval 恒等式是勾股定理在无穷维的推广",
    ],
    related: ["余弦定理", "欧几里得几何", "黎曼几何", "内积空间"],
  },
  euler_identity: {
    name: "欧拉恒等式",
    symbol: "e^(i*pi) + 1 = 0",
    statement: "自然常数 e, 虚数单位 i, 圆周率 pi, 加法单位元 0, 乘法单位元 1 通过基本运算联系在一起",
    conditions: ["复指数函数 e^z 的解析延拓成立", "复平面上极坐标表示有效"],
    proof_idea: "由欧拉公式 e^(i*theta) = cos(theta) + i*sin(theta), 代入 theta = pi: e^(i*pi) = cos(pi) + i*sin(pi) = -1 + 0*i = -1, 移项即得",
    generalizations: ["欧拉公式: e^(i*theta) = cos(theta) + i*sin(theta)", "复对数: log(z) = ln|z| + i(arg(z) + 2*k*pi), 多值性"],
    counterexamples: [],
    applications: [
      "交流电路分析: 阻抗的复数表示",
      "傅里叶变换的基础",
      "量子力学: 波函数的相位因子 e^(i*theta)",
    ],
    related: ["欧拉公式", "复分析", "傅里叶分析", "pi", "e", "i"],
  },
  schrodinger: {
    name: "薛定谔方程",
    symbol: "i*hbar*dPsi/dt = H*Psi",
    statement: "量子力学的基本运动方程, 描述量子态 (波函数) 随时间的演化",
    conditions: ["非相对论性量子力学 (粒子速度 << c)", "势场不显含时间时可用分离变量法", "单粒子近似"],
    proof_idea: "薛定谔方程是量子力学的公设之一, 不能从更基本的原理推导。思路: 从德布罗意物质波关系 E=p^2/(2m)+V 和平面波 Psi=A*e^(i(kx-omega*t)) 出发, 替换 E->i*hbar*d/dt, p->-i*hbar*grad, 得到含时薛定谔方程",
    generalizations: [
      "含势场的含时薛定谔方程: i*hbar*dPsi/dt = [-hbar^2/(2m)*laplacian + V(r,t)]*Psi",
      "定态薛定谔方程: H*psi = E*psi (分离变量, V 不依赖 t)",
      "狄拉克方程: 相对论性自旋 1/2 粒子的推广",
    ],
    counterexamples: [
      "光速粒子 (光子): 需用量子电动力学",
      "强引力场: 需结合广义相对论",
      "多体相互作用: 严格求解困难, 需近似方法",
    ],
    applications: [
      "氢原子能级计算 (玻尔模型的量子力学解释)",
      "谐振子能级 E_n = (n+1/2)*hbar*omega",
      "势垒隧穿效应 (扫描隧道显微镜原理)",
    ],
    related: ["海森堡矩阵力学", "费曼路径积分", "算子理论", "希尔伯特空间"],
  },
};

export class SymbolicMathEngine {
  answer(question: string): CalculationResult {
    const q = question.trim().toLowerCase();

    for (const [key, info] of Object.entries(CONSTANT_DB)) {
      if (q.includes(info.symbol) || q.includes(info.name) || q.includes(key)) {
        return this._buildResult(question, info);
      }
    }

    for (const [key, info] of Object.entries(THEOREM_DB)) {
      if (q.includes(info.name) || q.includes(key) || q.includes(info.symbol)) {
        return this._buildTheoremResult(question, info);
      }
    }

    const numMatch = q.match(/(\d+\.?\d*)\s*[\*\/\+\-\^]/);
    if (numMatch) {
      return this._handleExpression(q);
    }

    if (q.includes("无理") || q.includes("超越") || q.includes("有理")) {
      return this._answerNumberTypeQuestion(q);
    }

    return this._buildDefaultResult(question);
  }

  private _buildResult(q: string, info: ConstantInfo): CalculationResult {
    return {
      question: q,
      exactAnswer: info.symbol,
      approximate: info.approximation + "...(无限不循环, 无法完整写出)",
      derivation: [
        "定义: " + info.definition,
        ...info.properties.map((p, i) => (i + 1) + ". " + p),
        "数值展开: " + info.approximation.slice(0, 60) + "...(无限位数, 无周期)",
      ],
      explanation: "你问「" + q + "」—— 这不是一个能'算出有限位就结束'的问题。" +
        info.symbol + " 是一个无限不循环的数, 任何有限近似都只是近似。" +
        "它之所以重要, 不是因为它的数值, 而是因为它出现在" + info.appearances.length + "个不同领域的基础公式中:" +
        info.appearances.slice(0, 3).map((a, i) => (i + 1) + ". " + a).join("\n"),
      relatedInfo: { type: "constant", key: Object.entries(CONSTANT_DB).find(([, v]) => v.symbol === info.symbol)?.[0] ?? "" },
    };
  }

  private _buildTheoremResult(q: string, info: TheoremInfo): CalculationResult {
    return {
      question: q,
      exactAnswer: info.symbol,
      derivation: [
        "定理: " + info.statement,
        "适用条件: " + info.conditions.join(", "),
        "证明思路: " + info.proof_idea,
      ],
      explanation: "「" + info.name + "」不是一个公式可以简单套用的工具, 而是一个有适用边界的数学结构。" +
        "它的核心不是\"" + info.symbol + "\"这行字, 而是理解:" +
        "\n  -> 为什么在欧氏空间成立 (内积结构的必然结果)" +
        "\n  -> 为什么在非欧几何失效 (空间弯曲了)" +
        "\n  -> 它的推广是什么 (" + info.generalizations.slice(0, 2).join("; ") + ")" +
        (info.counterexamples.length > 0 ? "\n\n反例: " + info.counterexamples.slice(0, 2).join("; ") : ""),
      relatedInfo: { type: "theorem", key: Object.entries(THEOREM_DB).find(([, v]) => v.name === info.name)?.[0] ?? "" },
    };
  }

  private _handleExpression(q: string): CalculationResult {
    try {
      // 安全数学表达式求值: 只用数字、运算符、括号、空格, 不做代码注入
      const cleaned = q.replace(/[^0-9+\-*/^.()\s*pi]/g, "").trim();
      if (!cleaned) return this._buildDefaultResult(q);
      // 用递归下降解析器求值, 避免 Function/eval
      const result = this._safeEval(cleaned);
      if (typeof result === "number" && isFinite(result)) {
        return {
          question: q,
          exactAnswer: q,
          approximate: result.toFixed(10),
          derivation: ["代入数值计算: " + q + " = " + result.toFixed(10)],
          explanation: "数值结果是 " + result.toFixed(10) + ", 但注意: 如果表达式中包含 pi, 这个结果是近似的。" +
            "pi 本身是无限不循环的, 任何有限十进制表示都只是近似值。",
        };
      }
    } catch (_) { /* fall through */ }
    return this._buildDefaultResult(q);
  }

  /** 递归下降安全求值: 仅支持 + - * / ^ () 和数字/pi */
  private _safeEval(expr: string): number {
    let pos = 0;
    const source = expr.replace(/\s+/g, "");

    const peek = (): string => source[pos] ?? "";
    const consume = (): string => source[pos++];

    const parseNumber = (): number => {
      let numStr = "";
      while (pos < source.length && (/[0-9.]/.test(source[pos]))) numStr += consume();
      if (!numStr) throw new Error("unexpected char at " + pos);
      const v = parseFloat(numStr);
      if (isNaN(v)) throw new Error("invalid number: " + numStr);
      return v;
    };

    const parseConstant = (): number => {
      if (source.slice(pos, pos + 2) === "pi") { pos += 2; return Math.PI; }
      return NaN;
    };

    const parseTerm = (): number => {
      let left: number;
      if (peek() === "(") {
        consume(); // '('
        left = parseExpr();
        if (consume() !== ")") throw new Error("missing )");
      } else if (/^[0-9.]/.test(peek())) {
        left = parseNumber();
      } else {
        const c = parseConstant();
        if (isNaN(c)) throw new Error("unexpected: " + peek());
        left = c;
      }
      while (pos < source.length && (source[pos] === "*" || source[pos] === "/" || source[pos] === "^")) {
        const op = consume();
        const right = parseTerm();
        if (op === "*") left *= right;
        else if (op === "/") { if (right === 0) throw new Error("div by zero"); left /= right; }
        else if (op === "^") left = Math.pow(left, right);
      }
      return left;
    };

    const parseExpr = (): number => {
      let left = parseTerm();
      while (pos < source.length && (source[pos] === "+" || source[pos] === "-")) {
        const op = consume();
        const right = parseTerm();
        left = op === "+" ? left + right : left - right;
      }
      return left;
    };

    const result = parseExpr();
    if (pos !== source.length) throw new Error("trailing chars at " + pos);
    return result;
  }

  private _answerNumberTypeQuestion(q: string): CalculationResult {
    if (q.includes("pi") || q.includes("圆周率")) return this._buildResult(q, CONSTANT_DB.pi);
    if (q.includes("e") && (q.includes("自然") || q.includes("常数"))) return this._buildResult(q, CONSTANT_DB.e);
    return this._buildDefaultResult(q);
  }

  private _buildDefaultResult(q: string): CalculationResult {
    return {
      question: q,
      exactAnswer: "未知",
      derivation: ["未能在知识库中找到匹配的数学概念或定理"],
      explanation: "这个问题目前没有匹配到的精确数学知识。" +
        "如果你有具体的数学问题 (常量/定理/方程), 可以用更明确的术语提问," +
        "例如: " + String.fromCharCode(8216) + "pi 等于多少" + String.fromCharCode(8217) + ", " + String.fromCharCode(8216) + "勾股定理是什么" + String.fromCharCode(8217) + ", " + String.fromCharCode(8216) + "薛定谔方程怎么解" + String.fromCharCode(8217) + ".",
    };
  }
}

/** 别名，供 reasoner.ts / symbolic_engine.ts 使用 */
export const MathEngine = SymbolicMathEngine;


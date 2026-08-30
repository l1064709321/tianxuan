/**
 * MathCorpus — 数学表达式与定理语料生成器
 *
 * 生成内容:
 * 1. 算术表达式: a + b = c, 勾股定理, 二次方程求根公式等
 * 2. 代数恒等式: (a+b)², 平方差, 立方和差等
 * 3. 几何定理: 三角形内角和, 圆面积, 体积公式等
 * 4. 数论基础: 质数判定, 整除规则, 同余等
 * 5. 微积分概念: 导数定义, 积分公式, 极限等
 *
 * 每条记录带结构化元数据 (类型/难度/关键词), 便于向量库分类检索
 */
import { mulberry32 } from "./rng";

export interface MathEntry {
  expression: string;       // 数学表达式
  description: string;      // 中文描述
  category: string;         // 算术/代数/几何/数论/微积分/概率
  difficulty: number;       // 1-5
  keywords: string[];       // 检索关键词
}

export class MathCorpus {
  private rng: () => number;
  private seed: number;

  constructor(seed = 42) {
    this.seed = seed;
    this.rng = mulberry32(seed);
  }

  private pick<T>(arr: T[]): T {
    return arr[Math.floor(this.rng() * arr.length)];
  }

  private randInt(min: number, max: number): number {
    return min + Math.floor(this.rng() * (max - min + 1));
  }

  /** 生成单条数学知识 */
  generateOne(): MathEntry {
    const kind = Math.floor(this.rng() * 8);
    switch (kind) {
      case 0: return this._genArithmetic();
      case 1: return this._genAlgebra();
      case 2: return this._genGeometry();
      case 3: return this._genNumberTheory();
      case 4: return this._genCalculus();
      case 5: return this._genProbability();
      case 6: return this._genTrigonometry();
      case 7: return this._genMatrix();
      default: return this._genArithmetic();
    }
  }

  private _genArithmetic(): MathEntry {
    const a = this.randInt(2, 99), b = this.randInt(2, 99);
    const ops = ["加", "减", "乘", "除"];
    const op = this.pick(ops);
    let expr = "", desc = "", kw: string[] = [];
    switch (op) {
      case "加":
        expr = `${a} + ${b} = ${a + b}`;
        desc = `${a} 加 ${b} 等于 ${a + b},加法交换律:a+b=b+a`;
        kw = ["加法", "整数", "交换律"];
        break;
      case "减":
        expr = `${a} - ${b} = ${a - b}`;
        desc = `${a} 减 ${b} 等于 ${a - b},减法是加法的逆运算`;
        kw = ["减法", "整数", "逆运算"];
        break;
      case "乘":
        expr = `${a} × ${b} = ${a * b}`;
        desc = `${a} 乘以 ${b} 等于 ${a * b},乘法是重复加法`;
        kw = ["乘法", "整数", "积"];
        break;
      case "除": {
        const q = this.randInt(2, 20), r = this.randInt(0, b - 1);
        expr = `${q * b + r} ÷ ${b} = ${q}...${r}`;
        desc = `${q * b + r} 除以 ${b} 商 ${q} 余 ${r},带余除法`;
        kw = ["除法", "余数", "带余除法"];
        break;
      }
    }
    return { expression: expr, description: desc, category: "算术", difficulty: 1, keywords: kw };
  }

  private _genAlgebra(): MathEntry {
    const formulas = [
      { expr: "(a+b)² = a²+2ab+b²", desc: "完全平方公式:两数和的平方等于平方和加两倍积", kw: ["完全平方", "代数恒等式", "展开"] },
      { expr: "(a-b)² = a²-2ab+b²", desc: "完全平方差公式:两数差的平方", kw: ["完全平方", "差", "恒等式"] },
      { expr: "a²-b² = (a+b)(a-b)", desc: "平方差公式:两平方之差等于和乘差", kw: ["平方差", "因式分解", "恒等式"] },
      { expr: "a³+b³ = (a+b)(a²-ab+b²)", desc: "立方和公式", kw: ["立方和", "因式分解", "公式"] },
      { expr: "a³-b³ = (a-b)(a²+ab+b²)", desc: "立方差公式", kw: ["立方差", "因式分解", "公式"] },
      { expr: "ax²+bx+c=0 → x=(-b±√(b²-4ac))/2a", desc: "一元二次方程求根公式,判别式Δ=b²-4ac", kw: ["二次方程", "求根公式", "判别式"] },
      { expr: "S_n = n(a₁+aₙ)/2", desc: "等差数列求和公式:首项加末项乘项数除以二", kw: ["等差数列", "求和", "数列"] },
      { expr: "S_n = a₁(1-qⁿ)/(1-q)", desc: "等比数列求和公式", kw: ["等比数列", "求和", "数列"] },
      { expr: "log(ab) = log a + log b", desc: "对数乘法法则:积的对数等于对数之和", kw: ["对数", "运算律", "乘法"] },
      { expr: "log(a/b) = log a - log b", desc: "对数除法法则", kw: ["对数", "运算律", "除法"] },
    ];
    const f = this.pick(formulas);
    return { expression: f.expr, description: f.desc, category: "代数", difficulty: this.randInt(2, 3), keywords: f.kw };
  }

  private _genGeometry(): MathEntry {
    const formulas = [
      { expr: "S = πr²", desc: "圆面积公式:半径平方乘圆周率", kw: ["圆", "面积", "圆周率"] },
      { expr: "C = 2πr", desc: "圆周长公式:直径乘圆周率", kw: ["圆", "周长", "圆周率"] },
      { expr: "V = (4/3)πr³", desc: "球体积公式", kw: ["球", "体积", "圆周率"] },
      { expr: "S = πrl + πr²", desc: "圆锥表面积:侧面积加底面积", kw: ["圆锥", "表面积", "几何"] },
      { expr: "a²+b²=c²", desc: "勾股定理:直角三角形两直角边平方和等于斜边平方", kw: ["勾股定理", "直角三角形", "几何"] },
      { expr: "S = (a+b)h/2", desc: "梯形面积公式:上底加下底乘高除以二", kw: ["梯形", "面积", "几何"] },
      { expr: "S = ah", desc: "平行四边形面积:底乘高", kw: ["平行四边形", "面积", "几何"] },
      { expr: "S = 1/2·a·b·sinC", desc: "三角形面积正弦公式", kw: ["三角形", "面积", "正弦"] },
      { expr: "V = Sh", desc: "柱体体积:底面积乘高", kw: ["柱体", "体积", "几何"] },
      { expr: "V = 1/3·Sh", desc: "锥体体积:三分之一底面积乘高", kw: ["锥体", "体积", "几何"] },
      { expr: "l² = m² + n² - 2mn·cosA", desc: "余弦定理:任意三角形边长关系", kw: ["余弦定理", "三角形", "几何"] },
      { expr: "a/sinA = b/sinB = c/sinC = 2R", desc: "正弦定理:边长与对角正弦比值相等", kw: ["正弦定理", "三角形", "外接圆"] },
    ];
    const f = this.pick(formulas);
    return { expression: f.expr, description: f.desc, category: "几何", difficulty: this.randInt(1, 3), keywords: f.kw };
  }

  private _genNumberTheory(): MathEntry {
    const facts = [
      { expr: "p 是质数 ↔ p 只能被 1 和 p 整除", desc: "质数定义:大于1且只有1和自身两个正因数的自然数", kw: ["质数", "定义", "数论"] },
      { expr: "gcd(a,b)·lcm(a,b) = a·b", desc: "最大公约数与最小公倍数关系", kw: ["最大公约数", "最小公倍数", "数论"] },
      { expr: "a ≡ b (mod n) ↔ n|(a-b)", desc: "同余定义:差能被模数整除", kw: ["同余", "模运算", "数论"] },
      { expr: "费马小定理: a^(p-1) ≡ 1 (mod p)", desc: "p 为质数且 p∤a 时成立", kw: ["费马小定理", "同余", "数论"] },
      { expr: "欧拉定理: a^φ(n) ≡ 1 (mod n)", desc: "gcd(a,n)=1 时成立,φ为欧拉函数", kw: ["欧拉定理", "同余", "数论"] },
      { expr: "中国剩余定理: 两两互素模数的同余方程组有唯一解 mod M", desc: "模数两两互素时的解存在唯一性", kw: ["中国剩余定理", "同余", "数论"] },
      { expr: "算术基本定理: 任一大于1的整数可唯一分解为质数乘积", desc: "质因数分解的唯一性", kw: ["算术基本定理", "质因数", "数论"] },
      { expr: "ε 整除 n ↔ ε 是 n 的因数", desc: "整除定义", kw: ["整除", "因数", "数论"] },
    ];
    const f = this.pick(facts);
    return { expression: f.expr, description: f.desc, category: "数论", difficulty: this.randInt(2, 5), keywords: f.kw };
  }

  private _genCalculus(): MathEntry {
    const formulas = [
      { expr: "d/dx[x^n] = n·x^(n-1)", desc: "幂函数求导公式", kw: ["导数", "幂函数", "微积分"] },
      { expr: "d/dx[sin x] = cos x", desc: "正弦函数导数", kw: ["导数", "正弦", "三角函数"] },
      { expr: "d/dx[e^x] = e^x", desc: "指数函数导数等于自身", kw: ["导数", "指数", "e"] },
      { expr: "d/dx[ln x] = 1/x", desc: "自然对数导数", kw: ["导数", "对数", "ln"] },
      { expr: "∫x^n dx = x^(n+1)/(n+1) + C", desc: "幂函数不定积分", kw: ["积分", "幂函数", "原函数"] },
      { expr: "∫sin x dx = -cos x + C", desc: "正弦积分", kw: ["积分", "正弦", "三角"] },
      { expr: "∫e^x dx = e^x + C", desc: "指数积分", kw: ["积分", "指数", "e"] },
      { expr: "∫1/x dx = ln|x| + C", desc: "倒数积分", kw: ["积分", "对数", "ln"] },
      { expr: "∫_a^b f(x)dx = F(b)-F(a)", desc: "牛顿-莱布尼茨公式:定积分等于原函数差值", kw: ["定积分", "牛顿-莱布尼茨", "微积分"] },
      { expr: "lim(x→0) sinx/x = 1", desc: "重要极限:正弦极限", kw: ["极限", "重要极限", "正弦"] },
      { expr: "lim(x→∞) (1+1/x)^x = e", desc: "自然常数 e 的定义极限", kw: ["极限", "自然常数", "e"] },
      { expr: "d/dx[f·g] = f'g + fg'", desc: "乘积求导法则 (莱布尼茨法则)", kw: ["乘积法则", "导数", "莱布尼茨"] },
      { expr: "d/dx[f(g(x))] = f'(g(x))·g'(x)", desc: "链式法则:复合函数求导", kw: ["链式法则", "复合函数", "导数"] },
    ];
    const f = this.pick(formulas);
    return { expression: f.expr, description: f.desc, category: "微积分", difficulty: this.randInt(2, 4), keywords: f.kw };
  }

  private _genProbability(): MathEntry {
    const facts = [
      { expr: "P(A∪B) = P(A)+P(B)-P(A∩B)", desc: "概率加法公式:并集概率", kw: ["概率", "加法公式", "集合"] },
      { expr: "P(A|B) = P(A∩B)/P(B)", desc: "条件概率定义", kw: ["条件概率", "贝叶斯", "概率"] },
      { expr: "P(A∩B) = P(A|B)·P(B) = P(B|A)·P(A)", desc: "乘法公式/贝叶斯基础", kw: ["乘法公式", "贝叶斯", "概率"] },
      { expr: "E[X] = Σ x·P(x)", desc: "离散随机变量期望定义", kw: ["期望", "随机变量", "概率"] },
      { expr: "D[X] = E[X²]-(E[X])²", desc: "方差公式:平方的期望减期望的平方", kw: ["方差", "期望", "离散"] },
      { expr: "P(X=k) = C(n,k)·p^k·(1-p)^(n-k)", desc: "二项分布概率质量函数", kw: ["二项分布", "伯努利", "概率"] },
      { expr: "P(X=k) = λ^k·e^(-λ)/k!", desc: "泊松分布概率质量函数", kw: ["泊松分布", "稀有事件", "概率"] },
      { expr: "X~N(μ,σ²) → f(x)=1/(σ√2π)·e^(-(x-μ)²/(2σ²))", desc: "正态分布密度函数", kw: ["正态分布", "高斯", "概率"] },
    ];
    const f = this.pick(facts);
    return { expression: f.expr, description: f.desc, category: "概率统计", difficulty: this.randInt(2, 4), keywords: f.kw };
  }

  private _genTrigonometry(): MathEntry {
    const facts = [
      { expr: "sin²θ + cos²θ = 1", desc: "正弦余弦平方和恒等式", kw: ["三角恒等式", "平方和", "单位圆"] },
      { expr: "sin(2θ) = 2sinθcosθ", desc: "二倍角正弦公式", kw: ["二倍角", "正弦", "三角"] },
      { expr: "cos(2θ) = cos²θ-sin²θ = 2cos²θ-1", desc: "二倍角余弦公式", kw: ["二倍角", "余弦", "三角"] },
      { expr: "tan(θ₁+θ₂) = (tanθ₁+tanθ₂)/(1-tanθ₁·tanθ₂)", desc: "正切和角公式", kw: ["和角", "正切", "三角"] },
      { expr: "sin(θ₁±θ₂) = sinθ₁cosθ₂±cosθ₁sinθ₂", desc: "正弦和差角公式", kw: ["和差角", "正弦", "三角"] },
      { expr: "cos(θ₁±θ₂) = cosθ₁cosθ₂∓sinθ₁sinθ₂", desc: "余弦和差角公式", kw: ["和差角", "余弦", "三角"] },
      { expr: "sinθ + sinφ = 2sin((θ+φ)/2)cos((θ-φ)/2)", desc: "和化积公式:正弦和", kw: ["和化积", "正弦", "三角"] },
      { expr: "1+tan²θ = sec²θ", desc: "正切正割恒等式", kw: ["恒等式", "正切", "正割"] },
    ];
    const f = this.pick(facts);
    return { expression: f.expr, description: f.desc, category: "三角学", difficulty: this.randInt(1, 3), keywords: f.kw };
  }

  private _genMatrix(): MathEntry {
    const facts = [
      { expr: "det(AB) = det(A)·det(B)", desc: "行列式乘法公式:积的行列式等于行列式的积", kw: ["行列式", "矩阵", "乘法"] },
      { expr: "(AB)ᵀ = BᵀAᵀ", desc: "转置的乘法顺序反转", kw: ["转置", "矩阵", "乘法"] },
      { expr: "det(A⁻¹) = 1/det(A)", desc: "逆矩阵行列式", kw: ["逆矩阵", "行列式", "可逆"] },
      { expr: "A·A⁻¹ = I", desc: "逆矩阵定义:矩阵乘逆矩阵等于单位矩阵", kw: ["逆矩阵", "单位矩阵", "定义"] },
      { expr: "tr(AB) = tr(BA)", desc: "迹的循环性质", kw: ["迹", "矩阵", "循环"] },
      { expr: "rank(A) + nullity(A) = n", desc: "秩-零化度定理:列空间维数加核空间维数等于列数", kw: ["秩", "零化度", "线性方程"] },
    ];
    const f = this.pick(facts);
    return { expression: f.expr, description: f.desc, category: "线性代数", difficulty: this.randInt(2, 4), keywords: f.kw };
  }

  /** 生成 n 条数学知识记录 */
  generate(n: number): MathEntry[] {
    const results: MathEntry[] = [];
    for (let i = 0; i < n; i++) {
      results.push(this.generateOne());
    }
    return results;
  }

  /** 生成指定字符数的数学文本 (用于训练语料) */
  generateText(targetChars: number): string {
    const entries = this.generate(Math.ceil(targetChars / 40));
    return entries.map(e => `${e.expression}。${e.description}。`).join("");
  }
}

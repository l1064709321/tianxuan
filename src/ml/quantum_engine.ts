/**
 * QuantumMechanicsEngine — 量子力学方程求解器
 *
 * 不追求"解出数字", 追求"理解方程的结构和物理意义":
 *  - 识别方程类型 (含时/定态/一维/三维)
 *  - 给出求解思路 (分离变量/算子对角化/微扰论)
 *  - 对标准势场给出精确解的结构描述
 *  - 解释物理含义 (能级/波函数/概率密度/期望值)
 */
import { CNVectorStore } from "./cn_vectorstore";

export interface QMSolution {
  question: string;
  equation: string;
  type: string;
  solutionMethod: string;
  generalSolution: string;
  specificCases: Array<{
    potential: string;
    energyLevels: string;
    waveFunction: string;
    physicalMeaning: string;
  }>;
  keyConcepts: string[];
  limitations: string[];
}

export class QuantumMechanicsEngine {
  private standardPotentials: Record<string, {
    name: string;
    potential: string;
    energyLevels: string;
    waveFunctions: string[];
    meaning: string;
  }> = {
    "无限深方势阱": {
      name: "无限深方势阱 (一维)",
      potential: "V(x) = 0 (0<x<L), V=infinity (其他)",
      energyLevels: "E_n = n^2*pi^2*hbar^2/(2*m*L^2), n=1,2,3,...",
      waveFunctions: ["psi_n(x) = sqrt(2/L)*sin(n*pi*x/L),  0<x<L"],
      meaning: "粒子被束缚在 [0,L] 区间内, 能量量子化, n=1 是基态 (零点能不为零)",
    },
    "谐振子": {
      name: "一维谐振子",
      potential: "V(x) = (1/2)*m*omega^2*x^2",
      energyLevels: "E_n = (n + 1/2)*hbar*omega, n=0,1,2,...",
      waveFunctions: [
        "psi_n(x) = (m*omega/(pi*hbar))^(1/4) * 1/sqrt(2^n*n!) * H_n(sqrt(m*omega/hbar)*x) * exp(-m*omega*x^2/(2*hbar))",
        "其中 H_n 是厄米多项式",
      ],
      meaning: "量子力学最重要的可解模型, 零点能 E_0 = hbar*omega/2 != 0, 能级等间距",
    },
    "氢原子": {
      name: "氢原子 (库仑势, 三维)",
      potential: "V(r) = -e^2/(4*pi*epsilon_0*r)",
      energyLevels: "E_n = -13.6 eV / n^2, n=1,2,3,...",
      waveFunctions: [
        "psi_nlm(r,theta,phi) = R_nl(r) * Y_lm(theta,phi)",
        "R_nl: 径向部分 (拉盖尔多项式), Y_lm: 球谐函数",
        "量子数: n (主), l (角动量, 0<=l<=n-1), m (磁量子数, -l<=m<=l)",
      ],
      meaning: "玻尔模型的量子力学推广, 解释了原子光谱的精细结构, 能级简并度 = n^2",
    },
    "自由粒子": {
      name: "自由粒子 (V=0)",
      potential: "V(x) = 0 (全空间)",
      energyLevels: "E = hbar^2*k^2/(2*m), k in R (连续谱)",
      waveFunctions: ["psi(x) = A*e^(ikx) + B*e^(-ikx), 平面波解"],
      meaning: "无约束粒子, 动量本征态也是能量本征态, 能量连续而非量子化",
    },
    "势垒隧穿": {
      name: "方势垒隧穿",
      potential: "V(x) = V_0 (0<x<a), V=0 (其他)",
      energyLevels: "E < V_0 时透射系数 T != 0 (隧穿效应)",
      waveFunctions: [
        "E > V_0: 部分透射部分反射, T = [1 + V_0^2*sin^2(k_2*a)/(4*E*(E-V_0))]^(-1)",
        "E < V_0: 指数衰减穿透势垒, T approx exp(-2*kappa*a), kappa = sqrt(2*m*(V_0-E))/hbar",
      ],
      meaning: "量子力学特有现象, 经典物理中粒子无法穿越高于自身能量的势垒, 扫描隧道显微镜(TFM)的基础",
    },
  };

  answer(question: string): QMSolution {
    const q = question.trim().toLowerCase();

    let eqType = "含时薛定谔方程";
    let equation = "i*hbar * dPsi/dt = H*Psi";

    if (q.includes("定态") || q.includes("时间无关") || q.includes("stationary")) {
      eqType = "定态薛定谔方程";
      equation = "H*psi = E*psi,  其中 H = -hbar^2/(2m)*grad^2 + V(r)";
    } else if (q.includes("含时") || q.includes("time-dependent")) {
      eqType = "含时薛定谔方程";
      equation = "i*hbar * dPsi/dt = [-hbar^2/(2m)*grad^2 + V(r,t)]*Psi";
    } else if (q.includes("狄拉克") || q.includes("dirac")) {
      eqType = "狄拉克方程 (相对论性)";
      equation = "(i*gamma^mu*d_mu - m)*psi = 0";
    } else if (q.includes("克莱因") || q.includes("kg")) {
      eqType = "克莱因-戈尔登方程 (相对论性标量)";
      equation = "(box + m^2)*phi = 0,  即 d^2phi/dt^2 - grad^2(phi) + m^2*phi = 0";
    }

    const potentialKeys = Object.keys(this.standardPotentials);
    let matchedPotential = potentialKeys.find(k => q.includes(k)) ?? null;

    const specificCases: Array<{ potential: string; energyLevels: string; waveFunction: string; physicalMeaning: string }> = matchedPotential
      ? [{ potential: this.standardPotentials[matchedPotential].name, energyLevels: this.standardPotentials[matchedPotential].energyLevels, waveFunction: this.standardPotentials[matchedPotential].waveFunctions[0], physicalMeaning: this.standardPotentials[matchedPotential].meaning }] as Array<{ potential: string; energyLevels: string; waveFunction: string; physicalMeaning: string }>
      : this._getAllStandardCases();

    return {
      question,
      equation,
      type: eqType,
      solutionMethod: this._getSolutionMethod(eqType, matchedPotential),
      generalSolution: this._getGeneralSolution(eqType, matchedPotential),
      specificCases,
      keyConcepts: [
        "波函数 Psi(r,t) 是概率幅, |Psi|^2 是概率密度",
        "算符 H 是哈密顿量, 本征值 E 是能量本征值",
        "测量导致波函数坍缩到本征态",
        "不确定性原理: Delta_x * Delta_p >= hbar/2",
        "量子态叠加原理: 若 psi_1, psi_2 是解, 则 a*psi_1+b*psi_2 也是解",
      ],
      limitations: [
        "非相对论性: 粒子速度 << c, 高速需用狄拉克方程或QED",
        "单粒子近似: 多体问题需二次量子化/密度泛函理论(DFT)",
        "严格可解的问题很少, 大多数需数值方法 (有限差分/谱方法/Monte Carlo)",
        "测量问题: 波函数坍缩机制仍在诠释争论中 (哥本哈根/多世界/隐变量)",
      ],
    };
  }

  private _getSolutionMethod(eqType: string, potential: string | null): string {
    if (!potential) {
      return eqType.includes("定态")
        ? "分离变量法: Psi(r,t) = psi(r)*exp(-i*E*t/hbar), 代入得 H*psi=E*psi (本征值问题)\n" +
          "求解思路: 1) 写出哈密顿量 H = -hbar^2/(2m)*grad^2+V(r)  2) 选择合适坐标系  3) 分离变量  4) 求解常微分方程  5) 施加边界条件确定本征值"
        : "直接求解含时偏微分方程, 或对初态做傅里叶展开后逐分量演化";
    }
    const methods: Record<string, string> = {
      "无限深方势阱": "分离变量 + 边界条件 psi(0)=psi(L)=0 -> 正弦级数解, 能量量子化来自边界约束",
      "谐振子": "代数法 (升降算子): 定义 a, a_dag, 则 H = hbar*omega*(a_dag*a+1/2), 能级自然量子化\n" +
        "或微分方程法: 级数解 -> 厄米多项式 H_n(xi)",
      "氢原子": "球坐标分离变量: psi(r,theta,phi) = R(r)*Y(theta,phi)\n" +
        "径向方程 -> 关联拉盖尔多项式, 角向方程 -> 球谐函数 Y_lm\n" +
        "三个量子数 (n,l,m) 自然出现, 对应能量/角动量/角动量z分量",
      "自由粒子": "平面波解 psi = A*e^(ikx), 连续性边界条件 -> 动量本征态, 能量连续谱",
      "势垒隧穿": "分段常数势, 每段用指数/振荡解, 在边界处匹配 psi 和 psi' 的连续性",
    };
    return methods[potential] ?? "需根据具体势场选择合适方法";
  }

  private _getGeneralSolution(eqType: string, potential: string | null): string {
    if (eqType.includes("定态") && potential) {
      return "定态薛定谔方程 H*psi = E*psi 的通解结构:\n" +
        "  psi(r) = sum_n c_n * psi_n(r),  其中 psi_n 是第n个本征态\n" +
        "  任意态可展开为本征态的线性叠加 (叠加原理)\n" +
        "  时间演化: Psi(r,t) = sum_n c_n * psi_n(r) * exp(-i*E_n*t/hbar)";
    }
    return "薛定谔方程是线性偏微分方程, 通解 = 所有本征态的线性叠加 * 时间相位因子";
  }

  private _getAllStandardCases(): Array<{ potential: string; energyLevels: string; waveFunction: string; physicalMeaning: string }> {
    return Object.values(this.standardPotentials).map(p => ({
      potential: p.name,
      energyLevels: p.energyLevels,
      waveFunction: p.waveFunctions[0],
      physicalMeaning: p.meaning,
    }));
  }
}

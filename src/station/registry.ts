import { ComputeUnit, NeuralSystem, NeuralSystemKind, Stage, SystemStatus, TaskType } from "./types";

const SEED: NeuralSystem[] = [
  { kind: "lnn", name: "LNN 液态网络", role: "连续时间状态动力学", stage: "deferred", enabled: false, status: "shelf", justification: "候选: 连续时间 RNN(神经 ODE 类, arXiv:2006.04439), 与 Mamba/SSM 动力学重叠; 仅当状态建模出现可测缺口时评估", evidence: { metric: "时序/状态建模指标", baseline: 0, result: 0, status: "pending", note: "待评估; 官方依据 Liquid Time-constant Networks" } },
  { kind: "mamba", name: "Mamba", role: "线性状态流(快动力学)", stage: "phase1", enabled: false, status: "idle", roleTag: "central", justification: "对照基线: 线性状态建模 vs 纯 Transformer(同配置)", evidence: { metric: "事件语料 top-1(30k×2ep)", baseline: 49.1, result: 47.5, status: "pending", note: "SSM 内核 gradcheck=0.0000; 与 GRU 基线持平, 无缺口证据" } },
  { kind: "sparse-attention", name: "稀疏 Attention", role: "慢语义(事件级语义演化)", stage: "phase2", enabled: false, status: "idle", roleTag: "central", justification: "对照基线: 先验是否缺失注意力评估后定", evidence: { metric: "事件语料 top-1(30k×2ep)", baseline: 49.1, result: 47.8, status: "pending", note: "机制正确(gradcheck 0.0000)+gate 选择性双峰; 任务级提升证据未达成→不可 enabled" } },
  { kind: "titans", name: "Titans 记忆", role: "长期记忆 + L0 零算力直出", stage: "phase3", enabled: false, status: "idle", justification: "L0 记忆: 官方定位与长上下文/低成本命中一致", evidence: { metric: "L0 记忆命中率", baseline: 0, result: 0, status: "pending", note: "在线动量记忆已实现; 命中率待向量库召回量化" } },
  { kind: "cnn", name: "CNN", role: "局部感知 n-gram 特征", stage: "phase1", enabled: true, status: "idle", roleTag: "perception", justification: "局部字符模式, 独立预测头与循环/注意力输出融合", evidence: { metric: "事件语料 top-1(30k×4ep 同配置消融)", baseline: 47.5, result: 49.3, status: "pass", note: "gradcheck-cnn=0.0000; 修复 top-1 缺口 +1.8pt; 就后召回探针 9.7→6.5(小规模代价, 大语料复查)" } },
  { kind: "rwkv", name: "RWKV", role: "低延迟读入", stage: "deferred", enabled: false, status: "shelf", justification: "候选: 仅当推理延迟成为可测瓶颈时评估", evidence: { metric: "推理延迟", baseline: 0, result: 0, status: "pending" } },
  { kind: "kan", name: "KAN", role: "解码增强", stage: "deferred", enabled: false, status: "shelf", justification: "候选: 仅当解码精度出现可测缺口时评估", evidence: { metric: "解码 top-1", baseline: 0, result: 0, status: "pending" } },
  { kind: "moe", name: "MoE 专家路由", role: "动态专家选择 + 负载均衡", stage: "phase3", enabled: false, status: "idle", justification: "核心架构: 按输入embedding动态路由Top-K专家, 简单任务走浅专家, 复杂任务招募深度专家; 辅助损失防止专家垄断", evidence: { metric: "事件语料top-1 + 路由熵", baseline: 49.3, result: 0, status: "pending", note: "CharMultiNeuro已集成; 门控网络+4专家头+Titans在线学习; 训练进行中, 待验证" } },
  { kind: "snn", name: "SNN", role: "事件门控/稀疏唤醒", stage: "deferred", enabled: false, status: "shelf", justification: "否决: 收益绑定神经形态硬件,CPU+tfjs 无内核且训练不稳定", evidence: { metric: "CPU 能耗/收益", baseline: 0, result: 0, status: "fail", note: "CPU 文本世界无收益场景" } },
  { kind: "stdp", name: "STDP", role: "突触权重可塑性规则", stage: "phase1", enabled: false, status: "idle", justification: "实验: 局部 Hebbian 相关性与端到端损失叠加, stdpRate=0.01", evidence: { metric: "端到端 loss + 局部相关性", baseline: 0, result: 0, status: "pending", note: "STDP 作为附加正则叠加 BPTT 梯度" } },
  { kind: "stda", name: "STDA", role: "兴奋性/阈值适应", stage: "phase1", enabled: false, status: "idle", justification: "实验: 基于 h2 激活的滑动平均自适应输出头衰减, tau=50", evidence: { metric: "—", baseline: 0, result: 0, status: "pending", note: "STDA 与路由熵正则互补作用于不同层级" } },
];

/** 注册中心: 系统进站登记,enabled 前必须通过 justification 门槛 */
export class Registry {
  private systems = new Map<NeuralSystemKind, NeuralSystem>();

  constructor(seed: NeuralSystem[] = SEED) {
    for (const system of seed) this.systems.set(system.kind, system);
  }

  register(system: NeuralSystem): void {
    this.systems.set(system.kind, system);
  }

  unregister(kind: NeuralSystemKind): void {
    this.systems.delete(kind);
  }

  get(kind: NeuralSystemKind): NeuralSystem | undefined {
    return this.systems.get(kind);
  }

  enabled(): NeuralSystem[] {
    return [...this.systems.values()].filter((s) => s.enabled);
  }

  byStage(stage: Stage): NeuralSystem[] {
    return [...this.systems.values()].filter((s) => s.stage === stage);
  }

  setStatus(kind: NeuralSystemKind, status: SystemStatus): void {
    const system = this.systems.get(kind);
    if (system) system.status = status;
  }

  /** 按 unitId 反查所属神经系统(角色/证据) */
  unitSystem(unitId: string): NeuralSystem | undefined {
    return [...this.systems.values()].find((s) => s.compute?.unitId === unitId);
  }

  /** MoD 路由用: 指定任务类型神经链的启用计算单元,按深度升序 */
  computeUnits(type: TaskType): ComputeUnit[] {
    return [...this.systems.values()]
      .filter((s) => s.enabled && s.compute && s.compute.chains.includes(type))
      .sort((a, b) => (a.compute!.depth - b.compute!.depth))
      .map((s) => s.compute!);
  }

  summary(): Array<{
    kind: NeuralSystemKind;
    name: string;
    role: string;
    stage: Stage;
    enabled: boolean;
    status: SystemStatus;
    justification: string;
  }> {
    return [...this.systems.values()].map(({ kind, name, role, stage, enabled, status, justification }) => ({
      kind,
      name,
      role,
      stage,
      enabled,
      status,
      justification,
    }));
  }
}

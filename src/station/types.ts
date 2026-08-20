export type NeuralSystemKind =
  | "gru"
  | "lnn"
  | "mamba"
  | "sparse-attention"
  | "titans"
  | "cnn"
  | "snn"
  | "rwkv"
  | "kan"
  | "moe"
  | "stdp"
  | "stda";

export type Stage = "phase1" | "phase2" | "phase3" | "deferred";

export type SystemStatus = "idle" | "busy" | "error" | "shelf";

/** 任务类型: 决定走哪条神经链(借鉴 GPT-5 实时路由器的任务级路由) */
export type TaskType = "text" | "speech" | "vision" | "world";

/** 神经链角色: 感知神经 → 中央神经 → 输出神经(经黑板 z 协同, 神经间不得直接互调) */
export type NeuroRole = "perception" | "central" | "output";

/** 证据门槛: 每个神经 enabled 前必须量化「修复基线哪个缺口」 */
export interface Evidence {
  /** 指标名(如 "事件语料 top-1") */
  metric: string;
  /** 基线值 */
  baseline: number;
  /** 本神经结果 */
  result: number;
  status: "pass" | "pending" | "fail";
  /** 证据说明/实验环境 */
  note?: string;
}

export interface Task {
  id: string;
  /** 任务类型 → 选择神经链; v1 只有 text 链有真实计算单元 */
  type: TaskType;
  input: unknown;
  /** 0..1 预估复杂度,决定初始路由深度 */
  complexity: number;
  /** 本次任务的最大可执行层数(算力预算) */
  budget: number;
}

export interface WorldState {
  version: number;
  data: Record<string, unknown>;
}

/** MoD 式计算单元: 同一模型内的某一层,由路由按预算裁剪/纳入 */
export interface ComputeUnit {
  unitId: string;
  depth: number;
  /** 相对计算成本(归一化 FLOP) */
  cost: number;
  /** 该单元服务的神经链(任务类型); 空数组 = 不参与任何路由 */
  chains: TaskType[];
  /** 执行一层,可直接写共享状态 z,返回置信度 0..1 */
  forward(task: Task, state: WorldState): Promise<number>;
}

export interface NeuralSystem {
  kind: NeuralSystemKind;
  name: string;
  role: string;
  stage: Stage;
  enabled: boolean;
  status: SystemStatus;
  /** 启用门槛: 必须声明修复基线的哪个指标缺口(有证据才允许 enabled) */
  justification: string;
  /** 神经链角色(感知/中央/输出);缺省视为无角色 */
  roleTag?: NeuroRole;
  /** 证据量化(消融对照);无证据 = pending/fail, 不允许 enabled */
  evidence?: Evidence;
  /** 启用时承载的计算单元;缺省视为不参与路由(如纯记忆) */
  compute?: ComputeUnit;
}

export interface TaskResult {
  taskId: string;
  /** 实际执行到的最大深度(0 = 记忆直出) */
  depth: number;
  output: unknown;
  confidence: number;
  /** 实际执行的计算单元,按序 */
  units: string[];
  memoryHit: boolean;
}

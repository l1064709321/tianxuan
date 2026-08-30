import { AuditLog } from "./audit";
import { Blackboard } from "./blackboard";
import { CollaborationProbe } from "./probe";
import { Registry } from "./registry";
import { Task, TaskResult, WorldState } from "./types";

export interface ExecutorOptions {
  /** 置信度低于该值,在预算内继续扩深 */
  cascadeThreshold?: number;
  /** 中央神经点火低于该值(且样本数≥warmup) → 预算转向跳过该深度 */
  coldThreshold?: number;
  /** 点火评估暖机样本数 */
  warmup?: number;
  /** 最大执行深度 (默认4) */
  maxDepth?: number;
  /** 是否启用审计 (默认true) */
  enableAudit?: boolean;
  /** 是否启用探针 (默认true) */
  enableProbe?: boolean;
}

/** 默认配置 */
export const DEFAULT_EXECUTOR_OPTIONS: ExecutorOptions = {
  cascadeThreshold: 0.6,
  coldThreshold: 0.15,
  warmup: 8,
  maxDepth: 4,
  enableAudit: true,
  enableProbe: true,
};

/**
 * MoD(Mixture-of-Depths)式执行器
 * 共享同一模型: 按任务复杂度决定初始深度,置信不足则在预算内扩展深度
 * 不做"小网/大网"两套网络,路径裁剪替代网络复制(依据 DeepMind MoD 论文)
 */
export class Executor {
  private readonly cascadeThreshold: number;
  private readonly coldThreshold: number;
  private readonly warmup: number;
  private readonly maxDepth: number;
  private readonly enableAudit: boolean;
  private readonly enableProbe: boolean;

  constructor(
    private readonly registry: Registry,
    private readonly blackboard: Blackboard,
    private readonly audit: AuditLog,
    private readonly probe: CollaborationProbe,
    options: ExecutorOptions = {},
  ) {
    this.cascadeThreshold = options.cascadeThreshold ?? DEFAULT_EXECUTOR_OPTIONS.cascadeThreshold!;
    this.coldThreshold = options.coldThreshold ?? DEFAULT_EXECUTOR_OPTIONS.coldThreshold!;
    this.warmup = options.warmup ?? DEFAULT_EXECUTOR_OPTIONS.warmup!;
    this.maxDepth = options.maxDepth ?? DEFAULT_EXECUTOR_OPTIONS.maxDepth!;
    this.enableAudit = options.enableAudit ?? DEFAULT_EXECUTOR_OPTIONS.enableAudit!;
    this.enableProbe = options.enableProbe ?? DEFAULT_EXECUTOR_OPTIONS.enableProbe!;
  }

  async execute(task: Task): Promise<TaskResult> {
    const state = this.blackboard.worldState();

    // L0 — 记忆直出(零算力)
    const memKey = `mem:${task.id}`;
    const hit = this.blackboard.read(memKey);
    if (this.registry.get("titans")?.enabled && hit !== undefined) {
      this.audit.route({ taskId: task.id, depth: 0, units: [], confidence: 1, memoryHit: true });
      return { taskId: task.id, depth: 0, output: hit, confidence: 1, units: [], memoryHit: true };
    }

    const units = this.registry.computeUnits(task.type);
    if (units.length === 0) {
      this.audit.route({ taskId: task.id, depth: 0, units: [], confidence: 0, memoryHit: false });
      return { taskId: task.id, depth: 0, output: undefined, confidence: 0, units: [], memoryHit: false };
    }

    const maxDepth = this.maxDepth;
    const budget = Math.max(1, Math.min(task.budget || maxDepth, maxDepth));
    let targetDepth = Math.min(maxDepth, Math.max(1, Math.round(task.complexity * maxDepth)));
    const ranUnits: string[] = [];
    let lastConfidence = 0;

    for (let used = 0; used < budget && targetDepth <= maxDepth; used++) {
      let next = units.find((unit) => unit.depth <= targetDepth && !ranUnits.includes(unit.unitId));
      if (!next) {
        targetDepth += 1; // 该深度无单元: 预算内扩一层
        used -= 1;
        continue;
      }
      // 点火探针: 单元执行后把自身活跃度写入 z:act:<unitId>, 执行器记录
      const sys = this.registry.unitSystem(next.unitId);
      const ignition = Number(this.blackboard.read(`z:act:${next.unitId}`) ?? 0);
      if (this.enableProbe) {
        this.probe.record(task.id, next.unitId, sys?.kind ?? "?", ignition);
      }
      // 预算转向(根据多神经实际点火调整): 中央神经长期不点火 → 该深度让路, 预算留给他神经
      if (sys?.roleTag === "central" && this.enableProbe && this.probe.sampleCount(next.unitId) >= this.warmup && this.probe.ignitionAvg(next.unitId, 32) < this.coldThreshold) {
        if (this.enableAudit) {
          this.audit.route({
            taskId: task.id,
            depth: next.depth,
            units: [...ranUnits, next.unitId],
            confidence: 0,
            memoryHit: false,
            note: "cold-central",
          });
        }
        targetDepth = Math.min(maxDepth, next.depth + 1); // 让路: 跳到该深度之后
        continue;
      }
      ranUnits.push(next.unitId);
      const confidence = await next.forward(task, state);
      lastConfidence = confidence;
      if (this.enableAudit) {
        this.audit.route({
          taskId: task.id,
          depth: next.depth,
          units: [...ranUnits],
          confidence,
          memoryHit: false,
        });
      }
      if (confidence >= this.cascadeThreshold) {
        return this.result(task, ranUnits, units, confidence, state);
      }
      targetDepth = Math.min(maxDepth, next.depth + 1); // 置信不足 → 预算内扩到下一层

    }

    return this.result(task, ranUnits, units, lastConfidence, state);
  }

  private result(task: Task, ranUnits: string[], allUnits: { depth: number; unitId: string }[], confidence: number, state: WorldState): TaskResult {
    const depth = ranUnits.length === 0 ? 0 : Math.max(...allUnits.filter((u) => ranUnits.includes(u.unitId)).map((u) => u.depth));
    return {
      taskId: task.id,
      depth,
      output: this.blackboard.read(`o:${task.id}`) ?? null,
      confidence,
      units: ranUnits,
      memoryHit: false,
    };
  }
}

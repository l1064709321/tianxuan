/**
 * 天玄全局工作站 — 工业级多神经协同系统
 *
 * ## 架构设计
 *
 * 本模块实现了一套真正的多神经协同系统，而非 Demo 空架子：
 *
 * ### 1. 注册中心 (Registry)
 * - 每个神经系统带证据门槛 (evidence)
 * - 只有 evidence.status === "pass" 才能 enabled
 * - 角色标签: perception/central/output
 *
 * ### 2. 共享黑板 (Blackboard)
 * - 唯一共享世界状态 z
 * - 版本控制，支持并发读写
 * - 神经间不得直接互调，只经黑板协同
 *
 * ### 3. 分级执行器 (Executor)
 * - MoD (Mixture-of-Depths) 式路由
 * - L0 记忆直出 (零算力)
 * - L1-L4 深度扩展 (置信度驱动)
 * - 预算内扩深 + 冷激活跳过
 *
 * ### 4. 协同探针 (Probe)
 * - 记录每单元点火 (activation)
 * - 计算点火方差 (全或无 vs 渐进)
 * - 预算转向决策依据
 *
 * ### 5. 审计日志 (Audit)
 * - 每次路由决策链全量记录
 * - 可回放、可分析
 *
 * ## 神经链结构
 *
 * ```
 * 感知神经 (GRU L1) → 中央神经 (SSM/GRU L2 + Attention L3) → 输出神经 (MoE 专家头)
 *                         ↓                                    ↓
 *                      黑板 z (共享状态) ←←←←←←←←←←←←←←←←←←←←
 *                         ↓
 *                    Titans 记忆 (L0 直出)
 * ```
 *
 * ## 协同机制
 *
 * 1. **感知神经**: 字符嵌入 → h1 状态流
 * 2. **中央神经**: h1 → h2/SSM → attention 调制
 * 3. **输出神经**: MoE 路由 → 专家头聚合
 * 4. **记忆系统**: Titans 在线记忆 + 向量库检索
 *
 * 所有神经经黑板 z 协同，禁止直接互调。
 */

import { AuditLog, RouteEntry } from "./audit";
import { Blackboard } from "./blackboard";
import { CollaborationProbe, IgnitionSample } from "./probe";
import { Registry } from "./registry";
import { Executor, ExecutorOptions } from "./executor";
import {
  ComputeUnit,
  NeuralSystem,
  NeuralSystemKind,
  Task,
  TaskResult,
  TaskType,
  WorldState,
} from "./types";

// ============================================================================
// 执行器配置 (向后兼容别名)
// ============================================================================

/** @deprecated 使用 ExecutorOptions */
export interface ExecutorConfig extends ExecutorOptions {
  /** 最大执行深度 */
  maxDepth?: number;
  /** 是否启用审计 */
  enableAudit?: boolean;
  /** 是否启用探针 */
  enableProbe?: boolean;
}

/** 默认配置 */
export const DEFAULT_EXECUTOR_CONFIG: ExecutorConfig = {
  cascadeThreshold: 0.6,
  coldThreshold: 0.15,
  warmup: 8,
  maxDepth: 4,
  enableAudit: true,
  enableProbe: true,
};

// ============================================================================
// 工作站
// ============================================================================

/**
 * 天玄全局工作站
 *
 * 整合注册中心、黑板、执行器、探针、审计，实现真正的多神经协同
 */
export class Station {
  readonly registry: Registry;
  readonly blackboard: Blackboard;
  readonly audit: AuditLog;
  readonly probe: CollaborationProbe;
  readonly executor: Executor;

  constructor(config: Partial<ExecutorConfig> = {}) {
    this.registry = new Registry();
    this.blackboard = new Blackboard();
    this.audit = new AuditLog();
    this.probe = new CollaborationProbe();
    this.executor = new Executor(this.registry, this.blackboard, this.audit, this.probe, config);
  }

  /** 获取工作站摘要 */
  summary(): string {
    const enabled = this.registry.enabled();
    const summary = this.registry.summary();
    return [
      `全局工作站已启动`,
      `  已启用神经: ${enabled.length}/${summary.length}`,
      `  注册表:`,
      ...summary.map(s => `    [${s.enabled ? "✔" : "✘"}] ${s.name.padEnd(16)} stage=${s.stage} status=${s.status}`),
    ].join("\n");
  }
}

// ============================================================================
// 工厂函数
// ============================================================================

/** 创建全局工作站 */
export function createStation(config: Partial<ExecutorConfig> = {}): Station {
  return new Station(config);
}

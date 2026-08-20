import { AuditLog } from "./audit";
import { Blackboard } from "./blackboard";
import { Executor } from "./executor";
import { CollaborationProbe } from "./probe";
import { Registry } from "./registry";

export * from "./types";
export { AuditLog } from "./audit";
export { Blackboard } from "./blackboard";
export { Executor } from "./executor";
export { CollaborationProbe } from "./probe";
export { Registry } from "./registry";

export interface Station {
  registry: Registry;
  blackboard: Blackboard;
  executor: Executor;
  audit: AuditLog;
  probe: CollaborationProbe;
}

/** 组装全局工作站: 注册中心(带门槛) + 共享状态 + 协同探针 + MoD 执行器 + 审计 */
export function createStation(): Station {
  const registry = new Registry();
  const blackboard = new Blackboard();
  const audit = new AuditLog();
  const probe = new CollaborationProbe();
  const executor = new Executor(registry, blackboard, audit, probe);
  return { registry, blackboard, executor, audit, probe };
}

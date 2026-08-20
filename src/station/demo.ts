import { createStation } from "./index";
import { CollaborationProbe } from "./probe";
import { Task } from "./types";

async function main(): Promise<void> {
  const station = createStation();
  const { registry, blackboard, executor, audit } = station;

  console.log("全局工作站(MoD 式)已启动,进站神经系统:");
  for (const s of registry.summary()) {
    console.log(`  [${s.stage.padEnd(8)}] ${s.enabled ? "✔ 启用" : "✘ 货架"} ${s.name.padEnd(12)} — ${s.role}`);
    if (!s.enabled) console.log(`         门槛: ${s.justification}`);
  }

  // 演示 stub: 同一模型内的两层(共享预算底座,不复制网络)
  registry.register({
    kind: "mamba", name: "Mamba", role: "浅路径 stub", stage: "phase1", enabled: true, status: "idle", justification: "对照基线",
    compute: {
      unitId: "mamba:layer1", depth: 1, cost: 1, chains: ["text"],
      // 浅路径置信 0.45 不足 → 触发预算内扩深
      forward: async () => 0.45,
    },
  });
  registry.register({
    kind: "sparse-attention", name: "稀疏 Attention", role: "深路径 stub", stage: "phase2", enabled: true, status: "idle", justification: "慢语义路径",
    compute: {
      unitId: "sparse-attn:layer3", depth: 3, cost: 2, chains: ["text"],
      forward: async (task, state) => {
        state.data[`o:${task.id}`] = `deep-${task.id}`;
        return 0.92;
      },
    },
  });

  blackboard.write("mem:task-0", "记忆直出内容");
  const tasks: Task[] = [
    { id: "task-0", type: "text", input: "m", complexity: 0.1, budget: 4 }, // 记忆命中 AS L0 零算力
    { id: "task-1", type: "text", input: "x", complexity: 0.3, budget: 4 }, // 浅路径置信不足 → 预算内扩深
    { id: "task-2", type: "text", input: "y", complexity: 0.9, budget: 1 }, // 预算只有 1 层 → 深路径不可达
  ];
  for (const task of tasks) {
    const r = await executor.execute(task);
    const tag = r.memoryHit ? " [L0 记忆直出]" : "";
    console.log(`\n${task.id}: depth=${r.depth} 置信=${r.confidence} 单元=[${r.units.join(",")}] 输出=${JSON.stringify(r.output)}${tag}`);
  }

  console.log("\n审计回放:");
  for (const e of audit.all()) {
    console.log(`  #${e.seq} task=${e.taskId} depth=${e.depth} units=[${e.units.join(",")}] conf=${e.confidence}${e.memoryHit ? " memHit" : ""}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

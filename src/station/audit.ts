export interface RouteEntry {
  seq: number;
  taskId: string;
  depth: number;
  units: string[];
  confidence: number;
  memoryHit: boolean;
  /** 路由备注(如预算转向: cold-central 跳过) */
  note?: string;
  ts: number;
}

/** 审计日志: 每次路由的决策链全量记录,可回放 */
export class AuditLog {
  private entries: RouteEntry[] = [];

  route(entry: Omit<RouteEntry, "seq" | "ts">): void {
    this.entries.push({ ...entry, seq: this.entries.length + 1, ts: Date.now() });
  }

  lastNote(): string | undefined {
    const last = this.entries[this.entries.length - 1];
    return last?.note;
  }

  all(): RouteEntry[] {
    return [...this.entries];
  }
}

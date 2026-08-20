import { WorldState } from "./types";

/** 全局共享黑板(瘦身版): 唯一的 z 对象,普通读写,不做发布订阅 */
export class Blackboard {
  private state: WorldState = { version: 0, data: {} };

  worldState(): WorldState {
    return this.state;
  }

  read(key: string): unknown {
    return this.state.data[key];
  }

  write(key: string, value: unknown): void {
    this.state.data[key] = value;
    this.state.version += 1;
  }
}

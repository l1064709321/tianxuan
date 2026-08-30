/**
 * ReplayBuffer — 重放缓冲区 (模拟海马体-皮层重放巩固)
 *
 * 机制: 训练时记录 (h1, h2, x_id, y_id, loss) 过渡样本,
 * 每批训练时以概率 replayProb 从缓冲区采样, 用预测误差 loss 加权重新训练。
 * 这是人脑睡眠时"情景记忆重放"的离线版实现。
 */
export interface ReplaySample {
  h1: Float64Array;
  h2: Float64Array;
  xId: number;
  yId: number;
  predLoss: number;
  nextPredLoss: number;
}

export class ReplayBuffer {
  private buffer: ReplaySample[] = [];
  private maxCapacity: number;
  private writePos = 0;

  constructor(maxCapacity = 4096) {
    this.maxCapacity = maxCapacity;
  }

  push(sample: Omit<ReplaySample, 'predLoss' | 'nextPredLoss'> & { predLoss?: number; nextPredLoss?: number }): void {
    const s: ReplaySample = {
      ...sample,
      predLoss: sample.predLoss ?? 0,
      nextPredLoss: sample.nextPredLoss ?? 0,
    };
    if (this.buffer.length < this.maxCapacity) {
      this.buffer.push(s);
    } else {
      this.buffer[this.writePos] = s;
    }
    this.writePos = (this.writePos + 1) % this.maxCapacity;
  }

  sample(n: number, weightByLoss = true): ReplaySample[] {
    if (this.buffer.length === 0) return [];
    const k = Math.min(n, this.buffer.length);
    const weights = weightByLoss
      ? this.buffer.map(s => Math.max(0.01, s.predLoss + s.nextPredLoss))
      : this.buffer.map(() => 1);
    let total = 0;
    for (const w of weights) total += w;
    const result: ReplaySample[] = [];
    const available = this.buffer.slice();
    const availWeights = weights.slice();
    while (result.length < k && available.length > 0) {
      let r = Math.random() * total;
      for (let i = 0; i < available.length; i++) {
        r -= availWeights[i];
        if (r <= 0) {
          result.push(available[i]);
          total -= availWeights[i];
          availWeights.splice(i, 1);
          available.splice(i, 1);
          break;
        }
      }
    }
    return result;
  }

  size(): number { return this.buffer.length; }
  isEmpty(): boolean { return this.buffer.length === 0; }
  clear(): void { this.buffer = []; this.writePos = 0; }
  occupancy(): number { return this.buffer.length / this.maxCapacity; }
}

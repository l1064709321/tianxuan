/**
 * 架构稀疏化 — 参数剪枝与路由稀疏性控制
 *
 * ## 设计原则
 *
 * 1. **结构化剪枝**：按层/按头剪枝，保持模型结构完整
 * 2. **动态稀疏**：训练过程中逐步增加稀疏度
 * 3. **重要性评估**：基于梯度范数/激活能量评估参数重要性
 * 4. **MoE 路由稀疏**：控制专家激活的稀疏性
 *
 * ## 与 MoE 的关系
 *
 * MoE 提供"计算稀疏"（每步只激活 K 个专家），
 * 本模块提供"参数稀疏"（剪枝不重要的权重）。
 * 两者互补：MoE 减少计算量，剪枝减少参数量。
 */

import { Group } from "./model";

export interface PruningConfig {
  /** 目标稀疏度 (0-1, 1=完全稀疏) */
  targetSparsity: number;
  /** 剪枝方法: magnitude=幅值剪枝, gradient=梯度剪枝 */
  method: 'magnitude' | 'gradient';
  /** 剪枝粒度: element=元素级, head=头级, layer=层級 */
  granularity: 'element' | 'head' | 'layer';
  /** 渐进剪枝步数 */
  gradualSteps: number;
  /** 当前步数 */
  currentStep: number;
}

export const DEFAULT_PRUNING_CONFIG: PruningConfig = {
  targetSparsity: 0.3,
  method: 'magnitude',
  granularity: 'element',
  gradualSteps: 1000,
  currentStep: 0,
};

export interface PruningResult {
  originalParams: number;
  prunedParams: number;
  sparsity: number;
  removedIndices: number[];
}

/**
 * 架构稀疏化器
 */
export class ArchitecturePruner {
  readonly cfg: PruningConfig;
  private masks: Map<string, Float64Array> = new Map();
  private originalParams: Map<string, number> = new Map();

  constructor(cfg: Partial<PruningConfig> = {}) {
    this.cfg = { ...DEFAULT_PRUNING_CONFIG, ...cfg };
  }

  /**
   * 记录初始参数（用于剪枝后恢复）
   */
  snapshot(groups: Map<string, Group>): void {
    for (const [name, g] of groups) {
      this.originalParams.set(name, g.p.length);
      this.masks.set(name, new Float64Array(g.p.length).fill(1));
    }
  }

  /**
   * 计算当前稀疏度
   */
  currentSparsity(groups: Map<string, Group>): number {
    let total = 0;
    let zeros = 0;
    for (const [name, g] of groups) {
      const mask = this.masks.get(name);
      if (!mask) continue;
      total += g.p.length;
      for (let i = 0; i < g.p.length; i++) {
        if (mask[i] === 0) zeros++;
      }
    }
    return total > 0 ? zeros / total : 0;
  }

  /**
   * 幅值剪枝
   */
  magnitudePrune(groups: Map<string, Group>, sparsity: number): PruningResult {
    let removedIndices: number[] = [];
    let prunedParams = 0;

    for (const [name, g] of groups) {
      const absVals = Array.from(g.p).map((v, i) => ({ val: Math.abs(v), idx: i }));
      absVals.sort((a, b) => a.val - b.val);

      const toRemove = Math.floor(g.p.length * sparsity);
      const mask = this.masks.get(name)!;

      for (let i = 0; i < toRemove; i++) {
        mask[absVals[i].idx] = 0;
        removedIndices.push(absVals[i].idx);
      }
      prunedParams += toRemove;
    }

    return {
      originalParams: this.countParams(groups),
      prunedParams,
      sparsity,
      removedIndices,
    };
  }

  /**
   * 梯度剪枝
   */
  gradientPrune(groups: Map<string, Group>, sparsity: number): PruningResult {
    let removedIndices: number[] = [];
    let prunedParams = 0;

    for (const [name, g] of groups) {
      const gradAbs = Array.from(g.g).map((v, i) => ({ val: Math.abs(v), idx: i }));
      gradAbs.sort((a, b) => a.val - b.val);

      const toRemove = Math.floor(g.p.length * sparsity);
      const mask = this.masks.get(name)!;

      for (let i = 0; i < toRemove; i++) {
        mask[gradAbs[i].idx] = 0;
        removedIndices.push(gradAbs[i].idx);
      }
      prunedParams += toRemove;
    }

    return {
      originalParams: this.countParams(groups),
      prunedParams,
      sparsity,
      removedIndices,
    };
  }

  /**
   * 头级剪枝 (Transformer 专用)
   */
  pruneHeads(groups: Map<string, Group>, nHead: number, sparsity: number): number[] {
    const removedHeads: number[] = [];
    const keepRatio = 1 - sparsity;
    const headsToKeep = Math.max(1, Math.round(nHead * keepRatio));

    // 这里简化实现：实际需要根据注意力模式选择重要头
    // 暂时返回所有头都保留
    return removedHeads;
  }

  /**
   * 渐进剪枝
   */
  gradualPrune(groups: Map<string, Group>): void {
    this.cfg.currentStep++;
    const progress = Math.min(1, this.cfg.currentStep / this.cfg.gradualSteps);
    const currentSparsity = this.cfg.targetSparsity * progress;

    if (this.cfg.method === 'magnitude') {
      this.magnitudePrune(groups, currentSparsity);
    } else {
      this.gradientPrune(groups, currentSparsity);
    }
  }

  /**
   * 应用掩码到参数
   */
  applyMask(groups: Map<string, Group>): void {
    for (const [name, g] of groups) {
      const mask = this.masks.get(name);
      if (!mask) continue;
      for (let i = 0; i < g.p.length; i++) {
        g.p[i] *= mask[i];
        g.g[i] *= mask[i];
      }
    }
  }

  /**
   * 恢复参数
   */
  restore(groups: Map<string, Group>): void {
    // 简化：清空掩码
    for (const mask of this.masks.values()) {
      mask.fill(1);
    }
  }

  /**
   * 统计参数数量
   */
  private countParams(groups: Map<string, Group>): number {
    let total = 0;
    for (const g of groups.values()) total += g.p.length;
    return total;
  }

  /**
   * 生成剪枝报告
   */
  report(groups: Map<string, Group>): string {
    const original = this.countParams(groups);
    const sparsity = this.currentSparsity(groups);
    const pruned = Math.floor(original * (1 - sparsity));
    return [
      `== 架构稀疏化报告 ==`,
      `原始参数: ${original.toLocaleString()}`,
      `剪枝参数: ${pruned.toLocaleString()}`,
      `当前稀疏度: ${(sparsity * 100).toFixed(1)}%`,
      `目标稀疏度: ${(this.cfg.targetSparsity * 100).toFixed(1)}%`,
      `剪枝进度: ${this.cfg.currentStep}/${this.cfg.gradualSteps}`,
    ].join('\n');
  }
}

/**
 * MoE 路由稀疏性控制器
 *
 * 控制专家激活的分布，防止专家坍缩
 */
export class RouteSparsifier {
  private expertCounts: Map<string, number> = new Map();
  private totalSteps = 0;
  private targetLoadBalance: number = 0.1; // 每个专家应承担 1/N 的负载

  /**
   * 更新专家计数
   */
  updateExpertCounts(expertIds: number[], nExperts: number): void {
    this.totalSteps++;
    for (let i = 0; i < nExperts; i++) {
      const count = this.expertCounts.get(`expert_${i}`) ?? 0;
      this.expertCounts.set(`expert_${i}`, count + (expertIds.includes(i) ? 1 : 0));
    }
  }

  /**
   * 计算负载均衡损失
   */
  computeLoadBalanceLoss(nExperts: number): number {
    if (this.totalSteps === 0) return 0;
    const frac = Array.from({ length: nExperts }, (_, i) => {
      const count = this.expertCounts.get(`expert_${i}`) ?? 0;
      return count / this.totalSteps;
    });
    // Entropy-based load balance loss
    let loss = 0;
    for (const f of frac) {
      if (f > 0) loss -= f * Math.log(f);
    }
    // 归一化：最大熵 = log(nExperts)
    return 1 - loss / Math.log(nExperts);
  }

  /**
   * 获取专家使用分布
   */
  getDistribution(nExperts: number): number[] {
    if (this.totalSteps === 0) return Array(nExperts).fill(1 / nExperts);
    return Array.from({ length: nExperts }, (_, i) => {
      const count = this.expertCounts.get(`expert_${i}`) ?? 0;
      return count / this.totalSteps;
    });
  }

  /**
   * 重置计数
   */
  reset(): void {
    this.expertCounts.clear();
    this.totalSteps = 0;
  }
}

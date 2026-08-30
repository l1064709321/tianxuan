/**
 * 认知启发知识整合 — 从训练中提炼可复用知识
 *
 * ## 设计原则
 *
 * 1. **模式发现**：从字符序列中发现重复模式（n-gram、短语）
 * 2. **概念形成**：将相似模式聚类为概念
 * 3. **知识固化**：将提炼的知识写入向量库/Titans
 * 4. **迁移学习**：新知识可用于加速后续训练
 *
 * ## 与 STDP 的关系
 *
 * STDP 是突触层面的局部学习规则，
 * 本模块是系统层面的知识提炼，两者互补：
 * - STDP: 微观光滑权重调整
 * - Knowledge Consolidation: 宏观知识结构化
 */

import { Group } from "./model";

export interface KnowledgeItem {
  id: string;
  type: 'pattern' | 'phrase' | 'entity' | 'rule';
  content: string;
  confidence: number;
  frequency: number;
  context: string;       // 出现上下文
  embeddings: number[];  // 语义嵌入
  createdAt: number;     // 时间戳
}

export interface ConceptCluster {
  id: string;
  center: number[];      // 聚类中心（语义空间）
  members: string[];     // 成员知识 ID
  count: number;
  strength: number;      // 聚类强度
}

/**
 * 知识提炼器
 */
export class KnowledgeExtractor {
  private patterns: Map<string, { count: number; contexts: string[] }> = new Map();
  private phrases: Map<string, { count: number; contexts: string[] }> = new Map();
  private entities: Map<string, { count: number; type: string }> = new Map();
  private knowledgeBase: KnowledgeItem[] = [];
  private clusters: ConceptCluster[] = [];

  /**
   * 从训练序列中提取模式
   */
  extractPatterns(seqs: Array<{ ids: number[] }>, tokenizer: { decode: (ids: number[]) => string }): void {
    for (const seq of seqs) {
      const text = tokenizer.decode(seq.ids);

      // 提取 n-gram 模式
      for (let n = 2; n <= 5; n++) {
        for (let i = 0; i <= text.length - n; i++) {
          const gram = text.slice(i, i + n);
          const key = `${n}gram:${gram}`;
          const existing = this.patterns.get(key);
          if (existing) {
            existing.count++;
            if (existing.contexts.length < 3) {
              existing.contexts.push(text.slice(Math.max(0, i - 10), Math.min(text.length, i + n + 10)));
            }
          } else {
            this.patterns.set(key, { count: 1, contexts: [text.slice(Math.max(0, i - 10), Math.min(text.length, i + n + 10))] });
          }
        }
      }

      // 提取短语（标点分隔）
      const sentences = text.split(/[。！？、；]/);
      for (const sent of sentences) {
        const trimmed = sent.trim();
        if (trimmed.length >= 4 && trimmed.length <= 50) {
          const key = `phrase:${trimmed}`;
          const existing = this.phrases.get(key);
          if (existing) {
            existing.count++;
          } else {
            this.phrases.set(key, { count: 1, contexts: [text.slice(0, 100)] });
          }
        }
      }
    }
  }

  /**
   * 提取实体（简化版：基于位置和共现）
   */
  extractEntities(seqs: Array<{ ids: number[] }>, tokenizer: { decode: (ids: number[]) => string }): void {
    // 统计字符共现
    const coOccur: Map<string, Map<string, number>> = new Map();

    for (const seq of seqs) {
      const text = tokenizer.decode(seq.ids);
      const chars = text.split('');

      for (let i = 0; i < chars.length; i++) {
        if (!coOccur.has(chars[i])) coOccur.set(chars[i], new Map());
        const neighborMap = coOccur.get(chars[i])!;

        // 统计前后各 3 个字符的共现
        for (let j = Math.max(0, i - 3); j <= Math.min(chars.length - 1, i + 3); j++) {
          if (i !== j) {
            const count = neighborMap.get(chars[j]) ?? 0;
            neighborMap.set(chars[j], count + 1);
          }
        }
      }
    }

    // 识别高共现字符对（可能是实体）
    for (const [char, neighbors] of coOccur) {
      let totalCooccur = 0;
      for (const count of neighbors.values()) totalCooccur += count;
      const avgCooccur = totalCooccur / Math.max(neighbors.size, 1);

      // 如果某个字符与特定邻居频繁共现，可能是实体的一部分
      for (const [neighbor, count] of neighbors) {
        if (count / Math.max(totalCooccur, 1) > 0.3) {
          const entity = char + neighbor;
          const existing = this.entities.get(entity);
          if (existing) {
            existing.count++;
          } else {
            this.entities.set(entity, { count: 1, type: 'char_pair' });
          }
        }
      }
    }
  }

  /**
   * 构建知识项
   */
  buildKnowledgeItems(embedFunc: (text: string) => number[]): KnowledgeItem[] {
    const items: KnowledgeItem[] = [];
    const timestamp = Date.now();

    // 高频 n-gram 转为知识项
    for (const [key, data] of this.patterns) {
      if (data.count >= 5) {
        const [, content] = key.split(':');
        items.push({
          id: `pat_${key}`,
          type: 'pattern',
          content,
          confidence: Math.min(1, data.count / 20),
          frequency: data.count,
          context: data.contexts[0] ?? '',
          embeddings: embedFunc(content),
          createdAt: timestamp,
        });
      }
    }

    // 高频短语转为知识项
    for (const [key, data] of this.phrases) {
      if (data.count >= 3) {
        const [, content] = key.split(':');
        items.push({
          id: `phrase_${key}`,
          type: 'phrase',
          content,
          confidence: Math.min(1, data.count / 10),
          frequency: data.count,
          context: data.contexts[0] ?? '',
          embeddings: embedFunc(content),
          createdAt: timestamp,
        });
      }
    }

    // 实体转为知识项
    for (const [content, data] of this.entities) {
      if (data.count >= 10) {
        items.push({
          id: `ent_${content}`,
          type: 'entity',
          content,
          confidence: Math.min(1, data.count / 30),
          frequency: data.count,
          context: '',
          embeddings: embedFunc(content),
          createdAt: timestamp,
        });
      }
    }

    return items;
  }

  /**
   * 聚类知识项（简化版：基于 embedding 相似度）
   */
  clusterKnowledge(items: KnowledgeItem[], dim: number, threshold: number = 0.8): ConceptCluster[] {
    const clusters: ConceptCluster[] = [];

    for (const item of items) {
      let foundCluster = false;

      for (const cluster of clusters) {
        // 计算与聚类中心的相似度
        let dot = 0;
        let norm1 = 0;
        let norm2 = 0;
        for (let d = 0; d < dim; d++) {
          dot += item.embeddings[d] * cluster.center[d];
          norm1 += item.embeddings[d] * item.embeddings[d];
          norm2 += cluster.center[d] * cluster.center[d];
        }
        const sim = dot / (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-8);

        if (sim > threshold) {
          cluster.members.push(item.id);
          cluster.count++;
          cluster.strength = Math.max(cluster.strength, sim);
          // 更新中心（指数移动平均）
          for (let d = 0; d < dim; d++) {
            cluster.center[d] = cluster.center[d] * 0.9 + item.embeddings[d] * 0.1;
          }
          foundCluster = true;
          break;
        }
      }

      if (!foundCluster) {
        clusters.push({
          id: `cluster_${clusters.length}`,
          center: item.embeddings.slice(0, dim),
          members: [item.id],
          count: 1,
          strength: 1.0,
        });
      }
    }

    return clusters;
  }

  /**
   * 巩固知识到 Titans/向量库
   */
  consolidateToMemory(
    items: KnowledgeItem[],
    titansWrite: (content: string, embedding: number[]) => void,
    vectorStoreAdd: (text: string, embedding: number[]) => void
  ): void {
    for (const item of items) {
      if (item.confidence > 0.5) {
        titansWrite(item.content, item.embeddings);
        vectorStoreAdd(item.content, item.embeddings);
      }
    }
  }

  /**
   * 获取统计信息
   */
  getStats(): { patterns: number; phrases: number; entities: number; knowledgeItems: number; clusters: number } {
    return {
      patterns: this.patterns.size,
      phrases: this.phrases.size,
      entities: this.entities.size,
      knowledgeItems: this.knowledgeBase.length,
      clusters: this.clusters.length,
    };
  }

  /**
   * 重置提取器
   */
  reset(): void {
    this.patterns.clear();
    this.phrases.clear();
    this.entities.clear();
    this.knowledgeBase = [];
    this.clusters = [];
  }
}

/**
 * 知识检索器
 *
 * 用于推理时检索提炼的知识
 */
export class KnowledgeRetriever {
  private knowledge: Map<string, KnowledgeItem> = new Map();
  private clusters: Map<string, ConceptCluster> = new Map();

  addKnowledge(items: KnowledgeItem[]): void {
    for (const item of items) {
      this.knowledge.set(item.id, item);
    }
  }

  addClusters(clusters: ConceptCluster[]): void {
    for (const cluster of clusters) {
      this.clusters.set(cluster.id, cluster);
    }
  }

  /**
   * 基于内容检索知识
   */
  searchByContent(query: string, topK: number = 5): KnowledgeItem[] {
    const results: Array<{ item: KnowledgeItem; score: number }> = [];

    for (const item of this.knowledge.values()) {
      // 简单字符串匹配（实际应使用 embedding 相似度）
      const score = this.similarity(query, item.content);
      results.push({ item, score });
    }

    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map(r => r.item);
  }

  /**
   * 基于聚类检索
   */
  searchByCluster(clusterId: string, topK: number = 5): KnowledgeItem[] {
    const cluster = this.clusters.get(clusterId);
    if (!cluster) return [];

    return cluster.members
      .slice(0, topK)
      .map(id => this.knowledge.get(id))
      .filter(Boolean) as KnowledgeItem[];
  }

  private similarity(a: string, b: string): number {
    // Jaccard 相似度
    const setA = new Set(a.split(''));
    const setB = new Set(b.split(''));
    let intersection = 0;
    for (const ch of setA) {
      if (setB.has(ch)) intersection++;
    }
    const union = new Set([...setA, ...setB]).size;
    return union > 0 ? intersection / union : 0;
  }
}

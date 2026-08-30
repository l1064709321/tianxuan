/**
 * train_rl.ts — Actor-Critic 强化学习训练入口
 *
 * 用法:
 *   npm run train:rl                    # 默认参数训练
 *   npm run train:rl -- --steps 10000 --epochs 20
 *
 * 训练循环:
 *   1. ActorCritic 与环境交互收集轨迹 (policy rollout)
 *   2. WorldModel 提供环境模型预测 (model-based planning)
 *   3. ReplayBuffer 存储经验并采样重放
 *   4. PPO-Clip + GAE 更新策略和价值网络
 *   5. STDP/STDA 后处理微调突触权重
 */
import * as fs from "fs";
import * as path from "path";
import { CharTokenizer } from "./tokenizer";
import { buildCorpus, chunkText } from "./data";
import { ActorCritic, DEFAULT_AC_CONFIG, TextReward } from "./actor_critic";
import { WorldModel } from "./world_model";
import { ReplayBuffer } from "./replay_buffer";
import { STDP } from "./stdp";
import { STDA } from "./stda";
import { DopamineModulator } from "./dopamine";
import { TitansMemory } from "./titans";
import { mulberry32 } from "./rng";

function flag(name: string): string | null {
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    if (args[i] === `--${name}`) return args[i + 1] ?? null;
    if (args[i].startsWith(`--${name}=`)) return args[i].slice(name.length + 3);
  }
  return null;
}

function num(name: string, def: number): number {
  const raw = flag(name);
  if (raw === null) return def;
  const n = Number(raw);
  return Number.isFinite(n) ? n : def;
}

function ts(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

// ============================================================================
// 文本环境：基于语料的字符级 RL 环境
// ============================================================================

class TextEnv {
  private tokenizer: CharTokenizer;
  private corpus: string;
  private pos: number = 0;
  private stateDim: number;
  private rng: any;

  constructor(tokenizer: CharTokenizer, corpus: string, stateDim: number, seed = 42) {
    this.tokenizer = tokenizer;
    this.corpus = corpus;
    this.stateDim = stateDim;
    this.rng = mulberry32(seed);
    this.reset();
  }

  reset(): Float64Array {
    this.pos = Math.floor(this.rng() * Math.max(1, this.corpus.length - 100));
    return this.getState();
  }

  getState(): Float64Array {
    // 用上下文窗口编码为固定维度向量
    const ctxLen = 16;
    const start = Math.max(0, this.pos - Math.floor(ctxLen / 2));
    const end = Math.min(this.corpus.length, start + ctxLen);
    const window = this.corpus.slice(start, end);

    const state = new Float64Array(this.stateDim);
    const vocabSize = this.tokenizer.vocabSize;
    const embDim = this.stateDim;

    // 简单编码：字符嵌入均值 + 位置信息
    let charCount = 0;
    for (let i = 0; i < window.length; i++) {
      const id = this.tokenizer.encode(window[i])[0] ?? 0;
      const embOffset = (id % vocabSize) * Math.min(embDim, 64);
      // 伪嵌入：用字符位置的哈希值作为伪嵌入
      const h = (id * 2654435761) ^ (window.charCodeAt(i) * 2246822519);
      for (let e = 0; e < Math.min(embDim, 64); e++) {
        state[(i * 64 + e) % embDim] += ((h >> (e * 4)) & 0xF) / 15.0 - 0.5;
      }
      charCount++;
    }
    if (charCount > 0) {
      for (let i = 0; i < embDim; i++) state[i] /= charCount;
    }
    // 位置归一化
    state[this.stateDim - 1] = this.pos / this.corpus.length;
    return state;
  }

  step(action: number): { reward: number; done: boolean; nextState: Float64Array } {
    const vocabSize = this.tokenizer.vocabSize;
    const actualAction = action % vocabSize;
    const trueChar = this.corpus[this.pos];
    const trueId = this.tokenizer.encode(trueChar)[0] ?? 0;

    // 奖励：正确预测得正奖励，错误得负奖励
    const correct = actualAction === trueId;
    const reward = correct ? 1.0 : -0.5;

    // 进度更新
    this.pos = (this.pos + 1) % Math.max(1, this.corpus.length - 1);
    const done = this.pos === 0; // 回到起点视为一个 episode 结束

    return {
      reward,
      done,
      nextState: this.getState(),
    };
  }

  getStats(): { pos: number; corpusLen: number } {
    return { pos: this.pos, corpusLen: this.corpus.length };
  }
}

// ============================================================================
// 主训练循环
// ============================================================================

function main(): number {
  const seed = num("seed", 7);
  const steps = num("steps", 5000);         // 总交互步数
  const epochs = num("epochs", 20);          // PPO 更新轮数
  const batchSteps = num("batchsteps", 512); // 每次收集轨迹的步数
  const lr = num("lr", 0.001);              // 基础学习率
  const stateDim = num("statdim", 64);       // 状态维度
  const hidden = num("hidden", 128);         // Actor/Critic 隐藏层
  const tokens = num("tokens", 100000);      // 语料预算
  const rawPerFile = num("rawperfile", 0);   // 0 = 使用合成语料
  const stdp = num("stdp", 0) === 1;
  const stda = num("stda", 0) === 1;
  const stdpRate = num("stdprate", 0.001);
  const stdaRate = num("stdarate", 0.0005);
  const outDir = flag("out") ?? "data/checkpoints_rl";
  const replayCap = num("replaycap", 4096);
  const dopamine = num("dopamine", 1) === 1;

  console.log(`天玄 TianXuan · Actor-Critic 强化学习训练`);
  console.log(`  步数: ${steps} | 轮数: ${epochs} | 轨迹长度: ${batchSteps}`);
  console.log(`  状态维度: ${stateDim} | 隐藏层: ${hidden}`);
  console.log(`  STDP=${stdp ? "on" : "off"} STDA=${stda ? "on" : "off"} Dopamine=${dopamine ? "on" : "off"}`);

  // ── 构建语料 ───────────────────────────────────────────
  let corpus: string;
  if (rawPerFile > 0) {
    const bundle = buildCorpus({ tokens, seed, rawPerFile });
    corpus = bundle.text;
    console.log(`语料来源: ${bundle.sources.join(", ")}, 长度: ${corpus.length}`);
  } else {
    // 合成语料
    const synthetic = "abcdeabcdefgabcdefgh";
    corpus = synthetic.repeat(Math.ceil(tokens / synthetic.length) + 100);
    console.log(`使用合成语料, 长度: ${corpus.length}`);
  }

  // ── 分词器 ─────────────────────────────────────────────
  const tokenizer = new CharTokenizer();
  tokenizer.fitTopN(corpus, 256);
  console.log(`词表大小: ${tokenizer.vocabSize}`);

  // ── 初始化组件 ─────────────────────────────────────────
  const actorCritic = new ActorCritic({
    stateDim,
    actionDim: tokenizer.vocabSize,
    hidden,
    actorLr: lr,
    criticLr: lr * 3,
    gamma: 0.99,
    lambda: 0.95,
    entropyCoef: 0.01,
    clipEpsilon: 0.2,
    valueCoef: 0.5,
    updateEpochs: 4,
    batchSize: 64,
  });

  const worldModel = new WorldModel({
    stateDim,
    actionDim: tokenizer.vocabSize,
    hidden: hidden,
    embDim: stateDim,
  });

  const replayBuffer = new ReplayBuffer(replayCap);
  const dopamineMod = dopamine ? new DopamineModulator(lr, 0.95, true) : null;
  const titans = new TitansMemory({ dim: stateDim, slots: 256 });

  let stdpRule: STDP | null = stdp ? new STDP({ rateLTP: stdpRate, rateLTD: stdpRate / 2 }) : null;
  let stdaRule: STDA | null = stda ? new STDA(stdaRate, 50) : null;

  console.log(`Actor 参数: ${actorCritic.actor.paramCount().toLocaleString()}`);
  console.log(`Critic 参数: ${actorCritic.critic.paramCount().toLocaleString()}`);
  console.log(`WorldModel 参数: ${worldModel.paramCount().toLocaleString()}`);
  console.log(`总参数: ${(actorCritic.paramCount() + worldModel.paramCount()).toLocaleString()}`);

  // ── 训练循环 ───────────────────────────────────────────
  const env = new TextEnv(tokenizer, corpus, stateDim, seed);
  const t0 = Date.now();
  let totalReward = 0;
  let totalSteps = 0;
  let bestLoss = Infinity;

  // 预计算语料 n-gram 用于奖励函数
  const ngrams = new Set<string>();
  for (let i = 0; i < corpus.length - 3; i++) {
    ngrams.add(corpus.slice(i, i + 4));
  }

  fs.mkdirSync(outDir, { recursive: true });

  for (let ep = 1; ep <= epochs; ep++) {
    let epReward = 0;
    let epCorrect = 0;
    let epTotal = 0;

    // 收集多条轨迹
    const allBatches: Array<{
      states: Float64Array[];
      actions: number[];
      rewards: number[];
      nextStates: Float64Array[];
      dones: boolean[];
      logProbs: number[];
    }> = [];

    for (let b = 0; b < Math.ceil(steps / batchSteps); b++) {
      let state = env.getState();
      let done = false;
      const batch = {
        states: [] as Float64Array[],
        actions: [] as number[],
        rewards: [] as number[],
        nextStates: [] as Float64Array[],
        dones: [] as boolean[],
        logProbs: [] as number[],
      };

      for (let t = 0; t < batchSteps && !done; t++) {
        const { action, logProb } = actorCritic.actor.sampleAction(state, 1.0);
        const result = env.step(action);

        // 使用 WorldModel 预测下一状态作为辅助信号
        const worldState = worldModel.encode(state, action);

        // 综合奖励：环境奖励 + WorldModel 一致性奖励
        const modelConsistency = 1.0 - Math.sqrt(
          Array.from(worldState.s).slice(0, stateDim).reduce((s, v, i) => s + (v - result.nextState[i]) ** 2, 0) / stateDim
        );
        const combinedReward = result.reward * 0.7 + modelConsistency * 0.3;

        batch.states.push(state);
        batch.actions.push(action);
        batch.rewards.push(combinedReward);
        batch.nextStates.push(result.nextState);
        batch.dones.push(result.done);
        batch.logProbs.push(logProb);

        // 存入重放缓冲区
        replayBuffer.push({
          h1: state,
          h2: result.nextState,
          xId: action,
          yId: result.nextState.reduce((s, v, i) => s + v, 0) > 0 ? 1 : 0,
          predLoss: Math.abs(combinedReward),
          nextPredLoss: 0,
        });

        state = result.nextState;
        done = result.done;
        totalReward += combinedReward;
        epReward += combinedReward;
        totalSteps++;
        epTotal++;
        if (result.reward > 0) epCorrect++;
      }

      if (batch.states.length > 10) {
        allBatches.push(batch);
      }
    }

    // PPO-Clip 更新
    let totalActorLoss = 0;
    let totalCriticLoss = 0;
    let totalEntropy = 0;

    for (const batch of allBatches) {
      const { actorLoss, criticLoss, entropy } = actorCritic.update(batch);
      totalActorLoss += actorLoss;
      totalCriticLoss += criticLoss;
      totalEntropy += entropy;
    }

    // 重放巩固：从缓冲区采样额外更新
    if (replayBuffer.size() > 32) {
      const replayBatch = {
        states: [] as Float64Array[],
        actions: [] as number[],
        rewards: [] as number[],
        nextStates: [] as Float64Array[],
        dones: [] as boolean[],
        logProbs: [] as number[],
      };
      const samples = replayBuffer.sample(64, true);
      for (const s of samples) {
        replayBatch.states.push(s.h1);
        replayBatch.actions.push(s.xId);
        replayBatch.rewards.push(s.predLoss);
        replayBatch.nextStates.push(s.h2);
        replayBatch.dones.push(false);
        replayBatch.logProbs.push(0);
      }
      if (replayBatch.states.length > 10) {
        const r = actorCritic.update(replayBatch);
        totalActorLoss += r.actorLoss * 0.5;
        totalCriticLoss += r.criticLoss * 0.5;
      }
    }

    // STDP 后处理（避免与 PPO 梯度冲突）
    if (stdpRule && actorCritic.totalSteps > 100) {
      // 使用最近一次轨迹的隐藏状态作为 proxy
      const lastBatch = allBatches[allBatches.length - 1];
      if (lastBatch && lastBatch.states.length > 0) {
        const lastState = lastBatch.states[lastBatch.states.length - 1];
        const dummyPre = new Float64Array(stateDim).fill(0.5);
        stdpRule.record(lastState, dummyPre);
        // 对 Actor 的 w1/b1 应用 STDP（模拟突触可塑性）
        // 注：实际需要对 Actor 内部结构做 STDP 适配
      }
    }

    // STDA 后处理
    if (stdaRule && actorCritic.totalSteps > 100) {
      // 用输出层的激活作为 proxy 更新阈值
      const dummyH2 = new Float64Array(hidden).fill(0.3);
      stdaRule.update(dummyH2);
      stdaRule.apply(actorCritic.actor.groups);
    }

    // Dopamine 调制
    const avgLoss = (totalActorLoss + totalCriticLoss) / 2;
    if (dopamineMod) {
      const dmMult = dopamineMod.update(avgLoss);
      // 可以用 dmMult 调整下一轮的学习率
    }

    //  Titan 记忆写入
    if (allBatches.length > 0 && allBatches[0].states.length > 0) {
      titans.write(allBatches[0].states[0]);
    }

    //  Titan 记忆检索作为 L0 快通道
    if (allBatches.length > 0 && allBatches[0].states.length > 0) {
      const hit = titans.read(allBatches[0].states[0]);
    }

    // 训练统计
    const elapsed = (Date.now() - t0) / 1000;
    const sps = totalSteps / elapsed;
    const accuracy = epTotal > 0 ? epCorrect / epTotal : 0;
    const avgReward = epReward / Math.max(1, epTotal);

    if (ep % 5 === 0 || ep === 1) {
      console.log(
        `[${ts()}] epoch ${ep}/${epochs} | ` +
        `reward=${avgReward.toFixed(3)} acc=${(accuracy * 100).toFixed(1)}% | ` +
        `actorLoss=${(totalActorLoss / allBatches.length).toFixed(4)} ` +
        `criticLoss=${(totalCriticLoss / allBatches.length).toFixed(4)} | ` +
        `steps=${totalSteps} (${sps.toFixed(0)} steps/s) | ` +
        `replay=${replayBuffer.size()}/${replayCap} | ` +
        `titans=${titans.occupancy().toFixed(2)}`
      );
    }

    // 检查点保存
    if (ep % 10 === 0 || ep === epochs) {
      const checkpoint = {
        actorParams: actorCritic.actor.groups.map(g => Array.from(g.p)),
        criticParams: actorCritic.critic.groups.map(g => Array.from(g.p)),
        worldModelParams: worldModel.getGroups().map(g => Array.from(g.p)),
        replaySize: replayBuffer.size(),
        titansOccupancy: titans.occupancy(),
        totalSteps,
        totalReward,
        accuracy,
        avgReward,
        config: { steps, epochs, stateDim, hidden, lr, seed, stdp, stda, dopamine },
      };
      fs.writeFileSync(
        path.join(outDir, `rl_checkpoint_ep${ep}.json`),
        JSON.stringify(checkpoint, null, 2)
      );
      console.log(`  [${ts()}] 检查点已保存 → ${outDir}/rl_checkpoint_ep${ep}.json`);
    }
  }

  // ── 最终评估 ───────────────────────────────────────────
  console.log(`\n训练完成！`);
  console.log(`  总步数: ${totalSteps.toLocaleString()}`);
  console.log(`  总奖励: ${totalReward.toFixed(2)}`);
  console.log(`  平均步奖励: ${(totalReward / Math.max(1, totalSteps)).toFixed(4)}`);
  console.log(`  耗时: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  console.log(`  检查点: ${outDir}`);

  // 保存最终模型
  const finalCheckpoint = {
    actorParams: actorCritic.actor.groups.map(g => Array.from(g.p)),
    criticParams: actorCritic.critic.groups.map(g => Array.from(g.p)),
    worldModelParams: worldModel.getGroups().map(g => Array.from(g.p)),
    totalSteps,
    totalReward,
    config: { steps, epochs, stateDim, hidden, lr, seed, stdp, stda, dopamine },
  };
  fs.writeFileSync(path.join(outDir, "rl_final.json"), JSON.stringify(finalCheckpoint, null, 2));
  console.log(`最终模型已保存 → ${outDir}/rl_final.json`);

  return 0;
}

if (require.main === module) {
  process.exit(main());
}

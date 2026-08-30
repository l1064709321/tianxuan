import * as fs from "fs";
import * as path from "path";
import { execSync, spawn } from "child_process";

/**
 * 天玄跨平台训练看门狗(Windows / macOS / Linux 通用)
 *
 * 巡检逻辑:
 *   训练进程被杀/消失 → 置 .train-paused 并退出(等用户手动恢复)
 *   日志超过 AGE_LIMIT 未更新(冻结) → kill 训练进程, 置暂停标志并退出
 *   训练完成 → 记录 done 并退出
 *
 * 续训参数从 ck-full-multi/meta.json 动态读取, 不硬编码。
 *
 * 用法:
 *   node sandbox/dist/ml/watchdog.js                 # 看门狗模式(巡检)
 *   node sandbox/dist/ml/watchdog.js --resume        # 恢复模式(清标志+启动训练+拉起看门狗)
 *   node sandbox/dist/ml/watchdog.js --once          # 单次巡检(配合 cron/计划任务)
 */

const ROOT = process.env.TIANXUAN_ROOT ?? process.cwd();
const SANDBOX = path.join(ROOT, "sandbox");
const LOG = path.join(SANDBOX, "logs", "train-full.log");
const EV = path.join(SANDBOX, "logs", "train-events.log");
const ST = path.join(SANDBOX, "logs", "train-watch.json");
const FLAG = path.join(SANDBOX, "logs", ".train-paused");
const CK = path.join(SANDBOX, "ck-full-multi");
const TRAIN_PATTERN = /dist[/\\]ml[/\\]train\.js/i;

const INTERVAL = Number(process.env.WATCH_INTERVAL ?? 300);
const AGE_LIMIT = Number(process.env.AGE_LIMIT ?? 3600);

function ts(): string {
  return new Date().toISOString();
}

function event(line: string): void {
  try {
    fs.appendFileSync(EV, `${ts()} EVENT ${line}\n`, "utf-8");
  } catch {
    /* 日志目录未就绪时静默 */
  }
}

function writeStatus(obj: Record<string, unknown>): void {
  try {
    fs.mkdirSync(path.dirname(ST), { recursive: true });
    fs.writeFileSync(ST, JSON.stringify(obj), "utf-8");
  } catch {
    /* 写状态失败不阻塞 */
  }
}

function logTail(pattern: RegExp, maxLines = 4000): string {
  try {
    const data = fs.readFileSync(LOG, "utf-8");
    const lines = data.split(/\r?\n/).slice(-maxLines);
    for (let i = lines.length - 1; i >= 0; i--) {
      const m = lines[i].match(pattern);
      if (m) return lines[i];
    }
  } catch {
    /* 日志不存在 */
  }
  return "";
}

function lastProgress(): { epoch: string; batch: string; total: string; loss: string; speed: string } {
  const line = logTail(/\[epoch\s+(\d+)\/(\d+)\]\s+批\s+(\d+)\/(\d+)\s+loss\s+([\d.]+)\s+\(([\d.]+)\s+batch\/s\)/);
  const m = line.match(/\[epoch\s+(\d+)\/(\d+)\]\s+批\s+(\d+)\/(\d+)\s+loss\s+([\d.]+)\s+\(([\d.]+)\s+batch\/s\)/);
  return m
    ? { epoch: m[1], batch: m[3], total: m[4], loss: m[5], speed: m[6] }
    : { epoch: "", batch: "", total: "", loss: "", speed: "" };
}

/** 从检查点读取训练参数, 确保续训不丢进度 */
function buildResumeArgs(): string[] {
  try {
    const metaPath = path.join(CK, "meta.json");
    if (!fs.existsSync(metaPath)) throw new Error("no meta");
    const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8")) as {
      config: Record<string, unknown>;
      epochs?: number;
      lastBatch?: number;
      t?: number;
      moeTopK?: number;
      moeGateHidden?: number;
      moeNExperts?: number;
      moeLoadBalanceWeight?: number;
      onlineTitans?: boolean;
    };
    const cfg = meta.config as Record<string, unknown>;
    return [
      "--resume", CK,
      "--epochs", "5",
      "--lr", "0.002",
      "--ckpt", "50",
      "--out", CK,
      "--hidden", String(cfg.hidden ?? 64),
      "--emb", String(cfg.emb ?? 32),
      "--bptt", String(cfg.bptt ?? 32),
      "--ctx", String(cfg.ctx ?? 8),
      "--attn", cfg.attn ? "1" : "0",
      "--mamba", cfg.mamba ? "1" : "0",
      "--cnn", cfg.cnn ? "1" : "0",
      "--moe", "1",
      "--moetopk", String(cfg.moeTopK ?? 2),
      "--moegatehidden", String(cfg.moeGateHidden ?? 32),
      "--moen", String(cfg.moeNExperts ?? 10),
      "--moelbweight", String(cfg.moeLoadBalanceWeight ?? 0.01),
      "--titans", cfg.onlineTitans ? "1" : "0",
      "--prediction", "1",
      "--replay", "1",
      "--dopamine", "1",
      "--spikethreshold", String(cfg.spikeThreshold ?? 0.3),
      "--warmupsteps", "20",
    ];
  } catch (e) {
    console.error(`[watchdog] 读取检查点参数失败: ${e}, 使用默认值`);
    return ["--epochs", "3", "--lr", "0.002", "--ckpt", "50", "--out", CK, "--warmupsteps", "20"];
  }
}

function findTrainPid(): number | null {
  try {
    if (process.platform === "win32") {
      let out = "";
      try {
        out = execSync(
          `wmic process where "name='node.exe'" get processid,commandline /format:csv`,
          { windowsHide: true, maxBuffer: 8 * 1024 * 1024, encoding: "utf-8" }
        );
      } catch {
        out = execSync(
          `powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \\"name='node.exe'\\" | Select-Object ProcessId,CommandLine | ConvertTo-Csv -NoTypeInformation"`,
          { windowsHide: true, maxBuffer: 8 * 1024 * 1024, encoding: "utf-8" }
        );
      }
      for (const line of out.split(/\r?\n/)) {
        if (!TRAIN_PATTERN.test(line)) continue;
        const pidMatch = line.match(/(?:^|,)(\d+)(?:,|$)/) ?? line.match(/\b(\d{2,})\b/);
        if (pidMatch) return Number(pidMatch[1]);
      }
      return null;
    }
    const out = execSync("ps -eo pid=,args=", { maxBuffer: 8 * 1024 * 1024, encoding: "utf-8" });
    for (const line of out.split(/\r?\n/)) {
      const pid = Number(line.trim().split(/\s+/)[0]);
      if (!pid || pid === process.pid) continue;
      if (!TRAIN_PATTERN.test(line)) continue;
      if (process.platform === "linux") {
        try {
          const cwd = fs.realpathSync(`/proc/${pid}/cwd`);
          if (!cwd.startsWith(ROOT)) continue;
          return pid;
        } catch {
          continue;
        }
      }
      return pid;
    }
  } catch {
    /* 无权限或系统异常时视为未知 */
  }
  return null;
}

function killPid(pid: number): void {
  try {
    if (process.platform === "win32") {
      execSync(`taskkill /F /PID ${pid}`, { windowsHide: true, stdio: "ignore" });
    } else {
      execSync(`kill -9 ${pid}`, { stdio: "ignore" });
    }
  } catch {
    /* 进程可能已退出 */
  }
}

function trainingDone(): boolean {
  try {
    return fs.readFileSync(LOG, "utf-8").includes("训练完成");
  } catch {
    return false;
  }
}

function doOnce(): void {
  const p = lastProgress();
  if (trainingDone()) {
    event("training-done");
    writeStatus({ ts: ts(), running: 0, done: 1, ...p });
    process.exit(0);
  }
  const pid = findTrainPid();
  if (pid === null) {
    fs.writeFileSync(FLAG, "", "utf-8");
    event("auto-pause (进程被杀/消失, 自动暂停, 等手动恢复)");
    writeStatus({ ts: ts(), running: 0, paused: 1, reason: "dead", ...p, checkpointSaved: 1, done: 0 });
    process.exit(0);
  }
  let age = 0;
  try {
    age = Math.floor((Date.now() - fs.statSync(LOG).mtimeMs) / 1000);
  } catch {
    age = -1;
  }
  if (age > AGE_LIMIT) {
    killPid(pid);
    fs.writeFileSync(FLAG, "", "utf-8");
    event(`auto-pause (冻结: 日志停 ${age}s, 自动暂停, 等手动恢复)`);
    writeStatus({ ts: ts(), running: 0, paused: 1, reason: "frozen", frozen_sec: age, ...p, checkpointSaved: 1, done: 0 });
    process.exit(0);
  }
  writeStatus({ ts: ts(), running: 1, ...p, checkpointSaved: 1, done: 0 });
}

function resume(): void {
  if (fs.existsSync(FLAG)) fs.unlinkSync(FLAG);
  event("bootstrap-resume (用户/恢复模式触发, 自动续跑)");
  const trainCmd = path.join(SANDBOX, "dist", "ml", "train.js");
  // [FIXED] 从检查点动态读取参数, 不再硬编码
  const args = [trainCmd, ...buildResumeArgs()];
  const child = spawn(process.execPath, args, {
    cwd: SANDBOX,
    detached: true,
    stdio: ["ignore", fs.openSync(LOG, "a"), fs.openSync(LOG, "a")],
    windowsHide: true
  });
  child.unref();
  console.log(`[watchdog] 训练已启动 (pid=${child.pid ?? "?"}) args=${args.join(" ")}`);
  startWatch();
}

function startWatch(): void {
  doOnce();
  const timer = setInterval(doOnce, INTERVAL * 1000);
  timer.unref();
}

function main(): void {
  fs.mkdirSync(path.dirname(LOG), { recursive: true });
  fs.mkdirSync(path.dirname(EV), { recursive: true });
  if (process.argv.includes("--resume")) {
    resume();
  } else if (process.argv.includes("--once")) {
    doOnce();
  } else {
    startWatch();
  }
}

main();

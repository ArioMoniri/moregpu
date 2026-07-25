#!/usr/bin/env node
/**
 * watchdog.mjs — supervises long-running loops and the worker runtime for MoreGPU.
 *
 * Enforces the rules in loops.md:
 *   - per-iteration wall-clock budget (hang guard)
 *   - progress requirement (no-progress escalation)
 *   - clean kill + report instead of silent spinning
 *
 * Usage:
 *   node scripts/watchdog.mjs --cmd "npm test" --budget 300 --label test-loop
 *   node scripts/watchdog.mjs --cmd "node apps/worker/dist/main.js" --budget 0 --heartbeat 15
 *
 * A budget of 0 means "no hard wall-clock kill" (long-lived process) but heartbeat/no-progress
 * supervision still applies.
 */
import { spawn } from 'node:child_process';

function parseArgs(argv) {
  const args = { budget: 300, heartbeat: 10, maxNoProgress: 3, label: 'loop' };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    const v = argv[i + 1];
    if (k === '--cmd') { args.cmd = v; i++; }
    else if (k === '--budget') { args.budget = Number(v); i++; }
    else if (k === '--heartbeat') { args.heartbeat = Number(v); i++; }
    else if (k === '--max-no-progress') { args.maxNoProgress = Number(v); i++; }
    else if (k === '--label') { args.label = v; i++; }
  }
  return args;
}

function log(label, level, msg) {
  const ts = new Date().toISOString();
  process.stdout.write(`[watchdog:${label}] ${ts} ${level} ${msg}\n`);
}

export function superviseProcess({ cmd, budgetSec, heartbeatSec, label, onEvent = () => {} }) {
  return new Promise((resolve) => {
    const child = spawn(cmd, { shell: true, stdio: ['ignore', 'pipe', 'pipe'] });
    let lastOutputAt = Date.now();
    let killed = false;
    const started = Date.now();

    const emit = (type, detail) => onEvent({ type, detail, at: Date.now(), label });

    const bump = (chunk, stream) => {
      lastOutputAt = Date.now();
      process[stream].write(chunk);
    };
    child.stdout.on('data', (c) => bump(c, 'stdout'));
    child.stderr.on('data', (c) => bump(c, 'stderr'));

    const hb = setInterval(() => {
      const idleMs = Date.now() - lastOutputAt;
      const runMs = Date.now() - started;
      emit('heartbeat', { idleMs, runMs });
      if (budgetSec > 0 && runMs > budgetSec * 1000) {
        killed = true;
        log(label, 'HALT', `wall-clock budget ${budgetSec}s exceeded — killing (hang guard)`);
        emit('budget-exceeded', { runMs });
        child.kill('SIGKILL');
      } else if (heartbeatSec > 0 && idleMs > heartbeatSec * 1000 * 6) {
        killed = true;
        log(label, 'HALT', `no output for ${Math.round(idleMs / 1000)}s — presumed hung, killing`);
        emit('stall', { idleMs });
        child.kill('SIGKILL');
      }
    }, Math.max(1, heartbeatSec) * 1000);

    child.on('exit', (code, signal) => {
      clearInterval(hb);
      const runMs = Date.now() - started;
      if (killed) {
        log(label, 'FAIL', `terminated by watchdog after ${Math.round(runMs / 1000)}s`);
        emit('killed', { code, signal, runMs });
        resolve({ ok: false, killedByWatchdog: true, code, signal, runMs });
      } else if (code === 0) {
        log(label, 'OK', `completed cleanly in ${Math.round(runMs / 1000)}s`);
        emit('done', { code, runMs });
        resolve({ ok: true, code, runMs });
      } else {
        log(label, 'FAIL', `exited with code ${code} after ${Math.round(runMs / 1000)}s`);
        emit('exit-nonzero', { code, signal, runMs });
        resolve({ ok: false, code, signal, runMs });
      }
    });
  });
}

/**
 * Supervise an iterative loop of async work. Enforces the no-progress rule from loops.md:
 * each iteration must advance `progressFn()` (a monotonic number) or it counts as no-progress.
 */
export async function superviseLoop({ iterate, progressFn, maxNoProgress = 3, maxIterations = 100, label = 'loop' }) {
  let noProgress = 0;
  let lastProgress = -Infinity;
  for (let i = 0; i < maxIterations; i++) {
    const result = await iterate(i);
    const p = progressFn(result, i);
    if (p > lastProgress) {
      lastProgress = p;
      noProgress = 0;
      log(label, 'PROGRESS', `iteration ${i}: progress=${p}`);
    } else {
      noProgress++;
      log(label, 'WARN', `iteration ${i}: no progress (${noProgress}/${maxNoProgress})`);
      if (noProgress >= maxNoProgress) {
        log(label, 'HALT', `no progress for ${maxNoProgress} iterations — escalating, not spinning`);
        return { converged: false, reason: 'no-progress', iterations: i + 1, progress: lastProgress };
      }
    }
    if (result && result.done) {
      log(label, 'OK', `converged at iteration ${i}`);
      return { converged: true, iterations: i + 1, progress: lastProgress };
    }
  }
  return { converged: false, reason: 'max-iterations', iterations: maxIterations, progress: lastProgress };
}

// CLI entry
if (import.meta.url === `file://${process.argv[1]}`) {
  const args = parseArgs(process.argv);
  if (!args.cmd) {
    log(args.label, 'ERROR', 'missing --cmd');
    process.exit(2);
  }
  log(args.label, 'START', `cmd="${args.cmd}" budget=${args.budget}s heartbeat=${args.heartbeat}s`);
  const res = await superviseProcess({
    cmd: args.cmd,
    budgetSec: args.budget,
    heartbeatSec: args.heartbeat,
    label: args.label,
  });
  process.exit(res.ok ? 0 : 1);
}

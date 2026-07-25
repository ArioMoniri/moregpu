#!/usr/bin/env -S deno run --allow-net --allow-env --allow-read --allow-write
/**
 * MoreGPU coordinator / admin server — self-contained, multi-tenant-isolated, token-secured.
 *
 * Presents your worker fleet as ONE virtual GPU: a Slurm-like job queue accepts tensor jobs, shards
 * them across connected workers, AES-GCM seals each work unit on the wire, pools + verifies results,
 * and exposes a modern web dashboard (fleet, GPU slot, queue, live per-worker load/throttle, errors,
 * and Prometheus /metrics for Grafana). First run is an automated wizard that mints this pool's tokens.
 *
 *   deno run --allow-net --allow-env --allow-read --allow-write \
 *     https://raw.githubusercontent.com/ArioMoniri/moregpu/main/apps/coordinator/server.ts
 *
 * Flags: --help
 * Env: PORT(8787) MOREGPU_BIND(0.0.0.0) MOREGPU_HOST MOREGPU_CONFIG MOREGPU_TLS_CERT+MOREGPU_TLS_KEY MOREGPU_DUTY
 */
if (Deno.args.includes('--help') || Deno.args.includes('-h')) { printHelp(); Deno.exit(0); }

const PORT = Number(Deno.env.get('PORT') ?? 8787);
const BIND = Deno.env.get('MOREGPU_BIND') ?? '0.0.0.0';
const CONFIG_PATH = Deno.env.get('MOREGPU_CONFIG') ?? './.moregpu-server.json';
const ADVERTISE_HOST = Deno.env.get('MOREGPU_HOST') ?? 'localhost';
const DUTY_HINT = Number(Deno.env.get('MOREGPU_DUTY') ?? 0.6);
const CERT_PATH = Deno.env.get('MOREGPU_TLS_CERT');
const KEY_PATH = Deno.env.get('MOREGPU_TLS_KEY');
const C = { reset: '\x1b[0m', dim: '\x1b[2m', b: '\x1b[1m', cyan: '\x1b[36m', grn: '\x1b[32m', yel: '\x1b[33m', mag: '\x1b[35m', red: '\x1b[31m' };

// ---------- base64 + sealing ----------
function b64e(u8: Uint8Array): string { let s = ''; const K = 0x8000; for (let i = 0; i < u8.length; i += K) s += String.fromCharCode(...u8.subarray(i, i + K)); return btoa(s); }
function b64d(s: string): Uint8Array { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
function tokenB64url(n = 24): string { return b64e(crypto.getRandomValues(new Uint8Array(n))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''); }
async function importKey(raw: Uint8Array) { return crypto.subtle.importKey('raw', raw as BufferSource, 'AES-GCM', false, ['encrypt', 'decrypt']); }
async function seal(key: Uint8Array, plain: Uint8Array) { const iv = crypto.getRandomValues(new Uint8Array(12)); const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv: iv as BufferSource }, await importKey(key), plain as BufferSource); return { iv: b64e(iv), ct: b64e(new Uint8Array(ct)) }; }
async function unseal(key: Uint8Array, blob: { iv: string; ct: string }) { return new Uint8Array(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: b64d(blob.iv) as BufferSource }, await importKey(key), b64d(blob.ct) as BufferSource)); }
const f32ToB64 = (a: Float32Array) => b64e(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
const b64ToF32 = (s: string) => new Float32Array(b64d(s).buffer);
const constEq = (a: string, b: string) => { if (a.length !== b.length) return false; let d = 0; for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i); return d === 0; };

// ---------- structured log ring buffer (errors + debug) ----------
type Level = 'info' | 'warn' | 'error' | 'debug';
interface LogEntry { ts: number; level: Level; msg: string; ctx?: string }
const LOG: LogEntry[] = [];
function log(level: Level, msg: string, ctx?: string) {
  const e: LogEntry = { ts: Date.now(), level, msg, ctx };
  LOG.push(e); if (LOG.length > 500) LOG.shift();
  const col = level === 'error' ? C.red : level === 'warn' ? C.yel : level === 'debug' ? C.dim : C.cyan;
  console.log(`${col}[coord:${level}]${C.reset} ${msg}${ctx ? C.dim + ' · ' + ctx + C.reset : ''}`);
}

// ---------- config / wizard ----------
interface Config { adminToken: string; joinToken: string; tenantKeyB64: string; created: string; }
async function loadOrInitConfig(): Promise<{ cfg: Config; fresh: boolean }> {
  try { return { cfg: JSON.parse(await Deno.readTextFile(CONFIG_PATH)), fresh: false }; }
  catch {
    const cfg: Config = { adminToken: tokenB64url(24), joinToken: tokenB64url(18), tenantKeyB64: b64e(crypto.getRandomValues(new Uint8Array(32))), created: new Date().toISOString() };
    await Deno.writeTextFile(CONFIG_PATH, JSON.stringify(cfg, null, 2));
    return { cfg, fresh: true };
  }
}
const { cfg, fresh } = await loadOrInitConfig();
const TENANT_KEY = b64d(cfg.tenantKeyB64);
const scheme = CERT_PATH && KEY_PATH ? 'wss' : 'ws';
const httpScheme = CERT_PATH && KEY_PATH ? 'https' : 'http';
const RAW = 'https://raw.githubusercontent.com/ArioMoniri/moregpu/main';

function art() {
  return `${C.cyan}${C.b}
   __  __            ____ ____  _   _
  |  \\/  | ___  _ __/ ___|  _ \\| | | |   ${C.reset}${C.dim}native GPU compute pool${C.cyan}${C.b}
  | |\\/| |/ _ \\| '__| |  _| |_) | | | |
  | |  | | (_) | |  | |_| |  __/| |_| |
  |_|  |_|\\___/|_|   \\____|_|    \\___/${C.reset}`;
}
function wizardBanner() {
  const wsUrl = `${scheme}://${ADVERTISE_HOST}:${PORT}/ws`;
  console.log(art());
  console.log(`\n  ${C.b}${fresh ? C.grn + 'NEW POOL CREATED' : 'pool ready'}${C.reset}`);
  console.log(`  ${C.dim}dashboard${C.reset}  ${httpScheme}://${ADVERTISE_HOST}:${PORT}`);
  console.log(`  ${C.dim}admin token${C.reset}  ${C.yel}${cfg.adminToken}${C.reset}   ${C.dim}(controls the pool — keep secret)${C.reset}`);
  console.log(`  ${C.dim}join token${C.reset}   ${C.mag}${cfg.joinToken}${C.reset}   ${C.dim}(lets machines enroll)${C.reset}`);
  console.log(`  ${C.dim}config${C.reset}       ${CONFIG_PATH}`);
  console.log(`\n  ${C.b}Add a machine to this pool${C.reset} ${C.dim}(Linux/macOS)${C.reset}:`);
  console.log(`    ${C.grn}curl -fsSL ${RAW}/scripts/install.sh \\`);
  console.log(`      | MOREGPU_SERVER=${wsUrl} MOREGPU_TOKEN=${cfg.joinToken} sh${C.reset}`);
  console.log(`  ${C.dim}Windows:  $env:MOREGPU_SERVER="${wsUrl}"; $env:MOREGPU_TOKEN="${cfg.joinToken}"; irm ${RAW}/scripts/install.ps1 | iex${C.reset}`);
  console.log(`  ${C.dim}reboot-surviving service: add MOREGPU_SERVICE=1 · type /help in this console's HTTP: ${httpScheme}://${ADVERTISE_HOST}:${PORT}/help${C.reset}\n`);
}
function printHelp() {
  console.log(art());
  console.log(`
  ${C.b}MoreGPU admin server${C.reset} — presents your worker fleet as one virtual GPU.

  ${C.b}Run${C.reset}
    deno run --allow-net --allow-env --allow-read --allow-write ${RAW}/apps/coordinator/server.ts

  ${C.b}Environment${C.reset}
    PORT=8787              HTTP/WebSocket port
    MOREGPU_BIND=0.0.0.0   bind address (0.0.0.0 = reachable from other machines)
    MOREGPU_HOST=localhost hostname advertised in the wizard/worker command
    MOREGPU_CONFIG=./.moregpu-server.json   pool identity (tokens + key)
    MOREGPU_TLS_CERT / MOREGPU_TLS_KEY      PEM paths → serve https + wss
    MOREGPU_DUTY=0.6       duty-cycle ceiling hint sent to workers

  ${C.b}HTTP API${C.reset} ${C.dim}(admin endpoints need: Authorization: Bearer <admin token>)${C.reset}
    GET  /                 dashboard (fleet, GPU slot, queue, logs)
    GET  /help             this help as JSON/text
    GET  /health           liveness + fleet size (public)
    GET  /gpu              the pool as one virtual GPU device (admin)
    GET  /workers          connected workers + live load/throttle (admin)
    POST /submit           {"kernel":"matmul","size":1024}  run a job (admin)
    GET  /jobs             job queue + history (admin)
    GET  /jobs/:id         one job (admin)
    GET  /logs             recent errors/debug (admin)
    GET  /metrics          Prometheus metrics for Grafana (admin)

  ${C.b}Kernels${C.reset}  matmul · vector_add · vector_mul · saxpy · relu · scale   ${C.dim}(extensible)${C.reset}
`);
}

// ---------- metrics ----------
const M = { jobsTotal: 0, jobsDone: 0, jobsFailed: 0, shardsDone: 0, shardsFailed: 0, gpuShards: 0, cpuShards: 0 };
const KM: Record<string, number> = {}; // jobs per kernel

// ---------- worker registry ----------
interface Worker {
  id: string; backend: string; label: string; os: string; ws: WebSocket;
  load1: number; util: number; duty: number; busy: boolean; joinedAt: number;
  // contribution accounting
  shards: number; units: number; errors: number; totalMs: number;
  lastUnits: number; history: number[]; // per-sample units completed → sparkline trend
}
const workers = new Map<string, Worker>();
const pending = new Map<string, (r: { ok: boolean; sealedOut?: { iv: string; ct: string }; error?: string; backend?: string; ms?: number }) => void>();

function wireWorker(ws: WebSocket) {
  let id = '';
  ws.onmessage = (ev) => {
    let m: Record<string, unknown>;
    try { m = JSON.parse(ev.data as string); } catch { log('warn', 'bad frame from worker'); return; }
    if (m.t === 'register') {
      if (!constEq(String(m.joinToken ?? ''), cfg.joinToken)) { ws.send(JSON.stringify({ t: 'denied', reason: 'bad join token' })); log('warn', 'worker rejected: bad join token'); ws.close(); return; }
      const node = m.node as { id: string; backend: string; label: string; os: string };
      id = node.id;
      workers.set(id, { id, backend: node.backend, label: node.label, os: node.os, ws, load1: 0, util: 0, duty: DUTY_HINT, busy: false, joinedAt: Date.now(), shards: 0, units: 0, errors: 0, totalMs: 0, lastUnits: 0, history: [] });
      ws.send(JSON.stringify({ t: 'welcome', tenantKeyB64: b64e(TENANT_KEY), duty: DUTY_HINT }));
      log('info', `worker joined: ${id} (${node.label}, ${node.os}) · fleet=${workers.size}`);
      pumpQueue();
    } else if (m.t === 'heartbeat') {
      const w = workers.get(String(m.id)); if (w) { w.load1 = Number(m.load1) || 0; w.util = Number(m.util) || 0; w.duty = Number(m.duty) || w.duty; }
    } else if (m.t === 'result') {
      pending.get(String(m.shardId))?.({ ok: m.ok as boolean, sealedOut: m.sealedOut as { iv: string; ct: string } | undefined, error: m.error as string | undefined, backend: m.backend as string | undefined, ms: m.ms as number | undefined });
    }
  };
  ws.onclose = () => { if (id) { workers.delete(id); log('info', `worker left: ${id} · fleet=${workers.size}`); } };
  ws.onerror = () => log('warn', `socket error${id ? ' from ' + id : ''}`);
}

// ---------- contribution trend sampler (per-worker + pool throughput sparklines) ----------
const poolHistory: number[] = [];
let lastPoolUnits = 0;
setInterval(() => {
  let totalUnits = 0;
  for (const w of workers.values()) {
    const delta = Math.max(0, w.units - w.lastUnits); w.lastUnits = w.units;
    w.history.push(delta); if (w.history.length > 30) w.history.shift();
    totalUnits += w.units;
  }
  const pd = Math.max(0, totalUnits - lastPoolUnits); lastPoolUnits = totalUnits;
  poolHistory.push(pd); if (poolHistory.length > 30) poolHistory.shift();
}, 3000);

// ---------- kernels ----------
const ELEMENTWISE = new Set(['vector_add', 'vector_mul', 'saxpy', 'relu', 'scale']);
function cpuKernel(kernel: string, a: Float32Array, b: Float32Array | null, scalar: number): Float32Array {
  const n = a.length, o = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    switch (kernel) {
      case 'vector_add': o[i] = a[i] + (b as Float32Array)[i]; break;
      case 'vector_mul': o[i] = a[i] * (b as Float32Array)[i]; break;
      case 'saxpy': o[i] = scalar * a[i] + (b as Float32Array)[i]; break;
      case 'relu': o[i] = a[i] > 0 ? a[i] : 0; break;
      case 'scale': o[i] = a[i] * scalar; break;
    }
  }
  return o;
}
function cpuMatmul(a: Float32Array, b: Float32Array, Mm: number, N: number, K: number): Float32Array {
  const o = new Float32Array(Mm * N);
  for (let i = 0; i < Mm; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < K; k++) s += a[i * K + k] * b[k * N + j]; o[i * N + j] = s; }
  return o;
}

// ---------- jobs + Slurm-like queue ----------
type JobStatus = 'queued' | 'running' | 'done' | 'failed';
interface JobRec { id: string; status: JobStatus; kernel: string; size: number; sealed: boolean; submittedAt: number; ms?: number; gflops?: number; verified?: boolean; shards?: { worker: string; backend: string; work: number; ms: number }[]; error?: string; }
const jobs = new Map<string, JobRec>();
const queue: JobRec[] = [];
let jobSeq = 0, shardSeq = 0, draining = false;

function submit(kernel: string, size: number): JobRec {
  const id = `job-${++jobSeq}`;
  const rec: JobRec = { id, status: 'queued', kernel, size, sealed: true, submittedAt: Date.now() };
  jobs.set(id, rec); queue.push(rec); M.jobsTotal++; KM[kernel] = (KM[kernel] ?? 0) + 1;
  log('info', `queued ${id}: ${kernel} size=${size}`, `queue depth ${queue.length}`);
  pumpQueue();
  return rec;
}

async function pumpQueue() {
  if (draining) return;
  draining = true;
  try {
    while (queue.length && workers.size > 0) {
      const rec = queue.shift()!;
      rec.status = 'running';
      try { await runJob(rec); rec.status = 'done'; M.jobsDone++; }
      catch (e) { rec.status = 'failed'; rec.error = String(e); M.jobsFailed++; log('error', `${rec.id} failed: ${rec.error}`); }
    }
  } finally { draining = false; }
}

async function dispatchShard(w: Worker, jobId: string, payload: Record<string, unknown>): Promise<Float32Array> {
  const shardId = `s-${++shardSeq}`;
  const sealedIn = await seal(TENANT_KEY, new TextEncoder().encode(JSON.stringify(payload)));
  const done = new Promise<{ ok: boolean; sealedOut?: { iv: string; ct: string }; error?: string; backend?: string; ms?: number }>((res) => pending.set(shardId, res));
  w.busy = true;
  w.ws.send(JSON.stringify({ t: 'assign', shardId, jobId, sealedIn }));
  const r = await done; pending.delete(shardId); w.busy = false;
  if (!r.ok || !r.sealedOut) { M.shardsFailed++; w.errors++; throw new Error(`shard on ${w.id} failed: ${r.error}`); }
  M.shardsDone++; (r.backend?.startsWith('gpu') ? M.gpuShards++ : M.cpuShards++);
  const outObj = JSON.parse(new TextDecoder().decode(await unseal(TENANT_KEY, r.sealedOut)));
  const out = b64ToF32(outObj.out);
  w.shards++; w.units += out.length; w.totalMs += r.ms ?? 0; // contribution accounting
  return out;
}

async function runJob(rec: JobRec) {
  const fleet = [...workers.values()];
  const t0 = performance.now();
  if (rec.kernel === 'matmul') {
    const N = rec.size, A = new Float32Array(N * N).map(() => Math.random()), B = new Float32Array(N * N).map(() => Math.random());
    const rowsPer = Math.ceil(N / fleet.length);
    const parts = await Promise.all(fleet.map(async (w, i) => {
      const r0 = i * rowsPer, rows = Math.min(N, r0 + rowsPer) - r0; if (rows <= 0) return null;
      const out = await dispatchShard(w, rec.id, { kernel: 'matmul', a: f32ToB64(A.slice(r0 * N, (r0 + rows) * N)), b: f32ToB64(B), rows, N, K: N });
      return { r0, out, w };
    }));
    const Cm = new Float32Array(N * N); const sh: JobRec['shards'] = [];
    for (const p of parts) if (p) { Cm.set(p.out, p.r0 * N); sh.push({ worker: p.w.id, backend: p.w.label, work: p.out.length / N, ms: 0 }); }
    const wall = performance.now() - t0;
    if (N <= 640) { let md = 0; const ref = cpuMatmul(A, B, N, N, N); for (let i = 0; i < Cm.length; i++) md = Math.max(md, Math.abs(Cm[i] - ref[i])); rec.verified = md < 1e-2; }
    rec.gflops = (2 * N * N * N) / (wall / 1000) / 1e9; rec.ms = wall; rec.shards = sh;
  } else if (ELEMENTWISE.has(rec.kernel)) {
    const n = rec.size, a = new Float32Array(n).map(() => Math.random() * 2 - 1), b = new Float32Array(n).map(() => Math.random());
    const per = Math.ceil(n / fleet.length);
    const parts = await Promise.all(fleet.map(async (w, i) => {
      const s0 = i * per, len = Math.min(n, s0 + per) - s0; if (len <= 0) return null;
      const needB = rec.kernel !== 'relu' && rec.kernel !== 'scale';
      const out = await dispatchShard(w, rec.id, { kernel: rec.kernel, a: f32ToB64(a.subarray(s0, s0 + len)), b: needB ? f32ToB64(b.subarray(s0, s0 + len)) : undefined, scalar: 2, len });
      return { s0, out, w };
    }));
    const O = new Float32Array(n); const sh: JobRec['shards'] = [];
    for (const p of parts) if (p) { O.set(p.out, p.s0); sh.push({ worker: p.w.id, backend: p.w.label, work: p.out.length, ms: 0 }); }
    const ref = cpuKernel(rec.kernel, a, b, 2); let md = 0; for (let i = 0; i < n; i++) md = Math.max(md, Math.abs(O[i] - ref[i]));
    rec.verified = md < 1e-3; rec.ms = performance.now() - t0; rec.shards = sh;
  } else { throw new Error(`unknown kernel ${rec.kernel}`); }
  log('info', `${rec.id} done: ${rec.kernel} · ${(rec.ms ?? 0).toFixed(0)}ms · verified=${rec.verified}`);
}

// ---------- virtual GPU view ----------
function virtualGpu() {
  const fleet = [...workers.values()];
  const slots = fleet.length;
  const gpu = fleet.filter((w) => w.backend === 'gpu').length;
  const avgUtil = slots ? fleet.reduce((s, w) => s + w.util, 0) / slots : 0;
  const avgDuty = slots ? fleet.reduce((s, w) => s + w.duty, 0) / slots : 0;
  const totalUnits = fleet.reduce((s, w) => s + w.units, 0);
  const totalShards = fleet.reduce((s, w) => s + w.shards, 0);
  return { device: 'MoreGPU-Pool', slots, gpuSlots: gpu, cpuSlots: slots - gpu, busy: fleet.filter((w) => w.busy).length, avgUserUtil: +avgUtil.toFixed(3), avgPoolDuty: +avgDuty.toFixed(3), queueDepth: queue.length, totalUnits, totalShards, poolTrend: poolHistory, perKernel: KM, sealed: 'AES-256-GCM' };
}

// ---------- HTTP + WS ----------
const json = (o: unknown, status = 200) => new Response(JSON.stringify(o, null, 2), { status, headers: { 'content-type': 'application/json' } });
const authOk = (req: Request) => constEq((req.headers.get('authorization') ?? '').replace(/^Bearer\s+/i, '') || (req.headers.get('x-admin-token') ?? ''), cfg.adminToken);
const KERNELS = ['matmul', 'vector_add', 'vector_mul', 'saxpy', 'relu', 'scale'];

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  if (url.pathname === '/ws') { const { socket, response } = Deno.upgradeWebSocket(req); wireWorker(socket); return response; }
  if (url.pathname === '/health') return json({ ok: true, fleet: workers.size, queue: queue.length });
  if (url.pathname === '/help') return json({ kernels: KERNELS, endpoints: ['/ (dashboard)', '/health', '/gpu', '/workers', 'POST /submit', '/jobs', '/jobs/:id', '/logs', '/metrics'], auth: 'admin endpoints require Authorization: Bearer <admin token>' });
  // admin-gated
  if (['/gpu', '/workers', '/jobs', '/logs', '/metrics'].some((p) => url.pathname === p || url.pathname.startsWith('/jobs/')) || (req.method === 'POST' && url.pathname === '/submit')) {
    if (!authOk(req)) return json({ error: 'unauthorized — send Authorization: Bearer <admin token>' }, 401);
  }
  if (url.pathname === '/gpu') return json(virtualGpu());
  if (url.pathname === '/workers') {
    const total = [...workers.values()].reduce((s, w) => s + w.units, 0) || 1;
    return json([...workers.values()].map((w) => ({
      id: w.id, backend: w.backend, label: w.label, os: w.os, userUtil: w.util, poolDuty: w.duty, busy: w.busy,
      shards: w.shards, units: w.units, share: +(w.units / total).toFixed(3), errors: w.errors,
      avgMs: w.shards ? +(w.totalMs / w.shards).toFixed(1) : 0, uptimeS: Math.round((Date.now() - w.joinedAt) / 1000), trend: w.history,
    })));
  }
  if (url.pathname === '/logs') return json(LOG.slice(-200).reverse());
  if (url.pathname === '/metrics') return new Response(prometheus(), { headers: { 'content-type': 'text/plain; version=0.0.4' } });
  if (req.method === 'POST' && url.pathname === '/submit') {
    const body = await req.json().catch(() => ({})) as { kernel?: string; size?: number };
    const kernel = KERNELS.includes(body.kernel ?? '') ? body.kernel! : 'matmul';
    const size = Math.max(16, Math.min(kernel === 'matmul' ? 2048 : 8_000_000, Number(body.size ?? (kernel === 'matmul' ? 512 : 1_000_000))));
    if (workers.size === 0) { const r = submit(kernel, size); return json({ ...r, note: 'queued — no workers connected yet; will run when one joins' }, 202); }
    const rec = submit(kernel, size);
    // wait briefly for it to finish for a synchronous response
    for (let i = 0; i < 600 && rec.status !== 'done' && rec.status !== 'failed'; i++) await new Promise((r) => setTimeout(r, 50));
    return json(rec, rec.status === 'failed' ? 503 : 200);
  }
  if (url.pathname === '/jobs') return json([...jobs.values()].slice(-50).reverse());
  if (url.pathname.startsWith('/jobs/')) { const r = jobs.get(url.pathname.slice(6)); return r ? json(r) : json({ error: 'not found' }, 404); }
  return new Response(dashboard(), { headers: { 'content-type': 'text/html' } });
}

function prometheus(): string {
  const g = virtualGpu();
  const total = g.totalUnits || 1;
  const lines = [
    '# HELP moregpu_fleet Connected workers', '# TYPE moregpu_fleet gauge', `moregpu_fleet ${g.slots}`,
    `moregpu_gpu_slots ${g.gpuSlots}`, `moregpu_cpu_slots ${g.cpuSlots}`, `moregpu_busy ${g.busy}`,
    `moregpu_queue_depth ${g.queueDepth}`, `moregpu_avg_user_util ${g.avgUserUtil}`, `moregpu_avg_pool_duty ${g.avgPoolDuty}`,
    `moregpu_total_units ${g.totalUnits}`, `moregpu_total_shards ${g.totalShards}`,
    '# TYPE moregpu_jobs_total counter', `moregpu_jobs_total ${M.jobsTotal}`, `moregpu_jobs_done ${M.jobsDone}`, `moregpu_jobs_failed ${M.jobsFailed}`,
    `moregpu_shards_done ${M.shardsDone}`, `moregpu_shards_failed ${M.shardsFailed}`, `moregpu_gpu_shards ${M.gpuShards}`, `moregpu_cpu_shards ${M.cpuShards}`,
    '# HELP moregpu_worker_units Units of work completed per worker', '# TYPE moregpu_worker_units counter',
  ];
  for (const w of workers.values()) {
    const lbl = `{worker="${w.id.replace(/"/g, '')}",backend="${w.backend}"}`;
    lines.push(`moregpu_worker_units${lbl} ${w.units}`);
    lines.push(`moregpu_worker_share${lbl} ${(w.units / total).toFixed(4)}`);
    lines.push(`moregpu_worker_shards${lbl} ${w.shards}`);
    lines.push(`moregpu_worker_errors${lbl} ${w.errors}`);
    lines.push(`moregpu_worker_user_util${lbl} ${w.util}`);
    lines.push(`moregpu_worker_pool_duty${lbl} ${w.duty}`);
  }
  return lines.join('\n') + '\n';
}

const serveOpts: Deno.ServeTcpOptions & { cert?: string; key?: string } = { port: PORT, hostname: BIND, onListen: () => wizardBanner() };
if (CERT_PATH && KEY_PATH) { serveOpts.cert = await Deno.readTextFile(CERT_PATH); serveOpts.key = await Deno.readTextFile(KEY_PATH); }
Deno.serve(serveOpts, handler);

function dashboard(): string {
  return DASHBOARD_HTML;
}
const DASHBOARD_HTML = `<!doctype html><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1"><title>MoreGPU · admin</title>
<style>
:root{--bg:#0b0f17;--card:#121826;--line:#1f2937;--ink:#e5e7eb;--mut:#8b98ad;--acc:#6366f1;--grn:#34d399;--red:#f87171;--yel:#fbbf24}
*{box-sizing:border-box}body{margin:0;font:14px ui-sans-serif,system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--ink)}
.wrap{max-width:1080px;margin:0 auto;padding:28px 22px}
h1{font-size:20px;letter-spacing:-.3px;margin:0}.sub{color:var(--mut);font-size:13px;margin:2px 0 18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.card h3{margin:0 0 10px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);font-weight:600}
.big{font-size:30px;font-weight:700;letter-spacing:-1px}.mut{color:var(--mut)}
.gpu{background:linear-gradient(135deg,#1e213a,#141828);border:1px solid #2a2f52}
.bar{height:7px;border-radius:6px;background:#1f2637;overflow:hidden;margin-top:8px}.bar>i{display:block;height:100%;background:var(--acc)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600}
.gpuP{background:#132a22;color:var(--grn)}.cpuP{background:#2a2413;color:var(--yel)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}
input,select{background:#0e1420;border:1px solid var(--line);color:var(--ink);border-radius:9px;padding:9px}
button{background:var(--acc);color:#fff;border:0;border-radius:9px;padding:9px 16px;font-weight:600;cursor:pointer}
button.ghost{background:#1a2133;color:#c7d2fe}
pre{background:#0e1420;border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto;max-height:260px;font-size:12px;margin:0}
.lvl-error{color:var(--red)}.lvl-warn{color:var(--yel)}.lvl-info{color:#93c5fd}.lvl-debug{color:var(--mut)}
a{color:#a5b4fc}.sp{margin-top:16px}.k{display:inline-block;background:#1a2133;color:#c7d2fe;border-radius:7px;padding:3px 9px;margin:3px 4px 0 0;font-size:12px;font-family:ui-monospace,monospace}
</style>
<div class=wrap>
<h1>MoreGPU · admin</h1><div class=sub>Your worker fleet, presented as one virtual GPU.</div>
<div class=card style="margin-bottom:14px"><div class=row><b>Admin token</b><input id=tok type=password placeholder="paste from the server console wizard" style="flex:1;min-width:220px"><button class=ghost onclick=save()>Save</button><span id=authmsg class=mut></span></div></div>
<div class="grid">
  <div class="card gpu"><h3>Virtual GPU</h3><div class=big id=slots>—</div><div class=mut id=slotsub>slots</div>
    <div class=mut style="margin-top:10px">user load <span id=uu>–</span></div><div class=bar><i id=uubar style=width:0%></i></div>
    <div class=mut style="margin-top:6px">pool duty <span id=pd>–</span></div><div class=bar><i id=pdbar style="width:0%;background:var(--grn)"></i></div>
    <div class=mut style="margin-top:10px">throughput · <span id=units>0</span> units total</div><div id=tpspark style="margin-top:4px"></div>
  </div>
  <div class=card><h3>Queue</h3><div class=big id=q>0</div><div class=mut>waiting jobs</div>
    <div class=sp><span class=mut>done</span> <b id=jd>0</b> · <span class=mut>failed</span> <b id=jf>0</b></div></div>
  <div class=card><h3>Sealing</h3><div class=big style=font-size:20px>AES-256-GCM</div><div class=mut>every work unit, on the wire</div>
    <div class=sp><a href=/metrics target=_blank>/metrics</a> <span class=mut>· wire to Grafana</span></div></div>
</div>
<div class="card sp"><h3>Run a job</h3><div class=row>
  <select id=kernel><option>matmul</option><option>vector_add</option><option>vector_mul</option><option>saxpy</option><option>relu</option><option>scale</option></select>
  <label class=mut>size <input id=size value=512 style=width:110px></label><button onclick=submit()>Submit</button><span id=jobmsg class=mut></span></div></div>
<div class="card sp"><h3>Fleet — live contribution</h3><table><thead><tr><th>worker</th><th>type</th><th>share</th><th>trend</th><th>shards</th><th>units</th><th>avg ms</th><th>user load</th><th>pool duty</th><th>state</th></tr></thead><tbody id=fleet><tr><td class=mut colspan=10>connect a worker…</td></tr></tbody></table></div>
<div class="card sp"><h3>Per-kernel jobs</h3><div id=kernels class=mut>—</div></div>
<div class="card sp"><h3>Errors &amp; debug log</h3><pre id=logs>—</pre></div>
</div>
<script>
const K='moregpu_admin_token';document.getElementById('tok').value=localStorage.getItem(K)||'';
const H=()=>({'content-type':'application/json','authorization':'Bearer '+(localStorage.getItem(K)||'')});
function save(){localStorage.setItem(K,document.getElementById('tok').value.trim());refresh();}
function pct(x){return Math.round((x||0)*100)+'%';}
function fmt(n){n=n||0;return n>=1e9?(n/1e9).toFixed(1)+'G':n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':''+n;}
// inline SVG sparkline from an array of values
function spark(arr,w,h,col){arr=arr||[];if(!arr.length)return '<svg width='+w+' height='+h+'></svg>';
 const mx=Math.max(1,...arr);const step=w/Math.max(1,arr.length-1);
 const pts=arr.map((v,i)=>(i*step).toFixed(1)+','+(h-2-(v/mx)*(h-4)).toFixed(1)).join(' ');
 return '<svg width='+w+' height='+h+' viewBox="0 0 '+w+' '+h+'" preserveAspectRatio=none><polyline points="'+pts+'" fill=none stroke="'+col+'" stroke-width=1.5 stroke-linejoin=round stroke-linecap=round/></svg>';}
async function refresh(){
 try{
  const g=await (await fetch('/gpu',{headers:H()})).json();
  if(g.error){document.getElementById('authmsg').textContent='enter admin token';return;}
  document.getElementById('authmsg').textContent='';
  document.getElementById('slots').textContent=g.slots;
  document.getElementById('slotsub').textContent=g.gpuSlots+' GPU · '+g.cpuSlots+' CPU · '+g.busy+' busy';
  document.getElementById('uu').textContent=pct(g.avgUserUtil);document.getElementById('uubar').style.width=pct(g.avgUserUtil);
  document.getElementById('pd').textContent=pct(g.avgPoolDuty);document.getElementById('pdbar').style.width=pct(g.avgPoolDuty);
  document.getElementById('units').textContent=fmt(g.totalUnits);
  document.getElementById('tpspark').innerHTML=spark(g.poolTrend,190,30,'#6366f1');
  document.getElementById('q').textContent=g.queueDepth;
  const pk=g.perKernel||{};const ks=Object.keys(pk);
  document.getElementById('kernels').innerHTML=ks.length?ks.map(k=>'<span class=k>'+k+' · '+pk[k]+'</span>').join(' '):'no jobs yet';
  const w=await (await fetch('/workers',{headers:H()})).json();
  document.getElementById('fleet').innerHTML=w.length?w.map(x=>'<tr><td>'+x.id+(x.errors?' <span class=lvl-error>('+x.errors+' err)</span>':'')+'</td><td><span class="pill '+(x.backend==='gpu'?'gpuP':'cpuP')+'">'+x.backend+'</span></td><td><b>'+pct(x.share)+'</b></td><td>'+spark(x.trend,72,20,x.backend==='gpu'?'#34d399':'#fbbf24')+'</td><td>'+x.shards+'</td><td class=mut>'+fmt(x.units)+'</td><td class=mut>'+x.avgMs+'</td><td>'+pct(x.userUtil)+'</td><td>'+pct(x.poolDuty)+'</td><td class=mut>'+(x.busy?'working':'idle')+'</td></tr>').join(''):'<tr><td class=mut colspan=10>connect a worker…</td></tr>';
  const j=await (await fetch('/jobs',{headers:H()})).json();
  document.getElementById('jd').textContent=j.filter(x=>x.status==='done').length;
  document.getElementById('jf').textContent=j.filter(x=>x.status==='failed').length;
  const L=await (await fetch('/logs',{headers:H()})).json();
  document.getElementById('logs').innerHTML=L.slice(0,80).map(e=>'<span class=lvl-'+e.level+'>'+new Date(e.ts).toLocaleTimeString()+' ['+e.level+'] '+e.msg+(e.ctx?' · '+e.ctx:'')+'</span>').join('\\n');
 }catch(e){}
}
async function submit(){const m=document.getElementById('jobmsg');m.textContent='running…';
 const r=await fetch('/submit',{method:'POST',headers:H(),body:JSON.stringify({kernel:document.getElementById('kernel').value,size:+document.getElementById('size').value})});
 const j=await r.json();m.textContent=j.status==='done'?('done · '+(j.gflops?j.gflops.toFixed(1)+' GFLOP/s · ':'')+'verified='+j.verified):(j.note||j.error||j.status);refresh();}
refresh();setInterval(refresh,2500);
</script>`;

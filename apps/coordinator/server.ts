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
// Prefer the native TC39 Uint8Array<->base64 (Deno 2.x) — the manual String.fromCharCode loop is an order of
// magnitude slower and dominates a weight-push (GBs of base64). Feature-detected so older runtimes still work.
function b64e(u8: Uint8Array): string {
  const f = (u8 as unknown as { toBase64?: () => string }).toBase64;
  if (typeof f === 'function') return f.call(u8);
  let s = ''; const K = 0x8000; for (let i = 0; i < u8.length; i += K) s += String.fromCharCode(...u8.subarray(i, i + K)); return btoa(s);
}
function b64d(s: string): Uint8Array {
  const F = (Uint8Array as unknown as { fromBase64?: (s: string) => Uint8Array }).fromBase64;
  if (typeof F === 'function') return F(s);
  const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u;
}
// base64url, but never start with '-'/'_': a leading dash makes `worker_torch.py --token <tok>` unparseable
// (argparse reads it as a flag), and a leading dash is awkward in shells/URLs generally. Trim any from the front.
function tokenB64url(n = 24): string { return b64e(crypto.getRandomValues(new Uint8Array(n))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '').replace(/^[-_]+/, ''); }
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
  // ANSI Shadow "MOREGPU" with an indigo→pink→red 256-color gradient (matches the `moregpu` CLI).
  const rows = [
    '███╗   ███╗ ██████╗ ██████╗ ███████╗ ██████╗ ██████╗ ██╗   ██╗',
    '████╗ ████║██╔═══██╗██╔══██╗██╔════╝██╔════╝ ██╔══██╗██║   ██║',
    '██╔████╔██║██║   ██║██████╔╝█████╗  ██║  ███╗██████╔╝██║   ██║',
    '██║╚██╔╝██║██║   ██║██╔══██╗██╔══╝  ██║   ██║██╔═══╝ ██║   ██║',
    '██║ ╚═╝ ██║╚██████╔╝██║  ██║███████╗╚██████╔╝██║     ╚██████╔╝',
    '╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝      ╚═════╝ ',
  ];
  const cols = [63, 99, 135, 171, 205, 203];
  return '\n' + rows.map((r, i) => `  \x1b[38;5;${cols[i]}m${r}${C.reset}`).join('\n');
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
  console.log(`  ${C.dim}reboot-surviving service: add MOREGPU_SERVICE=1 · type /help in this console's HTTP: ${httpScheme}://${ADVERTISE_HOST}:${PORT}/help${C.reset}`);
  if (!(CERT_PATH && KEY_PATH) && BIND !== '127.0.0.1' && BIND !== 'localhost') {
    console.log(`\n  ${C.red}${C.b}⚠ no TLS and bound to ${BIND}${C.reset}${C.red} — the join handshake (including the tenant key) travels in plaintext.${C.reset}`);
    console.log(`  ${C.dim}For untrusted networks: set MOREGPU_TLS_CERT + MOREGPU_TLS_KEY (wss://), or MOREGPU_BIND=127.0.0.1 behind a VPN/tunnel.${C.reset}`);
  }
  console.log('');
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
    POST /workers/:id/control  pause · resume · set duty (ceil) · schedule · nick · remove (admin)

  ${C.b}Kernels${C.reset}  matmul · vector_add · vector_mul · saxpy · relu · scale · softmax · layernorm   ${C.dim}(extensible)${C.reset}
  ${C.b}Workers${C.reset}  a worker sets when it contributes with MOREGPU_SCHEDULE=always|idle-only|HH:MM-HH:MM
`);
}

// ---------- metrics ----------
const M = { jobsTotal: 0, jobsDone: 0, jobsFailed: 0, shardsDone: 0, shardsFailed: 0, gpuShards: 0, cpuShards: 0 };
const KM: Record<string, number> = {}; // jobs per kernel

// ---------- worker registry ----------
interface Worker {
  id: string; backend: string; label: string; os: string; ws: WebSocket;
  load1: number; util: number; duty: number; busy: boolean; joinedAt: number;
  // contribution accounting. `ops` is the CONSISTENT activity currency (one kernel shard OR one serving call
  // OR one training round = 1 op) — the trend + share are built from it so they never conflate LLM tokens
  // with kernel matrix-elements. `units` = kernel output-elements (detail); `tokens` = LLM tokens served (detail).
  shards: number; units: number; ops: number; tokens: number; errors: number; consecErrors: number; healthyBeats: number; busyCount: number; totalMs: number;
  lastOps: number; history: number[]; // per-sample ops completed → sparkline trend (consistent unit)
  pubkey?: CryptoKey; // Ed25519 public key for verifying this worker's result signatures
  pubkeyB64?: string; // raw public key (for the removed-worker denylist)
  paused: boolean; // scheduled-off or admin-paused → coordinator assigns it no new work
  pausedReason?: string | null; // 'admin' | 'schedule' | null — why it's paused
  ceil: number; // the worker's reported/administered duty CEILING (distinct from the live effective duty)
  schedule?: string; // the machine's own contribution schedule ("always" / "idle-only" / "HH:MM-HH:MM")
  nick?: string; // optional admin-set display label
}
const workers = new Map<string, Worker>();
const removedPubkeys = new Set<string>(); // workers an admin removed — refuse their re-registration (ban by key)
// Weight RESIDENCY: a named weight is cached on ONE worker; a resident matmul (bRef) runs there without
// re-sending the weight. Place different layers' weights on different workers to split a model (pipeline).
const weightHome = new Map<string, { worker: string; rows: number; cols: number; dtype: 'f32' | 'f16' }>(); // weightId → home worker + dims
const weightStore = new Map<string, Float32Array>(); // coordinator's own copy (DEQUANTIZED for f16), used to verify resident results
const REGISTER_TIMEOUT_MS = Number(Deno.env.get('MOREGPU_REGISTER_TIMEOUT_MS') ?? 8000);
const MAX_JOBS = Number(Deno.env.get('MOREGPU_MAX_JOBS') ?? 500); // cap retained job records (+ their output blobs)
const safeId = (s: string) => (String(s ?? '').replace(/[^A-Za-z0-9._:@-]/g, '').slice(0, 64) || 'worker');
/** Workers eligible for new work right now (not scheduled-off / admin-paused). */
function activeFleet(): Worker[] { return [...workers.values()].filter((w) => !w.paused); }
type ResultMsg = { ok: boolean; sealedOut?: { iv: string; ct: string }; error?: string; backend?: string; ms?: number; signed?: boolean };
interface Pending { resolve: (r: ResultMsg) => void; reject: (e: Error) => void; workerId: string; }
const pending = new Map<string, Pending>();
const SHARD_TIMEOUT_MS = Number(Deno.env.get('MOREGPU_SHARD_TIMEOUT_MS') ?? 60_000);
const MAX_SHARD_ATTEMPTS = Number(Deno.env.get('MOREGPU_MAX_SHARD_ATTEMPTS') ?? 3); // reassign a failed/timed-out shard to other workers
const AUTO_PAUSE_ERRORS = Number(Deno.env.get('MOREGPU_AUTO_PAUSE_ERRORS') ?? 4); // consecutive HARD failures before a worker is auto-paused
const AUTO_RESUME_BEATS = Number(Deno.env.get('MOREGPU_AUTO_RESUME_BEATS') ?? 3); // healthy heartbeats that auto-un-pause an errored worker
const MAX_CONCURRENT_JOBS = Number(Deno.env.get('MOREGPU_MAX_CONCURRENT_JOBS') ?? 4); // jobs the queue runs at once
const STALE_JOB_MS = Number(Deno.env.get('MOREGPU_STALE_JOB_MS') ?? 300_000); // fail a job that can't be scheduled within this

function wireWorker(ws: WebSocket) {
  let id = '';
  let registered = false;
  // Close a socket that connects but never authenticates, so an unauthenticated peer can't hold FDs/RAM open.
  const authTimer = setTimeout(() => { if (!registered) { try { ws.send(JSON.stringify({ t: 'denied', reason: 'no register within timeout' })); } catch { /* */ } try { ws.close(); } catch { /* */ } } }, REGISTER_TIMEOUT_MS);
  ws.onmessage = async (ev) => {
    let m: Record<string, unknown>;
    try { m = JSON.parse(ev.data as string); } catch { log('warn', 'bad frame from worker'); return; }
    if (m.t === 'register') {
      if (registered) return; // one register per socket
      if (!constEq(String(m.joinToken ?? ''), cfg.joinToken)) { ws.send(JSON.stringify({ t: 'denied', reason: 'bad join token' })); log('warn', 'worker rejected: bad join token'); ws.close(); return; }
      const node = m.node as { id: string; backend: string; label: string; os: string };
      const pubkeyB64 = typeof m.pubkey === 'string' ? m.pubkey : undefined;
      // Ban list: an admin-removed worker (by key) may not re-enroll.
      if (pubkeyB64 && removedPubkeys.has(pubkeyB64)) { ws.send(JSON.stringify({ t: 'denied', reason: 'removed by admin' })); ws.close(); return; }
      const wantId = safeId(node.id); // sanitize (also prevents /metrics label injection)
      // Reject an id that's already live so a token-holder can't hijack another worker's identity/shards.
      if (workers.has(wantId)) { ws.send(JSON.stringify({ t: 'denied', reason: 'worker id already registered' })); log('warn', `worker rejected: duplicate id ${wantId}`); ws.close(); return; }
      id = wantId;
      let pubkey: CryptoKey | undefined;
      try { if (pubkeyB64) pubkey = await crypto.subtle.importKey('raw', b64d(pubkeyB64) as BufferSource, { name: 'Ed25519' }, false, ['verify']); } catch { /* worker without a valid key runs unsigned */ }
      registered = true; clearTimeout(authTimer);
      workers.set(id, { id, backend: safeId(node.backend), label: safeId(node.label), os: safeId(node.os), ws, load1: 0, util: 0, duty: DUTY_HINT, ceil: DUTY_HINT, busy: false, joinedAt: Date.now(), shards: 0, units: 0, ops: 0, tokens: 0, errors: 0, consecErrors: 0, healthyBeats: 0, busyCount: 0, totalMs: 0, lastOps: 0, history: [], pubkey, pubkeyB64, paused: false, pausedReason: null });
      ws.send(JSON.stringify({ t: 'welcome', tenantKeyB64: b64e(TENANT_KEY), duty: DUTY_HINT }));
      log('info', `worker joined: ${id} (${safeId(node.label)}, ${safeId(node.os)})${pubkey ? ' · signed' : ''} · fleet=${workers.size}`);
      pumpQueue();
    } else if (m.t === 'heartbeat') {
      if (!registered || !id) return; // ignore heartbeats before auth; trust only the socket's own id, not m.id
      const w = workers.get(id);
      if (w) {
        w.load1 = Number(m.load1) || 0; w.util = Number(m.util) || 0; w.duty = Number(m.duty) || w.duty;
        if (typeof m.ceil === 'number') w.ceil = m.ceil;
        // A coordinator-owned time-window schedule must NOT be clobbered by the worker's self-report (the torch
        // worker always reports "always"); non-window schedules ("always"/"idle-only") stay worker-reported.
        if (typeof m.schedule === 'string' && !isWindow(w.schedule)) w.schedule = m.schedule;
        if (w.pausedReason === 'errors') {
          // auto-recover: a worker that heartbeats healthy for a few beats is un-paused automatically (no operator needed)
          if (m.paused === false) {
            w.healthyBeats++;
            if (w.healthyBeats >= AUTO_RESUME_BEATS) { w.paused = false; w.pausedReason = null; w.consecErrors = 0; w.healthyBeats = 0; log('info', `auto-resumed ${w.id} after recovery`); pumpQueue(); }
          } else w.healthyBeats = 0;
        } else if (isWindow(w.schedule) || w.pausedReason === 'schedule') {
          // the COORDINATOR owns the pause state for a time-window schedule → ignore the worker's paused report here
        } else { // the worker's own schedule/admin pause state (always / idle-only)
          if (typeof m.paused === 'boolean') w.paused = m.paused;
          w.pausedReason = (m.pausedReason as string | null) ?? null;
          if (m.paused === false) pumpQueue();
        }
      }
    } else if (m.t === 'cached') {
      if (!registered) return;
      const e = pendingCache.get(String(m.id)); if (e && e.workerId === id) { pendingCache.delete(String(m.id)); e.cb({ ok: !!m.ok, error: m.error as string | undefined }); } // ignore a cached ack from a worker other than the home

    } else if (m.t === 'train_reply' || m.t === 'model_reply') {
      if (!registered) return;
      const pm = m.t === 'train_reply' ? pendingTrain : pendingModel;
      const e = pm.get(String(m.reqId));
      if (e && e.workerId === id) { pm.delete(String(m.reqId)); e.cb({ ok: !!m.ok, sealed: m.sealed as { iv: string; ct: string } | undefined, error: m.error as string | undefined }); } // reject a reply for another worker's RPC
    } else if (m.t === 'result') {
      if (!registered) return;
      const p = pending.get(String(m.shardId));
      if (!p || p.workerId !== id) return; // forged/foreign result for another worker's shard → ignored
      const w = workers.get(id);
      let signed = false;
      if (m.ok && m.sealedOut && w?.pubkey && typeof m.sig === 'string') {
        const blob = m.sealedOut as { iv: string; ct: string };
        const okSig = await crypto.subtle.verify({ name: 'Ed25519' }, w.pubkey, b64d(m.sig) as BufferSource, new TextEncoder().encode(`${m.shardId}|${blob.iv}|${blob.ct}`));
        if (!okSig) { log('warn', `result signature INVALID from ${id} — rejecting shard`); p.reject(new Error('invalid result signature')); return; }
        signed = true;
      }
      p.resolve({ ok: m.ok as boolean, sealedOut: m.sealedOut as { iv: string; ct: string } | undefined, error: m.error as string | undefined, backend: m.backend as string | undefined, ms: m.ms as number | undefined, signed });
    }
  };
  ws.onclose = () => {
    clearTimeout(authTimer);
    if (!id) return;
    workers.delete(id);
    // reject any shard still in flight on this worker so the job fails fast instead of hanging the queue
    for (const [sid, p] of pending) if (p.workerId === id) { pending.delete(sid); p.reject(new Error(`worker ${id} disconnected mid-shard`)); }
    // same for in-flight train/model relay RPCs (else the awaiting HTTP handler hangs until the 120s timeout)
    for (const [rid, e] of pendingTrain) if (e.workerId === id) { pendingTrain.delete(rid); e.cb({ ok: false, error: `worker ${id} disconnected mid-rpc` }); }
    for (const [rid, e] of pendingModel) if (e.workerId === id) { pendingModel.delete(rid); e.cb({ ok: false, error: `worker ${id} disconnected mid-rpc` }); }
    // and any in-flight /weights cache ack destined for this worker (else that POST hangs for the full 30s)
    for (const [cid, e] of pendingCache) if (e.workerId === id) { pendingCache.delete(cid); e.cb({ ok: false, error: `worker ${id} disconnected before cache ack` }); }
    // drop weights that lived on this worker — they must be re-uploaded (the coordinator forgets its home)
    for (const [wid, home] of weightHome) if (home.worker === id) { weightHome.delete(wid); weightStore.delete(wid); }
    // forget any resident model / training session that lived here
    if (trainingHome === id) trainingHome = null;
    for (const [mid, wid] of modelHome) if (wid === id) modelHome.delete(mid);
    // a pipeline stage that lived here breaks the whole pipe → drop any shard plan that used this worker
    for (const [sid, plan] of shardPlans) if (plan.stages.some((s) => s.worker === id)) shardPlans.delete(sid);
    log('info', `worker left: ${id} · fleet=${workers.size}`);
  };
  ws.onerror = () => log('warn', `socket error${id ? ' from ' + id : ''}`);
}

// ---------- contribution trend sampler (per-worker + pool throughput sparklines) ----------
const poolHistory: number[] = [];
let lastPoolOps = 0;
setInterval(() => {
  let totalOps = 0;
  for (const w of workers.values()) {
    const delta = Math.max(0, w.ops - w.lastOps); w.lastOps = w.ops; // ops/interval — consistent across kernel + serving
    w.history.push(delta); if (w.history.length > 30) w.history.shift();
    totalOps += w.ops;
  }
  const pd = Math.max(0, totalOps - lastPoolOps); lastPoolOps = totalOps;
  poolHistory.push(pd); if (poolHistory.length > 30) poolHistory.shift();
}, 3000);

// ---------- coordinator-side schedule (time windows) ----------
// "always"/"idle-only" are honoured by the worker itself (self-reported via heartbeat). A time window
// "HH:MM-HH:MM" (the ON hours, may wrap past midnight, e.g. 22:00-07:00 = nights only) is evaluated HERE so it
// works for ANY worker type — the torch worker doesn't self-schedule — pausing/resuming the node as the wall
// clock enters/leaves the window. Set live from the admin panel; the coordinator owns the pause state for it.
const isWindow = (s?: string) => /^\d{1,2}:\d{2}-\d{1,2}:\d{2}$/.test((s ?? '').trim());
function scheduleOff(s: string): boolean {
  const [a, b] = s.split('-'); const [ah, am] = a.split(':').map(Number); const [bh, bm] = b.split(':').map(Number);
  if ([ah, am, bh, bm].some((n) => !Number.isFinite(n))) return false;
  const start = ah * 60 + am, end = bh * 60 + bm, now = new Date(), cur = now.getHours() * 60 + now.getMinutes();
  const active = start <= end ? (cur >= start && cur < end) : (cur >= start || cur < end); // wraps past midnight
  return !active;
}
function reconcileSchedules() {
  for (const w of workers.values()) {
    if (w.pausedReason === 'admin' || w.pausedReason === 'errors') continue; // admin/error pause outranks schedule
    if (isWindow(w.schedule)) {
      const off = scheduleOff(w.schedule!.trim());
      if (off && !(w.paused && w.pausedReason === 'schedule')) { w.paused = true; w.pausedReason = 'schedule'; log('info', `${w.id} scheduled-off (outside ${w.schedule})`); }
      else if (!off && w.pausedReason === 'schedule') { w.paused = false; w.pausedReason = null; log('info', `${w.id} scheduled-on (inside ${w.schedule})`); pumpQueue(); }
    } else if (w.pausedReason === 'schedule') { // schedule cleared back to always/idle-only → resume
      w.paused = false; w.pausedReason = null; pumpQueue();
    }
  }
}
setInterval(reconcileSchedules, 30_000);

// Backstop: fail any job that has sat un-schedulable in the queue too long, so a caller never hangs forever.
setInterval(() => {
  const now = Date.now();
  for (let i = queue.length - 1; i >= 0; i--) {
    const rec = queue[i];
    if (now - rec.submittedAt > STALE_JOB_MS) {
      queue.splice(i, 1); rec.status = 'failed'; rec.error = `stale: no worker available within ${Math.round(STALE_JOB_MS / 1000)}s`;
      jobInputs.delete(rec.id); jobSigned.delete(rec.id); // free the retained input buffers (runJob never ran for this job)
      M.jobsFailed++; log('warn', `${rec.id} failed: stale in queue (no active worker)`);
    }
  }
}, 15_000);

// ---------- kernels ----------
const ELEMENTWISE = new Set(['vector_add', 'vector_mul', 'saxpy', 'relu', 'scale', 'gelu']);
const geluF = (x: number) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
const ROWWISE = new Set(['softmax', 'layernorm']); // per-row reductions; sharded by whole rows
function cpuRowwise(kernel: string, a: Float32Array, cols: number): Float32Array {
  const rows = Math.floor(a.length / cols), o = new Float32Array(a.length);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    if (kernel === 'softmax') {
      let mx = -Infinity; for (let j = 0; j < cols; j++) mx = Math.max(mx, a[off + j]);
      let s = 0; for (let j = 0; j < cols; j++) { const e = Math.exp(a[off + j] - mx); o[off + j] = e; s += e; }
      for (let j = 0; j < cols; j++) o[off + j] /= s;
    } else {
      let m = 0; for (let j = 0; j < cols; j++) m += a[off + j]; m /= cols;
      let v = 0; for (let j = 0; j < cols; j++) { const d = a[off + j] - m; v += d * d; } v /= cols;
      const inv = 1 / Math.sqrt(v + 1e-5);
      for (let j = 0; j < cols; j++) o[off + j] = (a[off + j] - m) * inv;
    }
  }
  return o;
}
function cpuKernel(kernel: string, a: Float32Array, b: Float32Array | null, scalar: number): Float32Array {
  const n = a.length, o = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    switch (kernel) {
      case 'vector_add': o[i] = a[i] + (b as Float32Array)[i]; break;
      case 'vector_mul': o[i] = a[i] * (b as Float32Array)[i]; break;
      case 'saxpy': o[i] = scalar * a[i] + (b as Float32Array)[i]; break;
      case 'relu': o[i] = a[i] > 0 ? a[i] : 0; break;
      case 'scale': o[i] = a[i] * scalar; break;
      case 'gelu': o[i] = geluF(a[i]); break;
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
interface JobRec { id: string; status: JobStatus; kernel: string; size: number; sealed: boolean; submittedAt: number; ms?: number; gflops?: number; verified?: boolean; signed?: boolean; shards?: { worker: string; backend: string; work: number; ms: number }[]; error?: string; dataMode?: boolean; output?: string; outLen?: number; }
/** Real input a caller submitted (data mode); the pool computes on THIS data and returns the output. */
interface JobInput { a?: Float32Array; b?: Float32Array; scalar?: number; M?: number; N?: number; K?: number; bRef?: string; }
const residentCount = (wid: string) => [...weightHome.values()].filter((h) => h.worker === wid).length;
const pendingCache = new Map<string, { workerId: string; cb: (r: { ok: boolean; error?: string }) => void }>(); // /weights waits for the worker's ack (tagged with the home worker so a disconnect can reject it)
// On-pool fine-tuning is relayed to ONE native (torch) worker that runs the whole train step locally.
// The coordinator seals/accounts the RPC like a shard but can't verify a stochastic loss (it has no
// torch) — correctness is checked out-of-band against a seeded reference (see examples/lora_finetune.py).
// Relayed RPCs to a native (torch) worker: on-pool fine-tuning ('train') and resident-model serving
// ('model'). The coordinator seals/accounts them like a shard but can't verify the result (it has no
// torch) — training is checked out-of-band vs a seeded reference; serving is checked vs transformers
// exact-match by the client (examples/llm_serve.py).
type RelayReply = { ok: boolean; sealed?: { iv: string; ct: string }; error?: string };
type RelayPending = { workerId: string; cb: (r: RelayReply) => void };
const pendingTrain = new Map<string, RelayPending>();
const pendingModel = new Map<string, RelayPending>();
let relaySeq = 0;
let trainingHome: string | null = null; // worker id hosting the single (non-DiLoCo) training session
// TODO(review): a worker LRU-evicts its own resident models/shards past MAX_RESIDENT_MODELS, but the
// coordinator is never told, so modelHome/shardPlans keep pointing at an id the worker no longer holds
// (a later /model/forward then errors "not loaded"). Fix: have the worker report evicted ids on load
// replies so the coordinator can drop the stale bookkeeping, or mirror the worker's cap in lockstep.
const modelHome = new Map<string, string>(); // resident model id → worker id
// Async model loads: a download-free push can take longer than a public tunnel will hold one HTTP request
// open (trycloudflare 502s a long request), so /model/load?async streams in the BACKGROUND and the caller
// polls GET /model/status. This is also the answer to "heavy models over a slow link" — the request is short.
type LoadState = { status: 'loading' | 'ready' | 'error'; worker: string; model: string; started: number; info?: Record<string, unknown>; error?: string };
const modelLoads = new Map<string, LoadState>(); // model id → last load's progress/outcome
// Same idea for a download-free SHARD load: streaming each stage's slice across the fleet can outlast a tunnel's
// request timeout, so /model/shard?async loads the stages in the background and the caller polls /model/shard_status.
type ShardLoadState = { status: 'loading' | 'ready' | 'error'; model: string; started: number; stagesDone: number; stagesTotal: number; info?: unknown; error?: string; aborted?: boolean };
const shardLoads = new Map<string, ShardLoadState>();
// Pipeline-parallel sharding (GPT-2 family): a model's transformer layers are split into contiguous
// STAGES, one per worker — each worker holds ONLY its stage resident. The plan records, per shard id,
// the ordered stages (which worker owns which layer range + which is first/last); /model/shard_forward
// pipes the hidden state stage→stage (only [seq×hidden] activations cross the wire, never the weights).
interface ShardStage { worker: string; start: number; end: number; first: boolean; last: boolean }
const shardPlans = new Map<string, { model: string; stages: ShardStage[] }>();
// Generous by default: model_load/train_load download & instantiate a model on the worker, which for a
// heavy model or a slow link can far exceed 2min. Fine-tuning + big-model inference need this headroom.
const RELAY_TIMEOUT_MS = Number(Deno.env.get('MOREGPU_TRAIN_TIMEOUT_MS') ?? 600_000);
/** Native torch workers can fine-tune (autograd) and hold a whole model resident; WebGPU workers cannot. */
const torchWorkers = () => activeFleet().filter((w) => w.label.includes('torch'));
async function relayRPC(w: Worker, kind: 'train' | 'model', pend: Map<string, RelayPending>, op: string, payload: unknown): Promise<{ ok: boolean; data?: Record<string, unknown>; error?: string }> {
  const reqId = `${kind}-${++relaySeq}`;
  const sealed = await seal(TENANT_KEY, new TextEncoder().encode(JSON.stringify(payload)));
  // Bind the reply to THIS worker's id, so another worker can't resolve someone else's RPC (see train_reply handler).
  // TODO(review): a timeout here only abandons the coordinator's wait — the worker keeps running the compute
  // (and, for a train op, may still commit optimizer/step state). Needs a cancel/epoch protocol (send an
  // abort frame, or a generation counter the worker checks between steps) so a timed-out round is truly voided.
  const ack = new Promise<RelayReply>((res) => { const t = setTimeout(() => { pend.delete(reqId); res({ ok: false, error: `${kind} rpc timeout` }); }, RELAY_TIMEOUT_MS); pend.set(reqId, { workerId: w.id, cb: (r) => { clearTimeout(t); res(r); } }); });
  // Resident serving + training run through this relay, NOT the kernel-shard path — so without accounting here
  // they'd never show in the per-worker contribution / trend (flat while you chat). Mark the node busy for the
  // op's duration; on success credit ONE op (the consistent activity currency) for actual COMPUTE ops only —
  // never for weight-transfer/control ops (push_*/load/unload/arch/shard_unload), which the worker only
  // receives, not computes (crediting those would make a downloading node look like a top contributor).
  w.busyCount++; w.busy = true;
  try {
    w.ws.send(JSON.stringify({ t: kind, reqId, op, sealed }));
  } catch (e) { w.busyCount = Math.max(0, w.busyCount - 1); w.busy = w.busyCount > 0; return { ok: false, error: `send failed: ${e}` }; }
  try {
    const r = await ack;
    if (!r.ok) return { ok: false, error: r.error };
    if (!r.sealed) return { ok: true, data: {} };
    let data: Record<string, unknown>;
    try { data = JSON.parse(new TextDecoder().decode(await unseal(TENANT_KEY, r.sealed))) as Record<string, unknown>; }
    catch (e) { return { ok: false, error: `bad ${kind} reply: ${e}` }; }
    // NB: keep w.totalMs (→ avgMs) shard-only — mixing serve/train ms into a kernel-shard average corrupts it.
    if (SERVE_OPS.has(op)) { w.ops++; w.tokens += Math.max(0, Math.round(Number(data.n) || 0)); }
    else if (TRAIN_OPS.has(op)) { w.ops++; }
    return { ok: true, data };
  } finally { w.busyCount = Math.max(0, w.busyCount - 1); w.busy = w.busyCount > 0; }
}
// COMPUTE relay ops that count toward a node's contribution (one call = one op). Serving forwards/decodes and
// pipeline-shard stages, plus training inner-loops/steps. Everything else on the relay is transfer/setup.
const SERVE_OPS = new Set(['forward', 'generate', 'chat', 'shard_forward']);
const TRAIN_OPS = new Set(['inner', 'step']);
const trainRPC = (w: Worker, op: string, payload: unknown) => relayRPC(w, 'train', pendingTrain, op, payload);
const modelRPC = (w: Worker, op: string, payload: unknown) => relayRPC(w, 'model', pendingModel, op, payload);

// ── DOWNLOAD-FREE resident load: the COORDINATOR (admin box) fetches the weights from HF over HTTPS and
// streams the raw file bytes to the worker; the worker stages them (in RAM when it has /dev/shm — Linux/Colab
// — else a transient temp dir) and loads with the hub disabled, so it never downloads anything. This keeps
// "all weights on the admin, fleet = pure GPU compute". It's a ONE-TIME transfer (unlike the per-token
// pipeline pipe), so it tolerates a slow/flaky tunnel far better.
const HF_BASE = Deno.env.get('MOREGPU_HF_BASE') ?? 'https://huggingface.co';
const PUSH_CHUNK = Math.max(1 << 20, Number(Deno.env.get('MOREGPU_PUSH_CHUNK') ?? (32 << 20))); // 32 MiB raw / chunk — fewer sealed round-trips
// Safety rail against a runaway/oversized repo OOM-ing the coordinator or a donated worker (default 20 GiB,
// env-tunable). The index.json is metadata (KB) so it gets a much tighter cap of its own.
const PUSH_MAX_BYTES = Math.max(1 << 20, Number(Deno.env.get('MOREGPU_PUSH_MAX_BYTES') ?? (20 * (1 << 30))));
const PUSH_INDEX_MAX = 64 << 20; // a safetensors index is normally KB; refuse a multi-GB one before .text()
// model id must be a plain HF repo ref (owner/name or a canonical alias) — it goes straight into a URL.
const HF_REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*(\/[A-Za-z0-9][A-Za-z0-9._-]*)?$/;
// the finite set of hub files a causal-LM load may need; each is fetched, and a 404 is simply skipped.
const HF_AUX_FILES = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
  'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json', 'tokenizer.model', 'chat_template.jinja'];
const pushInFlight = new Set<string>(); // `${workerId}:${id}` currently being pushed — blocks a concurrent same-id race

async function hfFetch(model: string, file: string): Promise<Response | null> {
  const headers: Record<string, string> = {};
  const tok = Deno.env.get('HF_TOKEN'); if (tok) headers['authorization'] = `Bearer ${tok}`;
  const r = await fetch(`${HF_BASE}/${model}/resolve/main/${encodeURIComponent(file)}?download=true`, { headers, redirect: 'follow' });
  if (r.status === 404 || r.status === 403) { await r.body?.cancel(); return null; }
  if (!r.ok) { await r.body?.cancel(); throw new Error(`HF ${file} → HTTP ${r.status}`); }
  return r;
}

// Read a small metadata file into a string, but refuse (before buffering) if it declares more than `max` bytes.
async function hfFetchText(model: string, file: string, max: number): Promise<string | null> {
  const r = await hfFetch(model, file);
  if (!r) return null;
  const len = Number(r.headers.get('content-length') ?? 0);
  if (len > max) { await r.body?.cancel(); throw new Error(`${file} is ${len} bytes (> ${max} cap) — refusing to buffer`); }
  const buf = new Uint8Array(await r.arrayBuffer());
  if (buf.length > max) throw new Error(`${file} exceeded ${max} bytes — refusing`);
  return new TextDecoder().decode(buf);
}

async function streamFileToWorker(w: Worker, id: string, name: string, resp: Response, budget: { left: number }): Promise<number> {
  const reader = resp.body!.getReader();
  let pending = new Uint8Array(0), seq = 0, total = 0;
  const send = async (data: Uint8Array, last: boolean) => {
    const r = await modelRPC(w, 'push_chunk', { id, name, seq: seq++, data: b64e(data), last });
    if (!r.ok) throw new Error(`push_chunk ${name}#${seq}: ${r.error}`);
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length; budget.left -= value.length;
    if (budget.left < 0) { await reader.cancel(); throw new Error(`weight push exceeded ${PUSH_MAX_BYTES}-byte cap (raise MOREGPU_PUSH_MAX_BYTES) — aborted`); }
    const merged = new Uint8Array(pending.length + value.length); merged.set(pending); merged.set(value, pending.length);
    pending = merged;
    while (pending.length >= PUSH_CHUNK) { await send(pending.slice(0, PUSH_CHUNK), false); pending = pending.slice(PUSH_CHUNK); }
  }
  await send(pending, true); // final chunk (may be empty) → guarantees the file is created even if 0 bytes
  return total;
}

async function pushModelToWorker(w: Worker, model: string, id: string, fp16: boolean): Promise<Record<string, unknown>> {
  if (!HF_REPO_RE.test(model)) throw new Error(`bad model ref "${model}" — expected an HF repo id like "gpt2" or "Qwen/Qwen2.5-0.5B"`);
  const guard = `${w.id}:${id}`;
  if (pushInFlight.has(guard)) throw new Error(`a download-free push for "${id}" is already in progress on ${w.id} — wait for it to finish`);
  pushInFlight.add(guard);
  try {
    // resolve the weight files: a sharded checkpoint lists them in an index; otherwise it's a single safetensors.
    // The index itself MUST also be staged — transformers discovers shards only through model.safetensors.index.json.
    let weightFiles: string[];
    const indexText = await hfFetchText(model, 'model.safetensors.index.json', PUSH_INDEX_MAX);
    if (indexText !== null) {
      const idx = JSON.parse(indexText) as { weight_map?: Record<string, string> };
      weightFiles = [...new Set(Object.values(idx.weight_map ?? {}))];
      if (weightFiles.length === 0) throw new Error(`${model}: empty safetensors index`);
    } else {
      weightFiles = ['model.safetensors'];
    }
    const begin = await modelRPC(w, 'push_begin', { id, model });
    if (!begin.ok) throw new Error(`push_begin failed: ${begin.error}`);
    const budget = { left: PUSH_MAX_BYTES };
    try {
      let bytes = 0, sentWeights = 0;
      for (const f of HF_AUX_FILES) {            // config + tokenizer/generation files (skip any that 404)
        const resp = await hfFetch(model, f);
        if (resp) bytes += await streamFileToWorker(w, id, f, resp, budget);
      }
      if (indexText !== null) {                  // stage the shard index too, so from_pretrained can find the shards
        const b = new TextEncoder().encode(indexText);
        const r = await modelRPC(w, 'push_chunk', { id, name: 'model.safetensors.index.json', seq: 0, data: b64e(b), last: true });
        if (!r.ok) throw new Error(`push_chunk index: ${r.error}`); bytes += b.length;
      }
      for (const f of weightFiles) {             // the actual weights (single or sharded safetensors)
        const resp = await hfFetch(model, f);
        if (!resp) throw new Error(`weight file ${f} missing on HF (no safetensors?) — this model can't be pushed download-free`);
        bytes += await streamFileToWorker(w, id, f, resp, budget); sentWeights++;
      }
      if (sentWeights === 0) throw new Error(`${model}: no weight files found`);
      log('info', `weight-push ${model} → ${w.id}: streamed ${(bytes / 1e6).toFixed(1)} MB (${weightFiles.length} shard(s)), assembling…`);
      const end = await modelRPC(w, 'push_end', { id, model, fp16 });
      if (!end.ok) throw new Error(`push_end failed: ${end.error}`);
      return end.data ?? {};
    } catch (e) {
      await modelRPC(w, 'unload', { id }).catch(() => {}); // best-effort: drop any partial staging on the worker
      throw e;
    }
  } finally {
    pushInFlight.delete(guard);
  }
}

// ── DOWNLOAD-FREE PIPELINE SHARDING: the coordinator fetches the model ONCE and streams each worker only the
// tensors for ITS stage (its contiguous layer slice + embeddings on the first stage + final norm/head on the
// last), rebuilt as a small per-stage safetensors. The fleet never touches the HF hub. A layer tensor is any
// weight whose name contains ".h.<i>." (GPT-2) or ".layers.<i>." (Llama-family); everything else is non-layer
// (embeddings / final norm / lm_head) and goes to the end stages only.
type STHeader = Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
const LAYER_RE = /(?:^|\.)(?:h|layers)\.(\d+)\./;

async function hfFetchRange(model: string, file: string, start: number, end: number): Promise<Uint8Array> {
  const headers: Record<string, string> = { range: `bytes=${start}-${end}` };
  const tok = Deno.env.get('HF_TOKEN'); if (tok) headers['authorization'] = `Bearer ${tok}`;
  const r = await fetch(`${HF_BASE}/${model}/resolve/main/${encodeURIComponent(file)}?download=true`, { headers, redirect: 'follow' });
  if (!r.ok && r.status !== 206) { await r.body?.cancel(); throw new Error(`HF range ${file} [${start}-${end}] → HTTP ${r.status}`); }
  let buf = new Uint8Array(await r.arrayBuffer());
  if (r.status === 200 && buf.length > end - start + 1) buf = buf.slice(start, end + 1); // server ignored Range → slice locally
  return buf;
}

async function hfSafetensorsHeader(model: string, file: string): Promise<{ header: STHeader; headerLen: number }> {
  const lenBuf = await hfFetchRange(model, file, 0, 7);
  if (lenBuf.length < 8) throw new Error(`${file}: could not read safetensors header length`);
  const headerLen = Number(new DataView(lenBuf.buffer, lenBuf.byteOffset, 8).getBigUint64(0, true));
  if (!(headerLen > 0 && headerLen < (256 << 20))) throw new Error(`${file}: implausible safetensors header length ${headerLen}`);
  const hb = await hfFetchRange(model, file, 8, 8 + headerLen - 1);
  const parsed = JSON.parse(new TextDecoder().decode(hb)) as Record<string, unknown>;
  delete parsed['__metadata__'];
  return { header: parsed as STHeader, headerLen };
}

function stageTensors(header: STHeader, start: number, end: number, first: boolean, last: boolean): string[] {
  const names: string[] = [];
  for (const name of Object.keys(header)) {
    const m = LAYER_RE.exec(name);
    if (m) { const i = Number(m[1]); if (i >= start && i < end) names.push(name); }   // this stage's decoder blocks
    else if (first || last) names.push(name);                                          // embeddings / final norm / head
  }
  return names;
}

// Build a valid per-stage safetensors (recomputed contiguous offsets) and stream it as "model.safetensors".
async function streamStageSafetensors(w: Worker, id: string, model: string, file: string, header: STHeader, srcHeaderLen: number, names: string[], budget: { left: number }): Promise<number> {
  const newHeader: STHeader = {}; const parts: { srcStart: number; len: number }[] = []; let off = 0;
  const dataStart = 8 + srcHeaderLen;
  for (const name of names) {
    const t = header[name]; const len = t.data_offsets[1] - t.data_offsets[0];
    newHeader[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + len] };
    parts.push({ srcStart: dataStart + t.data_offsets[0], len }); off += len;
  }
  let hstr = JSON.stringify(newHeader);
  while ((8 + new TextEncoder().encode(hstr).length) % 8 !== 0) hstr += ' '; // safetensors: header padded so data is 8-byte aligned
  const hbytes = new TextEncoder().encode(hstr);
  const prefix = new Uint8Array(8); new DataView(prefix.buffer).setBigUint64(0, BigInt(hbytes.length), true);
  // chunked push of: [8-byte len][header][each tensor's bytes, in order] → appended into the worker's model.safetensors
  let pending = new Uint8Array(0), seq = 0, total = 0;
  const push = async (data: Uint8Array, last: boolean) => { const r = await modelRPC(w, 'push_chunk', { id, name: 'model.safetensors', seq: seq++, data: b64e(data), last }); if (!r.ok) throw new Error(`push_chunk safetensors: ${r.error}`); };
  const feed = async (b: Uint8Array) => {
    total += b.length; budget.left -= b.length;
    if (budget.left < 0) throw new Error(`shard push exceeded ${PUSH_MAX_BYTES}-byte cap (raise MOREGPU_PUSH_MAX_BYTES)`);
    const merged = new Uint8Array(pending.length + b.length); merged.set(pending); merged.set(b, pending.length); pending = merged;
    while (pending.length >= PUSH_CHUNK) { await push(pending.slice(0, PUSH_CHUNK), false); pending = pending.slice(PUSH_CHUNK); }
  };
  await feed(prefix); await feed(hbytes);
  for (const p of parts) await feed(await hfFetchRange(model, file, p.srcStart, p.srcStart + p.len - 1)); // Range-fetch just this tensor
  await push(pending, true);
  return total;
}

// Run ONE forward across a shard plan: stage 0 embeds input_ids → hidden; each next stage runs its blocks on
// the piped hidden; the last returns {argmax, logits?}. Only activations cross the wire. Re-checks each stage's
// worker is still connected (so a node churning mid-generation surfaces cleanly instead of hanging).
async function shardPipe(sid: string, input_ids: number[], returnLogits: boolean): Promise<{ ok: true; data: Record<string, unknown> } | { ok: false; error: string; disconnected?: boolean }> {
  const plan = shardPlans.get(sid);
  if (!plan || plan.stages.length === 0) return { ok: false, error: `shard ${sid} not loaded/ready` };
  for (const st of plan.stages) if (!workers.has(st.worker)) return { ok: false, error: `stage worker ${st.worker} disconnected — re-shard with POST /model/shard`, disconnected: true };
  let carry: Record<string, unknown> = {};
  for (let i = 0; i < plan.stages.length; i++) {
    const st = plan.stages[i]; const w = workers.get(st.worker)!;
    const payload: Record<string, unknown> = { id: sid, first: st.first, last: st.last };
    if (st.first) payload.input_ids = input_ids; else { payload.hidden = carry.hidden; payload.seq = carry.seq; }
    if (st.last && returnLogits) payload.return_logits = true;
    const r = await modelRPC(w, 'shard_forward', payload);
    if (!r.ok) return { ok: false, error: `shard_forward failed on ${st.worker} (stage ${i}, layers ${st.start}-${st.end}): ${r.error}` };
    carry = r.data!;
  }
  return { ok: true, data: carry };
}

// --- DiLoCo: the coordinator is the parameter server. It holds the GLOBAL adapter + an outer Nesterov
// momentum buffer; each round it averages the workers' post-inner-step adapters, applies the outer step,
// and broadcasts the new global. This is the genuine reduce path (matmul pooling only concatenates). ---
interface Diloco { workers: string[]; model: string; global: Map<string, Float32Array>; momentum: Map<string, Float32Array>; shape: Map<string, number[]>; round: number; busy: boolean; }
let diloco: Diloco | null = null;
function tensorsFromReply(data: Record<string, unknown>): Map<string, Float32Array> {
  const m = new Map<string, Float32Array>();
  const t = (data.tensors ?? {}) as Record<string, { data: string }>;
  for (const [n, v] of Object.entries(t)) m.set(n, b64ToF32(v.data));
  return m;
}
const jobs = new Map<string, JobRec>();
const jobInputs = new Map<string, JobInput>();
const jobSigned = new Map<string, { signed: number; total: number }>(); // Ed25519 verification tally per job
const queue: JobRec[] = [];
let jobSeq = 0, shardSeq = 0, inflight = 0;

function submit(kernel: string, size: number, input?: JobInput): JobRec {
  const id = `job-${++jobSeq}`;
  const rec: JobRec = { id, status: 'queued', kernel, size, sealed: true, submittedAt: Date.now(), dataMode: !!input?.a };
  jobs.set(id, rec); queue.push(rec); M.jobsTotal++; KM[kernel] = (KM[kernel] ?? 0) + 1;
  // cap retained job records (and their data-mode output blobs) — drop the oldest terminal jobs first
  if (jobs.size > MAX_JOBS) for (const [jid, j] of jobs) { if (jobs.size <= MAX_JOBS) break; if (j.status === 'done' || j.status === 'failed') jobs.delete(jid); }
  if (input?.a) jobInputs.set(id, input);
  log('info', `queued ${id}: ${kernel} ${input?.a ? 'data' : 'size=' + size}`, `queue depth ${queue.length}`);
  pumpQueue();
  return rec;
}

// Run up to MAX_CONCURRENT_JOBS jobs at once; each finished job re-pumps so the queue keeps flowing and
// one slow/large job never blocks the rest. Fire-and-forget: callers poll rec.status (never await this).
function pumpQueue() {
  while (inflight < MAX_CONCURRENT_JOBS && queue.length && activeFleet().length > 0) {
    const rec = queue.shift()!;
    rec.status = 'running';
    inflight++;
    runJob(rec)
      .then(() => { rec.status = 'done'; M.jobsDone++; })
      .catch((e) => { rec.status = 'failed'; rec.error = String(e); M.jobsFailed++; jobInputs.delete(rec.id); jobSigned.delete(rec.id); log('error', `${rec.id} failed: ${rec.error}`); })
      .finally(() => { inflight--; pumpQueue(); });
  }
}

/** Auto-pause a worker that keeps HARD-failing so the fleet stops handing it work (admin/auto resume).
 *  Never pauses the last active worker — that would strand the queue with nothing to run. */
function maybeAutoPause(w: Worker) {
  if (w.consecErrors >= AUTO_PAUSE_ERRORS && !w.paused && activeFleet().length > 1) {
    w.paused = true; w.pausedReason = 'errors'; w.healthyBeats = 0;
    log('warn', `auto-paused ${w.id} after ${w.consecErrors} consecutive hard failures`);
  }
}

/** Dispatch one shard, retrying on OTHER active workers if it fails/times out (so a dead worker doesn't
 *  fail the whole job). Prefers `preferred`, then the HEALTHIEST untried active worker (fewest recent errors). */
async function dispatchResilient(jobId: string, payload: Record<string, unknown>, preferred: Worker): Promise<{ out: Float32Array; worker: Worker }> {
  const tried = new Set<string>();
  let lastErr: unknown;
  for (let attempt = 0; attempt < MAX_SHARD_ATTEMPTS; attempt++) {
    const pool = activeFleet().filter((w) => !tried.has(w.id)).sort((a, b) => a.consecErrors - b.consecErrors);
    const candidate = (attempt === 0 && !preferred.paused && !tried.has(preferred.id)) ? preferred : pool[0];
    if (!candidate) break;
    tried.add(candidate.id);
    try {
      const out = await dispatchShard(candidate, jobId, payload);
      candidate.consecErrors = 0;
      return { out, worker: candidate };
    } catch (e) {
      lastErr = e; candidate.errors++;
      // Only a HARD compute failure counts toward auto-pause; a timeout/disconnect is transient backpressure,
      // not a reason to attrit a healthy-but-backlogged worker (a burst of concurrent shards must not pause it).
      if (String(e).includes('failed:')) { candidate.consecErrors++; maybeAutoPause(candidate); }
      log('warn', `shard reassign: ${candidate.id} failed (${String(e)}); attempt ${attempt + 1}/${MAX_SHARD_ATTEMPTS}`);
    }
  }
  throw lastErr ?? new Error('no active worker could complete the shard');
}

async function dispatchShard(w: Worker, jobId: string, payload: Record<string, unknown>): Promise<Float32Array> {
  const shardId = `s-${++shardSeq}-${tokenB64url(6)}`; // unguessable so results can't be forged by id
  const sealedIn = await seal(TENANT_KEY, new TextEncoder().encode(JSON.stringify(payload)));
  const done = new Promise<ResultMsg>((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(shardId); reject(new Error(`shard timeout (${SHARD_TIMEOUT_MS}ms) on ${w.id}`)); }, SHARD_TIMEOUT_MS);
    pending.set(shardId, { resolve: (r) => { clearTimeout(timer); resolve(r); }, reject: (e) => { clearTimeout(timer); reject(e); }, workerId: w.id });
  });
  w.busyCount++; w.busy = true; // a worker may run several concurrent shards under MAX_CONCURRENT_JOBS
  w.ws.send(JSON.stringify({ t: 'assign', shardId, jobId, sealedIn }));
  let r: ResultMsg;
  try { r = await done; } finally { pending.delete(shardId); w.busyCount = Math.max(0, w.busyCount - 1); w.busy = w.busyCount > 0; }
  if (!r.ok || !r.sealedOut) { M.shardsFailed++; throw new Error(`shard on ${w.id} failed: ${r.error}`); }
  M.shardsDone++; (r.backend?.startsWith('gpu') ? M.gpuShards++ : M.cpuShards++);
  const outObj = JSON.parse(new TextDecoder().decode(await unseal(TENANT_KEY, r.sealedOut)));
  const out = b64ToF32(outObj.out);
  w.shards++; w.ops++; w.units += out.length; w.totalMs += r.ms ?? 0; // contribution accounting (one shard = one op)
  const js = jobSigned.get(jobId) ?? { signed: 0, total: 0 }; js.total++; if (r.signed) js.signed++; jobSigned.set(jobId, js);
  return out;
}

async function runJob(rec: JobRec) {
  const fleet = activeFleet();
  if (fleet.length === 0) throw new Error('no active workers (all paused/scheduled-off)');
  const input = jobInputs.get(rec.id); jobInputs.delete(rec.id);
  const data = !!input?.a; // data mode: compute on the caller's real tensors and return the output
  const t0 = performance.now();
  // Resident matmul: A · (a weight that lives on its home worker). Runs whole on that worker — the weight
  // is never re-sent. This is the building block for splitting a model across workers (pipeline parallel).
  if (rec.kernel === 'matmul' && input?.bRef) {
    const home = weightHome.get(input.bRef);
    if (!home) throw new Error(`weight ${input.bRef} is not resident — upload it via POST /weights`);
    const w = workers.get(home.worker);
    if (!w || w.paused) throw new Error(`home worker for ${input.bRef} unavailable`);
    const A = input.a!, K = home.rows, N = home.cols;
    const M = input.M ?? Math.round(A.length / K);
    if (A.length !== M * K) throw new Error(`resident matmul shape mismatch (A=${A.length}, expected ${M}x${K})`);
    const out = await dispatchShard(w, rec.id, { kernel: 'matmul', a: f32ToB64(A), bRef: input.bRef, rows: M, N, K });
    const wall = performance.now() - t0;
    const B = weightStore.get(input.bRef);
    const tol = home.dtype === 'f16' ? Math.max(3e-2, K * 3e-3) : Math.max(1e-2, K * 1e-4); // f16 keeps ~3 sig digits
    if (B && M * N <= 640 * 640) { const ref = cpuMatmul(A, B, M, N, K); let md = 0; for (let i = 0; i < out.length; i++) md = Math.max(md, Math.abs(out[i] - ref[i])); rec.verified = md < tol; }
    rec.gflops = (2 * M * N * K) / (wall / 1000) / 1e9; rec.ms = wall;
    rec.shards = [{ worker: w.id, backend: w.label, work: out.length, ms: 0 }];
    rec.output = f32ToB64(out); rec.outLen = out.length;
    const js = jobSigned.get(rec.id); rec.signed = !!js && js.total > 0 && js.signed === js.total; jobSigned.delete(rec.id);
    log('info', `${rec.id} done: resident matmul ${input.bRef}@${w.id} · ${wall.toFixed(0)}ms · verified=${rec.verified}`);
    return;
  }
  if (rec.kernel === 'matmul') {
    // dims: data mode uses input.M/N/K (default square from array); else a random square of rec.size
    const M = data ? (input!.M ?? Math.round(Math.sqrt(input!.a!.length))) : rec.size;
    const K = data ? (input!.K ?? M) : rec.size;
    const N = data ? (input!.N ?? K) : rec.size;
    const A = data ? input!.a! : new Float32Array(M * K).map(() => Math.random());
    const B = data ? input!.b! : new Float32Array(K * N).map(() => Math.random());
    if (A.length !== M * K || B.length !== K * N) throw new Error(`matmul input shape mismatch (A=${A.length} expected ${M * K}, B=${B.length} expected ${K * N})`);
    const rowsPer = Math.ceil(M / fleet.length);
    const parts = await Promise.all(fleet.map(async (w, i) => {
      const r0 = i * rowsPer, rows = Math.min(M, r0 + rowsPer) - r0; if (rows <= 0) return null;
      const { out, worker } = await dispatchResilient(rec.id, { kernel: 'matmul', a: f32ToB64(A.slice(r0 * K, (r0 + rows) * K)), b: f32ToB64(B), rows, N, K }, w);
      return { r0, out, w: worker };
    }));
    const Cm = new Float32Array(M * N); const sh: JobRec['shards'] = [];
    for (const p of parts) if (p) { Cm.set(p.out, p.r0 * N); sh.push({ worker: p.w.id, backend: p.w.label, work: p.out.length, ms: 0 }); }
    const wall = performance.now() - t0;
    if (M * N <= 640 * 640) { let md = 0; const ref = cpuMatmul(A, B, M, N, K); for (let i = 0; i < Cm.length; i++) md = Math.max(md, Math.abs(Cm[i] - ref[i])); rec.verified = md < Math.max(1e-2, K * 1e-4); }
    rec.gflops = (2 * M * N * K) / (wall / 1000) / 1e9; rec.ms = wall; rec.shards = sh;
    if (data) { rec.output = f32ToB64(Cm); rec.outLen = Cm.length; }
  } else if (ELEMENTWISE.has(rec.kernel)) {
    const binary = rec.kernel !== 'relu' && rec.kernel !== 'scale' && rec.kernel !== 'gelu';
    const a = data ? input!.a! : new Float32Array(rec.size).map(() => Math.random() * 2 - 1);
    const b = data ? (input!.b ?? new Float32Array(a.length)) : new Float32Array(a.length).map(() => Math.random());
    const scalar = data ? (input!.scalar ?? 1) : 2;
    const n = a.length;
    if (binary && b.length !== n) throw new Error(`elementwise input length mismatch (a=${n}, b=${b.length})`);
    const per = Math.ceil(n / fleet.length);
    const parts = await Promise.all(fleet.map(async (w, i) => {
      const s0 = i * per, len = Math.min(n, s0 + per) - s0; if (len <= 0) return null;
      const { out, worker } = await dispatchResilient(rec.id, { kernel: rec.kernel, a: f32ToB64(a.subarray(s0, s0 + len)), b: binary ? f32ToB64(b.subarray(s0, s0 + len)) : undefined, scalar, len }, w);
      return { s0, out, w: worker };
    }));
    const O = new Float32Array(n); const sh: JobRec['shards'] = [];
    for (const p of parts) if (p) { O.set(p.out, p.s0); sh.push({ worker: p.w.id, backend: p.w.label, work: p.out.length, ms: 0 }); }
    const ref = cpuKernel(rec.kernel, a, b, scalar); let md = 0; for (let i = 0; i < n; i++) md = Math.max(md, Math.abs(O[i] - ref[i]));
    rec.verified = md < 1e-3; rec.ms = performance.now() - t0; rec.shards = sh;
    if (data) { rec.output = f32ToB64(O); rec.outLen = O.length; }
  } else if (ROWWISE.has(rec.kernel)) {
    // per-row op (softmax/layernorm): N = columns per row; shard whole rows across the fleet.
    const cols = Math.max(1, data ? (input!.N ?? 128) : 128);
    const a = data ? input!.a! : new Float32Array(rec.size * cols).map(() => Math.random() * 4 - 2);
    const rows = Math.floor(a.length / cols);
    if (rows < 1) throw new Error(`${rec.kernel}: need at least one full row of ${cols} columns`);
    const rowsPer = Math.ceil(rows / fleet.length);
    const parts = await Promise.all(fleet.map(async (w, i) => {
      const r0 = i * rowsPer, rn = Math.min(rows, r0 + rowsPer) - r0; if (rn <= 0) return null;
      const { out, worker } = await dispatchResilient(rec.id, { kernel: rec.kernel, a: f32ToB64(a.slice(r0 * cols, (r0 + rn) * cols)), cols }, w);
      return { r0, out, w: worker };
    }));
    const O = new Float32Array(rows * cols); const sh: JobRec['shards'] = [];
    for (const p of parts) if (p) { O.set(p.out, p.r0 * cols); sh.push({ worker: p.w.id, backend: p.w.label, work: p.out.length, ms: 0 }); }
    const ref = cpuRowwise(rec.kernel, a, cols); let md = 0; for (let i = 0; i < O.length; i++) md = Math.max(md, Math.abs(O[i] - ref[i]));
    rec.verified = md < 1e-4; rec.ms = performance.now() - t0; rec.shards = sh;
    if (data) { rec.output = f32ToB64(O); rec.outLen = O.length; }
  } else { throw new Error(`unknown kernel ${rec.kernel}`); }
  const js = jobSigned.get(rec.id); rec.signed = !!js && js.total > 0 && js.signed === js.total; jobSigned.delete(rec.id);
  log('info', `${rec.id} done: ${rec.kernel} · ${(rec.ms ?? 0).toFixed(0)}ms · verified=${rec.verified} · signed=${rec.signed}${data ? ' · data' : ''}`);
}

// ---------- device descriptor (the pool presented as a real GPU slot) ----------
const LIMITS = { maxMatmulDim: 2048, maxElements: 8_000_000, maxInputElements: 16_000_000 };
function deviceDescriptor() {
  const g = virtualGpu();
  return {
    name: 'MoreGPU-Pool',
    kind: 'virtual-gpu',
    backends: [...new Set([...workers.values()].map((w) => w.backend))],
    vendors: [...new Set([...workers.values()].map((w) => w.label.split(':')[1]?.split('/')[0] ?? w.backend))],
    slots: g.slots, gpuSlots: g.gpuSlots, cpuSlots: g.cpuSlots, busy: g.busy,
    kernels: KERNELS,
    limits: LIMITS,
    queue: { depth: g.queueDepth, running: g.busy },
    throughput: { totalOps: g.totalOps, totalUnits: g.totalUnits, totalShards: g.totalShards, tokensServed: g.totalTokens, trend: g.poolTrend },
    seal: 'AES-256-GCM',
    capabilities: { dataMode: true, verifiedResults: true, signedResults: true, adaptiveThrottle: true, asyncSubmit: true, sealedWire: true, tokenIsolated: true },
  };
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
  const totalOps = fleet.reduce((s, w) => s + w.ops, 0);
  const totalTokens = fleet.reduce((s, w) => s + w.tokens, 0);
  return { device: 'MoreGPU-Pool', slots, gpuSlots: gpu, cpuSlots: slots - gpu, busy: fleet.filter((w) => w.busy).length, avgUserUtil: +avgUtil.toFixed(3), avgPoolDuty: +avgDuty.toFixed(3), queueDepth: queue.length, totalUnits, totalShards, totalOps, totalTokens, poolTrend: poolHistory, perKernel: KM, sealed: 'AES-256-GCM' };
}

// ---------- HTTP + WS ----------
const json = (o: unknown, status = 200) => new Response(JSON.stringify(o, null, 2), { status, headers: { 'content-type': 'application/json' } });
const authOk = (req: Request) => constEq((req.headers.get('authorization') ?? '').replace(/^Bearer\s+/i, '') || (req.headers.get('x-admin-token') ?? ''), cfg.adminToken);
const KERNELS = ['matmul', 'vector_add', 'vector_mul', 'saxpy', 'relu', 'scale', 'gelu', 'softmax', 'layernorm'];

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  if (url.pathname === '/ws') { const { socket, response } = Deno.upgradeWebSocket(req); wireWorker(socket); return response; }
  if (url.pathname === '/health') return json({ ok: true, fleet: workers.size, queue: queue.length });
  if (url.pathname === '/help') return json({ kernels: KERNELS, endpoints: ['/ (dashboard)', '/health', '/device', '/gpu', '/workers', 'POST /workers/:id/control', 'POST /submit (?async=1)', '/jobs', '/jobs/:id', '/logs', '/metrics'], workerSchedule: 'MOREGPU_SCHEDULE=always|idle-only|HH:MM-HH:MM', auth: 'admin endpoints require Authorization: Bearer <admin token>' });
  // admin-gated
  if (['/gpu', '/device', '/workers', '/jobs', '/logs', '/metrics', '/weights', '/net'].some((p) => url.pathname === p || url.pathname.startsWith('/jobs/')) || url.pathname.startsWith('/workers/') || url.pathname.startsWith('/train/') || url.pathname.startsWith('/model/') || (req.method === 'POST' && url.pathname === '/submit')) {
    if (!authOk(req)) return json({ error: 'unauthorized — send Authorization: Bearer <admin token>' }, 401);
  }
  if (url.pathname === '/gpu') return json(virtualGpu());
  // Fleet network self-test: MoreGPU's viable workloads depend entirely on round-trip latency. Sub-ms (a LAN of
  // idle desktops) makes pipeline/expert sharding of big models viable; 100s of ms (a public tunnel) means only
  // single-node serving + coarse DiLoCo make sense. Measure each torch worker's RTT (min of a few timed pings)
  // and give an HONEST, latency-aware recommendation — no guessing whether your fleet can host a big model.
  if (url.pathname === '/net') {
    const cands = torchWorkers();
    const BW_BYTES = 512 << 10; // 512 KiB probe — enough to gauge throughput, small enough not to stall a slow link
    const bwBuf = new Uint8Array(BW_BYTES); // getRandomValues caps at 64 KiB/call → fill in chunks (entropy avoids any deflate skew)
    for (let o = 0; o < BW_BYTES; o += 65536) crypto.getRandomValues(bwBuf.subarray(o, Math.min(o + 65536, BW_BYTES)));
    const blob = b64e(bwBuf);
    const BW_CAP_MS = 10_000; // one slow/dead node must never hang the self-test → cap each probe and report what we got
    const cap = <T>(p: Promise<{ ok: boolean } & T>, ms: number) => Promise.race([p, new Promise<{ ok: false }>((res) => setTimeout(() => res({ ok: false }), ms))]);
    const rows = await Promise.all(cands.map(async (w) => {
      let rtt = Infinity;
      for (let i = 0; i < 4; i++) { const t0 = performance.now(); const r = await cap(modelRPC(w, 'ping', {}), 4000); if (r.ok) rtt = Math.min(rtt, performance.now() - t0); else break; }
      let mbps: number | null = null;
      const t1 = performance.now(); const br = await cap(modelRPC(w, 'ping', { blob }), BW_CAP_MS);
      if (br.ok) { const secs = Math.max(1e-3, (performance.now() - t1 - (Number.isFinite(rtt) ? rtt : 0)) / 1000); mbps = +((BW_BYTES / 1e6) / secs).toFixed(1); }
      // if the probe didn't finish in the cap, the link is slower than 512KB/10s ≈ 0.4 Mbps → report that honestly
      const slowNote = (mbps == null && Number.isFinite(rtt)) ? `< ${((BW_BYTES * 8 / 1e6) / (BW_CAP_MS / 1000)).toFixed(1)} Mbps (probe capped at ${BW_CAP_MS / 1000}s)` : undefined;
      return { id: w.id, backend: w.label, rtt_ms: Number.isFinite(rtt) ? +rtt.toFixed(2) : null, up_mbps: mbps, up_note: slowNote };
    }));
    const rtts = rows.map((r) => r.rtt_ms).filter((x): x is number => x != null).sort((a, b) => a - b);
    const median = rtts.length ? rtts[Math.floor(rtts.length / 2)] : null;
    const bws = rows.map((r) => r.up_mbps).filter((x): x is number => x != null).sort((a, b) => a - b);
    const medBw = bws.length ? bws[Math.floor(bws.length / 2)] : null;
    // Two independent limits: LATENCY (round-trips) gates single-request sharded DECODE; BANDWIDTH gates weight
    // transfer + throughput. A fat internet pipe does NOT lower RTT — so distinguish honestly.
    const latencyBound = median == null ? [] :
      median < 2 ? ['single-request tensor / expert parallelism (2·layers syncs/token) — needs a LAN, and you have one']
      : median < 20 ? ['single-request pipeline-sharded decode (a few hops/token) — workable; per-token tensor/expert parallelism is marginal']
      : ['single-request sharded decode is latency-bound at this RTT — it will be slow per token no matter your bandwidth (round-trips can\'t be bought with Mbps)'];
    const alwaysWorks = ['single-node resident serving (whole model on one node — any RTT)',
      'DiLoCo LoRA fine-tuning (syncs rarely + small adapters — latency-tolerant, fine over the internet if bandwidth allows)',
      'download-free model loading (bandwidth-bound — "if the internet allows" applies here: fast pipe ⇒ fast load)',
      'throughput serving for many users (batching hides per-request latency — a WAN with good bandwidth still gives high aggregate tok/s)'];
    const note = median == null ? 'no torch workers to probe' :
      median < 2 ? 'LAN-class RTT — the full menu is viable here, including single-request sharded/expert-parallel hosting of big models.' :
      median < 20 ? 'fast link — single-request pipeline sharding is viable; the internet-tolerant workloads all work.' :
      'WAN-class RTT — over the internet, use the latency-TOLERANT workloads (single-node serving, DiLoCo fine-tuning, download-free loading, throughput-batched serving). Single-request sharded DECODE stays slow: that is round-trip latency, which more bandwidth cannot fix.';
    return json({ ok: true, workers: rows, median_rtt_ms: median, median_up_mbps: medBw, latency_tolerant_anywhere: alwaysWorks, latency_bound_needs_low_rtt: latencyBound, note });
  }
  if (url.pathname === '/device') return json(deviceDescriptor());
  if (url.pathname === '/workers') {
    const totalOps = [...workers.values()].reduce((s, w) => s + w.ops, 0) || 1; // share of compute OPERATIONS (consistent unit)
    // which workers currently HOLD a resident model or a pipeline stage → they read "serving" even while idle between requests
    const serveWorkers = new Set<string>([...modelHome.values(), ...[...shardPlans.values()].flatMap((p) => p.stages.map((s) => s.worker))]);
    return json([...workers.values()].map((w) => ({
      id: w.id, nick: w.nick, backend: w.backend, label: w.label, os: w.os, userUtil: w.util, poolDuty: w.duty, ceil: w.ceil, busy: w.busy,
      paused: w.paused, pausedReason: w.pausedReason ?? null, schedule: w.schedule ?? 'always', serving: serveWorkers.has(w.id),
      shards: w.shards, ops: w.ops, units: w.units, tokens: w.tokens, share: +(w.ops / totalOps).toFixed(3), errors: w.errors,
      avgMs: w.shards ? +(w.totalMs / w.shards).toFixed(1) : 0, uptimeS: Math.round((Date.now() - w.joinedAt) / 1000), trend: w.history,
    })));
  }
  // Admin control of a single worker: pause/resume, cap its duty, set its schedule, relabel, or remove it.
  //   POST /workers/:id/control  { "action": "pause"|"resume"|"remove", "ceil": 0.4, "schedule": "22:00-07:00", "nick": "lab-3" }
  if (req.method === 'POST' && url.pathname.startsWith('/workers/') && url.pathname.endsWith('/control')) {
    const id = decodeURIComponent(url.pathname.slice('/workers/'.length, -'/control'.length));
    const w = workers.get(id);
    if (!w) return json({ error: 'worker not found' }, 404);
    const body = await req.json().catch(() => ({})) as { action?: string; ceil?: number; schedule?: string; nick?: string };
    if (body.action === 'remove') { if (w.pubkeyB64) removedPubkeys.add(w.pubkeyB64); try { w.ws.close(); } catch { /* */ } workers.delete(id); log('warn', `admin removed worker ${id}${w.pubkeyB64 ? ' (banned by key)' : ' (unsigned — could rejoin under a new id)'}`); return json({ ok: true, removed: id, banned: !!w.pubkeyB64 }); }
    const ctl: Record<string, unknown> = { t: 'control' };
    if (body.action === 'pause') { w.paused = true; w.pausedReason = 'admin'; ctl.pause = true; }
    if (body.action === 'resume') { w.paused = false; w.pausedReason = null; w.consecErrors = 0; ctl.pause = false; }
    if (typeof body.ceil === 'number' && isFinite(body.ceil)) { const c = Math.max(0.05, Math.min(1, body.ceil)); w.duty = c; w.ceil = c; ctl.ceil = c; }
    if (typeof body.schedule === 'string') { w.schedule = body.schedule.trim().toLowerCase().slice(0, 40); ctl.schedule = w.schedule; if (w.pausedReason === 'schedule') { w.paused = false; w.pausedReason = null; } reconcileSchedules(); } // apply the window now (coordinator-owned)
    if (typeof body.nick === 'string') w.nick = body.nick.slice(0, 40);
    try { if (Object.keys(ctl).length > 1) w.ws.send(JSON.stringify(ctl)); } catch { /* */ }
    log('info', `admin control → ${id}: ${JSON.stringify(body)}`);
    if (w.paused === false) pumpQueue(); // resuming may unblock the queue
    return json({ ok: true, id, paused: w.paused, ceil: w.duty, schedule: w.schedule ?? 'always', nick: w.nick });
  }
  // Weight residency: upload a named weight (pinned resident on one worker), or list what's cached.
  if (url.pathname === '/weights') {
    if (req.method === 'POST') {
      const body = await req.json().catch(() => ({})) as { id?: string; data?: string; rows?: number; cols?: number; worker?: string; dtype?: string };
      if (!body.id || typeof body.data !== 'string' || !body.rows || !body.cols) return json({ error: 'need {id, data (base64), rows, cols, dtype?, worker?}' }, 400);
      const dtype: 'f32' | 'f16' = body.dtype === 'f16' ? 'f16' : 'f32';
      const bytes = b64d(body.data);
      // Coordinator keeps a DEQUANTIZED f32 copy so its cpuMatmul verify matches the worker's f16 result.
      const arr = dtype === 'f16' ? Float32Array.from(new Float16Array(bytes.buffer)) : new Float32Array(bytes.buffer);
      if (arr.length !== body.rows * body.cols) return json({ error: `data length ${arr.length} != rows*cols ${body.rows * body.cols}` }, 400);
      // Cap per-weight size so a huge upload gets a clean 413 instead of OOM-ing the coordinator (a sealed
      // WS frame carries the base64 payload). Big projections (e.g. an LM head) should be tiled or host-side.
      const WCAP = Number(Deno.env.get('MOREGPU_MAX_WEIGHT_ELEMENTS') ?? 16_000_000);
      if (arr.length > WCAP) return json({ error: `weight too large (${arr.length} > ${WCAP} elements) — tile it or keep it host-side` }, 413);
      const active = activeFleet();
      if (active.length === 0) return json({ error: 'no active worker to hold the weight' }, 503);
      // home = explicit worker, else the active worker holding the fewest weights (spread the model across GPUs)
      const home = body.worker ? active.find((w) => w.id === body.worker) : active.slice().sort((a, b) => residentCount(a.id) - residentCount(b.id))[0];
      if (!home) return json({ error: `worker ${body.worker} not active` }, 404);
      const sealed = await seal(TENANT_KEY, new TextEncoder().encode(JSON.stringify({ data: body.data, rows: body.rows, cols: body.cols, dtype })));
      const ack = new Promise<{ ok: boolean; error?: string }>((res) => { const t = setTimeout(() => { pendingCache.delete(body.id!); res({ ok: false, error: 'cache ack timeout' }); }, 30_000); pendingCache.set(body.id!, { workerId: home.id, cb: (r) => { clearTimeout(t); res(r); } }); });
      home.ws.send(JSON.stringify({ t: 'cache', id: body.id, sealed }));
      const r = await ack;
      if (!r.ok) return json({ error: `cache failed on ${home.id}: ${r.error}` }, 502);
      weightHome.set(body.id, { worker: home.id, rows: body.rows, cols: body.cols, dtype });
      weightStore.set(body.id, arr);
      log('info', `weight ${body.id} (${body.rows}x${body.cols} ${dtype}) → resident on ${home.id}`);
      return json({ ok: true, id: body.id, worker: home.id, rows: body.rows, cols: body.cols, dtype });
    }
    return json([...weightHome.entries()].map(([id, h]) => ({ id, worker: h.worker, rows: h.rows, cols: h.cols })));
  }
  // On-pool fine-tuning (single native torch worker). load → repeated step → adapter pull.
  //   POST /train/load  { model, rank?, alpha?, lr?, seed?, targets?, worker? }
  //   POST /train/step  { input_ids:[...], labels?:[...], lr? }  → { loss, step }
  //   POST /train/adapter                                        → { step, tensors:{name:{data,shape}} }
  if (req.method === 'POST' && url.pathname === '/train/load') {
    const body = await req.json().catch(() => ({})) as { model?: string; worker?: string; rank?: number; alpha?: number; lr?: number; seed?: number; targets?: string[]; force?: boolean };
    if (!body.model) return json({ error: 'need {model}' }, 400);
    // refuse to clobber a live single-worker session unless {force:true} (would silently reset its adapter/optimizer/step)
    if (trainingHome && workers.has(trainingHome) && !body.force) return json({ error: `a training session is already live on ${trainingHome} — pass {force:true} to replace it`, worker: trainingHome }, 409);
    const cands = torchWorkers();
    if (cands.length === 0) return json({ error: 'no native (torch) worker connected — start apps/worker/worker_torch.py; the WebGPU workers cannot train (no autograd)' }, 503);
    const w = body.worker ? cands.find((x) => x.id === body.worker) : cands[0];
    if (!w) return json({ error: `torch worker ${body.worker} not active` }, 404);
    // A worker holds ONE global TRAIN slot (shared by /train and /train/diloco), so it can't host both at
    // once. TODO(review): full fix is to key training sessions by id on the worker; this guard prevents the
    // silent cross-session corruption until then.
    if (diloco && diloco.workers.includes(w.id)) return json({ error: `worker ${w.id} is part of a live DiLoCo group — it shares the single global TRAIN slot, so it can't also host a /train session; unload the DiLoCo group or pick another worker`, worker: w.id }, 409);
    const r = await trainRPC(w, 'load', { model: body.model, rank: body.rank ?? 8, alpha: body.alpha ?? 16, lr: body.lr ?? 1e-3, seed: body.seed ?? 0, targets: body.targets });
    if (!r.ok) return json({ error: r.error }, 502);
    trainingHome = w.id;
    log('info', `train session loaded ${body.model} on ${w.id} · trainable=${r.data?.trainable_params}`);
    return json({ ok: true, worker: w.id, ...r.data });
  }
  if (req.method === 'POST' && (url.pathname === '/train/step' || url.pathname === '/train/adapter')) {
    if (!trainingHome) return json({ error: 'no training session — POST /train/load first' }, 409);
    const w = workers.get(trainingHome);
    if (!w) { trainingHome = null; return json({ error: 'training worker disconnected — reload the session' }, 503); }
    if (url.pathname === '/train/adapter') {
      const r = await trainRPC(w, 'adapter', {});
      return r.ok ? json({ ok: true, ...r.data }) : json({ error: r.error }, 502);
    }
    const body = await req.json().catch(() => ({})) as { input_ids?: number[]; labels?: number[]; lr?: number };
    if (!Array.isArray(body.input_ids) || body.input_ids.length === 0) return json({ error: 'need {input_ids:[...]}' }, 400);
    const okInts = (a?: number[]) => !a || (a.length <= 100_000 && a.every((x) => Number.isInteger(x) && x >= 0));
    // labels may also carry HF's ignore index -100 to mask a position out of the loss (input_ids may not)
    const okLabels = (a?: number[]) => !a || (a.length <= 100_000 && a.every((x) => Number.isInteger(x) && (x === -100 || x >= 0)));
    if (!okInts(body.input_ids) || !okLabels(body.labels)) return json({ error: 'input_ids must be non-negative ints ≤100000; labels may also be -100 (ignore index)' }, 400);
    const r = await trainRPC(w, 'step', { input_ids: body.input_ids, labels: body.labels, lr: body.lr });
    if (!r.ok) return json({ error: r.error }, 502);
    return json({ ok: true, ...r.data });
  }
  // DiLoCo distributed LoRA: load the same seeded adapter on N workers, then run rounds (each worker does
  // H local steps on its shard; the coordinator averages + outer-Nesterov-steps + broadcasts the global).
  if (req.method === 'POST' && url.pathname === '/train/diloco/load') {
    const body = await req.json().catch(() => ({})) as { model?: string; rank?: number; alpha?: number; lr?: number; seed?: number; targets?: string[]; workers?: string[] };
    if (!body.model) return json({ error: 'need {model}' }, 400);
    const cands = torchWorkers();
    const group = (Array.isArray(body.workers) && body.workers.length) ? cands.filter((w) => body.workers!.includes(w.id)) : cands;
    if (group.length === 0) return json({ error: 'no native (torch) workers connected for DiLoCo' }, 503);
    // Each worker holds ONE global TRAIN slot shared by /train and /train/diloco. TODO(review): full fix is
    // per-session keying on the worker. Loading a DiLoCo group re-loads (and resets) every member's TRAIN slot
    // below, so it SUPERSEDES any single-worker /train session on those workers — end that session cleanly
    // rather than refusing (the old session's next /train/step then gets a clean 409). Only a busy round blocks.
    if (diloco && diloco.busy) return json({ error: 'a DiLoCo round is in progress — reload the group after it finishes' }, 409);
    if (trainingHome && group.some((w) => w.id === trainingHome)) { log('info', `DiLoCo supersedes the single-worker /train session on ${trainingHome}`); trainingHome = null; }
    const loads = await Promise.all(group.map((w) => trainRPC(w, 'load', { model: body.model, rank: body.rank ?? 8, alpha: body.alpha ?? 16, lr: body.lr ?? 1e-3, seed: body.seed ?? 0, targets: body.targets, no_dropout: true })));
    const okWorkers = group.filter((_, i) => loads[i].ok);
    if (okWorkers.length === 0) return json({ error: `train_load failed on all workers: ${loads[0]?.error}` }, 502);
    const init = await trainRPC(okWorkers[0], 'adapter', {});
    if (!init.ok) return json({ error: `adapter pull failed: ${init.error}` }, 502);
    const global = tensorsFromReply(init.data!);
    const shape = new Map<string, number[]>();
    for (const [n, v] of Object.entries((init.data!.tensors ?? {}) as Record<string, { shape: number[] }>)) shape.set(n, v.shape);
    const momentum = new Map<string, Float32Array>();
    for (const [n, a] of global) momentum.set(n, new Float32Array(a.length));
    diloco = { workers: okWorkers.map((w) => w.id), model: body.model, global, momentum, shape, round: 0, busy: false };
    const trainable = [...global.values()].reduce((s, a) => s + a.length, 0);
    log('info', `DiLoCo group loaded ${body.model} on ${diloco.workers.length} workers · adapter params=${trainable}`);
    return json({ ok: true, workers: diloco.workers, model: body.model, adapter_params: trainable });
  }
  if (req.method === 'POST' && url.pathname === '/train/diloco/round') {
    if (!diloco) return json({ error: 'no DiLoCo group — POST /train/diloco/load first' }, 409);
    if (diloco.busy) return json({ error: 'a DiLoCo round is already in progress — rounds are serialized' }, 409);
    const body = await req.json().catch(() => ({})) as { inner_steps?: number; lr?: number; outer_lr?: number; outer_momentum?: number; batches?: Record<string, number[][]> };
    const H = Math.max(1, Math.min(Number(body.inner_steps ?? 4), 1000));
    const lr = Number(body.lr ?? 1e-3), outerLr = Number(body.outer_lr ?? 0.7), mom = Number(body.outer_momentum ?? 0.9);
    if (![lr, outerLr, mom].every((x) => Number.isFinite(x))) return json({ error: 'lr/outer_lr/outer_momentum must be finite numbers' }, 400);
    const batches = (body.batches ?? {}) as Record<string, number[][]>;
    // Every DiLoCo batch bypasses the /train/step path, so apply the same id/size validation here: each
    // window is non-negative int token ids capped at 100000 tokens, capped at 100000 windows per worker.
    // (train_inner also applies the context-window guard on the worker.)
    const okInts = (a?: number[]) => !a || (a.length <= 100_000 && a.every((x) => Number.isInteger(x) && x >= 0));
    const okBatch = (rows?: number[][]) => !rows || (Array.isArray(rows) && rows.length <= 100_000 && rows.every((r) => Array.isArray(r) && okInts(r)));
    for (const [wid, rows] of Object.entries(batches)) if (!okBatch(rows)) return json({ error: `batches[${wid}] must be arrays of non-negative int token ids (each window ≤100000 tokens, ≤100000 windows)` }, 400);
    const live = diloco.workers.map((id) => workers.get(id)).filter((w): w is Worker => !!w);
    if (live.length === 0) { diloco = null; return json({ error: 'all DiLoCo workers disconnected' }, 503); }
    diloco.busy = true;
    try {
    const results = await Promise.all(live.map((w) => trainRPC(w, 'inner', { steps: H, lr, batches: batches[w.id] ?? batches['*'] ?? [] })));
    const okWorkers = live.map((w, i) => ({ w, r: results[i] })).filter((x) => x.r.ok && x.r.data?.tensors);
    // Reject a worker whose adapter carries ANY non-finite value (a diverged/exploded local step, or a poisoned
    // node) BEFORE it enters the average — one NaN/Inf would corrupt the whole global adapter for every worker.
    const nonFinite: string[] = [];
    const good = okWorkers.filter(({ w, r }) => {
      for (const arr of tensorsFromReply(r.data!).values()) for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) { nonFinite.push(w.id); return false; }
      return true;
    });
    if (nonFinite.length) log('warn', `DiLoCo: dropped ${nonFinite.join(', ')} from the average — non-finite adapter (diverged/poisoned)`);
    if (good.length === 0) return json({ error: `inner step produced no usable adapter${nonFinite.length ? ` (all non-finite: ${nonFinite.join(', ')})` : `: ${results[0]?.error}`}` }, 502);
    // average the surviving workers' post-inner-step adapters
    const avg = new Map<string, Float32Array>();
    for (const n of diloco.global.keys()) {
      const acc = new Float32Array(diloco.global.get(n)!.length); let cnt = 0;
      for (const { r } of good) { const a = tensorsFromReply(r.data!).get(n); if (a && a.length === acc.length) { for (let i = 0; i < acc.length; i++) acc[i] += a[i]; cnt++; } }
      if (cnt > 0) for (let i = 0; i < acc.length; i++) acc[i] /= cnt;
      avg.set(n, acc);
    }
    // outer Nesterov step on the pseudo-gradient Δ = global − avg
    for (const n of diloco.global.keys()) {
      const g = diloco.global.get(n)!, v = diloco.momentum.get(n)!, a = avg.get(n)!;
      for (let i = 0; i < g.length; i++) { const d = g[i] - a[i]; v[i] = mom * v[i] + d; g[i] = g[i] - outerLr * (d + mom * v[i]); }
    }
    // broadcast the new global to every worker, and VERIFY each accepted it: a worker that fails set_adapter
    // is now out of sync with the global adapter, so drop it from the group (its next round would otherwise
    // train from a stale adapter and silently corrupt the reduce). If ALL fail, dissolve the group.
    const tensors: Record<string, { data: string; shape: number[] }> = {};
    for (const [n, a] of diloco.global) tensors[n] = { data: f32ToB64(a), shape: diloco.shape.get(n)! };
    const bcast = await Promise.all(live.map((w) => trainRPC(w, 'set_adapter', { tensors })));
    const dropped = live.filter((_, i) => !bcast[i].ok).map((w) => w.id);
    if (dropped.length) {
      const bad = new Set(dropped);
      diloco.workers = diloco.workers.filter((wid) => !bad.has(wid));
      log('warn', `DiLoCo: set_adapter broadcast failed on ${dropped.join(', ')} — dropped from the group (out of sync)`);
    }
    if (diloco.workers.length === 0) { diloco = null; return json({ error: 'all DiLoCo workers fell out of sync (set_adapter broadcast failed) — group dissolved', dropped }, 502); }
    diloco.round++;
    const worker_losses = good.map(({ w, r }) => ({ worker: w.id, losses: r.data!.losses as number[] }));
    const avg_last_loss = worker_losses.reduce((s, x) => s + (x.losses[x.losses.length - 1] ?? 0), 0) / worker_losses.length;
    log('info', `DiLoCo round ${diloco.round}: ${good.length} workers · avg last loss ${avg_last_loss.toFixed(4)}${dropped.length ? ` · dropped ${dropped.length}` : ''}`);
    return json({ ok: true, round: diloco.round, workers: good.length, avg_last_loss, worker_losses, dropped: dropped.length ? dropped : undefined, dropped_nonfinite: nonFinite.length ? nonFinite : undefined });
    } finally { if (diloco) diloco.busy = false; }
  }
  if (req.method === 'POST' && url.pathname === '/train/diloco/adapter') {
    if (!diloco) return json({ error: 'no DiLoCo group' }, 409);
    const tensors: Record<string, { data: string; shape: number[] }> = {};
    for (const [n, a] of diloco.global) tensors[n] = { data: f32ToB64(a), shape: diloco.shape.get(n)! };
    return json({ ok: true, round: diloco.round, workers: diloco.workers, tensors });
  }
  // Resident-model serving: hold a whole model on a torch worker and run the ENTIRE forward per call —
  // ONE round-trip per token instead of the ~500 the fine-grained kernel path needs. THIS is fast serving.
  //   POST /model/load     { model, id?, fp16?, worker? }
  //   POST /model/forward  { id, input_ids:[...], return_logits?, topk? }  → { argmax, logits?, top? }
  if (req.method === 'POST' && url.pathname === '/model/load') {
    const body = await req.json().catch(() => ({})) as { model?: string; id?: string; fp16?: boolean; worker?: string; push?: boolean; async?: boolean };
    if (!body.model) return json({ error: 'need {model}' }, 400);
    const cands = torchWorkers();
    if (cands.length === 0) return json({ error: 'no native (torch) worker connected — start apps/worker/worker_torch.py; resident-model serving needs one' }, 503);
    const w = body.worker ? cands.find((x) => x.id === body.worker) : cands[0];
    if (!w) return json({ error: `torch worker ${body.worker} not active` }, 404);
    const mid = body.id ?? body.model;
    // Already resident on the target worker? reuse it instantly — a re-Load must not re-stream a live model.
    if (modelHome.get(mid) === w.id) {
      const s = modelLoads.get(mid);
      log('info', `resident model ${mid} already on ${w.id} — reusing (no re-load)`);
      return json({ ok: true, worker: w.id, id: mid, status: 'ready', reused: true, ...(s?.info ?? {}) });
    }
    if (body.push) {
      // download-free: coordinator fetches the weights and streams them to the worker (no hub access on the worker)
      if (body.async) {
        // return immediately (short request → survives a public tunnel); stream in the background, poll /model/status.
        if (modelLoads.get(mid)?.status === 'loading') return json({ error: `a load for "${mid}" is already in progress`, status: 'loading' }, 409);
        modelLoads.set(mid, { status: 'loading', worker: w.id, model: body.model!, started: Date.now() });
        pushModelToWorker(w, body.model!, mid, !!body.fp16)
          .then((info) => { modelHome.set(String(info.id ?? mid), w.id); modelLoads.set(mid, { status: 'ready', worker: w.id, model: body.model!, started: modelLoads.get(mid)!.started, info }); log('info', `resident model ${mid} weight-pushed to ${w.id} · ${info.n_params} params · staging=${info.staging}`); })
          .catch((e) => { modelLoads.set(mid, { status: 'error', worker: w.id, model: body.model!, started: modelLoads.get(mid)!.started, error: e instanceof Error ? e.message : String(e) }); log('warn', `weight-push ${mid} → ${w.id} failed: ${e instanceof Error ? e.message : e}`); });
        return json({ ok: true, worker: w.id, id: mid, status: 'loading', poll: `/model/status?id=${encodeURIComponent(mid)}` });
      }
      try {
        const info = await pushModelToWorker(w, body.model, mid, !!body.fp16);
        modelHome.set(String(info.id ?? mid), w.id);
        modelLoads.set(mid, { status: 'ready', worker: w.id, model: body.model!, started: Date.now(), info });
        log('info', `resident model ${info.id ?? mid} weight-pushed to ${w.id} · ${info.n_params} params · staging=${info.staging}`);
        return json({ ok: true, worker: w.id, ...info });
      } catch (e) { return json({ error: `weight push failed: ${e instanceof Error ? e.message : e}` }, 502); }
    }
    const r = await modelRPC(w, 'load', { model: body.model, id: mid, fp16: !!body.fp16 });
    if (!r.ok) return json({ error: r.error }, 502);
    const rid = String(r.data?.id ?? mid);
    modelHome.set(rid, w.id);
    modelLoads.set(rid, { status: 'ready', worker: w.id, model: body.model!, started: Date.now(), info: r.data }); // so /model/status is complete
    log('info', `resident model ${r.data?.id} loaded on ${w.id} · ${r.data?.n_params} params`);
    return json({ ok: true, worker: w.id, ...r.data });
  }
  // Poll an async download-free load: GET /model/status?id=<model id> → { status: loading|ready|error, … }
  if (req.method === 'GET' && url.pathname === '/model/status') {
    const id = url.searchParams.get('id') ?? [...modelLoads.keys()].pop();
    if (!id) return json({ status: 'unknown', models: [...modelHome.keys()] });
    if (modelHome.has(id)) { const s = modelLoads.get(id); return json({ status: 'ready', id, worker: modelHome.get(id), ...(s?.info ?? {}) }); }
    const s = modelLoads.get(id);
    if (!s) return json({ status: 'unknown', id });
    return json({ status: s.status, id, worker: s.worker, elapsed_ms: Date.now() - s.started, ...(s.error ? { error: s.error } : {}), ...(s.info ?? {}) });
  }
  if (req.method === 'POST' && (url.pathname === '/model/forward' || url.pathname === '/model/generate')) {
    const body = await req.json().catch(() => ({})) as { id?: string; input_ids?: number[]; return_logits?: boolean; topk?: number; max_new_tokens?: number };
    if (!body.id && modelHome.size > 1) return json({ error: `${modelHome.size} models resident — specify {id} (one of ${[...modelHome.keys()].join(', ')})` }, 400);
    const mid = body.id ?? [...modelHome.keys()][0];
    if (!mid || !modelHome.has(mid)) return json({ error: 'no resident model — POST /model/load first' }, 409);
    if (!Array.isArray(body.input_ids) || body.input_ids.length === 0) return json({ error: 'need {input_ids:[...]}' }, 400);
    if (body.input_ids.length > 100_000 || !body.input_ids.every((x) => Number.isInteger(x) && x >= 0)) return json({ error: 'input_ids must be non-negative ints, ≤100000' }, 400);
    const w = workers.get(modelHome.get(mid)!);
    if (!w) { modelHome.delete(mid); return json({ error: 'model worker disconnected — reload it' }, 503); }
    if (url.pathname === '/model/generate') {
      const n = Math.max(1, Math.min(Number(body.max_new_tokens ?? 16), 1024));
      const r = await modelRPC(w, 'generate', { id: mid, input_ids: body.input_ids, max_new_tokens: n });
      return r.ok ? json({ ok: true, ...r.data }) : json({ error: r.error }, 502);
    }
    const r = await modelRPC(w, 'forward', { id: mid, input_ids: body.input_ids, return_logits: !!body.return_logits, topk: body.topk });
    return r.ok ? json({ ok: true, ...r.data }) : json({ error: r.error }, 502);
  }
  // Text in → text out (the worker tokenizes with the model's own tokenizer). Powers the /chat page.
  if (req.method === 'POST' && url.pathname === '/model/chat') {
    const body = await req.json().catch(() => ({})) as { id?: string; prompt?: string; max_new_tokens?: number; do_sample?: boolean; temperature?: number };
    const mid = body.id ?? [...modelHome.keys()][0];
    if (!mid || !modelHome.has(mid)) return json({ error: 'no resident model — POST /model/load first', models: [...modelHome.keys()] }, 409);
    if (typeof body.prompt !== 'string' || !body.prompt.trim()) return json({ error: 'need {prompt}' }, 400);
    const w = workers.get(modelHome.get(mid)!);
    if (!w) { modelHome.delete(mid); return json({ error: 'model worker disconnected — reload it' }, 503); }
    const r = await modelRPC(w, 'chat', { id: mid, prompt: body.prompt.slice(0, 8000), max_new_tokens: Math.max(1, Math.min(Number(body.max_new_tokens ?? 96), 1024)), do_sample: !!body.do_sample, temperature: body.temperature });
    return r.ok ? json({ ok: true, worker: w.id, ...r.data }) : json({ error: r.error }, 502);
  }
  if (req.method === 'POST' && url.pathname === '/model/unload') {
    const body = await req.json().catch(() => ({})) as { id?: string };
    const mid = body.id ?? [...modelHome.keys()][0];
    if (!mid || !modelHome.has(mid)) return json({ error: 'no such resident model' }, 404);
    const w = workers.get(modelHome.get(mid)!);
    modelHome.delete(mid);
    if (w) await modelRPC(w, 'unload', { id: mid });
    log('info', `resident model ${mid} unloaded`);
    return json({ ok: true, id: mid });
  }
  // Pipeline-parallel sharding (GPT-2 family only). Split the layer range into contiguous stages across
  // the torch workers, load each stage on its worker, and record the plan.
  //   POST /model/shard          { model, id?, layers?, workers? }              → { id, stages:[{worker,start,end,first,last,params_held}] }
  //   POST /model/shard_forward  { id?, input_ids:[...], return_logits? }       → { argmax, logits? }  (pipes hidden state stage→stage)
  //   POST /model/shard_unload   { id? }                                        → unload every stage + drop the plan
  if (req.method === 'GET' && url.pathname === '/model/shard_status') {
    const id = url.searchParams.get('id') ?? [...shardLoads.keys()].pop();
    if (!id) return json({ status: 'unknown', shards: [...shardPlans.keys()] });
    const s = shardLoads.get(id);
    if (!s) return json({ status: shardPlans.get(id)?.stages.length ? 'ready' : 'unknown', id });
    return json({ status: s.status, id, model: s.model, stages_done: s.stagesDone, stages_total: s.stagesTotal, elapsed_ms: Date.now() - s.started, ...(s.error ? { error: s.error } : {}), ...(s.status === 'ready' && s.info ? { stages: s.info } : {}) });
  }
  if (req.method === 'POST' && url.pathname === '/model/shard') {
    const body = await req.json().catch(() => ({})) as { model?: string; id?: string; layers?: number; workers?: string[]; push?: boolean; async?: boolean };
    if (!body.model) return json({ error: 'need {model} (GPT-2 / Llama-family)' }, 400);
    if (body.push && !HF_REPO_RE.test(body.model)) return json({ error: `bad model ref "${body.model}" for download-free shard` }, 400);
    // explicit `workers` picks the stage order (stage i = workers[i]); otherwise use the torch fleet order
    const cands = (Array.isArray(body.workers) && body.workers.length)
      ? body.workers.map((wid) => torchWorkers().find((w) => w.id === wid)).filter((w): w is Worker => !!w)
      : torchWorkers();
    if (cands.length === 0) return json({ error: 'no native (torch) worker connected — start apps/worker/worker_torch.py; pipeline sharding needs ≥1 (≥2 for a real split)' }, 503);
    const sid = body.id ?? body.model;
    if (shardPlans.has(sid)) return json({ error: `shard ${sid} already loaded — POST /model/shard_unload first`, id: sid }, 409);
    shardPlans.set(sid, { model: body.model, stages: [] }); // RESERVE synchronously so a concurrent same-id shard 409s (TOCTOU)
    const fail = (msg: string, code: number) => { shardPlans.delete(sid); return json({ error: msg, id: sid }, code); };
    // DOWNLOAD-FREE: fetch config + the safetensors header ON THE COORDINATOR (no worker download), so the
    // fleet gets only its per-stage slice. Also gives the real layer count without a worker-side model_load.
    let configText: string | null = null, stHeader: STHeader | null = null, stHeaderLen = 0, nLayer = 0;
    if (body.push) {
      try {
        configText = await hfFetchText(body.model, 'config.json', PUSH_INDEX_MAX);
        if (!configText) return fail(`${body.model}: no config.json on HF`, 502);
        const cfg = JSON.parse(configText) as Record<string, unknown>;
        nLayer = Math.floor(Number(cfg.num_hidden_layers ?? cfg.n_layer ?? 0));
        if (await hfFetchText(body.model, 'model.safetensors.index.json', 4096).catch(() => null))
          return fail(`${body.model} ships SHARDED safetensors — download-free sharding needs a single-file model.safetensors (v1)`, 400);
        const h = await hfSafetensorsHeader(body.model, 'model.safetensors'); stHeader = h.header; stHeaderLen = h.headerLen;
      } catch (e) { return fail(`download-free shard preflight failed: ${e instanceof Error ? e.message : e}`, 502); }
    } else {
      // Resolve the model's REAL layer count (config-only, no weights) so gpt2-medium/large/xl shard correctly.
      const arch = await modelRPC(cands[0], 'arch', { model: body.model });
      if (arch.ok && Number(arch.data?.n_layer) > 0) nLayer = Math.floor(Number(arch.data!.n_layer));
    }
    if (!(nLayer > 0)) nLayer = Number(body.layers) > 0 ? Math.max(1, Math.floor(Number(body.layers))) : 0;
    if (!(nLayer > 0)) return fail(`could not determine layer count for ${body.model} — pass {layers:N} to shard explicitly`, 502);
    const nStages = Math.min(cands.length, nLayer);
    // split [0, nLayer) into nStages contiguous ranges, as even as possible (first `extra` stages get one more)
    const per = Math.floor(nLayer / nStages), extra = nLayer % nStages;
    const stages: ShardStage[] = [];
    let cursor = 0;
    for (let i = 0; i < nStages; i++) {
      const size = per + (i < extra ? 1 : 0);
      stages.push({ worker: cands[i].id, start: cursor, end: cursor + size, first: i === 0, last: i === nStages - 1 });
      cursor += size;
    }
    // Load every stage (streaming its slice for a push shard). Returns the stage info or throws after cleaning
    // up any stages already loaded — so a half-loaded pipe never lingers. Runnable in the background for async.
    const model = body.model, push = !!body.push;
    const runLoad = async (): Promise<(ShardStage & { params_held?: number; bytes?: number })[]> => {
      const info: (ShardStage & { params_held?: number; bytes?: number })[] = [];
      const unloadAll = async () => { for (const d of info) { const dw = workers.get(d.worker); if (dw) await modelRPC(dw, 'shard_unload', { id: sid }).catch(() => {}); } };
      for (const st of stages) {
        const w = workers.get(st.worker);
        if (!w) { await unloadAll(); throw new Error(`stage worker ${st.worker} vanished`); }
        let r: { ok: boolean; data?: Record<string, unknown>; error?: string }; let stageBytes = 0;
        if (push) {
          try {
            const begin = await modelRPC(w, 'push_begin', { id: sid, model }); if (!begin.ok) throw new Error(`push_begin: ${begin.error}`);
            const budget = { left: PUSH_MAX_BYTES };
            const cb = new TextEncoder().encode(configText!);
            const cr = await modelRPC(w, 'push_chunk', { id: sid, name: 'config.json', seq: 0, data: b64e(cb), last: true }); if (!cr.ok) throw new Error(`push config: ${cr.error}`);
            if (st.first) { // stream the tokenizer to the FIRST stage so a browser can chat this sharded model
              for (const f of ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json', 'tokenizer.model', 'chat_template.jinja', 'generation_config.json']) {
                const tr = await hfFetch(model, f); if (tr) await streamFileToWorker(w, sid, f, tr, budget);
              }
            }
            const names = stageTensors(stHeader!, st.start, st.end, st.first, st.last);
            stageBytes = await streamStageSafetensors(w, sid, model, 'model.safetensors', stHeader!, stHeaderLen, names, budget);
            r = await modelRPC(w, 'shard_load', { model, id: sid, start: st.start, end: st.end, first: st.first, last: st.last, push: true });
          } catch (e) { r = { ok: false, error: e instanceof Error ? e.message : String(e) }; }
        } else {
          r = await modelRPC(w, 'shard_load', { model, id: sid, start: st.start, end: st.end, first: st.first, last: st.last });
        }
        if (!r.ok) { await modelRPC(w, 'shard_unload', { id: sid }).catch(() => {}); await unloadAll(); throw new Error(`shard_load failed on ${st.worker} (layers ${st.start}-${st.end}): ${r.error}`); }
        info.push({ ...st, params_held: r.data?.params_held as number | undefined, bytes: stageBytes || undefined });
        const ls = shardLoads.get(sid); if (ls) ls.stagesDone = info.length; // progress for the poller
        if (shardLoads.get(sid)?.aborted) { await unloadAll(); throw new Error('shard load aborted (deadline)'); } // deadline fired → don't resurrect
      }
      if (shardLoads.get(sid)?.aborted) { await unloadAll(); throw new Error('shard load aborted (deadline)'); }
      shardPlans.set(sid, { model, stages }); // finalize the reservation with the real plan (unblocks shard_forward)
      log('info', `sharded ${model} (${nLayer} layers) → ${nStages} stages${push ? ' [download-free]' : ''}: ${stages.map((s) => `${s.worker}[${s.start}-${s.end})`).join(' → ')}`);
      return info;
    };
    if (body.async) {
      // return immediately (short request → survives a slow/tunneled link); stream stages in the background.
      const started = Date.now();
      shardLoads.set(sid, { status: 'loading', model, started, stagesDone: 0, stagesTotal: nStages });
      // Deadline: a stage stalling on a half-dead node would otherwise leave the load 'loading' forever (each
      // push_chunk only errors after RELAY_TIMEOUT_MS). On expiry flag aborted + unload finished stages. NB: this
      // can't cancel an in-flight push_chunk on the worker (no worker-side cancel yet) — it stops the bookkeeping
      // hang and blocks resurrection; runLoad checks `aborted` before finalizing.
      const DEADLINE_MS = Number(Deno.env.get('MOREGPU_SHARD_LOAD_DEADLINE_MS') ?? 1_800_000); // 30 min default
      const deadline = setTimeout(async () => {
        const s = shardLoads.get(sid);
        if (s && s.status === 'loading') {
          shardLoads.set(sid, { ...s, aborted: true, status: 'error', error: `shard load exceeded ${DEADLINE_MS}ms deadline (aborted)` });
          shardPlans.delete(sid);
          for (const st of stages) { const w = workers.get(st.worker); if (w) await modelRPC(w, 'shard_unload', { id: sid }).catch(() => {}); }
          log('warn', `shard ${sid} load timed out after ${DEADLINE_MS}ms — aborted`);
        }
      }, DEADLINE_MS);
      runLoad()
        .then((info) => { clearTimeout(deadline); if (shardLoads.get(sid)?.aborted) return; shardLoads.set(sid, { status: 'ready', model, started, stagesDone: nStages, stagesTotal: nStages, info }); })
        .catch((e) => { clearTimeout(deadline); if (shardLoads.get(sid)?.aborted) return; shardPlans.delete(sid); shardLoads.set(sid, { status: 'error', model, started, stagesDone: shardLoads.get(sid)?.stagesDone ?? 0, stagesTotal: nStages, error: e instanceof Error ? e.message : String(e) }); log('warn', `shard ${sid} failed: ${e instanceof Error ? e.message : e}`); });
      return json({ ok: true, id: sid, model, layers: nLayer, mode: push ? 'download-free' : 'download', stages_total: nStages, status: 'loading', poll: `/model/shard_status?id=${encodeURIComponent(sid)}` });
    }
    try {
      const info = await runLoad();
      shardLoads.set(sid, { status: 'ready', model, started: Date.now(), stagesDone: nStages, stagesTotal: nStages, info });
      return json({ ok: true, id: sid, model, layers: nLayer, mode: push ? 'download-free' : 'download', stages: info });
    } catch (e) { return fail(e instanceof Error ? e.message : String(e), 502); }
  }
  if (req.method === 'POST' && url.pathname === '/model/shard_forward') {
    const body = await req.json().catch(() => ({})) as { id?: string; input_ids?: number[]; return_logits?: boolean };
    const sid = body.id ?? [...shardPlans.keys()][0];
    if (!sid || !shardPlans.has(sid)) return json({ error: 'no sharded model — POST /model/shard first' }, 409);
    if (shardPlans.get(sid)!.stages.length === 0) return json({ error: `shard ${sid} is still loading`, id: sid }, 409);
    if (!Array.isArray(body.input_ids) || body.input_ids.length === 0) return json({ error: 'need {input_ids:[...]}' }, 400);
    if (body.input_ids.length > 100_000 || !body.input_ids.every((x) => Number.isInteger(x) && x >= 0)) return json({ error: 'input_ids must be non-negative ints, ≤100000' }, 400);
    const out = await shardPipe(sid, body.input_ids, !!body.return_logits);
    if (!out.ok) { if (out.disconnected) shardPlans.delete(sid); return json({ error: out.error, id: sid }, out.disconnected ? 503 : 502); }
    return json({ ok: true, id: sid, ...out.data });
  }
  // Coordinator-side pipelined GENERATE: run the whole greedy loop HERE (one client request), streaming one
  // NDJSON line per token. On the target LAN only the fast coordinator↔fleet hops run per token; over a slow
  // tunnel the client crosses it ONCE (steady bytes also stop an idle-proxy from 502-ing a long request).
  // This is the "possible, not fast" inference story: N tokens = 1 streamed request, not N slow round-trips.
  if (req.method === 'POST' && url.pathname === '/model/shard_generate') {
    const body = await req.json().catch(() => ({})) as { id?: string; input_ids?: number[]; max_new_tokens?: number };
    const sid = body.id ?? [...shardPlans.keys()][0];
    if (!sid || !shardPlans.has(sid)) return json({ error: 'no sharded model — POST /model/shard first' }, 409);
    if (shardPlans.get(sid)!.stages.length === 0) return json({ error: `shard ${sid} is still loading`, id: sid }, 409);
    if (!Array.isArray(body.input_ids) || body.input_ids.length === 0) return json({ error: 'need {input_ids:[...]}' }, 400);
    if (body.input_ids.length > 100_000 || !body.input_ids.every((x) => Number.isInteger(x) && x >= 0)) return json({ error: 'input_ids must be non-negative ints, ≤100000' }, 400);
    const n = Math.max(1, Math.min(Number(body.max_new_tokens ?? 32), 1024));
    const prompt = body.input_ids.slice();
    const seq = body.input_ids.slice();
    const enc = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        const send = (o: unknown) => { try { controller.enqueue(enc.encode(JSON.stringify(o) + '\n')); } catch { /* client gone */ } };
        const t0 = Date.now();
        try {
          for (let k = 0; k < n; k++) {
            const out = await shardPipe(sid, seq, false); // NB: no KV cache on the shard path → re-runs the growing seq each token (slow, but complete)
            if (!out.ok) { send({ error: out.error, at: k }); break; } // in-band terminal error (headers already sent → can't 502)
            const tok = Number(out.data.argmax); seq.push(tok);
            send({ token: tok, i: k, ms: Date.now() - t0 });
          }
          send({ done: true, tokens: seq.slice(prompt.length), n: seq.length - prompt.length, ms: Date.now() - t0 });
        } catch (e) { send({ error: e instanceof Error ? e.message : String(e) }); }
        finally { controller.close(); }
      },
    });
    return new Response(stream, { headers: { 'content-type': 'application/x-ndjson', 'cache-control': 'no-cache' } });
  }
  // Text-in/text-out chat over a SHARDED model — so all nodes contribute to a chat. The FIRST stage holds the
  // tokenizer: tokenize the prompt there, pipe the ids through every stage per token (shardPipe), decode there.
  if (req.method === 'POST' && url.pathname === '/model/shard_chat') {
    const body = await req.json().catch(() => ({})) as { id?: string; prompt?: string; max_new_tokens?: number };
    const sid = body.id ?? [...shardPlans.keys()][0];
    if (!sid || !shardPlans.has(sid) || shardPlans.get(sid)!.stages.length === 0) return json({ error: 'no ready sharded model — POST /model/shard first', id: sid }, 409);
    if (typeof body.prompt !== 'string' || !body.prompt.trim()) return json({ error: 'need {prompt}' }, 400);
    const plan = shardPlans.get(sid)!;
    const first = workers.get(plan.stages[0].worker);
    if (!first) { shardPlans.delete(sid); return json({ error: 'first-stage worker disconnected — re-shard', id: sid }, 503); }
    const tk = await modelRPC(first, 'shard_tok', { id: sid, prompt: body.prompt.slice(0, 8000) });
    if (!tk.ok) return json({ error: `tokenize failed (re-shard — the tokenizer streams to the first stage): ${tk.error}` }, 502);
    const seq = (tk.data!.input_ids as number[]).slice();
    const promptLen = seq.length, eos = Number(tk.data!.eos);
    const n = Math.max(1, Math.min(Number(body.max_new_tokens ?? 64), 512));
    const t0 = Date.now();
    for (let k = 0; k < n; k++) {
      const out = await shardPipe(sid, seq, false);
      if (!out.ok) { if (out.disconnected) shardPlans.delete(sid); return json({ error: out.error, id: sid }, out.disconnected ? 503 : 502); }
      const tok = Number(out.data.argmax); seq.push(tok);
      if (Number.isFinite(eos) && tok === eos) break;
    }
    const newTokens = seq.slice(promptLen);
    const dt = await modelRPC(first, 'shard_detok', { id: sid, tokens: newTokens });
    if (!dt.ok) return json({ error: `decode failed: ${dt.error}` }, 502);
    return json({ ok: true, id: sid, text: dt.data!.text, n: newTokens.length, ms: Date.now() - t0, workers: plan.stages.map((s) => s.worker) });
  }
  if (req.method === 'POST' && url.pathname === '/model/shard_unload') {
    const body = await req.json().catch(() => ({})) as { id?: string };
    const sid = body.id ?? [...shardPlans.keys()][0];
    if (!sid || !shardPlans.has(sid)) return json({ error: 'no such sharded model' }, 404);
    const plan = shardPlans.get(sid)!;
    shardPlans.delete(sid);
    for (const st of plan.stages) { const w = workers.get(st.worker); if (w) await modelRPC(w, 'shard_unload', { id: sid }); }
    log('info', `sharded model ${sid} unloaded (${plan.stages.length} stages)`);
    return json({ ok: true, id: sid, stages: plan.stages.length });
  }
  if (url.pathname === '/logs') return json(LOG.slice(-200).reverse());
  if (url.pathname === '/metrics') return new Response(prometheus(), { headers: { 'content-type': 'text/plain; version=0.0.4' } });
  if (req.method === 'POST' && url.pathname === '/submit') {
    const body = await req.json().catch(() => ({})) as { kernel?: string; size?: number; a?: string; b?: string; scalar?: number; M?: number; N?: number; K?: number; bRef?: string };
    const kernel = KERNELS.includes(body.kernel ?? '') ? body.kernel! : 'matmul';
    // Data mode: caller supplies real tensors (base64 Float32) → the pool computes on THEM and returns the output.
    let input: JobInput | undefined;
    if (typeof body.a === 'string') {
      const a = b64ToF32(body.a);
      const b = typeof body.b === 'string' ? b64ToF32(body.b) : undefined;
      const CAP = 16_000_000;
      if (a.length > CAP || (b && b.length > CAP)) return json({ error: 'input too large (>16M elements per buffer)' }, 413);
      input = { a, b, scalar: body.scalar, M: body.M, N: body.N, K: body.K, bRef: body.bRef };
    }
    const size = Math.max(16, Math.min(kernel === 'matmul' ? 2048 : 8_000_000, Number(body.size ?? (kernel === 'matmul' ? 512 : 1_000_000))));
    // Async mode (GPU-style submit-and-poll): return the job handle immediately; caller polls GET /jobs/:id.
    const isAsync = (body as { async?: boolean }).async === true || url.searchParams.get('async') === '1';
    if (activeFleet().length === 0) { const r = submit(kernel, size, input); return json({ ...r, poll: `/jobs/${r.id}`, note: 'queued — no active workers yet (none connected, or all paused/scheduled-off); will run when one is available' }, 202); }
    const rec = submit(kernel, size, input);
    if (isAsync) return json({ id: rec.id, status: rec.status, kernel: rec.kernel, poll: `/jobs/${rec.id}` }, 202);
    // sync: wait briefly for a result; clients must still check status (may still be queued/running)
    for (let i = 0; i < 600 && rec.status !== 'done' && rec.status !== 'failed'; i++) await new Promise((r) => setTimeout(r, 50));
    return json(rec, rec.status === 'failed' ? 503 : 200);
  }
  if (url.pathname === '/jobs') return json([...jobs.values()].slice(-50).reverse().map(({ output: _o, ...r }) => ({ ...r, hasOutput: !!_o })));
  if (url.pathname.startsWith('/jobs/')) { const r = jobs.get(url.pathname.slice(6)); return r ? json(r) : json({ error: 'not found' }, 404); }
  if (url.pathname === '/chat') return new Response(CHAT_HTML, { headers: { 'content-type': 'text/html' } });
  return new Response(dashboard(), { headers: { 'content-type': 'text/html' } });
}

function prometheus(): string {
  const g = virtualGpu();
  const totalOps = g.totalOps || 1; // share is a fraction of compute OPERATIONS (never tokens-vs-elements)
  const lines = [
    '# HELP moregpu_fleet Connected workers', '# TYPE moregpu_fleet gauge', `moregpu_fleet ${g.slots}`,
    '# TYPE moregpu_gpu_slots gauge', `moregpu_gpu_slots ${g.gpuSlots}`,
    '# TYPE moregpu_cpu_slots gauge', `moregpu_cpu_slots ${g.cpuSlots}`,
    '# TYPE moregpu_busy gauge', `moregpu_busy ${g.busy}`,
    '# TYPE moregpu_queue_depth gauge', `moregpu_queue_depth ${g.queueDepth}`,
    '# TYPE moregpu_avg_user_util gauge', `moregpu_avg_user_util ${g.avgUserUtil}`,
    '# TYPE moregpu_avg_pool_duty gauge', `moregpu_avg_pool_duty ${g.avgPoolDuty}`,
    '# TYPE moregpu_total_units counter', `moregpu_total_units ${g.totalUnits}`,
    '# TYPE moregpu_total_shards counter', `moregpu_total_shards ${g.totalShards}`,
    '# TYPE moregpu_total_ops counter', `moregpu_total_ops ${g.totalOps}`,
    '# HELP moregpu_tokens_served LLM tokens generated across the pool', '# TYPE moregpu_tokens_served counter', `moregpu_tokens_served ${g.totalTokens}`,
    '# TYPE moregpu_jobs_total counter', `moregpu_jobs_total ${M.jobsTotal}`, `moregpu_jobs_done ${M.jobsDone}`, `moregpu_jobs_failed ${M.jobsFailed}`,
    `moregpu_shards_done ${M.shardsDone}`, `moregpu_shards_failed ${M.shardsFailed}`, `moregpu_gpu_shards ${M.gpuShards}`, `moregpu_cpu_shards ${M.cpuShards}`,
    '# HELP moregpu_worker_units Kernel output-elements completed per worker (detail)', '# TYPE moregpu_worker_units counter',
  ];
  for (const w of workers.values()) {
    const lbl = `{worker="${w.id.replace(/"/g, '')}",backend="${w.backend}"}`;
    lines.push(`moregpu_worker_units${lbl} ${w.units}`);
    lines.push(`moregpu_worker_ops${lbl} ${w.ops}`);
    lines.push(`moregpu_worker_tokens${lbl} ${w.tokens}`);
    lines.push(`moregpu_worker_share${lbl} ${(w.ops / totalOps).toFixed(4)}`);
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

// Built-in worker: with --worker (or MOREGPU_SELF_WORKER=1) the admin's OWN machine joins its own pool
// as a compute slot — so a solo admin needs nothing else; submitting a job "just works" like a local GPU.
const SELF_WORKER = Deno.args.includes('--worker') || Deno.env.get('MOREGPU_SELF_WORKER') === '1';
if (SELF_WORKER) {
  try {
    const workerUrl = new URL('../worker/worker.ts', import.meta.url).href;
    const wsUrl = `ws://127.0.0.1:${PORT}/ws`;
    new Deno.Command(Deno.execPath(), {
      args: ['run', '--unstable-webgpu', '--allow-net', '--allow-env', '--allow-sys', workerUrl,
        '--server', wsUrl, '--token', cfg.joinToken, '--name', Deno.env.get('MOREGPU_NAME') ?? 'admin-slot'],
      stdout: 'inherit', stderr: 'inherit',
    }).spawn();
    log('info', 'built-in worker started — this machine is a pool slot (admin-slot). Submit a job; it just runs.');
  } catch (e) {
    log('warn', `could not start the built-in worker (the server needs --allow-run for --worker): ${e}`);
  }
}

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
button.mini{padding:4px 8px;font-size:11px;border-radius:7px;background:#26304a;color:#c7d2fe;margin:0 2px}
button.mini.danger{background:#3a1d24;color:#fca5a5}
.dutyin{width:50px;padding:4px 6px;font-size:11px}
.dutyslider{width:96px;vertical-align:middle;accent-color:#818cf8;cursor:pointer}.dutyval{display:inline-block;width:34px;font-size:11px;color:var(--mut);text-align:right}
.schedin{width:92px;padding:3px 6px;font-size:11px}
td.ctl{white-space:nowrap}
.logo{font-size:9px;line-height:1.05;margin:0 0 10px;font-family:ui-monospace,Menlo,monospace;overflow:auto;white-space:pre;background:linear-gradient(90deg,#5f5fff,#875fff,#af5fff,#d75fff,#ff5faf,#ff5f5f);-webkit-background-clip:text;background-clip:text;color:transparent}
pre{background:#0e1420;border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto;max-height:260px;font-size:12px;margin:0}
.lvl-error{color:var(--red)}.lvl-warn{color:var(--yel)}.lvl-info{color:#93c5fd}.lvl-debug{color:var(--mut)}
a{color:#a5b4fc}.sp{margin-top:16px}.k{display:inline-block;background:#1a2133;color:#c7d2fe;border-radius:7px;padding:3px 9px;margin:3px 4px 0 0;font-size:12px;font-family:ui-monospace,monospace}
</style>
<div class=wrap>
<pre class=logo>███╗   ███╗ ██████╗ ██████╗ ███████╗ ██████╗ ██████╗ ██╗   ██╗
████╗ ████║██╔═══██╗██╔══██╗██╔════╝██╔════╝ ██╔══██╗██║   ██║
██╔████╔██║██║   ██║██████╔╝█████╗  ██║  ███╗██████╔╝██║   ██║
██║╚██╔╝██║██║   ██║██╔══██╗██╔══╝  ██║   ██║██╔═══╝ ██║   ██║
██║ ╚═╝ ██║╚██████╔╝██║  ██║███████╗╚██████╔╝██║     ╚██████╔╝
╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝      ╚═════╝</pre>
<h1>MoreGPU · admin</h1><div class=sub>Your worker fleet, presented as one virtual GPU. No cap on how many machines can join.</div>
<div class=card style="margin-bottom:14px"><div class=row><b>Admin token</b><input id=tok type=password placeholder="paste from the server console wizard" style="flex:1;min-width:220px"><button class=ghost onclick=save()>Save</button><span id=authmsg class=mut></span></div></div>
<div class="grid">
  <div class="card gpu"><h3>Virtual GPU</h3><div class=big id=slots>—</div><div class=mut id=slotsub>slots</div>
    <div class=mut style="margin-top:10px">user load <span id=uu>–</span></div><div class=bar><i id=uubar style=width:0%></i></div>
    <div class=mut style="margin-top:6px">pool duty <span id=pd>–</span></div><div class=bar><i id=pdbar style="width:0%;background:var(--grn)"></i></div>
    <div class=mut style="margin-top:10px">throughput · <span id=ops>0</span> ops · <span id=toks>0</span> tokens · <span id=units>0</span> kernel-elts</div><div id=tpspark style="margin-top:4px"></div>
  </div>
  <div class=card><h3>Queue</h3><div class=big id=q>0</div><div class=mut>waiting jobs</div>
    <div class=sp><span class=mut>done</span> <b id=jd>0</b> · <span class=mut>failed</span> <b id=jf>0</b></div></div>
  <div class=card><h3>Sealing</h3><div class=big style=font-size:20px>AES-256-GCM</div><div class=mut>every work unit, on the wire</div>
    <div class=sp><a href=/metrics target=_blank>/metrics</a> <span class=mut>· wire to Grafana</span></div></div>
</div>
<div class="card sp"><h3>Run a job</h3><div class=row>
  <select id=kernel><option>matmul</option><option>vector_add</option><option>vector_mul</option><option>saxpy</option><option>relu</option><option>scale</option><option>softmax</option><option>layernorm</option></select>
  <label class=mut>size <input id=size value=512 style=width:110px></label><button onclick=submit()>Submit</button><span id=jobmsg class=mut></span></div></div>
<div class="card sp"><h3>Fleet — live contribution &amp; control</h3>
  <div class=row><input id=fsearch placeholder="filter by name / type / OS" style="flex:1;min-width:200px" oninput=renderFleet()><button class=ghost onclick=pauseAll(true)>Pause all</button><button class=ghost onclick=pauseAll(false)>Resume all</button><span id=fcount class=mut></span></div>
  <div style="overflow:auto"><table><thead><tr><th>worker</th><th>type</th><th title="share of compute OPERATIONS (kernel shards + serving calls + training rounds — one consistent unit; not tokens-vs-elements)">activity</th><th title="ops/interval: kernel shards + LLM serving calls + training rounds">trend</th><th>shards</th><th title="kernel output-elements (matmul/etc.)">units</th><th title="LLM tokens generated on this node">tokens</th><th>user load</th><th>pool duty</th><th>state</th><th>control</th></tr></thead><tbody id=fleet><tr><td class=mut colspan=11>connect a worker…</td></tr></tbody></table></div></div>
<div class="card sp"><h3>Network self-test <span class=mut>— what your fleet's link can actually do</span></h3>
  <div class=row><button onclick=netTest() id=netbtn class=g>Run self-test</button> <span id=netnote class=mut></span></div>
  <div id=nettable class=mut style="margin-top:8px">Latency gates single-request sharded decode (needs a LAN); bandwidth gates weight transfer + throughput. Run it to see which workloads THIS fleet supports.</div></div>
<div class="card sp"><h3>Per-kernel jobs</h3><div id=kernels class=mut>—</div></div>
<div class="card sp"><h3>Errors &amp; debug log</h3><pre id=logs>—</pre></div>
</div>
<script>
const K='moregpu_admin_token';document.getElementById('tok').value=localStorage.getItem(K)||'';
const H=()=>({'content-type':'application/json','authorization':'Bearer '+(localStorage.getItem(K)||'')});
function save(){localStorage.setItem(K,document.getElementById('tok').value.trim());refresh();}
function pct(x){return Math.round((x||0)*100)+'%';}
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];});}
function fmt(n){n=n||0;return n>=1e9?(n/1e9).toFixed(1)+'G':n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':''+n;}
// ---- fleet control (pause/resume/duty/remove) + high-worker-count rendering ----
let FLEET=[];
function ctl(id,body,silent){return fetch('/workers/'+encodeURIComponent(id)+'/control',{method:'POST',headers:H(),body:JSON.stringify(body)}).then(function(){if(!silent)refresh();}).catch(function(){});}
function pauseAll(p){Promise.all((FLEET||[]).map(function(x){return ctl(x.id,{action:p?'pause':'resume'},true);})).then(refresh);}
function netTest(){var b=document.getElementById('netbtn');b.disabled=true;b.textContent='probing…';document.getElementById('netnote').textContent='';
 fetch('/net',{headers:H()}).then(function(r){return r.json();}).then(function(d){
  b.disabled=false;b.textContent='Run self-test';
  if(d.error){document.getElementById('netnote').textContent='✗ '+d.error;return;}
  document.getElementById('netnote').textContent='median RTT '+(d.median_rtt_ms==null?'–':d.median_rtt_ms+' ms')+(d.median_up_mbps!=null?' · '+d.median_up_mbps+' Mbps up':'');
  var rows=(d.workers||[]).map(function(w){return '<tr><td>'+esc(w.id)+'</td><td>'+(w.rtt_ms==null?'–':w.rtt_ms+' ms')+'</td><td>'+(w.up_mbps==null?esc(w.up_note||'–'):w.up_mbps+' Mbps')+'</td></tr>';}).join('');
  var li=function(a){return (a||[]).map(function(x){return '<li>'+esc(x)+'</li>';}).join('');};
  document.getElementById('nettable').innerHTML=
    '<table><thead><tr><th>worker</th><th>RTT</th><th>up</th></tr></thead><tbody>'+rows+'</tbody></table>'+
    '<div style="margin-top:8px;color:var(--fg)">'+esc(d.note||'')+'</div>'+
    '<div style="margin-top:6px"><b>✓ works anywhere (latency-tolerant):</b><ul>'+li(d.latency_tolerant_anywhere)+'</ul>'+
    '<b>needs low RTT (single-request sharded decode):</b><ul>'+li(d.latency_bound_needs_low_rtt)+'</ul></div>';
 }).catch(function(e){b.disabled=false;b.textContent='Run self-test';document.getElementById('netnote').textContent='✗ '+e;});}
function renderFleet(){
 var el=document.getElementById('fleet');if(!el)return;
 var ae=document.activeElement;if(ae&&ae.classList&&(ae.classList.contains('dutyin')||ae.classList.contains('dutyslider')||ae.classList.contains('schedin')))return; // don't clobber a duty field/slider/schedule mid-edit
 var q=(document.getElementById('fsearch').value||'').toLowerCase();
 var list=(FLEET||[]).filter(function(x){return !q||((x.nick||'')+' '+x.id+' '+x.backend+' '+(x.os||'')+' '+(x.label||'')).toLowerCase().indexOf(q)>=0;});
 list.sort(function(a,b){return (b.share||0)-(a.share||0);});
 var shown=list.slice(0,200);
 document.getElementById('fcount').textContent=(FLEET||[]).length?('showing '+shown.length+' of '+FLEET.length+(q?' matched':'')+' workers'):'';
 el.innerHTML=shown.length?shown.map(function(x){var paused=x.paused;return '<tr>'+
   '<td>'+esc(x.nick||x.id)+(x.errors?' <span class=lvl-error>('+(x.errors|0)+' err)</span>':'')+(x.schedule&&x.schedule!=='always'?' <span class=mut>· '+esc(x.schedule)+'</span>':'')+'</td>'+
   '<td><span class="pill '+(x.backend==='gpu'?'gpuP':'cpuP')+'">'+esc(x.backend)+'</span></td>'+
   '<td><b>'+pct(x.share)+'</b></td>'+
   '<td>'+spark(x.trend,72,20,x.backend==='gpu'?'#34d399':'#fbbf24')+'</td>'+
   '<td>'+(x.shards|0)+'</td><td class=mut>'+fmt(x.units)+'</td><td class=mut>'+fmt(x.tokens||0)+'</td>'+
   '<td>'+pct(x.userUtil)+'</td><td>'+pct(x.poolDuty)+'</td>'+
   '<td>'+(paused?(x.pausedReason==='schedule'?'<span class=mut>scheduled-off</span>':(x.pausedReason==='errors'?'<span class=lvl-error>auto-paused</span>':'<span class=lvl-warn>paused</span>')):(x.busy?'working':(x.serving?'<span style="color:#34d399">serving</span>':'idle')))+'</td>'+
   '<td class=ctl>'+
     '<button class=mini data-act="'+(paused?'resume':'pause')+'" data-id="'+esc(x.id)+'" title="'+(paused?'resume':'pause')+'">'+(paused?'▶':'⏸')+'</button>'+
     '<input type=range class=dutyslider min=0.05 max=1 step=0.05 value='+(x.ceil!=null?x.ceil:(x.poolDuty||0.6))+' data-id="'+esc(x.id)+'" title="usage ceiling — drag to change live">'+
     '<span class=dutyval>'+Math.round((x.ceil!=null?x.ceil:(x.poolDuty||0.6))*100)+'%</span>'+
     '<input class=schedin value="'+esc(x.schedule||'always')+'" data-id="'+esc(x.id)+'" title="when this worker contributes — always · idle-only · a window like 22:00-07:00 (its off-hours). Applied on Enter/blur.">'+
     '<button class="mini danger" data-act=remove data-id="'+esc(x.id)+'">✕</button>'+
   '</td></tr>';}).join(''):'<tr><td class=mut colspan=10>connect a worker…</td></tr>';
}
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
  document.getElementById('ops').textContent=fmt(g.totalOps||0);
  document.getElementById('toks').textContent=fmt(g.totalTokens||0);
  document.getElementById('units').textContent=fmt(g.totalUnits);
  document.getElementById('tpspark').innerHTML=spark(g.poolTrend,190,30,'#6366f1');
  document.getElementById('q').textContent=g.queueDepth;
  const pk=g.perKernel||{};const ks=Object.keys(pk);
  document.getElementById('kernels').innerHTML=ks.length?ks.map(k=>'<span class=k>'+k+' · '+pk[k]+'</span>').join(' '):'no jobs yet';
  FLEET=await (await fetch('/workers',{headers:H()})).json();
  renderFleet();
  const j=await (await fetch('/jobs',{headers:H()})).json();
  document.getElementById('jd').textContent=j.filter(x=>x.status==='done').length;
  document.getElementById('jf').textContent=j.filter(x=>x.status==='failed').length;
  const L=await (await fetch('/logs',{headers:H()})).json();
  document.getElementById('logs').innerHTML=L.slice(0,80).map(e=>'<span class=lvl-'+esc(e.level)+'>'+new Date(e.ts).toLocaleTimeString()+' ['+esc(e.level)+'] '+esc(e.msg)+(e.ctx?' · '+esc(e.ctx):'')+'</span>').join('\\n');
 }catch(e){}
}
async function submit(){const m=document.getElementById('jobmsg');m.textContent='running…';
 const r=await fetch('/submit',{method:'POST',headers:H(),body:JSON.stringify({kernel:document.getElementById('kernel').value,size:+document.getElementById('size').value})});
 const j=await r.json();m.textContent=j.status==='done'?('done · '+(j.gflops?j.gflops.toFixed(1)+' GFLOP/s · ':'')+'verified='+j.verified):(j.note||j.error||j.status);refresh();}
document.getElementById('fleet').addEventListener('click',function(ev){
 var b=ev.target.closest&&ev.target.closest('button[data-act]');if(!b)return;
 var id=b.getAttribute('data-id'),act=b.getAttribute('data-act');
 if(act==='remove'){if(!confirm('Remove worker '+id+'?'))return;ctl(id,{action:'remove'});}
 else ctl(id,{action:act});
});
// usage slider: live label while dragging, apply the ceiling on release (no full refresh, so it stays smooth)
document.getElementById('fleet').addEventListener('input',function(ev){var s=ev.target;if(!s.classList||!s.classList.contains('dutyslider'))return;var v=s.parentNode.querySelector('.dutyval');if(v)v.textContent=Math.round(s.value*100)+'%';});
document.getElementById('fleet').addEventListener('change',function(ev){var s=ev.target;if(!s.classList||!s.classList.contains('dutyslider'))return;ctl(s.getAttribute('data-id'),{ceil:parseFloat(s.value)},true);});
// schedule: apply on blur/Enter (always · idle-only · HH:MM-HH:MM). Blank → always.
document.getElementById('fleet').addEventListener('change',function(ev){var s=ev.target;if(!s.classList||!s.classList.contains('schedin'))return;ctl(s.getAttribute('data-id'),{schedule:(s.value||'always').trim()});});
document.getElementById('fleet').addEventListener('keydown',function(ev){if(ev.key==='Enter'&&ev.target.classList&&ev.target.classList.contains('schedin'))ev.target.blur();});
refresh();setInterval(refresh,2500);
</script>`;

// ---- /chat : a minimal chatbot page to test a served model (text↔text via the worker's tokenizer) ----
const CHAT_HTML = `<!doctype html><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1"><title>MoreGPU · chat</title>
<style>
:root{--bg:#0b0f17;--card:#111827;--line:#1f2937;--fg:#e5e7eb;--mut:#8792a8;--acc:#818cf8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:20px}
.logo{font:9px/1.05 ui-monospace,Menlo,monospace;white-space:pre;background:linear-gradient(90deg,#5f5fff,#875fff,#af5fff,#d75fff,#ff5faf,#ff5f5f);-webkit-background-clip:text;background-clip:text;color:transparent;margin:0 0 6px}
h1{font-size:18px;margin:0 0 12px}a{color:#a5b4fc}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
input,button{font:inherit;background:#0e1420;color:var(--fg);border:1px solid var(--line);border-radius:9px;padding:8px 11px}
input[type=password],#model{min-width:150px}#prompt{flex:1;min-width:220px}
button{background:var(--acc);border-color:var(--acc);color:#0b1020;font-weight:600;cursor:pointer}
button.g{background:#0e1420;color:var(--fg);border-color:var(--line);font-weight:500}
.mut{color:var(--mut);font-size:13px}
.chat{min-height:260px;max-height:56vh;overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:10px;margin:10px 0}
.msg{padding:9px 12px;border-radius:12px;max-width:88%;white-space:pre-wrap;word-wrap:break-word}
.msg.user{align-self:flex-end;background:#312e81;border:1px solid #4338ca}
.msg.bot{align-self:flex-start;background:#0e1420;border:1px solid var(--line)}
</style>
<div class=wrap>
<pre class=logo>███╗   ███╗ ██████╗ ██████╗ ███████╗ ██████╗ ██████╗ ██╗   ██╗
██║ ╚═╝ ██║╚██████╔╝██║  ██║███████╗╚██████╔╝██║     ╚██████╔╝</pre>
<h1>MoreGPU · chat <span class=mut>— test a model served on your pool · <a href="/">admin panel</a></span></h1>
<div class=bar>
<input id=tok type=password placeholder="admin token">
<input id=model value="gpt2" placeholder="HF model (gpt2 · Qwen/Qwen2.5-0.5B …)">
<input id=worker placeholder="worker (optional, e.g. colab-cuda)" style="min-width:150px">
<label class=mut title="OFF (default, fast): the worker downloads the model itself — quick on a machine with good internet (e.g. a Colab GPU). ON: the coordinator streams the weights so the worker never touches the HF hub and keeps nothing on its disk — but it's SLOWER (sealed streaming, ~minutes on Colab)."><input id=pushfree type=checkbox> download-free</label>
<label class=mut title="Split the model across ALL torch workers (pipeline parallel) so EVERY node contributes to each token — for a model too big for one node. Slower per token (each token hops between nodes), always download-free. Ignores the worker field."><input id=shardmode type=checkbox> shard across fleet</label>
<button onclick=load()>Load</button>
<span id=status class=mut></span>
</div>
<div id=chat class=chat></div>
<div class=bar>
<input id=prompt placeholder="message…  (Enter to send)" onkeydown="if(event.key==='Enter')send()">
<button onclick=send() id=sendb disabled title="Load a model first — Send enables when it's ready">Send</button>
<label class=mut>max <input id=mnt type=number value=96 style="width:64px"></label>
<label class=mut><input id=samp type=checkbox> sample</label>
</div>
<p class=mut><b>Load</b> a model, wait for the status to say <b>✓ ready</b> (Send stays disabled until then), then chat. Leave <b>download-free</b> <b>off</b> for quick interactive chat on a Colab GPU (the worker downloads the model itself — fast). Turn it <b>on</b> only when you want the fleet to touch <em>nothing</em> on disk — the coordinator streams the sealed weights (no HF hub on the worker, staged in RAM), which is slower (~minutes on Colab). Pin a specific <b>worker</b> to place the model.</p>
</div>
<script>
var K='moregpu_admin_token',tokEl=document.getElementById('tok');tokEl.value=localStorage.getItem(K)||'';
tokEl.onchange=function(){localStorage.setItem(K,tokEl.value.trim());};
function H(){return {'content-type':'application/json','authorization':'Bearer '+(localStorage.getItem(K)||tokEl.value.trim())};}
var MID=null,SHARDED=false;
function add(role,text){var d=document.getElementById('chat'),el=document.createElement('div');el.className='msg '+role;el.textContent=text;d.appendChild(el);d.scrollTop=d.scrollHeight;return el;}
function enableSend(){var sb=document.getElementById('sendb');sb.disabled=false;sb.title='';}
function ready(m,r,s){MID=r.id||m;SHARDED=false;enableSend();s.textContent='✓ ready — '+MID+' on '+(r.worker||'?')+(r.device?' · '+r.device:'')+' · '+((r.n_params||0)).toLocaleString()+' params'+(r.mode==='download-free'?' · download-free ('+(r.staging||'?')+')':'')+' — type a message and Send';}
function readyShard(m,r,s){MID=m;SHARDED=true;enableSend();var st=(r.stages||[]).map(function(x){return x.worker+'['+x.start+'-'+x.end+')';}).join(' → ');s.textContent='✓ ready — '+m+' SHARDED across '+((r.stages||[]).length)+' nodes: '+st+' — every node contributes per token';}
function pollShard(m,s,t0){fetch('/model/shard_status?id='+encodeURIComponent(m),{headers:H()}).then(function(r){return r.json();}).then(function(r){
  if(r.status==='ready'){readyShard(m,r,s);return;}
  if(r.status==='error'){s.textContent='✗ '+r.error;return;}
  s.textContent='sharding '+m+' across the fleet… '+Math.round((Date.now()-t0)/1000)+'s ('+(r.stages_done||0)+'/'+(r.stages_total||'?')+' stages streamed)';setTimeout(function(){pollShard(m,s,t0);},2000);
 }).catch(function(){setTimeout(function(){pollShard(m,s,t0);},2500);});}
function poll(m,s,t0){fetch('/model/status?id='+encodeURIComponent(m),{headers:H()}).then(function(r){return r.json();}).then(function(r){
  if(r.status==='ready'){ready(m,r,s);return;}
  if(r.status==='error'){s.textContent='✗ '+r.error;return;}
  s.textContent='streaming '+m+'… '+Math.round((Date.now()-t0)/1000)+'s (weights → worker)';setTimeout(function(){poll(m,s,t0);},2000);
 }).catch(function(){setTimeout(function(){poll(m,s,t0);},2500);});}
function load(){localStorage.setItem(K,tokEl.value.trim());MID=null;var sb=document.getElementById('sendb');sb.disabled=true;sb.title='Loading… Send enables when the model is ready';var m=document.getElementById('model').value.trim(),wk=document.getElementById('worker').value.trim(),pf=document.getElementById('pushfree').checked,sh=document.getElementById('shardmode').checked,s=document.getElementById('status');
 if(sh){ // SHARD across all nodes — always download-free + async; every node contributes per token
  s.textContent='sharding '+m+' across the fleet… (Send disabled until ready)';
  fetch('/model/shard',{method:'POST',headers:H(),body:JSON.stringify({model:m,id:m,push:true,async:true})}).then(function(r){return r.json();}).then(function(r){
   if(r.error){if(/already loaded/i.test(r.error)){pollShard(m,s,Date.now());return;}s.textContent='✗ '+r.error;return;} // attach to an in-flight/ready shard instead of erroring
   if(r.status==='loading'){pollShard(m,s,Date.now());return;}
   readyShard(m,r,s);}).catch(function(e){s.textContent='✗ '+e;});
  return;
 }
 s.textContent=(pf?'streaming ':'loading ')+m+'… (Send disabled until ready)';
 // async for download-free (a big push outlives a public tunnel's request timeout) — return fast, then poll status
 var b={model:m,id:m,push:pf};if(wk)b.worker=wk;if(pf)b.async=true;
 fetch('/model/load',{method:'POST',headers:H(),body:JSON.stringify(b)}).then(function(r){return r.json();}).then(function(r){
  if(r.error){s.textContent='✗ '+r.error;return;}
  if(r.status==='loading'){poll(m,s,Date.now());return;}
  ready(m,r,s);}).catch(function(e){s.textContent='✗ '+e;});}
function send(){var p=document.getElementById('prompt'),text=p.value.trim();if(!text)return;if(!MID){alert('Load a model first (top bar).');return;}
 add('user',text);p.value='';var rep=add('bot','…'),sb=document.getElementById('sendb');sb.disabled=true;
 var mnt=(+document.getElementById('mnt').value)||96;
 var url=SHARDED?'/model/shard_chat':'/model/chat';
 var body=SHARDED?{id:MID,prompt:text,max_new_tokens:mnt}:{id:MID,prompt:text,max_new_tokens:mnt,do_sample:document.getElementById('samp').checked};
 fetch(url,{method:'POST',headers:H(),body:JSON.stringify(body)}).then(function(r){return r.json();}).then(function(r){
  rep.textContent=r.error?('✗ '+r.error):((r.text||'(empty)')+(r.ms?'   ·  '+(r.ms/1000).toFixed(1)+'s '+(SHARDED?('across '+((r.workers||[]).length)+' nodes'):('on '+r.worker)):''));sb.disabled=false;}).catch(function(e){rep.textContent='✗ '+e;sb.disabled=false;});}
// deep link: /chat?model=NAME&shard=1 → prefill the model, tick "shard across fleet", and auto-Load once a token is present.
// (token is never taken from the URL — it stays in localStorage; the model auto-shards across every node into one pipe.)
(function(){var q=new URLSearchParams(location.search),m=q.get('model'),sh=q.get('shard');
 if(m)document.getElementById('model').value=m;
 if(sh&&sh!=='0')document.getElementById('shardmode').checked=true;
 if(!(m||sh))return;
 if((localStorage.getItem(K)||'').trim()){load();return;}
 document.getElementById('status').textContent='enter your admin token above — this '+(sh&&sh!=='0'?'sharded ':'')+'model then loads automatically';
 tokEl.addEventListener('change',function once(){if((tokEl.value||'').trim()){tokEl.removeEventListener('change',once);load();}});
})();
</script>`;

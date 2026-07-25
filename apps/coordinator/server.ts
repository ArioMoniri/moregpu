#!/usr/bin/env -S deno run --allow-net --allow-env --allow-read --allow-write
/**
 * MoreGPU coordinator / admin server — self-contained, multi-tenant-isolated, token-secured.
 *
 * First run is an automated WIZARD: it generates this pool's tokens + tenant key, writes a config
 * file, and prints the exact one-liner your workers should run. Every admin who runs this gets their
 * OWN isolated pool (own join token, own admin token, own encryption key) — nobody shares another's.
 *
 * Workers connect OUTBOUND over WebSocket and must present the join token. The admin submits jobs over
 * HTTP with the admin token. Each work unit is AES-GCM sealed on the wire. Optional native TLS (wss).
 *
 *   deno run --allow-net --allow-env --allow-read --allow-write \
 *     https://raw.githubusercontent.com/ArioMoniri/moregpu/main/apps/coordinator/server.ts
 *
 * Env: PORT (8787) · MOREGPU_CONFIG (./.moregpu-server.json) · MOREGPU_HOST (advertised host)
 *      MOREGPU_TLS_CERT + MOREGPU_TLS_KEY (paths → serves https/wss) · MOREGPU_DUTY (worker duty hint)
 */
const PORT = Number(Deno.env.get('PORT') ?? 8787);
const CONFIG_PATH = Deno.env.get('MOREGPU_CONFIG') ?? './.moregpu-server.json';
const ADVERTISE_HOST = Deno.env.get('MOREGPU_HOST') ?? 'localhost';
const DUTY_HINT = Number(Deno.env.get('MOREGPU_DUTY') ?? 0.6);
const CERT_PATH = Deno.env.get('MOREGPU_TLS_CERT');
const KEY_PATH = Deno.env.get('MOREGPU_TLS_KEY');

// ---------- base64 + sealing ----------
function b64e(u8: Uint8Array): string { let s = ''; const C = 0x8000; for (let i = 0; i < u8.length; i += C) s += String.fromCharCode(...u8.subarray(i, i + C)); return btoa(s); }
function b64d(s: string): Uint8Array { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
function tokenB64url(n = 24): string { return b64e(crypto.getRandomValues(new Uint8Array(n))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''); }
async function importKey(raw: Uint8Array) { return crypto.subtle.importKey('raw', raw as BufferSource, 'AES-GCM', false, ['encrypt', 'decrypt']); }
async function seal(key: Uint8Array, plain: Uint8Array) { const iv = crypto.getRandomValues(new Uint8Array(12)); const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv: iv as BufferSource }, await importKey(key), plain as BufferSource); return { iv: b64e(iv), ct: b64e(new Uint8Array(ct)) }; }
async function unseal(key: Uint8Array, blob: { iv: string; ct: string }) { return new Uint8Array(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: b64d(blob.iv) as BufferSource }, await importKey(key), b64d(blob.ct) as BufferSource)); }
const f32ToB64 = (a: Float32Array) => b64e(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
const b64ToF32 = (s: string) => new Float32Array(b64d(s).buffer);
const constEq = (a: string, b: string) => { if (a.length !== b.length) return false; let d = 0; for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i); return d === 0; };

// ---------- config / wizard ----------
interface Config { adminToken: string; joinToken: string; tenantKeyB64: string; created: string; }
async function loadOrInitConfig(): Promise<{ cfg: Config; fresh: boolean }> {
  try {
    return { cfg: JSON.parse(await Deno.readTextFile(CONFIG_PATH)), fresh: false };
  } catch {
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

function wizardBanner() {
  const wsUrl = `${scheme}://${ADVERTISE_HOST}:${PORT}/ws`;
  console.log(`
┌──────────────────────────────────────────────────────────────────────────┐
│  MoreGPU pool  ${fresh ? '· NEW POOL CREATED' : '· existing config loaded'}
├──────────────────────────────────────────────────────────────────────────┤
│  Admin dashboard : ${httpScheme}://${ADVERTISE_HOST}:${PORT}
│  Admin token     : ${cfg.adminToken}
│  Worker join tok : ${cfg.joinToken}
│  Config file     : ${CONFIG_PATH}   (keep it safe — it is this pool's identity)
├──────────────────────────────────────────────────────────────────────────┤
│  Add a machine to THIS pool (Linux/macOS):
│    curl -fsSL ${RAW}/scripts/install.sh \\
│      | MOREGPU_SERVER=${wsUrl} MOREGPU_TOKEN=${cfg.joinToken} sh
│
│  Windows (PowerShell):
│    $env:MOREGPU_SERVER="${wsUrl}"; $env:MOREGPU_TOKEN="${cfg.joinToken}"
│    irm ${RAW}/scripts/install.ps1 | iex
│
│  Install as a reboot-surviving service: add  MOREGPU_SERVICE=1  to either.
└──────────────────────────────────────────────────────────────────────────┘
`);
}

// ---------- worker registry ----------
interface Worker { id: string; backend: string; label: string; os: string; ws: WebSocket; }
const workers = new Map<string, Worker>();
const pending = new Map<string, (r: { ok: boolean; sealedOut?: { iv: string; ct: string }; error?: string; backend?: string; ms?: number }) => void>();

function wireWorker(ws: WebSocket) {
  let id = '';
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data as string);
    if (m.t === 'register') {
      if (!constEq(String(m.joinToken ?? ''), cfg.joinToken)) { ws.send(JSON.stringify({ t: 'denied', reason: 'bad join token' })); ws.close(); return; }
      id = m.node.id;
      workers.set(id, { id, backend: m.node.backend, label: m.node.label, os: m.node.os, ws });
      ws.send(JSON.stringify({ t: 'welcome', tenantKeyB64: b64e(TENANT_KEY), duty: DUTY_HINT }));
      console.log(`[coord] + worker ${id} (${m.node.label}, ${m.node.os}) · fleet=${workers.size}`);
    } else if (m.t === 'result') {
      pending.get(m.shardId)?.({ ok: m.ok, sealedOut: m.sealedOut, error: m.error, backend: m.backend, ms: m.ms });
    }
  };
  ws.onclose = () => { if (id) { workers.delete(id); console.log(`[coord] - worker ${id} · fleet=${workers.size}`); } };
}

// ---------- jobs ----------
interface JobRec { id: string; status: 'running' | 'done' | 'failed'; kernel: string; N: number; ms?: number; gflops?: number; verified?: boolean; shards?: { worker: string; backend: string; rows: number; ms: number }[]; error?: string; }
const jobs = new Map<string, JobRec>();
let jobSeq = 0, shardSeq = 0;

function cpuMatmul(a: Float32Array, b: Float32Array, M: number, N: number, K: number): Float32Array {
  const o = new Float32Array(M * N);
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < K; k++) s += a[i * K + k] * b[k * N + j]; o[i * N + j] = s; }
  return o;
}

async function runMatmulJob(N: number): Promise<JobRec> {
  const id = `job-${++jobSeq}`;
  const rec: JobRec = { id, status: 'running', kernel: 'matmul', N };
  jobs.set(id, rec);
  const fleet = [...workers.values()];
  if (fleet.length === 0) { rec.status = 'failed'; rec.error = 'no workers connected'; return rec; }

  const A = new Float32Array(N * N).map(() => Math.random());
  const B = new Float32Array(N * N).map(() => Math.random());
  const rowsPer = Math.ceil(N / fleet.length);

  const t0 = performance.now();
  const shardResults = await Promise.all(fleet.map(async (w, i) => {
    const r0 = i * rowsPer, r1 = Math.min(N, r0 + rowsPer), rows = r1 - r0;
    if (rows <= 0) return null;
    const shardId = `s-${++shardSeq}`;
    const Ablk = A.slice(r0 * N, r1 * N);
    const sealedIn = await seal(TENANT_KEY, new TextEncoder().encode(JSON.stringify({ kernel: 'matmul', a: f32ToB64(Ablk), b: f32ToB64(B), rows, N, K: N })));
    const done = new Promise<{ ok: boolean; sealedOut?: { iv: string; ct: string }; error?: string; backend?: string; ms?: number }>((res) => pending.set(shardId, res));
    w.ws.send(JSON.stringify({ t: 'assign', shardId, jobId: id, sealedIn }));
    const r = await done; pending.delete(shardId);
    if (!r.ok || !r.sealedOut) throw new Error(`shard on ${w.id} failed: ${r.error}`);
    const outObj = JSON.parse(new TextDecoder().decode(await unseal(TENANT_KEY, r.sealedOut)));
    return { r0, rows, out: b64ToF32(outObj.out), worker: w.id, backend: r.backend ?? w.label, ms: r.ms ?? 0 };
  })).catch((e) => { rec.status = 'failed'; rec.error = String(e); return null; });

  if (!shardResults || rec.status === 'failed') return rec;
  const C = new Float32Array(N * N);
  for (const s of shardResults) if (s) C.set(s.out, s.r0 * N);
  const wall = performance.now() - t0;

  let verified: boolean | undefined;
  if (N <= 640) { let md = 0; const ref = cpuMatmul(A, B, N, N, N); for (let i = 0; i < C.length; i++) md = Math.max(md, Math.abs(C[i] - ref[i])); verified = md < 1e-2; }

  rec.status = 'done';
  rec.ms = wall;
  rec.gflops = (2 * N * N * N) / (wall / 1000) / 1e9;
  rec.verified = verified;
  rec.shards = shardResults.filter(Boolean).map((s) => ({ worker: s!.worker, backend: s!.backend, rows: s!.rows, ms: s!.ms }));
  console.log(`[coord] ${id}: ${N}x${N} across ${rec.shards.length} workers · ${wall.toFixed(0)}ms · ${rec.gflops.toFixed(1)} GFLOP/s · verified=${verified}`);
  return rec;
}

// ---------- HTTP + WS ----------
const json = (o: unknown, status = 200) => new Response(JSON.stringify(o, null, 2), { status, headers: { 'content-type': 'application/json' } });
const authOk = (req: Request) => constEq((req.headers.get('authorization') ?? '').replace(/^Bearer\s+/i, '') || (req.headers.get('x-admin-token') ?? ''), cfg.adminToken);

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  if (url.pathname === '/ws') { const { socket, response } = Deno.upgradeWebSocket(req); wireWorker(socket); return response; }
  if (url.pathname === '/health') return json({ ok: true, fleet: workers.size });
  if (url.pathname === '/workers') { if (!authOk(req)) return json({ error: 'unauthorized' }, 401); return json([...workers.values()].map((w) => ({ id: w.id, backend: w.backend, label: w.label, os: w.os }))); }
  if (req.method === 'POST' && url.pathname === '/submit') {
    if (!authOk(req)) return json({ error: 'unauthorized — send Authorization: Bearer <admin token>' }, 401);
    const body = await req.json().catch(() => ({}));
    const rec = await runMatmulJob(Math.max(16, Math.min(2048, Number(body.size ?? 512))));
    return json(rec, rec.status === 'failed' ? 503 : 200);
  }
  if (url.pathname.startsWith('/jobs/')) { if (!authOk(req)) return json({ error: 'unauthorized' }, 401); const r = jobs.get(url.pathname.slice(6)); return r ? json(r) : json({ error: 'not found' }, 404); }
  return new Response(dashboard(), { headers: { 'content-type': 'text/html' } });
}

const serveOpts: Deno.ServeTcpOptions & { cert?: string; key?: string } = { port: PORT, onListen: () => wizardBanner() };
if (CERT_PATH && KEY_PATH) { serveOpts.cert = await Deno.readTextFile(CERT_PATH); serveOpts.key = await Deno.readTextFile(KEY_PATH); }
Deno.serve(serveOpts, handler);

function dashboard(): string {
  return `<!doctype html><meta charset=utf8><title>MoreGPU admin</title>
<style>body{font:14px ui-sans-serif,system-ui;margin:0;color:#0f172a;background:#f8fafc}
.wrap{max-width:820px;margin:0 auto;padding:36px 24px}h1{letter-spacing:-.5px;margin:0 0 4px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:18px 20px;margin:14px 0}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}input{border:1px solid #cbd5e1;border-radius:8px;padding:9px}
button{background:#4f46e5;color:#fff;border:0;border-radius:8px;padding:9px 16px;font-weight:600;cursor:pointer}
button.ghost{background:#eef2ff;color:#4338ca}#out{white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12.5px;background:#0f172a;color:#e2e8f0;border-radius:10px;padding:16px;margin-top:12px;min-height:40px}
.muted{color:#64748b}.pill{display:inline-block;background:#ecfdf5;color:#047857;border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600}</style>
<div class=wrap>
<h1>MoreGPU · admin</h1><p class=muted>Pooled GPU/CPU compute across the machines you've joined to this pool.</p>
<div class=card><div class=row><b>Admin token</b><input id=tok type=password placeholder="paste the admin token from your server console" style=flex:1>
<button class=ghost onclick=save()>Save</button></div><p class=muted style=margin:8px_0_0>Stored only in this browser. The server printed it in the setup wizard.</p></div>
<div class=card><div class=row><span class=pill id=fleetpill>fleet…</span><span id=fleet class=muted></span></div></div>
<div class=card><div class=row><b>Run a job</b><label class=muted>matmul N <input id=n value=512 style=width:90px></label>
<button onclick=submit()>Submit</button></div><div id=out>—</div></div>
</div>
<script>
const tokKey='moregpu_admin_token';
document.getElementById('tok').value=localStorage.getItem(tokKey)||'';
function hdr(){return {'content-type':'application/json','authorization':'Bearer '+(localStorage.getItem(tokKey)||'')};}
function save(){localStorage.setItem(tokKey,document.getElementById('tok').value.trim());refresh();}
async function refresh(){try{const r=await fetch('/workers',{headers:hdr()});if(r.status===401){document.getElementById('fleetpill').textContent='enter admin token';document.getElementById('fleet').textContent='';return;}
const w=await r.json();document.getElementById('fleetpill').textContent=w.length+' worker'+(w.length===1?'':'s');
document.getElementById('fleet').textContent=w.map(x=>x.id+' ('+x.backend+', '+x.os+')').join('  ·  ')||'none yet — run the join one-liner on a PC';}catch(e){}}
async function submit(){const o=document.getElementById('out');o.textContent='running…';
const r=await fetch('/submit',{method:'POST',headers:hdr(),body:JSON.stringify({size:+document.getElementById('n').value})});
o.textContent=JSON.stringify(await r.json(),null,2);refresh();}
refresh();setInterval(refresh,3000);
</script>`;
}

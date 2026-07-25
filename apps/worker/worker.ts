#!/usr/bin/env -S deno run --unstable-webgpu --allow-net --allow-env --allow-sys
/**
 * MoreGPU worker agent — a single self-contained file. Runs on any OS with Deno.
 *
 *  • Uses the real GPU (WebGPU → Metal/Vulkan/D3D12) when present.
 *  • When there is NO GPU, it contributes to the same pool with its CPU — running the identical
 *    kernels in JS/WASM. CPU machines add compute too.
 *  • Stays polite: an adaptive DUTY-CYCLE throttle interleaves compute with sleeps, so the interactive
 *    user keeps their machine responsive and average power draw (electricity) stays down.
 *  • Connects OUTBOUND over WebSocket (NAT/firewall friendly) and must present the pool's JOIN TOKEN.
 *  • Work units arrive AES-GCM sealed; the worker decrypts with the tenant key, computes, re-seals.
 *
 * One-liner (Deno installed):
 *   deno run --unstable-webgpu --allow-net --allow-env --allow-sys <this-url> \
 *     --server wss://ADMIN:8787/ws --token <join-token>
 */

const args = new Map<string, string>();
for (let i = 0; i < Deno.args.length; i++) { const a = Deno.args[i]; if (a.startsWith('--')) args.set(a.slice(2), Deno.args[i + 1] ?? 'true'); }
const SERVER = args.get('server') ?? Deno.env.get('MOREGPU_SERVER') ?? 'ws://localhost:8787/ws';
const TOKEN = args.get('token') ?? Deno.env.get('MOREGPU_TOKEN') ?? '';
const NAME = args.get('name') ?? Deno.env.get('MOREGPU_NAME') ?? `${(() => { try { return Deno.hostname(); } catch { return 'worker'; } })()}-${crypto.randomUUID().slice(0, 6)}`;
// Duty cycle CEILING: the most of this machine the pool may ever use (fraction of time computing).
// The EFFECTIVE duty adapts DOWN from this ceiling in real time based on the machine's own load, so
// the moment the user works their PC harder, the pool's share shrinks and the user is not disturbed.
let CEIL = args.has('throttle') ? Math.max(0.05, Math.min(1, Number(args.get('throttle')))) : NaN;
const MIN_DUTY = 0.05;
// Keep TOTAL system utilization under this; the pool only ever uses the slack below it. Lower = more
// headroom reserved for the user. Configurable per machine.
const MAX_UTIL = Math.max(0.3, Math.min(0.98, Number(Deno.env.get('MOREGPU_MAX_UTIL') ?? args.get('max-util') ?? 0.85)));
const CORES = Math.max(1, navigator.hardwareConcurrency || 4);
let emaUtil = 0; // smoothed system utilization (excluding transient spikes)
let lastDuty = MIN_DUTY;

/** Sample the machine's own load and return the duty we're allowed right now (adaptive per-user). */
function effectiveDuty(): number {
  const ceil = Number.isNaN(CEIL) ? 0.6 : CEIL;
  let load1 = 0;
  try { load1 = Deno.loadavg()[0]; } catch { load1 = 0; }
  if (load1 <= 0) return ceil; // no load signal (e.g. Windows) → fall back to the static ceiling
  const util = Math.min(1, load1 / CORES);
  emaUtil = emaUtil === 0 ? util : emaUtil + 0.4 * (util - emaUtil);
  // Slack below the utilization cap that the pool is permitted to consume.
  const slack = Math.max(0, (MAX_UTIL - emaUtil) / MAX_UTIL);
  lastDuty = Math.max(MIN_DUTY, Math.min(ceil, MIN_DUTY + (ceil - MIN_DUTY) * slack));
  return lastDuty;
}

// ---------- WGSL kernels (identical to @moregpu/gpu) ----------
const WGSL = {
  vector_add: `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= arrayLength(&a)) { return; } o[i] = a[i] + b[i]; }`,
  matmul: `@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4<u32>;
@compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let M=d.x; let N=d.y; let K=d.z; let r=g.y; let c=g.x;
  if (r>=M || c>=N) { return; }
  var acc=0.0; for (var k=0u;k<K;k=k+1u){ acc=acc+A[r*K+k]*B[k*N+c]; } C[r*N+c]=acc; }`,
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, Math.max(0, ms)));

// ---------- backends ----------
interface Backend { kind: 'gpu' | 'cpu'; label: string; matmul(a: Float32Array, b: Float32Array, M: number, N: number, K: number): Promise<Float32Array>; vectorAdd(a: Float32Array, b: Float32Array): Promise<Float32Array>; }

async function makeGpuBackend(): Promise<Backend | null> {
  const gpu = (navigator as { gpu?: GPU }).gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  async function run(code: string, storage: Float32Array[], uniform: Uint32Array | null, outLen: number, dispatch: [number, number, number]) {
    const inBufs = storage.map((arr) => { const b = device.createBuffer({ size: Math.max(4, arr.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, arr as BufferSource); return b; });
    const outBytes = Math.max(4, outLen * 4);
    const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const entries: GPUBindGroupEntry[] = inBufs.map((buffer, i) => ({ binding: i, resource: { buffer } }));
    entries.push({ binding: storage.length, resource: { buffer: outBuf } });
    let uBuf: GPUBuffer | null = null;
    if (uniform) { uBuf = device.createBuffer({ size: uniform.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(uBuf, 0, uniform as BufferSource); entries.push({ binding: storage.length + 1, resource: { buffer: uBuf } }); }
    const mod = device.createShaderModule({ code });
    const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(dispatch[0], dispatch[1], dispatch[2]); pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes); device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readBuf.getMappedRange().slice(0, outLen * 4)); readBuf.unmap();
    [...inBufs, outBuf, readBuf].forEach((b) => b.destroy()); uBuf?.destroy();
    return out;
  }
  const info = adapter.info ?? {};
  return { kind: 'gpu', label: `gpu:${info.vendor || 'webgpu'}/${info.architecture || 'native'}`,
    vectorAdd: (a, b) => run(WGSL.vector_add, [a, b], null, a.length, [Math.ceil(a.length / 64), 1, 1]),
    matmul: (a, b, M, N, K) => run(WGSL.matmul, [a, b], new Uint32Array([M, N, K, 0]), M * N, [Math.ceil(N / 16), Math.ceil(M / 16), 1]) };
}

/** CPU backend: computes in small row-chunks and sleeps between them to honor the duty cycle,
 *  so the machine stays responsive for its user and average power draw stays low. */
function makeCpuBackend(): Backend {
  return { kind: 'cpu', label: `cpu:${Deno.build.arch}`,
    async vectorAdd(a, b) { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] + b[i]; return o; },
    async matmul(a, b, M, N, K) {
      const o = new Float32Array(M * N);
      const chunk = Math.max(1, Math.min(M, Math.ceil(32768 / (K + 1)))); // ~constant work per chunk
      for (let i0 = 0; i0 < M; i0 += chunk) {
        const t = performance.now();
        const iEnd = Math.min(M, i0 + chunk);
        for (let i = i0; i < iEnd; i++) { const ai = i * K, oi = i * N; for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < K; k++) s += a[ai + k] * b[k * N + j]; o[oi + j] = s; } }
        const busy = performance.now() - t;
        const d = effectiveDuty(); if (d < 1) await sleep(busy * (1 / d - 1)); // adapts to current user load
      }
      return o;
    } };
}

// ---------- sealing ----------
function b64e(u8: Uint8Array): string { let s = ''; const C = 0x8000; for (let i = 0; i < u8.length; i += C) s += String.fromCharCode(...u8.subarray(i, i + C)); return btoa(s); }
function b64d(s: string): Uint8Array { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
async function importKey(raw: Uint8Array) { return crypto.subtle.importKey('raw', raw as BufferSource, 'AES-GCM', false, ['encrypt', 'decrypt']); }
async function seal(key: Uint8Array, plain: Uint8Array) { const iv = crypto.getRandomValues(new Uint8Array(12)); const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv: iv as BufferSource }, await importKey(key), plain as BufferSource); return { iv: b64e(iv), ct: b64e(new Uint8Array(ct)) }; }
async function unseal(key: Uint8Array, blob: { iv: string; ct: string }) { return new Uint8Array(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: b64d(blob.iv) as BufferSource }, await importKey(key), b64d(blob.ct) as BufferSource)); }
const f32ToB64 = (a: Float32Array) => b64e(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
const b64ToF32 = (s: string) => new Float32Array(b64d(s).buffer);

// ---------- elementwise tensor kernels (index-sharded; memory-bound so run on CPU, duty-throttled) ----------
const ELEMENTWISE = new Set(['vector_add', 'vector_mul', 'saxpy', 'relu', 'scale']);
async function runElementwise(kernel: string, a: Float32Array, b: Float32Array | null, scalar: number): Promise<Float32Array> {
  const n = a.length, o = new Float32Array(n);
  const chunk = Math.max(4096, Math.ceil(n / 32));
  for (let i0 = 0; i0 < n; i0 += chunk) {
    const t = performance.now(); const e = Math.min(n, i0 + chunk);
    for (let i = i0; i < e; i++) {
      switch (kernel) {
        case 'vector_add': o[i] = a[i] + (b as Float32Array)[i]; break;
        case 'vector_mul': o[i] = a[i] * (b as Float32Array)[i]; break;
        case 'saxpy': o[i] = scalar * a[i] + (b as Float32Array)[i]; break;
        case 'relu': o[i] = a[i] > 0 ? a[i] : 0; break;
        case 'scale': o[i] = a[i] * scalar; break;
      }
    }
    const d = effectiveDuty(); if (d < 1) await sleep((performance.now() - t) * (1 / d - 1));
  }
  return o;
}

// ---------- row-wise tensor kernels (softmax/layernorm; sharded by whole rows) ----------
const ROWWISE = new Set(['softmax', 'layernorm']);
function runRowwise(kernel: string, a: Float32Array, cols: number): Float32Array {
  const rows = Math.floor(a.length / cols), o = new Float32Array(a.length);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    if (kernel === 'softmax') {
      let mx = -Infinity; for (let j = 0; j < cols; j++) mx = Math.max(mx, a[off + j]);
      let s = 0; for (let j = 0; j < cols; j++) { const e = Math.exp(a[off + j] - mx); o[off + j] = e; s += e; }
      for (let j = 0; j < cols; j++) o[off + j] /= s;
    } else { // layernorm
      let m = 0; for (let j = 0; j < cols; j++) m += a[off + j]; m /= cols;
      let v = 0; for (let j = 0; j < cols; j++) { const d = a[off + j] - m; v += d * d; } v /= cols;
      const inv = 1 / Math.sqrt(v + 1e-5);
      for (let j = 0; j < cols; j++) o[off + j] = (a[off + j] - m) * inv;
    }
  }
  return o;
}

// ---------- work loop ----------
const FORCE_CPU = args.has('cpu') || Deno.env.get('MOREGPU_FORCE_CPU') === '1';
const backend = (FORCE_CPU ? null : await makeGpuBackend().catch(() => null)) ?? makeCpuBackend();
console.log(`[worker] ${NAME} · backend=${backend.label} · server=${SERVER}`);
if (!TOKEN) console.log('[worker] warning: no join token set (--token / MOREGPU_TOKEN) — the server will reject me');
let tenantKey: Uint8Array | null = null;

function connect() {
  const ws = new WebSocket(SERVER);
  let hb: ReturnType<typeof setInterval> | undefined;
  ws.onopen = () => {
    ws.send(JSON.stringify({ t: 'register', joinToken: TOKEN, node: { id: NAME, backend: backend.kind, label: backend.label, os: Deno.build.os } }));
    // Heartbeat: report live load + the adaptive duty so the admin panel can show throttle in real time.
    hb = setInterval(() => {
      let load1 = 0; try { load1 = Deno.loadavg()[0]; } catch { /* */ }
      try { ws.send(JSON.stringify({ t: 'heartbeat', id: NAME, load1, cores: CORES, util: +(load1 / CORES).toFixed(3), duty: +effectiveDuty().toFixed(3) })); } catch { /* */ }
    }, 4000);
  };
  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data as string);
    if (msg.t === 'denied') { console.error(`[worker] rejected by server: ${msg.reason}. Check the join token.`); try { ws.close(); } catch { /* */ } Deno.exit(1); }
    if (msg.t === 'welcome') { tenantKey = b64d(msg.tenantKeyB64); if (Number.isNaN(CEIL)) CEIL = typeof msg.duty === 'number' ? msg.duty : 0.6; console.log(`[worker] joined pool · duty ceiling=${(CEIL * 100).toFixed(0)}% · adaptive (backs off as your machine gets busier, keeps total load < ${(MAX_UTIL * 100).toFixed(0)}%)`); return; }
    if (msg.t === 'assign') {
      const t0 = performance.now();
      try {
        if (!tenantKey) throw new Error('no tenant key');
        const req = JSON.parse(new TextDecoder().decode(await unseal(tenantKey, msg.sealedIn)));
        let out: Float32Array;
        if (req.kernel === 'matmul') out = await backend.matmul(b64ToF32(req.a), b64ToF32(req.b), req.rows, req.N, req.K);
        else if (ROWWISE.has(req.kernel)) out = runRowwise(req.kernel, b64ToF32(req.a), req.cols);
        else if (ELEMENTWISE.has(req.kernel)) out = await runElementwise(req.kernel, b64ToF32(req.a), req.b ? b64ToF32(req.b) : null, req.scalar ?? 1);
        else throw new Error(`unknown kernel ${req.kernel}`);
        const sealedOut = await seal(tenantKey, new TextEncoder().encode(JSON.stringify({ out: f32ToB64(out) })));
        const ms = performance.now() - t0;
        ws.send(JSON.stringify({ t: 'result', shardId: msg.shardId, jobId: msg.jobId, ok: true, sealedOut, ms, backend: backend.label }));
        console.log(`[worker] ${msg.shardId} done in ${ms.toFixed(1)}ms on ${backend.kind}`);
        const cool = effectiveDuty(); if (cool < 1) await sleep(ms * (1 / cool - 1)); // adaptive cool-down between shards
      } catch (e) { ws.send(JSON.stringify({ t: 'result', shardId: msg.shardId, jobId: msg.jobId, ok: false, error: String(e) })); }
    }
  };
  ws.onclose = () => { if (hb) clearInterval(hb); console.log('[worker] disconnected, retrying in 2s'); setTimeout(connect, 2000); };
  ws.onerror = () => { try { ws.close(); } catch { /* */ } };
}
connect();

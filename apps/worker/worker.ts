#!/usr/bin/env -S deno run --unstable-webgpu --allow-net --allow-env --allow-sys
/**
 * MoreGPU worker agent — a single self-contained file. Runs on any OS with Deno.
 *
 *  • On a GPU worker, ALL kernels run on the real GPU (WebGPU → Metal/Vulkan/D3D12): tiled matmul,
 *    elementwise/activations (relu/scale/gelu/add/mul/saxpy), and row-wise softmax/layernorm reductions.
 *    On a machine with no GPU, the identical kernels run on the CPU. CPU machines add compute too.
 *    (Elementwise/row-wise ops are memory-bound, so on the GPU they mainly relieve the worker's CPU
 *    rather than run dramatically faster — but every kernel now executes on-device.)
 *  • Recovers from GPU device loss by falling back to the CPU backend mid-flight.
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

// ---------- scheduling + remote control (the USER decides WHEN this machine is lent; the ADMIN can override) ----------
// MOREGPU_SCHEDULE / --schedule:  "always" (default) · "idle-only" (only when the machine is idle) ·
//   "HH:MM-HH:MM" active window in local time (may wrap past midnight, e.g. "22:00-07:00" = nights only).
// While outside the window / not idle / admin-paused, the worker takes NO new work (duty 0) and reports
// "paused" so the coordinator stops assigning to it. In-flight shards always finish (work is never dropped).
let SCHEDULE = (args.get('schedule') ?? Deno.env.get('MOREGPU_SCHEDULE') ?? 'always').trim().toLowerCase();
let adminPaused = false; // toggled by an admin 'control' frame from the coordinator
const IDLE_UTIL = Math.max(0.05, Math.min(0.9, Number(Deno.env.get('MOREGPU_IDLE_UTIL') ?? 0.25)));
function inWindow(now: Date): boolean {
  const m = SCHEDULE.match(/^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$/);
  if (!m) return true; // not a time window → always (unless 'idle-only', handled below)
  const cur = now.getHours() * 60 + now.getMinutes();
  const s = (+m[1]) * 60 + (+m[2]), e = (+m[3]) * 60 + (+m[4]);
  return s <= e ? (cur >= s && cur < e) : (cur >= s || cur < e); // wrap past midnight
}
function scheduleActive(): boolean {
  if (SCHEDULE === 'idle-only') return emaUtil > 0 ? emaUtil < IDLE_UTIL : true;
  if (SCHEDULE === 'always' || SCHEDULE === '') return true;
  return inWindow(new Date());
}
/** May the pool give this machine new work right now? */
function isActive(): boolean { return !adminPaused && scheduleActive(); }

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

// ---------- WGSL kernels ----------
// Workgroup-tiled dense fp32 GEMM: each 16×16 workgroup cooperatively stages a 16×16 tile of A and B
// into shared memory, so each A/B element is read from global memory once per tile instead of once per
// output — the standard tiling optimization. Boundary tiles are zero-padded via select(); no early
// return (all invocations must reach every workgroupBarrier). Still fp32, no tensor cores.
const WGSL = {
  matmul: `@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4<u32>;
var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let M=d.x; let N=d.y; let K=d.z;
  let row=g.y; let col=g.x; let lr=l.y; let lc=l.x;
  var acc=0.0;
  let tiles=(K+15u)/16u;
  for (var t=0u; t<tiles; t=t+1u) {
    let aCol=t*16u+lc; let bRow=t*16u+lr;
    As[lr*16u+lc]=select(0.0, A[row*K+aCol], row<M && aCol<K);
    Bs[lr*16u+lc]=select(0.0, B[bRow*N+col], bRow<K && col<N);
    workgroupBarrier();
    for (var k=0u;k<16u;k=k+1u){ acc=acc+As[lr*16u+k]*Bs[k*16u+lc]; }
    workgroupBarrier();
  }
  if (row<M && col<N){ C[row*N+col]=acc; }
}`,
  // fp16 tiled GEMM: f16 A/B storage (half the memory + bandwidth), f32 accumulate, f32 output.
  // `enable f16;` is a module directive and must be the first line — so this is a separate module.
  matmulF16: `enable f16;
@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4<u32>;
var<workgroup> As: array<f16, 256>;
var<workgroup> Bs: array<f16, 256>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let M=d.x; let N=d.y; let K=d.z;
  let row=g.y; let col=g.x; let lr=l.y; let lc=l.x;
  var acc=0.0;
  let tiles=(K+15u)/16u;
  for (var t=0u; t<tiles; t=t+1u) {
    let aCol=t*16u+lc; let bRow=t*16u+lr;
    As[lr*16u+lc]=select(f16(0.0), A[row*K+aCol], row<M && aCol<K);
    Bs[lr*16u+lc]=select(f16(0.0), B[bRow*N+col], bRow<K && col<N);
    workgroupBarrier();
    for (var k=0u;k<16u;k=k+1u){ acc=acc+f32(As[lr*16u+k])*f32(Bs[k*16u+lc]); }
    workgroupBarrier();
  }
  if (row<M && col<N){ C[row*N+col]=acc; }
}`,
  // Elementwise unary (op: 0=relu 1=scale 2=gelu), one thread per element. p.x=op, p.y=scalar.
  unary: `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i=g.x; if (i>=arrayLength(&a)) { return; }
  let x=a[i]; let op=p.x;
  if (op < 0.5) { o[i]=max(x,0.0); }
  else if (op < 1.5) { o[i]=x*p.y; }
  else { let c=0.7978845608028654; o[i]=0.5*x*(1.0+tanh(clamp(c*(x+0.044715*x*x*x), -30.0, 30.0))); }
// ^ clamp the tanh argument: it grows as x³, and some GPU tanh impls (Metal) overflow to NaN on huge
//   inputs; tanh saturates to ±1 by |arg|≈20, so clamping is exact and just prevents the NaN.
}`,
  // Elementwise binary (op: 0=add 1=mul 2=saxpy). p.x=op, p.y=scalar.
  binary: `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i=g.x; if (i>=arrayLength(&a)) { return; }
  let op=p.x;
  if (op < 0.5) { o[i]=a[i]+b[i]; }
  else if (op < 1.5) { o[i]=a[i]*b[i]; }
  else { o[i]=p.y*a[i]+b[i]; }
}`,
  // Softmax over each row (one workgroup per row): max-reduce → exp → sum-reduce → normalize. d.x=cols.
  softmax: `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> d: vec4<u32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let cols=d.x; let base=wg.x*cols; let tid=l.x;
  var m=-3.4e38;
  for (var j=tid; j<cols; j=j+256u) { m=max(m, a[base+j]); }
  red[tid]=m; workgroupBarrier();
  for (var s=128u; s>0u; s=s>>1u) { if (tid<s) { red[tid]=max(red[tid], red[tid+s]); } workgroupBarrier(); }
  let mx=red[0]; workgroupBarrier();
  var sum=0.0;
  for (var j=tid; j<cols; j=j+256u) { sum=sum+exp(a[base+j]-mx); }
  red[tid]=sum; workgroupBarrier();
  for (var s=128u; s>0u; s=s>>1u) { if (tid<s) { red[tid]=red[tid]+red[tid+s]; } workgroupBarrier(); }
  let sm=red[0]; workgroupBarrier();
  for (var j=tid; j<cols; j=j+256u) { o[base+j]=exp(a[base+j]-mx)/sm; }
}`,
  // LayerNorm over each row (one workgroup per row): mean → variance → normalize. d.x=cols.
  layernorm: `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> d: vec4<u32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let cols=d.x; let base=wg.x*cols; let tid=l.x; let n=f32(cols);
  var s=0.0;
  for (var j=tid; j<cols; j=j+256u) { s=s+a[base+j]; }
  red[tid]=s; workgroupBarrier();
  for (var k=128u; k>0u; k=k>>1u) { if (tid<k) { red[tid]=red[tid]+red[tid+k]; } workgroupBarrier(); }
  let mean=red[0]/n; workgroupBarrier();
  var v=0.0;
  for (var j=tid; j<cols; j=j+256u) { let dd=a[base+j]-mean; v=v+dd*dd; }
  red[tid]=v; workgroupBarrier();
  for (var k=128u; k>0u; k=k>>1u) { if (tid<k) { red[tid]=red[tid]+red[tid+k]; } workgroupBarrier(); }
  let inv=1.0/sqrt(red[0]/n + 1e-5); workgroupBarrier();
  for (var j=tid; j<cols; j=j+256u) { o[base+j]=(a[base+j]-mean)*inv; }
}`,
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, Math.max(0, ms)));

// ---------- backends ----------
interface Backend {
  kind: 'gpu' | 'cpu'; label: string; hasF16?: boolean;
  matmul(a: Float32Array, b: Float32Array, M: number, N: number, K: number): Promise<Float32Array>;
  // fp16 matmul: f16 A/B (Uint16 bit patterns), f32 output. Present only on f16-capable GPU workers.
  matmulF16?(a: Uint16Array, b: Uint16Array, M: number, N: number, K: number): Promise<Float32Array>;
  // Optional GPU paths for the other kernels; when absent the worker uses the CPU reference implementations.
  elementwise?(kernel: string, a: Float32Array, b: Float32Array | null, scalar: number): Promise<Float32Array>;
  rowwise?(kernel: string, a: Float32Array, cols: number): Promise<Float32Array>;
  shard?: ShardRuntime; // present on a GPU backend: lets this worker HOLD a pipeline MIDDLE stage (WGSL transformer forward)
}

// ════════════ WEBGPU MODEL-SHARD RUNTIME ════════════════════════════════════════════════════════════════
// Lets a WebGPU worker hold a pipeline MIDDLE stage: parse the coordinator's streamed per-stage safetensors
// slice + config, then run a numerically-correct Llama/Qwen2-style decoder-block forward (hidden→hidden) in
// WGSL. Verified against a torch oracle to ~1e-6 (RMSNorm, GQA attention + RoPE + causal mask, SwiGLU, residuals).
interface ShardCfg { H: number; NH: number; NKV: number; HD: number; INT: number; eps: number; theta: number }
function f16ToF32(h: number): number {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
// safetensors bytes → { name: {data: Float32Array (dequantized F32/F16/BF16), shape} }
function parseSafetensors(buf: Uint8Array): Map<string, { data: Float32Array; shape: number[] }> {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + headerLen)));
  const dataStart = 8 + headerLen, out = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const m = meta as { dtype: string; shape: number[]; data_offsets: [number, number] };
    const raw = buf.subarray(dataStart + m.data_offsets[0], dataStart + m.data_offsets[1]);
    const n = m.shape.reduce((a, b) => a * b, 1); let data: Float32Array;
    if (m.dtype === 'F32') data = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    else if (m.dtype === 'F16') { data = new Float32Array(n); const u = new Uint16Array(raw.buffer, raw.byteOffset, n); for (let i = 0; i < n; i++) data[i] = f16ToF32(u[i]); }
    else if (m.dtype === 'BF16') { data = new Float32Array(n); const u = new Uint16Array(raw.buffer, raw.byteOffset, n); const f = new Uint32Array(data.buffer); for (let i = 0; i < n; i++) f[i] = u[i] << 16; }
    else throw new Error(`unsupported safetensors dtype ${m.dtype} for ${name}`);
    out.set(name, { data, shape: m.shape });
  }
  return out;
}
// HF rotary cos/sin from absolute positions (matches transformers to ~1e-7): inv_freq=theta^-(2i/HD), halves duplicated.
function computeCosSin(positions: number[], HD: number, theta: number): { cos: Float32Array; sin: Float32Array } {
  const S = positions.length, cos = new Float32Array(S * HD), sin = new Float32Array(S * HD);
  for (let s = 0; s < S; s++) for (let i = 0; i < HD / 2; i++) {
    const fr = positions[s] / Math.pow(theta, (2 * i) / HD);
    cos[s * HD + i] = Math.cos(fr); cos[s * HD + i + HD / 2] = Math.cos(fr);
    sin[s * HD + i] = Math.sin(fr); sin[s * HD + i + HD / 2] = Math.sin(fr);
  }
  return { cos, sin };
}
const SHARD_WGSL = {
  linear: `@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> w:array<f32>;@group(0) @binding(2) var<storage,read> bias:array<f32>;@group(0) @binding(3) var<storage,read_write> y:array<f32>;struct U{R:u32,N:u32,K:u32,hasBias:u32};@group(0) @binding(4) var<uniform> u:U;
@compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) g:vec3<u32>){let r=g.y;let n=g.x;if(r>=u.R||n>=u.N){return;}var a=0.0;for(var k=0u;k<u.K;k=k+1u){a=a+x[r*u.K+k]*w[n*u.K+k];}if(u.hasBias==1u){a=a+bias[n];}y[r*u.N+n]=a;}`,
  rmsnorm: `@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> w:array<f32>;@group(0) @binding(2) var<storage,read_write> y:array<f32>;struct U{H:u32,eps:f32};@group(0) @binding(3) var<uniform> u:U;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) wid:vec3<u32>){let r=wid.x;var s=0.0;for(var i=0u;i<u.H;i=i+1u){let v=x[r*u.H+i];s=s+v*v;}let inv=inverseSqrt(s/f32(u.H)+u.eps);for(var i=0u;i<u.H;i=i+1u){y[r*u.H+i]=x[r*u.H+i]*inv*w[i];}}`,
  rope: `@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> cosb:array<f32>;@group(0) @binding(2) var<storage,read> sinb:array<f32>;@group(0) @binding(3) var<storage,read_write> y:array<f32>;struct U{SEQ:u32,NHEADS:u32,HD:u32};@group(0) @binding(4) var<uniform> u:U;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let idx=g.x;let total=u.SEQ*u.NHEADS*u.HD;if(idx>=total){return;}let d=idx%u.HD;let rest=idx/u.HD;let h=rest%u.NHEADS;let s=rest/u.NHEADS;let half=u.HD/2u;let base=(s*u.NHEADS+h)*u.HD;let xv=x[base+d];var rot:f32;if(d<half){rot=-x[base+d+half];}else{rot=x[base+d-half];}y[idx]=xv*cosb[s*u.HD+d]+rot*sinb[s*u.HD+d];}`,
  attn: `@group(0) @binding(0) var<storage,read> q:array<f32>;@group(0) @binding(1) var<storage,read> k:array<f32>;@group(0) @binding(2) var<storage,read> v:array<f32>;@group(0) @binding(3) var<storage,read_write> ctx:array<f32>;struct U{SEQ:u32,NH:u32,NKV:u32,HD:u32};@group(0) @binding(4) var<uniform> u:U;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;let h=g.y;if(i>=u.SEQ||h>=u.NH){return;}let kv=h/(u.NH/u.NKV);let scale=1.0/sqrt(f32(u.HD));let qbase=(i*u.NH+h)*u.HD;var sc:array<f32,512>;var mx=-3.0e38;for(var j=0u;j<=i;j=j+1u){let kb=(j*u.NKV+kv)*u.HD;var dot=0.0;for(var d=0u;d<u.HD;d=d+1u){dot=dot+q[qbase+d]*k[kb+d];}let s2=dot*scale;sc[j]=s2;if(s2>mx){mx=s2;}}var den=0.0;for(var j=0u;j<=i;j=j+1u){let e=exp(sc[j]-mx);sc[j]=e;den=den+e;}let ob=i*(u.NH*u.HD)+h*u.HD;for(var d=0u;d<u.HD;d=d+1u){var acc=0.0;for(var j=0u;j<=i;j=j+1u){let vb=(j*u.NKV+kv)*u.HD;acc=acc+sc[j]*v[vb+d];}ctx[ob+d]=acc/den;}}`,
  swiglu: `@group(0) @binding(0) var<storage,read> gate:array<f32>;@group(0) @binding(1) var<storage,read> up:array<f32>;@group(0) @binding(2) var<storage,read_write> out:array<f32>;struct U{n:u32};@group(0) @binding(3) var<uniform> u:U;@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;if(i>=u.n){return;}let x=gate[i];out[i]=(x/(1.0+exp(-x)))*up[i];}`,
  add: `@group(0) @binding(0) var<storage,read> a:array<f32>;@group(0) @binding(1) var<storage,read> b:array<f32>;@group(0) @binding(2) var<storage,read_write> o:array<f32>;struct U{n:u32};@group(0) @binding(3) var<uniform> u:U;@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;if(i>=u.n){return;}o[i]=a[i]+b[i];}`,
};
class ShardRuntime {
  private cache = new Map<string, GPUComputePipeline>();
  constructor(private dev: GPUDevice) {}
  private pipe(code: string): GPUComputePipeline { let p = this.cache.get(code); if (!p) { p = this.dev.createComputePipeline({ layout: 'auto', compute: { module: this.dev.createShaderModule({ code }), entryPoint: 'main' } }); this.cache.set(code, p); } return p; }
  private async run(code: string, storage: Float32Array[], uniform: ArrayBufferView, outLen: number, dispatch: [number, number, number]): Promise<Float32Array> {
    const dev = this.dev;
    const inBufs = storage.map((arr) => { const b = dev.createBuffer({ size: Math.max(4, (arr.byteLength + 3) & ~3), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(b, 0, arr as BufferSource); return b; });
    const outBytes = Math.max(4, outLen * 4);
    const outBuf = dev.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = dev.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const uBuf = dev.createBuffer({ size: Math.max(16, uniform.byteLength), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(uBuf, 0, uniform as BufferSource);
    const entries: GPUBindGroupEntry[] = inBufs.map((buffer, i) => ({ binding: i, resource: { buffer } }));
    entries.push({ binding: storage.length, resource: { buffer: outBuf } }, { binding: storage.length + 1, resource: { buffer: uBuf } });
    const pipe = this.pipe(code); const bind = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const enc = dev.createCommandEncoder(); const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(dispatch[0], dispatch[1], dispatch[2]); pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes); dev.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readBuf.getMappedRange().slice(0, outLen * 4)); readBuf.unmap();
    [...inBufs, outBuf, readBuf, uBuf].forEach((b) => b.destroy());
    return out;
  }
  private u32(...v: number[]) { return new Uint32Array(v); }
  private linear(x: Float32Array, w: Float32Array, bias: Float32Array | null, R: number, N: number, Kk: number) { return this.run(SHARD_WGSL.linear, [x, w, bias ?? new Float32Array(1)], this.u32(R, N, Kk, bias ? 1 : 0), R * N, [Math.ceil(N / 16), Math.ceil(R / 16), 1]); }
  private rmsnorm(x: Float32Array, w: Float32Array, R: number, H: number, eps: number) { const b = new ArrayBuffer(8); new Uint32Array(b, 0, 1)[0] = H; new Float32Array(b, 4, 1)[0] = eps; return this.run(SHARD_WGSL.rmsnorm, [x, w], new Uint8Array(b), R * H, [R, 1, 1]); }
  private rope(t: Float32Array, cos: Float32Array, sin: Float32Array, SEQ: number, nh: number, HD: number) { return this.run(SHARD_WGSL.rope, [t, cos, sin], this.u32(SEQ, nh, HD), SEQ * nh * HD, [Math.ceil(SEQ * nh * HD / 64), 1, 1]); }
  private async layer(h: Float32Array, g: (n: string, opt?: boolean) => Float32Array | null, cos: Float32Array, sin: Float32Array, SEQ: number, c: ShardCfg): Promise<Float32Array> {
    const { H, NH, NKV, HD, INT, eps } = c;
    const req = (n: string) => { const w = g(n); if (!w) throw new Error(`missing weight ${n}`); return w; };
    const ln1 = await this.rmsnorm(h, req('input_layernorm.weight'), SEQ, H, eps);
    // q/k/v bias is present on Qwen2/Qwen2.5, absent on Llama/SmolLM — optional (null → no bias add).
    const q = await this.linear(ln1, req('self_attn.q_proj.weight'), g('self_attn.q_proj.bias', true), SEQ, NH * HD, H);
    const k = await this.linear(ln1, req('self_attn.k_proj.weight'), g('self_attn.k_proj.bias', true), SEQ, NKV * HD, H);
    const v = await this.linear(ln1, req('self_attn.v_proj.weight'), g('self_attn.v_proj.bias', true), SEQ, NKV * HD, H);
    const qR = await this.rope(q, cos, sin, SEQ, NH, HD), kR = await this.rope(k, cos, sin, SEQ, NKV, HD);
    const ctx = await this.run(SHARD_WGSL.attn, [qR, kR, v], this.u32(SEQ, NH, NKV, HD), SEQ * NH * HD, [SEQ, NH, 1]);
    const attnOut = await this.linear(ctx, req('self_attn.o_proj.weight'), null, SEQ, H, NH * HD);
    const hMid = await this.run(SHARD_WGSL.add, [h, attnOut], this.u32(SEQ * H), SEQ * H, [Math.ceil(SEQ * H / 64), 1, 1]);
    const ln2 = await this.rmsnorm(hMid, req('post_attention_layernorm.weight'), SEQ, H, eps);
    const gate = await this.linear(ln2, req('mlp.gate_proj.weight'), null, SEQ, INT, H);
    const up = await this.linear(ln2, req('mlp.up_proj.weight'), null, SEQ, INT, H);
    const act = await this.run(SHARD_WGSL.swiglu, [gate, up], this.u32(SEQ * INT), SEQ * INT, [Math.ceil(SEQ * INT / 64), 1, 1]);
    const mlpOut = await this.linear(act, req('mlp.down_proj.weight'), null, SEQ, H, INT);
    return await this.run(SHARD_WGSL.add, [hMid, mlpOut], this.u32(SEQ * H), SEQ * H, [Math.ceil(SEQ * H / 64), 1, 1]);
  }
  async stage(hidden: Float32Array, positions: number[], weights: Map<string, { data: Float32Array }>, start: number, end: number, c: ShardCfg): Promise<Float32Array> {
    const SEQ = positions.length; const { cos, sin } = computeCosSin(positions, c.HD, c.theta); let h = hidden;
    for (let li = start; li < end; li++) { const g = (n: string, opt?: boolean) => { const w = weights.get(`model.layers.${li}.${n}`); if (!w) { if (opt) return null; throw new Error(`missing weight model.layers.${li}.${n}`); } return w.data; }; h = await this.layer(h, g, cos, sin, SEQ, c); }
    return h;
  }
}

let gpuLost = false; // set if the GPU device is lost (Metal reset, driver crash) → we fall back to CPU
async function makeGpuBackend(): Promise<Backend | null> {
  const gpu = (navigator as { gpu?: GPU }).gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  // fp16 (half precision) if the adapter supports it — requestDevice REJECTS an unsupported feature,
  // so gate on it. f16 weights halve residency memory + matmul bandwidth (with f32 accumulation).
  const hasF16 = adapter.features.has('shader-f16');
  // Ask for the adapter's real buffer limits (default caps storage buffers at 128 MiB on Apple silicon,
  // which would fail a large data-mode matmul); clamp to what the adapter actually supports.
  const device = await adapter.requestDevice({
    requiredFeatures: hasF16 ? ['shader-f16'] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  gpuLost = false;
  device.lost.then((info) => { gpuLost = true; console.error(`[worker] GPU device lost (${(info as GPUDeviceLostInfo)?.reason ?? 'unknown'}); falling back to CPU.`); }).catch(() => {});
  async function run(code: string, storage: (Float32Array | Uint16Array)[], uniform: Uint32Array | Float32Array | null, outLen: number, dispatch: [number, number, number]) {
    // buffer size must be a multiple of 4 bytes; an f16 (Uint16) array with an odd element count is 2 mod 4.
    const inBufs = storage.map((arr) => { const b = device.createBuffer({ size: Math.max(4, (arr.byteLength + 3) & ~3), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, arr as BufferSource); return b; });
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
  const ELEM_OP: Record<string, number> = { relu: 0, scale: 1, gelu: 2, vector_add: 0, vector_mul: 1, saxpy: 2 };
  // WebGPU caps workgroups per dispatch dimension (usually 65535); dispatch in chunks so large inputs
  // (data mode allows up to 16M elements) never exceed it — each chunk is an independent dispatch.
  const MAXW = device.limits.maxComputeWorkgroupsPerDimension || 65535;
  const ELEM_PER = MAXW * 64;   // max elements per elementwise dispatch (workgroup_size 64)
  return { kind: 'gpu', label: `gpu:${info.vendor || 'webgpu'}/${info.architecture || 'native'}${hasF16 ? '+f16' : ''}`, hasF16,
    shard: new ShardRuntime(device), // this GPU worker can HOLD a pipeline middle-stage
    matmul: (a, b, M, N, K) => run(WGSL.matmul, [a, b], new Uint32Array([M, N, K, 0]), M * N, [Math.ceil(N / 16), Math.ceil(M / 16), 1]),
    matmulF16: hasF16 ? ((a, b, M, N, K) => run(WGSL.matmulF16, [a, b], new Uint32Array([M, N, K, 0]), M * N, [Math.ceil(N / 16), Math.ceil(M / 16), 1])) : undefined,
    elementwise: async (kernel, a, b, scalar) => {
      const op = ELEM_OP[kernel] ?? 0;
      const unary = kernel === 'relu' || kernel === 'scale' || kernel === 'gelu';
      const code = unary ? WGSL.unary : WGSL.binary;
      const uni = new Float32Array([op, scalar, 0, 0]);
      if (a.length <= ELEM_PER) {
        const wg: [number, number, number] = [Math.ceil(a.length / 64), 1, 1];
        return unary ? run(code, [a], uni, a.length, wg) : run(code, [a, b as Float32Array], uni, a.length, wg);
      }
      const out = new Float32Array(a.length);
      for (let s = 0; s < a.length; s += ELEM_PER) {
        const aa = a.subarray(s, Math.min(a.length, s + ELEM_PER));
        const wg: [number, number, number] = [Math.ceil(aa.length / 64), 1, 1];
        const chunk = unary ? await run(code, [aa], uni, aa.length, wg) : await run(code, [aa, (b as Float32Array).subarray(s, s + aa.length)], uni, aa.length, wg);
        out.set(chunk, s);
      }
      return out;
    },
    rowwise: async (kernel, a, cols) => {
      const code = kernel === 'softmax' ? WGSL.softmax : WGSL.layernorm;
      const uni = new Uint32Array([cols, 0, 0, 0]);
      const rows = Math.max(1, Math.floor(a.length / cols));
      if (rows <= MAXW) return run(code, [a], uni, a.length, [rows, 1, 1]);
      const out = new Float32Array(a.length);
      for (let r = 0; r < rows; r += MAXW) {
        const rn = Math.min(rows, r + MAXW) - r;
        const aa = a.subarray(r * cols, (r + rn) * cols);
        const chunk = await run(code, [aa], uni, aa.length, [rn, 1, 1]);
        out.set(chunk, r * cols);
      }
      return out;
    } };
}

/** CPU backend: computes in small row-chunks and sleeps between them to honor the duty cycle,
 *  so the machine stays responsive for its user and average power draw stays low. */
function makeCpuBackend(): Backend {
  return { kind: 'cpu', label: `cpu:${Deno.build.arch}`,
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

// ---------- Ed25519 result signing (per-worker authenticity + tamper-evidence) ----------
// The worker signs every result with a fresh keypair; the coordinator verifies against the public key
// it registered. A worker cannot forge or tamper another worker's result, and altered bytes are caught.
const SIGN_KEYS = await crypto.subtle.generateKey({ name: 'Ed25519' }, true, ['sign', 'verify']) as CryptoKeyPair;
const PUBKEY_B64 = b64e(new Uint8Array(await crypto.subtle.exportKey('raw', SIGN_KEYS.publicKey)));
async function signResult(shardId: string, blob: { iv: string; ct: string }): Promise<string> {
  const sig = await crypto.subtle.sign({ name: 'Ed25519' }, SIGN_KEYS.privateKey, new TextEncoder().encode(`${shardId}|${blob.iv}|${blob.ct}`));
  return b64e(new Uint8Array(sig));
}

// ---------- elementwise tensor kernels (index-sharded; memory-bound so run on CPU, duty-throttled) ----------
const ELEMENTWISE = new Set(['vector_add', 'vector_mul', 'saxpy', 'relu', 'scale', 'gelu']);
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
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
        case 'gelu': o[i] = gelu(a[i]); break;
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
let backend = (FORCE_CPU ? null : await makeGpuBackend().catch(() => null)) ?? makeCpuBackend();
console.log(`[worker] ${NAME} · backend=${backend.label} · server=${SERVER}`);

/** Run one shard on the current backend; on a GPU failure/device-loss, permanently fall back to CPU and retry. */
async function computeShard(req: { kernel: string; a: string; b?: string; bRef?: string; scalar?: number; rows?: number; N?: number; K?: number; cols?: number }): Promise<Float32Array> {
  if (backend.kind === 'gpu' && gpuLost) { console.error('[worker] GPU marked lost — switching to CPU backend.'); backend = makeCpuBackend(); }
  const run = async (): Promise<Float32Array> => {
    if (req.kernel === 'matmul') {
      const A = b64ToF32(req.a);
      if (req.bRef) {  // resident-weight matmul: B comes from this worker's cache (never re-sent)
        const rw = residentWeights.get(req.bRef);
        if (!rw) throw new Error(`weight ${req.bRef} not resident on this worker`);
        if (rw.dtype === 'f16') {
          if (backend.kind === 'gpu' && backend.matmulF16) {                    // f16 A × f16 B, f32 accumulate
            return await backend.matmulF16(f32ToF16bits(A), rw.data as Uint16Array, req.rows!, req.N!, req.K!);
          }
          return await backend.matmul(A, f16bitsToF32(rw.data as Uint16Array), req.rows!, req.N!, req.K!);  // dequantize fallback
        }
        return await backend.matmul(A, rw.data as Float32Array, req.rows!, req.N!, req.K!);
      }
      return await backend.matmul(A, b64ToF32(req.b!), req.rows!, req.N!, req.K!);
    }
    if (ROWWISE.has(req.kernel)) return backend.rowwise ? await backend.rowwise(req.kernel, b64ToF32(req.a), req.cols!) : runRowwise(req.kernel, b64ToF32(req.a), req.cols!);
    if (ELEMENTWISE.has(req.kernel)) return backend.elementwise ? await backend.elementwise(req.kernel, b64ToF32(req.a), req.b ? b64ToF32(req.b) : null, req.scalar ?? 1) : await runElementwise(req.kernel, b64ToF32(req.a), req.b ? b64ToF32(req.b) : null, req.scalar ?? 1);
    throw new Error(`unknown kernel ${req.kernel}`);
  };
  try { return await run(); }
  catch (e) {
    if (backend.kind === 'gpu') { console.error(`[worker] GPU compute failed (${e}); switching to CPU and retrying.`); backend = makeCpuBackend(); return await run(); }
    throw e;
  }
}
// ── MODEL SHARD (pipeline MIDDLE stage) — staging + dispatch for the sealed 'model' RPC ──
// The coordinator streams this stage's config.json + per-stage safetensors via push_begin/push_chunk (download-
// free), then shard_load parses them into GPU-ready weights and shard_forward runs the stage's decoder blocks
// (hidden→hidden). This is what makes a WebGPU device a real shard host (not just an offloaded-kernel donor).
const SHARD_STAGE = new Map<string, { weights: Map<string, { data: Float32Array }>; cfg: ShardCfg; start: number; end: number }>();
const SHARD_PUSH = new Map<string, Map<string, Uint8Array[]>>(); // id → filename → streamed chunks (in-memory staging)
function cfgFromJson(txt: string): ShardCfg {
  const c = JSON.parse(txt) as Record<string, number>;
  const H = c.hidden_size, NH = c.num_attention_heads, NKV = c.num_key_value_heads ?? NH;
  return { H, NH, NKV, HD: c.head_dim ?? Math.floor(H / NH), INT: c.intermediate_size, eps: c.rms_norm_eps ?? 1e-6, theta: c.rope_theta ?? 10000 };
}
async function modelDispatch(op: string, p: Record<string, unknown>): Promise<Record<string, unknown>> {
  const id = String(p.id ?? p.model ?? '');
  if (op === 'ping') return { ok: true, pong: true, n: String((p.blob as string) ?? '').length }; // RTT/throughput probe
  if (op === 'push_begin') { SHARD_PUSH.set(id, new Map()); return { ok: true, id, staging: 'ram', resumed: false, sizes: {} }; }
  if (op === 'push_chunk') { const files = SHARD_PUSH.get(id) ?? new Map<string, Uint8Array[]>(); SHARD_PUSH.set(id, files); const name = String(p.name); const arr = files.get(name) ?? []; arr.push(b64d(String(p.data))); files.set(name, arr); return { ok: true }; }
  if (op === 'push_end') return { ok: true };
  if (op === 'shard_load') {
    if (p.first || p.last) throw new Error('a WebGPU worker can host only a MIDDLE stage (no embeddings/tokenizer/head)');
    if (!backend.shard) throw new Error('this worker has no GPU shard runtime (CPU-only backend)');
    const files = SHARD_PUSH.get(id); if (!files) throw new Error('shard weights not staged — push_begin/push_chunk must precede shard_load');
    const join = (name: string): Uint8Array | null => { const parts = files.get(name); if (!parts) return null; const n = parts.reduce((a, b) => a + b.length, 0); const out = new Uint8Array(n); let o = 0; for (const pt of parts) { out.set(pt, o); o += pt.length; } return out; };
    const cfgBytes = join('config.json'); if (!cfgBytes) throw new Error('no config.json staged');
    const stBytes = join('model.safetensors'); if (!stBytes) throw new Error('no model.safetensors staged');
    const cfg = cfgFromJson(new TextDecoder().decode(cfgBytes)); const weights = parseSafetensors(stBytes);
    const start = Number(p.start), end = Number(p.end);
    SHARD_STAGE.set(id, { weights, cfg, start, end }); SHARD_PUSH.delete(id);
    let held = 0; for (const w of weights.values()) held += w.data.length;
    return { ok: true, params_held: held, layers: end - start };
  }
  if (op === 'shard_forward') {
    const st = SHARD_STAGE.get(id); if (!st) throw new Error(`shard ${id} not loaded`);
    if (p.cached && Number(p.pos ?? 0) > 0) throw new Error('WebGPU middle stage is uncached — a pos>0 KV-decode step is not supported yet; re-feed the whole sequence (uncached)');
    const seq = Number(p.seq); const hidden = b64ToF32(String(p.hidden));
    const positions = Array.from({ length: seq }, (_, i) => i);
    const out = await backend.shard!.stage(hidden, positions, st.weights, st.start, st.end, st.cfg);
    return { ok: true, hidden: f32ToB64(out), seq, hidden_dim: st.cfg.H };
  }
  if (op === 'shard_reset') return { ok: true };
  if (op === 'shard_unload') { SHARD_STAGE.delete(id); SHARD_PUSH.delete(id); return { ok: true }; }
  throw new Error(`webgpu worker: unsupported model op '${op}' (this worker hosts MIDDLE stages only)`);
}
if (!TOKEN) console.log('[worker] warning: no join token set (--token / MOREGPU_TOKEN) — the server will reject me');
let tenantKey: Uint8Array | null = null;
// Weight RESIDENCY: named tensors cached on this worker so a matmul can reference one by id (bRef)
// instead of re-uploading it every call. This is what lets a model be split across workers (place each
// layer's weights on a different worker → activations pipeline through; weights are sent ONCE).
const residentWeights = new Map<string, { data: Float32Array | Uint16Array; dtype: 'f32' | 'f16' }>();
const b64ToU16 = (s: string) => new Uint16Array(b64d(s).buffer);
const f32ToF16bits = (a: Float32Array) => new Uint16Array(new Float16Array(a).buffer);
const f16bitsToF32 = (u: Uint16Array) => Float32Array.from(new Float16Array(u.buffer, u.byteOffset, u.length));

function connect() {
  const ws = new WebSocket(SERVER);
  let hb: ReturnType<typeof setInterval> | undefined;
  ws.onopen = () => {
    // CAPABILITIES: a GPU worker can hold a pipeline MIDDLE stage (WGSL transformer forward) → advertise 'shard';
    // it has no embeddings/tokenizer/head, so NOT 'shardEnds', no autograd (NOT 'train'), no whole-model residency.
    const caps = backend.shard ? ['kernel', 'shard'] : ['kernel'];
    ws.send(JSON.stringify({ t: 'register', joinToken: TOKEN, pubkey: PUBKEY_B64, node: { id: NAME, backend: backend.kind, label: backend.label, os: Deno.build.os, caps } }));
    // Heartbeat: report live load, adaptive duty, the ceiling, and why (if) we're paused.
    hb = setInterval(() => {
      let load1 = 0; try { load1 = Deno.loadavg()[0]; } catch { /* */ }
      const active = isActive();
      const reason = adminPaused ? 'admin' : (!scheduleActive() ? 'schedule' : null);
      try { ws.send(JSON.stringify({ t: 'heartbeat', id: NAME, load1, cores: CORES, util: +(load1 / CORES).toFixed(3), duty: active ? +effectiveDuty().toFixed(3) : 0, ceil: +(Number.isNaN(CEIL) ? 0.6 : CEIL).toFixed(2), paused: !active, pausedReason: reason, schedule: SCHEDULE })); } catch { /* */ }
    }, 4000);
  };
  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data as string);
    if (msg.t === 'denied') { console.error(`[worker] rejected by server: ${msg.reason}. Check the join token.`); try { ws.close(); } catch { /* */ } Deno.exit(1); }
    if (msg.t === 'welcome') { tenantKey = b64d(msg.tenantKeyB64); if (Number.isNaN(CEIL)) CEIL = typeof msg.duty === 'number' ? msg.duty : 0.6; console.log(`[worker] joined pool · duty ceiling=${(CEIL * 100).toFixed(0)}% · schedule=${SCHEDULE} · adaptive (backs off as your machine gets busier, keeps total load < ${(MAX_UTIL * 100).toFixed(0)}%)`); return; }
    if (msg.t === 'control') { // the admin remotely pauses/resumes, caps duty, or sets a schedule for this machine
      if (typeof msg.pause === 'boolean') adminPaused = msg.pause;
      if (typeof msg.ceil === 'number') CEIL = Math.max(MIN_DUTY, Math.min(1, msg.ceil));
      if (typeof msg.schedule === 'string') SCHEDULE = msg.schedule.trim().toLowerCase();
      const active = isActive();
      const reason = adminPaused ? 'admin' : (!scheduleActive() ? 'schedule' : null);
      console.log(`[worker] admin control → ${adminPaused ? 'PAUSED' : 'active'} · duty ceiling=${((Number.isNaN(CEIL) ? 0.6 : CEIL) * 100).toFixed(0)}% · schedule=${SCHEDULE}`);
      try { ws.send(JSON.stringify({ t: 'heartbeat', id: NAME, load1: 0, cores: CORES, util: +emaUtil.toFixed(3), duty: active ? +effectiveDuty().toFixed(3) : 0, ceil: +(Number.isNaN(CEIL) ? 0.6 : CEIL).toFixed(2), paused: !active, pausedReason: reason, schedule: SCHEDULE })); } catch { /* */ }
      return;
    }
    if (msg.t === 'cache') { // coordinator asks this worker to hold a named weight resident
      try {
        if (!tenantKey) throw new Error('no tenant key');
        const w = JSON.parse(new TextDecoder().decode(await unseal(tenantKey, msg.sealed)));
        const dtype = w.dtype === 'f16' ? 'f16' : 'f32';
        residentWeights.set(msg.id, { data: dtype === 'f16' ? b64ToU16(w.data) : b64ToF32(w.data), dtype });
        console.log(`[worker] cached weight ${msg.id} (${w.rows}x${w.cols} ${dtype}) · resident=${residentWeights.size}`);
        ws.send(JSON.stringify({ t: 'cached', id: msg.id, ok: true }));
      } catch (e) { ws.send(JSON.stringify({ t: 'cached', id: msg.id, ok: false, error: String(e) })); }
      return;
    }
    if (msg.t === 'uncache') { residentWeights.delete(msg.id); return; }
    if (msg.t === 'assign') {
      const t0 = performance.now();
      try {
        if (!tenantKey) throw new Error('no tenant key');
        const req = JSON.parse(new TextDecoder().decode(await unseal(tenantKey, msg.sealedIn)));
        const out = await computeShard(req);
        const sealedOut = await seal(tenantKey, new TextEncoder().encode(JSON.stringify({ out: f32ToB64(out) })));
        const ms = performance.now() - t0;
        const sig = await signResult(msg.shardId, sealedOut);
        ws.send(JSON.stringify({ t: 'result', shardId: msg.shardId, jobId: msg.jobId, ok: true, sealedOut, sig, ms, backend: backend.label }));
        console.log(`[worker] ${msg.shardId} done in ${ms.toFixed(1)}ms on ${backend.kind}`);
        const cool = effectiveDuty(); if (cool < 1) await sleep(ms * (1 / cool - 1)); // adaptive cool-down between shards
      } catch (e) { ws.send(JSON.stringify({ t: 'result', shardId: msg.shardId, jobId: msg.jobId, ok: false, error: String(e) })); }
    }
    if (msg.t === 'model') { // sealed pipeline-shard RPC (push_begin/chunk, shard_load/forward/reset/unload, ping)
      try {
        if (!tenantKey) throw new Error('no tenant key');
        const payload = JSON.parse(new TextDecoder().decode(await unseal(tenantKey, msg.sealed)));
        const res = await modelDispatch(String(msg.op), payload);
        const sealed = await seal(tenantKey, new TextEncoder().encode(JSON.stringify(res)));
        ws.send(JSON.stringify({ t: 'model_reply', reqId: msg.reqId, ok: true, sealed }));
      } catch (e) { ws.send(JSON.stringify({ t: 'model_reply', reqId: msg.reqId, ok: false, error: String(e) })); }
    }
  };
  ws.onclose = () => { if (hb) clearInterval(hb); console.log('[worker] disconnected, retrying in 2s'); setTimeout(connect, 2000); };
  ws.onerror = () => { try { ws.close(); } catch { /* */ } };
}
connect();

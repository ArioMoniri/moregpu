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

// PLATFORM SHIM: this one file runs BOTH under Deno (CLI) and in a BROWSER TAB (served as a bundle by an
// HTML page). Only the host APIs differ — everything below (WebSocket, WebGPU, crypto.subtle, WGSL) is
// identical. In a browser, config comes from globalThis.MOREGPU (set by the page) or ?server=&token= URL params.
const _D = (globalThis as { Deno?: { args: string[]; env: { get(k: string): string | undefined }; build: { os: string; arch: string }; hostname(): string; loadavg(): number[]; exit(c: number): never } }).Deno;
const _CFG = ((globalThis as { MOREGPU?: Record<string, string> }).MOREGPU) ?? {};
const _URLQ = (() => { try { return new URLSearchParams((globalThis as { location?: { search: string } }).location?.search ?? ''); } catch { return new URLSearchParams(); } })();
const PLAT = {
  os: _D ? _D.build.os : 'browser', arch: _D ? _D.build.arch : ((globalThis as { navigator?: { platform?: string } }).navigator?.platform ?? 'web'),
  hostname: () => { try { return _D ? _D.hostname() : 'browser'; } catch { return 'browser'; } },
  loadavg: (): number[] => { try { return _D ? _D.loadavg() : [0, 0, 0]; } catch { return [0, 0, 0]; } }, // browser has no load avg → the duty throttle just uses CPU-EMA
  exit: (c: number) => { if (_D) _D.exit(c); else throw new Error(`worker exit(${c})`); },
  env: (k: string): string | undefined => (_D ? _D.env.get(k) : (_CFG[k] ?? undefined)),
};
const args = new Map<string, string>();
if (_D) for (let i = 0; i < _D.args.length; i++) { const a = _D.args[i]; if (a.startsWith('--')) args.set(a.slice(2), _D.args[i + 1] ?? 'true'); }
const _pick = (k: string, env: string) => args.get(k) ?? _CFG[k] ?? _URLQ.get(k) ?? PLAT.env(env) ?? undefined;
const SERVER = _pick('server', 'MOREGPU_SERVER') ?? 'ws://localhost:8787/ws';
const TOKEN = _pick('token', 'MOREGPU_TOKEN') ?? '';
const NAME = _pick('name', 'MOREGPU_NAME') ?? `${PLAT.hostname()}-${crypto.randomUUID().slice(0, 6)}`;
// Duty cycle CEILING: the most of this machine the pool may ever use (fraction of time computing).
// The EFFECTIVE duty adapts DOWN from this ceiling in real time based on the machine's own load, so
// the moment the user works their PC harder, the pool's share shrinks and the user is not disturbed.
let CEIL = args.has('throttle') ? Math.max(0.05, Math.min(1, Number(args.get('throttle')))) : NaN;
const MIN_DUTY = 0.05;
// Keep TOTAL system utilization under this; the pool only ever uses the slack below it. Lower = more
// headroom reserved for the user. Configurable per machine.
const MAX_UTIL = Math.max(0.3, Math.min(0.98, Number(PLAT.env('MOREGPU_MAX_UTIL') ?? args.get('max-util') ?? 0.85)));
const CORES = Math.max(1, navigator.hardwareConcurrency || 4);
let emaUtil = 0; // smoothed system utilization (excluding transient spikes)
let lastDuty = MIN_DUTY;

// ---------- scheduling + remote control (the USER decides WHEN this machine is lent; the ADMIN can override) ----------
// MOREGPU_SCHEDULE / --schedule:  "always" (default) · "idle-only" (only when the machine is idle) ·
//   "HH:MM-HH:MM" active window in local time (may wrap past midnight, e.g. "22:00-07:00" = nights only).
// While outside the window / not idle / admin-paused, the worker takes NO new work (duty 0) and reports
// "paused" so the coordinator stops assigning to it. In-flight shards always finish (work is never dropped).
let SCHEDULE = (args.get('schedule') ?? PLAT.env('MOREGPU_SCHEDULE') ?? 'always').trim().toLowerCase();
let adminPaused = false; // toggled by an admin 'control' frame from the coordinator
const IDLE_UTIL = Math.max(0.05, Math.min(0.9, Number(PLAT.env('MOREGPU_IDLE_UTIL') ?? 0.25)));
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
  try { load1 = PLAT.loadavg()[0]; } catch { load1 = 0; }
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
// ===== BPE TOKENIZER (verified byte-level BPE) BEGIN =====
// Lets a FIRST-stage worker (browser or Deno) turn TEXT -> token ids locally, so the client
// need not pre-tokenize. Pure TS (TextEncoder/TextDecoder/RegExp) — runs under Deno AND a browser.
// Loads a model tokenizer from a HF tokenizer.json OR a vocab.json + merges.txt pair. Verified to
// EXACTLY reproduce the reference tokenizer (byte-level BPE: bytes<->unicode, regex pre-tokenize,
// rank-based merges, special-token splitting; gpt2 + qwen2/gpt4 pre-tokenizer regex variants).
// ---------------------------------------------------------------------------
// Byte <-> unicode table (the exact GPT-2 construction).
// ---------------------------------------------------------------------------

function bytesToUnicode(): { byteToChar: string[]; charToByte: Map<string, number> } {
  const bs: number[] = [];
  for (let i = "!".charCodeAt(0); i <= "~".charCodeAt(0); i++) bs.push(i);
  for (let i = "¡".charCodeAt(0); i <= "¬".charCodeAt(0); i++) bs.push(i);
  for (let i = "®".charCodeAt(0); i <= "ÿ".charCodeAt(0); i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const byteToChar: string[] = new Array(256);
  const charToByte = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCharCode(cs[i]);
    byteToChar[bs[i]] = ch;
    charToByte.set(ch, bs[i]);
  }
  return { byteToChar, charToByte };
}

// ---------------------------------------------------------------------------
// Pre-tokenizer regexes.
// ---------------------------------------------------------------------------

// Classic GPT-2 / RoBERTa pattern.
const GPT2_PATTERN =
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// Qwen2 / GPT-4-family pattern. The upstream tokenizer.json writes the contraction
// group as `(?i:'s|'t|...)`; JS engines do not universally support inline `(?i:)`
// modifiers, so we expand it into explicit ASCII case classes (exactly equivalent
// for these ASCII-only contractions). Digits are isolated one at a time via \p{N}.
const QWEN2_PATTERN =
  "'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])" +
  "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" +
  "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

const NAMED_PATTERNS: Record<string, string> = {
  gpt2: GPT2_PATTERN,
  qwen2: QWEN2_PATTERN,
  gpt4: QWEN2_PATTERN, // same family
};

// Translate a pattern that came from a tokenizer.json Split rule into a JS-safe
// source: expand any `(?i:...)` groups (ASCII letters -> [aA] classes) so the
// pattern works in browsers that lack inline-modifier support.
function toJsSafePattern(src: string): string {
  let out = "";
  let i = 0;
  while (i < src.length) {
    if (src.startsWith("(?i:", i)) {
      // Find the matching close paren.
      let depth = 1;
      let j = i + 4;
      let inner = "";
      for (; j < src.length && depth > 0; j++) {
        const c = src[j];
        if (c === "\\") {
          inner += c + (src[j + 1] ?? "");
          j++;
          continue;
        }
        if (c === "(") depth++;
        else if (c === ")") {
          depth--;
          if (depth === 0) break;
        }
        inner += c;
      }
      out += "(?:" + expandAsciiCaseInsensitive(inner) + ")";
      i = j + 1;
    } else {
      out += src[i];
      i++;
    }
  }
  return out;
}

function expandAsciiCaseInsensitive(s: string): string {
  let out = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === "\\") {
      out += c + (s[i + 1] ?? "");
      i++;
      continue;
    }
    if (/[a-z]/.test(c)) out += "[" + c + c.toUpperCase() + "]";
    else if (/[A-Z]/.test(c)) out += "[" + c.toLowerCase() + c + "]";
    else out += c;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Types.
// ---------------------------------------------------------------------------

interface AddedToken {
  id: number;
  content: string;
  special: boolean;
  lstrip?: boolean;
  rstrip?: boolean;
  singleWord?: boolean;
  normalized?: boolean;
}

type PretokVariant = "gpt2" | "qwen2" | "gpt4";

interface TokenizerConfig {
  vocab: Record<string, number>;
  merges: Array<[string, string]>;
  addedTokens?: AddedToken[];
  /** Named pre-tokenizer variant. Ignored if `pretokenizerRegex` is given. */
  pretokenizer?: PretokVariant;
  /** Explicit pre-tokenizer regex source (as found in a tokenizer.json Split). */
  pretokenizerRegex?: string;
  /** ByteLevel add_prefix_space (GPT-2/Qwen2 default: false). */
  addPrefixSpace?: boolean;
  /** Apply Unicode NFC normalization before pre-tokenization (Qwen2's tokenizer.json declares a NFC normalizer). */
  nfc?: boolean;
  unkToken?: string;
}

// ---------------------------------------------------------------------------
// The tokenizer.
// ---------------------------------------------------------------------------

class BpeTokenizer {
  readonly vocab: Map<string, number>;
  readonly idToToken: Map<number, string>;
  readonly bpeRanks: Map<string, number>;
  readonly addedTokens: AddedToken[];
  readonly addPrefixSpace: boolean;
  readonly nfc: boolean;
  readonly unkToken?: string;

  private readonly byteToChar: string[];
  private readonly charToByte: Map<string, number>;
  private readonly pat: RegExp;
  private readonly specialById: Map<number, AddedToken>;
  private readonly specialByContent: Map<string, AddedToken>;
  private readonly splitRe: RegExp | null;
  private readonly encoder = new TextEncoder();
  private readonly decoder = new TextDecoder("utf-8");
  private readonly cache = new Map<string, string[]>();

  constructor(cfg: TokenizerConfig) {
    // vocab / merges.
    this.vocab = new Map(Object.entries(cfg.vocab));
    this.idToToken = new Map();
    for (const [tok, id] of this.vocab) this.idToToken.set(id, tok);

    this.bpeRanks = new Map();
    for (let i = 0; i < cfg.merges.length; i++) {
      const [a, b] = cfg.merges[i];
      this.bpeRanks.set(a + " " + b, i);
    }

    // byte-level table.
    const bt = bytesToUnicode();
    this.byteToChar = bt.byteToChar;
    this.charToByte = bt.charToByte;

    // pre-tokenizer regex.
    let patternSrc: string;
    if (cfg.pretokenizerRegex) {
      patternSrc = toJsSafePattern(cfg.pretokenizerRegex);
    } else {
      const name = cfg.pretokenizer ?? "gpt2";
      patternSrc = NAMED_PATTERNS[name];
      if (!patternSrc) throw new Error("unknown pretokenizer variant: " + name);
    }
    this.pat = new RegExp(patternSrc, "gu");

    this.addPrefixSpace = cfg.addPrefixSpace ?? false;
    this.nfc = cfg.nfc ?? false;
    this.unkToken = cfg.unkToken;

    // added / special tokens.
    this.addedTokens = cfg.addedTokens ? cfg.addedTokens.slice() : [];
    this.specialById = new Map();
    this.specialByContent = new Map();
    for (const t of this.addedTokens) {
      this.specialById.set(t.id, t);
      this.specialByContent.set(t.content, t);
      if (!this.idToToken.has(t.id)) this.idToToken.set(t.id, t.content);
      if (!this.vocab.has(t.content)) this.vocab.set(t.content, t.id);
    }
    // Build a splitter that keeps added-token contents as separate chunks
    // (longest content first so overlapping tokens match greedily).
    if (this.addedTokens.length) {
      const contents = this.addedTokens
        .map((t) => t.content)
        .sort((a, b) => b.length - a.length)
        .map(escapeRegex);
      this.splitRe = new RegExp("(" + contents.join("|") + ")");
    } else {
      this.splitRe = null;
    }
  }

  // ---- loaders ----------------------------------------------------------

  /** Load from a parsed HuggingFace `tokenizer.json` object. */
  static fromTokenizerJson(obj: any): BpeTokenizer {
    const model = obj.model ?? {};
    const vocab: Record<string, number> = model.vocab ?? {};
    const merges = normalizeMerges(model.merges ?? []);

    const added: AddedToken[] = (obj.added_tokens ?? []).map((t: any) => ({
      id: t.id,
      content: t.content,
      special: !!t.special,
      lstrip: !!t.lstrip,
      rstrip: !!t.rstrip,
      singleWord: !!t.single_word,
      normalized: !!t.normalized,
    }));

    const { regex, addPrefixSpace } = readPreTokenizer(obj.pre_tokenizer);

    return new BpeTokenizer({
      vocab,
      merges,
      addedTokens: added,
      pretokenizerRegex: regex ?? undefined,
      pretokenizer: regex ? undefined : "gpt2",
      addPrefixSpace,
      nfc: hasNFCNormalizer(obj.normalizer),
      unkToken: model.unk_token ?? undefined,
    });
  }

  /**
   * Load from a `vocab.json` object + raw `merges.txt` text.
   * `opts.pretokenizer` selects the regex variant ("gpt2" default, or "qwen2").
   */
  static fromVocabAndMerges(
    vocab: Record<string, number>,
    mergesText: string,
    opts: {
      pretokenizer?: PretokVariant;
      pretokenizerRegex?: string;
      addedTokens?: AddedToken[];
      addPrefixSpace?: boolean;
      unkToken?: string;
    } = {},
  ): BpeTokenizer {
    const merges: Array<[string, string]> = [];
    for (const raw of mergesText.split("\n")) {
      const line = raw.replace(/\r$/, "");
      // Skip only the optional `#version:` header — NOT legitimate merges whose left
      // piece is the '#' byte-token (e.g. `# #`, `## ##`, `#$ #$`), which are real BPE merges.
      if (!line || line.startsWith("#version")) continue;
      const sp = line.indexOf(" ");
      if (sp < 0) continue;
      merges.push([line.slice(0, sp), line.slice(sp + 1)]);
    }
    return new BpeTokenizer({
      vocab,
      merges,
      addedTokens: opts.addedTokens,
      pretokenizer: opts.pretokenizer ?? "gpt2",
      pretokenizerRegex: opts.pretokenizerRegex,
      addPrefixSpace: opts.addPrefixSpace,
      unkToken: opts.unkToken,
    });
  }

  // ---- core BPE ---------------------------------------------------------

  private bpe(token: string): string[] {
    const cached = this.cache.get(token);
    if (cached) return cached;

    let word = Array.from(token);
    if (word.length < 2) {
      this.cache.set(token, word);
      return word;
    }

    for (;;) {
      // find the lowest-rank adjacent pair.
      let minRank = Infinity;
      let first = "";
      let second = "";
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.bpeRanks.get(word[i] + " " + word[i + 1]);
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          first = word[i];
          second = word[i + 1];
        }
      }
      if (minRank === Infinity) break;

      // merge every non-overlapping occurrence of (first, second).
      const newWord: string[] = [];
      let i = 0;
      while (i < word.length) {
        let j = -1;
        for (let k = i; k < word.length; k++) {
          if (word[k] === first) {
            j = k;
            break;
          }
        }
        if (j === -1) {
          for (let k = i; k < word.length; k++) newWord.push(word[k]);
          break;
        }
        for (let k = i; k < j; k++) newWord.push(word[k]);
        i = j;
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
          newWord.push(first + second);
          i += 2;
        } else {
          newWord.push(word[i]);
          i += 1;
        }
      }
      word = newWord;
      if (word.length === 1) break;
    }

    this.cache.set(token, word);
    return word;
  }

  /** Encode one pre-token (already isolated by the regex) into ids. */
  private encodePiece(piece: string, out: number[]): void {
    const bytes = this.encoder.encode(piece);
    let s = "";
    for (let i = 0; i < bytes.length; i++) s += this.byteToChar[bytes[i]];
    for (const sub of this.bpe(s)) {
      const id = this.vocab.get(sub);
      if (id === undefined) {
        if (this.unkToken && this.vocab.has(this.unkToken)) {
          out.push(this.vocab.get(this.unkToken)!);
          continue;
        }
        throw new Error("piece not in vocab: " + JSON.stringify(sub));
      }
      out.push(id);
    }
  }

  private encodeNormal(text: string, out: number[]): void {
    if (!text) return;
    const matches = text.matchAll(this.pat);
    for (const m of matches) this.encodePiece(m[0], out);
  }

  // ---- public API -------------------------------------------------------

  encode(text: string): number[] {
    if (this.nfc) text = text.normalize("NFC");   // Qwen2 (and any NFC-normalizer tokenizer) canonicalizes first
    if (this.addPrefixSpace && text.length && !/^\s/.test(text)) {
      text = " " + text;
    }
    const out: number[] = [];
    if (!this.splitRe) {
      this.encodeNormal(text, out);
      return out;
    }
    // Split on added/special tokens; odd chunks are the special tokens.
    const parts = text.split(this.splitRe);
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      if (!part) continue;
      const special = this.specialByContent.get(part);
      if (special) {
        out.push(special.id);
      } else {
        this.encodeNormal(part, out);
      }
    }
    return out;
  }

  decode(ids: number[], opts: { skipSpecialTokens?: boolean } = {}): string {
    const skip = opts.skipSpecialTokens ?? false;
    let bytes: number[] = [];
    let result = "";
    const flush = () => {
      if (bytes.length) {
        result += this.decoder.decode(new Uint8Array(bytes));
        bytes = [];
      }
    };
    for (const id of ids) {
      const special = this.specialById.get(id);
      if (special) {
        flush();
        if (!skip) result += special.content;
        continue;
      }
      const tok = this.idToToken.get(id);
      if (tok === undefined) continue;
      for (const ch of tok) {
        const b = this.charToByte.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    flush();
    return result;
  }

  get vocabSize(): number {
    return this.vocab.size;
  }
}

// ---------------------------------------------------------------------------
// helpers.
// ---------------------------------------------------------------------------

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// merges can be ["a b", ...] or [["a","b"], ...] depending on tokenizer version.
function normalizeMerges(merges: any[]): Array<[string, string]> {
  const out: Array<[string, string]> = [];
  for (const m of merges) {
    if (typeof m === "string") {
      const sp = m.indexOf(" ");
      out.push([m.slice(0, sp), m.slice(sp + 1)]);
    } else if (Array.isArray(m) && m.length === 2) {
      out.push([m[0], m[1]]);
    }
  }
  return out;
}

// Detect whether a tokenizer.json `normalizer` node applies NFC (directly, or nested in a Sequence).
function hasNFCNormalizer(node: any): boolean {
  if (!node) return false;
  if (node.type === "NFC") return true;
  if (Array.isArray(node.normalizers)) return node.normalizers.some(hasNFCNormalizer);
  return false;
}

// Read a tokenizer.json pre_tokenizer node -> { explicit regex?, add_prefix_space }.
function readPreTokenizer(
  pt: any,
): { regex: string | null; addPrefixSpace: boolean } {
  let regex: string | null = null;
  let addPrefixSpace = false;
  const visit = (node: any) => {
    if (!node) return;
    if (Array.isArray(node)) {
      for (const n of node) visit(n);
      return;
    }
    switch (node.type) {
      case "Sequence":
        visit(node.pretokenizers);
        break;
      case "Split":
        if (node.pattern && typeof node.pattern.Regex === "string") {
          regex = node.pattern.Regex;
        }
        break;
      case "ByteLevel":
        if (typeof node.add_prefix_space === "boolean") {
          addPrefixSpace = node.add_prefix_space;
        }
        break;
      default:
        // Some configs nest a list under `pretokenizers` without a Sequence type.
        if (node.pretokenizers) visit(node.pretokenizers);
        break;
    }
  };
  visit(pt);
  return { regex, addPrefixSpace };
}
// ===== BPE TOKENIZER END =====

// ===== SHARD RUNTIME (extractable for tests) BEGIN =====
interface ShardCfg { H: number; NH: number; NKV: number; HD: number; INT: number; eps: number; theta: number }
// A loaded weight is EITHER dequantized fp32 (`data`) or QUANTIZED (`q`): packed int8/int4 (4/u32 or
// 8-nibbles/u32) plus a per-row (int8) / per-group (int4) f32 scale. i8/u8 are transient raw forms held
// between parse and assembleQuant. Quantized weights stay ~1 B/weight (int8) or ~0.5 B/weight (int4) in
// host RAM and on the GPU upload — a browser/mobile tab holds a bigger slice; a push-streamed stage is smaller.
interface QInfo { kind: 'i8' | 'i4'; wq: Uint32Array; scale: Float32Array; N: number; K: number; group: number; ng: number }
interface WEntry { data?: Float32Array; i8?: Int8Array; u8?: Uint8Array; q?: QInfo; shape: number[] }
function f16ToF32(h: number): number {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
// safetensors bytes → { name: WEntry }. F32/F16/BF16 → dequantized `data`; I8/U8 kept RAW (quantized weights,
// paired with their `.scale` sibling by assembleQuant — never dequantized, so the shard stays small).
function parseSafetensors(buf: Uint8Array): Map<string, WEntry> {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + headerLen)));
  const dataStart = 8 + headerLen, out = new Map<string, WEntry>();
  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const m = meta as { dtype: string; shape: number[]; data_offsets: [number, number] };
    const raw = buf.subarray(dataStart + m.data_offsets[0], dataStart + m.data_offsets[1]);
    const n = m.shape.reduce((a, b) => a * b, 1);
    if (m.dtype === 'I8') { out.set(name, { i8: new Int8Array(raw.slice().buffer), shape: m.shape }); continue; }
    if (m.dtype === 'U8') { out.set(name, { u8: raw.slice(), shape: m.shape }); continue; }
    let data: Float32Array;
    if (m.dtype === 'F32') data = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    else if (m.dtype === 'F16') { data = new Float32Array(n); const u = new Uint16Array(raw.buffer, raw.byteOffset, n); for (let i = 0; i < n; i++) data[i] = f16ToF32(u[i]); }
    else if (m.dtype === 'BF16') { data = new Float32Array(n); const u = new Uint16Array(raw.buffer, raw.byteOffset, n); const f = new Uint32Array(data.buffer); for (let i = 0; i < n; i++) f[i] = u[i] << 16; }
    else throw new Error(`unsupported safetensors dtype ${m.dtype} for ${name}`);
    out.set(name, { data, shape: m.shape });
  }
  return out;
}
// Pair each raw int8/int4 weight `NAME` with its `NAME.scale` sibling into a QInfo (packing the bytes 4/u32),
// then drop the scale entry. A quantized checkpoint stores block Linear weights as I8 [N,K] (int8, per-row
// scale [N]) or U8 [N,K/2] (packed int4, per-group scale [N,ng]); norms/embeddings/head stay fp32.
function assembleQuant(weights: Map<string, WEntry>): void {
  const toU32 = (bytes: Uint8Array): Uint32Array => { const pad = new Uint8Array(Math.ceil(bytes.length / 4) * 4); pad.set(bytes); return new Uint32Array(pad.buffer); };
  for (const [name, e] of [...weights]) {
    if (!e.i8 && !e.u8) continue;
    const scaleE = weights.get(name + '.scale'); if (!scaleE?.data) throw new Error(`quantized weight ${name} has no ${name}.scale`);
    const scale = scaleE.data;
    if (e.i8) {
      const [N, K] = e.shape; // int8 weight is [N,K]
      weights.set(name, { q: { kind: 'i8', wq: toU32(new Uint8Array(e.i8.buffer, e.i8.byteOffset, e.i8.length)), scale, N, K, group: K, ng: 1 }, shape: [N, K] });
    } else {
      const N = e.shape[0], K = e.shape[1] * 2; // packed int4 is [N, K/2]
      const ng = scale.length / N; const group = K / ng;
      if (K % 2 !== 0 || K % group !== 0 || N <= 0) throw new Error(`int4 weight ${name}: needs K even and K%group==0 (N=${N} K=${K} group=${group})`);
      weights.set(name, { q: { kind: 'i4', wq: toU32(e.u8!), scale, N, K, group, ng }, shape: [N, K] });
    }
    weights.delete(name + '.scale');
  }
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
  // VEC4 GEMM — reads x/w as vec4<f32> (128-bit loads), BIT-IDENTICAL to `linear` (explicit .x/.y/.z/.w adds keep
  // the exact scalar accumulation order). This GEMM is memory-bandwidth-bound and Apple's unified cache already
  // captures the reuse shared-memory TILING exists for (tiling measured SLOWER on Metal); wider loads are the win —
  // ~1.6x (decode) to ~2.3x (prefill vec4x4), zero accuracy change. Needs K%4==0 (every Qwen/Llama linear); the
  // caller falls back to `linear` otherwise. Same bindings/uniform as `linear`, so the SAME f32 buffers upload.
  linearVec4: `@group(0) @binding(0) var<storage,read> x:array<vec4<f32>>;@group(0) @binding(1) var<storage,read> w:array<vec4<f32>>;@group(0) @binding(2) var<storage,read> bias:array<f32>;@group(0) @binding(3) var<storage,read_write> y:array<f32>;struct U{R:u32,N:u32,K:u32,hasBias:u32};@group(0) @binding(4) var<uniform> u:U;
@compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) g:vec3<u32>){let r=g.y;let n=g.x;if(r>=u.R||n>=u.N){return;}let kv=u.K/4u;let xb=r*kv;let wb=n*kv;var a=0.0;for(var q=0u;q<kv;q=q+1u){let xv=x[xb+q];let wv=w[wb+q];a=a+xv.x*wv.x;a=a+xv.y*wv.y;a=a+xv.z*wv.z;a=a+xv.w*wv.w;}if(u.hasBias==1u){a=a+bias[n];}y[r*u.N+n]=a;}`,
  // VEC4X4 — vec4 loads + 4 output cols/thread (each x-vec4 reused across 4 weight rows). Fastest for PREFILL (R>1);
  // dispatch X uses a /64 divisor (16 threads × 4 cols). Bit-identical; needs K%4==0.
  linearVec4x4: `@group(0) @binding(0) var<storage,read> x:array<vec4<f32>>;@group(0) @binding(1) var<storage,read> w:array<vec4<f32>>;@group(0) @binding(2) var<storage,read> bias:array<f32>;@group(0) @binding(3) var<storage,read_write> y:array<f32>;struct U{R:u32,N:u32,K:u32,hasBias:u32};@group(0) @binding(4) var<uniform> u:U;
@compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) g:vec3<u32>){let r=g.y;if(r>=u.R){return;}let nb=g.x*4u;let kv=u.K/4u;let xb=r*kv;let n0=nb;let n1=nb+1u;let n2=nb+2u;let n3=nb+3u;var a0=0.0;var a1=0.0;var a2=0.0;var a3=0.0;for(var q=0u;q<kv;q=q+1u){let xv=x[xb+q];if(n0<u.N){let wv=w[n0*kv+q];a0=a0+xv.x*wv.x;a0=a0+xv.y*wv.y;a0=a0+xv.z*wv.z;a0=a0+xv.w*wv.w;}if(n1<u.N){let wv=w[n1*kv+q];a1=a1+xv.x*wv.x;a1=a1+xv.y*wv.y;a1=a1+xv.z*wv.z;a1=a1+xv.w*wv.w;}if(n2<u.N){let wv=w[n2*kv+q];a2=a2+xv.x*wv.x;a2=a2+xv.y*wv.y;a2=a2+xv.z*wv.z;a2=a2+xv.w*wv.w;}if(n3<u.N){let wv=w[n3*kv+q];a3=a3+xv.x*wv.x;a3=a3+xv.y*wv.y;a3=a3+xv.z*wv.z;a3=a3+xv.w*wv.w;}}if(n0<u.N){var v=a0;if(u.hasBias==1u){v=v+bias[n0];}y[r*u.N+n0]=v;}if(n1<u.N){var v=a1;if(u.hasBias==1u){v=v+bias[n1];}y[r*u.N+n1]=v;}if(n2<u.N){var v=a2;if(u.hasBias==1u){v=v+bias[n2];}y[r*u.N+n2]=v;}if(n3<u.N){var v=a3;if(u.hasBias==1u){v=v+bias[n3];}y[r*u.N+n3]=v;}}`,
  rmsnorm: `@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> w:array<f32>;@group(0) @binding(2) var<storage,read_write> y:array<f32>;struct U{H:u32,eps:f32};@group(0) @binding(3) var<uniform> u:U;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) wid:vec3<u32>){let r=wid.x;var s=0.0;for(var i=0u;i<u.H;i=i+1u){let v=x[r*u.H+i];s=s+v*v;}let inv=inverseSqrt(s/f32(u.H)+u.eps);for(var i=0u;i<u.H;i=i+1u){y[r*u.H+i]=x[r*u.H+i]*inv*w[i];}}`,
  rope: `@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> cosb:array<f32>;@group(0) @binding(2) var<storage,read> sinb:array<f32>;@group(0) @binding(3) var<storage,read_write> y:array<f32>;struct U{SEQ:u32,NHEADS:u32,HD:u32};@group(0) @binding(4) var<uniform> u:U;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let idx=g.y*nwg.x*64u+g.x;let total=u.SEQ*u.NHEADS*u.HD;if(idx>=total){return;}let d=idx%u.HD;let rest=idx/u.HD;let h=rest%u.NHEADS;let s=rest/u.NHEADS;let half=u.HD/2u;let base=(s*u.NHEADS+h)*u.HD;let xv=x[base+d];var rot:f32;if(d<half){rot=-x[base+d+half];}else{rot=x[base+d-half];}y[idx]=xv*cosb[s*u.HD+d]+rot*sinb[s*u.HD+d];}`,
  // (The former fixed-`array<f32,512>` `attn` kernel is gone — the uncached path now uses `cachedAttn` with past=0,
  //  the same online-softmax kernel, which has no sequence cap. That removes the silent >512-token corruption.)
  swiglu: `@group(0) @binding(0) var<storage,read> gate:array<f32>;@group(0) @binding(1) var<storage,read> up:array<f32>;@group(0) @binding(2) var<storage,read_write> out:array<f32>;struct U{n:u32};@group(0) @binding(3) var<uniform> u:U;@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=g.y*nwg.x*64u+g.x;if(i>=u.n){return;}let x=gate[i];out[i]=(x/(1.0+exp(-x)))*up[i];}`,
  add: `@group(0) @binding(0) var<storage,read> a:array<f32>;@group(0) @binding(1) var<storage,read> b:array<f32>;@group(0) @binding(2) var<storage,read_write> o:array<f32>;struct U{n:u32};@group(0) @binding(3) var<uniform> u:U;@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=g.y*nwg.x*64u+g.x;if(i>=u.n){return;}o[i]=a[i]+b[i];}`,
  // FIRST-stage embedding gather: out[s,:] = embed_tokens.weight[ids[s],:]. ids uploaded as u32 bit-patterns via a Float32Array view.
  embed: `@group(0) @binding(0) var<storage,read> ids:array<u32>;@group(0) @binding(1) var<storage,read> w:array<f32>;@group(0) @binding(2) var<storage,read_write> out:array<f32>;struct U{seq:u32,H:u32};@group(0) @binding(3) var<uniform> u:U;@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=g.y*nwg.x*64u+g.x;let total=u.seq*u.H;if(i>=total){return;}let s=i/u.H;let d=i%u.H;out[i]=w[ids[s]*u.H+d];}`,
  // ONLINE-softmax cached attention: streams the whole KV cache (running max/denom/context), so ANY length — no
  // fixed scores array (removes the seq<=512 cap) — and supports KV-cached decode (q for the new tokens attends
  // over the full cache). causal: token i (abs pos past+i) attends 0..(past+i). GQA kv=h/(NH/NKV). dispatch [seqNew,NH,1].
  cachedAttn: `@group(0) @binding(0) var<storage,read> q:array<f32>;@group(0) @binding(1) var<storage,read> k:array<f32>;@group(0) @binding(2) var<storage,read> v:array<f32>;@group(0) @binding(3) var<storage,read_write> ctx:array<f32>;struct U{seqNew:u32,NH:u32,NKV:u32,HD:u32,past:u32};@group(0) @binding(4) var<uniform> u:U;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;let h=g.y;if(i>=u.seqNew||h>=u.NH){return;}let kv=h/(u.NH/u.NKV);let scale=1.0/sqrt(f32(u.HD));let qbase=(i*u.NH+h)*u.HD;let limit=u.past+i;var m=-3.0e38;var l=0.0;var acc:array<f32,256>;for(var d=0u;d<u.HD;d=d+1u){acc[d]=0.0;}for(var j=0u;j<=limit;j=j+1u){let kb=(j*u.NKV+kv)*u.HD;var dot=0.0;for(var d=0u;d<u.HD;d=d+1u){dot=dot+q[qbase+d]*k[kb+d];}let s=dot*scale;let newm=max(m,s);let corr=exp(m-newm);let p=exp(s-newm);l=l*corr+p;let vb=(j*u.NKV+kv)*u.HD;for(var d=0u;d<u.HD;d=d+1u){acc[d]=acc[d]*corr+p*v[vb+d];}m=newm;}let ob=i*(u.NH*u.HD)+h*u.HD;for(var d=0u;d<u.HD;d=d+1u){ctx[ob+d]=acc[d]/l;}}`,
  // QUANTIZED linears — dequantize in the shader (portable base-WebGPU int/float ops, NO packed-dot / 8-bit-storage
  // feature). Weights packed 4/u32: int8 symmetric per-row (byte b of word = flat idx&3; scale[N]); int4 symmetric
  // per-group (2 nibbles/byte along K, scale[N,ng]). Matches SHARD_WGSL.linear layout/dispatch [ceil(N/16),ceil(R/16),1].
  linearQ8: `@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> wq:array<u32>;@group(0) @binding(2) var<storage,read> scale:array<f32>;@group(0) @binding(3) var<storage,read> bias:array<f32>;@group(0) @binding(4) var<storage,read_write> y:array<f32>;struct U{R:u32,N:u32,K:u32,hasBias:u32};@group(0) @binding(5) var<uniform> u:U;
@compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) g:vec3<u32>){let r=g.y;let n=g.x;if(r>=u.R||n>=u.N){return;}let sc=scale[n];let wbase=n*u.K;let xrow=r*u.K;var acc=0.0;for(var k=0u;k<u.K;k=k+1u){let idx=wbase+k;let word=wq[idx>>2u];let bv=(word>>((idx&3u)*8u))&0xFFu;var wi=i32(bv);if(wi>127){wi=wi-256;}acc=acc+x[xrow+k]*(f32(wi)*sc);}if(u.hasBias==1u){acc=acc+bias[n];}y[r*u.N+n]=acc;}`,
  linearQ4: `@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> wq:array<u32>;@group(0) @binding(2) var<storage,read> scale:array<f32>;@group(0) @binding(3) var<storage,read> bias:array<f32>;@group(0) @binding(4) var<storage,read_write> y:array<f32>;struct U{R:u32,N:u32,K:u32,group:u32,ng:u32,hasBias:u32};@group(0) @binding(5) var<uniform> u:U;
@compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) g:vec3<u32>){let r=g.y;let n=g.x;if(r>=u.R||n>=u.N){return;}let rowBytes=u.K/2u;let byteBase=n*rowBytes;let xrow=r*u.K;let scBase=n*u.ng;var acc=0.0;for(var k=0u;k<u.K;k=k+1u){let bi=byteBase+(k>>1u);let word=wq[bi>>2u];let bv=(word>>((bi&3u)*8u))&0xFFu;var nib:u32;if((k&1u)==0u){nib=bv&0xFu;}else{nib=(bv>>4u)&0xFu;}var wi=i32(nib);if(wi>7){wi=wi-16;}let sc=scale[scBase+(k/u.group)];acc=acc+x[xrow+k]*(f32(wi)*sc);}if(u.hasBias==1u){acc=acc+bias[n];}y[r*u.N+n]=acc;}`,
};
interface LayerKV { K: Float32Array; V: Float32Array; len: number }
class ShardRuntime {
  private cache = new Map<string, GPUComputePipeline>();
  constructor(private dev: GPUDevice) {}
  private pipe(code: string): GPUComputePipeline { let p = this.cache.get(code); if (!p) { p = this.dev.createComputePipeline({ layout: 'auto', compute: { module: this.dev.createShaderModule({ code }), entryPoint: 'main' } }); this.cache.set(code, p); } return p; }
  private async run(code: string, storage: ArrayBufferView[], uniform: ArrayBufferView, outLen: number, dispatch: [number, number, number]): Promise<Float32Array> {
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
  // Dispatch for a 1-D @workgroup_size(64) kernel over `total` elements, folded into 2-D so it never exceeds
  // maxComputeWorkgroupsPerDimension (65535). The kernels reconstruct the flat index as g.y*nwg.x*64 + g.x. A
  // large prefill (e.g. 3B swiglu at seq≈400 → 66k workgroups) used to overflow one dimension and fail the stage.
  private d1(total: number): [number, number, number] { const wg = Math.ceil(total / 64), MAX = 65535; return wg <= MAX ? [wg, 1, 1] : [MAX, Math.ceil(wg / MAX), 1]; }
  private linear(x: Float32Array, w: Float32Array, bias: Float32Array | null, R: number, N: number, Kk: number) {
    const args = [x, w, bias ?? new Float32Array(1)], uni = this.u32(R, N, Kk, bias ? 1 : 0);
    // vec4 loads are BIT-IDENTICAL to the naive kernel and ~1.6-2.3x faster (memory-bandwidth-bound GEMM; wider
    // loads beat shared-memory tiling on Apple's unified cache). Needs K%4==0 (every Qwen/Llama linear); vec4x4
    // (4 output cols/thread) wins for prefill (R>1), plain vec4 for a 1-row decode. Else fall back to the naive kernel.
    if (Kk % 4 === 0) {
      if (R > 1) return this.run(SHARD_WGSL.linearVec4x4, args, uni, R * N, [Math.ceil(N / 64), Math.ceil(R / 16), 1]);
      return this.run(SHARD_WGSL.linearVec4, args, uni, R * N, [Math.ceil(N / 16), Math.ceil(R / 16), 1]);
    }
    return this.run(SHARD_WGSL.linear, args, uni, R * N, [Math.ceil(N / 16), Math.ceil(R / 16), 1]);
  }
  private linearQ8(x: Float32Array, q: QInfo, bias: Float32Array | null, R: number) { return this.run(SHARD_WGSL.linearQ8, [x, q.wq, q.scale, bias ?? new Float32Array(1)], this.u32(R, q.N, q.K, bias ? 1 : 0), R * q.N, [Math.ceil(q.N / 16), Math.ceil(R / 16), 1]); }
  private linearQ4(x: Float32Array, q: QInfo, bias: Float32Array | null, R: number) { return this.run(SHARD_WGSL.linearQ4, [x, q.wq, q.scale, bias ?? new Float32Array(1)], this.u32(R, q.N, q.K, q.group, q.ng, bias ? 1 : 0), R * q.N, [Math.ceil(q.N / 16), Math.ceil(R / 16), 1]); }
  // Route one linear to fp32 or the int8/int4 kernel based on how the weight was loaded (see WEntry/assembleQuant).
  private matmul(x: Float32Array, e: WEntry, bias: Float32Array | null, R: number, N: number, K: number) { return e.q ? (e.q.kind === 'i8' ? this.linearQ8(x, e.q, bias, R) : this.linearQ4(x, e.q, bias, R)) : this.linear(x, e.data!, bias, R, N, K); }
  private rmsnorm(x: Float32Array, w: Float32Array, R: number, H: number, eps: number) { const b = new ArrayBuffer(8); new Uint32Array(b, 0, 1)[0] = H; new Float32Array(b, 4, 1)[0] = eps; return this.run(SHARD_WGSL.rmsnorm, [x, w], new Uint8Array(b), R * H, [R, 1, 1]); }
  private rope(t: Float32Array, cos: Float32Array, sin: Float32Array, SEQ: number, nh: number, HD: number) { return this.run(SHARD_WGSL.rope, [t, cos, sin], this.u32(SEQ, nh, HD), SEQ * nh * HD, this.d1(SEQ * nh * HD)); }
  private async layer(h: Float32Array, g: (n: string, opt?: boolean) => WEntry | null, cos: Float32Array, sin: Float32Array, SEQ: number, c: ShardCfg): Promise<Float32Array> {
    const { H, NH, NKV, HD, INT, eps } = c;
    const req = (n: string) => { const w = g(n); if (!w) throw new Error(`missing weight ${n}`); return w; };
    const nf = (n: string) => req(n).data!;                 // fp32-only weights (norms)
    const bias = (n: string) => g(n, true)?.data ?? null;   // q/k/v bias present on Qwen2/2.5, absent on Llama/SmolLM
    const ln1 = await this.rmsnorm(h, nf('input_layernorm.weight'), SEQ, H, eps);
    const q = await this.matmul(ln1, req('self_attn.q_proj.weight'), bias('self_attn.q_proj.bias'), SEQ, NH * HD, H);
    const k = await this.matmul(ln1, req('self_attn.k_proj.weight'), bias('self_attn.k_proj.bias'), SEQ, NKV * HD, H);
    const v = await this.matmul(ln1, req('self_attn.v_proj.weight'), bias('self_attn.v_proj.bias'), SEQ, NKV * HD, H);
    const qR = await this.rope(q, cos, sin, SEQ, NH, HD), kR = await this.rope(k, cos, sin, SEQ, NKV, HD);
    const ctx = await this.run(SHARD_WGSL.cachedAttn, [qR, kR, v], this.u32(SEQ, NH, NKV, HD, 0), SEQ * NH * HD, [SEQ, NH, 1]); // online-softmax, past=0 → full causal attention, NO seq cap
    const attnOut = await this.matmul(ctx, req('self_attn.o_proj.weight'), null, SEQ, H, NH * HD);
    const hMid = await this.run(SHARD_WGSL.add, [h, attnOut], this.u32(SEQ * H), SEQ * H, this.d1(SEQ * H));
    const ln2 = await this.rmsnorm(hMid, nf('post_attention_layernorm.weight'), SEQ, H, eps);
    const gate = await this.matmul(ln2, req('mlp.gate_proj.weight'), null, SEQ, INT, H);
    const up = await this.matmul(ln2, req('mlp.up_proj.weight'), null, SEQ, INT, H);
    const act = await this.run(SHARD_WGSL.swiglu, [gate, up], this.u32(SEQ * INT), SEQ * INT, this.d1(SEQ * INT));
    const mlpOut = await this.matmul(act, req('mlp.down_proj.weight'), null, SEQ, H, INT);
    return await this.run(SHARD_WGSL.add, [hMid, mlpOut], this.u32(SEQ * H), SEQ * H, this.d1(SEQ * H));
  }
  async stage(hidden: Float32Array, positions: number[], weights: Map<string, WEntry>, start: number, end: number, c: ShardCfg): Promise<Float32Array> {
    const SEQ = positions.length; const { cos, sin } = computeCosSin(positions, c.HD, c.theta); let h = hidden;
    for (let li = start; li < end; li++) { const g = (n: string, opt?: boolean) => { const w = weights.get(`model.layers.${li}.${n}`); if (!w) { if (opt) return null; throw new Error(`missing weight model.layers.${li}.${n}`); } return w; }; h = await this.layer(h, g, cos, sin, SEQ, c); }
    return h;
  }
  // FIRST stage: token ids → embeddings (gather rows of embed_tokens.weight). ids uploaded as u32 bit-patterns.
  embed(ids: number[], embW: Float32Array, H: number): Promise<Float32Array> {
    const idsF32 = new Float32Array(new Uint32Array(ids).buffer);
    return this.run(SHARD_WGSL.embed, [idsF32, embW], this.u32(ids.length, H), ids.length * H, this.d1(ids.length * H));
  }
  // LAST stage: final RMSNorm → lm_head → logits[seq,vocab]. headW is embed_tokens.weight when tied.
  async lmHead(hidden: Float32Array, normW: Float32Array, headW: Float32Array, seq: number, H: number, vocab: number, eps: number): Promise<Float32Array> {
    const normed = await this.rmsnorm(hidden, normW, seq, H, eps);
    return await this.linear(normed, headW, null, seq, vocab, H);
  }
  newCache(NL: number): LayerKV[] { return Array.from({ length: NL }, () => ({ K: new Float32Array(0), V: new Float32Array(0), len: 0 })); }
  // One cached decoder block: `seqNew` new tokens at absolute position `past`; appends K/V to the layer cache
  // and attends over the full cache with online softmax (any length; used for both prefill and 1-token decode).
  private async cachedLayer(h: Float32Array, g: (n: string, opt?: boolean) => WEntry | null, seqNew: number, past: number, kvc: LayerKV, c: ShardCfg): Promise<Float32Array> {
    const { H, NH, NKV, HD, INT, eps, theta } = c;
    const req = (n: string) => { const w = g(n); if (!w) throw new Error(`missing weight ${n}`); return w; };
    const nf = (n: string) => req(n).data!;
    const bias = (n: string) => g(n, true)?.data ?? null;
    const ln1 = await this.rmsnorm(h, nf('input_layernorm.weight'), seqNew, H, eps);
    const q = await this.matmul(ln1, req('self_attn.q_proj.weight'), bias('self_attn.q_proj.bias'), seqNew, NH * HD, H);
    const k = await this.matmul(ln1, req('self_attn.k_proj.weight'), bias('self_attn.k_proj.bias'), seqNew, NKV * HD, H);
    const v = await this.matmul(ln1, req('self_attn.v_proj.weight'), bias('self_attn.v_proj.bias'), seqNew, NKV * HD, H);
    const positions = Array.from({ length: seqNew }, (_, i) => past + i);
    const { cos, sin } = computeCosSin(positions, HD, theta);
    const qR = await this.rope(q, cos, sin, seqNew, NH, HD), kR = await this.rope(k, cos, sin, seqNew, NKV, HD);
    const Kfull = new Float32Array(kvc.K.length + kR.length); Kfull.set(kvc.K, 0); Kfull.set(kR, kvc.K.length);
    const Vfull = new Float32Array(kvc.V.length + v.length); Vfull.set(kvc.V, 0); Vfull.set(v, kvc.V.length);
    kvc.K = Kfull; kvc.V = Vfull; kvc.len = past + seqNew;
    const ctx = await this.run(SHARD_WGSL.cachedAttn, [qR, Kfull, Vfull], this.u32(seqNew, NH, NKV, HD, past), seqNew * NH * HD, [seqNew, NH, 1]);
    const attnOut = await this.matmul(ctx, req('self_attn.o_proj.weight'), null, seqNew, H, NH * HD);
    const hMid = await this.run(SHARD_WGSL.add, [h, attnOut], this.u32(seqNew * H), seqNew * H, this.d1(seqNew * H));
    const ln2 = await this.rmsnorm(hMid, nf('post_attention_layernorm.weight'), seqNew, H, eps);
    const gate = await this.matmul(ln2, req('mlp.gate_proj.weight'), null, seqNew, INT, H);
    const up = await this.matmul(ln2, req('mlp.up_proj.weight'), null, seqNew, INT, H);
    const act = await this.run(SHARD_WGSL.swiglu, [gate, up], this.u32(seqNew * INT), seqNew * INT, this.d1(seqNew * INT));
    const mlpOut = await this.matmul(act, req('mlp.down_proj.weight'), null, seqNew, H, INT);
    return await this.run(SHARD_WGSL.add, [hMid, mlpOut], this.u32(seqNew * H), seqNew * H, this.d1(seqNew * H));
  }
  // Cached stage — same call serves PREFILL (past=0, seqNew=SEQ) and DECODE (past=M, seqNew=1). Mutates `cache`.
  async cachedStage(hidden: Float32Array, past: number, weights: Map<string, WEntry>, start: number, end: number, cache: LayerKV[], c: ShardCfg): Promise<Float32Array> {
    const seqNew = Math.floor(hidden.length / c.H); let h = hidden;
    for (let li = start; li < end; li++) { const g = (n: string, opt?: boolean) => { const w = weights.get(`model.layers.${li}.${n}`); if (!w) { if (opt) return null; throw new Error(`missing weight model.layers.${li}.${n}`); } return w; }; h = await this.cachedLayer(h, g, seqNew, past, cache[li - start], c); }
    return h;
  }
}
// ===== SHARD RUNTIME END =====

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
  return { kind: 'cpu', label: `cpu:${PLAT.arch}`,
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
// INT8 ACTIVATION WIRE (opt-in) — a stage's [seq,H] hidden crosses the wire as symmetric per-ROW (per-token)
// int8 + one f32 scale/row instead of fp32 (~4x smaller). BYTE-IDENTICAL with worker_torch.py's actq_encode
// (both stages of a boundary must agree): scale = fround(maxAbs/127) [f64 divide → f32], q = round-half-away-
// from-zero of the f64 quotient, clamp [-127,127]. Lossy (per-hop, compounds) — used only where the plan enables it.
function actqEncode(h: Float32Array, seq: number, H: number): { qB64: string; scaleB64: string } {
  const q = new Int8Array(seq * H), scale = new Float32Array(seq);
  for (let i = 0; i < seq; i++) {
    const base = i * H; let maxAbs = 0;
    for (let j = 0; j < H; j++) { const a = Math.abs(h[base + j]); if (a > maxAbs) maxAbs = a; }
    const s = Math.fround(maxAbs / 127); scale[i] = s; const denom = s === 0 ? 1 : s;
    for (let j = 0; j < H; j++) { const v = h[base + j] / denom; let qi = Math.floor(Math.abs(v) + 0.5); if (v < 0) qi = -qi; q[base + j] = qi > 127 ? 127 : qi < -127 ? -127 : qi; }
  }
  return { qB64: b64e(new Uint8Array(q.buffer)), scaleB64: b64e(new Uint8Array(scale.buffer)) };
}
function actqDecode(qB64: string, scaleB64: string, seq: number, H: number): Float32Array {
  const q = new Int8Array(b64d(qB64).buffer), scale = new Float32Array(b64d(scaleB64).buffer), out = new Float32Array(seq * H);
  for (let i = 0; i < seq; i++) { const s = scale[i], base = i * H; for (let j = 0; j < H; j++) out[base + j] = Math.fround(q[base + j] * s); }
  return out;
}

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
const FORCE_CPU = args.has('cpu') || PLAT.env('MOREGPU_FORCE_CPU') === '1';
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
const SHARD_STAGE = new Map<string, { weights: Map<string, WEntry>; cfg: ShardCfg; start: number; end: number; first: boolean; last: boolean; vocab: number; kv: Map<string, LayerKV[]>; tok: BpeTokenizer | null; eos: number | null }>();
// Resolve the model's EOS token id from the streamed generation_config.json (eos_token_id, int or list) or, as a
// fallback, tokenizer_config.json's eos_token string mapped through the tokenizer's vocab — so a WebGPU first stage
// can stop generation on EOS just like the torch worker (which uses tk.eos_token_id).
function resolveEos(join: (n: string) => Uint8Array | null, tok: BpeTokenizer | null): number | null {
  try { const g = join('generation_config.json'); if (g) { const e = (JSON.parse(new TextDecoder().decode(g)) as { eos_token_id?: number | number[] }).eos_token_id; if (Array.isArray(e) && e.length) return Number(e[0]); if (typeof e === 'number') return e; } } catch { /* */ }
  try { const t = join('tokenizer_config.json'); if (t && tok) { const raw = (JSON.parse(new TextDecoder().decode(t)) as { eos_token?: string | { content?: string } }).eos_token; const content = typeof raw === 'string' ? raw : raw?.content; if (content) { const id = tok.vocab.get(content); if (typeof id === 'number') return id; } } } catch { /* */ }
  return null;
}
const SHARD_PUSH = new Map<string, Map<string, Uint8Array[]>>(); // id → filename → streamed chunks (in-memory staging)
const SHARD_PUSH_BYTES = new Map<string, number>(); // id → staged bytes, for the cap below
// Resource bounds so a buggy/hostile coordinator can't OOM the worker: cap concurrent live decode sessions (LRU-evict
// the idle ones), the number of concurrent push-staging ids (evict an abandoned push whose shard_load never arrived),
// and the total staged bytes across all pushes. All overridable via env.
const _envn = (k: string, d: number) => { const v = _D?.env.get(k); const n = v ? Number(v) : NaN; return Number.isFinite(n) && n > 0 ? n : d; };
const MAX_KV_SESSIONS = _envn('MOREGPU_MAX_KV_SESSIONS', 8);
const MAX_STAGING = _envn('MOREGPU_MAX_STAGING', 4);
const MAX_STAGING_BYTES = _envn('MOREGPU_MAX_STAGING_BYTES', 8 * 2 ** 30); // 8 GiB across all live pushes
const dropPush = (id: string) => { SHARD_PUSH.delete(id); SHARD_PUSH_BYTES.delete(id); };
function cfgFromJson(txt: string): { cfg: ShardCfg; vocab: number } {
  const c = JSON.parse(txt) as Record<string, number>;
  const H = c.hidden_size, NH = c.num_attention_heads, NKV = c.num_key_value_heads ?? NH;
  return { cfg: { H, NH, NKV, HD: c.head_dim ?? Math.floor(H / NH), INT: c.intermediate_size, eps: c.rms_norm_eps ?? 1e-6, theta: c.rope_theta ?? 10000 }, vocab: c.vocab_size };
}
async function modelDispatch(op: string, p: Record<string, unknown>): Promise<Record<string, unknown>> {
  const id = String(p.id ?? p.model ?? '');
  if (op === 'ping') return { ok: true, pong: true, n: String((p.blob as string) ?? '').length }; // RTT/throughput probe
  if (op === 'push_begin') { while (SHARD_PUSH.size >= MAX_STAGING && !SHARD_PUSH.has(id)) { const oldest = SHARD_PUSH.keys().next().value; if (oldest === undefined) break; dropPush(oldest); } SHARD_PUSH.set(id, new Map()); SHARD_PUSH_BYTES.set(id, 0); return { ok: true, id, staging: 'ram', resumed: false, sizes: {} }; }
  if (op === 'push_chunk') { const files = SHARD_PUSH.get(id) ?? new Map<string, Uint8Array[]>(); SHARD_PUSH.set(id, files); const chunk = b64d(String(p.data)); let total = chunk.length; for (const b of SHARD_PUSH_BYTES.values()) total += b; if (total > MAX_STAGING_BYTES) throw new Error(`push staging exceeded ${MAX_STAGING_BYTES} bytes across pushes — raise MOREGPU_MAX_STAGING_BYTES`); SHARD_PUSH_BYTES.set(id, (SHARD_PUSH_BYTES.get(id) ?? 0) + chunk.length); const name = String(p.name); const arr = files.get(name) ?? []; arr.push(chunk); files.set(name, arr); return { ok: true }; }
  if (op === 'push_end') return { ok: true };
  if (op === 'shard_load') {
    if (!backend.shard) throw new Error('this worker has no GPU shard runtime (CPU-only backend)');
    const files = SHARD_PUSH.get(id); if (!files) throw new Error('shard weights not staged — push_begin/push_chunk must precede shard_load');
    const join = (name: string): Uint8Array | null => { const parts = files.get(name); if (!parts) return null; const n = parts.reduce((a, b) => a + b.length, 0); const out = new Uint8Array(n); let o = 0; for (const pt of parts) { out.set(pt, o); o += pt.length; } return out; };
    const cfgBytes = join('config.json'); if (!cfgBytes) throw new Error('no config.json staged');
    const stBytes = join('model.safetensors'); if (!stBytes) throw new Error('no model.safetensors staged');
    const { cfg, vocab } = cfgFromJson(new TextDecoder().decode(cfgBytes)); const weights = parseSafetensors(stBytes);
    assembleQuant(weights); // pair any int8/int4 block weights with their .scale (quantized checkpoints); fp32 shards untouched
    const first = !!p.first, last = !!p.last;
    if (first && !weights.get('model.embed_tokens.weight')) throw new Error('first stage needs model.embed_tokens.weight (not streamed)');
    if (last && !weights.get('model.norm.weight')) throw new Error('last stage needs model.norm.weight (not streamed)');
    // Optional: a FIRST stage may carry the model's tokenizer.json so it can accept raw TEXT (encode on-stage),
    // making a browser/Deno tab a self-contained first stage — no client-side tokenizer needed. Ids still work too.
    const tokBytes = first ? join('tokenizer.json') : null;
    const tok = tokBytes ? BpeTokenizer.fromTokenizerJson(JSON.parse(new TextDecoder().decode(tokBytes))) : null;
    const eos = tok ? resolveEos(join, tok) : null;
    SHARD_STAGE.set(id, { weights, cfg, start: Number(p.start), end: Number(p.end), first, last, vocab, kv: new Map(), tok, eos }); dropPush(id);
    let held = 0; for (const w of weights.values()) held += (w.data?.length ?? (w.q ? w.q.N * w.q.K : 0));
    return { ok: true, params_held: held, layers: Number(p.end) - Number(p.start) };
  }
  if (op === 'shard_forward') {
    const st = SHARD_STAGE.get(id); if (!st) throw new Error(`shard ${id} not loaded`);
    const rt = backend.shard!; const { H, eps } = st.cfg;
    const cached = !!p.cached && typeof p.session === 'string';
    const past = cached ? Math.max(0, Number(p.pos ?? 0)) : 0;   // absolute start position (RoPE + causal offset)
    const w = (n: string) => st.weights.get(n)?.data;
    // stage INPUT → hidden [seqNew, H]
    let h: Float32Array, seqNew: number;
    if (st.first) {
      // input_ids from the client, OR raw text tokenized on-stage (if a tokenizer.json was staged with this first stage).
      let ids: number[];
      if (Array.isArray(p.input_ids)) ids = p.input_ids as number[];
      else if (typeof p.prompt === 'string' && st.tok) ids = st.tok.encode(p.prompt);
      else throw new Error('first stage: need input_ids, or prompt text + a staged tokenizer.json');
      seqNew = ids.length; const embW = w('model.embed_tokens.weight')!; h = await rt.embed(ids, embW, H);
    }
    else { seqNew = Number(p.seq); h = (p.actq === 'int8' && p.hidden_q) ? actqDecode((p.hidden_q as { qB64: string }).qB64, (p.hidden_q as { scaleB64: string }).scaleB64, seqNew, H) : b64ToF32(String(p.hidden)); }
    // run this stage's decoder blocks (cached KV or uncached full-sequence)
    if (cached) {
      const key = String(p.session); let kv = st.kv.get(key);
      // A pos-0 call (re)prefills → fresh cache. A pos>0 decode MUST match the cache's current length, else the
      // caller (a reconnect, a lost/evicted session, a coordinator bug) would attend over a zero-filled phantom
      // prefix and silently return wrong tokens — reject it so the coordinator resets + re-prefills instead.
      if (past === 0) { while (st.kv.size >= MAX_KV_SESSIONS && !st.kv.has(key)) { const old = st.kv.keys().next().value; if (old === undefined) break; st.kv.delete(old); } kv = rt.newCache(st.end - st.start); st.kv.set(key, kv); } // LRU-evict the oldest idle session
      else if (!kv || (kv[0] && kv[0].len !== past)) throw new Error(`KV desync for session ${key}: ${kv?.[0]?.len ?? 'no'} tokens cached but pos=${past} — reset the session and re-prefill`);
      else { st.kv.delete(key); st.kv.set(key, kv); } // LRU touch: this session is now most-recently-used
      h = await rt.cachedStage(h, past, st.weights, st.start, st.end, kv!, st.cfg);
    } else {
      const positions = Array.from({ length: seqNew }, (_, i) => i);
      h = await rt.stage(h, positions, st.weights, st.start, st.end, st.cfg);
    }
    // stage OUTPUT: last → logits + argmax; else → hidden for the next stage
    if (st.last) {
      const normW = w('model.norm.weight')!, headW = w('lm_head.weight') ?? w('model.embed_tokens.weight')!;
      // Only the LAST token's logits drive argmax / return_logits, so run the LM head on just that row — saves
      // O(seq·vocab) compute and a seq·vocab·4 output buffer (hundreds of MB for a long prefill on a 150k-vocab model,
      // which could also trip maxStorageBufferBindingSize/maxBufferSize).
      const hLast = seqNew > 1 ? h.subarray((seqNew - 1) * H, seqNew * H) : h;
      const logits = await rt.lmHead(hLast, normW, headW, 1, H, st.vocab, eps);
      const lastRow = logits.subarray(0, st.vocab);
      let argmax = 0, mx = -Infinity; for (let i = 0; i < st.vocab; i++) if (lastRow[i] > mx) { mx = lastRow[i]; argmax = i; }
      const res: Record<string, unknown> = { ok: true, argmax };
      if (p.return_logits) res.logits = f32ToB64(lastRow.slice(0));
      if (cached) res.past = past + seqNew;
      return res;
    }
    // hidden → next stage: opt-in int8 activation wire (p.actq_out set by the coordinator for non-last-input boundaries).
    const res: Record<string, unknown> = { ok: true, seq: seqNew, hidden_dim: H };
    if (p.actq_out === 'int8') { res.hidden_q = actqEncode(h, seqNew, H); res.actq = 'int8'; } else res.hidden = f32ToB64(h);
    if (cached) res.past = past + seqNew;
    return res;
  }
  if (op === 'shard_reset') { const st = SHARD_STAGE.get(id); if (st) { if (typeof p.session === 'string') st.kv.delete(String(p.session)); else st.kv.clear(); } return { ok: true }; }
  if (op === 'shard_unload') { SHARD_STAGE.delete(id); dropPush(id); return { ok: true }; }
  // TEXT ↔ ids on the FIRST stage's on-device tokenizer, so /model/shard_chat works when the first stage is a WebGPU
  // worker (mirrors worker_torch's shard_tok/shard_detok). Needs a tokenizer.json to have been streamed to this stage.
  if (op === 'shard_tok') { const st = SHARD_STAGE.get(id); if (!st?.tok) throw new Error('no tokenizer for this shard — the tokenizer.json streams to the first stage; a WebGPU first stage needs it'); return { ok: true, input_ids: st.tok.encode(String(p.prompt ?? '').slice(0, 8000)), eos: st.eos }; }
  if (op === 'shard_detok') { const st = SHARD_STAGE.get(id); if (!st?.tok) throw new Error('no tokenizer for this shard'); return { ok: true, text: st.tok.decode((p.tokens as number[]) ?? [], { skipSpecialTokens: true }) }; }
  throw new Error(`webgpu worker: unsupported model op '${op}'`);
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
    // CAPABILITIES: a GPU worker can hold ANY pipeline stage — a middle decoder-block slice ('shard') AND the
    // ends ('shardEnds': embedding gather on the first stage, final norm + LM head + argmax on the last), so it
    // can even host a whole small model solo. No autograd (NOT 'train') and no torch whole-model residency path.
    const caps = backend.shard ? ['kernel', 'shard', 'shardEnds'] : ['kernel'];
    ws.send(JSON.stringify({ t: 'register', joinToken: TOKEN, pubkey: PUBKEY_B64, node: { id: NAME, backend: backend.kind, label: backend.label, os: PLAT.os, caps } }));
    // Heartbeat: report live load, adaptive duty, the ceiling, and why (if) we're paused.
    hb = setInterval(() => {
      let load1 = 0; try { load1 = PLAT.loadavg()[0]; } catch { /* */ }
      const active = isActive();
      const reason = adminPaused ? 'admin' : (!scheduleActive() ? 'schedule' : null);
      try { ws.send(JSON.stringify({ t: 'heartbeat', id: NAME, load1, cores: CORES, util: +(load1 / CORES).toFixed(3), duty: active ? +effectiveDuty().toFixed(3) : 0, ceil: +(Number.isNaN(CEIL) ? 0.6 : CEIL).toFixed(2), paused: !active, pausedReason: reason, schedule: SCHEDULE })); } catch { /* */ }
    }, 4000);
  };
  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data as string);
    if (msg.t === 'denied') { console.error(`[worker] rejected by server: ${msg.reason}. Check the join token.`); try { ws.close(); } catch { /* */ } PLAT.exit(1); }
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

/**
 * MoreGPU WebGPU VISION executor (milestone M6, ADR-0114). Imported by worker.ts, and runs unchanged under Deno,
 * in a browser bundle and in Node (vitest).
 *
 * This module does INFERENCE and JEPA-style feature extraction only. It does not train on WebGPU.
 *
 *   op-graph JSON + safetensors ──compile──▶ launch list (+ memory plan) ──▶ runCpu()          (CPU reference)
 *                                                                        └─▶ createGpuRunner() (WGSL on WebGPU)
 *
 * What makes the CPU path a reference for the GPU path:
 *  - Every WGSL kernel has a CPU MIRROR. The mirror reads the SAME uniform array (u32 params, f32 bit-cast), sees
 *    the SAME binding windows and runs the SAME per-output algorithm: loop order, reduction trees and fp32
 *    rounding (Math.fround after every op).
 *  - Both paths execute the SAME compiled launch list, including the binding-limit tiling (gather/scatter
 *    copies + windowed bindings). A tiled CPU run is therefore bit-identical to the untiled one, and the GPU runs
 *    exactly what the CPU reference verified.
 *  - The GPU can still differ from the mirror in the last bits: FMA contraction, and exp/tanh/sqrt/division
 *    accuracy (WGSL allows a few ulp). That is why tests compare both paths to the PyTorch goldens at 1e-5 rel.
 *
 * Kernels (fp32 compute; with shader-f16, conv/convT/matmul can store weights as f16 and accumulate in f32):
 *  - conv: conv2d/3d as an implicit GEMM (the tiled 16×16 GEMM from worker.ts's WGSL.matmul, with the im2col
 *    gather fused into the A-tile load).
 *  - convT: transposed conv (gather form).
 *  - pool: max/avg 2d/3d.
 *  - upsample: nearest, bi-/trilinear.
 *  - norm: group/instance/layer norm (one workgroup per row, 256-wide tree reduction).
 *  - chanop: per-channel affine (folded batch norm, as PyTorch's own CPU eval does) and prelu.
 *  - unary: relu, leaky, gelu (erf/tanh), sigmoid, tanh, silu, …
 *  - binary: broadcast add/sub/mul/div/max/min.
 *  - linered: softmax/argmax/mean along any dim.
 *  - matmul: batched tiled GEMM, derived from WGSL.matmul (linear/addmm/mm/bmm/matmul).
 *  - attention: SDPA via online softmax, extended from SHARD_WGSL.cachedAttn (non-causal or causal, any batch×heads).
 *  - copy (strided gather: permute/expand/select/slice) and pad.
 * 2D ops are embedded as 3D with a trailing W=1, so one kernel serves both ranks bit-identically.
 */

// ════════════════════════════════════════ types ════════════════════════════════════════
export interface Tensor { shape: number[]; data: Float32Array }
export interface OpNode { op: string; inputs: (string | null)[]; attrs?: Record<string, unknown>; output: string }
export interface OpGraph { version: number; inputs: { name: string; shape: number[] }[]; nodes: OpNode[]; outputs: string[]; weights?: string }
export interface CompileOptions {
  /** Largest byte size of ONE storage binding (device.limits.maxStorageBufferBindingSize). Default 128 MiB. */
  maxBindingBytes?: number;
  /** Largest single buffer (device.limits.maxBufferSize). Default 256 MiB. */
  maxBufferBytes?: number;
  /** Store conv/convT/linear weights as f16 (needs 'shader-f16' on the GPU). The CPU mirror rounds identically. */
  f16Weights?: boolean;
  /** Batch norm: 'folded' (scale/shift at load, PyTorch's CPU eval formula; default) or 'explicit'. */
  batchNorm?: 'folded' | 'explicit';
}

const fr = Math.fround;
const numel = (s: readonly number[]) => s.reduce((a, b) => a * b, 1);
const ALIGN = 64; // elements: 256-byte minStorageBufferOffsetAlignment for f32 windows
const alignDown = (e: number) => Math.floor(e / ALIGN) * ALIGN;

// ════════════════════════════════════════ fp16 ════════════════════════════════════════
const _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);
function halfBits(v: number): number {
  _f[0] = v; const x = _u[0];
  const sign = (x >>> 16) & 0x8000, exp = (x >>> 23) & 0xff; let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let h = mant >>> shift;
    const rem = mant & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rem > half || (rem === half && (h & 1))) h++;
    return sign | h;
  }
  let h = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
  return sign | h;
}
function halfToF32(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
/** f32 → IEEE binary16 bit patterns, round-to-nearest-even (what the GPU's f16 storage holds). */
export function f32ToF16Bits(a: Float32Array): Uint16Array { const o = new Uint16Array(a.length); for (let i = 0; i < a.length; i++) o[i] = halfBits(a[i]); return o; }
export function f16BitsToF32(u: Uint16Array): Float32Array { const o = new Float32Array(u.length); for (let i = 0; i < u.length; i++) o[i] = halfToF32(u[i]); return o; }

// ════════════════════════════════════════ safetensors ════════════════════════════════════════
/** safetensors bytes → name → f32 tensor (F32/F16/BF16/F64/I64/I32/I16/I8/U8 widened to f32). */
export function parseSafetensors(buf: Uint8Array): Map<string, Tensor> {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const hlen = Number(dv.getBigUint64(0, true));
  if (hlen <= 0 || 8 + hlen > buf.byteLength) throw new Error('safetensors: bad header length');
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + hlen))) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const base = 8 + hlen, out = new Map<string, Tensor>();
  for (const [name, m] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [s, e] = m.data_offsets, n = numel(m.shape), off = base + s;
    if (base + e > buf.byteLength) throw new Error(`safetensors: ${name} out of range`);
    const d = new Float32Array(n);
    const get: Record<string, (i: number) => number> = {
      F32: (i) => dv.getFloat32(off + 4 * i, true), F64: (i) => dv.getFloat64(off + 8 * i, true),
      F16: (i) => halfToF32(dv.getUint16(off + 2 * i, true)), BF16: (i) => { _u[0] = dv.getUint16(off + 2 * i, true) << 16; return _f[0]; },
      I64: (i) => Number(dv.getBigInt64(off + 8 * i, true)), I32: (i) => dv.getInt32(off + 4 * i, true), I16: (i) => dv.getInt16(off + 2 * i, true),
      I8: (i) => dv.getInt8(off + i), U8: (i) => dv.getUint8(off + i), BOOL: (i) => dv.getUint8(off + i),
    };
    const g = get[m.dtype];
    if (!g) throw new Error(`safetensors: unsupported dtype ${m.dtype} for ${name}`);
    for (let i = 0; i < n; i++) d[i] = g(i);
    out.set(name, { shape: m.shape.slice(), data: d });
  }
  return out;
}

// ════════════════════════════════════════ WGSL kernels ════════════════════════════════════════
// Conventions shared by every kernel and its CPU mirror:
//  - Bindings: inputs b0..b{n-1} (read-only storage), then `ob` (read_write), then the uniform `u`.
//  - Uniform: 64 u32 in array<vec4<u32>,16>. P(k) for k<4 is binding k's WINDOW START (in elements, with the output
//    binding using P(3)). Every access is `bk[globalIndex - P(k)]`, so the whole-tensor and windowed launches share
//    one code path. P(4) is the thread/row count and P(5) a global index base. Kernel params start at P(8).
//  - 1-D grids: workgroup_size 64; index = gid.y·(nwg.x·64) + gid.x (a 2-D dispatch when > 65535 groups).
const UNI = (bind: number) => `struct UB { p: array<vec4<u32>, 16> };
@group(0) @binding(${bind}) var<uniform> u: UB;
fn P(i: u32) -> u32 { return u.p[i / 4u][i % 4u]; }
fn PF(i: u32) -> f32 { return bitcast<f32>(P(i)); }
fn PI(i: u32) -> i32 { return bitcast<i32>(P(i)); }`;
function header(nIn: number, f16In: number[], f16: boolean): string {
  let s = f16 ? 'enable f16;\n' : '';
  for (let i = 0; i < nIn; i++) s += `@group(0) @binding(${i}) var<storage, read> b${i}: array<${f16 && f16In.includes(i) ? 'f16' : 'f32'}>;\n`;
  s += `@group(0) @binding(${nIn}) var<storage, read_write> ob: array<f32>;\n` + UNI(nIn + 1) + '\n';
  return s;
}
const MAIN1 = `@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * nwg.x * 64u + gid.x;
  if (i >= P(4)) { return; }`;

// erf: Eigen's float rational approximation (|x| clamped to 4, where erf is ±1 in fp32); the CPU mirror is identical.
const ERF_WGSL = `fn erf_approx(a: f32) -> f32 {
  let x = clamp(a, -4.0, 4.0); let x2 = x * x;
  var p = x2 * -2.72614225801306e-10 + 2.77068142495902e-08;
  p = x2 * p + -2.10102402082508e-06; p = x2 * p + -5.69250639462346e-05; p = x2 * p + -7.34990630326855e-04;
  p = x2 * p + -2.95459980854025e-03; p = x2 * p + -1.60960333262415e-02; p = x * p;
  var q = x2 * -1.45660718464996e-05 + -2.13374055278905e-04;
  q = x2 * q + -1.68282697438203e-03; q = x2 * q + -7.37332916720468e-03; q = x2 * q + -1.42647390514189e-02;
  return p / q;
}`;
const ERF_A = [-2.72614225801306e-10, 2.77068142495902e-08, -2.10102402082508e-06, -5.69250639462346e-05, -7.34990630326855e-04, -2.95459980854025e-03, -1.60960333262415e-02].map(fr);
const ERF_B = [-1.45660718464996e-05, -2.13374055278905e-04, -1.68282697438203e-03, -7.37332916720468e-03, -1.42647390514189e-02].map(fr);
function erfF(a: number): number {
  const x = Math.min(4, Math.max(-4, a)), x2 = fr(x * x);
  let p = fr(fr(x2 * ERF_A[0]) + ERF_A[1]);
  for (let k = 2; k < 7; k++) p = fr(fr(x2 * p) + ERF_A[k]);
  p = fr(x * p);
  let q = fr(fr(x2 * ERF_B[0]) + ERF_B[1]);
  for (let k = 2; k < 5; k++) q = fr(fr(x2 * q) + ERF_B[k]);
  return fr(p / q);
}
// Max-pool / softmax running-max seed. It must be a FINITE literal that is representable in f32: Tint (Chrome/Dawn)
// rejects -3.40282347e38 because it rounds past FLT_MAX, while naga (Deno/wgpu) accepts it. The same value is used
// in the CPU mirror.
const NEG_BIG = fr(-3.402823e38);
const TANH_CLAMP = 15; // tanh(15) == 1 in fp32; clamping keeps Metal-style tanh overflow (NaN) away

// ── conv: implicit GEMM. Params: 8 N,9 Cin,10 ID(tile),11 IH,12 IW,13 Cout,14 OD(tile),15 OH,16 OW,17-19 K,20-22 S,
//    23-25 Pad,26-28 Dil,29 G,30 hasBias,31 inZ0,32 outZ0,33 IDg (global input depth),34 mBase.
//    Bindings: b0 x, b1 w, b2 bias. Dispatch (ceil(Og/16), ceil(M/16), N·G).
const CONV = (f16: boolean) => header(3, [1], f16) + `
var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let N = P(8u); let Cin = P(9u); let ID = P(10u); let IH = P(11u); let IW = P(12u);
  let Cout = P(13u); let OD = P(14u); let OH = P(15u); let OW = P(16u);
  let KD = P(17u); let KH = P(18u); let KW = P(19u);
  let G = P(29u); let Cg = Cin / G; let Og = Cout / G;
  let K = Cg * KD * KH * KW; let M = OD * OH * OW;
  let n = wg.z / G; let g = wg.z % G;
  let row = P(34u) + wg.y * 16u + l.y;
  let col = wg.x * 16u + l.x;
  let ow = row % OW; let t1 = row / OW; let oh = t1 % OH; let od = t1 / OH;
  let ozg = od + P(32u);
  var acc = 0.0;
  let tiles = (K + 15u) / 16u;
  for (var t = 0u; t < tiles; t++) {
    let ka = t * 16u + l.x;
    var av = 0.0;
    if (row < M && ka < K) {
      let kw = ka % KW; let r1 = ka / KW; let kh = r1 % KH; let r2 = r1 / KH; let kd = r2 % KD; let ci = r2 / KD;
      let izg = i32(ozg * P(20u) + kd * P(26u)) - i32(P(23u));
      let iy = i32(oh * P(21u) + kh * P(27u)) - i32(P(24u));
      let ix = i32(ow * P(22u) + kw * P(28u)) - i32(P(25u));
      if (izg >= 0 && izg < i32(P(33u)) && iy >= 0 && iy < i32(IH) && ix >= 0 && ix < i32(IW)) {
        let iz = u32(izg) - P(31u);
        av = b0[(((n * Cin + g * Cg + ci) * ID + iz) * IH + u32(iy)) * IW + u32(ix) - P(0u)];
      }
    }
    As[l.y * 16u + l.x] = av;
    let kb = t * 16u + l.y;
    var bv = 0.0;
    if (col < Og && kb < K) { bv = f32(b1[(g * Og + col) * K + kb - P(1u)]); }
    Bs[l.y * 16u + l.x] = bv;
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) { acc = acc + As[l.y * 16u + k] * Bs[k * 16u + l.x]; }
    workgroupBarrier();
  }
  if (row < M && col < Og) {
    var v = acc;
    if (P(30u) == 1u) { v = v + b2[g * Og + col - P(2u)]; }
    ob[(n * Cout + g * Og + col) * M + row - P(3u)] = v;
  }
}`;
function cpuConv(b: Float32Array[], u: Uint32Array): void {
  const [x, w, bias, o] = b;
  const N = u[8], Cin = u[9], ID = u[10], IH = u[11], IW = u[12], Cout = u[13], OD = u[14], OH = u[15], OW = u[16];
  const KD = u[17], KH = u[18], KW = u[19], sD = u[20], sH = u[21], sW = u[22], pD = u[23], pH = u[24], pW = u[25];
  const dD = u[26], dH = u[27], dW = u[28], G = u[29], hasB = u[30], inZ0 = u[31], outZ0 = u[32], IDg = u[33], mBase = u[34];
  const Cg = Cin / G, Og = Cout / G, K = Cg * KD * KH * KW, M = OD * OH * OW;
  // k → (ci, kd, kh, kw), hoisted (pure index math; the accumulation order is still k = 0..K-1 per output)
  const kci = new Int32Array(K), kdd = new Int32Array(K), khh = new Int32Array(K), kww = new Int32Array(K);
  for (let ka = 0; ka < K; ka++) { kww[ka] = ka % KW; const r1 = (ka / KW) | 0; khh[ka] = r1 % KH; const r2 = (r1 / KH) | 0; kdd[ka] = r2 % KD; kci[ka] = (r2 / KD) | 0; }
  const acc = new Float32Array(Og);
  const rowEnd = Math.min(M, mBase + 65535 * 16);
  for (let n = 0; n < N; n++) for (let g = 0; g < G; g++) for (let row = mBase; row < rowEnd; row++) {
    const ow = row % OW, t1 = (row / OW) | 0, oh = t1 % OH, od = (t1 / OH) | 0, ozg = od + outZ0;
    acc.fill(0);
    for (let ka = 0; ka < K; ka++) {
      const izg = ozg * sD + kdd[ka] * dD - pD, iy = oh * sH + khh[ka] * dH - pH, ix = ow * sW + kww[ka] * dW - pW;
      let av = 0;
      if (izg >= 0 && izg < IDg && iy >= 0 && iy < IH && ix >= 0 && ix < IW) av = x[(((n * Cin + g * Cg + kci[ka]) * ID + (izg - inZ0)) * IH + iy) * IW + ix - u[0]];
      if (av === 0) continue; // acc + 0·w == acc exactly (w finite)
      for (let col = 0; col < Og; col++) acc[col] = fr(acc[col] + fr(av * w[(g * Og + col) * K + ka - u[1]]));
    }
    for (let col = 0; col < Og; col++) {
      let v = acc[col];
      if (hasB === 1) v = fr(v + bias[g * Og + col - u[2]]);
      o[(n * Cout + g * Og + col) * M + row - u[3]] = v;
    }
  }
}

// ── convT: gather-form transposed conv, one thread per output. Same params as conv (34 mBase unused).
//    Weight layout [Cin, Cout/G, KD, KH, KW].
const CONVT = (f16: boolean) => header(3, [1], f16) + MAIN1 + `
  let N = P(8u); let Cin = P(9u); let ID = P(10u); let IH = P(11u); let IW = P(12u);
  let Cout = P(13u); let OD = P(14u); let OH = P(15u); let OW = P(16u);
  let KD = P(17u); let KH = P(18u); let KW = P(19u);
  let sD = i32(P(20u)); let sH = i32(P(21u)); let sW = i32(P(22u));
  let G = P(29u); let Cg = Cin / G; let Og = Cout / G;
  let ox = i % OW; var r = i / OW; let oy = r % OH; r = r / OH; let oz = r % OD; r = r / OD; let co = r % Cout; let n = r / Cout;
  let g = co / Og; let col = co % Og; let ozg = oz + P(32u);
  var acc = 0.0;
  for (var ci = 0u; ci < Cg; ci++) {
    for (var kd = 0u; kd < KD; kd++) {
      let tz = i32(ozg + P(23u)) - i32(kd * P(26u));
      if (tz < 0 || (tz % sD) != 0) { continue; }
      let izg = tz / sD;
      if (izg >= i32(P(33u))) { continue; }
      for (var kh = 0u; kh < KH; kh++) {
        let ty = i32(oy + P(24u)) - i32(kh * P(27u));
        if (ty < 0 || (ty % sH) != 0) { continue; }
        let iy = ty / sH;
        if (iy >= i32(IH)) { continue; }
        for (var kw = 0u; kw < KW; kw++) {
          let tx = i32(ox + P(25u)) - i32(kw * P(28u));
          if (tx < 0 || (tx % sW) != 0) { continue; }
          let ix = tx / sW;
          if (ix >= i32(IW)) { continue; }
          let xv = b0[(((n * Cin + g * Cg + ci) * ID + (u32(izg) - P(31u))) * IH + u32(iy)) * IW + u32(ix) - P(0u)];
          let wv = f32(b1[((((g * Cg + ci) * Og + col) * KD + kd) * KH + kh) * KW + kw - P(1u)]);
          acc = acc + xv * wv;
        }
      }
    }
  }
  var v = acc;
  if (P(30u) == 1u) { v = v + b2[co - P(2u)]; }
  ob[i - P(3u)] = v;
}`;
function cpuConvT(b: Float32Array[], u: Uint32Array): void {
  const [x, w, bias, o] = b;
  const Cin = u[9], ID = u[10], IH = u[11], IW = u[12], Cout = u[13], OD = u[14], OH = u[15], OW = u[16];
  const KD = u[17], KH = u[18], KW = u[19], sD = u[20], sH = u[21], sW = u[22], pD = u[23], pH = u[24], pW = u[25];
  const dD = u[26], dH = u[27], dW = u[28], G = u[29], hasB = u[30], inZ0 = u[31], outZ0 = u[32], IDg = u[33];
  const Cg = Cin / G, Og = Cout / G, T = u[4];
  for (let i = 0; i < T; i++) {
    const ox = i % OW; let r = (i / OW) | 0; const oy = r % OH; r = (r / OH) | 0; const oz = r % OD; r = (r / OD) | 0; const co = r % Cout, n = (r / Cout) | 0;
    const g = (co / Og) | 0, col = co % Og, ozg = oz + outZ0;
    let acc = 0;
    for (let ci = 0; ci < Cg; ci++) for (let kd = 0; kd < KD; kd++) {
      const tz = ozg + pD - kd * dD;
      if (tz < 0 || tz % sD !== 0) continue;
      const izg = tz / sD;
      if (izg >= IDg) continue;
      for (let kh = 0; kh < KH; kh++) {
        const ty = oy + pH - kh * dH;
        if (ty < 0 || ty % sH !== 0) continue;
        const iy = ty / sH;
        if (iy >= IH) continue;
        for (let kw = 0; kw < KW; kw++) {
          const tx = ox + pW - kw * dW;
          if (tx < 0 || tx % sW !== 0) continue;
          const ix = tx / sW;
          if (ix >= IW) continue;
          acc = fr(acc + fr(x[(((n * Cin + g * Cg + ci) * ID + (izg - inZ0)) * IH + iy) * IW + ix - u[0]] * w[((((g * Cg + ci) * Og + col) * KD + kd) * KH + kh) * KW + kw - u[1]]));
        }
      }
    }
    o[i - u[3]] = hasB === 1 ? fr(acc + bias[co - u[2]]) : acc;
  }
}

// ── pool: 8 NC,9 ID(tile),10 IH,11 IW,12 OD(tile),13 OH,14 OW,15-17 K,18-20 S,21-23 Pad,24-26 Dil,27 mode(0 max,1 avg),
//    28 countIncludePad,29 divisorOverride(0=none),30 inZ0,31 outZ0,32 IDg.
const POOL = () => header(1, [], false) + MAIN1 + `
  let ID = P(9u); let IH = P(10u); let IW = P(11u); let OD = P(12u); let OH = P(13u); let OW = P(14u);
  let KD = P(15u); let KH = P(16u); let KW = P(17u);
  let IDg = i32(P(32u));
  let ox = i % OW; var r = i / OW; let oy = r % OH; r = r / OH; let oz = r % OD; let nc = r / OD;
  let zs = i32((oz + P(31u)) * P(18u)) - i32(P(21u));
  let ys = i32(oy * P(19u)) - i32(P(22u));
  let xs = i32(ox * P(20u)) - i32(P(23u));
  let base = nc * ID * IH * IW;
  if (P(27u) == 0u) {
    var m = -3.402823e38;
    for (var kd = 0u; kd < KD; kd++) {
      let z = zs + i32(kd * P(24u));
      if (z < 0 || z >= IDg) { continue; }
      for (var kh = 0u; kh < KH; kh++) {
        let y = ys + i32(kh * P(25u));
        if (y < 0 || y >= i32(IH)) { continue; }
        for (var kw = 0u; kw < KW; kw++) {
          let x = xs + i32(kw * P(26u));
          if (x < 0 || x >= i32(IW)) { continue; }
          let v = b0[base + ((u32(z) - P(30u)) * IH + u32(y)) * IW + u32(x) - P(0u)];
          if (v > m) { m = v; }
        }
      }
    }
    ob[i - P(3u)] = m;
  } else {
    let ze = min(zs + i32(KD), IDg + i32(P(21u))); let ye = min(ys + i32(KH), i32(IH) + i32(P(22u))); let xe = min(xs + i32(KW), i32(IW) + i32(P(23u)));
    let poolSize = (ze - zs) * (ye - ys) * (xe - xs);
    let z0 = max(zs, 0); let y0 = max(ys, 0); let x0 = max(xs, 0);
    let z1 = min(ze, IDg); let y1 = min(ye, i32(IH)); let x1 = min(xe, i32(IW));
    var s = 0.0;
    for (var z = z0; z < z1; z++) {
      for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) { s = s + b0[base + ((u32(z) - P(30u)) * IH + u32(y)) * IW + u32(x) - P(0u)]; }
      }
    }
    var dv = f32(poolSize);
    if (P(29u) != 0u) { dv = f32(P(29u)); } else if (P(28u) == 0u) { dv = f32((z1 - z0) * (y1 - y0) * (x1 - x0)); }
    ob[i - P(3u)] = s / dv;
  }
}`;
function cpuPool(b: Float32Array[], u: Uint32Array): void {
  const [x, o] = b;
  const ID = u[9], IH = u[10], IW = u[11], OD = u[12], OH = u[13], OW = u[14], KD = u[15], KH = u[16], KW = u[17];
  const sD = u[18], sH = u[19], sW = u[20], pD = u[21], pH = u[22], pW = u[23], dD = u[24], dH = u[25], dW = u[26];
  const mode = u[27], cip = u[28], div = u[29], inZ0 = u[30], outZ0 = u[31], IDg = u[32], T = u[4];
  for (let i = 0; i < T; i++) {
    const ox = i % OW; let r = (i / OW) | 0; const oy = r % OH; r = (r / OH) | 0; const oz = r % OD, nc = (r / OD) | 0;
    const zs = (oz + outZ0) * sD - pD, ys = oy * sH - pH, xs = ox * sW - pW, base = nc * ID * IH * IW;
    if (mode === 0) {
      let m = NEG_BIG;
      for (let kd = 0; kd < KD; kd++) {
        const z = zs + kd * dD; if (z < 0 || z >= IDg) continue;
        for (let kh = 0; kh < KH; kh++) {
          const y = ys + kh * dH; if (y < 0 || y >= IH) continue;
          for (let kw = 0; kw < KW; kw++) {
            const xx = xs + kw * dW; if (xx < 0 || xx >= IW) continue;
            const v = x[base + ((z - inZ0) * IH + y) * IW + xx - u[0]];
            if (v > m) m = v;
          }
        }
      }
      o[i - u[3]] = m;
    } else {
      const ze = Math.min(zs + KD, IDg + pD), ye = Math.min(ys + KH, IH + pH), xe = Math.min(xs + KW, IW + pW);
      const poolSize = (ze - zs) * (ye - ys) * (xe - xs);
      const z0 = Math.max(zs, 0), y0 = Math.max(ys, 0), x0 = Math.max(xs, 0), z1 = Math.min(ze, IDg), y1 = Math.min(ye, IH), x1 = Math.min(xe, IW);
      let s = 0;
      for (let z = z0; z < z1; z++) for (let y = y0; y < y1; y++) for (let xx = x0; xx < x1; xx++) s = fr(s + x[base + ((z - inZ0) * IH + y) * IW + xx - u[0]]);
      const dv = div !== 0 ? div : (cip === 0 ? (z1 - z0) * (y1 - y0) * (x1 - x0) : poolSize);
      o[i - u[3]] = fr(s / dv);
    }
  }
}

// ── upsample: 8 NC,9 ID(tile),10 IH,11 IW,12 OD(tile),13 OH,14 OW,15 mode(0 nearest,1 linear),16 alignCorners,
//    17-19 scale d/h/w (f32),20 inZ0,21 outZ0,22 IDg,23 ODg. PyTorch's nearest_idx / area_pixel_compute_source_index.
const UPS = () => header(1, [], false) + `
fn nidx(dst: u32, isz: u32, osz: u32, sc: f32) -> u32 {
  if (osz == isz) { return dst; }
  if (osz == 2u * isz) { return dst >> 1u; }
  return min(u32(floor(f32(dst) * sc)), isz - 1u);
}
struct LW { i0: u32, i1: u32, l0: f32, l1: f32 };
fn lidx(dst: u32, isz: u32, sc: f32, ac: u32) -> LW {
  var src: f32;
  if (ac == 1u) { src = sc * f32(dst); } else { src = max(sc * (f32(dst) + 0.5) - 0.5, 0.0); }
  let i0 = min(u32(floor(src)), isz - 1u);
  let l1 = min(max(src - f32(i0), 0.0), 1.0);
  var i1 = i0; if (i0 < isz - 1u) { i1 = i0 + 1u; }
  return LW(i0, i1, 1.0 - l1, l1);
}
` + MAIN1 + `
  let ID = P(9u); let IH = P(10u); let IW = P(11u); let OD = P(12u); let OH = P(13u); let OW = P(14u);
  let ox = i % OW; var r = i / OW; let oy = r % OH; r = r / OH; let oz = r % OD; let nc = r / OD;
  let ozg = oz + P(21u); let z0 = P(20u);
  let base = nc * ID * IH * IW - P(0u);
  if (P(15u) == 0u) {
    let iz = nidx(ozg, P(22u), P(23u), PF(17u)) - z0; let iy = nidx(oy, IH, OH, PF(18u)); let ix = nidx(ox, IW, OW, PF(19u));
    ob[i - P(3u)] = b0[base + (iz * IH + iy) * IW + ix];
  } else {
    let ac = P(16u);
    let wz = lidx(ozg, P(22u), PF(17u), ac); let wy = lidx(oy, IH, PF(18u), ac); let wx = lidx(ox, IW, PF(19u), ac);
    let za = (wz.i0 - z0) * IH; let zb = (wz.i1 - z0) * IH;
    let t00 = wx.l0 * b0[base + (za + wy.i0) * IW + wx.i0] + wx.l1 * b0[base + (za + wy.i0) * IW + wx.i1];
    let t01 = wx.l0 * b0[base + (za + wy.i1) * IW + wx.i0] + wx.l1 * b0[base + (za + wy.i1) * IW + wx.i1];
    let t10 = wx.l0 * b0[base + (zb + wy.i0) * IW + wx.i0] + wx.l1 * b0[base + (zb + wy.i0) * IW + wx.i1];
    let t11 = wx.l0 * b0[base + (zb + wy.i1) * IW + wx.i0] + wx.l1 * b0[base + (zb + wy.i1) * IW + wx.i1];
    let tz0 = wy.l0 * t00 + wy.l1 * t01;
    let tz1 = wy.l0 * t10 + wy.l1 * t11;
    ob[i - P(3u)] = wz.l0 * tz0 + wz.l1 * tz1;
  }
}`;
function nidxF(dst: number, isz: number, osz: number, sc: number): number {
  if (osz === isz) return dst;
  if (osz === 2 * isz) return dst >> 1;
  return Math.min(Math.floor(fr(dst * sc)), isz - 1);
}
function lidxF(dst: number, isz: number, sc: number, ac: number): [number, number, number, number] {
  const src = ac === 1 ? fr(sc * dst) : Math.max(fr(fr(sc * fr(dst + 0.5)) - 0.5), 0);
  const i0 = Math.min(Math.floor(src), isz - 1);
  const l1 = Math.min(Math.max(fr(src - i0), 0), 1);
  return [i0, i0 < isz - 1 ? i0 + 1 : i0, fr(1 - l1), l1];
}
function cpuUpsample(b: Float32Array[], u: Uint32Array): void {
  const [x, o] = b; const uf = new Float32Array(u.buffer, u.byteOffset, u.length);
  const ID = u[9], IH = u[10], IW = u[11], OD = u[12], OH = u[13], OW = u[14], mode = u[15], ac = u[16];
  const sd = uf[17], sh = uf[18], sw = uf[19], z0 = u[20], outZ0 = u[21], IDg = u[22], ODg = u[23], T = u[4];
  for (let i = 0; i < T; i++) {
    const ox = i % OW; let r = (i / OW) | 0; const oy = r % OH; r = (r / OH) | 0; const oz = r % OD, nc = (r / OD) | 0;
    const ozg = oz + outZ0, base = nc * ID * IH * IW - u[0];
    if (mode === 0) {
      const iz = nidxF(ozg, IDg, ODg, sd) - z0, iy = nidxF(oy, IH, OH, sh), ix = nidxF(ox, IW, OW, sw);
      o[i - u[3]] = x[base + (iz * IH + iy) * IW + ix];
    } else {
      const wz = lidxF(ozg, IDg, sd, ac), wy = lidxF(oy, IH, sh, ac), wx = lidxF(ox, IW, sw, ac);
      const za = (wz[0] - z0) * IH, zb = (wz[1] - z0) * IH;
      const L = (zz: number, yy: number) => fr(fr(wx[2] * x[base + (zz + yy) * IW + wx[0]]) + fr(wx[3] * x[base + (zz + yy) * IW + wx[1]]));
      const t00 = L(za, wy[0]), t01 = L(za, wy[1]), t10 = L(zb, wy[0]), t11 = L(zb, wy[1]);
      const tz0 = fr(fr(wy[2] * t00) + fr(wy[3] * t01)), tz1 = fr(fr(wy[2] * t10) + fr(wy[3] * t11));
      o[i - u[3]] = fr(fr(wz[2] * tz0) + fr(wz[3] * tz1));
    }
  }
}

// ── unary: 8 op, 9 alpha(f32), 10 beta(f32), 5 gBase. ops: 0 relu,1 leaky,2 sigmoid,3 tanh,4 gelu-erf,5 gelu-tanh,
//    6 silu,7 abs,8 neg,9 exp,10 sqrt,11 rsqrt,12 identity,13 affine(x·alpha+beta)
export const UNARY_OPS = { relu: 0, leaky_relu: 1, sigmoid: 2, tanh: 3, gelu: 4, gelu_tanh: 5, silu: 6, abs: 7, neg: 8, exp: 9, sqrt: 10, rsqrt: 11, identity: 12, affine: 13 } as const;
const UNARY = () => header(1, [], false) + ERF_WGSL + '\n' + MAIN1 + `
  let gi = P(5u) + i;
  let x = b0[gi - P(0u)]; let op = P(8u);
  var y = x;
  switch op {
    case 0u: { y = max(x, 0.0); if (x != x) { y = x; } }
    case 1u: { if (x > 0.0) { y = x; } else { y = x * PF(9u); } }
    case 2u: { y = 1.0 / (1.0 + exp(-x)); }
    case 3u: { y = tanh(clamp(x, -${TANH_CLAMP}.0, ${TANH_CLAMP}.0)); }
    case 4u: { y = 0.5 * x * (1.0 + erf_approx(x * 0.70710678118654752)); }
    case 5u: { let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x); y = 0.5 * x * (1.0 + tanh(clamp(inner, -${TANH_CLAMP}.0, ${TANH_CLAMP}.0))); }
    case 6u: { y = x / (1.0 + exp(-x)); }
    case 7u: { y = abs(x); }
    case 8u: { y = -x; }
    case 9u: { y = exp(x); }
    case 10u: { y = sqrt(x); }
    case 11u: { y = 1.0 / sqrt(x); }
    case 13u: { y = x * PF(9u) + PF(10u); }
    default: { y = x; }
  }
  ob[gi - P(3u)] = y;
}`;
const TC = (v: number) => Math.min(TANH_CLAMP, Math.max(-TANH_CLAMP, v));
function unaryF(op: number, x: number, a: number, bb: number): number {
  switch (op) {
    case 0: return x > 0 ? x : (x !== x ? x : 0);
    case 1: return x > 0 ? x : fr(x * a);
    case 2: return fr(1 / fr(1 + fr(Math.exp(-x))));
    case 3: return fr(Math.tanh(TC(x)));
    case 4: return fr(fr(fr(0.5 * x) * fr(1 + erfF(fr(x * fr(0.70710678118654752))))));
    case 5: { const inner = fr(fr(0.7978845608028654) * fr(x + fr(fr(fr(fr(0.044715) * x) * x) * x))); return fr(fr(0.5 * x) * fr(1 + fr(Math.tanh(TC(inner))))); }
    case 6: return fr(x / fr(1 + fr(Math.exp(-x))));
    case 7: return Math.abs(x);
    case 8: return -x;
    case 9: return fr(Math.exp(x));
    case 10: return fr(Math.sqrt(x));
    case 11: return fr(1 / fr(Math.sqrt(x)));
    case 13: return fr(fr(x * a) + bb);
    default: return x;
  }
}
function cpuUnary(b: Float32Array[], u: Uint32Array): void {
  const [x, o] = b; const uf = new Float32Array(u.buffer, u.byteOffset, u.length);
  const T = u[4], g0 = u[5], op = u[8], a = uf[9], bb = uf[10];
  for (let i = 0; i < T; i++) { const gi = g0 + i; o[gi - u[3]] = unaryF(op, x[gi - u[0]], a, bb); }
}

// ── binary (numpy broadcast, rank ≤ 6): 8 op(0 add,1 sub,2 mul,3 div,4 max,5 min), 9 alpha(f32),
//    11-16 out dims, 17-22 a strides, 23-28 b strides (0 = broadcast), 5 gBase.
const BINARY = () => header(2, [], false) + MAIN1 + `
  let gi = P(5u) + i;
  var r = gi; var ai = 0u; var bi = 0u;
  for (var d = 5i; d >= 0i; d--) {
    let dim = P(11u + u32(d)); let c = r % dim; r = r / dim;
    ai = ai + c * P(17u + u32(d)); bi = bi + c * P(23u + u32(d));
  }
  let a = b0[ai - P(0u)]; let bv = b1[bi - P(1u)];
  var y = 0.0;
  switch P(8u) {
    case 0u: { y = a + PF(9u) * bv; }
    case 1u: { y = a - PF(9u) * bv; }
    case 2u: { y = a * bv; }
    case 3u: { y = a / bv; }
    case 4u: { y = max(a, bv); }
    default: { y = min(a, bv); }
  }
  ob[gi - P(3u)] = y;
}`;
function cpuBinary(b: Float32Array[], u: Uint32Array): void {
  const [A, B, o] = b; const uf = new Float32Array(u.buffer, u.byteOffset, u.length);
  const T = u[4], g0 = u[5], op = u[8], al = uf[9];
  for (let i = 0; i < T; i++) {
    const gi = g0 + i; let r = gi, ai = 0, bi = 0;
    for (let d = 5; d >= 0; d--) { const dim = u[11 + d], c = r % dim; r = (r / dim) | 0; ai += c * u[17 + d]; bi += c * u[23 + d]; }
    const a = A[ai - u[0]], bv = B[bi - u[1]];
    let y: number;
    switch (op) {
      case 0: y = fr(a + fr(al * bv)); break;
      case 1: y = fr(a - fr(al * bv)); break;
      case 2: y = fr(a * bv); break;
      case 3: y = fr(a / bv); break;
      case 4: y = Math.max(a, bv); break;
      default: y = Math.min(a, bv);
    }
    o[gi - u[3]] = y;
  }
}

// ── chanop (per-channel, channel = (gi / inner) % C): 8 C, 9 inner, 10 mode, 5 gBase.
//    mode 0 affine y = x·s[c] + t[c] (folded BN) · 1 prelu (x>0 ? x : s[c]·x) · 2 explicit BN: s=[mean|invstd], t=[w|b].
const CHANOP = () => header(3, [], false) + MAIN1 + `
  let gi = P(5u) + i; let C = P(8u);
  let c = (gi / P(9u)) % C;
  let x = b0[gi - P(0u)];
  var y = 0.0;
  if (P(10u) == 0u) { y = x * b1[c - P(1u)] + b2[c - P(2u)]; }
  else if (P(10u) == 1u) { if (x > 0.0) { y = x; } else { y = b1[c - P(1u)] * x; } }
  else { y = (x - b1[c - P(1u)]) * b1[C + c - P(1u)] * b2[c - P(2u)] + b2[C + c - P(2u)]; }
  ob[gi - P(3u)] = y;
}`;
function cpuChanop(b: Float32Array[], u: Uint32Array): void {
  const [x, s, t, o] = b; const T = u[4], g0 = u[5], C = u[8], inner = u[9], mode = u[10];
  for (let i = 0; i < T; i++) {
    const gi = g0 + i, c = ((gi / inner) | 0) % C, v = x[gi - u[0]];
    let y: number;
    if (mode === 0) y = fr(fr(v * s[c - u[1]]) + t[c - u[2]]);
    else if (mode === 1) y = v > 0 ? v : fr(s[c - u[1]] * v);
    else y = fr(fr(fr(fr(v - s[c - u[1]]) * s[C + c - u[1]]) * t[c - u[2]]) + t[C + c - u[2]]);
    o[gi - u[3]] = y;
  }
}

// ── norm (one 256-thread workgroup per row): 4 rows, 8 L, 9 inner, 10 cpg, 11 G, 12 eps(f32), 13 affine
//    (0 none, 1 per-channel c = g·cpg + j/inner, 2 per-element j), 14 rowBase. Two-pass mean/var, tree-reduced.
const NORM = () => header(3, [], false) + `
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let lr = wg.y * nwg.x + wg.x;
  if (lr >= P(4u)) { return; }
  let r = P(14u) + lr; let L = P(8u); let base = r * L; let tid = l.x;
  var s = 0.0;
  for (var j = tid; j < L; j += 256u) { s = s + b0[base + j - P(0u)]; }
  red[tid] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k = k >> 1u) { if (tid < k) { red[tid] = red[tid] + red[tid + k]; } workgroupBarrier(); }
  let mean = red[0] / f32(L);
  workgroupBarrier();
  var v = 0.0;
  for (var j = tid; j < L; j += 256u) { let d = b0[base + j - P(0u)] - mean; v = v + d * d; }
  red[tid] = v;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k = k >> 1u) { if (tid < k) { red[tid] = red[tid] + red[tid + k]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(red[0] / f32(L) + PF(12u));
  let g = r % P(11u);
  for (var j = tid; j < L; j += 256u) {
    var y = (b0[base + j - P(0u)] - mean) * inv;
    if (P(13u) == 1u) { let c = g * P(10u) + j / P(9u); y = y * b1[c - P(1u)] + b2[c - P(2u)]; }
    else if (P(13u) == 2u) { y = y * b1[j - P(1u)] + b2[j - P(2u)]; }
    ob[base + j - P(3u)] = y;
  }
}`;
function treeReduce(red: Float32Array): number {
  for (let k = 128; k > 0; k >>= 1) for (let t = 0; t < k; t++) red[t] = fr(red[t] + red[t + k]);
  return red[0];
}
function cpuNorm(b: Float32Array[], u: Uint32Array): void {
  const [x, w, bb, o] = b; const uf = new Float32Array(u.buffer, u.byteOffset, u.length);
  const rows = u[4], L = u[8], inner = u[9], cpg = u[10], G = u[11], eps = uf[12], aff = u[13], r0 = u[14];
  const red = new Float32Array(256);
  for (let lr = 0; lr < rows; lr++) {
    const r = r0 + lr, base = r * L - u[0];
    for (let t = 0; t < 256; t++) { let s = 0; for (let j = t; j < L; j += 256) s = fr(s + x[base + j]); red[t] = s; }
    const mean = fr(treeReduce(red) / L);
    for (let t = 0; t < 256; t++) { let v = 0; for (let j = t; j < L; j += 256) { const d = fr(x[base + j] - mean); v = fr(v + fr(d * d)); } red[t] = v; }
    const inv = fr(1 / fr(Math.sqrt(fr(fr(treeReduce(red) / L) + eps))));
    const g = r % G, ob = r * L - u[3];
    for (let j = 0; j < L; j++) {
      let y = fr(fr(x[base + j] - mean) * inv);
      if (aff === 1) { const c = g * cpg + ((j / inner) | 0); y = fr(fr(y * w[c - u[1]]) + bb[c - u[2]]); }
      else if (aff === 2) y = fr(fr(y * w[j - u[1]]) + bb[j - u[2]]);
      o[ob + j] = y;
    }
  }
}

// ── linered (one thread per line along a dim of size S with stride `inner`): 8 S, 9 inner, 10 mode
//    (0 softmax, 1 argmax (first max, f32 index), 2 mean, 3 sum), 5 lineBase. Line li → outer = li / inner, q = li % inner.
const LINERED = () => header(1, [], false) + MAIN1 + `
  let li = P(5u) + i; let S = P(8u); let inner = P(9u);
  let outer = li / inner; let q = li % inner;
  let base = outer * S * inner + q;
  let mode = P(10u);
  if (mode == 0u) {
    var m = -3.402823e38;
    for (var k = 0u; k < S; k++) { m = max(m, b0[base + k * inner - P(0u)]); }
    var s = 0.0;
    for (var k = 0u; k < S; k++) { s = s + exp(b0[base + k * inner - P(0u)] - m); }
    for (var k = 0u; k < S; k++) { ob[base + k * inner - P(3u)] = exp(b0[base + k * inner - P(0u)] - m) / s; }
  } else if (mode == 1u) {
    var best = b0[base - P(0u)]; var bi = 0u;
    for (var k = 1u; k < S; k++) { let v = b0[base + k * inner - P(0u)]; if (v > best) { best = v; bi = k; } }
    ob[li - P(3u)] = f32(bi);
  } else {
    var s = 0.0;
    for (var k = 0u; k < S; k++) { s = s + b0[base + k * inner - P(0u)]; }
    if (mode == 2u) { s = s / f32(S); }
    ob[li - P(3u)] = s;
  }
}`;
function cpuLinered(b: Float32Array[], u: Uint32Array): void {
  const [x, o] = b; const T = u[4], l0 = u[5], S = u[8], inner = u[9], mode = u[10];
  for (let i = 0; i < T; i++) {
    const li = l0 + i, outer = (li / inner) | 0, q = li % inner, base = outer * S * inner + q - u[0], ob = outer * S * inner + q - u[3];
    if (mode === 0) {
      let m = NEG_BIG;
      for (let k = 0; k < S; k++) m = Math.max(m, x[base + k * inner]);
      let s = 0;
      for (let k = 0; k < S; k++) s = fr(s + fr(Math.exp(fr(x[base + k * inner] - m))));
      for (let k = 0; k < S; k++) o[ob + k * inner] = fr(fr(Math.exp(fr(x[base + k * inner] - m))) / s);
    } else if (mode === 1) {
      let best = x[base], bi = 0;
      for (let k = 1; k < S; k++) { const v = x[base + k * inner]; if (v > best) { best = v; bi = k; } }
      o[li - u[3]] = bi;
    } else {
      let s = 0;
      for (let k = 0; k < S; k++) s = fr(s + x[base + k * inner]);
      o[li - u[3]] = mode === 2 ? fr(s / S) : s;
    }
  }
}

// ── matmul (tiled 16×16, from worker.ts WGSL.matmul): 8 rowEnd, 9 N, 10 K, 11 transB (B is [N,K]), 12 aBatchStride,
//    13 bBatchStride, 14 hasBias, 15 alpha(f32), 16 beta(f32), 17 rowBase, 18 batches (grid z), 19 outBatchStride, 20 batchBase.
//    Bindings b0 A, b1 B, b2 bias[N]. Dispatch (ceil(N/16), ceil(rows/16), batches).
const MATMUL = (f16: boolean) => header(3, [1], f16) + `
var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let rowEnd = P(8u); let N = P(9u); let K = P(10u);
  let bIdx = P(20u) + wg.z;
  let row = P(17u) + wg.y * 16u + l.y; let col = wg.x * 16u + l.x;
  let aB = bIdx * P(12u); let bB = bIdx * P(13u);
  var acc = 0.0;
  let tiles = (K + 15u) / 16u;
  for (var t = 0u; t < tiles; t++) {
    let aCol = t * 16u + l.x; let bRow = t * 16u + l.y;
    var av = 0.0;
    if (row < rowEnd && aCol < K) { av = b0[aB + row * K + aCol - P(0u)]; }
    var bv = 0.0;
    if (bRow < K && col < N) {
      if (P(11u) == 1u) { bv = f32(b1[bB + col * K + bRow - P(1u)]); } else { bv = f32(b1[bB + bRow * N + col - P(1u)]); }
    }
    As[l.y * 16u + l.x] = av; Bs[l.y * 16u + l.x] = bv;
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) { acc = acc + As[l.y * 16u + k] * Bs[k * 16u + l.x]; }
    workgroupBarrier();
  }
  if (row < rowEnd && col < N) {
    var v = acc * PF(15u);
    if (P(14u) == 1u) { v = v + PF(16u) * b2[col - P(2u)]; }
    ob[bIdx * P(19u) + row * N + col - P(3u)] = v;
  }
}`;
function cpuMatmul(b: Float32Array[], u: Uint32Array): void {
  const [A, B, bias, o] = b; const uf = new Float32Array(u.buffer, u.byteOffset, u.length);
  const rowEnd = u[8], N = u[9], K = u[10], tB = u[11], aBS = u[12], bBS = u[13], hasB = u[14], al = uf[15], be = uf[16];
  const r0 = u[17], nb = u[18], oBS = u[19], b0 = u[20];
  const acc = new Float32Array(N);
  for (let bi = b0; bi < b0 + nb; bi++) for (let row = r0; row < rowEnd; row++) {
    acc.fill(0);
    const aOff = bi * aBS + row * K - u[0], bOff = bi * bBS - u[1];
    for (let k = 0; k < K; k++) {
      const av = A[aOff + k];
      if (av === 0) continue;
      if (tB === 1) for (let c = 0; c < N; c++) acc[c] = fr(acc[c] + fr(av * B[bOff + c * K + k]));
      else for (let c = 0; c < N; c++) acc[c] = fr(acc[c] + fr(av * B[bOff + k * N + c]));
    }
    for (let c = 0; c < N; c++) {
      let v = fr(acc[c] * al);
      if (hasB === 1) v = fr(v + fr(be * bias[c - u[2]]));
      o[bi * oBS + row * N + c - u[3]] = v;
    }
  }
}

// ── attention (online softmax, extended from SHARD_WGSL.cachedAttn): threads = BH·Sq (4), 5 threadBase, 8 Sq, 9 Sk,
//    10 D, 11 Dv, 12 scale(f32), 13 causal (j ≤ i). q/k/v/out are [BH, S, D]. D, Dv ≤ 256.
const ATTN = () => header(3, [], false) + MAIN1 + `
  let t = P(5u) + i; let Sq = P(8u); let Sk = P(9u); let D = P(10u); let Dv = P(11u);
  let bh = t / Sq; let qi = t % Sq;
  let qb = (bh * Sq + qi) * D - P(0u);
  var m = -3.0e38; var lsum = 0.0;
  var acc: array<f32, 256>;
  for (var d = 0u; d < Dv; d++) { acc[d] = 0.0; }
  var lim = Sk; if (P(13u) == 1u) { lim = min(Sk, qi + 1u); }
  for (var j = 0u; j < lim; j++) {
    let kb = (bh * Sk + j) * D - P(1u);
    var dot = 0.0;
    for (var d = 0u; d < D; d++) { dot = dot + b0[qb + d] * b1[kb + d]; }
    let s = dot * PF(12u);
    let nm = max(m, s); let corr = exp(m - nm); let p = exp(s - nm);
    lsum = lsum * corr + p;
    let vb = (bh * Sk + j) * Dv - P(2u);
    for (var d = 0u; d < Dv; d++) { acc[d] = acc[d] * corr + p * b2[vb + d]; }
    m = nm;
  }
  let obase = (bh * Sq + qi) * Dv - P(3u);
  for (var d = 0u; d < Dv; d++) { ob[obase + d] = acc[d] / lsum; }
}`;
function cpuAttn(b: Float32Array[], u: Uint32Array): void {
  const [q, k, v, o] = b; const uf = new Float32Array(u.buffer, u.byteOffset, u.length);
  const T = u[4], t0 = u[5], Sq = u[8], Sk = u[9], D = u[10], Dv = u[11], sc = uf[12], causal = u[13];
  const acc = new Float32Array(Dv);
  for (let i = 0; i < T; i++) {
    const t = t0 + i, bh = (t / Sq) | 0, qi = t % Sq, qb = (bh * Sq + qi) * D - u[0];
    let m = -3.0e38, l = 0;
    acc.fill(0);
    const lim = causal === 1 ? Math.min(Sk, qi + 1) : Sk;
    for (let j = 0; j < lim; j++) {
      const kb = (bh * Sk + j) * D - u[1];
      let dot = 0;
      for (let d = 0; d < D; d++) dot = fr(dot + fr(q[qb + d] * k[kb + d]));
      const s = fr(dot * sc), nm = Math.max(m, s), corr = fr(Math.exp(fr(m - nm))), p = fr(Math.exp(fr(s - nm)));
      l = fr(fr(l * corr) + p);
      const vb = (bh * Sk + j) * Dv - u[2];
      for (let d = 0; d < Dv; d++) acc[d] = fr(fr(acc[d] * corr) + fr(p * v[vb + d]));
      m = nm;
    }
    const ob = (bh * Sq + qi) * Dv - u[3];
    for (let d = 0; d < Dv; d++) o[ob + d] = fr(acc[d] / l);
  }
}

// ── copy (strided gather, rank ≤ 6): 11-16 iteration dims, 17-22 in strides, 23 inOff, 24-29 out strides, 30 outOff, 5 gBase.
const COPY = () => header(1, [], false) + MAIN1 + `
  let gi = P(5u) + i;
  var r = gi; var ii = P(23u); var oi = P(30u);
  for (var d = 5i; d >= 0i; d--) {
    let dim = P(11u + u32(d)); let c = r % dim; r = r / dim;
    ii = ii + c * P(17u + u32(d)); oi = oi + c * P(24u + u32(d));
  }
  ob[oi - P(3u)] = b0[ii - P(0u)];
}`;
function cpuCopy(b: Float32Array[], u: Uint32Array): void {
  const [x, o] = b; const T = u[4], g0 = u[5];
  for (let i = 0; i < T; i++) {
    let r = g0 + i, ii = u[23], oi = u[30];
    for (let d = 5; d >= 0; d--) { const dim = u[11 + d], c = r % dim; r = (r / dim) | 0; ii += c * u[17 + d]; oi += c * u[24 + d]; }
    o[oi - u[3]] = x[ii - u[0]];
  }
}

// ── pad (rank ≤ 6): 11-16 out dims, 17-22 in dims, 23-28 padBefore (i32), 29 mode (0 constant,1 replicate,2 reflect),
//    30 value(f32), 5 gBase.
const PAD = () => header(1, [], false) + MAIN1 + `
  let gi = P(5u) + i;
  var r = gi; var ii = 0u; var stride = 1u; var inside = true;
  for (var d = 5i; d >= 0i; d--) {
    let od = P(11u + u32(d)); let idim = i32(P(17u + u32(d)));
    let c = i32(r % od); r = r / od;
    var s = c - PI(23u + u32(d));
    if (s < 0 || s >= idim) {
      if (P(29u) == 0u) { inside = false; s = 0; }
      else if (P(29u) == 1u) { s = clamp(s, 0, idim - 1); }
      else { if (s < 0) { s = -s; } if (s >= idim) { s = 2 * (idim - 1) - s; } }
    }
    ii = ii + u32(s) * stride; stride = stride * u32(idim);
  }
  if (inside) { ob[gi - P(3u)] = b0[ii - P(0u)]; } else { ob[gi - P(3u)] = PF(30u); }
}`;
function cpuPad(b: Float32Array[], u: Uint32Array): void {
  const [x, o] = b; const uf = new Float32Array(u.buffer, u.byteOffset, u.length); const T = u[4], g0 = u[5], mode = u[29];
  for (let i = 0; i < T; i++) {
    let r = g0 + i, ii = 0, stride = 1, inside = true;
    for (let d = 5; d >= 0; d--) {
      const od = u[11 + d], idim = u[17 + d], c = r % od; r = (r / od) | 0;
      let s = c - (u[23 + d] | 0);
      if (s < 0 || s >= idim) {
        if (mode === 0) { inside = false; s = 0; } else if (mode === 1) s = Math.min(Math.max(s, 0), idim - 1);
        else { if (s < 0) s = -s; if (s >= idim) s = 2 * (idim - 1) - s; }
      }
      ii += s * stride; stride *= idim;
    }
    o[g0 + i - u[3]] = inside ? x[ii - u[0]] : uf[30];
  }
}

export const WGSL_KERNELS: Record<string, (f16: boolean) => string> = {
  conv: CONV, convT: CONVT, pool: POOL, upsample: UPS, unary: UNARY, binary: BINARY, chanop: CHANOP, norm: NORM,
  linered: LINERED, matmul: MATMUL, attention: ATTN, copy: COPY, pad: PAD,
};
export const CPU_KERNELS: Record<string, (b: Float32Array[], u: Uint32Array) => void> = {
  conv: cpuConv, convT: cpuConvT, pool: cpuPool, upsample: cpuUpsample, unary: cpuUnary, binary: cpuBinary, chanop: cpuChanop,
  norm: cpuNorm, linered: cpuLinered, matmul: cpuMatmul, attention: cpuAttn, copy: cpuCopy, pad: cpuPad,
};
/** Kernels with an f16-weight-storage variant (binding 1 = weights as array<f16>). */
export const F16_KERNELS = ['conv', 'convT', 'matmul'];
export function wgslSource(kernel: string, f16: boolean): string {
  const k = WGSL_KERNELS[kernel];
  if (!k) throw new Error(`unknown kernel ${kernel}`);
  if (f16 && !F16_KERNELS.includes(kernel)) throw new Error(`kernel ${kernel} has no f16 variant`);
  return k(f16);
}
// ════════════════════════════════════════ op table ════════════════════════════════════════
/** The op names this executor accepts (base names; overload suffixes are stripped). MUST equal the keys of
 *  apps/worker/vision_ops.json (a test pins it), which is the contract with the Python lowering. */
export const SUPPORTED_OPS: readonly string[] = [
  'aten.convolution', 'aten.conv2d', 'aten.conv3d', 'aten.conv_transpose2d', 'aten.conv_transpose3d',
  'aten.linear', 'aten.addmm', 'aten.mm', 'aten.bmm', 'aten.matmul', 'aten.scaled_dot_product_attention',
  'aten.batch_norm', 'aten._native_batch_norm_legit_no_training', 'aten.instance_norm', 'aten.group_norm', 'aten.native_group_norm',
  'aten.layer_norm', 'aten.native_layer_norm',
  'aten.max_pool2d', 'aten.max_pool3d', 'aten.max_pool2d_with_indices', 'aten.max_pool3d_with_indices', 'aten.avg_pool2d', 'aten.avg_pool3d',
  'aten.adaptive_avg_pool2d', 'aten.adaptive_avg_pool3d',
  'aten.upsample_nearest2d', 'aten.upsample_nearest3d', 'aten.upsample_bilinear2d', 'aten.upsample_trilinear3d',
  'aten.cat', 'aten.add', 'aten.sub', 'aten.mul', 'aten.div',
  'aten.relu', 'aten.relu_', 'aten.leaky_relu', 'aten.leaky_relu_', 'aten.gelu', 'aten.sigmoid', 'aten.tanh', 'aten.silu',
  'aten.abs', 'aten.neg', 'aten.exp', 'aten.sqrt', 'aten.rsqrt', 'aten.prelu',
  'aten._softmax', 'aten.softmax', 'aten.argmax', 'aten.mean',
  'aten.view', 'aten.reshape', 'aten._unsafe_view', 'aten.flatten', 'aten.unsqueeze', 'aten.squeeze', 'aten.contiguous', 'aten.clone',
  'aten.alias', 'aten.detach', 'aten.dropout',
  'aten.permute', 'aten.transpose', 'aten.t', 'aten.expand', 'aten.select', 'aten.slice', 'aten.pad', 'aten.constant_pad_nd',
];
const SUPPORTED = new Set(SUPPORTED_OPS);
/** 'aten.add.Tensor' → 'aten.add'; 'aten.relu' stays. */
export function baseOp(op: string): string { const p = op.split('.'); return p.length > 2 ? `${p[0]}.${p[1]}` : op; }

// ════════════════════════════════════════ compiler ════════════════════════════════════════
type VKind = 'input' | 'const' | 'act' | 'scratch';
interface Val { name: string; shape: number[]; kind: VKind; root: string; data?: Float32Array; f16?: boolean }
export interface Bind { value: string; start: number; len: number }
export type Launch =
  | { kind: 'kernel'; kernel: string; f16: boolean; binds: Bind[]; u: Uint32Array; grid: [number, number, number]; node: number }
  | { kind: 'copy'; src: string; srcOff: number; dst: string; dstOff: number; len: number; node: number };
export interface MemoryPlan {
  maxBindingBytes: number;
  maxBufferBytes: number;
  /** bytes of each pooled buffer (slot) */
  slots: number[];
  totalBytes: number;
  /** what one-buffer-per-value would cost */
  naiveBytes: number;
  peakLiveBytes: number;
  constBytes: number;
  launches: number;
  copies: number;
  /** per value: slot, lifetime [def, lastUse] in launch indices (inputs def = -1, graph outputs lastUse = ∞) */
  values: Record<string, { slot: number; def: number; lastUse: number; bytes: number; aliasOf?: string }>;
  tiled: { node: string; op: string; strategy: 'spatial-slab' | 'flat' | 'rows' | 'inner-chunk'; tiles: number }[];
  bindings: { value: string; bytes: number }[];
}

class UB {
  u = new Uint32Array(64);
  f = new Float32Array(this.u.buffer);
  s(i: number, v: number): this { this.u[i] = v >>> 0; return this; }
  si(i: number, v: number): this { this.u[i] = (v | 0) >>> 0; return this; }
  sf(i: number, v: number): this { this.f[i] = v; return this; }
}
const MAXG = 65535;
function grid1(T: number): [number, number, number] { const g = Math.ceil(T / 64); const x = Math.min(g, MAXG); return [x, Math.ceil(g / x), 1]; }
const contig = (s: number[]) => { const st = new Array(s.length).fill(1); for (let i = s.length - 2; i >= 0; i--) st[i] = st[i + 1] * s[i + 1]; return st; };
const pad6 = (a: number[], fill: number) => [...new Array(6 - a.length).fill(fill), ...a];
function A<T>(n: OpNode, k: string, d: T): T { const v = n.attrs?.[k]; return (v === undefined || v === null ? d : v) as T; }
const normDim = (d: number, r: number) => (d < 0 ? d + r : d);
const listN = (v: unknown, n: number, d: number): number[] => {
  if (v === undefined || v === null) return new Array(n).fill(d);
  if (typeof v === 'number') return new Array(n).fill(v);
  const a = (v as number[]).map(Number);
  if (a.length === 0) return new Array(n).fill(d);
  return a.length === 1 ? new Array(n).fill(a[0]) : a;
};

class Builder {
  vals = new Map<string, Val>();
  launches: Launch[] = [];
  tiled: MemoryPlan['tiled'] = [];
  bindings: MemoryPlan['bindings'] = [];
  node = -1; nodeName = ''; nodeOp = '';
  private nScratch = 0;
  constructor(public weights: Map<string, Tensor>, public lim: number, public opts: CompileOptions) {}

  val(name: string | null | undefined): Val {
    if (name === null || name === undefined) throw new Error(`${this.nodeOp}: missing required input`);
    const v = this.vals.get(name);
    if (v) return v;
    const w = this.weights.get(name);
    if (w) { const c: Val = { name, shape: w.shape.slice(), kind: 'const', root: name, data: w.data }; this.vals.set(name, c); return c; }
    throw new Error(`unknown value '${name}' (not a graph input, earlier node output or weight) in ${this.nodeOp}`);
  }
  opt(name: string | null | undefined): Val | null { return name === null || name === undefined ? null : this.val(name); }
  act(name: string, shape: number[]): Val {
    if (this.vals.has(name)) throw new Error(`value '${name}' defined twice`);
    const v: Val = { name, shape, kind: 'act', root: name }; this.vals.set(name, v); return v;
  }
  scratch(n: number): Val { const name = `__tile${this.nScratch++}`; const v: Val = { name, shape: [n], kind: 'scratch', root: name }; this.vals.set(name, v); return v; }
  konst(name: string, data: Float32Array, shape: number[], f16 = false): Val {
    const ex = this.vals.get(name); if (ex) return ex;
    const v: Val = { name, shape, kind: 'const', root: name, data: f16 ? f16BitsToF32(f32ToF16Bits(data)) : data, f16 };
    this.vals.set(name, v); return v;
  }
  zero1(): Val { return this.konst('__zero1', new Float32Array(1), [1]); }
  /** weight operand for conv/convT/matmul: an f16-rounded twin when f16 weights are on */
  weight(v: Val): Val { return this.opts.f16Weights && v.kind === 'const' ? this.konst(`${v.name}::f16`, v.data!, v.shape, true) : v; }
  alias(name: string, src: Val, shape: number[]): Val {
    if (numel(shape) !== numel(src.shape)) throw new Error(`${this.nodeOp}: view ${src.shape} → ${shape} changes numel`);
    const v: Val = src.kind === 'const' ? { name, shape, kind: 'const', root: name, data: src.data } : { name, shape, kind: src.kind, root: src.root };
    this.vals.set(name, v); return v;
  }
  esz(value: string): number { return this.vals.get(value)!.f16 ? 2 : 4; }
  bytesOf(b: Bind): number { return Math.ceil((b.len * this.esz(b.value)) / 4) * 4; }
  win(v: Val, start: number, end: number): Bind { const s = alignDown(start); return { value: v.name, start: s, len: Math.max(1, end - s) }; }
  whole(v: Val): Bind { return { value: v.name, start: 0, len: Math.max(1, numel(v.shape)) }; }
  fits(...bs: Bind[]): boolean { return bs.every((b) => this.bytesOf(b) <= this.lim); }
  kernel(kernel: string, binds: Bind[], ub: UB, grid: [number, number, number], f16 = false): void {
    if (grid[0] * grid[1] * grid[2] === 0) return;
    for (const b of binds) {
      const bytes = this.bytesOf(b);
      if (bytes > this.lim) throw new Error(`${this.nodeOp} '${this.nodeName}': binding '${b.value}' of ${bytes} B exceeds maxStorageBufferBindingSize (${this.lim} B) and this operand cannot be tiled further`);
      this.bindings.push({ value: b.value, bytes });
    }
    const u = new Uint32Array(ub.u);
    for (let k = 0; k < binds.length - 1; k++) u[k] = binds[k].start;
    u[3] = binds[binds.length - 1].start;
    this.launches.push({ kind: 'kernel', kernel, f16, binds, u, grid, node: this.node });
  }
  copy(src: Val, srcOff: number, dst: Val, dstOff: number, len: number): void {
    if (len > 0) this.launches.push({ kind: 'copy', src: src.name, srcOff, dst: dst.name, dstOff, len, node: this.node });
  }
  tile(strategy: MemoryPlan['tiled'][number]['strategy'], tiles: number): void { this.tiled.push({ node: this.nodeName, op: this.nodeOp, strategy, tiles }); }
  refuse(what: string): never { throw new Error(`${this.nodeOp} '${this.nodeName}': ${what} exceeds maxStorageBufferBindingSize (${this.lim} B) even after tiling`); }

  // ── tiling drivers ──
  /** Element-wise over `total` output elements. `bindsFor(s,e)` returns windows (input windows first, output last). */
  flat(total: number, bindsFor: (s: number, e: number) => Bind[], mk: (s: number, e: number) => UB, kernel: string): void {
    const all = bindsFor(0, total);
    if (this.fits(...all)) { this.kernel(kernel, all, mk(0, total), grid1(total)); return; }
    const E = Math.floor(this.lim / 4 / ALIGN) * ALIGN;
    if (E < ALIGN) this.refuse('one 64-element chunk');
    let n = 0;
    for (let s = 0; s < total; s += E) { const e = Math.min(total, s + E); this.kernel(kernel, bindsFor(s, e), mk(s, e), grid1(e - s)); n++; }
    this.tile('flat', n);
  }
  /** Split `units` contiguous rows across launches. Each entry of `win` is a windowed binding: value, element offset of
   *  unit 0, elements per unit. `build(u0, u1, windows)` emits the launch(es) for units [u0, u1). */
  rows(units: number, win: { v: Val; off: number; ue: number }[], build: (u0: number, u1: number, w: Bind[]) => void): void {
    const wAll = win.map((w) => this.win(w.v, w.off, w.off + units * w.ue));
    if (this.fits(...wAll)) { build(0, units, wAll); return; }
    let per = units;
    for (const w of win) per = Math.min(per, Math.floor((this.lim / 4 - (ALIGN - 1)) / Math.max(1, w.ue)));
    if (per < 1) this.refuse('a single row/unit');
    let n = 0;
    for (let u0 = 0; u0 < units; u0 += per) {
      const u1 = Math.min(units, u0 + per);
      build(u0, u1, win.map((w) => this.win(w.v, w.off + u0 * w.ue, w.off + u1 * w.ue)));
      n++;
    }
    this.tile('rows', n);
  }
}

interface Sp { N: number; C: number; D: number; H: number; W: number }
/** [N,C,H,W] → D=H, H=W, W=1 (2D embedded in 3D); [N,C,D,H,W] as is. */
function sp3(shape: number[]): Sp {
  if (shape.length === 4) return { N: shape[0], C: shape[1], D: shape[2], H: shape[3], W: 1 };
  if (shape.length === 5) return { N: shape[0], C: shape[1], D: shape[2], H: shape[3], W: shape[4] };
  throw new Error(`expected a 4-D or 5-D tensor, got [${shape}]`);
}
const to3 = (a: number[], fill: number) => (a.length === 2 ? [a[0], a[1], fill] : a.slice(0, 3));

export class VisionModel {
  private constructor(
    public readonly graph: OpGraph,
    public readonly inputs: { name: string; shape: number[] }[],
    public readonly outputs: string[],
    public readonly launches: Launch[],
    public readonly plan: MemoryPlan,
    public readonly f16Weights: boolean,
    /** @internal */ public readonly vals: Map<string, Val>,
    /** @internal */ public readonly slotOf: Map<string, number>,
  ) {}

  static compile(graph: OpGraph, weights: Map<string, Tensor>, opts: CompileOptions = {}): VisionModel {
    if (!graph || graph.version !== 1) throw new Error(`op-graph version must be 1 (got ${graph?.version})`);
    if (!Array.isArray(graph.inputs) || !Array.isArray(graph.nodes) || !Array.isArray(graph.outputs)) throw new Error('op-graph needs inputs/nodes/outputs arrays');
    const bad = [...new Set(graph.nodes.map((n) => baseOp(n.op)).filter((o) => !SUPPORTED.has(o)))];
    if (bad.length) throw new Error(`unsupported op(s): ${bad.join(', ')} — see apps/worker/vision_ops.json`);
    const lim = opts.maxBindingBytes ?? 128 * 1024 * 1024, maxBuf = opts.maxBufferBytes ?? 256 * 1024 * 1024;
    const B = new Builder(weights, lim, opts);
    for (const i of graph.inputs) B.vals.set(i.name, { name: i.name, shape: i.shape.slice(), kind: 'input', root: i.name });
    graph.nodes.forEach((nd, idx) => { B.node = idx; B.nodeName = nd.output; B.nodeOp = baseOp(nd.op); lowerNode(B, nd); });
    for (const o of graph.outputs) B.val(o);
    const { plan, slotOf } = planMemory(B, graph, lim, maxBuf);
    return new VisionModel(graph, graph.inputs.map((i) => ({ name: i.name, shape: i.shape.slice() })), graph.outputs.slice(), B.launches, plan, !!opts.f16Weights, B.vals, slotOf);
  }
  shapeOf(name: string): number[] { return this.vals.get(name)!.shape.slice(); }
}

// ── lowering ──
function lowerNode(B: Builder, nd: OpNode): void {
  const op = baseOp(nd.op), ins = nd.inputs ?? [], out = nd.output;
  const X = () => B.val(ins[0]);
  switch (op) {
    // views
    case 'aten.view': case 'aten.reshape': case 'aten._unsafe_view': {
      const x = X(); const want = (A<number[]>(nd, 'size', A<number[]>(nd, 'shape', []))).map(Number);
      const known = want.filter((d) => d !== -1).reduce((a, b) => a * b, 1);
      B.alias(out, x, want.map((d) => (d === -1 ? numel(x.shape) / known : d))); return;
    }
    case 'aten.flatten': {
      const x = X(), r = x.shape.length; if (r === 0) { B.alias(out, x, [1]); return; }
      const s = normDim(A(nd, 'start_dim', 0), r), e = normDim(A(nd, 'end_dim', -1), r);
      B.alias(out, x, [...x.shape.slice(0, s), numel(x.shape.slice(s, e + 1)), ...x.shape.slice(e + 1)]); return;
    }
    case 'aten.unsqueeze': { const x = X(); const d = normDim(A(nd, 'dim', 0), x.shape.length + 1); const s = x.shape.slice(); s.splice(d, 0, 1); B.alias(out, x, s); return; }
    case 'aten.squeeze': {
      const x = X(), r = x.shape.length; const dv = nd.attrs?.dim;
      const dims = dv === undefined || dv === null ? x.shape.map((_, i) => i) : (Array.isArray(dv) ? dv : [dv]).map((d) => normDim(Number(d), r));
      B.alias(out, x, x.shape.filter((s, i) => !(dims.includes(i) && s === 1))); return;
    }
    case 'aten.contiguous': case 'aten.clone': case 'aten.alias': case 'aten.detach': case 'aten.dropout': {
      if (op === 'aten.dropout' && A(nd, 'train', false) && Number(A(nd, 'p', 0)) > 0) throw new Error('aten.dropout with train=true is not inference');
      const x = X(); B.alias(out, x, x.shape.slice()); return;
    }
    // strided copies
    case 'aten.permute': { const x = X(); permute(B, x, out, (A<number[]>(nd, 'dims', [])).map((d) => normDim(d, x.shape.length))); return; }
    case 'aten.transpose': {
      const x = X(), r = x.shape.length, p = x.shape.map((_, i) => i);
      const a = normDim(A(nd, 'dim0', 0), r), b = normDim(A(nd, 'dim1', 1), r); [p[a], p[b]] = [p[b], p[a]];
      permute(B, x, out, p); return;
    }
    case 'aten.t': { const x = X(); if (x.shape.length < 2) B.alias(out, x, x.shape.slice()); else permute(B, x, out, [1, 0]); return; }
    case 'aten.expand': {
      const x = X(), size = (A<number[]>(nd, 'size', [])).map(Number), r = size.length, lead = r - x.shape.length, st = contig(x.shape);
      const oshape = size.map((s, i) => (i < lead ? s : s === -1 ? x.shape[i - lead] : s));
      const ist = oshape.map((s, i) => (i < lead ? 0 : x.shape[i - lead] === 1 && s !== 1 ? 0 : st[i - lead]));
      strided(B, x, out, oshape, ist, 0); return;
    }
    case 'aten.select': {
      const x = X(), r = x.shape.length, d = normDim(A(nd, 'dim', 0), r); let idx = Number(A(nd, 'index', 0)); if (idx < 0) idx += x.shape[d];
      const st = contig(x.shape);
      strided(B, x, out, x.shape.filter((_, i) => i !== d), st.filter((_, i) => i !== d), idx * st[d]); return;
    }
    case 'aten.slice': {
      const x = X(), r = x.shape.length, d = normDim(A(nd, 'dim', 0), r), n = x.shape[d], step = Number(A(nd, 'step', 1));
      let s = Number(A(nd, 'start', 0)), e = Number(A(nd, 'end', n));
      if (s < 0) s += n; if (e < 0) e += n; s = Math.min(Math.max(s, 0), n); e = Math.min(Math.max(e, s), n);
      const st = contig(x.shape), shp = x.shape.slice(); shp[d] = Math.ceil((e - s) / step);
      const ist = st.slice(); ist[d] = st[d] * step;
      strided(B, x, out, shp, ist, s * st[d]); return;
    }
    case 'aten.pad': case 'aten.constant_pad_nd': {
      const x = X(), pads = (A<number[]>(nd, 'pad', [])).map(Number), mode = op === 'aten.constant_pad_nd' ? 'constant' : String(A(nd, 'mode', 'constant'));
      const value = Number(A(nd, 'value', 0) ?? 0);
      padOp(B, x, out, pads, mode, value); return;
    }
    case 'aten.cat': cat(B, ins.filter((n): n is string => !!n).map((n) => B.val(n)), out, Number(A(nd, 'dim', 0))); return;
    // elementwise
    case 'aten.add': case 'aten.sub': case 'aten.mul': case 'aten.div': {
      if (op === 'aten.div' && A(nd, 'rounding_mode', null) !== null) throw new Error('aten.div rounding_mode is unsupported');
      const a = X(); let b: Val;
      if (ins[1]) b = B.val(ins[1]);
      else { const o = Number(A(nd, 'other', NaN)); if (Number.isNaN(o)) throw new Error(`${op}: missing 'other'`); b = B.konst(`__scalar:${out}`, new Float32Array([o]), []); }
      binary(B, a, b, out, { 'aten.add': 0, 'aten.sub': 1, 'aten.mul': 2, 'aten.div': 3 }[op]!, Number(A(nd, 'alpha', 1))); return;
    }
    case 'aten.relu': case 'aten.relu_': unary(B, X(), out, UNARY_OPS.relu); return;
    case 'aten.leaky_relu': case 'aten.leaky_relu_': unary(B, X(), out, UNARY_OPS.leaky_relu, Number(A(nd, 'negative_slope', 0.01))); return;
    case 'aten.gelu': unary(B, X(), out, String(A(nd, 'approximate', 'none')) === 'tanh' ? UNARY_OPS.gelu_tanh : UNARY_OPS.gelu); return;
    case 'aten.sigmoid': unary(B, X(), out, UNARY_OPS.sigmoid); return;
    case 'aten.tanh': unary(B, X(), out, UNARY_OPS.tanh); return;
    case 'aten.silu': unary(B, X(), out, UNARY_OPS.silu); return;
    case 'aten.abs': unary(B, X(), out, UNARY_OPS.abs); return;
    case 'aten.neg': unary(B, X(), out, UNARY_OPS.neg); return;
    case 'aten.exp': unary(B, X(), out, UNARY_OPS.exp); return;
    case 'aten.sqrt': unary(B, X(), out, UNARY_OPS.sqrt); return;
    case 'aten.rsqrt': unary(B, X(), out, UNARY_OPS.rsqrt); return;
    case 'aten.prelu': {
      const x = X(), w = B.val(ins[1]); const C = numel(w.shape);
      if (C !== 1 && C !== x.shape[1]) throw new Error('aten.prelu: weight must have 1 or C elements');
      chanop(B, x, out, 1, w, B.zero1(), C, numel(x.shape.slice(2))); return;
    }
    // norms
    case 'aten.batch_norm': case 'aten._native_batch_norm_legit_no_training': {
      if (op === 'aten.batch_norm' && A(nd, 'training', false)) throw new Error('aten.batch_norm training=true is unsupported (inference only)');
      const x = X(); batchNorm(B, x, out, B.opt(ins[1]), B.opt(ins[2]), B.val(ins[3]), B.val(ins[4]), Number(A(nd, 'eps', 1e-5))); return;
    }
    case 'aten.instance_norm': {
      const x = X(), w = B.opt(ins[1]), b = B.opt(ins[2]), eps = Number(A(nd, 'eps', 1e-5));
      if (!A(nd, 'use_input_stats', true)) { batchNorm(B, x, out, w, b, B.val(ins[3]), B.val(ins[4]), eps); return; }
      const C = x.shape[1], L = numel(x.shape.slice(2));
      norm(B, x, out, x.shape[0] * C, L, L, 1, C, eps, w, b, 1); return;
    }
    case 'aten.group_norm': case 'aten.native_group_norm': {
      const x = X(), G = Number(op === 'aten.group_norm' ? A(nd, 'num_groups', 1) : A(nd, 'group', 1)), C = x.shape[1], sp = numel(x.shape.slice(2));
      if (C % G) throw new Error(`group_norm: C=${C} not divisible by groups=${G}`);
      norm(B, x, out, x.shape[0] * G, (C / G) * sp, sp, C / G, G, Number(A(nd, 'eps', 1e-5)), B.opt(ins[1]), B.opt(ins[2]), 1); return;
    }
    case 'aten.layer_norm': case 'aten.native_layer_norm': {
      const x = X(), ns = (A<number[]>(nd, 'normalized_shape', [x.shape[x.shape.length - 1]])).map(Number), L = numel(ns);
      norm(B, x, out, numel(x.shape) / L, L, 1, 1, 1, Number(A(nd, 'eps', 1e-5)), B.opt(ins[1]), B.opt(ins[2]), 2); return;
    }
    // conv / pool / upsample
    case 'aten.convolution': case 'aten.conv2d': case 'aten.conv3d': case 'aten.conv_transpose2d': case 'aten.conv_transpose3d': {
      const transposed = op.startsWith('aten.conv_transpose') || (op === 'aten.convolution' && !!A(nd, 'transposed', false));
      if (typeof nd.attrs?.padding === 'string') throw new Error(`${op}: string padding '${nd.attrs.padding}' is unsupported — lower it to explicit ints`);
      conv(B, X(), B.val(ins[1]), B.opt(ins[2]), out, nd, transposed); return;
    }
    case 'aten.max_pool2d': case 'aten.max_pool3d': case 'aten.max_pool2d_with_indices': case 'aten.max_pool3d_with_indices':
      pool(B, X(), out, nd, 0); return;
    case 'aten.avg_pool2d': case 'aten.avg_pool3d': pool(B, X(), out, nd, 1); return;
    case 'aten.adaptive_avg_pool2d': case 'aten.adaptive_avg_pool3d': {
      // divisible sizes only: every window then has the same extent in/out, so this IS avg_pool with kernel = stride
      const x = X(), ns = x.shape.length - 2, I = x.shape.slice(2), O = listN(nd.attrs?.output_size, ns, 1);
      if (O.length !== ns) throw new Error(`${op}: output_size [${O}] does not match ${ns} spatial dims`);
      const k = I.map((v, i) => {
        if (O[i] <= 0 || v % O[i]) throw new Error(`${op}: input size ${v} is not divisible by output size ${O[i]} (only evenly dividing adaptive pools are supported)`);
        return v / O[i];
      });
      pool(B, x, out, { op: nd.op, inputs: nd.inputs, output: nd.output, attrs: { kernel_size: k, stride: k, padding: 0, ceil_mode: false, count_include_pad: true, divisor_override: null } }, 1); return;
    }
    case 'aten.upsample_nearest2d': case 'aten.upsample_nearest3d': upsample(B, X(), out, nd, 0); return;
    case 'aten.upsample_bilinear2d': case 'aten.upsample_trilinear3d': upsample(B, X(), out, nd, 1); return;
    // reductions
    case 'aten._softmax': case 'aten.softmax': { const x = X(); const d = normDim(Number(A(nd, 'dim', -1)), x.shape.length); linered(B, x, out, d, d + 1, 0, false); return; }
    case 'aten.argmax': {
      const x = X(), dv = nd.attrs?.dim, keep = !!A(nd, 'keepdim', false);
      if (dv === null || dv === undefined) { const flat = B.alias(`${out}::flat`, x, [numel(x.shape)]); linered(B, flat, out, 0, 1, 1, false, keep ? x.shape.map(() => 1) : []); return; }
      const d = normDim(Number(dv), x.shape.length); linered(B, x, out, d, d + 1, 1, keep); return;
    }
    case 'aten.mean': {
      const x = X(), r = x.shape.length, dv = nd.attrs?.dim;
      const dims = (dv === null || dv === undefined || (Array.isArray(dv) && dv.length === 0) ? x.shape.map((_, i) => i) : (Array.isArray(dv) ? dv : [dv]).map((d) => normDim(Number(d), r))).sort((a, b) => a - b);
      for (let i = 1; i < dims.length; i++) if (dims[i] !== dims[i - 1] + 1) throw new Error('aten.mean: reduced dims must be contiguous');
      linered(B, x, out, dims[0], dims[dims.length - 1] + 1, 2, !!A(nd, 'keepdim', false)); return;
    }
    // matmul family
    case 'aten.linear': {
      const x = X(), w = B.val(ins[1]), b = B.opt(ins[2]); const K = x.shape[x.shape.length - 1], N = w.shape[0];
      if (w.shape[1] !== K) throw new Error(`aten.linear: weight [${w.shape}] vs input K=${K}`);
      gemm(B, { a: x, b: w, bias: b, M: numel(x.shape) / K, N, K, transB: true, batches: 1, aBS: 0, bBS: 0, alpha: 1, beta: 1, out, outShape: [...x.shape.slice(0, -1), N] }); return;
    }
    case 'aten.addmm': {
      const bias = B.val(ins[0]), m1 = B.val(ins[1]), m2 = B.val(ins[2]); const [M, K] = m1.shape, N = m2.shape[1];
      const alpha = Number(A(nd, 'alpha', 1)), beta = Number(A(nd, 'beta', 1));
      if (numel(bias.shape) === N) { gemm(B, { a: m1, b: m2, bias, M, N, K, transB: false, batches: 1, aBS: 0, bBS: 0, alpha, beta, out, outShape: [M, N] }); return; }
      gemm(B, { a: m1, b: m2, bias: null, M, N, K, transB: false, batches: 1, aBS: 0, bBS: 0, alpha, beta: 1, out: `${out}::mm`, outShape: [M, N] });
      binary(B, B.val(`${out}::mm`), bias, out, 0, beta); return;
    }
    case 'aten.mm': { const a = X(), b = B.val(ins[1]); gemm(B, { a, b, bias: null, M: a.shape[0], N: b.shape[1], K: a.shape[1], transB: false, batches: 1, aBS: 0, bBS: 0, alpha: 1, beta: 1, out, outShape: [a.shape[0], b.shape[1]] }); return; }
    case 'aten.bmm': case 'aten.matmul': matmul(B, X(), B.val(ins[1]), out); return;
    case 'aten.scaled_dot_product_attention': attention(B, nd); return;
  }
  throw new Error(`unsupported op ${op}`);
}

// ── lowering helpers ──
function newOut(B: Builder, out: string, shape: number[]): Val { return B.act(out, shape); }

function foldConst(B: Builder, x: Val, out: string, oshape: number[], ist: number[], inOff: number): boolean {
  if (x.kind !== 'const') return false;
  const n = numel(oshape), d = new Float32Array(n), r = oshape.length;
  for (let i = 0; i < n; i++) { let rem = i, ii = inOff; for (let k = r - 1; k >= 0; k--) { const c = rem % oshape[k]; rem = Math.floor(rem / oshape[k]); ii += c * ist[k]; } d[i] = x.data![ii]; }
  B.konst(out, d, oshape); return true;
}
function strided(B: Builder, x: Val, out: string, oshape: number[], ist: number[], inOff: number): void {
  if (oshape.length > 6) throw new Error(`${B.nodeOp}: rank > 6 unsupported`);
  // identity (contiguous, full) → alias
  const cs = contig(x.shape);
  if (inOff === 0 && numel(oshape) === numel(x.shape) && oshape.length === x.shape.length && oshape.every((s, i) => s === x.shape[i] && (s === 1 || ist[i] === cs[i]))) { B.alias(out, x, oshape); return; }
  if (foldConst(B, x, out, oshape, ist, inOff)) return;
  const o = newOut(B, out, oshape), T = numel(oshape);
  const dims = pad6(oshape, 1), ins = pad6(ist, 0), outs = pad6(contig(oshape), 0);
  B.flat(T, (s, e) => [B.whole(x), B.win(o, s, e)], (s, e) => {
    const ub = new UB().s(4, e - s).s(5, s).s(23, inOff).s(30, 0);
    for (let k = 0; k < 6; k++) { ub.s(11 + k, dims[k]).s(17 + k, ins[k]).s(24 + k, outs[k]); }
    return ub;
  }, 'copy');
}
function permute(B: Builder, x: Val, out: string, p: number[]): void {
  const st = contig(x.shape); strided(B, x, out, p.map((d) => x.shape[d]), p.map((d) => st[d]), 0);
}
function padOp(B: Builder, x: Val, out: string, pads: number[], mode: string, value: number): void {
  const r = x.shape.length, before = new Array(r).fill(0), after = new Array(r).fill(0);
  for (let i = 0; i * 2 < pads.length; i++) { before[r - 1 - i] = pads[2 * i]; after[r - 1 - i] = pads[2 * i + 1] ?? 0; }
  const oshape = x.shape.map((s, i) => s + before[i] + after[i]);
  const m = { constant: 0, replicate: 1, reflect: 2 }[mode];
  if (m === undefined) throw new Error(`aten.pad mode '${mode}' unsupported`);
  if (pads.every((p) => p === 0)) { B.alias(out, x, oshape); return; }
  const o = newOut(B, out, oshape), T = numel(oshape), od = pad6(oshape, 1), id = pad6(x.shape, 1), pb = pad6(before, 0);
  B.flat(T, (s, e) => [B.whole(x), B.win(o, s, e)], (s, e) => {
    const ub = new UB().s(4, e - s).s(5, s).s(29, m).sf(30, value);
    for (let k = 0; k < 6; k++) ub.s(11 + k, od[k]).s(17 + k, id[k]).si(23 + k, pb[k]);
    return ub;
  }, 'pad');
}
function cat(B: Builder, xs: Val[], out: string, dim: number): void {
  const r = xs[0].shape.length, d = normDim(dim, r);
  const oshape = xs[0].shape.slice(); oshape[d] = xs.reduce((a, x) => a + x.shape[d], 0);
  const o = newOut(B, out, oshape), outer = numel(oshape.slice(0, d)), inner = numel(oshape.slice(d + 1)), orun = oshape[d] * inner;
  let off = 0;
  for (const x of xs) {
    const run = x.shape[d] * inner;
    if (outer <= 4096) for (let q = 0; q < outer; q++) B.copy(x, q * run, o, q * orun + off * inner, run);
    else {
      const it = pad6([outer, run], 1), ist = pad6([run, 1], 0), ost = pad6([orun, 1], 0);
      const ub = new UB().s(4, outer * run).s(5, 0).s(23, 0).s(30, off * inner);
      for (let k = 0; k < 6; k++) ub.s(11 + k, it[k]).s(17 + k, ist[k]).s(24 + k, ost[k]);
      B.kernel('copy', [B.whole(x), B.whole(o)], ub, grid1(outer * run));
    }
    off += x.shape[d];
  }
}
function unary(B: Builder, x: Val, out: string, op: number, alpha = 0, beta = 0): void {
  const o = newOut(B, out, x.shape.slice()), T = numel(x.shape);
  B.flat(T, (s, e) => [B.win(x, s, e), B.win(o, s, e)], (s, e) => new UB().s(4, e - s).s(5, s).s(8, op).sf(9, alpha).sf(10, beta), 'unary');
}
function binary(B: Builder, a: Val, b: Val, out: string, op: number, alpha: number): void {
  const r = Math.max(a.shape.length, b.shape.length);
  if (r > 6) throw new Error(`${B.nodeOp}: rank > 6 unsupported`);
  const as = pad6(a.shape, 1), bs = pad6(b.shape, 1), os = as.map((v, i) => {
    if (v !== bs[i] && v !== 1 && bs[i] !== 1) throw new Error(`${B.nodeOp}: cannot broadcast [${a.shape}] with [${b.shape}]`);
    return Math.max(v, bs[i]);
  });
  const oshape = os.slice(6 - r), o = newOut(B, out, oshape), T = numel(oshape);
  const ast = contig(as).map((s, i) => (as[i] === 1 && os[i] !== 1 ? 0 : s)), bst = contig(bs).map((s, i) => (bs[i] === 1 && os[i] !== 1 ? 0 : s));
  const same = (v: Val) => numel(v.shape) === T;
  B.flat(T, (s, e) => [same(a) ? B.win(a, s, e) : B.whole(a), same(b) ? B.win(b, s, e) : B.whole(b), B.win(o, s, e)], (s, e) => {
    const ub = new UB().s(4, e - s).s(5, s).s(8, op).sf(9, alpha);
    for (let k = 0; k < 6; k++) ub.s(11 + k, os[k]).s(17 + k, ast[k]).s(23 + k, bst[k]);
    return ub;
  }, 'binary');
}
function chanop(B: Builder, x: Val, out: string, mode: number, s: Val, t: Val, C: number, inner: number): void {
  const o = newOut(B, out, x.shape.slice()), T = numel(x.shape);
  B.flat(T, (a, e) => [B.win(x, a, e), B.whole(s), B.whole(t), B.win(o, a, e)], (a, e) => new UB().s(4, e - a).s(5, a).s(8, C).s(9, inner).s(10, mode), 'chanop');
}
function batchNorm(B: Builder, x: Val, out: string, w: Val | null, b: Val | null, rm: Val, rv: Val, eps: number): void {
  const C = x.shape[1], inner = numel(x.shape.slice(2));
  const mean = rm.data!, v = rv.data!, wd = w?.data, bd = b?.data;
  if (!mean || !v) throw new Error(`${B.nodeOp}: running stats must be weights`);
  if (B.opts.batchNorm === 'explicit') {
    const s = new Float32Array(2 * C), t = new Float32Array(2 * C);
    for (let c = 0; c < C; c++) { s[c] = mean[c]; s[C + c] = fr(1 / fr(Math.sqrt(fr(v[c] + eps)))); t[c] = wd ? wd[c] : 1; t[C + c] = bd ? bd[c] : 0; }
    chanop(B, x, out, 2, B.konst(`__bn_s:${out}`, s, [2 * C]), B.konst(`__bn_t:${out}`, t, [2 * C]), C, inner); return;
  }
  // PyTorch CPU eval: invstd = 1/sqrt(var+eps); scale = w·invstd; shift = b − mean·scale; y = x·scale + shift
  const s = new Float32Array(C), t = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    const inv = fr(1 / fr(Math.sqrt(fr(v[c] + fr(eps)))));
    s[c] = fr((wd ? wd[c] : 1) * inv); t[c] = fr((bd ? bd[c] : 0) - fr(mean[c] * s[c]));
  }
  chanop(B, x, out, 0, B.konst(`__bn_s:${out}`, s, [C]), B.konst(`__bn_t:${out}`, t, [C]), C, inner);
}
function norm(B: Builder, x: Val, out: string, rows: number, L: number, inner: number, cpg: number, G: number, eps: number, w: Val | null, b: Val | null, affineMode: number): void {
  const o = newOut(B, out, x.shape.slice());
  const nAff = affineMode === 1 ? G * cpg : L;
  let mode = 0, wv = B.zero1(), bv = B.zero1();
  if (w || b) {
    mode = affineMode;
    wv = w ?? B.konst(`__ones:${nAff}`, new Float32Array(nAff).fill(1), [nAff]);
    bv = b ?? B.konst(`__zeros:${nAff}`, new Float32Array(nAff), [nAff]);
  }
  B.rows(rows, [{ v: x, off: 0, ue: L }, { v: o, off: 0, ue: L }], (r0, r1, win) => {
    const n = r1 - r0, gx = Math.min(n, MAXG);
    B.kernel('norm', [win[0], B.whole(wv), B.whole(bv), win[1]], new UB().s(4, n).s(8, L).s(9, inner).s(10, cpg).s(11, G).sf(12, eps).s(13, mode).s(14, r0), [gx, Math.ceil(n / gx), 1]);
  });
}
/** softmax/argmax/mean over dims [d0, d1) of x. */
function linered(B: Builder, x: Val, out: string, d0: number, d1: number, mode: number, keepdim: boolean, forceShape?: number[]): void {
  const outer = numel(x.shape.slice(0, d0)), S = numel(x.shape.slice(d0, d1)), inner = numel(x.shape.slice(d1));
  const oshape = mode === 0 ? x.shape.slice() : forceShape ?? (keepdim ? x.shape.map((s, i) => (i >= d0 && i < d1 ? 1 : s)) : [...x.shape.slice(0, d0), ...x.shape.slice(d1)]);
  const o = newOut(B, out, oshape), oue = mode === 0 ? S : 1;
  const launch = (xb: Bind, ob: Bind, lines: number, lineBase: number, inn: number) =>
    B.kernel('linered', [xb, ob], new UB().s(4, lines).s(5, lineBase).s(8, S).s(9, inn).s(10, mode), grid1(lines));
  const wx = B.whole(x), wo = B.whole(o);
  if (B.fits(wx, wo)) { launch(wx, wo, outer * inner, 0, inner); return; }
  if (inner === 1) { B.rows(outer, [{ v: x, off: 0, ue: S }, { v: o, off: 0, ue: oue }], (u0, u1, w) => launch(w[0], w[1], u1 - u0, u0, 1)); return; }
  const perOuterFits = (S * inner + ALIGN) * 4 <= B.lim;
  if (perOuterFits) { B.rows(outer, [{ v: x, off: 0, ue: S * inner }, { v: o, off: 0, ue: oue * inner }], (u0, u1, w) => launch(w[0], w[1], (u1 - u0) * inner, u0 * inner, inner)); return; }
  // inner-chunk: gather [outer, S, Q] tiles with buffer copies (no binding limit on copies), run, scatter back
  const Q = Math.floor(B.lim / 4 / (outer * S));
  if (Q < 1) B.refuse(`one inner column (${outer}×${S} elements)`);
  const tin = B.scratch(outer * S * Math.min(Q, inner)), tout = B.scratch(outer * oue * Math.min(Q, inner));
  let n = 0;
  for (let q0 = 0; q0 < inner; q0 += Q) {
    const Qc = Math.min(Q, inner - q0);
    for (let a = 0; a < outer * S; a++) B.copy(x, a * inner + q0, tin, a * Qc, Qc);
    B.kernel('linered', [{ value: tin.name, start: 0, len: outer * S * Qc }, { value: tout.name, start: 0, len: outer * oue * Qc }], new UB().s(4, outer * Qc).s(5, 0).s(8, S).s(9, Qc).s(10, mode), grid1(outer * Qc));
    for (let a = 0; a < outer * oue; a++) B.copy(tout, a * Qc, o, a * inner + q0, Qc);
    n++;
  }
  B.tile('inner-chunk', n);
}
interface GemmSpec { a: Val; b: Val; bias: Val | null; M: number; N: number; K: number; transB: boolean; batches: number; aBS: number; bBS: number; alpha: number; beta: number; out: string; outShape: number[] }
function gemm(B: Builder, g: GemmSpec): void {
  const o = newOut(B, g.out, g.outShape), bw = B.weight(g.b), f16 = !!bw.f16, bias = g.bias ?? B.zero1(), oBS = g.M * g.N;
  const base = (rowBase: number, rowEnd: number, batchBase: number, nb: number) => new UB().s(8, rowEnd).s(9, g.N).s(10, g.K).s(11, g.transB ? 1 : 0)
    .s(12, g.aBS).s(13, g.bBS).s(14, g.bias ? 1 : 0).sf(15, g.alpha).sf(16, g.beta).s(17, rowBase).s(18, nb).s(19, oBS).s(20, batchBase);
  const launchRows = (binds: Bind[], r0: number, r1: number, b0: number, nb: number) => {
    for (let s = r0; s < r1; s += MAXG * 16) {
      const e = Math.min(r1, s + MAXG * 16);
      B.kernel('matmul', binds, base(s, e, b0, nb), [Math.ceil(g.N / 16), Math.ceil((e - s) / 16), nb], f16);
    }
  };
  const all = [B.whole(g.a), B.whole(bw), B.whole(bias), B.whole(o)];
  if (B.fits(...all) && g.batches <= MAXG) { launchRows(all, 0, g.M, 0, g.batches); return; }
  // one launch group per batch; within a batch, rows of A/out are windowed (B.rows records its own split)
  const before = B.tiled.length;
  for (let bi = 0; bi < g.batches; bi++) {
    const bBind = g.bBS === 0 ? B.whole(bw) : B.win(bw, bi * g.bBS, bi * g.bBS + g.K * g.N);
    if (!B.fits(bBind)) B.refuse(`the B operand '${g.b.name}' (${g.K}×${g.N})`);
    B.rows(g.M, [{ v: g.a, off: bi * g.aBS, ue: g.K }, { v: o, off: bi * oBS, ue: g.N }], (r0, r1, w) => launchRows([w[0], bBind, B.whole(bias), w[1]], r0, r1, bi, 1));
  }
  if (B.tiled.length === before) B.tile('rows', g.batches);
}
function matmul(B: Builder, a0: Val, b0: Val, out: string): void {
  let a = a0, b = b0;
  const aVec = a.shape.length === 1, bVec = b.shape.length === 1;
  if (aVec) a = B.alias(`${out}::a2`, a, [1, a.shape[0]]);
  if (bVec) b = B.alias(`${out}::b2`, b, [b.shape[0], 1]);
  const M = a.shape[a.shape.length - 2], K = a.shape[a.shape.length - 1], N = b.shape[b.shape.length - 1];
  if (b.shape[b.shape.length - 2] !== K) throw new Error(`${B.nodeOp}: [${a.shape}] @ [${b.shape}] inner dims differ`);
  const ab = a.shape.slice(0, -2), bb = b.shape.slice(0, -2), r = Math.max(ab.length, bb.length);
  const pa = [...new Array(r - ab.length).fill(1), ...ab], pb = [...new Array(r - bb.length).fill(1), ...bb];
  const batch = pa.map((v, i) => { if (v !== pb[i] && v !== 1 && pb[i] !== 1) throw new Error(`${B.nodeOp}: batch dims [${ab}] vs [${bb}] do not broadcast`); return Math.max(v, pb[i]); });
  const nb = numel(batch);
  const expandTo = (v: Val, pv: number[], tail: number[], tag: string): { v: Val; bs: number } => {
    if (numel(pv) === nb) return { v, bs: numel(tail) };
    if (numel(pv) === 1) return { v, bs: 0 };
    const st = contig([...pv, ...tail]).map((s, i) => (i < pv.length && pv[i] === 1 && batch[i] !== 1 ? 0 : s));
    const name = `${out}::${tag}`; strided(B, v, name, [...batch, ...tail], st, 0);
    return { v: B.val(name), bs: numel(tail) };
  };
  const A2 = expandTo(a, pa, [M, K], 'ea'), B2 = expandTo(b, pb, [K, N], 'eb');
  let oshape = [...batch, M, N];
  if (aVec) oshape = [...batch, N];
  if (bVec) oshape = aVec ? [...batch] : [...batch, M];
  if (aVec || bVec) {
    gemm(B, { a: A2.v, b: B2.v, bias: null, M, N, K, transB: false, batches: nb, aBS: A2.bs, bBS: B2.bs, alpha: 1, beta: 1, out: `${out}::mm`, outShape: [...batch, M, N] });
    B.alias(out, B.val(`${out}::mm`), oshape); return;
  }
  gemm(B, { a: A2.v, b: B2.v, bias: null, M, N, K, transB: false, batches: nb, aBS: A2.bs, bBS: B2.bs, alpha: 1, beta: 1, out, outShape: oshape });
}
function attention(B: Builder, nd: OpNode): void {
  const ins = nd.inputs;
  if (ins[3]) throw new Error('scaled_dot_product_attention: attn_mask is unsupported');
  if (Number(A(nd, 'dropout_p', 0)) !== 0) throw new Error('scaled_dot_product_attention: dropout_p must be 0 (inference)');
  if (A(nd, 'enable_gqa', false)) throw new Error('scaled_dot_product_attention: enable_gqa is unsupported');
  const q = B.val(ins[0]), k = B.val(ins[1]), v = B.val(ins[2]);
  const r = q.shape.length, Sq = q.shape[r - 2], D = q.shape[r - 1], Sk = k.shape[r - 2], Dv = v.shape[r - 1];
  const BH = numel(q.shape.slice(0, -2));
  if (numel(k.shape.slice(0, -2)) !== BH || numel(v.shape.slice(0, -2)) !== BH || k.shape[r - 1] !== D) throw new Error('scaled_dot_product_attention: q/k/v batch or head dims differ');
  if (D > 256 || Dv > 256) throw new Error('scaled_dot_product_attention: head_dim > 256 unsupported');
  const sa = nd.attrs?.scale, scale = sa === null || sa === undefined ? fr(1 / Math.sqrt(D)) : fr(Number(sa)), causal = A(nd, 'is_causal', false) ? 1 : 0;
  const o = newOut(B, nd.output, [...q.shape.slice(0, -1), Dv]);
  B.rows(BH, [{ v: q, off: 0, ue: Sq * D }, { v: k, off: 0, ue: Sk * D }, { v, off: 0, ue: Sk * Dv }, { v: o, off: 0, ue: Sq * Dv }], (u0, u1, w) =>
    B.kernel('attention', w, new UB().s(4, (u1 - u0) * Sq).s(5, u0 * Sq).s(8, Sq).s(9, Sk).s(10, D).s(11, Dv).sf(12, scale).s(13, causal), grid1((u1 - u0) * Sq)));
}

/** Tile a spatial kernel along the outermost (D) axis: gather an input slab (+halo) and scatter the output slab with
 *  buffer copies, so every binding stays under maxStorageBufferBindingSize. Untiled when everything fits. */
function slab(B: Builder, kernel: string, f16: boolean, x: Val, o: Val, d: { N: number; Cin: number; ID: number; IH: number; IW: number; Cout: number; OD: number; OH: number; OW: number },
  extra: Bind[], params: (ub: UB) => void, tileP: (ub: UB, inZ0: number, TZ: number, outZ0: number, TO: number) => void,
  grids: (TO: number, ub: UB) => { ub: UB; grid: [number, number, number] }[], inRange: (z0: number, z1: number) => [number, number]): void {
  const inB = (TZ: number) => d.N * d.Cin * TZ * d.IH * d.IW, outB = (TO: number) => d.N * d.Cout * TO * d.OH * d.OW;
  const emit = (xb: Bind, ob: Bind, inZ0: number, TZ: number, outZ0: number, TO: number) => {
    const ub = new UB(); params(ub); tileP(ub, inZ0, TZ, outZ0, TO);
    for (const g of grids(TO, ub)) B.kernel(kernel, [xb, ...extra, ob], g.ub, g.grid, f16);
  };
  if (inB(d.ID) * 4 <= B.lim && outB(d.OD) * 4 <= B.lim) { emit(B.whole(x), B.whole(o), 0, d.ID, 0, d.OD); return; }
  const fitsT = (T: number) => {
    for (let z0 = 0; z0 < d.OD; z0 += T) { const [a, b] = inRange(z0, Math.min(d.OD, z0 + T)); if (inB(b - a) * 4 > B.lim || outB(Math.min(T, d.OD - z0)) * 4 > B.lim) return false; }
    return true;
  };
  let T = d.OD;
  while (T > 1 && !fitsT(T)) T = Math.ceil(T / 2);
  if (!fitsT(T)) B.refuse(`one output depth-slice (input slab ${inB(inRange(0, 1)[1] - inRange(0, 1)[0]) * 4} B, output ${outB(1) * 4} B)`);
  let maxTZ = 1;
  for (let z0 = 0; z0 < d.OD; z0 += T) { const [a, b] = inRange(z0, Math.min(d.OD, z0 + T)); maxTZ = Math.max(maxTZ, b - a); }
  const tin = B.scratch(inB(maxTZ)), tout = B.scratch(outB(T));
  const inPlane = d.IH * d.IW, outPlane = d.OH * d.OW;
  let n = 0;
  for (let z0 = 0; z0 < d.OD; z0 += T) {
    const z1 = Math.min(d.OD, z0 + T), TO = z1 - z0, [a, b] = inRange(z0, z1), TZ = b - a;
    for (let nc = 0; nc < d.N * d.Cin; nc++) B.copy(x, (nc * d.ID + a) * inPlane, tin, nc * TZ * inPlane, TZ * inPlane);
    emit({ value: tin.name, start: 0, len: inB(TZ) }, { value: tout.name, start: 0, len: outB(TO) }, a, TZ, z0, TO);
    for (let nc = 0; nc < d.N * d.Cout; nc++) B.copy(tout, nc * TO * outPlane, o, (nc * d.OD + z0) * outPlane, TO * outPlane);
    n++;
  }
  B.tile('spatial-slab', n);
}
const clampR = (a: number, b: number, n: number): [number, number] => { const lo = Math.min(Math.max(a, 0), n - 1); return [lo, Math.max(lo + 1, Math.min(b, n))]; };

function conv(B: Builder, x: Val, w0: Val, b: Val | null, out: string, nd: OpNode, transposed: boolean): void {
  const s = sp3(x.shape), ns = x.shape.length - 2, G = Number(A(nd, 'groups', 1));
  const k = to3(w0.shape.slice(2), 1), st = to3(listN(nd.attrs?.stride, ns, 1), 1), pd = to3(listN(nd.attrs?.padding, ns, 0), 0);
  const dl = to3(listN(nd.attrs?.dilation, ns, 1), 1), op = to3(listN(nd.attrs?.output_padding, ns, 0), 0);
  const Cout = transposed ? w0.shape[1] * G : w0.shape[0];
  if ((transposed ? w0.shape[0] : w0.shape[1] * G) !== s.C) throw new Error(`${B.nodeOp}: weight [${w0.shape}] does not match input channels ${s.C} (groups ${G})`);
  const I = [s.D, s.H, s.W];
  const O = I.map((v, i) => (transposed ? (v - 1) * st[i] - 2 * pd[i] + dl[i] * (k[i] - 1) + op[i] + 1 : Math.floor((v + 2 * pd[i] - dl[i] * (k[i] - 1) - 1) / st[i]) + 1));
  if (O.some((v) => v <= 0)) throw new Error(`${B.nodeOp}: empty output [${O}]`);
  const oshape = ns === 2 ? [s.N, Cout, O[0], O[1]] : [s.N, Cout, O[0], O[1], O[2]];
  const o = newOut(B, out, oshape), w = B.weight(w0), bias = b ?? B.zero1();
  const params = (ub: UB) => {
    ub.s(8, s.N).s(9, s.C).s(11, s.H).s(12, s.W).s(13, Cout).s(15, O[1]).s(16, O[2]);
    for (let i = 0; i < 3; i++) ub.s(17 + i, k[i]).s(20 + i, st[i]).s(23 + i, pd[i]).s(26 + i, dl[i]);
    ub.s(29, G).s(30, b ? 1 : 0).s(33, s.D);
  };
  const tileP = (ub: UB, inZ0: number, TZ: number, outZ0: number, TO: number) => { ub.s(10, TZ).s(14, TO).s(31, inZ0).s(32, outZ0); };
  const Og = Cout / G;
  const grids = (TO: number, ub: UB) => {
    if (transposed) { const T = s.N * Cout * TO * O[1] * O[2]; return [{ ub: cloneUB(ub).s(4, T), grid: grid1(T) }]; }
    const M = TO * O[1] * O[2], res: { ub: UB; grid: [number, number, number] }[] = [];
    for (let m0 = 0; m0 < M; m0 += MAXG * 16) { const u2 = cloneUB(ub); u2.s(34, m0); res.push({ ub: u2, grid: [Math.ceil(Og / 16), Math.ceil(Math.min(M - m0, MAXG * 16) / 16), s.N * G] }); }
    return res;
  };
  const inRange = (z0: number, z1: number): [number, number] => transposed
    ? clampR(Math.ceil((z0 + pd[0] - (k[0] - 1) * dl[0]) / st[0]), Math.floor((z1 - 1 + pd[0]) / st[0]) + 1, s.D)
    : clampR(z0 * st[0] - pd[0], (z1 - 1) * st[0] - pd[0] + (k[0] - 1) * dl[0] + 1, s.D);
  slab(B, transposed ? 'convT' : 'conv', !!w.f16, x, o, { N: s.N, Cin: s.C, ID: s.D, IH: s.H, IW: s.W, Cout, OD: O[0], OH: O[1], OW: O[2] },
    [B.whole(w), B.whole(bias)], params, tileP, grids, inRange);
}
function cloneUB(ub: UB): UB { const n = new UB(); n.u.set(ub.u); return n; }
function pool(B: Builder, x: Val, out: string, nd: OpNode, mode: number): void {
  if (A(nd, 'ceil_mode', false)) throw new Error(`${B.nodeOp}: ceil_mode=true is unsupported`);
  const s = sp3(x.shape), ns = x.shape.length - 2;
  const k = to3(listN(nd.attrs?.kernel_size, ns, 1), 1);
  const stRaw = nd.attrs?.stride, st = to3(Array.isArray(stRaw) && stRaw.length === 0 || stRaw === null || stRaw === undefined ? k.slice(0, ns) : listN(stRaw, ns, 1), 1);
  const pd = to3(listN(nd.attrs?.padding, ns, 0), 0), dl = to3(listN(nd.attrs?.dilation, ns, 1), 1);
  const I = [s.D, s.H, s.W], O = I.map((v, i) => Math.floor((v + 2 * pd[i] - (mode === 0 ? dl[i] : 1) * (k[i] - 1) - 1) / st[i]) + 1);
  const oshape = ns === 2 ? [s.N, s.C, O[0], O[1]] : [s.N, s.C, O[0], O[1], O[2]];
  const o = newOut(B, out, oshape);
  const dv = nd.attrs?.divisor_override;
  const params = (ub: UB) => {
    ub.s(8, s.N * s.C).s(10, s.H).s(11, s.W).s(13, O[1]).s(14, O[2]);
    for (let i = 0; i < 3; i++) ub.s(15 + i, k[i]).s(18 + i, st[i]).s(21 + i, pd[i]).s(24 + i, mode === 0 ? dl[i] : 1);
    ub.s(27, mode).s(28, A(nd, 'count_include_pad', true) ? 1 : 0).s(29, dv === null || dv === undefined ? 0 : Number(dv)).s(32, s.D);
  };
  const tileP = (ub: UB, inZ0: number, TZ: number, outZ0: number, TO: number) => { ub.s(9, TZ).s(12, TO).s(30, inZ0).s(31, outZ0); };
  const grids = (TO: number, ub: UB) => { const T = s.N * s.C * TO * O[1] * O[2]; const u2 = cloneUB(ub).s(4, T); return [{ ub: u2, grid: grid1(T) }]; };
  const inRange = (z0: number, z1: number) => clampR(z0 * st[0] - pd[0], (z1 - 1) * st[0] - pd[0] + (k[0] - 1) * (mode === 0 ? dl[0] : 1) + 1, s.D);
  slab(B, 'pool', false, x, o, { N: 1, Cin: s.N * s.C, ID: s.D, IH: s.H, IW: s.W, Cout: s.N * s.C, OD: O[0], OH: O[1], OW: O[2] }, [], params, tileP, grids, inRange);
}
function upsample(B: Builder, x: Val, out: string, nd: OpNode, mode: number): void {
  const s = sp3(x.shape), ns = x.shape.length - 2, I = [s.D, s.H, s.W].slice(0, ns);
  const osz = nd.attrs?.output_size as number[] | null | undefined;
  let sf = nd.attrs?.scale_factors as number[] | null | undefined;
  if (!sf) { const alt = ['scales_d', 'scales_h', 'scales_w'].slice(3 - ns).map((kk) => nd.attrs?.[kk]); if (alt.every((v) => typeof v === 'number')) sf = alt as number[]; }
  const O = osz && osz.length ? osz.map(Number) : sf ? I.map((v, i) => Math.floor(v * Number(sf![i]))) : null;
  if (!O) throw new Error(`${B.nodeOp}: needs output_size or scale_factors`);
  const ac = mode === 1 && A(nd, 'align_corners', false) ? 1 : 0;
  const useSf = !(osz && osz.length) && !!sf;
  const sc = I.map((iv, i) => {
    if (mode === 1 && ac) return O[i] > 1 ? fr((iv - 1) / (O[i] - 1)) : 0;
    return useSf && Number(sf![i]) > 0 ? fr(1 / Number(sf![i])) : fr(iv / O[i]);
  });
  const I3 = to3(I, 1), O3 = to3(O, 1), sc3 = to3(sc, 1);
  const oshape = [s.N, s.C, ...O];
  const o = newOut(B, out, oshape);
  const params = (ub: UB) => {
    ub.s(8, s.N * s.C).s(10, I3[1]).s(11, I3[2]).s(13, O3[1]).s(14, O3[2]).s(15, mode).s(16, ac);
    ub.sf(17, sc3[0]).sf(18, sc3[1]).sf(19, sc3[2]).s(22, I3[0]).s(23, O3[0]);
  };
  const tileP = (ub: UB, inZ0: number, TZ: number, outZ0: number, TO: number) => { ub.s(9, TZ).s(12, TO).s(20, inZ0).s(21, outZ0); };
  const grids = (TO: number, ub: UB) => { const T = s.N * s.C * TO * O3[1] * O3[2]; return [{ ub: cloneUB(ub).s(4, T), grid: grid1(T) }]; };
  const inRange = (z0: number, z1: number): [number, number] => mode === 0
    ? clampR(nidxF(z0, I3[0], O3[0], sc3[0]), nidxF(z1 - 1, I3[0], O3[0], sc3[0]) + 1, I3[0])
    : clampR(lidxF(z0, I3[0], sc3[0], ac)[0], lidxF(z1 - 1, I3[0], sc3[0], ac)[1] + 1, I3[0]);
  slab(B, 'upsample', false, x, o, { N: 1, Cin: s.N * s.C, ID: I3[0], IH: I3[1], IW: I3[2], Cout: s.N * s.C, OD: O3[0], OH: O3[1], OW: O3[2] }, [], params, tileP, grids, inRange);
}

// ════════════════════════════════════════ memory planner ════════════════════════════════════════
function planMemory(B: Builder, graph: OpGraph, lim: number, maxBuf: number): { plan: MemoryPlan; slotOf: Map<string, number> } {
  const L = B.launches, def = new Map<string, number>(), last = new Map<string, number>();
  const rootOf = (n: string) => B.vals.get(n)!.root;
  const planned = (n: string) => { const v = B.vals.get(n)!; return v.kind !== 'const'; };
  const touch = (n: string, t: number, write: boolean) => {
    if (!planned(n)) return;
    const r = rootOf(n);
    if (write && !def.has(r)) def.set(r, t);
    last.set(r, Math.max(last.get(r) ?? -1, t));
  };
  for (const i of graph.inputs) { def.set(i.name, -1); last.set(i.name, -1); }
  L.forEach((l, t) => {
    if (l.kind === 'kernel') { l.binds.forEach((b, j) => touch(b.value, t, j === l.binds.length - 1)); }
    else { touch(l.src, t, false); touch(l.dst, t, true); }
  });
  const outRoots = new Set(graph.outputs.filter(planned).map(rootOf));
  for (const r of outRoots) { if (!def.has(r)) def.set(r, -1); last.set(r, Number.POSITIVE_INFINITY); }
  const bytesOf = (r: string) => Math.max(4, numel(B.vals.get(r)!.shape) * 4);
  const roots = [...def.keys()].sort((a, b) => def.get(a)! - def.get(b)! || (a < b ? -1 : 1));
  const slots: number[] = [], free: number[] = [], slotOf = new Map<string, number>();
  const byDef = new Map<number, string[]>(), byLast = new Map<number, string[]>();
  for (const r of roots) {
    if (!last.has(r)) last.set(r, def.get(r)!);
    (byDef.get(def.get(r)!) ?? byDef.set(def.get(r)!, []).get(def.get(r)!)!).push(r);
    const lu = last.get(r)!; if (Number.isFinite(lu)) (byLast.get(lu) ?? byLast.set(lu, []).get(lu)!).push(r);
  }
  let live = 0, peak = 0;
  for (let t = -1; t < L.length; t++) {
    for (const r of byDef.get(t) ?? []) {
      const need = bytesOf(r);
      if (need > maxBuf) throw new Error(`value '${r}' needs ${need} B > maxBufferSize (${maxBuf} B)`);
      let best = -1;
      for (const s of free) if (slots[s] >= need && (best < 0 || slots[s] < slots[best])) best = s;
      if (best < 0 && free.length) { best = free.reduce((a, b) => (slots[b] > slots[a] ? b : a)); slots[best] = need; }
      if (best < 0) { best = slots.length; slots.push(need); } else free.splice(free.indexOf(best), 1);
      slotOf.set(r, best); live += need; peak = Math.max(peak, live);
    }
    for (const r of byLast.get(t) ?? []) { free.push(slotOf.get(r)!); live -= bytesOf(r); }
  }
  const values: MemoryPlan['values'] = {};
  let naive = 0, constBytes = 0;
  for (const [n, v] of B.vals) {
    if (v.kind === 'const') { constBytes += numel(v.shape) * (v.f16 ? 2 : 4); continue; }
    const r = v.root;
    if (!slotOf.has(r)) continue; // declared but never touched (e.g. a dead view)
    const e: MemoryPlan['values'][string] = { slot: slotOf.get(r)!, def: def.get(r)!, lastUse: last.get(r)!, bytes: bytesOf(r) };
    if (r !== n) e.aliasOf = r; else naive += bytesOf(r);
    values[n] = e;
  }
  const plan: MemoryPlan = {
    maxBindingBytes: lim, maxBufferBytes: maxBuf, slots, totalBytes: slots.reduce((a, b) => a + b, 0), naiveBytes: naive, peakLiveBytes: peak, constBytes,
    launches: L.filter((l) => l.kind === 'kernel').length, copies: L.filter((l) => l.kind === 'copy').length, values, tiled: B.tiled, bindings: B.bindings,
  };
  return { plan, slotOf };
}

// ════════════════════════════════════════ CPU reference runner ════════════════════════════════════════
function checkInputs(model: VisionModel, inputs: Record<string, Tensor>): void {
  for (const i of model.inputs) {
    const t = inputs[i.name];
    if (!t) throw new Error(`missing input '${i.name}'`);
    if (t.shape.length !== i.shape.length || t.shape.some((v, k) => v !== i.shape[k])) throw new Error(`input '${i.name}' shape [${t.shape}] != graph shape [${i.shape}]`);
    if (t.data.length !== numel(i.shape)) throw new Error(`input '${i.name}' has ${t.data.length} elements, shape needs ${numel(i.shape)}`);
  }
}
/** Execute the compiled launch list on the CPU mirrors of the WGSL kernels. */
export function runCpu(model: VisionModel, inputs: Record<string, Tensor>): Record<string, Tensor> {
  checkInputs(model, inputs);
  const slots = model.plan.slots.map((b) => new Float32Array(b / 4));
  const buf = (name: string): Float32Array => {
    const v = model.vals.get(name)!;
    if (v.kind === 'const') return v.data!;
    return slots[model.slotOf.get(v.root)!];
  };
  for (const i of model.inputs) buf(i.name).set(inputs[i.name].data);
  for (const l of model.launches) {
    if (l.kind === 'copy') { buf(l.dst).set(buf(l.src).subarray(l.srcOff, l.srcOff + l.len), l.dstOff); continue; }
    const views = l.binds.map((b) => buf(b.value).subarray(b.start, b.start + b.len));
    CPU_KERNELS[l.kernel](views, l.u);
  }
  const out: Record<string, Tensor> = {};
  for (const o of model.outputs) { const shape = model.shapeOf(o); out[o] = { shape, data: buf(o).slice(0, numel(shape)) }; }
  return out;
}

// ════════════════════════════════════════ WebGPU runner ════════════════════════════════════════
export interface GpuRunner { run(inputs: Record<string, Tensor>): Promise<Record<string, Tensor>>; destroy(): void; readonly label: string }
const PIPES = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();
function pipeline(device: GPUDevice, kernel: string, f16: boolean): GPUComputePipeline {
  let m = PIPES.get(device); if (!m) { m = new Map(); PIPES.set(device, m); }
  const key = `${kernel}:${f16}`;
  let p = m.get(key);
  if (!p) { p = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: wgslSource(kernel, f16) }), entryPoint: 'main' } }); m.set(key, p); }
  return p;
}
/** Upload weights once (resident), pre-build every bind group/uniform, and run the launch list in one submit. */
export async function createGpuRunner(device: GPUDevice, model: VisionModel): Promise<GpuRunner> {
  if (model.f16Weights && !device.features.has('shader-f16')) throw new Error('model compiled with f16Weights but the device lacks shader-f16');
  if (model.plan.maxBindingBytes > device.limits.maxStorageBufferBindingSize) throw new Error(`model planned for ${model.plan.maxBindingBytes} B bindings; device allows ${device.limits.maxStorageBufferBindingSize} — compile with maxBindingBytes = device.limits.maxStorageBufferBindingSize`);
  const U = GPUBufferUsage;
  const owned: GPUBuffer[] = [];
  const mk = (size: number, usage: number) => { const b = device.createBuffer({ size: Math.max(4, Math.ceil(size / 4) * 4), usage }); owned.push(b); return b; };
  const slotBufs = model.plan.slots.map((b) => mk(b, U.STORAGE | U.COPY_SRC | U.COPY_DST));
  const constBufs = new Map<string, GPUBuffer>();
  const bufOf = (name: string): GPUBuffer => {
    const v = model.vals.get(name)!;
    if (v.kind !== 'const') return slotBufs[model.slotOf.get(v.root)!];
    let b = constBufs.get(name);
    if (!b) {
      const src = v.f16 ? f32ToF16Bits(v.data!) : v.data!;
      const bytes = new Uint8Array(Math.max(4, Math.ceil(src.byteLength / 4) * 4)); bytes.set(new Uint8Array(src.buffer, src.byteOffset, src.byteLength));
      b = mk(bytes.byteLength, U.STORAGE | U.COPY_SRC | U.COPY_DST); device.queue.writeBuffer(b, 0, bytes); constBufs.set(name, b);
    }
    return b;
  };
  device.pushErrorScope('validation');
  type Step = { kind: 'k'; pipe: GPUComputePipeline; bg: GPUBindGroup; grid: [number, number, number] } | { kind: 'c'; src: GPUBuffer; so: number; dst: GPUBuffer; dO: number; n: number };
  const steps: Step[] = [];
  for (const l of model.launches) {
    if (l.kind === 'copy') { steps.push({ kind: 'c', src: bufOf(l.src), so: l.srcOff * 4, dst: bufOf(l.dst), dO: l.dstOff * 4, n: l.len * 4 }); continue; }
    const pipe = pipeline(device, l.kernel, l.f16);
    const ub = mk(256, U.UNIFORM | U.COPY_DST); device.queue.writeBuffer(ub, 0, l.u as Uint32Array<ArrayBuffer>);
    const entries: GPUBindGroupEntry[] = l.binds.map((b, i) => {
      const esz = model.vals.get(b.value)!.f16 ? 2 : 4;
      return { binding: i, resource: { buffer: bufOf(b.value), offset: b.start * esz, size: Math.ceil((b.len * esz) / 4) * 4 } };
    });
    entries.push({ binding: l.binds.length, resource: { buffer: ub } });
    steps.push({ kind: 'k', pipe, bg: device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries }), grid: l.grid });
  }
  const buildErr = await device.popErrorScope();
  if (buildErr) { for (const b of owned) b.destroy(); throw new Error(`WebGPU validation error while building the vision pipelines: ${buildErr.message}`); }
  let chain: Promise<unknown> = Promise.resolve();
  const runOnce = async (inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> => {
    checkInputs(model, inputs);
    for (const i of model.inputs) { const d = inputs[i.name].data; device.queue.writeBuffer(bufOf(i.name), 0, d.buffer as ArrayBuffer, d.byteOffset, d.byteLength); }
    const enc = device.createCommandEncoder();
    let pass: GPUComputePassEncoder | null = null;
    for (const s of steps) {
      if (s.kind === 'k') { if (!pass) pass = enc.beginComputePass(); pass.setPipeline(s.pipe); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(s.grid[0], s.grid[1], s.grid[2]); }
      else { if (pass) { pass.end(); pass = null; } enc.copyBufferToBuffer(s.src, s.so, s.dst, s.dO, s.n); }
    }
    if (pass) pass.end();
    const reads: { name: string; shape: number[]; rb: GPUBuffer }[] = [];
    for (const o of model.outputs) {
      const shape = model.shapeOf(o), n = Math.max(4, numel(shape) * 4);
      const rb = device.createBuffer({ size: n, usage: U.COPY_DST | U.MAP_READ });
      enc.copyBufferToBuffer(bufOf(o), 0, rb, 0, n); reads.push({ name: o, shape, rb });
    }
    device.pushErrorScope('validation');
    device.queue.submit([enc.finish()]);
    const runErr = await device.popErrorScope();
    if (runErr) { for (const r of reads) r.rb.destroy(); throw new Error(`WebGPU validation error during vision inference: ${runErr.message}`); }
    const out: Record<string, Tensor> = {};
    for (const r of reads) {
      await r.rb.mapAsync(GPUMapMode.READ);
      out[r.name] = { shape: r.shape, data: new Float32Array(r.rb.getMappedRange().slice(0, numel(r.shape) * 4)) };
      r.rb.unmap(); r.rb.destroy();
    }
    return out;
  };
  return {
    label: `webgpu${model.f16Weights ? '+f16w' : ''}`,
    run: (inputs) => { const p = chain.then(() => runOnce(inputs)); chain = p.catch(() => undefined); return p; },
    destroy: () => { for (const b of owned) b.destroy(); owned.length = 0; },
  };
}

// ════════════════════════════════════════ sliding window (MONAI semantics) ════════════════════════════════════════
/** Per-dim window starts, exactly monai.data.utils.dense_patch_slices ∘ _get_scan_interval (image already ≥ roi). */
export function slidingWindowStarts(image: number[], roi: number[], overlap: number | number[]): number[][] {
  const ov = Array.isArray(overlap) ? overlap : image.map(() => overlap);
  return image.map((sz, i) => {
    let interval = roi[i] === sz ? roi[i] : Math.trunc(roi[i] * (1 - ov[i]));
    if (interval <= 0) interval = 1;
    const num = Math.ceil(sz / interval);
    let scan = 1;
    for (let d = 0; d < num; d++) if (d * interval + roi[i] >= sz) { scan = d + 1; break; }
    return Array.from({ length: scan }, (_, idx) => { const s = idx * interval; return s - Math.max(s + roi[i] - sz, 0); });
  });
}
/** monai.data.utils.compute_importance_map (float32). */
export function computeImportanceMap(roi: number[], mode: 'gaussian' | 'constant', sigmaScale: number | number[] = 0.125): Float32Array {
  const n = numel(roi);
  if (mode === 'constant') return new Float32Array(n).fill(1);
  const ss = Array.isArray(sigmaScale) ? sigmaScale : roi.map(() => sigmaScale);
  const g = roi.map((p, i) => {
    const sigma = p * ss[i], den = fr(-2 * sigma ** 2);
    return Float32Array.from({ length: p }, (_, j) => { const x = fr(-(p - 1) / 2 + j); return fr(Math.exp(fr(fr(x * x) / den))); });
  });
  let m = Float32Array.from(g[0]);
  for (let i = 1; i < roi.length; i++) {
    const nm = new Float32Array(m.length * roi[i]);
    for (let a = 0; a < m.length; a++) for (let b = 0; b < roi[i]; b++) nm[a * roi[i] + b] = fr(m[a] * g[i][b]);
    m = nm;
  }
  let mn = Infinity; for (const v of m) mn = Math.min(mn, v);
  const floor = fr(Math.max(mn, 1e-3));
  for (let i = 0; i < m.length; i++) if (m[i] < floor) m[i] = floor;
  return m;
}
export interface SlidingWindowOptions { overlap?: number | number[]; mode?: 'gaussian' | 'constant'; sigmaScale?: number | number[]; cval?: number; onWindow?: (i: number, n: number) => void }
/** monai.inferers.sliding_window_inference (non-buffered, same-resolution output): symmetric constant pad when the
 *  image is smaller than roi, dense scan, importance-weighted accumulation in window order, divide by the count map, crop. */
export async function slidingWindowInference(input: Tensor, roi: number[], predictor: (w: Tensor) => Tensor | Promise<Tensor>, opts: SlidingWindowOptions = {}): Promise<Tensor> {
  const nsd = input.shape.length - 2;
  if (roi.length !== nsd) throw new Error(`roi rank ${roi.length} != spatial rank ${nsd}`);
  const overlap = opts.overlap ?? 0.25, mode = opts.mode ?? 'constant', cval = opts.cval ?? 0;
  for (const o of Array.isArray(overlap) ? overlap : [overlap]) if (!(o >= 0 && o < 1)) throw new Error(`overlap must be in [0, 1), got ${o}`);
  const [Bn, C, ...sp0] = input.shape;
  const padB = sp0.map((s, i) => Math.floor(Math.max(roi[i] - s, 0) / 2));
  const img = sp0.map((s, i) => Math.max(s, roi[i]));
  // pad (constant cval) into [B, C, ...img]
  const pst = contig(img), vol = numel(img), vol0 = numel(sp0);
  const padded = new Float32Array(Bn * C * vol).fill(cval);
  const coord = new Array(nsd).fill(0);
  for (let bc = 0; bc < Bn * C; bc++) for (let i = 0; i < vol0; i++) {
    let r = i, o = 0; for (let d = nsd - 1; d >= 0; d--) { coord[d] = r % sp0[d]; r = Math.floor(r / sp0[d]); o += (coord[d] + padB[d]) * pst[d]; }
    padded[bc * vol + o] = input.data[bc * vol0 + i];
  }
  const starts = slidingWindowStarts(img, roi, overlap);
  const wins: number[][] = [[]];
  for (const ds of starts) { const nx: number[][] = []; for (const w of wins) for (const s of ds) nx.push([...w, s]); wins.splice(0, wins.length, ...nx); }
  const imp = computeImportanceMap(roi, mode, opts.sigmaScale ?? 0.125), rv = numel(roi);
  // offsets of each roi voxel inside the padded image (relative to the window start)
  const rel = new Int32Array(rv);
  for (let i = 0; i < rv; i++) { let r = i, o = 0; for (let d = nsd - 1; d >= 0; d--) { const c = r % roi[d]; r = Math.floor(r / roi[d]); o += c * pst[d]; } rel[i] = o; }
  const count = new Float32Array(vol);
  for (const w of wins) { const b0 = w.reduce((a, s, d) => a + s * pst[d], 0); for (let i = 0; i < rv; i++) count[b0 + rel[i]] = fr(count[b0 + rel[i]] + imp[i]); }
  let outC = -1, output = new Float32Array(0);
  const total = wins.length * Bn;
  for (let idx = 0; idx < total; idx++) {
    const b = Math.floor(idx / wins.length), w = wins[idx % wins.length], b0 = w.reduce((a, s, d) => a + s * pst[d], 0);
    const win = new Float32Array(C * rv);
    for (let c = 0; c < C; c++) { const src = (b * C + c) * vol + b0; for (let i = 0; i < rv; i++) win[c * rv + i] = padded[src + rel[i]]; }
    const pred = await predictor({ shape: [1, C, ...roi], data: win });
    if (pred.shape.length !== nsd + 2 || pred.shape[0] !== 1 || pred.shape.slice(2).some((v, i) => v !== roi[i])) throw new Error(`predictor output [${pred.shape}] must be [1, C', ${roi}] (multi-resolution outputs are unsupported)`);
    if (outC < 0) { outC = pred.shape[1]; output = new Float32Array(Bn * outC * vol); }
    for (let c = 0; c < outC; c++) { const dst = (b * outC + c) * vol + b0; for (let i = 0; i < rv; i++) output[dst + rel[i]] = fr(output[dst + rel[i]] + fr(pred.data[c * rv + i] * imp[i])); }
    opts.onWindow?.(idx, total);
  }
  for (let bc = 0; bc < Bn * outC; bc++) for (let i = 0; i < vol; i++) output[bc * vol + i] = fr(output[bc * vol + i] / count[i]);
  // crop the padding back off
  const res = new Float32Array(Bn * outC * vol0);
  for (let bc = 0; bc < Bn * outC; bc++) for (let i = 0; i < vol0; i++) {
    let r = i, o = 0; for (let d = nsd - 1; d >= 0; d--) { const c = r % sp0[d]; r = Math.floor(r / sp0[d]); o += (c + padB[d]) * pst[d]; }
    res[bc * vol0 + i] = output[bc * vol + o];
  }
  return { shape: [Bn, outC, ...sp0], data: res };
}

// ════════════════════════════════════════ worker op family (vision_*) ════════════════════════════════════════
export const VISION_OPS = ['vision_caps', 'vision_load', 'vision_plan', 'vision_infer', 'vision_sliding_window', 'vision_unload'] as const;
export interface VisionContext { device: GPUDevice | null; takeStaged(id: string): Map<string, Uint8Array> | null }
interface Session { model: VisionModel; gpu: GpuRunner | null }
const SESSIONS = new Map<string, Session>();
const MAX_SESSIONS = 4;
const ID_RE = /^[A-Za-z0-9_.:-]{1,128}$/;
function b64d(s: string): Uint8Array { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
function b64e(u: Uint8Array): string { let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000)); return btoa(s); }
function tensorIn(t: unknown, what: string): Tensor {
  const o = t as { shape?: number[]; b64?: string };
  if (!o || !Array.isArray(o.shape) || typeof o.b64 !== 'string') throw new Error(`${what}: expected {shape, b64}`);
  const u = b64d(o.b64); if (u.byteLength % 4) throw new Error(`${what}: byte length not a multiple of 4`);
  const data = new Float32Array(u.buffer, 0, u.byteLength / 4);
  if (data.length !== numel(o.shape)) throw new Error(`${what}: ${data.length} values for shape [${o.shape}]`);
  return { shape: o.shape.map(Number), data };
}
const tensorOut = (t: Tensor) => ({ shape: t.shape, b64: b64e(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength)) });
function planSummary(p: MemoryPlan) {
  const { bindings, values, ...rest } = p;
  return { ...rest, values: Object.keys(values).length, maxBindingUsed: bindings.reduce((a, b) => Math.max(a, b.bytes), 0) };
}
function session(id: unknown): Session { const s = SESSIONS.get(String(id)); if (!s) throw new Error(`no vision model '${id}' loaded (vision_load first)`); return s; }

/** The worker's sealed `model` RPC routes every op starting with `vision_` here. */
export async function visionDispatch(op: string, p: Record<string, unknown>, ctx: VisionContext): Promise<Record<string, unknown>> {
  const dev = ctx.device;
  switch (op) {
    case 'vision_caps':
      return { ok: true, webgpu: !!dev, backend: dev ? 'webgpu' : 'cpu-reference', f16: !!dev?.features.has('shader-f16'), ops: [...SUPPORTED_OPS],
        limits: dev ? { maxStorageBufferBindingSize: dev.limits.maxStorageBufferBindingSize, maxBufferSize: dev.limits.maxBufferSize } : null,
        training: false, scope: 'inference + JEPA feature extraction' };
    case 'vision_load': {
      const id = String(p.id ?? '');
      if (!ID_RE.test(id)) throw new Error(`bad vision model id '${id}'`);
      const files = ctx.takeStaged(id) ?? new Map<string, Uint8Array>();
      let graph = p.graph as OpGraph | string | undefined;
      if (typeof graph === 'string') graph = JSON.parse(graph) as OpGraph;
      if (!graph) { const g = files.get('graph.json'); if (!g) throw new Error('vision_load: no graph (pass `graph` or stage graph.json)'); graph = JSON.parse(new TextDecoder().decode(g)) as OpGraph; }
      const st = files.get('model.safetensors');
      if (!st) throw new Error('vision_load: no model.safetensors staged (push_begin/push_chunk it under this id first)');
      const lim = Math.min(Number(p.maxBindingBytes ?? Infinity), dev?.limits.maxStorageBufferBindingSize ?? 128 * 1024 * 1024);
      const maxBuf = dev?.limits.maxBufferSize ?? 256 * 1024 * 1024;
      const f16 = !!p.f16 && !!dev?.features.has('shader-f16');
      const model = VisionModel.compile(graph, parseSafetensors(st), { maxBindingBytes: lim, maxBufferBytes: maxBuf, f16Weights: f16 });
      const prev = SESSIONS.get(id); if (prev) { prev.gpu?.destroy(); SESSIONS.delete(id); }
      while (SESSIONS.size >= MAX_SESSIONS) { const k = SESSIONS.keys().next().value as string; SESSIONS.get(k)?.gpu?.destroy(); SESSIONS.delete(k); }
      const gpu = dev ? await createGpuRunner(dev, model) : null;
      SESSIONS.set(id, { model, gpu });
      return { ok: true, id, backend: gpu ? gpu.label : 'cpu-reference', f16, inputs: model.inputs, outputs: model.outputs, plan: planSummary(model.plan) };
    }
    case 'vision_plan': return { ok: true, plan: planSummary(session(p.id).model.plan) };
    case 'vision_infer': {
      const s = session(p.id), t0 = performance.now();
      const raw = (p.inputs ?? {}) as Record<string, unknown>, inputs: Record<string, Tensor> = {};
      for (const [k, v] of Object.entries(raw)) inputs[k] = tensorIn(v, `input '${k}'`);
      const out = s.gpu ? await s.gpu.run(inputs) : runCpu(s.model, inputs);
      const outputs: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(out)) outputs[k] = tensorOut(v);
      return { ok: true, outputs, ms: performance.now() - t0, backend: s.gpu ? s.gpu.label : 'cpu-reference' };
    }
    case 'vision_sliding_window': {
      const s = session(p.id), t0 = performance.now();
      const x = tensorIn(p.input, 'input');
      const inName = String(p.input_name ?? s.model.inputs[0].name), outName = String(p.output ?? s.model.outputs[0]);
      if (!s.model.outputs.includes(outName)) throw new Error(`no output '${outName}'`);
      let windows = 0;
      const pred = async (w: Tensor) => { windows++; return (s.gpu ? await s.gpu.run({ [inName]: w }) : runCpu(s.model, { [inName]: w }))[outName]; };
      const roi = (p.roi as number[] | undefined) ?? s.model.inputs[0].shape.slice(2);
      const y = await slidingWindowInference(x, roi, pred, { overlap: (p.overlap as number | undefined) ?? 0.25, mode: (p.mode as 'gaussian' | 'constant' | undefined) ?? 'gaussian', sigmaScale: (p.sigma_scale as number | undefined) ?? 0.125, cval: Number(p.cval ?? 0) });
      return { ok: true, output: tensorOut(y), windows, ms: performance.now() - t0 };
    }
    case 'vision_unload': { const s = SESSIONS.get(String(p.id)); s?.gpu?.destroy(); SESSIONS.delete(String(p.id)); return { ok: true }; }
  }
  throw new Error(`unsupported vision op '${op}'`);
}

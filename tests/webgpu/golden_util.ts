// Shared golden-loading + comparison helpers for the WebGPU vision tests. Pure TS with no Node or Deno APIs, so
// vitest (Node), the Deno WebGPU test and the Playwright browser entry can all use it. Callers do their own I/O.

export interface TJson { shape: number[]; b64: string }
export interface KernelCase {
  name: string;
  graph: { version: number; inputs: { name: string; shape: number[] }[]; nodes: { op: string; inputs: (string | null)[]; attrs: Record<string, unknown>; output: string }[]; outputs: string[] };
  tensors: Record<string, TJson>;
  inputs: Record<string, TJson>;
  expected: Record<string, TJson>;
  tol: number;
  ops: string[];
}
export interface KernelGoldens { version: number; torch: string; cases: KernelCase[] }

export function b64ToF32(s: string): Float32Array {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer, 0, u8.byteLength >> 2);
}

export function f32ToB64(a: Float32Array): string {
  const u8 = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000));
  return btoa(s);
}

export function tensorOf(t: TJson): { shape: number[]; data: Float32Array } {
  return { shape: t.shape.slice(), data: b64ToF32(t.b64) };
}

export function tensorMap(ts: Record<string, TJson>): Map<string, { shape: number[]; data: Float32Array }> {
  const m = new Map<string, { shape: number[]; data: Float32Array }>();
  for (const [k, v] of Object.entries(ts)) m.set(k, tensorOf(v));
  return m;
}

/** ‖a−b‖∞ / ‖b‖∞: the fp32 "1e-5 rel" criterion, normalised by the tensor's scale so near-zero outputs caused by
 *  cancellation do not produce meaningless per-element ratios. */
export function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity;
  let num = 0, den = 0;
  for (let i = 0; i < b.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (!(d <= num)) num = d; // also catches NaN
    const m = Math.abs(b[i]);
    if (m > den) den = m;
  }
  return num / Math.max(den, 1e-30);
}

export function shapeEq(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

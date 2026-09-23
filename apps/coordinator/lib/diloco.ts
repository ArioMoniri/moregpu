// DiLoCo outer loop (ADR-0106) — identical math to apps/worker/moregpu_worker/train/diloco.py.
// Outer Nesterov on Δ = global − avg:  v ← μv + Δ ;  global ← global − η(Δ + μv)   (PyTorch SGD nesterov form)

export type Tensors = Map<string, Float32Array>;
export interface Weighted { tensors: Tensors; weight: number }

export class OuterState {
  constructor(public global: Tensors, public momentum: Tensors, public round = 0) {}
  static init(g: Tensors): OuterState {
    const gl = new Map<string, Float32Array>(), m = new Map<string, Float32Array>();
    for (const [k, v] of g) { gl.set(k, Float32Array.from(v)); m.set(k, new Float32Array(v.length)); }
    return new OuterState(gl, m, 0);
  }
}

export function weightedAverage(items: Weighted[]): Tensors {
  const total = items.reduce((s, x) => s + x.weight, 0);
  if (!items.length || !(total > 0)) throw new Error('weightedAverage needs positive total weight');
  const out = new Map<string, Float32Array>();
  for (const k of items[0]!.tensors.keys()) {
    const n = items[0]!.tensors.get(k)!.length, acc = new Float64Array(n);
    for (const { tensors, weight } of items) {
      const a = tensors.get(k);
      if (!a || a.length !== n) throw new Error(`${k}: length mismatch`);
      const w = weight / total;
      for (let i = 0; i < n; i++) acc[i]! += a[i]! * w;
    }
    out.set(k, Float32Array.from(acc));
  }
  return out;
}

/** Running statistics (e.g. BatchNorm buffers) are averaged and never outer-stepped (extrapolation could make a variance negative). */
export const BUFFER_PREFIX = 'buffer:';

export function outerStep(st: OuterState, avg: Tensors, lr: number, mom: number, start?: Tensors): OuterState {
  for (const [k, g] of st.global) {
    if (k.startsWith(BUFFER_PREFIX)) { g.set(avg.get(k)!); continue; }
    // Δ is taken from where the workers actually started (the decoded broadcast when it was lossy, e.g. bf16),
    // so the broadcast rounding residual is not fed into the outer momentum every round.
    const s0 = start?.get(k);
    const v = st.momentum.get(k)!, a = avg.get(k)!;
    for (let i = 0; i < g.length; i++) {
      const d = Math.fround((s0 ? s0[i]! : g[i]!) - a[i]!);
      v[i] = Math.fround(Math.fround(mom * v[i]!) + d);
      g[i] = Math.fround(g[i]! - Math.fround(lr * Math.fround(d + Math.fround(mom * v[i]!))));
    }
  }
  st.round++;
  return st;
}

export function dropNonFinite<T extends { id: string; tensors: Tensors }>(items: T[]): { kept: T[]; dropped: string[] } {
  const kept: T[] = [], dropped: string[] = [];
  for (const it of items) {
    const finite = [...it.tensors.values()].every((a) => a.every((x) => Number.isFinite(x)));
    if (finite) kept.push(it); else dropped.push(it.id);
  }
  return { kept, dropped };
}

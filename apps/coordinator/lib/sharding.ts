// Deterministic sample assignment (ADR-0107) — identical to apps/worker/moregpu_worker/train/sharding.py.
const M = (1n << 64n) - 1n;

function* splitmix(state: bigint): Generator<bigint> {
  for (;;) {
    state = (state + 0x9E3779B97F4A7C15n) & M;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & M;
    z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & M;
    yield z ^ (z >> 31n);
  }
}

export function permutation(n: number, seed: number, epoch: number): number[] {
  const rng = splitmix((BigInt(seed) * 0x100000001B3n + BigInt(epoch)) & M);
  const p = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Number(rng.next().value! % BigInt(i + 1));
    [p[i], p[j]] = [p[j]!, p[i]!];
  }
  return p;
}

export interface StreamState { n: number; seed: number; epoch: number; cursor: number }
export class SampleStream {
  private perm: number[];
  constructor(public n: number, public seed: number, public epoch = 0, public cursor = 0) {
    if (n <= 0) throw new Error('manifest is empty');
    this.perm = permutation(n, seed, epoch);
  }
  take(k: number): number[] {
    const out: number[] = [];
    while (out.length < k) {
      if (this.cursor >= this.n) { this.epoch++; this.cursor = 0; this.perm = permutation(this.n, this.seed, this.epoch); }
      const m = Math.min(k - out.length, this.n - this.cursor);
      out.push(...this.perm.slice(this.cursor, this.cursor + m)); this.cursor += m;
    }
    return out;
  }
  state(): StreamState { return { n: this.n, seed: this.seed, epoch: this.epoch, cursor: this.cursor }; }
  static fromState(s: StreamState): SampleStream { return new SampleStream(s.n, s.seed, s.epoch, s.cursor); }
}

export function split(indices: number[], sizes: number[]): number[][] {
  if (sizes.reduce((a, b) => a + b, 0) !== indices.length) throw new Error('sizes do not sum to the index count');
  const out: number[][] = []; let o = 0;
  for (const s of sizes) { out.push(indices.slice(o, o + s)); o += s; }
  return out;
}

export function allocate(perWorker: number, n: number, speeds: number[], mode: 'fixed' | 'proportional' = 'fixed', remaining?: number): number[] {
  const total = perWorker * n; let alloc: number[];
  if (mode === 'fixed') alloc = Array(n).fill(perWorker);
  else {
    const s = speeds.map((x) => Math.max(Number(x), 1e-9)), sum = s.reduce((a, b) => a + b, 0);
    const raw = s.map((x) => (total * x) / sum);
    alloc = raw.map((r) => Math.max(1, Math.floor(r)));
    const order = [...Array(n).keys()].sort((a, b) => (raw[b]! - Math.floor(raw[b]!)) - (raw[a]! - Math.floor(raw[a]!)));
    let k = 0; while (alloc.reduce((a, b) => a + b, 0) < total) { alloc[order[k % n]!]!++; k++; }
    while (alloc.reduce((a, b) => a + b, 0) > total) { let i = 0; for (let j = 1; j < n; j++) if (alloc[j]! > alloc[i]!) i = j; alloc[i]!--; }
  }
  const tot = alloc.reduce((a, b) => a + b, 0);
  if (remaining !== undefined && remaining < tot) {
    alloc = alloc.map((a) => Math.floor((a * remaining) / tot));
    let i = 0; while (alloc.reduce((a, b) => a + b, 0) < remaining) { alloc[i % n]!++; i++; }
  }
  return alloc;
}

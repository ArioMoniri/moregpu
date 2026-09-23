// Distributed vision batch inference (ADR-0112): a pull-based work queue over the workers that hold the model.
// Heterogeneity is handled by pull rate (a fast worker simply takes more items), failed items are retried on another
// worker (up to maxAttempts), a worker that fails `deadAfter` times in a row is retired, and once the queue drains an
// idle worker duplicates ("steals") the oldest in-flight item that has been running longer than stealAfterMs — the first
// result wins, so one slow node never sets the batch's wall time.

export interface RpcResult { ok: boolean; data?: Record<string, unknown>; error?: string }
export type Rpc = (workerId: string, op: string, payload: Record<string, unknown>) => Promise<RpcResult>;
export interface BatchItem { ref: unknown; out: string; mask?: unknown }
export interface BatchOpts { tta?: string; overlap?: number; sw_batch?: number; blend?: string; normalize?: unknown;
  maxAttempts?: number; deadAfter?: number; stealAfterMs?: number; telemetry?: (r: Record<string, unknown>) => void; now?: () => number }
export interface ItemResult { index: number; worker?: string; ok: boolean; data?: Record<string, unknown>; error?: string; attempts: number; ms?: number }

export class VisionBatch {
  results: ItemResult[] = [];
  status: 'pending' | 'running' | 'done' | 'failed' | 'cancelled' = 'pending';
  deadWorkers: string[] = [];
  retries = 0; stolen = 0;
  private queue: number[]; private attempts: number[]; private finished = new Set<number>();
  private inflight = new Map<number, { t0: number; workers: Set<string> }>();
  private cancelled = false; private t0 = 0;
  private now: () => number;
  perWorker = new Map<string, number>();

  constructor(public id: string, public model: string, public items: BatchItem[], public workers: string[], private rpc: Rpc, private o: BatchOpts = {}) {
    this.queue = items.map((_, i) => i); this.attempts = items.map(() => 0);
    this.now = o.now ?? (() => Date.now());
  }

  cancel() { this.cancelled = true; }

  progress() {
    return { id: this.id, model: this.model, status: this.status, items: this.items.length, done: this.results.filter((r) => r.ok).length,
      failed: this.results.filter((r) => !r.ok).length, inflight: this.inflight.size, retries: this.retries, stolen: this.stolen,
      dead_workers: this.deadWorkers, per_worker: Object.fromEntries(this.perWorker) };
  }

  private next(w: string): number | undefined {
    const i = this.queue.shift();
    if (i !== undefined) return i;
    // steal: oldest in-flight item not already running on this worker, running longer than the threshold
    const lim = this.o.stealAfterMs ?? 30_000;
    let best: number | undefined, bestT = Infinity;
    for (const [idx, f] of this.inflight) if (!f.workers.has(w) && this.now() - f.t0 > lim && f.t0 < bestT && f.workers.size < 2) { best = idx; bestT = f.t0; }
    if (best !== undefined) this.stolen++;
    return best;
  }

  private async loop(w: string) {
    let consecutive = 0;
    const deadAfter = this.o.deadAfter ?? 2, maxAttempts = this.o.maxAttempts ?? 3;
    while (!this.cancelled) {
      const idx = this.next(w);
      if (idx === undefined) {
        if (!this.inflight.size || this.finished.size === this.items.length) return;
        await new Promise((r) => setTimeout(r, Math.min(50, (this.o.stealAfterMs ?? 30_000) / 4)));
        continue;
      }
      const f = this.inflight.get(idx) ?? { t0: this.now(), workers: new Set<string>() };
      f.workers.add(w); this.inflight.set(idx, f);
      const it = this.items[idx]!; const t = this.now();
      this.attempts[idx]!++;
      const r = await this.rpc(w, 'vision_predict', { id: this.model, ref: it.ref, out: it.out, mask: it.mask, tta: this.o.tta ?? 'none',
        overlap: this.o.overlap ?? 0.5, sw_batch: this.o.sw_batch ?? 8, blend: this.o.blend ?? 'gaussian', normalize: this.o.normalize });
      f.workers.delete(w);
      if (this.finished.has(idx)) continue;                // a stolen duplicate already finished it
      if (r.ok) {
        this.finished.add(idx); this.inflight.delete(idx); consecutive = 0;
        this.perWorker.set(w, (this.perWorker.get(w) ?? 0) + 1);
        this.results.push({ index: idx, worker: w, ok: true, data: r.data, attempts: this.attempts[idx]!, ms: this.now() - t });
        continue;
      }
      consecutive++;
      if (!f.workers.size) {
        if (this.attempts[idx]! < maxAttempts) { this.retries++; this.inflight.delete(idx); this.queue.push(idx); }
        else { this.finished.add(idx); this.inflight.delete(idx); this.results.push({ index: idx, worker: w, ok: false, error: r.error, attempts: this.attempts[idx]! }); }
      }
      if (consecutive >= deadAfter && /disconnect|closed|timeout/i.test(r.error ?? '')) { this.deadWorkers.push(w); return; }
    }
  }

  async run() {
    this.status = 'running'; this.t0 = this.now();
    // finish as soon as every item has a result — a straggler whose item was stolen and completed elsewhere is not waited on
    const allLoops = Promise.all(this.workers.map((w) => this.loop(w)));
    const allDone = new Promise<void>((res) => { const t = setInterval(() => { if (this.finished.size === this.items.length || this.cancelled) { clearInterval(t); res(); } }, 5); allLoops.then(() => { clearInterval(t); res(); }); });
    await Promise.race([allLoops, allDone]);
    // everyone died with items left: report them
    for (const idx of this.queue) if (!this.finished.has(idx)) this.results.push({ index: idx, ok: false, error: 'no live worker left', attempts: this.attempts[idx]! });
    this.results.sort((a, b) => a.index - b.index);
    const p = this.progress();
    this.status = this.cancelled ? 'cancelled' : p.failed && !p.done ? 'failed' : 'done';
    const wall = (this.now() - this.t0) / 1000;
    this.o.telemetry?.({ schema: 'moregpu.telemetry/1', kind: 'job', ts: new Date().toISOString(), job: this.id, op: 'vision_batch', session: null,
      worker: null, items: this.items.length, items_done: p.done, items_failed: p.failed, retries: this.retries, wall_s: wall });
    return { ...this.progress(), wall_s: wall };
  }
}

import { describe, it, expect } from 'vitest';
import { VisionBatch, type Rpc } from './vision_batch.ts';

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
function fleet(spec: Record<string, { ms: number; fail?: (item: number, attempt: number) => boolean; dieAfter?: number }>) {
  const done: Record<string, number[]> = {}; const calls: Record<string, number> = {};
  const rpc: Rpc = async (w, op, p) => {
    const s = spec[w]!; calls[w] = (calls[w] ?? 0) + 1;
    if (s.dieAfter !== undefined && calls[w]! > s.dieAfter) return { ok: false, error: 'worker disconnected' };
    await sleep(s.ms);
    const item = Number((p.ref as { uri: string }).uri.split('#')[1]);
    if (s.fail?.(item, calls[w]!)) return { ok: false, error: 'boom' };
    (done[w] ??= []).push(item);
    return { ok: true, data: { path: `/out/${item}.npy`, dice: { '1': 0.9 }, timings: { compute_s: s.ms / 1000 } } };
  };
  return { rpc, done };
}
const items = (n: number) => Array.from({ length: n }, (_, i) => ({ ref: { uri: `file://v#${i}` }, out: `case${i}` }));

describe('VisionBatch (pull-based work queue with stealing + churn retry)', () => {
  it('fast workers take more items; all items complete once', async () => {
    const f = fleet({ fast: { ms: 2 }, slow: { ms: 20 } });
    const b = new VisionBatch('j', 'm', items(20), ['fast', 'slow'], f.rpc, {});
    const r = await b.run();
    expect(r.done).toBe(20); expect(r.failed).toBe(0);
    expect(f.done.fast!.length).toBeGreaterThan(f.done.slow!.length);
    expect(new Set(b.results.map((x) => x.index)).size).toBe(20);
  });
  it('a killed worker: its items are retried elsewhere', async () => {
    const f = fleet({ a: { ms: 3 }, b: { ms: 3, dieAfter: 2 } });
    const b = new VisionBatch('j', 'm', items(12), ['a', 'b'], f.rpc, {});
    const r = await b.run();
    expect(r.done).toBe(12); expect(r.retries).toBeGreaterThan(0);
    expect(b.deadWorkers).toContain('b');
  });
  it('work stealing: when the queue drains, an idle fast worker duplicates the straggler and the first result wins', async () => {
    const f = fleet({ fast: { ms: 2 }, snail: { ms: 400 } });
    const b = new VisionBatch('j', 'm', items(6), ['fast', 'snail'], f.rpc, { stealAfterMs: 20 });
    const t0 = Date.now(); const r = await b.run();
    expect(r.done).toBe(6); expect(r.stolen).toBeGreaterThan(0);
    expect(Date.now() - t0).toBeLessThan(380);
  });
  it('an item failing everywhere is reported after max attempts', async () => {
    const f = fleet({ a: { ms: 1, fail: (i) => i === 3 }, b: { ms: 1, fail: (i) => i === 3 } });
    const b = new VisionBatch('j', 'm', items(5), ['a', 'b'], f.rpc, { maxAttempts: 3 });
    const r = await b.run();
    expect(r.done).toBe(4); expect(r.failed).toBe(1);
    expect(b.results.find((x) => x.index === 3)!.error).toMatch(/boom/);
  });
  it('emits a job telemetry record and supports cancel', async () => {
    const recs: any[] = [];
    const f = fleet({ a: { ms: 5 } });
    const b = new VisionBatch('j', 'm', items(50), ['a'], f.rpc, { telemetry: (r) => recs.push(r) });
    const p = b.run(); await sleep(20); b.cancel(); const r = await p;
    expect(r.status).toBe('cancelled'); expect(r.done).toBeLessThan(50);
    expect(recs.at(-1).kind).toBe('job'); expect(recs.at(-1).items).toBe(50);
  });
});

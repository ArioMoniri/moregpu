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

describe('VisionBatch with a per-worker-kind task callback (mixed torch + WebGPU fleet)', () => {
  it('each worker gets the op/payload its kind needs; items are served by both kinds; results keep item order', async () => {
    const kind: Record<string, 'torch' | 'webgpu'> = { t1: 'torch', g1: 'webgpu' };
    const seen: Array<[string, string, any]> = [];
    const rpc: Rpc = async (w, op, p) => {
      seen.push([w, op, p]);
      await sleep(kind[w] === 'torch' ? 3 : 5);
      const i = kind[w] === 'torch' ? p.i : p.inputs.x.i;
      return { ok: true, data: { shape: [1], data: `y${i}`, kind: kind[w] } };
    };
    const its = Array.from({ length: 10 }, (_, i) => ({ i }));
    const b = new VisionBatch('j', 'm', its, ['t1', 'g1'], rpc, {
      task: (w, it) => kind[w] === 'torch'
        ? { op: 'vision_infer', payload: { id: 'm', i: (it as { i: number }).i } }
        : { op: 'vision_infer', payload: { id: 'm', inputs: { x: { i: (it as { i: number }).i } } } },
    });
    const r = await b.run();
    expect(r.done).toBe(10); expect(r.failed).toBe(0);
    expect(b.results.map((x) => x.data!.data)).toEqual(its.map((_, i) => `y${i}`));
    expect(r.per_worker.t1! + r.per_worker.g1!).toBe(10);
    expect(r.per_worker.t1).toBeGreaterThan(0); expect(r.per_worker.g1).toBeGreaterThan(0);
    expect(seen.every(([w, op, p]) => op === 'vision_infer' && (kind[w] === 'torch' ? 'i' in p : 'inputs' in p))).toBe(true);
  });
  it('without a task callback the default op is still vision_predict (existing behaviour)', async () => {
    const ops: string[] = [];
    const rpc: Rpc = async (_w, op) => { ops.push(op); return { ok: true, data: {} }; };
    await new VisionBatch('j', 'm', items(3), ['a'], rpc).run();
    expect(ops).toEqual(['vision_predict', 'vision_predict', 'vision_predict']);
  });
});

describe('VisionBatch split=tiles (one volume across all workers)', () => {
  it('each item is split into one part per worker, merged once, part failures retried on another worker', async () => {
    const calls: Array<[string, string, any]> = [];
    let failOnce = true;
    const rpc: Rpc = async (w, op, p) => {
      calls.push([w, op, p]);
      await new Promise((r) => setTimeout(r, 2));
      if (op === 'vision_predict_part') {
        if (w === 'b' && failOnce) { failOnce = false; return { ok: false, error: 'worker disconnected' }; }
        return { ok: true, data: { k: p.part[0], n: p.part[1], n_units: 1, kind: '3d', timings: { compute_s: 0.001 } } };
      }
      if (op === 'vision_merge_write') return { ok: true, data: { path: `/out/${p.out}.npy`, n_parts: p.parts.length, dice: { '1': 0.9 } } };
      return { ok: false, error: 'unexpected' };
    };
    const b = new VisionBatch('j', 'm', items(2), ['a', 'b', 'c'], rpc, { split: 'tiles' });
    const r = await b.run();
    expect(r.done).toBe(2); expect(r.failed).toBe(0);
    const merges = calls.filter((c) => c[1] === 'vision_merge_write');
    expect(merges).toHaveLength(2);
    expect(merges[0]![2].parts.map((x: any) => x.k).sort()).toEqual([0, 1, 2]);   // 3 live workers → 3 parts
    expect(merges[1]![2].parts.map((x: any) => x.k).sort()).toEqual([0, 1]);      // b died → re-split across 2
    expect(b.deadWorkers).toEqual(['b']);
    expect(b.results[0]!.data!.latency_s).toBeGreaterThan(0);
    expect(r.retries).toBeGreaterThan(0);
  });
});

describe('VisionBatch robustness (review P2)', () => {
  it('a fast-failing worker is retired after repeated failures and cannot burn every retry', async () => {
    const f = fleet({ bad: { ms: 0, fail: () => true }, good: { ms: 3 } });
    const b = new VisionBatch('j', 'm', items(10), ['bad', 'good'], f.rpc, { maxAttempts: 3 });
    const r = await b.run();
    expect(r.done).toBe(10); expect(r.failed).toBe(0); expect(b.deadWorkers).toContain('bad');
  });
  it('tiled mode refuses parts whose geometry disagrees before merging', async () => {
    const calls: string[] = [];
    const rpc: Rpc = async (w, op, p) => {
      calls.push(op);
      if (op === 'vision_predict_part') return { ok: true, data: { k: p.part[0], n: p.part[1], n_units: 1, kind: '3d', vol_shape: w === 'evil' ? [9999, 9999, 9999] : [8, 8, 8] } };
      return { ok: true, data: { path: '/o.npy' } };
    };
    const b = new VisionBatch('j', 'm', items(1), ['a', 'evil'], rpc, { split: 'tiles', maxAttempts: 1 });
    const r = await b.run();
    expect(r.failed).toBe(1); expect(calls).not.toContain('vision_merge_write');
  });
});

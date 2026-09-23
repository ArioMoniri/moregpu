import { describe, it, expect } from 'vitest';
import { TrainSession, type Rpc, type RpcResult, type CheckpointStore, type SessionConfig } from './train_session.ts';
import { decodeTensors, encodeTensors, b64ToBytes, bytesToB64, chunkBytes, concatBytes, type WireHeader } from './tensorwire.ts';

// A fake worker: state = {w: Float32Array(4)}; inner step moves each element by -lr * (#refs) * bias.
class FakeWorker {
  state = new Map<string, Float32Array>([['w', new Float32Array([1, 2, 3, 4])]]);
  out: Uint8Array[] = []; inParts: Uint8Array[] = []; inHdr: WireHeader | null = null;
  calls: string[] = []; dead = false; poison = false; failPut = false;
  constructor(public bias: number, public speed = 1) {}
  async handle(op: string, p: any): Promise<RpcResult> {
    this.calls.push(op);
    if (this.dead) return { ok: false, error: 'disconnected' };
    const enc = async () => {
      const { header, blob } = await encodeTensors(this.state, { w: [4] }, p.sync_dtype === 'bf16' ? 'bf16' : 'f32');
      this.out = chunkBytes(blob, p.chunk_bytes ?? 1 << 20);
      return { header, nchunks: this.out.length, chunk0: bytesToB64(this.out[0]!) };
    };
    switch (op) {
      case 'task_init': return { ok: true, data: { describe: { task: p.task } } };
      case 'task_state_get': return { ok: true, data: await enc() };
      case 'task_inner': {
        const w = this.state.get('w')!;
        for (let i = 0; i < w.length; i++) w[i] = this.poison ? NaN : w[i]! - p.lr * p.refs.length * this.bias;
        return { ok: true, data: { report: { samples: p.refs.length, losses: [1, 0.5], timings: { compute_s: p.refs.length / this.speed, data_s: 0, serialize_s: 0 }, metrics: {} }, ...(await enc()) } };
      }
      case 'task_state_chunk': return { ok: true, data: { data: bytesToB64(this.out[p.k]!) } };
      case 'task_state_put': {
        if (this.failPut) return { ok: false, error: 'put failed' };
        if (p.k === 0) { this.inHdr = p.header; this.inParts = []; }
        this.inParts.push(b64ToBytes(p.data));
        if (p.k < p.n - 1) return { ok: true, data: { applied: false } };
        this.state = await decodeTensors(this.inHdr!, concatBytes(this.inParts));
        return { ok: true, data: { applied: true, deserialize_s: 0 } };
      }
      case 'task_after_outer': return { ok: true, data: { target_sha256: 'same' } };
      case 'task_eval': return { ok: true, data: { metrics: { loss: 0.1 } } };
      case 'task_close': return { ok: true, data: {} };
      default: return { ok: false, error: `unknown ${op}` };
    }
  }
}

function fleet(...ws: FakeWorker[]) {
  const map = new Map(ws.map((w, i) => [`w${i}`, w]));
  const rpc: Rpc = (id, op, p) => map.get(id)!.handle(op, p);
  return { map, rpc, ids: [...map.keys()] };
}
const CFG: SessionConfig = { task: 'toy', manifest_len: 32, batch: 2, inner_steps: 2, lr: 0.1, outer_lr: 1, outer_momentum: 0, seed: 1, chunk_bytes: 6 };

describe('TrainSession', () => {
  it('init pulls worker 0 state and broadcasts it (chunked)', async () => {
    const f = fleet(new FakeWorker(1), new FakeWorker(1));
    f.map.get('w1')!.state.set('w', new Float32Array([9, 9, 9, 9]));
    const s = new TrainSession('s', CFG, f.ids, f.rpc);
    const r = await s.init();
    expect(r.params).toBe(4);
    expect(Array.from(f.map.get('w1')!.state.get('w')!)).toEqual([1, 2, 3, 4]);
    expect(f.map.get('w1')!.calls.filter((c) => c === 'task_state_put').length).toBeGreaterThan(1);
  });

  it('weights the average by samples seen and counts samples exactly', async () => {
    const f = fleet(new FakeWorker(1), new FakeWorker(3));
    const s = new TrainSession('s', { ...CFG }, f.ids, f.rpc);
    await s.init();
    const r = await s.runRound();
    // each worker got 4 refs; w0 moved -0.4, w1 moved -1.2; equal samples → mean −0.8; η=1 μ=0 → global = mean
    expect(Array.from(s.st!.global.get('w')!).map((x) => +x.toFixed(5))).toEqual([0.2, 1.2, 2.2, 3.2]);
    expect(r.samples).toBe(8); expect(s.samplesSeen).toBe(8);
    expect(Array.from(f.map.get('w0')!.state.get('w')!)).toEqual(Array.from(s.st!.global.get('w')!));
  });

  it('proportional allocation follows measured speed after the first round', async () => {
    const f = fleet(new FakeWorker(1, 1), new FakeWorker(1, 3));
    const s = new TrainSession('s', { ...CFG, alloc: 'proportional' }, f.ids, f.rpc);
    await s.init(); await s.runRound();
    const calls: number[] = [];
    const orig = f.rpc;
    const spy: Rpc = async (id, op, p) => { if (op === 'task_inner') calls.push((p.refs as number[]).length); return orig(id, op, p); };
    (s as any).rpc = spy;
    await s.runRound();
    expect(calls.reduce((a, b) => a + b)).toBe(8);
    expect(calls[1]!).toBeGreaterThan(calls[0]!);
  });

  it('stops exactly at target_samples', async () => {
    const f = fleet(new FakeWorker(1), new FakeWorker(1));
    const s = new TrainSession('s', { ...CFG, target_samples: 13 }, f.ids, f.rpc);
    await s.init();
    while (!s.isDone()) await s.runRound();
    expect(s.samplesSeen).toBe(13);
    await expect(s.runRound()).rejects.toThrow(/stopping rule/);
  });

  it('drops a non-finite worker from the average but resyncs it', async () => {
    const f = fleet(new FakeWorker(1), new FakeWorker(1));
    const s = new TrainSession('s', CFG, f.ids, f.rpc);
    await s.init();
    f.map.get('w1')!.poison = true;
    const r = await s.runRound();
    expect(r.dropped_nonfinite).toEqual(['w1']);
    expect(Number.isFinite(f.map.get('w1')!.state.get('w')![0]!)).toBe(true);
    expect(s.samplesSeen).toBe(4);
  });

  it('a dead worker is dropped; all dead → error and the sample stream is not consumed', async () => {
    const f = fleet(new FakeWorker(1), new FakeWorker(1));
    const s = new TrainSession('s', CFG, f.ids, f.rpc);
    await s.init();
    f.map.get('w1')!.dead = true;
    const r = await s.runRound();
    expect(r.dropped).toContain('w1'); expect(s.workers).toEqual(['w0']);
    f.map.get('w0')!.dead = true;
    const before = s.stream.state();
    await expect(s.runRound()).rejects.toThrow();
    expect(s.stream.state()).toEqual(before);
  });

  it('emits telemetry whose breakdown sums to the round wall time', async () => {
    let t = 0; const now = () => (t += 10);
    const recs: any[] = [];
    const f = fleet(new FakeWorker(1, 1000), new FakeWorker(1, 1000));   // compute 4 ms < wall (realistic)
    const s = new TrainSession('s', CFG, f.ids, f.rpc, { now, telemetry: (r) => recs.push(r) });
    await s.init(); await s.runRound();
    const wr = recs.filter((r) => r.kind === 'worker_round');
    expect(wr).toHaveLength(2);
    for (const r of wr) {
      expect(r.schema).toBe('moregpu.telemetry/1');
      const sum = r.compute_s + r.data_s + r.serialize_s + r.network_s + r.wait_s;
      expect(Math.abs(sum - r.wall_s)).toBeLessThan(1e-9 + 0.05 * r.wall_s);
      expect(r.config_hash).toMatch(/^[0-9a-f]{64}$/);
    }
    expect(recs.filter((r) => r.kind === 'round')).toHaveLength(1);
  });

  it('checkpoint → resume continues bit-identically', async () => {
    const files = new Map<string, Uint8Array>();
    const store: CheckpointStore = {
      write: async (n, d) => { files.set(n, typeof d === 'string' ? new TextEncoder().encode(d) : d); },
      read: async (n) => files.get(n) ?? null,
      list: async (p) => [...files.keys()].filter((k) => k.startsWith(p)),
      remove: async (n) => { files.delete(n); },
    };
    const cfg = { ...CFG, outer_lr: 0.7, outer_momentum: 0.9, checkpoint_every: 1, keep_checkpoints: 2 };
    const a = fleet(new FakeWorker(1), new FakeWorker(2));
    const s = new TrainSession('s', cfg, a.ids, a.rpc, { store });
    await s.init(); await s.runRound(); await s.runRound(); await s.runRound();
    expect([...files.keys()].filter((k) => k.endsWith('.json'))).toHaveLength(2);
    // reference: continue the original for one more round
    await s.runRound(); const ref = Array.from(s.st!.global.get('w')!);
    // resume from the round-3 checkpoint on fresh workers
    files.forEach((_, k) => { if (k.includes('round-000004')) files.delete(k); });
    const b = fleet(new FakeWorker(1), new FakeWorker(2));
    const r = await TrainSession.resume('s', store, b.rpc, {}, b.ids);
    expect(r.round).toBe(3); expect(r.samplesSeen).toBe(24);
    await r.runRound();
    expect(Array.from(r.st!.global.get('w')!)).toEqual(ref);
  });

  it('bf16 broadcast keeps the decoded broadcast as the reference and eval/close work', async () => {
    const f = fleet(new FakeWorker(1));
    const s = new TrainSession('s', { ...CFG, broadcast_dtype: 'bf16', eval: { refs: [1], kind: 'loss', every: 1 } }, f.ids, f.rpc);
    await s.init();
    const r = await s.runRound();
    expect(r.eval).toEqual({ loss: 0.1 });
    expect(s.lastBroadcast!.get('w')![0]).toBe(f.map.get('w0')!.state.get('w')![0]);
    await s.close(); expect(s.status).toBe('closed');
    expect(s.describe().round).toBe(1);
  });

  it('validates config and serializes rounds', async () => {
    expect(() => new TrainSession('x', { ...CFG, manifest_len: 0 }, ['a'], async () => ({ ok: true }))).toThrow();
    expect(() => new TrainSession('x', CFG, [], async () => ({ ok: true }))).toThrow();
    const f = fleet(new FakeWorker(1));
    const s = new TrainSession('s', CFG, f.ids, f.rpc); await s.init();
    const p = s.runRound(); await expect(s.runRound()).rejects.toThrow(/serialized/); await p;
  });
});

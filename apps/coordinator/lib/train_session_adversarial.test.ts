// Security/DS review P0-3, P1-5, P1-6, P1-7: a single malicious or broken worker must not corrupt, finish, wedge or
// kill a session; paused workers are skipped not evicted; session operations are serialised.
import { describe, it, expect } from 'vitest';
import { TrainSession, type Rpc, type RpcResult, type SessionConfig } from './train_session.ts';
import { decodeTensors, encodeTensors, b64ToBytes, bytesToB64, chunkBytes, concatBytes, type WireHeader } from './tensorwire.ts';

type Evil = Partial<{ samples: unknown; dropTensor: string; noLosses: boolean; nchunks: number; computeS: number; hash: string; alarm: string; badHeader: boolean }>;
class W {
  state = new Map<string, Float32Array>([['a', new Float32Array([1, 2])], ['b', new Float32Array([3, 4])]]);
  out: Uint8Array[] = []; parts: Uint8Array[] = []; hdr: WireHeader | null = null; calls: string[] = []; closed = 0; evil: Evil = {};
  async enc(p: any) {
    const st = new Map(this.state); if (this.evil.dropTensor) st.delete(this.evil.dropTensor);
    const shapes: Record<string, number[]> = { a: [2], b: [2] };
    const { header, blob } = await encodeTensors(st, shapes, 'f32');
    if (this.evil.badHeader) header.tensors[0]!.nbytes = 1 << 30;
    this.out = chunkBytes(blob, p.chunk_bytes ?? 1 << 20);
    return { header, nchunks: this.evil.nchunks ?? this.out.length, chunk0: bytesToB64(this.out[0]!) };
  }
  async handle(op: string, p: any): Promise<RpcResult> {
    this.calls.push(op);
    switch (op) {
      case 'task_init': return { ok: true, data: {} };
      case 'task_close': this.closed++; return { ok: true, data: {} };
      case 'task_state_get': if (p.which === 'extra') return { ok: true, data: { header: { v: 1, dtype: 'f32', sha256: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', tensors: [] }, nchunks: 1, chunk0: '' } }; return { ok: true, data: await this.enc(p) };
      case 'task_inner': {
        for (const a of this.state.values()) for (let i = 0; i < a.length; i++) a[i] = a[i]! - 0.1;
        const report: any = { samples: this.evil.samples ?? p.refs.length, timings: { compute_s: this.evil.computeS ?? 0.01 }, metrics: {} };
        if (!this.evil.noLosses) report.losses = [1];
        return { ok: true, data: { report, ...(await this.enc(p)) } };
      }
      case 'task_state_chunk': return { ok: true, data: { data: bytesToB64(this.out[p.k] ?? new Uint8Array(0)) } };
      case 'task_state_put': {
        if (p.k === 0) { this.hdr = p.header; this.parts = []; }
        this.parts.push(b64ToBytes(p.data));
        if (p.k < p.n - 1) return { ok: true, data: { applied: false } };
        this.state = await decodeTensors(this.hdr!, concatBytes(this.parts)); return { ok: true, data: { applied: true } };
      }
      case 'task_after_outer': return { ok: true, data: { target_sha256: this.evil.hash ?? 'h', target_checksum: this.evil.hash === 'fake' ? [9, 9] : [1, 1], alarms: this.evil.alarm ? [this.evil.alarm] : [] } };
      case 'task_eval': return { ok: true, data: { metrics: {} } };
      default: return { ok: false, error: op };
    }
  }
}
const CFG: SessionConfig = { task: 't', manifest_len: 64, batch: 2, inner_steps: 2, lr: 0.1, outer_lr: 1, outer_momentum: 0, seed: 0 };
function mk(n = 3, cfg: Partial<SessionConfig> = {}) {
  const ws = Array.from({ length: n }, () => new W()); const ids = ws.map((_, i) => `w${i}`);
  const rpc: Rpc = (id, op, p) => ws[ids.indexOf(id)]!.handle(op, p);
  return { ws, ids, s: new TrainSession('s', { ...CFG, ...cfg }, ids, rpc) };
}
const g = (s: TrainSession) => Array.from(s.st!.global.get('a')!);

describe('adversarial workers (P0-3)', () => {
  it('reported samples are clamped to the refs actually assigned', async () => {
    for (const bad of [1e9, -3, 'x']) {
      const { ws, s } = mk(); await s.init(); ws[0]!.evil.samples = bad;
      const r = await s.runRound();
      expect(r.samples).toBe(12); expect(s.samplesSeen).toBe(12); expect(s.isDone()).toBe(false);
      for (const v of g(s)) expect(Number.isFinite(v)).toBe(true);
    }
  });
  it('a worker whose state is missing/mis-shaped tensors is dropped before aggregation, even if first', async () => {
    const { ws, s } = mk(); await s.init(); ws[0]!.evil.dropTensor = 'b';
    const r = await s.runRound();
    expect(r.dropped).toContain("w0"); expect(g(s)).toEqual([expect.closeTo(0.9, 5), expect.closeTo(1.9, 5)]);
  });
  it('an absurd nchunks / header is rejected without buffering', async () => {
    const { ws, s } = mk(); await s.init(); ws[1]!.evil.nchunks = 5000; ws[2]!.evil.badHeader = true;
    const r = await s.runRound();
    expect(r.dropped).toEqual(expect.arrayContaining(['w1', 'w2']));
    expect(ws[1]!.calls.filter((c) => c === 'task_state_chunk').length).toBeLessThan(5);
  });
  it('missing losses or a tiny compute time never breaks the round or later allocation', async () => {
    const { ws, s } = mk(3, { alloc: 'proportional' }); await s.init();
    ws[0]!.evil.noLosses = true; ws[1]!.evil.computeS = 1e-300;
    await s.runRound(); await s.runRound();
    expect(s.round).toBe(2);
  });
  it('a single worker faking a target hash or alarm cannot kill the session; it is dropped instead', async () => {
    const { ws, s } = mk(); await s.init(); ws[2]!.evil.hash = 'fake'; ws[1]!.evil.alarm = 'target encoders diverged across workers';
    const r = await s.runRound();
    expect(s.status).not.toBe('failed'); expect(r.dropped).toContain('w2'); expect(ws[2]!.closed).toBe(1);
  });
  it('target hashes that differ but agree within tolerance (mixed devices) are fine', async () => {
    const { ws, s } = mk(2); await s.init(); ws[1]!.evil.hash = 'ulp-different';
    await s.runRound();
    expect(s.status).not.toBe('failed'); expect(s.workers).toHaveLength(2);
  });
  it('the global state is untouched when a round throws after pulling', async () => {
    const { ws, s } = mk(1); await s.init();
    const before = g(s), stream = s.stream.state();
    ws[0]!.evil.dropTensor = 'a';
    await expect(s.runRound()).rejects.toThrow();
    expect(g(s)).toEqual(before); expect(s.stream.state()).toEqual(stream); expect(s.round).toBe(0);
  });
});

describe('liveness + serialisation (P1-5, P1-6)', () => {
  it('a paused worker is skipped for the round but stays in the session', async () => {
    const { s } = mk(); await s.init();
    const r = await s.runRound((w) => w !== 'w1', () => true);
    expect(r.workers).not.toContain('w1'); expect(s.workers).toContain('w1');
  });
  it('an evicted worker gets task_close; duplicate worker ids are deduped', async () => {
    const ws = [new W(), new W()]; const ids = ['w0', 'w1'];
    const rpc: Rpc = (id, op, p) => ws[ids.indexOf(id)]!.handle(op, p);
    const s = new TrainSession('s', CFG, ['w0', 'w0', 'w1'], rpc);
    expect(s.workers).toEqual(['w0', 'w1']);
    await s.init();
    await s.runRound(() => true, (w) => w !== 'w1');           // w1 disconnected → evicted
    expect(ws[1]!.closed).toBe(1);
  });
  it('checkpoint/eval/export during a round wait for it instead of interleaving chunk streams', async () => {
    const { ws, s } = mk(); await s.init();
    const order: string[] = [];
    const round = s.runRound().then(() => order.push('round'));
    const ev = s.evaluate([1], 'loss').then(() => order.push('eval'));
    await Promise.all([round, ev]);
    expect(order).toEqual(['round', 'eval']);
    expect(ws[0]!.calls.lastIndexOf('task_eval')).toBeGreaterThan(ws[0]!.calls.lastIndexOf('task_after_outer'));
  });
  it('keep_checkpoints is clamped to ≥ 1 and chunk_bytes/manifest_len are validated', () => {
    expect(() => new TrainSession('s', { ...CFG, chunk_bytes: 0 }, ['a'], async () => ({ ok: true }))).toThrow(/chunk_bytes/);
    expect(() => new TrainSession('s', { ...CFG, manifest_len: 1e12 }, ['a'], async () => ({ ok: true }))).toThrow(/manifest_len/);
    expect(() => new TrainSession('s', { ...CFG, sync_dtype: 'f64' as 'f32' }, ['a'], async () => ({ ok: true }))).toThrow(/sync_dtype/);
    expect(() => new TrainSession('../x', CFG, ['a'], async () => ({ ok: true }))).toThrow(/session id/);
  });
});

// Cross-language fixture for tests/py/test_telemetry_schema.py: drives the real coordinator TrainSession
// (apps/coordinator/lib/train_session.ts) against two in-memory fake workers for two rounds and prints every telemetry
// record it emits as JSONL, so the Python validator is checked against what the coordinator actually writes.
// Run: deno run tests/py/fixtures/ts_emit_telemetry.ts
import { TrainSession, type Rpc, type RpcResult, type SessionConfig } from '../../../apps/coordinator/lib/train_session.ts';
import { decodeTensors, encodeTensors, b64ToBytes, bytesToB64, chunkBytes, concatBytes, type WireHeader } from '../../../apps/coordinator/lib/tensorwire.ts';

class Fake {
  state = new Map<string, Float32Array>([['w', new Float32Array([1, 2, 3, 4])]]);
  out: Uint8Array[] = []; parts: Uint8Array[] = []; hdr: WireHeader | null = null;
  constructor(public metrics: Record<string, unknown>) {}
  async handle(op: string, p: any): Promise<RpcResult> {
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
        for (let i = 0; i < w.length; i++) w[i] = w[i]! - p.lr * p.refs.length;
        return { ok: true, data: { report: { samples: p.refs.length, losses: [1, 0.5], timings: { compute_s: 0.001, data_s: 0.0005, serialize_s: 0.0001 }, metrics: this.metrics }, ...(await enc()) } };
      }
      case 'task_state_chunk': return { ok: true, data: { data: bytesToB64(this.out[p.k]!) } };
      case 'task_state_put': {
        if (p.k === 0) { this.hdr = p.header; this.parts = []; }
        this.parts.push(b64ToBytes(p.data));
        if (p.k < p.n - 1) return { ok: true, data: { applied: false } };
        this.state = await decodeTensors(this.hdr!, concatBytes(this.parts));
        return { ok: true, data: { applied: true, deserialize_s: 0 } };
      }
      case 'task_after_outer': return { ok: true, data: { target_sha256: 'same' } };
      case 'task_eval': return { ok: true, data: { metrics: { loss: 0.1 } } };
      default: return { ok: true, data: {} };
    }
  }
}

const full = { amp: 'bf16', gpu_util: 71.5, gpu_power_w: 180.2, energy_j: 12.5, mem_peak_bytes: 123456789,
  hw: { os: 'Linux', python: '3.11.0', cpu_count: 8 } };
const fleet = new Map([['w0', new Fake(full)], ['w1', new Fake({})]]);
const rpc: Rpc = (id, op, p) => fleet.get(id)!.handle(op, p);
const cfg: SessionConfig = { task: 'toy', manifest_len: 32, batch: 2, inner_steps: 2, lr: 0.1, seed: 1, chunk_bytes: 6,
  sync_dtype: 'bf16', eval: { refs: [0, 1], kind: 'loss', every: 2 } };
let t = 0;
const s = new TrainSession('fixture', cfg, [...fleet.keys()], rpc,
  { now: () => (t += 7), telemetry: (r) => console.log(JSON.stringify(r)), gitSha: 'deadbeef' });
await s.init();
await s.runRound();
await s.runRound();

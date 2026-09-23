// Generic training session driven by the coordinator (ADR-0104/0105/0106/0107/0111).
// Pure logic over an injected RPC, so it is unit-tested with fake workers (train_session.test.ts) and wired to the
// real sealed relay in server.ts. The coordinator is the DiLoCo parameter server: it assigns deterministic sample
// shards, pulls each worker's post-inner-step state, takes a samples-weighted average, applies outer Nesterov,
// broadcasts, and writes one telemetry record per (round, worker) plus one per round.

import { OuterState, weightedAverage, outerStep, dropNonFinite, type Tensors } from './diloco.ts';
import { SampleStream, allocate, split, type StreamState } from './sharding.ts';
import { lrAt, progress } from './schedule.ts';
import { decodeTensors, encodeTensors, b64ToBytes, bytesToB64, chunkBytes, concatBytes, type WireHeader, type WireDtype } from './tensorwire.ts';

export interface RpcResult { ok: boolean; data?: Record<string, unknown>; error?: string }
export type Rpc = (workerId: string, op: string, payload: Record<string, unknown>) => Promise<RpcResult>;

export interface SessionConfig {
  task: string;
  cfg?: Record<string, unknown>;
  amp?: 'auto' | 'bf16' | 'fp16' | 'fp32';
  seed?: number;
  deterministic?: boolean;
  manifest_len: number;              // number of samples addressable by index on every worker
  batch: number;                     // samples per inner step
  inner_steps: number;               // H
  lr: number;
  outer_lr?: number;                 // η (default 0.7)
  outer_momentum?: number;           // μ (default 0.9)
  alloc?: 'fixed' | 'proportional';
  sync_dtype?: WireDtype;            // worker → coordinator
  broadcast_dtype?: 'f32' | 'bf16' | 'fp16';
  chunk_bytes?: number;
  target_samples?: number;           // stop exactly here
  max_rounds?: number;
  checkpoint_every?: number;         // rounds; 0 = off
  keep_checkpoints?: number;
  eval?: { refs: unknown[]; kind: string; every: number };
  lr_schedule?: { kind: 'constant' | 'cosine'; warmup_frac?: number; min_lr?: number };
  /** alarm substrings that fail the session (default: EMA divergence). The study adds 'collapse'. */
  stop_on_alarm?: string[];
}

export interface CheckpointStore {
  write(name: string, data: Uint8Array | string): Promise<void>;
  read(name: string): Promise<Uint8Array | null>;
  list(prefix: string): Promise<string[]>;
  remove(name: string): Promise<void>;
}

export interface SessionDeps {
  now?: () => number;                            // ms
  telemetry?: (rec: Record<string, unknown>) => void;
  store?: CheckpointStore;
  log?: (level: string, msg: string) => void;
  gitSha?: string;
}

export interface RoundSummary {
  round: number; workers: string[]; dropped: string[]; dropped_nonfinite: string[]; samples: number; samples_seen: number;
  avg_last_loss: number; lr: number; wall_s: number; reduce_s: number; hook_s: number; eval_s: number; bytes_up: number; bytes_down: number; alarms: string[];
  eval?: Record<string, unknown>; monitors?: Record<string, unknown>; done: boolean;
}

const SCHEMA = 'moregpu.telemetry/1';
/** Session ids name files under MOREGPU_TRAIN_DIR / MOREGPU_TELEMETRY_DIR: no dots, no slashes. */
export const SESSION_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const WIRE_DTYPES = ['f32', 'bf16', 'fp16', 'int8delta'];
const MAX_STATE_BYTES = 16 * 2 ** 30;   // refuse any single worker payload above 16 GiB
const bytesPer = (dt: string) => (dt === 'f32' ? 4 : 2);
const numel = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

/** Validate a worker-supplied payload header BEFORE fetching or allocating anything (review P0-3). */
export function validateHeader(h: WireHeader, expect: Record<string, number[]>, nchunks: number, chunk: number): number {
  if (!h || !Array.isArray(h.tensors) || !WIRE_DTYPES.includes(h.dtype) || !/^[0-9a-f]{64}$/.test(String(h.sha256))) throw new Error('malformed state header');
  const names = Object.keys(expect);
  if (names.length) {
    if (h.tensors.length !== names.length) throw new Error(`state has ${h.tensors.length} tensors, expected ${names.length}`);
  }
  let off = 0;
  for (const e of h.tensors) {
    if (!Array.isArray(e.shape) || !e.shape.every((d) => Number.isInteger(d) && d >= 0)) throw new Error(`${e.name}: bad shape`);
    if (names.length) {
      const want = expect[e.name];
      if (!want || want.length !== e.shape.length || want.some((d, i) => d !== e.shape[i])) throw new Error(`${e.name}: unexpected tensor or shape`);
    }
    const n = numel(e.shape);
    let need: number;
    if (h.dtype === 'int8delta') {
      if (!Number.isInteger(e.block) || e.block! <= 0 || e.nblocks !== Math.max(1, Math.ceil(n / e.block!))) throw new Error(`${e.name}: bad int8 blocking`);
      need = 4 * e.nblocks! + n;
    } else need = bytesPer(h.dtype) * n;
    if (e.offset !== off || e.nbytes !== need) throw new Error(`${e.name}: bad offset/size`);
    off += need;
    if (off > MAX_STATE_BYTES) throw new Error('state payload too large');
  }
  const want = Math.max(1, Math.ceil(off / chunk));
  if (!Number.isInteger(nchunks) || nchunks !== want) throw new Error(`nchunks ${nchunks} does not match the payload (${want})`);
  return off;
}
const cleanSamples = (v: unknown, max: number) => { const n = Number(v); return Number.isInteger(n) && n >= 0 && n <= max ? n : max; };
const cleanLosses = (v: unknown): number[] => (Array.isArray(v) ? v.map(Number).filter((x) => Number.isFinite(x)) : []);


export async function configHash(cfg: SessionConfig): Promise<string> {
  const canon = JSON.stringify(cfg, Object.keys(flatten(cfg)).sort());
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(canon)));
  return Array.from(d, (x) => x.toString(16).padStart(2, '0')).join('');
}
function flatten(o: unknown, out: Record<string, true> = {}): Record<string, true> {
  if (o && typeof o === 'object') for (const [k, v] of Object.entries(o as Record<string, unknown>)) { out[k] = true; flatten(v, out); }
  return out;
}

export class TrainSession {
  st: OuterState | null = null;
  shapes: Record<string, number[]> = {};
  stream: SampleStream;
  samplesSeen = 0;
  perWorkerSeen = new Map<string, number>();
  speeds = new Map<string, number>();
  lastBroadcast: Tensors | null = null;
  history: RoundSummary[] = [];
  status: 'new' | 'ready' | 'running' | 'done' | 'failed' | 'closed' = 'new';
  busy = false;
  lastError = '';
  hash = '';
  private now: () => number;

  private lock: Promise<unknown> = Promise.resolve();

  constructor(public id: string, public cfg: SessionConfig, public workers: string[], private rpc: Rpc, private deps: SessionDeps = {}) {
    if (!SESSION_ID_RE.test(id)) throw new Error('session id must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}');
    const intIn = (v: unknown, lo: number, hi: number) => Number.isInteger(v) && (v as number) >= lo && (v as number) <= hi;
    if (!intIn(cfg.manifest_len, 1, 1e8)) throw new Error('manifest_len must be an integer in [1, 1e8]');
    if (!intIn(cfg.batch, 1, 1e6) || !intIn(cfg.inner_steps, 1, 1e5)) throw new Error('batch and inner_steps must be positive integers (batch ≤ 1e6, inner_steps ≤ 1e5)');
    if (!(Number.isFinite(cfg.lr) && cfg.lr > 0)) throw new Error('lr must be a positive number');
    if (cfg.chunk_bytes !== undefined && !intIn(cfg.chunk_bytes, 4, 64 << 20)) throw new Error('chunk_bytes must be an integer in [4 B, 64 MiB]');
    if (cfg.sync_dtype !== undefined && !WIRE_DTYPES.includes(cfg.sync_dtype)) throw new Error('sync_dtype must be f32|bf16|fp16|int8delta');
    if (cfg.broadcast_dtype !== undefined && !['f32', 'bf16', 'fp16'].includes(cfg.broadcast_dtype)) throw new Error('broadcast_dtype must be f32|bf16|fp16');
    if (cfg.alloc !== undefined && !['fixed', 'proportional'].includes(cfg.alloc)) throw new Error('alloc must be fixed|proportional');
    if (cfg.target_samples !== undefined && !intIn(cfg.target_samples, 1, 1e13)) throw new Error('target_samples must be a positive integer');
    if (cfg.max_rounds !== undefined && !intIn(cfg.max_rounds, 1, 1e7)) throw new Error('max_rounds must be a positive integer');
    for (const k of ['outer_lr', 'outer_momentum'] as const) if (cfg[k] !== undefined && !Number.isFinite(cfg[k])) throw new Error(`${k} must be finite`);
    if (cfg.keep_checkpoints !== undefined) cfg.keep_checkpoints = Math.max(1, Math.floor(Number(cfg.keep_checkpoints) || 1));
    this.workers = workers = [...new Set(workers)];
    if (!workers.length) throw new Error('a training session needs at least one worker');
    this.stream = new SampleStream(cfg.manifest_len, cfg.seed ?? 0);
    this.now = deps.now ?? (() => performance.now());
  }

  get round(): number { return this.st?.round ?? 0; }
  /** Serialise every operation that talks to the workers' per-session state (rounds, checkpoints, eval, export, add, close). */
  private exclusive<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.lock.then(fn, fn);
    this.lock = run.catch(() => undefined);
    return run;
  }
  private closeWorkers(ids: string[]) { for (const w of ids) this.rpc(w, 'task_close', { session: this.id }).catch(() => undefined); }
  get chunk(): number { return this.cfg.chunk_bytes ?? (4 << 20); }
  private log(level: string, msg: string) { this.deps.log?.(level, `train ${this.id}: ${msg}`); }

  async call(w: string, op: string, payload: Record<string, unknown>): Promise<Record<string, unknown>> {
    const r = await this.rpc(w, op, { session: this.id, ...payload });
    if (!r.ok) throw new Error(`${w} ${op}: ${r.error ?? 'failed'}`);
    return r.data ?? {};
  }

  /** Pull an encoded state whose first reply carried header/nchunks/chunk0. Returns tensors + bytes received. */
  private async pull(w: string, first: Record<string, unknown>): Promise<{ tensors: Tensors; bytes: number; header: WireHeader }> {
    const header = first.header as WireHeader, n = Number(first.nchunks);
    validateHeader(header, this.shapes, n, this.chunk);            // reject before fetching/allocating anything
    const parts = [b64ToBytes(String(first.chunk0 ?? ''))]; let bytes = String(first.chunk0 ?? '').length;
    for (let k = 1; k < n; k++) {
      const c = await this.call(w, 'task_state_chunk', { k });
      parts.push(b64ToBytes(String(c.data))); bytes += String(c.data).length;
    }
    const ref = header.dtype === 'int8delta' ? this.lastBroadcast ?? undefined : undefined;
    const tensors = await decodeTensors(header, concatBytes(parts), ref);
    return { tensors, bytes, header };
  }

  private async push(w: string, header: WireHeader, blob: Uint8Array): Promise<{ bytes: number; deser_s: number }> {
    const parts = chunkBytes(blob, this.chunk); let bytes = 0, deser = 0;
    for (let k = 0; k < parts.length; k++) {
      const data = bytesToB64(parts[k]!); bytes += data.length;
      const r = await this.call(w, 'task_state_put', { header: k === 0 ? header : null, k, n: parts.length, data });
      if (k === parts.length - 1) { if (!r.applied) throw new Error(`${w}: state not applied`); deser = Number(r.deserialize_s ?? 0); }
    }
    return { bytes, deser_s: deser };
  }

  private async encodeGlobal(): Promise<{ header: WireHeader; blob: Uint8Array }> {
    const enc = await encodeTensors(this.st!.global, this.shapes, this.cfg.broadcast_dtype ?? 'f32');
    // What the workers will hold is the DECODED broadcast; keep exactly that as the int8-delta reference.
    this.lastBroadcast = await decodeTensors(enc.header, enc.blob);
    return enc;
  }

  private async broadcast(targets: string[]): Promise<{ ok: string[]; failed: string[]; bytes: Map<string, number>; secs: Map<string, number>; deser: Map<string, number> }> {
    const { header, blob } = await this.encodeGlobal();
    const bytes = new Map<string, number>(), secs = new Map<string, number>(), deser = new Map<string, number>();
    const res = await Promise.all(targets.map(async (w) => {
      const t0 = this.now();
      try { const r = await this.push(w, header, blob); bytes.set(w, r.bytes); secs.set(w, (this.now() - t0) / 1000); deser.set(w, r.deser_s); return true; }
      catch (e) { this.log('warn', `broadcast to ${w} failed: ${(e as Error).message}`); return false; }
    }));
    return { ok: targets.filter((_, i) => res[i]), failed: targets.filter((_, i) => !res[i]), bytes, secs, deser };
  }

  async init(): Promise<Record<string, unknown>> {
    this.hash = await configHash(this.cfg);
    const init = { task: this.cfg.task, cfg: this.cfg.cfg ?? {}, amp: this.cfg.amp ?? 'auto', seed: this.cfg.seed ?? 0, deterministic: !!this.cfg.deterministic };
    const r = await Promise.all(this.workers.map((w) => this.rpc(w, 'task_init', { session: this.id, ...init })));
    const bad = this.workers.filter((_, i) => !r[i]!.ok);
    if (bad.length === this.workers.length) { this.status = 'failed'; throw new Error(`task_init failed on every worker: ${r[0]!.error}`); }
    if (bad.length) { this.log('warn', `task_init failed on ${bad.join(', ')} — excluded`); this.workers = this.workers.filter((w) => !bad.includes(w)); this.closeWorkers(bad); }
    const first = await this.call(this.workers[0]!, 'task_state_get', { dtype: 'f32', chunk_bytes: this.chunk });
    const { tensors, header } = await this.pull(this.workers[0]!, first);
    for (const e of header.tensors) this.shapes[e.name] = e.shape;
    this.st = OuterState.init(tensors);
    const b = await this.broadcast(this.workers);
    if (!b.ok.length) { this.status = 'failed'; throw new Error('initial broadcast failed on every worker'); }
    this.workers = b.ok;
    this.status = 'ready';
    const describe = (r[this.workers.indexOf(this.workers[0]!)]?.data?.describe) ?? null;
    return { ok: true, session: this.id, workers: this.workers, params: [...tensors.values()].reduce((s, a) => s + a.length, 0), describe, config_hash: this.hash };
  }

  /** Workers that joined after init: initialise the task and hand them the current global. */
  addWorker(w: string): Promise<boolean> { return this.exclusive(() => this.addWorkerUnlocked(w)); }

  private async addWorkerUnlocked(w: string): Promise<boolean> {
    const init = { task: this.cfg.task, cfg: this.cfg.cfg ?? {}, amp: this.cfg.amp ?? 'auto', seed: this.cfg.seed ?? 0, deterministic: !!this.cfg.deterministic };
    const r = await this.rpc(w, 'task_init', { session: this.id, replace: true, ...init });
    if (!r.ok) return false;
    const { header, blob } = await this.encodeGlobal();
    try { await this.push(w, header, blob); } catch { return false; }
    if (!this.workers.includes(w)) this.workers.push(w);
    return true;
  }

  isDone(): boolean {
    if (this.cfg.target_samples !== undefined && this.samplesSeen >= this.cfg.target_samples) return true;
    if (this.cfg.max_rounds !== undefined && this.round >= this.cfg.max_rounds) return true;
    return false;
  }

  /** `live(w)`: may take part this round (e.g. not paused); `connected(w)`: still attached (else evicted). */
  async runRound(live: (w: string) => boolean = () => true, connected: (w: string) => boolean = live): Promise<RoundSummary> {
    if (!this.st) throw new Error('session not initialised');
    if (this.busy) throw new Error('a round is already in progress — rounds are serialized');
    if (this.isDone()) throw new Error('session already reached its stopping rule');
    this.busy = true; this.status = 'running';
    try { return await this.exclusive(() => this.roundInner(live, connected)); }
    catch (e) { this.lastError = (e as Error).message; throw e; }
    finally { this.busy = false; if (this.status === 'running') this.status = this.isDone() ? 'done' : 'ready'; }
  }

  private async roundInner(live: (w: string) => boolean, connected: (w: string) => boolean): Promise<RoundSummary> {
    const tRound = this.now();
    const gone = this.workers.filter((w) => !connected(w));
    const ws = this.workers.filter((w) => connected(w) && live(w));     // paused workers sit this round out, not evicted
    if (!ws.length) { this.status = 'failed'; throw new Error('all workers of this session are gone'); }
    const per = this.cfg.batch * this.cfg.inner_steps;
    const remaining = this.cfg.target_samples !== undefined ? this.cfg.target_samples - this.samplesSeen : undefined;
    const sizes = allocate(per, ws.length, ws.map((w) => this.speeds.get(w) ?? 1), this.cfg.alloc ?? 'fixed', remaining);
    const streamBefore = this.stream.state();
    const roundTotal = sizes.reduce((a, b) => a + b, 0);
    const pMid = progress(this.cfg, this.samplesSeen, roundTotal, this.round);
    const lr = lrAt(this.cfg, pMid);
    // global optimizer steps this round (matched global batch = batch × workers); identical info for every worker's hook
    const hSteps = roundTotal / (this.cfg.batch * ws.length);
    const shards = split(this.stream.take(roundTotal), sizes);
    type R = { w: string; tensors: Tensors; samples: number; report: Record<string, any>; t_inner: number; t_pull: number; bytes_up: number; wire_bytes_up: number; wire_err?: unknown };
    const results = await Promise.all(ws.map(async (w, i): Promise<R | { w: string; error: string }> => {
      const refs = shards[i]!;
      if (!refs.length) return { w, error: 'no samples this round' };
      const steps = Math.max(1, Math.min(refs.length, Math.round(refs.length / this.cfg.batch)));
      const t0 = this.now();
      try {
        const first = await this.call(w, 'task_inner', { refs, steps, lr, sync_dtype: this.cfg.sync_dtype ?? 'f32', chunk_bytes: this.chunk });
        const t1 = this.now();
        const { tensors, bytes, header } = await this.pull(w, first);
        const t2 = this.now();
        const report = first.report as Record<string, any>;
        report.losses = cleanLosses(report.losses);
        return { w, tensors, samples: cleanSamples(report.samples, refs.length), report, t_inner: (t1 - t0) / 1000, t_pull: (t2 - t1) / 1000, bytes_up: Math.floor(bytes * 3 / 4), wire_bytes_up: bytes, wire_err: header.error };
      } catch (e) { return { w, error: (e as Error).message }; }
    }));
    const okR = results.filter((r): r is R => 'tensors' in r);
    const failed = results.filter((r): r is { w: string; error: string } => 'error' in r && r.error !== 'no samples this round');
    const { kept, dropped: nonFinite } = dropNonFinite(okR.map((r) => ({ id: r.w, ...r })));
    if (!kept.length) {
      // nothing usable: roll the sample stream back so these samples are not counted as seen
      this.stream = SampleStream.fromState(streamBefore);
      throw new Error(`round produced no usable state${nonFinite.length ? ` (non-finite: ${nonFinite.join(', ')})` : failed.length ? `: ${failed[0]!.error}` : ''}`);
    }
    const tReduce0 = this.now();
    // atomic: compute on copies, swap in only on success (a throw leaves the global, momentum and stream untouched)
    let next: OuterState;
    try {
      const avg = weightedAverage(kept.map((r) => ({ tensors: r.tensors, weight: r.samples })));
      const lossy = (this.cfg.broadcast_dtype ?? 'f32') !== 'f32';
      const cp = (m: Tensors) => new Map([...m].map(([k, v]) => [k, Float32Array.from(v)]));
      next = new OuterState(cp(this.st!.global), cp(this.st!.momentum), this.st!.round);
      outerStep(next, avg, this.cfg.outer_lr ?? 0.7, this.cfg.outer_momentum ?? 0.9, lossy ? this.lastBroadcast ?? undefined : undefined);
    } catch (e) { this.stream = SampleStream.fromState(streamBefore); throw e; }
    this.st = next;
    const reduce_s = (this.now() - tReduce0) / 1000;
    let roundSamples = 0;
    for (const r of kept) {
      roundSamples += r.samples; this.perWorkerSeen.set(r.w, (this.perWorkerSeen.get(r.w) ?? 0) + r.samples);
      const c = Number(r.report.timings?.compute_s ?? 0);
      if (Number.isFinite(c) && c > 1e-6) this.speeds.set(r.w, Math.min(1e7, Math.max(1e-3, r.samples / c)));
    }
    this.samplesSeen += roundSamples;
    // broadcast to every live worker (including ones dropped for non-finite state: they must resync)
    const b = await this.broadcast(ws);
    const failedIds = new Set(failed.map((f) => f.w));
    const alarms: string[] = [];
    // after_outer hook (e.g. JEPA EMA target update) on all synced workers; compare target hashes
    const tHook = this.now();
    const after = await Promise.all(b.ok.map((w) => this.rpc(w, 'task_after_outer', { session: this.id, round: this.round, progress: pMid, h: hSteps })));
    const hook_s = (this.now() - tHook) / 1000;
    // EMA target consensus: identical hashes, or checksums equal within tolerance (mixed CPU/CUDA/MPS ulp noise);
    // otherwise a strict majority wins and the minority is dropped (non-fatal); no majority = divergence (fatal).
    const minority: string[] = [];
    const hs = b.ok.map((w, i) => ({ w, h: after[i]?.data?.target_sha256 as string | undefined, c: after[i]?.data?.target_checksum as number[] | undefined }));
    const withH = hs.filter((x) => x.h !== undefined);
    if (new Set(withH.map((x) => x.h)).size > 1) {
      const close = (a?: number[], c?: number[]) => !!a && !!c && a.length === c.length && a.every((v, i) => Math.abs(v - c[i]!) <= 1e-4 * Math.max(1, Math.abs(v)));
      const groups: Array<typeof withH> = [];
      for (const x of withH) { const g = groups.find((gr) => gr[0]!.h === x.h || close(gr[0]!.c, x.c)); if (g) g.push(x); else groups.push([x]); }
      groups.sort((a, b2) => b2.length - a.length);
      if (groups.length > 1) {
        if (groups[0]!.length * 2 > withH.length) {
          for (const gr of groups.slice(1)) for (const x of gr) minority.push(x.w);
          alarms.push(`EMA target mismatch: dropped ${minority.join(', ')} (majority kept)`);
        } else alarms.push('target encoders diverged across workers (EMA hash mismatch, no majority)');
      }
    }
    // worker-reported alarms (e.g. collapse) count only when a strict majority of workers report them; 'diverged' is
    // reserved for the coordinator's own check above
    const counts = new Map<string, number>();
    for (const a of after) for (const al of new Set(((a.data?.alarms as string[] | undefined) ?? []).map(String))) if (!al.includes('diverged')) counts.set(al, (counts.get(al) ?? 0) + 1);
    for (const [al, c] of counts) if (c * 2 > after.length && !alarms.includes(al)) alarms.push(al);
    const good = b.ok.filter((w) => !minority.includes(w));
    const monitors = (after[b.ok.indexOf(good[0]!)]?.data?.monitors ?? undefined) as Record<string, unknown> | undefined;
    const evicted = [...new Set([...b.failed, ...minority, ...gone])];
    this.workers = this.workers.filter((w) => good.includes(w) || (!ws.includes(w) && !gone.includes(w)));
    this.closeWorkers(evicted);
    const dropped = [...new Set([...b.failed, ...failedIds, ...gone, ...minority, ...nonFinite])];
    if (!this.workers.length) { this.status = 'failed'; throw new Error('all workers fell out of sync — session failed'); }
    let evalOut: Record<string, unknown> | undefined;
    const tEval = this.now();
    if (this.cfg.eval && this.cfg.eval.every > 0 && this.round % this.cfg.eval.every === 0) {
      const e = await this.rpc(this.workers[0]!, 'task_eval', { session: this.id, refs: this.cfg.eval.refs, kind: this.cfg.eval.kind });
      evalOut = e.ok ? (e.data?.metrics as Record<string, unknown>) : { error: e.error };
    }
    const eval_s = (this.now() - tEval) / 1000;
    const wall = (this.now() - tRound) / 1000;
    const losses = kept.map((r) => (r.report.losses as number[]).at(-1)).filter((x): x is number => Number.isFinite(x));
    const summary: RoundSummary = {
      round: this.round, workers: kept.map((r) => r.w), dropped, dropped_nonfinite: nonFinite, samples: roundSamples, samples_seen: this.samplesSeen,
      avg_last_loss: losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : (null as unknown as number), lr, wall_s: wall, reduce_s, hook_s, eval_s,
      bytes_up: okR.reduce((s, r) => s + r.bytes_up, 0), bytes_down: Math.floor([...b.bytes.values()].reduce((s, x) => s + x, 0) * 3 / 4),
      alarms, eval: evalOut, monitors, done: this.isDone(),
    };
    this.history.push(summary);
    if (this.history.length > 2000) this.history.splice(0, this.history.length - 2000);   // bounded; telemetry JSONL is the full record
    this.emitTelemetry(summary, okR, b.secs, b.bytes, wall, b.deser);
    const fatal = alarms.filter((a) => (this.cfg.stop_on_alarm ?? ['diverged']).some((k) => a.includes(k)));
    if (fatal.length) {
      this.status = 'failed'; this.lastError = `stopped on alarm: ${fatal.join('; ')}`;
      this.log('warn', this.lastError);
      throw new Error(this.lastError);
    }
    if ((this.cfg.checkpoint_every ?? 0) > 0 && this.round % this.cfg.checkpoint_every! === 0) await this.checkpointUnlocked();
    return summary;
  }

  private emitTelemetry(s: RoundSummary, rs: Array<{ w: string; samples: number; report: Record<string, any>; t_inner: number; t_pull: number; bytes_up: number; wire_bytes_up: number; wire_err?: unknown }>, bsecs: Map<string, number>, bbytes: Map<string, number>, wall: number, bdeser: Map<string, number> = new Map()) {
    const tel = this.deps.telemetry; if (!tel) return;
    const ts = new Date().toISOString();
    for (const r of rs) {
      const t = r.report.timings ?? {};
      // serialize_s = worker-side encode of its state + worker-side decode/apply of the broadcast
      const compute = Number(t.compute_s ?? 0), data = Number(t.data_s ?? 0), ser = Number(t.serialize_s ?? 0) + (bdeser.get(r.w) ?? 0);
      const inflight = r.t_inner + r.t_pull + (bsecs.get(r.w) ?? 0);
      const network = Math.max(0, inflight - compute - data - ser);
      const wait = Math.max(0, wall - compute - data - ser - network);
      const m = r.report.metrics ?? {};
      tel({ schema: SCHEMA, kind: 'worker_round', ts, session: this.id, task: this.cfg.task, round: s.round, worker: r.w,
        wall_s: wall, compute_s: compute, data_s: data, serialize_s: ser, network_s: network, wait_s: wait,
        bytes_up: r.bytes_up, bytes_down: Math.floor((bbytes.get(r.w) ?? 0) * 3 / 4), wire_bytes_up: r.wire_bytes_up, wire_bytes_down: bbytes.get(r.w) ?? 0, samples: r.samples, samples_seen: this.perWorkerSeen.get(r.w) ?? 0,
        samples_per_s: compute > 0 ? r.samples / compute : null, loss_last: (r.report.losses as number[]).at(-1) ?? null,
        amp: m.amp ?? null, gpu_util: m.gpu_util ?? null, gpu_power_w: m.gpu_power_w ?? null, energy_j: m.energy_j ?? null,
        mem_peak_bytes: m.mem_peak_bytes ?? null, hw: m.hw ?? null, wire_error: r.wire_err ?? null,
        git_sha: this.deps.gitSha ?? null, config_hash: this.hash });
    }
    tel({ schema: SCHEMA, kind: 'round', ts, session: this.id, task: this.cfg.task, round: s.round, wall_s: wall, reduce_s: s.reduce_s, hook_s: s.hook_s, eval_s: s.eval_s,
      workers: s.workers, dropped: s.dropped, samples: s.samples, samples_seen: s.samples_seen, avg_last_loss: s.avg_last_loss, lr: s.lr,
      bytes_up: s.bytes_up, bytes_down: s.bytes_down, alarms: s.alarms, eval: s.eval ?? null, monitors: s.monitors ?? null,
      git_sha: this.deps.gitSha ?? null, config_hash: this.hash });
  }

  // ---------------------------------------------------------------- checkpoint / resume (ADR-0106)
  checkpoint(): Promise<string | null> { return this.exclusive(() => this.checkpointUnlocked()); }

  private async checkpointUnlocked(): Promise<string | null> {
    const store = this.deps.store; if (!store || !this.st) return null;
    const name = `${this.id}/round-${String(this.round).padStart(6, '0')}`;
    const g = await encodeTensors(this.st.global, this.shapes, 'f32');
    const m = await encodeTensors(this.st.momentum, this.shapes, 'f32');
    await store.write(`${name}.global.bin`, g.blob);
    await store.write(`${name}.momentum.bin`, m.blob);
    // non-synced, PER-WORKER task state (inner optimizer moments, step counters, JEPA EMA target) — one file per worker
    const extra: Record<string, WireHeader> = {};
    for (const w of this.workers) {
      try {
        const first = await this.call(w, 'task_state_get', { which: 'extra', chunk_bytes: this.chunk });
        const hdr = first.header as WireHeader;
        if (!hdr.tensors.length) continue;
        const parts = [b64ToBytes(String(first.chunk0 ?? ''))];
        for (let k = 1; k < Number(first.nchunks); k++) parts.push(b64ToBytes(String((await this.call(w, 'task_state_chunk', { k })).data)));
        await store.write(`${name}.extra.${encodeURIComponent(w)}.bin`, concatBytes(parts)); extra[w] = hdr;
      } catch (e) { this.log('warn', `checkpoint: no extra task state from ${w} (${(e as Error).message})`); }
    }
    const meta = { v: 1, id: this.id, cfg: this.cfg, workers: this.workers, round: this.round, samples_seen: this.samplesSeen,
      per_worker_seen: Object.fromEntries(this.perWorkerSeen), speeds: Object.fromEntries(this.speeds), stream: this.stream.state(),
      global: g.header, momentum: m.header, extra, config_hash: this.hash };
    await store.write(`${name}.json`, JSON.stringify(meta));   // written last: a checkpoint exists iff its .json exists
    const keep = this.cfg.keep_checkpoints ?? 3;
    const all = (await store.list(`${this.id}/`)).filter((n) => n.endsWith('.json')).sort();
    for (const old of all.slice(0, Math.max(0, all.length - keep))) {
      const base = old.slice(0, -5);
      for (const suf of ['.json', '.global.bin', '.momentum.bin']) await store.remove(base + suf);
      for (const f of await store.list(`${this.id}/`)) if (f.startsWith(`${base}.extra.`)) await store.remove(f);
    }
    return name;
  }

  static async resume(id: string, store: CheckpointStore, rpc: Rpc, deps: SessionDeps, workers?: string[]): Promise<TrainSession> {
    const all = (await store.list(`${id}/`)).filter((n) => n.endsWith('.json')).sort();
    if (!all.length) throw new Error(`no checkpoint for session ${id}`);
    const base = all[all.length - 1]!.slice(0, -5);
    const meta = JSON.parse(new TextDecoder().decode((await store.read(`${base}.json`))!));
    const s = new TrainSession(id, meta.cfg, workers ?? meta.workers, rpc, { ...deps, store });
    const g = await decodeTensors(meta.global, (await store.read(`${base}.global.bin`))!);
    const m = await decodeTensors(meta.momentum, (await store.read(`${base}.momentum.bin`))!);
    s.st = new OuterState(g, m, meta.round);
    for (const e of (meta.global as WireHeader).tensors) s.shapes[e.name] = e.shape;
    s.samplesSeen = meta.samples_seen; s.perWorkerSeen = new Map(Object.entries(meta.per_worker_seen));
    s.speeds = new Map(Object.entries(meta.speeds)); s.stream = SampleStream.fromState(meta.stream as StreamState);
    s.hash = meta.config_hash;
    const init = { task: s.cfg.task, cfg: s.cfg.cfg ?? {}, amp: s.cfg.amp ?? 'auto', seed: s.cfg.seed ?? 0, deterministic: !!s.cfg.deterministic };
    const r = await Promise.all(s.workers.map((w) => rpc(w, 'task_init', { session: id, replace: true, ...init })));
    s.workers = s.workers.filter((_, i) => r[i]!.ok);
    if (!s.workers.length) throw new Error('resume: task_init failed on every worker');
    const b = await s.broadcast(s.workers);
    s.workers = b.ok;
    // restore non-synced per-worker task state. A worker that was in the checkpoint gets its own state back (exact
    // resume); a replacement worker gets the shared part (e.g. the EMA target) from the first saved worker, without
    // another worker's optimizer moments (its inner optimizer starts fresh — logged).
    const saved = (meta.extra ?? {}) as Record<string, WireHeader>;
    const savedIds = Object.keys(saved);
    if (savedIds.length) {
      const ok = await Promise.all(s.workers.map(async (w) => {
        const own = saved[w] !== undefined, src = own ? w : savedIds[0]!;
        let header = saved[src]!, blob = (await store.read(`${base}.extra.${encodeURIComponent(src)}.bin`))!;
        if (!own) {
          const t = await decodeTensors(header, blob);
          for (const k of [...t.keys()]) if (k.startsWith('opt.')) t.delete(k);
          const shapes: Record<string, number[]> = {}; for (const e of header.tensors) shapes[e.name] = e.shape;
          ({ header, blob } = await encodeTensors(t, shapes, 'f32'));
          deps.log?.('warn', `resume ${id}: ${w} was not in the checkpoint — shared task state restored, inner optimizer starts fresh`);
        }
        try {
          const parts = chunkBytes(blob, s.chunk);
          for (let k = 0; k < parts.length; k++) await s.call(w, 'task_state_put', { which: 'extra', header: k === 0 ? header : null, k, n: parts.length, data: bytesToB64(parts[k]!) });
          return true;
        } catch { return false; }
      }));
      s.workers = s.workers.filter((_, i) => ok[i]);
      if (!s.workers.length) throw new Error('resume: restoring task state failed on every worker');
    }
    s.status = s.isDone() ? 'done' : 'ready';
    return s;
  }

  // ---------------------------------------------------------------- misc
  evaluate(refs: unknown[], kind: string, worker?: string): Promise<Record<string, unknown>> {
    return this.exclusive(async () => (await this.call(worker ?? this.workers[0]!, 'task_eval', { refs, kind })).metrics as Record<string, unknown>);
  }
  export(fmt: string, path: string, worker?: string, which?: string): Promise<Record<string, unknown>> {
    return this.exclusive(() => this.call(worker ?? this.workers[0]!, 'task_export', { fmt, path, ...(which ? { which } : {}) }));
  }
  async globalTensors(dtype: 'f32' | 'bf16' | 'fp16' = 'f32') { return encodeTensors(this.st!.global, this.shapes, dtype); }
  close(): Promise<void> {
    return this.exclusive(async () => {
      await Promise.all(this.workers.map((w) => this.rpc(w, 'task_close', { session: this.id })));
      this.status = 'closed';
    });
  }
  describe(): Record<string, unknown> {
    return { id: this.id, task: this.cfg.task, status: this.status, workers: this.workers, round: this.round, samples_seen: this.samplesSeen,
      target_samples: this.cfg.target_samples ?? null, per_worker_seen: Object.fromEntries(this.perWorkerSeen),
      speeds: Object.fromEntries(this.speeds), busy: this.busy, last_error: this.lastError || null, config_hash: this.hash,
      last: this.history.at(-1) ?? null };
  }
}

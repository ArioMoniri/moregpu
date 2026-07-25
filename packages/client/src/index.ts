/**
 * @moregpu/client — a small SDK for driving a MoreGPU pool from an application (AI/ML batch jobs,
 * Monte Carlo, linear-algebra building blocks, …). It wraps the admin HTTP API. Works anywhere `fetch`
 * exists (Deno, Node 18+, browsers); inject a `fetch` for testing.
 *
 *   const pool = new MoreGPUClient({ baseUrl: 'http://admin:8787', adminToken: '…' });
 *   const job = await pool.submit('matmul', 1024);      // one job
 *   const jobs = await pool.submitBatch([{kernel:'relu',size:1_000_000}, {kernel:'matmul',size:512}]);
 *   const gpu = await pool.gpu();                         // pool state (virtual GPU)
 */

export type Kernel = 'matmul' | 'vector_add' | 'vector_mul' | 'saxpy' | 'relu' | 'scale' | 'gelu' | 'softmax' | 'layernorm';

export interface JobSpec { kernel: Kernel; size: number; }

/** Real tensor input for a data-mode job (the pool computes on THESE values and returns the output). */
export interface TensorInput { a: ArrayLike<number>; b?: ArrayLike<number>; scalar?: number; M?: number; N?: number; K?: number; }

export interface RunResult { job: Job; output: Float32Array; }

// base64 <-> Float32Array, working in browsers, Deno and Node 18+ (btoa/atob are global in all three).
function f32ToB64(f: Float32Array): string {
  const u = new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
  let s = ''; const C = 0x8000;
  for (let i = 0; i < u.length; i += C) s += String.fromCharCode(...u.subarray(i, i + C));
  return btoa(s);
}
function b64ToF32(b64: string): Float32Array {
  const bin = atob(b64); const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new Float32Array(u.buffer);
}

export interface Job {
  id: string;
  status: 'queued' | 'running' | 'done' | 'failed';
  kernel: string;
  size: number;
  sealed?: boolean;
  ms?: number;
  gflops?: number;
  verified?: boolean;
  /** true when every shard's result carried a valid Ed25519 signature from its worker. */
  signed?: boolean;
  shards?: { worker: string; backend: string; work: number; ms: number }[];
  error?: string;
  note?: string;
  /** base64 Float32 output (data mode only). Use MoreGPUClient.run() to get it decoded. */
  output?: string;
  outLen?: number;
  dataMode?: boolean;
}

export interface VirtualGpu {
  device: string; slots: number; gpuSlots: number; cpuSlots: number; busy: number;
  avgUserUtil: number; avgPoolDuty: number; queueDepth: number;
  totalUnits: number; totalShards: number; poolTrend: number[]; perKernel: Record<string, number>; sealed: string;
}

export interface DeviceDescriptor {
  name: string; kind: string; backends: string[]; vendors: string[];
  slots: number; gpuSlots: number; cpuSlots: number; busy: number;
  kernels: string[];
  limits: { maxMatmulDim: number; maxElements: number; maxInputElements: number };
  queue: { depth: number; running: number };
  throughput: { totalUnits: number; totalShards: number; trend: number[] };
  seal: string;
  capabilities: Record<string, boolean>;
}

export interface WorkerInfo {
  id: string; backend: 'gpu' | 'cpu'; label: string; os: string;
  userUtil: number; poolDuty: number; busy: boolean;
  shards: number; units: number; share: number; errors: number; avgMs: number; uptimeS: number; trend: number[];
}

export interface ClientOptions {
  baseUrl: string;
  adminToken: string;
  /** Inject a fetch implementation (defaults to global fetch). */
  fetch?: typeof globalThis.fetch;
}

export class MoreGPUClient {
  private readonly base: string;
  private readonly token: string;
  private readonly f: typeof globalThis.fetch;

  constructor(opts: ClientOptions) {
    this.base = opts.baseUrl.replace(/\/+$/, '');
    this.token = opts.adminToken;
    this.f = opts.fetch ?? globalThis.fetch;
  }

  private authHeaders(): Record<string, string> {
    return { 'content-type': 'application/json', authorization: `Bearer ${this.token}` };
  }

  private async get<T>(path: string, auth = true): Promise<T> {
    const res = await this.f(`${this.base}${path}`, auth ? { headers: this.authHeaders() } : {});
    if (!res.ok) throw new Error(`moregpu: GET ${path} → ${res.status}`);
    return (await res.json()) as T;
  }

  /** Liveness + fleet size (no auth required). */
  health(): Promise<{ ok: boolean; fleet: number; queue: number }> {
    return this.get('/health', false);
  }

  /** The pool as one virtual GPU (slots, contribution totals, throughput trend, per-kernel counts). */
  gpu(): Promise<VirtualGpu> {
    return this.get('/gpu');
  }

  /** Device descriptor — the pool as a GPU slot: backends, kernels, limits, queue, capabilities. */
  device(): Promise<DeviceDescriptor> {
    return this.get('/device');
  }

  /** Submit without waiting: returns a job handle immediately (GPU-style async queue). Poll with waitFor(). */
  async submitAsync(kernel: Kernel, size: number): Promise<{ id: string; status: string; poll: string }> {
    const res = await this.f(`${this.base}/submit?async=1`, { method: 'POST', headers: this.authHeaders(), body: JSON.stringify({ kernel, size }) });
    return (await res.json()) as { id: string; status: string; poll: string };
  }

  /** Poll a job until it is done or failed (or the timeout elapses). */
  async waitFor(id: string, opts: { intervalMs?: number; timeoutMs?: number } = {}): Promise<Job> {
    const interval = opts.intervalMs ?? 200, deadline = Date.now() + (opts.timeoutMs ?? 60_000);
    for (;;) {
      const job = await this.job(id);
      if (job.status === 'done' || job.status === 'failed') return job;
      if (Date.now() > deadline) throw new Error(`moregpu: waitFor(${id}) timed out in status ${job.status}`);
      await new Promise((r) => setTimeout(r, interval));
    }
  }

  /** Connected workers with live contribution (share, units, trend, avg latency, util/duty). */
  workers(): Promise<WorkerInfo[]> {
    return this.get('/workers');
  }

  /** Submit one job and wait for the server's synchronous result. */
  async submit(kernel: Kernel, size: number): Promise<Job> {
    const res = await this.f(`${this.base}/submit`, {
      method: 'POST',
      headers: this.authHeaders(),
      body: JSON.stringify({ kernel, size }),
    });
    const job = (await res.json()) as Job;
    if (!res.ok && res.status !== 202) throw new Error(`moregpu: submit ${kernel} → ${res.status}: ${job.error ?? ''}`);
    return job;
  }

  /** Submit many jobs concurrently. NOTE: the pool runs queued jobs sequentially (one job at a time,
   *  sharded across the fleet); concurrency here just keeps the queue full. */
  submitBatch(specs: JobSpec[]): Promise<Job[]> {
    return Promise.all(specs.map((s) => this.submit(s.kernel, s.size)));
  }

  /**
   * DATA MODE — send your own tensors and get the pooled result back (sealed end-to-end).
   * The returned `output` is the computed Float32Array; `job.verified` is set when the pool checked it
   * against the CPU reference. Always confirm `job.status === 'done'` before using the output.
   */
  async run(kernel: Kernel, input: TensorInput): Promise<RunResult> {
    const a = input.a instanceof Float32Array ? input.a : Float32Array.from(input.a);
    const b = input.b === undefined ? undefined : input.b instanceof Float32Array ? input.b : Float32Array.from(input.b);
    const body: Record<string, unknown> = { kernel, a: f32ToB64(a) };
    if (b) body.b = f32ToB64(b);
    if (input.scalar !== undefined) body.scalar = input.scalar;
    for (const k of ['M', 'N', 'K'] as const) if (input[k] !== undefined) body[k] = input[k];
    const res = await this.f(`${this.base}/submit`, { method: 'POST', headers: this.authHeaders(), body: JSON.stringify(body) });
    const job = (await res.json()) as Job & { output?: string };
    if (!res.ok && res.status !== 202) throw new Error(`moregpu: run ${kernel} → ${res.status}: ${job.error ?? ''}`);
    // Don't silently return an empty output for a queued/running/failed job — the caller would feed [] downstream.
    if (job.status !== 'done') throw new Error(`moregpu: run ${kernel} not done (status=${job.status}${job.error ? ', ' + job.error : job.note ? ', ' + job.note : ''})`);
    return { job, output: job.output ? b64ToF32(job.output) : new Float32Array(0) };
  }

  /** Convenience: C = A(M×K) · B(K×N), computed on the pool, returned as a Float32Array. */
  async matmul(A: ArrayLike<number>, B: ArrayLike<number>, M: number, N: number, K: number): Promise<Float32Array> {
    const { output } = await this.run('matmul', { a: A, b: B, M, N, K });
    return output;
  }

  // ---- composition helpers for inference (compose the verified primitives above) ----

  /** Dense layer y = X(M×K)·W(K×N) [+ b(N)]; bias is row-broadcast client-side. */
  async linear(X: ArrayLike<number>, W: ArrayLike<number>, b: ArrayLike<number> | null, M: number, K: number, N: number): Promise<Float32Array> {
    const y = await this.matmul(X, W, M, N, K);
    if (b) { const bb = Float32Array.from(b); for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) { const p = i * N + j; y[p] = (y[p] ?? 0) + (bb[j] ?? 0); } }
    return y;
  }

  /** Chain dense layers with an activation between them. layers = [{ W, b?, out }]. */
  async mlp(X: ArrayLike<number>, layers: { W: ArrayLike<number>; b?: ArrayLike<number> | null; out: number }[], M: number, K: number, act: Kernel = 'relu'): Promise<Float32Array> {
    let cur: ArrayLike<number> = X, k = K;
    for (const L of layers) { const lin = await this.linear(cur, L.W, L.b ?? null, M, k, L.out); cur = (await this.run(act, { a: lin })).output; k = L.out; }
    return Float32Array.from(cur);
  }

  /** Single-head scaled dot-product attention softmax(Q·Kᵀ/√d)·V. Q,K,V row-major seq×d → seq×d. */
  async attention(Q: ArrayLike<number>, K: ArrayLike<number>, V: ArrayLike<number>, seq: number, d: number, scale?: number): Promise<Float32Array> {
    const s = scale ?? 1 / Math.sqrt(d);
    const Ka = Float32Array.from(K), Kt = new Float32Array(seq * d);
    for (let r = 0; r < seq; r++) for (let c = 0; c < d; c++) Kt[c * seq + r] = Ka[r * d + c] ?? 0;
    const scores = await this.matmul(Q, Kt, seq, seq, d);
    const scaled = (await this.run('scale', { a: scores, scalar: s })).output;
    const attn = (await this.run('softmax', { a: scaled, M: seq, N: seq })).output;
    return this.matmul(attn, V, seq, d, seq);
  }

  /** Reductions via the GEMM trick (convenience; run single-worker/single-thread, not for throughput). */
  async dot(a: ArrayLike<number>, b: ArrayLike<number>): Promise<number> { return (await this.matmul(a, b, 1, 1, a.length))[0] ?? NaN; }
  async sum(a: ArrayLike<number>): Promise<number> { return (await this.matmul(a, new Float32Array(a.length).fill(1), 1, 1, a.length))[0] ?? NaN; }
  async mean(a: ArrayLike<number>): Promise<number> { return (await this.sum(a)) / a.length; }
  async norm(a: ArrayLike<number>): Promise<number> { return Math.sqrt(await this.dot(a, a)); }

  /** Fetch one job's status/result by id. */
  job(id: string): Promise<Job> {
    return this.get(`/jobs/${encodeURIComponent(id)}`);
  }

  /** Recent jobs (queue + history). */
  jobs(): Promise<Job[]> {
    return this.get('/jobs');
  }
}

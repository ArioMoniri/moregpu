// Mixed-fleet vision (ADR-0114, M6): one model served by native torch workers AND WebGPU workers (Deno / browser).
//
//   /vision/load {fleet: 'webgpu'|'all'}: the model is loaded natively on a torch worker, lowered there
//   (vision_lower target 'wgsl', include_bytes) into the WGSL executor's op-graph + safetensors, streamed to every
//   worker advertising the `vision` cap (push_begin / push_chunk / push_end, the same staging path as model weights),
//   then vision_load'ed there. Inference then goes to any holder; the two kinds speak different payloads, which this
//   module normalises:
//     torch   vision_infer {id, shape, data}                       → {shape, data}
//     webgpu  vision_infer {id, inputs: {<graph input>: {shape, b64}}} → {outputs: {<name>: {shape, b64}}, backend, ms}
//   Every reply becomes {shape, data (b64 little-endian f32), kind, backend?}.
// Pure functions + an injected RPC, so it is unit-tested without sockets (vision_fleet.test.ts).
import type { Rpc, RpcResult } from './vision_batch.ts';

export type WorkerKind = 'torch' | 'webgpu';
export type Fleet = 'native' | 'webgpu' | 'all';
export interface TensorB64 { shape: number[]; data: string }
/** The lowered graph's io: which graph input receives the tensor, which output is returned. */
export interface VisionIO { input: string; output: string; shape?: number[] }

export function fleetWants(fleet: Fleet | undefined): { torch: boolean; webgpu: boolean } {
  switch (fleet ?? 'native') {
    case 'native': return { torch: true, webgpu: false };
    case 'webgpu': return { torch: false, webgpu: true };
    case 'all': return { torch: true, webgpu: true };
  }
  throw new Error(`fleet must be 'native', 'webgpu' or 'all' (got '${fleet}')`);
}

/** Static input shape for the lowering: explicit, else [1, in_chans, ...img_size] from the loaded model's meta. */
export function exampleShape(meta: Record<string, unknown>, explicit: number[] | undefined): number[] | undefined {
  if (explicit?.length) return explicit.map(Number);
  const enc = meta?.encoder as { in_chans?: number; img_size?: number[] } | undefined;
  if (enc && Number.isFinite(Number(enc.in_chans)) && Array.isArray(enc.img_size) && enc.img_size.length) return [1, Number(enc.in_chans), ...enc.img_size.map(Number)];
  return undefined;
}

export function inferCall(kind: WorkerKind, id: string, x: TensorB64, io: VisionIO | undefined): { op: string; payload: Record<string, unknown> } {
  if (kind === 'torch') return { op: 'vision_infer', payload: { id, shape: x.shape, data: x.data } };
  if (!io?.input) throw new Error('webgpu worker: the lowered graph input name is unknown (load with fleet webgpu|all first)');
  return { op: 'vision_infer', payload: { id, inputs: { [io.input]: { shape: x.shape, b64: x.data } } } };
}

export function normalizeInferReply(kind: WorkerKind, data: Record<string, unknown>, io: VisionIO | undefined): TensorB64 & { backend?: string; ms?: number } {
  if (kind === 'torch') return { shape: (data.shape as number[]) ?? [], data: String(data.data ?? '') };
  const outs = (data.outputs ?? {}) as Record<string, { shape: number[]; b64: string }>;
  const o = (io?.output && outs[io.output]) || Object.values(outs)[0];
  if (!o) throw new Error('webgpu worker returned no output');
  return { shape: o.shape, data: o.b64, backend: data.backend as string | undefined, ms: data.ms as number | undefined };
}

/** An Rpc that routes by worker kind (torch → the train relay, webgpu → the sealed model RPC) and normalises
 *  vision_infer replies to {shape, data, kind}. Other ops pass through unchanged. */
export function makeFleetRpc(kindOf: (worker: string) => WorkerKind | undefined, torch: Rpc, model: Rpc, io: VisionIO | undefined): Rpc {
  return async (w, op, payload) => {
    const kind = kindOf(w) ?? 'torch';
    const r = await (kind === 'webgpu' ? model : torch)(w, op, payload);
    if (!r.ok || op !== 'vision_infer') return r;
    try { return { ok: true, data: { ...normalizeInferReply(kind, r.data ?? {}, io), kind } }; }
    catch (e) { return { ok: false, error: (e as Error).message }; }
  };
}

function b64FromBytes(u: Uint8Array): string {
  let s = '';
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
  return btoa(s);
}
function bytesFromB64(s: string): Uint8Array {
  const bin = atob(s); const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}
export { b64FromBytes, bytesFromB64 };

export function b64Chunks(bytes: Uint8Array, chunk: number): string[] {
  if (!bytes.length) return [''];
  const out: string[] = [];
  for (let o = 0; o < bytes.length; o += chunk) out.push(b64FromBytes(bytes.subarray(o, Math.min(bytes.length, o + chunk))));
  return out;
}

/** Stage files on a WebGPU worker under `id` (push_begin → push_chunk… → push_end), then vision_load it there. */
export async function pushVisionArtifact(rpc: Rpc, worker: string, id: string, files: { name: string; bytes: Uint8Array }[], chunk: number,
  load: Record<string, unknown> = {}): Promise<RpcResult> {
  const begin = await rpc(worker, 'push_begin', { id, model: id });
  if (!begin.ok) return { ok: false, error: `push_begin: ${begin.error}` };
  let total = 0;
  for (const f of files) {
    const parts = b64Chunks(f.bytes, chunk);
    for (let seq = 0; seq < parts.length; seq++) {
      const r = await rpc(worker, 'push_chunk', { id, name: f.name, seq, data: parts[seq], last: seq === parts.length - 1 });
      if (!r.ok) { await rpc(worker, 'vision_unload', { id }).catch(() => undefined); return { ok: false, error: `push_chunk ${f.name}#${seq}: ${r.error}` }; }
    }
    total += f.bytes.length;
  }
  const end = await rpc(worker, 'push_end', { id });
  if (!end.ok) return { ok: false, error: `push_end: ${end.error}` };
  const l = await rpc(worker, 'vision_load', { ...load, id });
  return l.ok ? { ok: true, data: { ...(l.data ?? {}), bytes: total } } : { ok: false, error: `vision_load: ${l.error}` };
}

export function maxAbsDiffB64(a: string, b: string): number {
  const x = bytesFromB64(a), y = bytesFromB64(b);
  if (x.length !== y.length || x.length % 4) return Infinity;
  const fa = new Float32Array(x.buffer, x.byteOffset, x.length / 4), fb = new Float32Array(y.buffer, y.byteOffset, y.length / 4);
  let m = 0;
  for (let i = 0; i < fa.length; i++) { const d = Math.abs(fa[i] - fb[i]); if (!(d <= m)) m = Number.isNaN(d) ? Infinity : d; }
  return m;
}

/** max|Δ| of every served output vs the reference worker's output for the same input: overall and per serving worker. */
export function parityReport(outputs: Array<(TensorB64 & { worker?: string }) | null | undefined>, reference: Array<TensorB64 | null | undefined>, refWorker: string) {
  const per: Record<string, number> = {};
  let max = 0;
  outputs.forEach((o, i) => {
    const r = reference[i];
    const d = !o || !r || o.shape.join(',') !== r.shape.join(',') ? Infinity : maxAbsDiffB64(o.data, r.data);
    const w = o?.worker ?? '?';
    per[w] = Math.max(per[w] ?? 0, d);
    max = Math.max(max, d);
  });
  return { reference: refWorker, max_abs: max, per_worker: per };
}

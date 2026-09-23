import { describe, it, expect } from 'vitest';
import {
  fleetWants, inferCall, normalizeInferReply, makeFleetRpc, pushVisionArtifact, b64Chunks, maxAbsDiffB64, parityReport,
  exampleShape, type WorkerKind,
} from './vision_fleet.ts';
import type { Rpc } from './vision_batch.ts';

const f32b64 = (a: number[]) => Buffer.from(new Float32Array(a).buffer).toString('base64');
const IO = { input: 'x', output: 'output', shape: [1, 3, 4, 4] };

describe('fleet selection', () => {
  it("fleet 'native' | 'webgpu' | 'all' (default native)", () => {
    expect(fleetWants(undefined)).toEqual({ torch: true, webgpu: false });
    expect(fleetWants('native')).toEqual({ torch: true, webgpu: false });
    expect(fleetWants('webgpu')).toEqual({ torch: false, webgpu: true });
    expect(fleetWants('all')).toEqual({ torch: true, webgpu: true });
    expect(() => fleetWants('gpu' as never)).toThrow(/fleet/);
  });
  it('example shape: explicit, else [1, in_chans, ...img_size] from the loaded model meta', () => {
    expect(exampleShape({ encoder: { in_chans: 3, img_size: [32, 32] } }, undefined)).toEqual([1, 3, 32, 32]);
    expect(exampleShape({ encoder: { in_chans: 1, img_size: [16, 16, 16] } }, [2, 1, 16, 16, 16])).toEqual([2, 1, 16, 16, 16]);
    expect(exampleShape({}, undefined)).toBeUndefined();
  });
});

describe('infer request / reply normalisation per worker kind', () => {
  const x = { shape: [1, 3, 4, 4], data: f32b64(new Array(48).fill(1)) };
  it('torch: vision_infer {id, shape, data}; webgpu: vision_infer {id, inputs: {<graph input>: {shape, b64}}}', () => {
    expect(inferCall('torch', 'm', x, IO)).toEqual({ op: 'vision_infer', payload: { id: 'm', shape: x.shape, data: x.data } });
    expect(inferCall('webgpu', 'm', x, IO)).toEqual({ op: 'vision_infer', payload: { id: 'm', inputs: { x: { shape: x.shape, b64: x.data } } } });
    expect(() => inferCall('webgpu', 'm', x, undefined)).toThrow(/graph input/);
  });
  it('replies are normalised to {shape, data}', () => {
    expect(normalizeInferReply('torch', { ok: true, shape: [1, 2], data: 'AA' }, IO)).toMatchObject({ shape: [1, 2], data: 'AA' });
    const g = normalizeInferReply('webgpu', { ok: true, outputs: { output: { shape: [1, 2], b64: 'BB' } }, backend: 'webgpu:llvmpipe', ms: 3 }, IO);
    expect(g).toMatchObject({ shape: [1, 2], data: 'BB', backend: 'webgpu:llvmpipe' });
    expect(normalizeInferReply('webgpu', { outputs: { other: { shape: [2], b64: 'CC' } } }, { ...IO, output: 'missing' })).toMatchObject({ data: 'CC' });
    expect(() => normalizeInferReply('webgpu', { outputs: {} }, IO)).toThrow(/no output/);
  });
  it('makeFleetRpc routes by kind (torch → trainRPC, webgpu → modelRPC) and normalises vision_infer replies', async () => {
    const calls: string[] = [];
    const kinds: Record<string, WorkerKind> = { t: 'torch', g: 'webgpu' };
    const torch: Rpc = async (w, op) => { calls.push(`train:${w}:${op}`); return { ok: true, data: { shape: [1], data: 'T' } }; };
    const model: Rpc = async (w, op) => { calls.push(`model:${w}:${op}`); return { ok: true, data: { outputs: { output: { shape: [1], b64: 'G' } } } }; };
    const rpc = makeFleetRpc((w) => kinds[w], torch, model, IO);
    expect((await rpc('t', 'vision_infer', {})).data).toMatchObject({ shape: [1], data: 'T', kind: 'torch' });
    expect((await rpc('g', 'vision_infer', {})).data).toMatchObject({ shape: [1], data: 'G', kind: 'webgpu' });
    expect((await rpc('g', 'vision_unload', {})).ok).toBe(true);
    expect(calls).toEqual(['train:t:vision_infer', 'model:g:vision_infer', 'model:g:vision_unload']);
    const bad = makeFleetRpc(() => 'webgpu', torch, async () => ({ ok: true, data: { outputs: {} } }), IO);
    expect(await bad('g', 'vision_infer', {})).toMatchObject({ ok: false, error: expect.stringMatching(/no output/) });
  });
});

describe('streaming the lowered artefact to a WebGPU worker', () => {
  it('b64Chunks splits bytes into chunk-sized base64 pieces (an empty file still gets one chunk)', () => {
    const b = new Uint8Array(10).map((_, i) => i);
    const parts = b64Chunks(b, 4);
    expect(parts).toHaveLength(3);
    expect(Buffer.concat(parts.map((p) => Buffer.from(p, 'base64')))).toEqual(Buffer.from(b));
    expect(b64Chunks(new Uint8Array(0), 4)).toEqual(['']);
  });
  it('push_begin → push_chunk (graph.json, model.safetensors) → push_end → vision_load, stopping at the first error', async () => {
    const calls: Array<[string, any]> = [];
    const rpc: Rpc = async (_w, op, p) => { calls.push([op, p]); return { ok: true, data: op === 'vision_load' ? { id: p.id, backend: 'webgpu' } : {} }; };
    const graph = new TextEncoder().encode('{"version":1}');
    const st = new Uint8Array(9).fill(7);
    const r = await pushVisionArtifact(rpc, 'g', 'seg', [{ name: 'graph.json', bytes: graph }, { name: 'model.safetensors', bytes: st }], 4);
    expect(r).toMatchObject({ ok: true, data: { id: 'seg', backend: 'webgpu', bytes: graph.length + st.length } });
    expect(calls.map((c) => c[0])).toEqual(['push_begin', ...new Array(4 + 3).fill('push_chunk'), 'push_end', 'vision_load']); // 13 B graph → 4 chunks, 9 B weights → 3
    const chunks = calls.filter((c) => c[0] === 'push_chunk').map((c) => c[1]);
    expect(chunks.filter((c) => c.name === 'model.safetensors').map((c) => [c.seq, c.last])).toEqual([[0, false], [1, false], [2, true]]);
    expect(chunks.every((c) => c.id === 'seg')).toBe(true);
    const failing: Rpc = async (_w, op) => (op === 'push_chunk' ? { ok: false, error: 'staging cap' } : { ok: true, data: {} });
    const bad = await pushVisionArtifact(failing, 'g', 'seg', [{ name: 'graph.json', bytes: graph }], 4);
    expect(bad).toMatchObject({ ok: false, error: expect.stringMatching(/push_chunk graph\.json.*staging cap/) });
  });
});

describe('parity between workers', () => {
  it('max|Δ| over little-endian f32 base64 tensors', () => {
    expect(maxAbsDiffB64(f32b64([1, 2, 3]), f32b64([1, 2.5, 2]))).toBeCloseTo(1, 6);
    expect(maxAbsDiffB64(f32b64([1]), f32b64([1, 2]))).toBe(Infinity);
    expect(maxAbsDiffB64(f32b64([NaN]), f32b64([0]))).toBe(Infinity);
  });
  it('parityReport: overall + per worker', () => {
    const outs = [{ shape: [2], data: f32b64([1, 2]), worker: 'g' }, { shape: [2], data: f32b64([3, 4]), worker: 't' }];
    const ref = [{ shape: [2], data: f32b64([1, 2.001]) }, { shape: [2], data: f32b64([3, 4]) }];
    const p = parityReport(outs, ref, 't');
    expect(p.reference).toBe('t');
    expect(p.max_abs).toBeCloseTo(0.001, 5);
    expect(p.per_worker.g).toBeCloseTo(0.001, 5); expect(p.per_worker.t).toBe(0);
    expect(parityReport([{ shape: [3], data: f32b64([1, 2, 3]), worker: 'g' }], ref.slice(0, 1), 't').max_abs).toBe(Infinity);
  });
});

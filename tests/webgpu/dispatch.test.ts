// The worker-facing `vision_*` op family (routed from worker.ts's sealed `model` RPC) on the CPU path, and the
// minimal worker.ts wiring (import, routing, capability flag) checked statically. worker.ts itself cannot be
// imported in a test: it connects to a coordinator at module load.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { visionDispatch, VISION_OPS, SUPPORTED_OPS } from '../../apps/worker/vision_wgsl.ts';
import { type TJson, relErr, b64ToF32, f32ToB64 } from './golden_util.ts';

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));
const bytes = (p: string) => new Uint8Array(readFileSync(here(p)));

function staged(name: string): Map<string, Uint8Array> {
  const g = bytes(`../goldens/wgsl/${name}.graph.json`);
  const w = bytes(`../goldens/wgsl/${name}.safetensors`);
  return new Map([['graph.json', g], ['model.safetensors', w]]);
}
const ctx = (files: Map<string, Uint8Array> | null) => ({ device: null, takeStaged: (_id: string) => files });

describe('visionDispatch (CPU fallback path)', () => {
  it('vision_caps reports webgpu=false, the op list and limits', async () => {
    const r = await visionDispatch('vision_caps', {}, ctx(null));
    expect(r.ok).toBe(true);
    expect(r.webgpu).toBe(false);
    expect(r.backend).toBe('cpu-reference');
    expect(r.ops).toEqual([...SUPPORTED_OPS]);
    expect(r.training).toBe(false);
    expect(VISION_OPS).toEqual(['vision_caps', 'vision_load', 'vision_plan', 'vision_infer', 'vision_sliding_window', 'vision_unload']);
  });
  it('load → plan → infer → unload round trip matches the golden', async () => {
    const r1 = await visionDispatch('vision_load', { id: 's2d' }, ctx(staged('seg2d_tiny')));
    expect(r1).toMatchObject({ ok: true, id: 's2d', backend: 'cpu-reference', inputs: [{ name: 'x', shape: [1, 3, 16, 20] }], outputs: ['mask'] });
    const plan = await visionDispatch('vision_plan', { id: 's2d' }, ctx(null));
    expect((plan.plan as { totalBytes: number }).totalBytes).toBeGreaterThan(0);
    const io = JSON.parse(readFileSync(here('../goldens/wgsl/seg2d_tiny.io.json'), 'utf8')) as { inputs: Record<string, TJson>; expected: Record<string, TJson> };
    const r2 = await visionDispatch('vision_infer', { id: 's2d', inputs: { x: io.inputs.x } }, ctx(null));
    const out = (r2.outputs as Record<string, TJson>).mask;
    expect(out.shape).toEqual([1, 2, 16, 20]);
    expect(relErr(b64ToF32(out.b64), b64ToF32(io.expected.mask.b64))).toBeLessThanOrEqual(1e-5);
    expect(typeof r2.ms).toBe('number');
    expect(await visionDispatch('vision_unload', { id: 's2d' }, ctx(null))).toEqual({ ok: true });
    await expect(visionDispatch('vision_infer', { id: 's2d', inputs: { x: io.inputs.x } }, ctx(null))).rejects.toThrow(/no vision model 's2d'/);
  });
  it('vision_load accepts an inline graph object and requires staged weights', async () => {
    const graph = JSON.parse(readFileSync(here('../goldens/wgsl/seg2d_tiny.graph.json'), 'utf8'));
    const w = new Map([['model.safetensors', bytes('../goldens/wgsl/seg2d_tiny.safetensors')]]);
    expect((await visionDispatch('vision_load', { id: 'inl', graph }, ctx(w))).ok).toBe(true);
    await expect(visionDispatch('vision_load', { id: 'nw', graph }, ctx(new Map()))).rejects.toThrow(/model\.safetensors/);
    await visionDispatch('vision_unload', { id: 'inl' }, ctx(null));
  });
  it('vision_sliding_window runs the MONAI-equivalent driver over a loaded model', async () => {
    await visionDispatch('vision_load', { id: 'u' }, ctx(staged('unet3d_tiny')));
    const vol = { shape: [1, 1, 32, 32, 40], data: Float32Array.from({ length: 32 * 32 * 40 }, (_, i) => Math.cos(i * 0.01)) };
    const b64 = f32ToB64(vol.data);
    const r = await visionDispatch('vision_sliding_window', { id: 'u', input: { shape: vol.shape, b64 }, output: 'logits', roi: [32, 32, 32], overlap: 0.5, mode: 'gaussian' }, ctx(null));
    expect((r.output as TJson).shape).toEqual([1, 3, 32, 32, 40]);
    expect(r.windows).toBe(2); // D,H: roi == image → 1 window each; W: interval 16 → starts [0, 8]
    await visionDispatch('vision_unload', { id: 'u' }, ctx(null));
  }, 60_000);
  it('unknown op / bad ids are errors', async () => {
    await expect(visionDispatch('vision_nope', {}, ctx(null))).rejects.toThrow(/unsupported vision op/);
    await expect(visionDispatch('vision_load', { id: '../x' }, ctx(null))).rejects.toThrow(/id/);
  });
});

describe('worker.ts wiring (static)', () => {
  const src = readFileSync(here('../../apps/worker/worker.ts'), 'utf8');
  it('imports the vision module LAZILY (signed single-file installs must still start) and routes vision_* ops to it', () => {
    expect(src).toMatch(/import\('\.\/vision_wgsl\.ts'\)\.catch\(/);
    expect(src).not.toMatch(/^import .*vision_wgsl/m); // no static import: a missing sibling would abort startup
    expect(src).toMatch(/op\.startsWith\('vision_'\)/);
    expect(src).toMatch(/vision\.visionDispatch\(op, p,/);
  });
  it("advertises the 'vision' capability only when a WebGPU device exists and the module loaded", () => {
    expect(src).toMatch(/if \(backend\.device && visionReady\) caps\.push\('vision'\)/);
  });
});

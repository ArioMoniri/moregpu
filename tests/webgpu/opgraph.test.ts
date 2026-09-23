// Whole-model parity through the op-graph executor (CPU reference), op-table agreement with vision_ops.json,
// and the memory planner (lifetimes, buffer reuse, binding-size tiling).
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  VisionModel, runCpu, parseSafetensors, SUPPORTED_OPS, baseOp, type OpGraph,
} from '../../apps/worker/vision_wgsl.ts';
import { type KernelGoldens, type TJson, tensorOf, relErr, shapeEq } from './golden_util.ts';

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));
const OPS = JSON.parse(readFileSync(here('../../apps/worker/vision_ops.json'), 'utf8')) as { ops: Record<string, unknown> };
const G: KernelGoldens = JSON.parse(readFileSync(here('../goldens/wgsl/kernels.json'), 'utf8'));

function loadModel(name: string, opts: Parameters<typeof VisionModel.compile>[2] = {}) {
  const graph = JSON.parse(readFileSync(here(`../goldens/wgsl/${name}.graph.json`), 'utf8')) as OpGraph;
  const weights = parseSafetensors(new Uint8Array(readFileSync(here(`../goldens/wgsl/${graph.weights}`))));
  const io = JSON.parse(readFileSync(here(`../goldens/wgsl/${name}.io.json`), 'utf8')) as { inputs: Record<string, TJson>; expected: Record<string, TJson> };
  return { graph, weights, io, model: VisionModel.compile(graph, weights, opts) };
}
const inputsOf = (io: { inputs: Record<string, TJson> }) => Object.fromEntries(Object.entries(io.inputs).map(([k, v]) => [k, tensorOf(v)]));

describe('op table (vision_ops.json) agrees with the executor', () => {
  it('SUPPORTED_OPS == keys of vision_ops.json', () => {
    expect([...SUPPORTED_OPS].sort()).toEqual(Object.keys(OPS.ops).sort());
  });
  it('covers the ops ADR-0114 / M6 names', () => {
    for (const op of ['aten.convolution', 'aten.relu', 'aten.add', 'aten.mul', 'aten.addmm', 'aten.linear', 'aten.layer_norm',
      'aten.native_group_norm', 'aten.group_norm', 'aten.instance_norm', 'aten.batch_norm', 'aten.max_pool2d', 'aten.max_pool3d',
      'aten.avg_pool2d', 'aten.avg_pool3d', 'aten.upsample_nearest2d', 'aten.upsample_nearest3d', 'aten.upsample_bilinear2d',
      'aten.upsample_trilinear3d', 'aten.cat', 'aten._softmax', 'aten.sigmoid', 'aten.gelu', 'aten.leaky_relu', 'aten.permute',
      'aten.view', 'aten.reshape', 'aten.argmax', 'aten.scaled_dot_product_attention']) expect(SUPPORTED_OPS).toContain(op);
  });
  it('every op used by a golden graph is supported', () => {
    const used = new Set<string>();
    for (const c of G.cases) for (const n of c.graph.nodes) used.add(baseOp(n.op));
    for (const m of ['unet3d_tiny', 'seg2d_tiny', 'vit_tiny']) {
      const g = JSON.parse(readFileSync(here(`../goldens/wgsl/${m}.graph.json`), 'utf8')) as OpGraph;
      for (const n of g.nodes) used.add(baseOp(n.op));
    }
    for (const op of used) expect(SUPPORTED_OPS, op).toContain(op);
  });
  it('baseOp strips the overload', () => {
    expect(baseOp('aten.add.Tensor')).toBe('aten.add');
    expect(baseOp('aten._softmax.default')).toBe('aten._softmax');
    expect(baseOp('aten.relu')).toBe('aten.relu');
  });
  it('rejects unsupported ops with the full list', () => {
    const g: OpGraph = { version: 1, inputs: [{ name: 'x', shape: [2, 2] }], nodes: [
      { op: 'aten.fft_fft.default', inputs: ['x'], attrs: {}, output: 'a' },
      { op: 'aten.cumsum.default', inputs: ['a'], attrs: { dim: 0 }, output: 'b' }], outputs: ['b'] };
    expect(() => VisionModel.compile(g, new Map())).toThrow(/unsupported op.*aten\.fft_fft.*aten\.cumsum/s);
  });
  it('rejects a bad graph version and unknown value names', () => {
    expect(() => VisionModel.compile({ version: 2, inputs: [], nodes: [], outputs: [] } as unknown as OpGraph, new Map())).toThrow(/version/);
    const g: OpGraph = { version: 1, inputs: [{ name: 'x', shape: [2] }], nodes: [{ op: 'aten.relu.default', inputs: ['nope'], attrs: {}, output: 'y' }], outputs: ['y'] };
    expect(() => VisionModel.compile(g, new Map())).toThrow(/unknown value 'nope'/);
  });
});

describe('model parity via op-graph (CPU reference vs PyTorch)', () => {
  it('tiny 3D UNet (MONAI BasicUNet) + channel softmax + argmax', () => {
    const { model, io } = loadModel('unet3d_tiny');
    const out = runCpu(model, inputsOf(io));
    for (const k of ['logits', 'probs']) {
      const e = tensorOf(io.expected[k]);
      expect(shapeEq(out[k].shape, e.shape)).toBe(true);
      expect(relErr(out[k].data, e.data), k).toBeLessThanOrEqual(1e-5);
    }
    // labels: exact wherever the reference top-2 margin is not a numerical tie
    const probs = tensorOf(io.expected.probs), lab = tensorOf(io.expected.labels);
    const C = probs.shape[1], S = probs.data.length / C;
    let checked = 0;
    for (let s = 0; s < S; s++) {
      const p = Array.from({ length: C }, (_, c) => probs.data[c * S + s]).sort((a, b) => b - a);
      if (p[0] - p[1] < 1e-4) continue;
      checked++;
      expect(out.labels.data[s]).toBe(lab.data[s]);
    }
    expect(checked).toBeGreaterThan(S * 0.95);
  }, 120_000);
  it('small 2D seg net (BN eval, GroupNorm, bilinear, skip cat, sigmoid)', () => {
    const { model, io } = loadModel('seg2d_tiny');
    const out = runCpu(model, inputsOf(io));
    expect(relErr(out.mask.data, tensorOf(io.expected.mask).data)).toBeLessThanOrEqual(1e-5);
  });
  it('ViT-Tiny width (patch-embed, cls/pos, SDPA, GELU MLP) → logits + JEPA features', () => {
    const { model, io } = loadModel('vit_tiny');
    const out = runCpu(model, inputsOf(io));
    for (const k of ['logits', 'features']) {
      const e = tensorOf(io.expected[k]);
      expect(shapeEq(out[k].shape, e.shape), k).toBe(true);
      expect(relErr(out[k].data, e.data), k).toBeLessThanOrEqual(1e-5);
    }
  }, 60_000);
  it('input shape mismatch is rejected', () => {
    const { model } = loadModel('seg2d_tiny');
    expect(() => runCpu(model, { x: { shape: [1, 3, 8, 8], data: new Float32Array(192) } })).toThrow(/shape/);
    expect(() => runCpu(model, {})).toThrow(/missing input 'x'/);
  });
});

describe('memory planner', () => {
  it('reuses buffers: slot bytes < naive bytes, and no two live values share a slot', () => {
    const { model } = loadModel('unet3d_tiny');
    const p = model.plan;
    expect(p.slots.length).toBeGreaterThan(0);
    expect(p.slots.length).toBeLessThan(Object.keys(p.values).length);
    expect(p.totalBytes).toBeLessThan(p.naiveBytes);
    expect(p.peakLiveBytes).toBeLessThanOrEqual(p.totalBytes);
    expect(p.totalBytes).toBe(p.slots.reduce((a, b) => a + b, 0));
    const vals = Object.entries(p.values);
    for (let i = 0; i < vals.length; i++) for (let j = i + 1; j < vals.length; j++) {
      const [na, a] = vals[i], [nb, b] = vals[j];
      if (a.slot !== b.slot || a.aliasOf || b.aliasOf) continue; // a view shares its root's storage by design
      const overlap = a.def <= b.lastUse && b.def <= a.lastUse;
      expect(overlap, `${na} [${a.def},${a.lastUse}] and ${nb} [${b.def},${b.lastUse}] share slot ${a.slot}`).toBe(false);
    }
    for (const o of ['logits', 'probs', 'labels']) expect(p.values[o].lastUse).toBe(Number.POSITIVE_INFINITY);
    expect(p.tiled).toEqual([]);
  });
  it('views (reshape/flatten/transpose-free) alias their source instead of copying', () => {
    const { model } = loadModel('vit_tiny');
    const v = model.plan.values;
    const flat = Object.keys(v).find((k) => k.startsWith('flatten'))!;
    expect(v[flat].aliasOf).toBe('conv2d');
  });
  it('tiles spatial kernels when a binding would exceed maxStorageBufferBindingSize — result is bit-identical', () => {
    const base = loadModel('unet3d_tiny');
    const LIM = 256 * 1024; // activations are 512 KiB–1.5 MiB; one instance-norm row (32³·4 B) still fits
    const small = loadModel('unet3d_tiny', { maxBindingBytes: LIM });
    expect(small.model.plan.maxBindingBytes).toBe(LIM);
    const strategies = new Set(small.model.plan.tiled.map((t) => t.strategy));
    expect(strategies.has('spatial-slab')).toBe(true);
    for (const t of small.model.plan.tiled) expect(t.tiles).toBeGreaterThan(1);
    expect(strategies.has('inner-chunk')).toBe(true); // channel softmax / argmax over a 32³ volume
    for (const b of small.model.plan.bindings) expect(b.bytes).toBeLessThanOrEqual(LIM);
    const a = runCpu(base.model, inputsOf(base.io)), b = runCpu(small.model, inputsOf(small.io));
    for (const k of ['logits', 'probs', 'labels']) expect(Array.from(b[k].data)).toEqual(Array.from(a[k].data));
  }, 120_000);
  it('splits flat ranges / slabs for seg2d under an 8 KiB binding limit — bit-identical', () => {
    const LIM = 8 * 1024;
    const base = loadModel('seg2d_tiny'), small = loadModel('seg2d_tiny', { maxBindingBytes: LIM });
    expect(small.model.plan.tiled.length).toBeGreaterThan(0);
    for (const b of small.model.plan.bindings) expect(b.bytes, b.value).toBeLessThanOrEqual(LIM);
    const a = runCpu(base.model, inputsOf(base.io)), b = runCpu(small.model, inputsOf(small.io));
    expect(Array.from(b.mask.data)).toEqual(Array.from(a.mask.data));
  });
  it('every kernel golden under a 1 KiB binding limit either tiles bit-identically or refuses clearly', () => {
    const LIM = 1024;
    let tiled = 0, refused = 0;
    const strategies = new Set<string>();
    for (const c of G.cases) {
      const w = new Map(Object.entries(c.tensors).map(([k, v]) => [k, tensorOf(v)]));
      const inputs = Object.fromEntries(Object.entries(c.inputs).map(([k, v]) => [k, tensorOf(v)]));
      let small: VisionModel;
      try { small = VisionModel.compile(c.graph, w, { maxBindingBytes: LIM }); } catch (e) {
        expect(String(e), c.name).toMatch(/maxStorageBufferBindingSize/);
        refused++;
        continue;
      }
      for (const b of small.plan.bindings) expect(b.bytes, `${c.name} ${b.value}`).toBeLessThanOrEqual(LIM);
      if (small.plan.tiled.length) tiled++;
      for (const t of small.plan.tiled) strategies.add(t.strategy);
      const a = runCpu(VisionModel.compile(c.graph, w), inputs), b = runCpu(small, inputs);
      for (const k of Object.keys(a)) expect(Array.from(b[k].data), `${c.name}:${k}`).toEqual(Array.from(a[k].data));
    }
    expect(tiled).toBeGreaterThanOrEqual(25);
    expect(refused).toBeLessThan(G.cases.length / 3);
    for (const s of ['spatial-slab', 'flat', 'rows', 'inner-chunk']) expect(strategies, s).toContain(s);
  }, 60_000);
  it('refuses clearly when a single unit cannot fit a binding', () => {
    expect(() => loadModel('vit_tiny', { maxBindingBytes: 1024 })).toThrow(/maxStorageBufferBindingSize/);
  });
});

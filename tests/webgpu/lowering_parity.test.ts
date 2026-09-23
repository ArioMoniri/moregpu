// Cross-component parity: graphs produced by the PYTHON LOWERING (lowering.py target 'wgsl', written by
// tests/goldens/make_lowering_goldens.py) run through the TS executor's CPU reference and must equal PyTorch eager
// within 1e-5 (‖Δ‖∞/‖ref‖∞). This is what makes a mixed fleet (torch + WebGPU workers) return the same numbers.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { VisionModel, runCpu, parseSafetensors, SUPPORTED_OPS, baseOp, type OpGraph } from '../../apps/worker/vision_wgsl.ts';
import { type TJson, tensorOf, relErr, shapeEq } from './golden_util.ts';

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));
function load(name: string) {
  const graph = JSON.parse(readFileSync(here(`../goldens/wgsl/lowered/${name}.graph.json`), 'utf8')) as OpGraph;
  const weights = parseSafetensors(new Uint8Array(readFileSync(here(`../goldens/wgsl/lowered/${name}.safetensors`))));
  const io = JSON.parse(readFileSync(here(`../goldens/wgsl/lowered/${name}.io.json`), 'utf8')) as { inputs: Record<string, TJson>; expected: Record<string, TJson> };
  return { graph, weights, io };
}

describe('python-lowered op-graphs run on the TS executor (CPU reference) == PyTorch', () => {
  for (const name of ['unet3d', 'segvit', 'vit_tiny']) {
    it(`${name}: schema accepted, output within 1e-5 rel of PyTorch`, () => {
      const { graph, weights, io } = load(name);
      expect(graph.version).toBe(1);
      expect(graph.weights).toBe('model.safetensors');
      for (const n of graph.nodes) expect(SUPPORTED_OPS, n.op).toContain(baseOp(n.op));
      const model = VisionModel.compile(graph, weights);
      const inputs = Object.fromEntries(Object.entries(io.inputs).map(([k, v]) => [k, tensorOf(v)]));
      const out = runCpu(model, inputs);
      for (const [k, v] of Object.entries(io.expected)) {
        const e = tensorOf(v);
        expect(out[k], `output ${k}`).toBeDefined();
        expect(shapeEq(out[k].shape, e.shape), `${k} shape ${out[k].shape} vs ${e.shape}`).toBe(true);
        expect(relErr(out[k].data, e.data), k).toBeLessThanOrEqual(1e-5);
      }
    }, 120_000);
  }
  it('the ViT-encoder graphs use select (from unbind), never getitem/unbind', () => {
    const ops = new Set(load('segvit').graph.nodes.map((n) => baseOp(n.op)));
    expect(ops.has('aten.select')).toBe(true);
    expect([...ops].some((o) => o.includes('unbind') || o.includes('getitem'))).toBe(false);
  });
});

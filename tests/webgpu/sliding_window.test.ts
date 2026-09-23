// Sliding-window driver vs monai.inferers.sliding_window_inference (goldens from make_wgsl_goldens.py), with the
// tiny BasicUNet op-graph as the predictor on the CPU reference path. Contract: ≤ 1e-5 rel.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  VisionModel, runCpu, parseSafetensors, slidingWindowInference, slidingWindowStarts, computeImportanceMap, type OpGraph, type Tensor,
} from '../../apps/worker/vision_wgsl.ts';
import { type TJson, tensorOf, relErr, shapeEq } from './golden_util.ts';

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));
const SW = JSON.parse(readFileSync(here('../goldens/wgsl/sliding_window.json'), 'utf8')) as {
  model: string; output: string;
  cases: { name: string; roi: number[]; overlap: number; mode: 'gaussian' | 'constant'; sigma_scale: number; input: TJson; expected: TJson }[];
};

describe('sliding-window geometry (MONAI semantics)', () => {
  it('scan starts: interval=int(roi*(1-overlap)), last window flush with the edge', () => {
    expect(slidingWindowStarts([40, 36, 44], [32, 32, 32], 0.25)).toEqual([[0, 8], [0, 4], [0, 12]]);
    expect(slidingWindowStarts([96], [32], 0.5)).toEqual([[0, 16, 32, 48, 64]]);
    expect(slidingWindowStarts([32], [32], 0.25)).toEqual([[0]]); // roi == image → one window
    expect(slidingWindowStarts([33], [32], 0.99)).toEqual([[0, 1]]); // interval floors to ≥1
  });
  it('gaussian importance map = separable exp(-x²/2σ²), σ=roi·sigma_scale, clamped at max(min,1e-3)', () => {
    const m = computeImportanceMap([4, 5], 'gaussian', 0.125);
    expect(m.length).toBe(20);
    const g = (n: number, i: number) => Math.exp(-((i - (n - 1) / 2) ** 2) / (2 * (n * 0.125) ** 2));
    let mn = Infinity;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 5; j++) mn = Math.min(mn, g(4, i) * g(5, j));
    const floor = Math.max(mn, 1e-3);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 5; j++) expect(m[i * 5 + j]).toBeCloseTo(Math.max(g(4, i) * g(5, j), floor), 6);
    expect(Array.from(computeImportanceMap([2, 3], 'constant', 0.125))).toEqual([1, 1, 1, 1, 1, 1]);
  });
});

describe('sliding_window_inference parity with MONAI (CPU reference)', () => {
  const graph = JSON.parse(readFileSync(here(`../goldens/wgsl/${SW.model}.graph.json`), 'utf8')) as OpGraph;
  const weights = parseSafetensors(new Uint8Array(readFileSync(here(`../goldens/wgsl/${graph.weights}`))));
  const model = VisionModel.compile(graph, weights);
  const predictor = (w: Tensor): Tensor => runCpu(model, { x: w })[SW.output];
  for (const c of SW.cases) {
    it(c.name, async () => {
      const x = tensorOf(c.input), e = tensorOf(c.expected);
      const y = await slidingWindowInference(x, c.roi, predictor, { overlap: c.overlap, mode: c.mode, sigmaScale: c.sigma_scale });
      expect(shapeEq(y.shape, e.shape), `${y.shape} vs ${e.shape}`).toBe(true);
      expect(relErr(y.data, e.data)).toBeLessThanOrEqual(1e-5);
    }, 180_000);
  }
  it('rejects roi rank mismatch and bad overlap', async () => {
    const x = { shape: [1, 1, 8, 8, 8], data: new Float32Array(512) };
    await expect(slidingWindowInference(x, [4, 4], (w) => w)).rejects.toThrow(/roi/);
    await expect(slidingWindowInference(x, [4, 4, 4], (w) => w, { overlap: 1 })).rejects.toThrow(/overlap/);
  });
  it('identity predictor reproduces the input exactly (weights normalise out)', async () => {
    const x = { shape: [2, 1, 9, 7, 10], data: Float32Array.from({ length: 2 * 9 * 7 * 10 }, (_, i) => Math.sin(i)) };
    const y = await slidingWindowInference(x, [4, 4, 4], (w) => w, { overlap: 0.5, mode: 'gaussian' });
    expect(relErr(y.data, x.data)).toBeLessThanOrEqual(1e-6);
  });
});

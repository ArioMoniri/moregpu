// CPU-reference kernels vs PyTorch goldens, one small op-graph per kernel (tests/goldens/wgsl/kernels.json).
// The CPU reference mirrors each WGSL kernel's algorithm (same launches, uniforms, loop order and fp32 rounding),
// so a pass here plus the WGSL-vs-CPU agreement in deno_webgpu_test.ts together pin the GPU path to PyTorch.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  VisionModel, runCpu, WGSL_KERNELS, CPU_KERNELS, f32ToF16Bits, f16BitsToF32, wgslSource,
} from '../../apps/worker/vision_wgsl.ts';
import { type KernelGoldens, tensorMap, tensorOf, relErr, shapeEq } from './golden_util.ts';

const G: KernelGoldens = JSON.parse(readFileSync(fileURLToPath(new URL('../goldens/wgsl/kernels.json', import.meta.url)), 'utf8'));

describe('kernel goldens (CPU reference vs PyTorch, fp32)', () => {
  it('has a meaningful number of cases', () => {
    expect(G.cases.length).toBeGreaterThanOrEqual(90);
  });
  for (const c of G.cases) {
    it(c.name, () => {
      const model = VisionModel.compile(c.graph, tensorMap(c.tensors));
      const inputs: Record<string, { shape: number[]; data: Float32Array }> = {};
      for (const [k, v] of Object.entries(c.inputs)) inputs[k] = tensorOf(v);
      const out = runCpu(model, inputs);
      for (const [name, exp] of Object.entries(c.expected)) {
        const e = tensorOf(exp);
        const got = out[name];
        expect(got, `missing output ${name}`).toBeDefined();
        expect(shapeEq(got.shape, e.shape), `${name} shape ${got.shape} vs ${e.shape}`).toBe(true);
        if (c.name.startsWith('argmax')) expect(Array.from(got.data)).toEqual(Array.from(e.data));
        else expect(relErr(got.data, e.data), `${c.name}:${name}`).toBeLessThanOrEqual(c.tol);
      }
    });
  }
});

describe('fp16 weight storage (CPU mirror of the shader-f16 variants)', () => {
  it('f16 bit conversion is exact on representable values and rounds to nearest-even', () => {
    const v = new Float32Array([0, -0, 1, -2.5, 65504, 6.103515625e-5, 5.960464477539063e-8, 1 + 2 ** -11, 1 + 3 * 2 ** -11, 70000, -Infinity]);
    const back = f16BitsToF32(f32ToF16Bits(v));
    expect(Array.from(back.slice(0, 7))).toEqual(Array.from(v.slice(0, 7)));
    expect(back[7]).toBe(1); // tie → even
    expect(back[8]).toBe(1 + 2 * 2 ** -10); // tie → even (up)
    expect(back[9]).toBe(Infinity); // overflow
    expect(back[10]).toBe(-Infinity);
  });
  for (const name of ['conv2d_3x3_pad1', 'conv3d_stride_dil_groups', 'conv_transpose3d_k2s2', 'linear_3d_input', 'patch_embed_conv2d_k4s4']) {
    it(`${name} with f16 weights stays within fp16 tolerance`, () => {
      const c = G.cases.find((x) => x.name === name)!;
      const model = VisionModel.compile(c.graph, tensorMap(c.tensors), { f16Weights: true });
      expect(model.f16Weights).toBe(true);
      const inputs: Record<string, { shape: number[]; data: Float32Array }> = {};
      for (const [k, v] of Object.entries(c.inputs)) inputs[k] = tensorOf(v);
      const out = runCpu(model, inputs);
      const [oname, exp] = Object.entries(c.expected)[0];
      const err = relErr(out[oname].data, tensorOf(exp).data);
      expect(err).toBeLessThanOrEqual(3e-3);
      expect(err).toBeGreaterThan(0); // really rounded the weights
    });
  }
});

describe('every WGSL kernel has a CPU mirror', () => {
  it('kernel tables line up', () => {
    const names = Object.keys(WGSL_KERNELS).sort();
    expect(names.length).toBeGreaterThanOrEqual(12);
    expect(Object.keys(CPU_KERNELS).sort()).toEqual(names);
    for (const k of names) {
      const src = wgslSource(k, false);
      expect(src).toContain('@compute');
      expect(src).toContain('fn main');
    }
  });
  it('f16 variants exist for the weight-heavy kernels and enable f16 first', () => {
    for (const k of ['conv', 'convT', 'matmul']) {
      const src = wgslSource(k, true);
      expect(src.startsWith('enable f16;')).toBe(true);
      expect(src).toContain('array<f16>');
    }
  });
});

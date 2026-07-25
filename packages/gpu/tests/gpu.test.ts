import { describe, it, expect } from 'vitest';
import {
  cpuVectorAdd,
  cpuSaxpy,
  cpuMatmul,
  maxAbsDiff,
  getKernel,
  KERNELS,
  VECTOR_ADD,
  MATMUL,
  hasWebGpu,
  acquireGpu,
} from '../src/index.js';

describe('CPU reference kernels (golden oracle + fallback)', () => {
  it('vector_add', () => {
    expect(Array.from(cpuVectorAdd(new Float32Array([1, 2, 3]), new Float32Array([10, 20, 30])))).toEqual([11, 22, 33]);
  });

  it('saxpy', () => {
    expect(Array.from(cpuSaxpy(new Float32Array([1, 2, 3]), new Float32Array([10, 10, 10]), 2))).toEqual([12, 14, 16]);
  });

  it('matmul (2x3 · 3x2)', () => {
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]] → [[58,64],[139,154]]
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([7, 8, 9, 10, 11, 12]);
    expect(Array.from(cpuMatmul(a, b, { M: 2, N: 2, K: 3 }))).toEqual([58, 64, 139, 154]);
  });

  it('matmul identity is a no-op', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const id = new Float32Array([1, 0, 0, 1]);
    expect(Array.from(cpuMatmul(a, id, { M: 2, N: 2, K: 2 }))).toEqual([1, 2, 3, 4]);
  });

  it('maxAbsDiff detects agreement and disagreement', () => {
    expect(maxAbsDiff(new Float32Array([1, 2]), new Float32Array([1, 2]))).toBe(0);
    expect(maxAbsDiff(new Float32Array([1, 2]), new Float32Array([1, 5]))).toBe(3);
  });
});

describe('kernel registry', () => {
  it('exposes the real kernels', () => {
    expect(Object.keys(KERNELS).sort()).toEqual(['matmul', 'saxpy', 'vector_add']);
    expect(getKernel('vector_add')).toBe(VECTOR_ADD);
  });

  it('throws on an unknown kernel', () => {
    expect(() => getKernel('nope')).toThrow(/unknown kernel/i);
  });

  it('every kernel carries valid WGSL metadata', () => {
    for (const spec of Object.values(KERNELS)) {
      expect(spec.code).toContain('@compute');
      expect(spec.code).toContain('fn main');
      expect(spec.workgroupSize).toBeGreaterThan(0);
      expect(spec.inputs).toBeGreaterThanOrEqual(1);
    }
  });

  it('matmul declares its dimension uniforms', () => {
    expect(MATMUL.uniforms).toEqual(['M', 'N', 'K']);
  });
});

describe('GPU acquisition degrades cleanly without a device', () => {
  it('reports WebGPU availability honestly', () => {
    expect(typeof hasWebGpu()).toBe('boolean');
  });

  it('throws a clear error when no GPU runtime is present (Node/Vitest)', async () => {
    if (hasWebGpu()) return; // on a WebGPU runtime this path is covered by the Deno smoke test
    await expect(acquireGpu()).rejects.toThrow(/no WebGPU|no usable GPU/i);
  });
});

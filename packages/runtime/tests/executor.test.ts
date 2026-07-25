import { describe, it, expect } from 'vitest';
import {
  CpuKernelExecutor,
  NativeAccelExecutor,
  selectExecutor,
  BUILTIN_KERNELS,
  type KernelExecutor,
} from '../src/index.js';

describe('CpuKernelExecutor — built-in kernels', () => {
  const cpu = new CpuKernelExecutor();

  it('reports the wasm-cpu backend', () => {
    expect(cpu.backend).toBe('wasm-cpu');
  });

  it('runs vector_add', async () => {
    const out = await cpu.execute('vector_add', { a: [1, 2, 3], b: [10, 20, 30] });
    expect(out).toEqual([11, 22, 33]);
  });

  it('runs vector_scale with a scalar', async () => {
    const out = await cpu.execute('vector_scale', { a: [1, 2, 3], scalar: 2.5 });
    expect(out).toEqual([2.5, 5, 7.5]);
  });

  it('runs saxpy (scalar*x + y)', async () => {
    const out = await cpu.execute('saxpy', { x: [1, 2, 3], y: [10, 10, 10], scalar: 2 });
    expect(out).toEqual([12, 14, 16]); // 2*[1,2,3] + [10,10,10]
  });

  it('runs dot as a reduction to a single element', async () => {
    const out = await cpu.execute('dot', { a: [1, 2, 3], b: [4, 5, 6] });
    expect(out).toEqual([32]); // 1*4 + 2*5 + 3*6
  });

  it('rejects an unknown kernel by name', async () => {
    await expect(cpu.execute('nonexistent_kernel', { a: [1], b: [1] })).rejects.toThrow(/unknown kernel/i);
  });

  it('validates mismatched input lengths', async () => {
    await expect(cpu.execute('vector_add', { a: [1, 2], b: [1] })).rejects.toThrow(/length/i);
  });

  it('exposes its supported kernel names', () => {
    expect(cpu.supports('vector_add')).toBe(true);
    expect(cpu.supports('made_up')).toBe(false);
    expect(BUILTIN_KERNELS).toContain('vector_add');
  });
});

describe('selectExecutor — tiered adaptivity (ADR 0007)', () => {
  it('prefers a provided WebGPU executor over CPU', () => {
    const fakeGpu: KernelExecutor = { backend: 'webgpu', supports: () => true, execute: async () => [0] };
    const chosen = selectExecutor({ webgpu: fakeGpu, cpu: new CpuKernelExecutor() });
    expect(chosen.backend).toBe('webgpu');
  });

  it('prefers native-accel over WebGPU and CPU when present', () => {
    const fakeGpu: KernelExecutor = { backend: 'webgpu', supports: () => true, execute: async () => [0] };
    const chosen = selectExecutor({ nativeAccel: new NativeAccelExecutor(), webgpu: fakeGpu, cpu: new CpuKernelExecutor() });
    expect(chosen.backend).toBe('native-accel');
  });

  it('falls back to CPU when no GPU executor is available', () => {
    const chosen = selectExecutor({ webgpu: null, cpu: new CpuKernelExecutor() });
    expect(chosen.backend).toBe('wasm-cpu');
  });
});

describe('tensor-core boundary (Fix 1)', () => {
  it('the WGSL/CPU tier does NOT advertise tensor-core gemm', () => {
    expect(new CpuKernelExecutor().supports('gemm_tensor')).toBe(false);
  });

  it('only native-accel advertises gemm_tensor', () => {
    expect(new NativeAccelExecutor().supports('gemm_tensor')).toBe(true);
  });

  it('native-accel refuses gemm_tensor honestly until the addon is installed', async () => {
    await expect(new NativeAccelExecutor().execute('gemm_tensor', { a: [1], b: [1] })).rejects.toThrow(/addon|not installed/i);
  });
});

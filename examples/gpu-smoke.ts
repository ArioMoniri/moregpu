/**
 * Real-GPU smoke test. Runs the @moregpu/gpu executor on the physical GPU and checks its output
 * against the CPU reference. Run it with Deno (native WebGPU → Metal/Vulkan/D3D12):
 *
 *   deno run --unstable-webgpu --allow-read examples/gpu-smoke.ts
 */
import { GpuExecutor, acquireGpu, cpuVectorAdd, cpuMatmul, maxAbsDiff } from '../packages/gpu/dist/index.js';

const { device, info } = await acquireGpu();
console.log(`GPU: ${info.vendor} ${info.architecture} ${info.device}`.trim());
const gpu = new GpuExecutor(device);

// 1) vector_add over 1M elements
{
  const N = 1_000_000;
  const a = new Float32Array(N).map((_, i) => i % 1000);
  const b = new Float32Array(N).map((_, i) => (i * 3) % 1000);
  const t0 = performance.now();
  const out = await gpu.vectorAdd(a, b);
  const ms = performance.now() - t0;
  const diff = maxAbsDiff(out, cpuVectorAdd(a, b));
  console.log(`vector_add  N=${N}  ${ms.toFixed(1)}ms  maxAbsDiff=${diff}  ${diff === 0 ? 'OK' : 'FAIL'}`);
}

// 2) matmul 512x512 · 512x512 — real GPU throughput
{
  const M = 512, N = 512, K = 512;
  const A = new Float32Array(M * K).map(() => Math.random());
  const B = new Float32Array(K * N).map(() => Math.random());
  const t0 = performance.now();
  const C = await gpu.matmul(A, B, { M, N, K });
  const ms = performance.now() - t0;
  const gflops = (2 * M * N * K) / (ms / 1000) / 1e9;
  const diff = maxAbsDiff(C, cpuMatmul(A, B, { M, N, K }));
  const ok = diff < 1e-2; // fp accumulation tolerance
  console.log(`matmul ${M}x${N}x${K}  ${ms.toFixed(1)}ms  ~${gflops.toFixed(1)} GFLOP/s  maxAbsDiff=${diff.toExponential(1)}  ${ok ? 'OK' : 'FAIL'}`);
  if (!ok) Deno.exit(1);
}

console.log('SMOKE_OK');

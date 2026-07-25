/**
 * Native GPU pool demo — the whole point of MoreGPU, running for real.
 *
 * Spins up N worker nodes (each an independent GPU executor on the physical device), shards a large
 * matmul by row-blocks across them, runs every shard on the real GPU, pools the row-blocks back into
 * one matrix, and verifies the pooled result against the CPU reference.
 *
 *   deno run --unstable-webgpu --allow-read examples/pool-demo.ts
 */
import { GpuExecutor, acquireGpu, cpuMatmul, maxAbsDiff } from '../packages/gpu/dist/index.js';

const WORKERS = 4;
const M = 1024, N = 1024, K = 1024; // C[MxN] = A[MxK] · B[KxN]

const { device, info } = await acquireGpu();
console.log(`\n  MoreGPU pool · device: ${(info.device || info.vendor || 'gpu').trim() || 'gpu'} · ${WORKERS} workers\n`);

// Each "node" is its own executor sharing the one physical GPU (stand-in for N machines).
const nodes = Array.from({ length: WORKERS }, (_, i) => ({ id: `worker-${i}`, gpu: new GpuExecutor(device) }));

const A = new Float32Array(M * K).map(() => Math.random());
const B = new Float32Array(K * N).map(() => Math.random());

// Shard C by contiguous row-blocks — an embarrassingly-parallel split: C[r0:r1] = A[r0:r1] · B.
const rowsPer = Math.ceil(M / WORKERS);
const shards = nodes.map((node, i) => {
  const r0 = i * rowsPer;
  const r1 = Math.min(M, r0 + rowsPer);
  return { node, r0, r1, rows: r1 - r0 };
}).filter((s) => s.rows > 0);

const t0 = performance.now();
const results = await Promise.all(
  shards.map(async (s) => {
    const Ablock = A.subarray(s.r0 * K, s.r1 * K);
    const t = performance.now();
    const Cblock = await s.node.gpu.matmul(new Float32Array(Ablock), B, { M: s.rows, N, K });
    const ms = performance.now() - t;
    console.log(`  ${s.node.id}: rows ${s.r0}–${s.r1}  (${s.rows}×${N})  ${ms.toFixed(1)}ms on GPU`);
    return { s, Cblock };
  }),
);
const wall = performance.now() - t0;

// Pool: stitch row-blocks back together in order.
const C = new Float32Array(M * N);
for (const { s, Cblock } of results) C.set(Cblock, s.r0 * N);

// Verify the pooled GPU result against the CPU reference.
const diff = maxAbsDiff(C, cpuMatmul(A, B, { M, N, K }));
const gflops = (2 * M * N * K) / (wall / 1000) / 1e9;
console.log(`\n  pooled ${M}×${N} result in ${wall.toFixed(1)}ms  ·  ~${gflops.toFixed(1)} GFLOP/s aggregate`);
console.log(`  verified vs CPU reference: maxAbsDiff=${diff.toExponential(1)}  ${diff < 1e-2 ? '✓ MATCH' : '✗ MISMATCH'}\n`);
if (!(diff < 1e-2)) Deno.exit(1);
console.log('  POOL_OK\n');

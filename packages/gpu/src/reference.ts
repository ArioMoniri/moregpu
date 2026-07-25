/**
 * CPU reference implementations of the GPU kernels. Two jobs:
 *  1. Verify GPU output in tests and in redundant-compute checks (golden oracle).
 *  2. Serve as the automatic fallback when no GPU is present (see selectBackend).
 * The GPU and CPU paths must agree within floating-point tolerance.
 */

export interface MatmulDims {
  M: number;
  N: number;
  K: number;
}

export function cpuVectorAdd(a: Float32Array, b: Float32Array): Float32Array {
  if (a.length !== b.length) throw new Error('gpu/ref: vector_add length mismatch');
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! + b[i]!;
  return out;
}

export function cpuSaxpy(x: Float32Array, y: Float32Array, scalar: number): Float32Array {
  if (x.length !== y.length) throw new Error('gpu/ref: saxpy length mismatch');
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = scalar * x[i]! + y[i]!;
  return out;
}

export function cpuMatmul(a: Float32Array, b: Float32Array, dims: MatmulDims): Float32Array {
  const { M, N, K } = dims;
  if (a.length !== M * K) throw new Error('gpu/ref: A has wrong length');
  if (b.length !== K * N) throw new Error('gpu/ref: B has wrong length');
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let acc = 0;
      for (let k = 0; k < K; k++) acc += a[i * K + k]! * b[k * N + j]!;
      out[i * N + j] = acc;
    }
  }
  return out;
}

/** Max absolute difference between two vectors — used to assert GPU/CPU agreement. */
export function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return Infinity;
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

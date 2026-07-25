/**
 * WGSL compute kernels that run on the physical GPU (Metal / Vulkan / D3D12 via wgpu or Dawn).
 * Each kernel is a self-contained compute shader plus the metadata the executor needs to bind
 * buffers and size the dispatch. These are the real work MoreGPU pools across nodes.
 */

export interface KernelSpec {
  name: string;
  /** WGSL source. */
  code: string;
  /** Workgroup size along X declared in the shader (for dispatch math). */
  workgroupSize: number;
  /** Number of read-only storage inputs (bound 0..inputs-1). */
  inputs: number;
  /** Optional scalar uniforms (f32), bound after the storage buffers. */
  uniforms?: string[];
}

/** out[i] = a[i] + b[i] */
export const VECTOR_ADD: KernelSpec = {
  name: 'vector_add',
  workgroupSize: 64,
  inputs: 2,
  code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&a)) { return; }
  out[i] = a[i] + b[i];
}`,
};

/** out[i] = scalar * x[i] + y[i] (SAXPY) */
export const SAXPY: KernelSpec = {
  name: 'saxpy',
  workgroupSize: 64,
  inputs: 2,
  uniforms: ['scalar'],
  code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&x)) { return; }
  out[i] = params.x * x[i] + y[i];
}`,
};

/**
 * Tiled dense matmul: C[MxN] = A[MxK] * B[KxN]. Dispatched over an (N, M) grid of 16x16 tiles.
 * Dimensions arrive as a uniform vec4<u32> (M, N, K, _). This is the showcase kernel that actually
 * exercises the GPU's throughput.
 */
export const MATMUL: KernelSpec = {
  name: 'matmul',
  workgroupSize: 16, // 16x16 tile
  inputs: 2,
  uniforms: ['M', 'N', 'K'],
  code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let M = dims.x; let N = dims.y; let K = dims.z;
  let row = gid.y; let col = gid.x;
  if (row >= M || col >= N) { return; }
  var acc = 0.0;
  for (var k = 0u; k < K; k = k + 1u) {
    acc = acc + A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = acc;
}`,
};

export const KERNELS: Record<string, KernelSpec> = {
  [VECTOR_ADD.name]: VECTOR_ADD,
  [SAXPY.name]: SAXPY,
  [MATMUL.name]: MATMUL,
};

export function getKernel(name: string): KernelSpec {
  const k = KERNELS[name];
  if (!k) throw new Error(`gpu: unknown kernel "${name}"`);
  return k;
}

/**
 * Core domain types shared across MoreGPU. These describe *what* flows through the
 * system; `messages.ts` describes *how* it is framed on the wire.
 */

/**
 * Compute backend a worker can offer:
 *  - 'native-accel' — signed N-API addon (CUDA/ROCm/Metal/Vulkan-coopmat); the ONLY tier that reaches
 *    tensor-core / cooperative-matrix GEMM, hence the only one that may make matmul claims (ADR 0007).
 *  - 'webgpu'       — headless WGSL compute; portable but tensor-core-INDEPENDENT (no wmma in shipping Dawn).
 *  - 'wasm-cpu'     — WASM SIMD+threads fallback when no usable GPU exists.
 */
export type Backend = 'native-accel' | 'webgpu' | 'wasm-cpu';

/** Which native accelerator stack a 'native-accel' worker is bound to. */
export type NativeAccel = 'cuda' | 'rocm' | 'metal' | 'vulkan';

export type Vendor = 'nvidia' | 'amd' | 'intel' | 'apple' | 'unknown';

export type OS = 'windows' | 'macos' | 'linux' | 'unknown';

/** What a single donor node can do, reported at registration and refreshed on heartbeat. */
export interface NodeCapability {
  nodeId: string;
  backend: Backend;
  vendor: Vendor;
  os: OS;
  /** Usable VRAM in MB. 0 for CPU-only nodes. */
  vramMB: number;
  logicalCores: number;
  /** WebGPU maxComputeWorkgroupsPerDimension-ish ceiling; 0 for CPU nodes. */
  maxComputeInvocations: number;
  supportsF16: boolean;
  /** True only on a native-accel worker with cooperative-matrix/wmma GEMM available (ADR 0007). */
  wmmaSupported?: boolean;
  /** The native accelerator stack, when backend is 'native-accel'. */
  nativeAccel?: NativeAccel;
}

/** Live utilization a node reports so the scheduler can throttle without hurting the local user. */
export interface NodeLoad {
  /** 0..1 fraction of the node currently consumed by the interactive user. */
  interactiveLoad: number;
  /** 0..1 fraction we are allowed to use right now (after adaptive throttle). */
  availableFraction: number;
  temperatureC?: number;
  onBattery?: boolean;
}

/** A whole job an admin submits; the scheduler shards it across nodes when the kernel allows. */
export interface Job {
  jobId: string;
  tenantId: string;
  kernel: string;
  /** Opaque, already-encrypted work payload descriptor; protocol never sees plaintext. */
  totalUnits: number;
  shardable: boolean;
  /** Reduction used to pool shard outputs, e.g. 'concat' | 'sum' | 'mean' | 'none'. */
  reduce: ReduceOp;
}

export type ReduceOp = 'concat' | 'sum' | 'mean' | 'none';

/** One shard of a job assigned to one node. */
export interface Shard {
  shardId: string;
  jobId: string;
  index: number;
  unitStart: number;
  unitEnd: number;
  /** Sealed (encrypted) work-unit blob, base64. The coordinator relays; it does not decrypt. */
  sealed: string;
}

/** Result of executing one shard, returned by a worker. */
export interface ShardResult {
  shardId: string;
  jobId: string;
  ok: boolean;
  /** Sealed (encrypted) output blob, base64, when ok. */
  sealed?: string;
  error?: string;
  elapsedMs?: number;
}

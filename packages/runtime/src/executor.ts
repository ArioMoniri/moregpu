import type { Backend } from '@moregpu/protocol';

/** Inputs a kernel can receive. Arrays are the data; scalar is an optional coefficient. */
export interface KernelInput {
  a?: number[];
  b?: number[];
  x?: number[];
  y?: number[];
  scalar?: number;
}

/**
 * A compute backend able to run named kernels. The CPU implementation lives here; the WebGPU
 * implementation is injected by the worker app (it needs a real `navigator.gpu`, i.e. a headless
 * browser or a native wgpu binding). Both satisfy this one interface so the worker is backend-agnostic.
 */
export interface KernelExecutor {
  readonly backend: Backend;
  supports(kernel: string): boolean;
  execute(kernel: string, input: KernelInput): Promise<number[]>;
}

/** The reference kernel set every backend must implement identically (used for redundant verification). */
export const BUILTIN_KERNELS = ['vector_add', 'vector_scale', 'saxpy', 'dot'] as const;
export type BuiltinKernel = (typeof BUILTIN_KERNELS)[number];

function requireEqualLength(a: number[], b: number[]): void {
  if (a.length !== b.length) {
    throw new Error(`executor: input length mismatch (${a.length} vs ${b.length})`);
  }
}

/**
 * Pure-JS/WASM-friendly CPU executor. Deterministic, dependency-free, and used both as the fallback
 * when no GPU is present and as the golden oracle for verifying GPU results by redundant compute.
 */
export class CpuKernelExecutor implements KernelExecutor {
  readonly backend: Backend = 'wasm-cpu';

  supports(kernel: string): boolean {
    return (BUILTIN_KERNELS as readonly string[]).includes(kernel);
  }

  async execute(kernel: string, input: KernelInput): Promise<number[]> {
    switch (kernel) {
      case 'vector_add': {
        const { a = [], b = [] } = input;
        requireEqualLength(a, b);
        return a.map((v, i) => v + b[i]!);
      }
      case 'vector_scale': {
        const { a = [], scalar = 1 } = input;
        return a.map((v) => v * scalar);
      }
      case 'saxpy': {
        const { x = [], y = [], scalar = 1 } = input;
        requireEqualLength(x, y);
        return x.map((v, i) => scalar * v + y[i]!);
      }
      case 'dot': {
        const { a = [], b = [] } = input;
        requireEqualLength(a, b);
        return [a.reduce((acc, v, i) => acc + v * b[i]!, 0)];
      }
      default:
        throw new Error(`executor: unknown kernel "${kernel}"`);
    }
  }
}

/**
 * The native-accelerator tier (ADR 0007). This is the ONLY executor that advertises `gemm_tensor`
 * (tensor-core / cooperative-matrix GEMM), because shipping WGSL cannot express wmma. In production a
 * signed N-API addon (cuBLASLt / rocWMMA / MPS-MLX) backs it; here it is a stub that reports the tier
 * honestly and refuses `gemm_tensor` at runtime until the addon is installed. Portable kernels delegate
 * to the CPU oracle so the interface stays uniform.
 */
export class NativeAccelExecutor implements KernelExecutor {
  readonly backend: Backend = 'native-accel';
  private readonly cpu = new CpuKernelExecutor();

  supports(kernel: string): boolean {
    return kernel === 'gemm_tensor' || this.cpu.supports(kernel);
  }

  async execute(kernel: string, input: KernelInput): Promise<number[]> {
    if (kernel === 'gemm_tensor') {
      throw new Error('executor: native-accel addon (CUDA/ROCm/Metal) not installed — cannot run gemm_tensor');
    }
    return this.cpu.execute(kernel, input);
  }
}

/**
 * Pick the best available executor, honoring ADR 0007's tiering:
 * native-accel → WebGPU → CPU. Tensor-core GEMM only runs when a native-accel executor is present.
 */
export function selectExecutor(opts: {
  nativeAccel?: KernelExecutor | null;
  webgpu: KernelExecutor | null;
  cpu: KernelExecutor;
}): KernelExecutor {
  return opts.nativeAccel ?? opts.webgpu ?? opts.cpu;
}

/**
 * GPU device acquisition. Uses the standard WebGPU API (`navigator.gpu`), which is provided natively
 * by Deno (`--unstable-webgpu`, wgpu → Metal/Vulkan/D3D12), by the Node `webgpu`/Dawn addon, and by
 * browsers. No vendor lock-in: the same code path lights up whatever real GPU is present.
 */

export interface GpuHandle {
  adapter: GPUAdapter;
  device: GPUDevice;
  info: { vendor: string; architecture: string; device: string };
}

/** True if a WebGPU entry point exists in this runtime (does not guarantee a usable adapter). */
export function hasWebGpu(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu != null;
}

/** Acquire a real GPU device, or throw a clear error if none is available. */
export async function acquireGpu(): Promise<GpuHandle> {
  if (!hasWebGpu()) {
    throw new Error('gpu: no WebGPU in this runtime (run under Deno --unstable-webgpu, Node webgpu addon, or a browser)');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('gpu: requestAdapter returned null (no usable GPU / driver)');
  const device = await adapter.requestDevice();
  const raw = (adapter.info ?? {}) as Partial<GPUAdapterInfo>;
  return {
    adapter,
    device,
    info: {
      vendor: String(raw.vendor ?? 'unknown'),
      architecture: String(raw.architecture ?? 'unknown'),
      device: String(raw.device ?? 'unknown'),
    },
  };
}

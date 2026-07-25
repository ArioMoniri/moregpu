/**
 * Real GPU kernel executor. Given a WGSL KernelSpec, storage inputs, optional scalar uniforms, an
 * output length and a dispatch size, it uploads buffers, runs the compute pass on the physical GPU,
 * and reads results back. High-level helpers (vectorAdd / saxpy / matmul) wrap the common kernels.
 */
import { VECTOR_ADD, SAXPY, MATMUL, type KernelSpec } from './kernels.js';
import type { MatmulDims } from './reference.js';

export interface RunRequest {
  spec: KernelSpec;
  /** Read-only storage inputs, bound at 0..inputs-1. */
  storage: Float32Array[];
  /** Optional uniform payload (typed-array view), bound after the storage + output buffers. */
  uniform?: Uint32Array | Float32Array;
  /** Number of f32 elements in the output buffer. */
  outputLength: number;
  /** Workgroup counts [x, y, z]. */
  dispatch: [number, number, number];
}

export class GpuExecutor {
  constructor(private readonly device: GPUDevice) {}

  async run(req: RunRequest): Promise<Float32Array> {
    const { device } = this;
    const { spec, storage, uniform, outputLength, dispatch } = req;

    const inputBuffers = storage.map((arr) => {
      const buf = device.createBuffer({
        size: Math.max(4, arr.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, arr as GPUAllowSharedBufferSource);
      return buf;
    });

    const outBytes = Math.max(4, outputLength * 4);
    const outBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const entries: GPUBindGroupEntry[] = [
      ...inputBuffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
      { binding: storage.length, resource: { buffer: outBuffer } },
    ];
    let uniformBuffer: GPUBuffer | undefined;
    if (uniform) {
      uniformBuffer = device.createBuffer({ size: uniform.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(uniformBuffer, 0, uniform as GPUAllowSharedBufferSource);
      entries.push({ binding: storage.length + 1, resource: { buffer: uniformBuffer } });
    }

    const module = device.createShaderModule({ code: spec.code });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatch[0], dispatch[1], dispatch[2]);
    pass.end();
    encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, outBytes);
    device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0, outputLength * 4));
    readBuffer.unmap();

    inputBuffers.forEach((b) => b.destroy());
    outBuffer.destroy();
    readBuffer.destroy();
    uniformBuffer?.destroy();
    return result;
  }

  vectorAdd(a: Float32Array, b: Float32Array): Promise<Float32Array> {
    return this.run({
      spec: VECTOR_ADD,
      storage: [a, b],
      outputLength: a.length,
      dispatch: [Math.ceil(a.length / VECTOR_ADD.workgroupSize), 1, 1],
    });
  }

  saxpy(x: Float32Array, y: Float32Array, scalar: number): Promise<Float32Array> {
    const uniform = new Float32Array([scalar, 0, 0, 0]);
    return this.run({
      spec: SAXPY,
      storage: [x, y],
      uniform,
      outputLength: x.length,
      dispatch: [Math.ceil(x.length / SAXPY.workgroupSize), 1, 1],
    });
  }

  matmul(a: Float32Array, b: Float32Array, dims: MatmulDims): Promise<Float32Array> {
    const uniform = new Uint32Array([dims.M, dims.N, dims.K, 0]);
    const tiles = MATMUL.workgroupSize;
    return this.run({
      spec: MATMUL,
      storage: [a, b],
      uniform,
      outputLength: dims.M * dims.N,
      dispatch: [Math.ceil(dims.N / tiles), Math.ceil(dims.M / tiles), 1],
    });
  }
}

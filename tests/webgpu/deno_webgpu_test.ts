// Real WGSL on a real (or software) WebGPU adapter under Deno. Every test is IGNORED when no adapter is available,
// so this file is safe to run anywhere:
//
//   deno test --unstable-webgpu --allow-read tests/webgpu/deno_webgpu_test.ts
//
// On Linux without a GPU, Mesa's lavapipe (apt install mesa-vulkan-drivers) gives Deno/wgpu a Vulkan software
// adapter, with shader-f16. It checks the same PyTorch goldens as the vitest CPU-reference suite (tests/webgpu/*.test.ts).
import {
  VisionModel, runCpu, createGpuRunner, parseSafetensors, wgslSource, WGSL_KERNELS, F16_KERNELS, slidingWindowInference,
  type OpGraph, type Tensor,
} from '../../apps/worker/vision_wgsl.ts';
import { type KernelGoldens, type TJson, tensorMap, tensorOf, relErr, shapeEq } from './golden_util.ts';

const adapter: GPUAdapter | null = await (async () => {
  try { return (await navigator.gpu?.requestAdapter()) ?? null; } catch { return null; }
})();
const hasF16 = !!adapter?.features.has('shader-f16');
const device: GPUDevice | null = adapter
  ? await adapter.requestDevice({
    requiredFeatures: hasF16 ? ['shader-f16'] : [],
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  })
  : null;
if (device) {
  const info = adapter!.info;
  console.log(`[webgpu] adapter: vendor=${info?.vendor || '?'} arch=${info?.architecture || '?'} desc=${info?.description || '?'} f16=${hasF16}`);
} else console.log('[webgpu] no WebGPU adapter — real-GPU tests are ignored');

const T = { ignore: !device, sanitizeResources: false, sanitizeOps: false };
const url = (p: string) => new URL(p, import.meta.url);
const G: KernelGoldens = JSON.parse(await Deno.readTextFile(url('../goldens/wgsl/kernels.json')));
const inputsOf = (ins: Record<string, TJson>) => Object.fromEntries(Object.entries(ins).map(([k, v]) => [k, tensorOf(v)]));
function assert(c: unknown, msg: string): asserts c { if (!c) throw new Error(msg); }

async function loadModel(name: string, opts: Parameters<typeof VisionModel.compile>[2] = {}) {
  const graph = JSON.parse(await Deno.readTextFile(url(`../goldens/wgsl/${name}.graph.json`))) as OpGraph;
  const weights = parseSafetensors(await Deno.readFile(url(`../goldens/wgsl/${graph.weights}`)));
  const io = JSON.parse(await Deno.readTextFile(url(`../goldens/wgsl/${name}.io.json`))) as { inputs: Record<string, TJson>; expected: Record<string, TJson> };
  return { model: VisionModel.compile(graph, weights, opts), io };
}

Deno.test({ name: 'webgpu: every WGSL kernel (and f16 variant) compiles without errors', ...T, fn: async () => {
  for (const k of Object.keys(WGSL_KERNELS)) {
    for (const f16 of hasF16 && F16_KERNELS.includes(k) ? [false, true] : [false]) {
      const mod = device!.createShaderModule({ code: wgslSource(k, f16) });
      const info = await mod.getCompilationInfo();
      const errs = info.messages.filter((m) => m.type === 'error');
      assert(errs.length === 0, `${k}${f16 ? '/f16' : ''}: ${errs.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; ')}`);
    }
  }
} });

Deno.test({ name: 'webgpu: kernel goldens on the GPU (fp32, ≤1e-5 rel)', ...T, fn: async () => {
  const fails: string[] = [];
  for (const c of G.cases) {
    const model = VisionModel.compile(c.graph, tensorMap(c.tensors));
    const gpu = await createGpuRunner(device!, model);
    try {
      const out = await gpu.run(inputsOf(c.inputs));
      for (const [name, exp] of Object.entries(c.expected)) {
        const e = tensorOf(exp), got = out[name];
        if (!got || !shapeEq(got.shape, e.shape)) { fails.push(`${c.name}:${name} shape`); continue; }
        const err = c.name.startsWith('argmax') ? (got.data.every((v, i) => v === e.data[i]) ? 0 : 1) : relErr(got.data, e.data);
        if (!(err <= c.tol)) fails.push(`${c.name}:${name} err=${err.toExponential(2)}`);
      }
    } finally { gpu.destroy(); }
  }
  assert(fails.length === 0, `GPU kernel failures:\n${fails.join('\n')}`);
  console.log(`[webgpu] ${G.cases.length} kernel cases passed on the GPU`);
} });

Deno.test({ name: 'webgpu: binding-limit tiling on the GPU (1 KiB / 512 B limits) matches the goldens', ...T, fn: async () => {
  const fails: string[] = [];
  let tiled = 0;
  for (const LIM of [1024, 512]) for (const c of G.cases) {
    let model: VisionModel;
    try { model = VisionModel.compile(c.graph, tensorMap(c.tensors), { maxBindingBytes: LIM }); } catch { continue; }
    if (!model.plan.tiled.length) continue;
    tiled++;
    const gpu = await createGpuRunner(device!, model);
    try {
      const out = await gpu.run(inputsOf(c.inputs));
      for (const [name, exp] of Object.entries(c.expected)) {
        const e = tensorOf(exp);
        const err = c.name.startsWith('argmax') ? (out[name].data.every((v, i) => v === e.data[i]) ? 0 : 1) : relErr(out[name].data, e.data);
        if (!(err <= c.tol)) fails.push(`${c.name}@${LIM}:${name} err=${err}`);
      }
    } finally { gpu.destroy(); }
  }
  assert(tiled >= 40, `only ${tiled} tiled cases`);
  assert(fails.length === 0, fails.join('\n'));
} });

for (const [name, outs] of [['unet3d_tiny', ['logits', 'probs']], ['seg2d_tiny', ['mask']], ['vit_tiny', ['logits', 'features']]] as const) {
  Deno.test({ name: `webgpu: ${name} parity on the GPU (fp32) and agreement with the CPU reference`, ...T, fn: async () => {
    const { model, io } = await loadModel(name);
    const gpu = await createGpuRunner(device!, model);
    try {
      const inputs = inputsOf(io.inputs);
      const out = await gpu.run(inputs);
      const cpu = runCpu(model, inputs);
      for (const k of outs) {
        const e = tensorOf(io.expected[k]);
        const err = relErr(out[k].data, e.data), vsCpu = relErr(out[k].data, cpu[k].data);
        console.log(`[webgpu] ${name}.${k}: vs torch ${err.toExponential(2)}, vs CPU-ref ${vsCpu.toExponential(2)}`);
        assert(err <= 1e-5, `${name}.${k} err ${err}`);
      }
      // second run reuses resident weights + uniforms (no re-upload) and must be identical
      const again = await gpu.run(inputs);
      for (const k of outs) assert(again[k].data.every((v, i) => v === out[k].data[i]), `${name}.${k} not deterministic`);
    } finally { gpu.destroy(); }
  } });
}

Deno.test({ name: 'webgpu: UNet under a 256 KiB binding limit (spatial slabs + inner chunks) on the GPU', ...T, fn: async () => {
  const { model, io } = await loadModel('unet3d_tiny', { maxBindingBytes: 256 * 1024 });
  assert(model.plan.tiled.length > 0, 'expected tiling');
  const gpu = await createGpuRunner(device!, model);
  try {
    const out = await gpu.run(inputsOf(io.inputs));
    const err = relErr(out.logits.data, tensorOf(io.expected.logits).data);
    assert(err <= 1e-5, `err ${err}`);
  } finally { gpu.destroy(); }
} });

Deno.test({ name: 'webgpu: fp16 weight storage (shader-f16) — ViT and UNet within fp16 tolerance', ignore: !device || !hasF16, sanitizeResources: false, sanitizeOps: false, fn: async () => {
  for (const [name, k] of [['vit_tiny', 'logits'], ['unet3d_tiny', 'logits'], ['seg2d_tiny', 'mask']] as const) {
    const { model, io } = await loadModel(name, { f16Weights: true });
    const gpu = await createGpuRunner(device!, model);
    try {
      const out = await gpu.run(inputsOf(io.inputs));
      const cpu = runCpu(model, inputsOf(io.inputs)); // the CPU mirror rounds weights to f16 identically
      const err = relErr(out[k].data, tensorOf(io.expected[k]).data), vsCpu = relErr(out[k].data, cpu[k].data);
      console.log(`[webgpu] f16 ${name}.${k}: vs torch-fp32 ${err.toExponential(2)}, vs CPU f16 mirror ${vsCpu.toExponential(2)}`);
      assert(err <= 5e-3, `${name} f16 err ${err}`);
      assert(vsCpu <= 1e-5, `${name} f16 GPU vs CPU mirror ${vsCpu}`);
    } finally { gpu.destroy(); }
  }
} });

Deno.test({ name: 'webgpu: sliding-window inference on the GPU equals MONAI (≤1e-5)', ...T, fn: async () => {
  const SW = JSON.parse(await Deno.readTextFile(url('../goldens/wgsl/sliding_window.json'))) as {
    model: string; output: string; cases: { name: string; roi: number[]; overlap: number; mode: 'gaussian' | 'constant'; sigma_scale: number; input: TJson; expected: TJson }[];
  };
  const { model } = await loadModel(SW.model);
  const gpu = await createGpuRunner(device!, model);
  try {
    for (const c of SW.cases) {
      const y = await slidingWindowInference(tensorOf(c.input), c.roi, async (w: Tensor) => (await gpu.run({ x: w }))[SW.output],
        { overlap: c.overlap, mode: c.mode, sigmaScale: c.sigma_scale });
      const err = relErr(y.data, tensorOf(c.expected).data);
      console.log(`[webgpu] sliding window ${c.name}: ${err.toExponential(2)}`);
      assert(err <= 1e-5, `${c.name} err ${err}`);
    }
  } finally { gpu.destroy(); }
} });

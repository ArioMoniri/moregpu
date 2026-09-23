// Browser entry for tests/webgpu/browser.spec.ts. The spec bundles it with esbuild and loads it in a page served over
// http://127.0.0.1 (WebGPU needs a secure context). It then runs the kernel goldens and the model parity cases on
// navigator.gpu with the SAME vision_wgsl.ts that worker.ts ships.
import { VisionModel, createGpuRunner, parseSafetensors, wgslSource, WGSL_KERNELS, type OpGraph } from '../../apps/worker/vision_wgsl.ts';
import { type KernelGoldens, type TJson, tensorMap, tensorOf, relErr, shapeEq } from './golden_util.ts';

interface Result { name: string; ok: boolean; err: number | string }

async function probe(): Promise<{ adapter: boolean; info: string; f16: boolean }> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return { adapter: false, info: 'no navigator.gpu', f16: false };
  const a = await gpu.requestAdapter();
  if (!a) return { adapter: false, info: 'no adapter', f16: false };
  const i = a.info;
  return { adapter: true, info: `${i?.vendor || '?'}/${i?.architecture || '?'}`, f16: a.features.has('shader-f16') };
}

async function run(opts: { limit?: number } = {}): Promise<Result[]> {
  const a = (await (navigator as Navigator & { gpu: GPU }).gpu.requestAdapter())!;
  const device = await a.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize, maxBufferSize: a.limits.maxBufferSize } });
  const res: Result[] = [];
  // Tint (Dawn) is stricter than naga (wgpu): compile every kernel here too and report errors by name.
  // (f32 variants; the device is requested without shader-f16 — the Deno test covers the f16 variants)
  for (const k of Object.keys(WGSL_KERNELS)) {
    const info = await device.createShaderModule({ code: wgslSource(k, false) }).getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error').map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
    res.push({ name: `compile:${k}`, ok: errs.length === 0, err: errs.length ? errs.join('; ') : 0 });
  }
  const G = (await (await fetch('/goldens/kernels.json')).json()) as KernelGoldens;
  const cases = opts.limit ? G.cases.slice(0, opts.limit) : G.cases;
  for (const c of cases) {
    try {
      const model = VisionModel.compile(c.graph, tensorMap(c.tensors));
      const gpu = await createGpuRunner(device, model);
      const out = await gpu.run(Object.fromEntries(Object.entries(c.inputs).map(([k, v]) => [k, tensorOf(v)])));
      gpu.destroy();
      let worst = 0;
      for (const [name, exp] of Object.entries(c.expected)) {
        const e = tensorOf(exp), got = out[name];
        if (!shapeEq(got.shape, e.shape)) { worst = Infinity; continue; }
        const err = c.name.startsWith('argmax') ? (got.data.every((v, i) => v === e.data[i]) ? 0 : 1) : relErr(got.data, e.data);
        worst = Math.max(worst, err);
      }
      res.push({ name: c.name, ok: worst <= c.tol, err: worst });
    } catch (e) { res.push({ name: c.name, ok: false, err: String(e) }); }
  }
  for (const [m, outs] of [['seg2d_tiny', ['mask']], ['vit_tiny', ['logits', 'features']], ['unet3d_tiny', ['logits', 'probs']]] as const) {
    try {
      const graph = (await (await fetch(`/goldens/${m}.graph.json`)).json()) as OpGraph;
      const weights = parseSafetensors(new Uint8Array(await (await fetch(`/goldens/${graph.weights}`)).arrayBuffer()));
      const io = (await (await fetch(`/goldens/${m}.io.json`)).json()) as { inputs: Record<string, TJson>; expected: Record<string, TJson> };
      const gpu = await createGpuRunner(device, VisionModel.compile(graph, weights));
      const out = await gpu.run(Object.fromEntries(Object.entries(io.inputs).map(([k, v]) => [k, tensorOf(v)])));
      gpu.destroy();
      let worst = 0;
      for (const k of outs) worst = Math.max(worst, relErr(out[k].data, tensorOf(io.expected[k]).data));
      res.push({ name: `model:${m}`, ok: worst <= 1e-5, err: worst });
    } catch (e) { res.push({ name: `model:${m}`, ok: false, err: String(e) }); }
  }
  device.destroy();
  return res;
}

(globalThis as unknown as { visionSuite: unknown }).visionSuite = { probe, run };

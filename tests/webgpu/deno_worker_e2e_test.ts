// End-to-end wiring: spawn the REAL apps/worker/worker.ts against a minimal fake coordinator (WebSocket). Then drive
// the sealed `model` RPC: push_begin/push_chunk (graph.json + model.safetensors) → vision_caps → vision_load →
// vision_infer, and compare with the seg2d golden. Runs with or without a GPU. With a WebGPU adapter it also checks
// that the worker advertised the 'vision' capability and served the ops on WebGPU.
//
//   deno test --unstable-webgpu --allow-read --allow-net --allow-run --allow-env tests/webgpu/deno_worker_e2e_test.ts
import { type TJson, b64ToF32, relErr } from './golden_util.ts';

const url = (p: string) => new URL(p, import.meta.url);
const b64e = (u: Uint8Array) => { let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000)); return btoa(s); };
const b64d = (s: string) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const key = crypto.getRandomValues(new Uint8Array(32));
const aes = await crypto.subtle.importKey('raw', key, 'AES-GCM', false, ['encrypt', 'decrypt']);
async function seal(obj: unknown) { const iv = crypto.getRandomValues(new Uint8Array(12)); const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, aes, new TextEncoder().encode(JSON.stringify(obj))); return { iv: b64e(iv), ct: b64e(new Uint8Array(ct)) }; }
async function unseal(b: { iv: string; ct: string }) { return JSON.parse(new TextDecoder().decode(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: b64d(b.iv) }, aes, b64d(b.ct)))); }
function assert(c: unknown, msg: string): asserts c { if (!c) throw new Error(msg); }

Deno.test({ name: 'worker.ts e2e: vision_* over the sealed model RPC (fake coordinator)', sanitizeResources: false, sanitizeOps: false, fn: async () => {
  let registered: { node: { caps: string[]; label: string } } | null = null;
  let sock: WebSocket | null = null;
  const pending = new Map<string, (m: { ok: boolean; sealed?: { iv: string; ct: string }; error?: string }) => void>();
  const ready = Promise.withResolvers<void>();
  const server = Deno.serve({ port: 0, hostname: '127.0.0.1', onListen: () => {} }, (req) => {
    const { socket, response } = Deno.upgradeWebSocket(req);
    socket.onmessage = (ev) => {
      const m = JSON.parse(ev.data as string);
      if (m.t === 'register') { registered = m; sock = socket; socket.send(JSON.stringify({ t: 'welcome', tenantKeyB64: b64e(key), epoch: 0, duty: 1 })); ready.resolve(); }
      if (m.t === 'model_reply') pending.get(m.reqId)?.(m);
    };
    return response;
  });
  const port = (server.addr as Deno.NetAddr).port;
  const proc = new Deno.Command(Deno.execPath(), {
    args: ['run', '--unstable-webgpu', '--allow-net', '--allow-env', '--allow-sys', url('../../apps/worker/worker.ts').pathname, '--server', `ws://127.0.0.1:${port}/ws`, '--token', 't', '--name', 'e2e-vision'],
    stdout: 'piped', stderr: 'piped',
  }).spawn();
  let n = 0;
  const rpc = async (op: string, payload: Record<string, unknown>) => {
    const reqId = `r${n++}`;
    const got = new Promise<{ ok: boolean; sealed?: { iv: string; ct: string }; error?: string }>((r) => pending.set(reqId, r));
    sock!.send(JSON.stringify({ t: 'model', reqId, op, sealed: await seal(payload) }));
    const m = await got;
    if (!m.ok) throw new Error(`${op}: ${m.error}`);
    return await unseal(m.sealed!) as Record<string, unknown>;
  };
  try {
    await Promise.race([ready.promise, new Promise((_, rej) => setTimeout(() => rej(new Error('worker did not register')), 30_000))]);
    await new Promise((r) => setTimeout(r, 300)); // let the worker process the welcome (tenant key) first
    const caps = registered!.node.caps, gpu = registered!.node.label.startsWith('gpu');
    console.log(`[e2e] worker label=${registered!.node.label} caps=${caps.join(',')}`);
    assert(caps.includes('vision') === gpu, `vision cap must follow the GPU backend (gpu=${gpu}, caps=${caps})`);
    const vc = await rpc('vision_caps', {});
    assert(vc.webgpu === gpu && Array.isArray(vc.ops) && (vc.ops as string[]).includes('aten.convolution'), `caps ${JSON.stringify(vc).slice(0, 200)}`);
    // stage graph + weights through the existing download-free push path, under the model id
    const id = 'seg2d';
    await rpc('push_begin', { id });
    for (const [name, file] of [['graph.json', 'seg2d_tiny.graph.json'], ['model.safetensors', 'seg2d_tiny.safetensors']]) {
      const bytes = await Deno.readFile(url(`../goldens/wgsl/${file}`));
      for (let o = 0; o < bytes.length; o += 4096) await rpc('push_chunk', { id, name, data: b64e(bytes.subarray(o, o + 4096)) });
    }
    await rpc('push_end', { id });
    const ld = await rpc('vision_load', { id });
    assert(ld.ok === true && (ld.outputs as string[])[0] === 'mask', `load ${JSON.stringify(ld).slice(0, 200)}`);
    const io = JSON.parse(await Deno.readTextFile(url('../goldens/wgsl/seg2d_tiny.io.json'))) as { inputs: Record<string, TJson>; expected: Record<string, TJson> };
    const inf = await rpc('vision_infer', { id, inputs: { x: io.inputs.x } });
    const mask = (inf.outputs as Record<string, TJson>).mask;
    const err = relErr(b64ToF32(mask.b64), b64ToF32(io.expected.mask.b64));
    console.log(`[e2e] vision_infer on ${inf.backend}: rel err ${err.toExponential(2)} in ${(inf.ms as number).toFixed(1)} ms`);
    assert(err <= 1e-5, `seg2d via worker err ${err}`);
    assert(String(inf.backend).startsWith(gpu ? 'webgpu' : 'cpu-reference'), `backend ${inf.backend}`);
    assert((await rpc('vision_unload', { id })).ok === true, 'unload');
  } finally {
    try { proc.kill('SIGTERM'); } catch { /* */ }
    await proc.output().catch(() => {});
    await server.shutdown();
  }
} });

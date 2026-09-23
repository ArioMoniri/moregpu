import { describe, it, expect, vi } from 'vitest';
import { MoreGPUClient } from '../src/index.js';

function recorder(reply: (url: string, init?: RequestInit) => unknown = () => ({ ok: true })) {
  const calls: { url: string; method: string; body: any }[] = [];
  const f = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), method: init?.method ?? 'GET', body: init?.body ? JSON.parse(init.body as string) : undefined });
    return { ok: true, status: 200, json: async () => reply(String(url), init) } as Response;
  }) as unknown as typeof globalThis.fetch;
  return { c: new MoreGPUClient({ baseUrl: 'http://h:1', adminToken: 'T', fetch: f }), calls };
}

describe('training sessions + JEPA', () => {
  it('trainJepa builds a jepa_2p5d session with total_steps from target_samples', async () => {
    const { c, calls } = recorder();
    await c.trainJepa({ synthetic: { n: 8 }, manifest_len: 8, batch: 4, inner_steps: 2, lr: 1e-3, target_samples: 40, workers: ['a'] });
    expect(calls[0]!.url).toBe('http://h:1/train/sessions'); expect(calls[0]!.method).toBe('POST');
    expect(calls[0]!.body).toMatchObject({ task: 'jepa_2p5d', cfg: { model: 'tiny', patch: 16, synthetic: { n: 8 }, total_steps: 10 }, target_samples: 40, workers: ['a'] });
  });
  it('session lifecycle hits the documented routes', async () => {
    const { c, calls } = recorder();
    await c.trainSessionRound('s/1', 3); await c.trainSessionEval('s/1', [1], 'knn'); await c.trainSessionExport('s/1', 'onnx', '/x');
    await c.trainSessionTelemetry('s/1', 10); await c.trainSessionResume('s/1'); await c.trainSessionDelete('s/1'); await c.trainSessionState('s/1', 'bf16');
    expect(calls.map((x) => `${x.method} ${x.url.replace('http://h:1', '')}`)).toEqual([
      'POST /train/sessions/s%2F1/round', 'POST /train/sessions/s%2F1/eval', 'POST /train/sessions/s%2F1/export',
      'GET /train/sessions/s%2F1/telemetry?n=10', 'POST /train/sessions/resume', 'DELETE /train/sessions/s%2F1', 'GET /train/sessions/s%2F1/state?dtype=bf16']);
    expect(calls[0]!.body).toEqual({ rounds: 3 });
  });
  it('errors carry the server message', async () => {
    const f = vi.fn(async () => ({ ok: false, status: 409, json: async () => ({ error: 'already running' }) }) as Response) as unknown as typeof globalThis.fetch;
    await expect(new MoreGPUClient({ baseUrl: 'http://h:1', adminToken: 'T', fetch: f }).trainSessionRun('x')).rejects.toThrow(/409: already running/);
  });
});

describe('vision + data + net', () => {
  it('visionInfer encodes f32 and decodes the reply', async () => {
    const { c, calls } = recorder(() => ({ data: btoa(String.fromCharCode(...new Uint8Array(new Float32Array([1.5, -2]).buffer))), shape: [1, 2], worker: 'w' }));
    const r = await c.visionInfer('m', new Float32Array([1, 2, 3]), [1, 3]);
    expect(Array.from(r.output)).toEqual([1.5, -2]); expect(calls[0]!.body.shape).toEqual([1, 3]);
  });
  it('visionBatch tiles, dataPush sha256, net + caps', async () => {
    const { c, calls } = recorder();
    await c.visionBatch({ id: 'm', items: [{ ref: { uri: 'file://v.npy' }, out: 'o' }], split: 'tiles' });
    await c.dataPush('b', new TextEncoder().encode('abc'), '.txt');
    await c.net(50, 64); await c.workerCaps('w1'); await c.visionJob('j', true);
    expect(calls[0]!.body.split).toBe('tiles');
    expect(calls[1]!.body.sha256).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
    expect(calls[2]!.url).toBe('http://h:1/net?pings=50&sustained_mb=64');
    expect(calls[4]!.url).toBe('http://h:1/vision/jobs/j?results=1');
  });
  it('pushSafetensors: sha256 + .safetensors suffix, returns the pushed:// ref', async () => {
    const hdr = new TextEncoder().encode(JSON.stringify({ w: { dtype: 'F32', shape: [1], data_offsets: [0, 4] } }));
    const st = new Uint8Array(8 + hdr.length + 4);
    new DataView(st.buffer).setBigUint64(0, BigInt(hdr.length), true); st.set(hdr, 8); new DataView(st.buffer).setFloat32(8 + hdr.length, 1, true);
    const { c, calls } = recorder(() => ({ ok: true, uri: 'pushed://enc', results: [] }));
    const r = await c.pushSafetensors(st, 'enc', ['w1']);
    const sha = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', st)), (x) => x.toString(16).padStart(2, '0')).join('');
    expect(calls[0]!.url).toBe('http://h:1/data/push');
    expect(calls[0]!.body).toMatchObject({ id: 'enc', sha256: sha, suffix: '.safetensors', workers: ['w1'] });
    expect(r).toMatchObject({ ok: true, uri: 'pushed://enc', sha256: sha, size: st.length, ref: { path: 'pushed://enc', sha256: sha } });
    expect((await c.pushSafetensors(st)).uri).toBe(`pushed://st-${sha.slice(0, 16)}`);
  });
  it('pushSafetensors refuses pickles and junk before sending anything', async () => {
    const { c, calls } = recorder();
    for (const bad of [new Uint8Array([0x50, 0x4b, 3, 4, 0, 0, 0, 0, 0]), new Uint8Array([0x80, 2, 0x7d, 0x71, 0, 0x2e, 0, 0, 0]), new Uint8Array(0)]) {
      await expect(c.pushSafetensors(bad)).rejects.toThrow(/safetensors/);
    }
    expect(calls.length).toBe(0);
  });
  it('legacy LoRA DiLoCo parity methods', async () => {
    const { c, calls } = recorder();
    await c.dilocoLoad({ model: 'gpt2' }); await c.dilocoRound({ batches: { '*': [[1, 2]] } }); await c.trainStep({ input_ids: [1] });
    expect(calls.map((x) => x.url.replace('http://h:1', ''))).toEqual(['/train/diloco/load', '/train/diloco/round', '/train/step']);
  });
});

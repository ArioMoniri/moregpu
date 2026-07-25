import { describe, it, expect, vi } from 'vitest';
import { MoreGPUClient } from '../src/index.js';

function mockFetch(handler: (url: string, init?: RequestInit) => { status?: number; body: unknown }) {
  return vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    const { status = 200, body } = handler(String(url), init);
    return { ok: status >= 200 && status < 300, status, json: async () => body } as Response;
  }) as unknown as typeof globalThis.fetch;
}

const opts = (fetch: typeof globalThis.fetch) => ({ baseUrl: 'http://admin:8787/', adminToken: 'SECRET', fetch });

describe('MoreGPUClient — request building', () => {
  it('submit POSTs to /submit with kernel+size and a Bearer token', async () => {
    let seen: { url: string; init?: RequestInit } | undefined;
    const f = mockFetch((url, init) => { seen = { url, init }; return { body: { id: 'job-1', status: 'done', kernel: 'matmul', size: 512, verified: true } }; });
    const job = await new MoreGPUClient(opts(f)).submit('matmul', 512);
    expect(seen!.url).toBe('http://admin:8787/submit'); // trailing slash normalized
    expect(seen!.init!.method).toBe('POST');
    expect((seen!.init!.headers as Record<string, string>).authorization).toBe('Bearer SECRET');
    expect(JSON.parse(seen!.init!.body as string)).toEqual({ kernel: 'matmul', size: 512 });
    expect(job.verified).toBe(true);
  });

  it('health does NOT send an auth header (public endpoint)', async () => {
    let hadAuth = true;
    const f = mockFetch((_url, init) => { hadAuth = !!(init?.headers); return { body: { ok: true, fleet: 2, queue: 0 } }; });
    const h = await new MoreGPUClient(opts(f)).health();
    expect(hadAuth).toBe(false);
    expect(h.fleet).toBe(2);
  });

  it('gpu and workers send auth and hit the right paths', async () => {
    const urls: string[] = [];
    const f = mockFetch((url) => { urls.push(url); return { body: url.endsWith('/gpu') ? { slots: 3 } : [{ id: 'w1' }] }; });
    const c = new MoreGPUClient(opts(f));
    await c.gpu(); await c.workers();
    expect(urls).toEqual(['http://admin:8787/gpu', 'http://admin:8787/workers']);
  });

  it('submitBatch issues one submit per spec', async () => {
    let n = 0;
    const f = mockFetch(() => { n++; return { body: { id: `job-${n}`, status: 'done', kernel: 'relu', size: 1 } }; });
    const jobs = await new MoreGPUClient(opts(f)).submitBatch([
      { kernel: 'relu', size: 100 }, { kernel: 'matmul', size: 256 }, { kernel: 'scale', size: 50 },
    ]);
    expect(jobs).toHaveLength(3);
    expect(n).toBe(3);
  });

  it('treats HTTP 202 (queued, no workers) as a non-error job', async () => {
    const f = mockFetch(() => ({ status: 202, body: { id: 'job-9', status: 'queued', kernel: 'matmul', size: 512, note: 'queued — no workers yet' } }));
    const job = await new MoreGPUClient(opts(f)).submit('matmul', 512);
    expect(job.status).toBe('queued');
  });

  it('device() fetches the pool device descriptor', async () => {
    const f = mockFetch((url) => { expect(url).toBe('http://admin:8787/device'); return { body: { name: 'MoreGPU-Pool', kernels: ['matmul'], capabilities: { dataMode: true } } }; });
    const d = await new MoreGPUClient(opts(f)).device();
    expect(d.name).toBe('MoreGPU-Pool');
    expect(d.capabilities.dataMode).toBe(true);
  });

  it('submitAsync returns a handle and waitFor polls until done', async () => {
    let polls = 0;
    const f = mockFetch((url) => {
      if (url.includes('/submit')) return { status: 202, body: { id: 'job-1', status: 'queued', poll: '/jobs/job-1' } };
      polls++; return { body: { id: 'job-1', status: polls < 2 ? 'running' : 'done', kernel: 'matmul' } };
    });
    const c = new MoreGPUClient(opts(f));
    const handle = await c.submitAsync('matmul', 512);
    expect(handle.id).toBe('job-1');
    const job = await c.waitFor('job-1', { intervalMs: 1 });
    expect(job.status).toBe('done');
    expect(polls).toBeGreaterThanOrEqual(2);
  });

  it('throws on an auth failure', async () => {
    const f = mockFetch(() => ({ status: 401, body: { error: 'unauthorized' } }));
    await expect(new MoreGPUClient(opts(f)).workers()).rejects.toThrow(/401/);
  });

  it('run() sends base64 tensors + dims and decodes the returned output', async () => {
    const enc = (f: Float32Array) => { const u = new Uint8Array(f.buffer); let s = ''; for (const x of u) s += String.fromCharCode(x); return btoa(s); };
    const C = new Float32Array([1, 2, 3, 4]); // identity matmul returns A unchanged
    let body: Record<string, unknown> | undefined;
    const f = mockFetch((_url, init) => { body = JSON.parse(init!.body as string); return { body: { id: 'j', status: 'done', kernel: 'matmul', size: 2, verified: true, output: enc(C) } }; });
    const { job, output } = await new MoreGPUClient(opts(f)).run('matmul', { a: [1, 2, 3, 4], b: [1, 0, 0, 1], M: 2, N: 2, K: 2 });
    expect(job.verified).toBe(true);
    expect(Array.from(output)).toEqual([1, 2, 3, 4]);
    expect(body!.kernel).toBe('matmul');
    expect(typeof body!.a).toBe('string');
    expect(typeof body!.b).toBe('string');
    expect(body!.M).toBe(2);
  });

  it('matmul() convenience returns the decoded product', async () => {
    const enc = (f: Float32Array) => { const u = new Uint8Array(f.buffer); let s = ''; for (const x of u) s += String.fromCharCode(x); return btoa(s); };
    const f = mockFetch(() => ({ body: { id: 'j', status: 'done', kernel: 'matmul', output: enc(new Float32Array([58, 64, 139, 154])) } }));
    const out = await new MoreGPUClient(opts(f)).matmul([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], 2, 2, 3);
    expect(Array.from(out)).toEqual([58, 64, 139, 154]);
  });

  it('url-encodes job ids', async () => {
    let seen = '';
    const f = mockFetch((url) => { seen = url; return { body: { id: 'a/b', status: 'done', kernel: 'matmul', size: 1 } }; });
    await new MoreGPUClient(opts(f)).job('a/b');
    expect(seen).toBe('http://admin:8787/jobs/a%2Fb');
  });
});

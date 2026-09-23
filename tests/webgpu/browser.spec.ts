// Playwright: the WGSL vision kernels in a real browser (Chromium WebGPU). SKIPPED when the browser exposes no
// WebGPU adapter. Run:
//
//   npx playwright test -c tests/webgpu/playwright.config.ts
//
// Headless Chromium on a GPU-less Linux box can still run it through SwiftShader. The config passes
// --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader --enable-features=Vulkan --use-angle=swiftshader.
import { test, expect } from '@playwright/test';
import { createServer, type Server } from 'node:http';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join, normalize } from 'node:path';
import { build } from 'esbuild';

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));
const GOLD = here('../goldens/wgsl');
let server: Server;
let base = '';

test.beforeAll(async () => {
  const bundle = await build({ entryPoints: [here('./browser_entry.ts')], bundle: true, format: 'esm', write: false, platform: 'browser', target: 'es2022' });
  const js = bundle.outputFiles[0].text;
  server = createServer((req, res) => {
    const u = new URL(req.url ?? '/', 'http://x');
    if (u.pathname === '/') { res.setHeader('content-type', 'text/html'); res.end('<!doctype html><title>vision</title><script type="module" src="/suite.js"></script>'); return; }
    if (u.pathname === '/suite.js') { res.setHeader('content-type', 'text/javascript'); res.end(js); return; }
    if (u.pathname.startsWith('/goldens/')) {
      const f = normalize(join(GOLD, u.pathname.slice('/goldens/'.length)));
      if (f.startsWith(GOLD) && existsSync(f)) { res.end(readFileSync(f)); return; }
    }
    res.statusCode = 404; res.end();
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', () => r()));
  const addr = server.address();
  base = `http://127.0.0.1:${typeof addr === 'object' && addr ? addr.port : 0}`;
});
test.afterAll(async () => { await new Promise((r) => server.close(r)); });

test('WGSL vision kernels + model parity in the browser (skipped without WebGPU)', async ({ page }) => {
  test.setTimeout(600_000);
  await page.goto(`${base}/`);
  await page.waitForFunction(() => 'visionSuite' in globalThis);
  const probe = await page.evaluate(() => (globalThis as unknown as { visionSuite: { probe(): Promise<{ adapter: boolean; info: string; f16: boolean }> } }).visionSuite.probe());
  test.skip(!probe.adapter, `no WebGPU adapter in this browser (${probe.info})`);
  console.log(`[browser] WebGPU adapter ${probe.info} f16=${probe.f16}`);
  const res = await page.evaluate(() => (globalThis as unknown as { visionSuite: { run(): Promise<{ name: string; ok: boolean; err: number | string }[]> } }).visionSuite.run());
  const bad = res.filter((r) => !r.ok);
  console.log(`[browser] ${res.length - bad.length}/${res.length} passed`);
  expect(bad, JSON.stringify(bad, null, 1)).toEqual([]);
  expect(res.length).toBeGreaterThanOrEqual(90 + 13 + 3); // kernel goldens + kernel compiles + 3 models
});

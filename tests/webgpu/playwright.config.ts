// Playwright config for the WebGPU browser test only (tests/webgpu/browser.spec.ts).
// Flags: WebGPU is behind --enable-unsafe-webgpu on Linux, and SwiftShader provides a software adapter when there is no GPU.
// MOREGPU_WEBGPU=0 launches Chromium without the flags, which exercises the skip path.
import { defineConfig } from '@playwright/test';

const WEBGPU_ARGS = ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan', '--use-angle=swiftshader'];

export default defineConfig({
  testDir: '.',
  testMatch: /browser\.spec\.ts$/,
  timeout: 600_000,
  reporter: 'list',
  use: {
    browserName: 'chromium',
    headless: true,
    launchOptions: {
      args: process.env.MOREGPU_WEBGPU === '0' ? [] : WEBGPU_ARGS,
    },
  },
});

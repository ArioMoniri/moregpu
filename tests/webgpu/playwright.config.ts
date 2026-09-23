// Playwright config for the WebGPU browser test only (tests/webgpu/browser.spec.ts).
// Flags: WebGPU is behind --enable-unsafe-webgpu on Linux, and SwiftShader provides a software adapter when there is no GPU.
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  testMatch: /browser\.spec\.ts$/,
  timeout: 600_000,
  reporter: 'list',
  use: {
    browserName: 'chromium',
    headless: true,
    launchOptions: {
      args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan', '--use-angle=swiftshader'],
    },
  },
});

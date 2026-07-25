import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

const pkg = (p: string) => fileURLToPath(new URL(p, import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      '@moregpu/protocol': pkg('./packages/protocol/src/index.ts'),
      '@moregpu/crypto': pkg('./packages/crypto/src/index.ts'),
      '@moregpu/scheduler': pkg('./packages/scheduler/src/index.ts'),
      '@moregpu/runtime': pkg('./packages/runtime/src/index.ts'),
      '@moregpu/coordinator': pkg('./packages/coordinator/src/index.ts'),
      '@moregpu/transport': pkg('./packages/transport/src/index.ts'),
      '@moregpu/gpu': pkg('./packages/gpu/src/index.ts'),
      '@moregpu/integrity': pkg('./packages/integrity/src/index.ts'),
    },
  },
  test: {
    include: ['packages/**/*.test.ts', 'apps/**/*.test.ts', 'tests/**/*.test.ts'],
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json-summary', 'html'],
      include: ['packages/**/src/**/*.ts', 'apps/**/src/**/*.ts'],
      exclude: ['**/*.test.ts', '**/dist/**', '**/*.d.ts'],
      thresholds: {
        lines: 80,
        branches: 70,
        functions: 80,
        statements: 80,
      },
    },
  },
});

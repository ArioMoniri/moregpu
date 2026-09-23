import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { lrAt, progress } from './schedule.ts';

const G = JSON.parse(readFileSync(new URL('../../../tests/goldens/lr_schedule.json', import.meta.url), 'utf8'));
describe('lr schedule', () => {
  it('matches the golden', () => {
    for (const c of G) expect(lrAt(c.cfg, progress(c.cfg, c.seen, c.round_samples, c.round))).toBeCloseTo(c.lr, 12);
  });
  it('rejects unknown kinds', () => expect(() => lrAt({ lr: 1, lr_schedule: { kind: 'step' as 'cosine' } }, 0.3)).toThrow());
});

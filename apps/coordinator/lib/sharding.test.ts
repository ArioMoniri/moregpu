import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { permutation, SampleStream, split, allocate } from './sharding.ts';

const G = JSON.parse(readFileSync(new URL('../../../tests/goldens/sharding.json', import.meta.url), 'utf8'));

describe('sharding (identical to moregpu_worker.train.sharding)', () => {
  it('matches the SplitMix64 golden', () => expect(permutation(G.n, G.seed, G.epoch)).toEqual(G.perm));
  it('stream crosses epochs and resumes from state', () => {
    const s = new SampleStream(10, 3); const a = [...s.take(7), ...s.take(7)];
    expect(a.slice(0, 10)).toEqual(permutation(10, 3, 0));
    expect(a.slice(10)).toEqual(permutation(10, 3, 1).slice(0, 4));
    const s3 = new SampleStream(10, 3); s3.take(7);
    expect(SampleStream.fromState(s3.state()).take(7)).toEqual(a.slice(7));
  });
  it('split + allocate', () => {
    expect(split([0, 1, 2, 3, 4, 5], [1, 2, 3])).toEqual([[0], [1, 2], [3, 4, 5]]);
    expect(allocate(8, 3, [1, 2, 1], 'fixed')).toEqual([8, 8, 8]);
    const p = allocate(8, 3, [1, 2, 1], 'proportional');
    expect(p.reduce((a, b) => a + b)).toBe(24); expect(p[1]!).toBeGreaterThan(p[0]!);
    expect(allocate(8, 3, [1, 1, 1], 'fixed', 5).reduce((a, b) => a + b)).toBe(5);
  });
});

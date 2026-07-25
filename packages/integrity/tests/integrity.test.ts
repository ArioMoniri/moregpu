import { describe, it, expect } from 'vitest';
import {
  reproducibilityRings,
  exactQuorum,
  canaryVerdict,
  ReputationBook,
  type ReplicaResult,
} from '../src/index.js';

describe('reproducibilityRings — BOINC-style homogeneous redundancy', () => {
  it('groups nodes into rings by numeric class (vendor+driver)', () => {
    const rings = reproducibilityRings([
      { nodeId: 'a', vendor: 'nvidia', driverClass: '550' },
      { nodeId: 'b', vendor: 'nvidia', driverClass: '550' },
      { nodeId: 'c', vendor: 'amd', driverClass: 'rocm6' },
    ]);
    expect(rings.get('nvidia:550')?.sort()).toEqual(['a', 'b']);
    expect(rings.get('amd:rocm6')).toEqual(['c']);
  });

  it('never mixes vendors in one ring (bit-exact quorum only within a ring)', () => {
    const rings = reproducibilityRings([
      { nodeId: 'a', vendor: 'nvidia', driverClass: '550' },
      { nodeId: 'b', vendor: 'intel', driverClass: 'arc' },
    ]);
    expect(rings.size).toBe(2);
  });
});

describe('exactQuorum — bit-identical integer majority', () => {
  const r = (nodeId: string, checksum: string): ReplicaResult => ({ nodeId, checksum });

  it('agrees when all replicas are bit-identical', () => {
    const v = exactQuorum([r('a', 'H1'), r('b', 'H1'), r('c', 'H1')]);
    expect(v.agreed).toBe(true);
    expect(v.value).toBe('H1');
    expect(v.dissenters).toEqual([]);
  });

  it('takes the majority and names dissenters', () => {
    const v = exactQuorum([r('a', 'H1'), r('b', 'H1'), r('c', 'H2')]);
    expect(v.agreed).toBe(true);
    expect(v.value).toBe('H1');
    expect(v.dissenters).toEqual(['c']);
  });

  it('does not agree without a strict majority', () => {
    const v = exactQuorum([r('a', 'H1'), r('b', 'H2'), r('c', 'H3')]);
    expect(v.agreed).toBe(false);
    expect(v.value).toBeUndefined();
  });

  it('requires at least two replicas to claim agreement', () => {
    const v = exactQuorum([r('a', 'H1')]);
    expect(v.agreed).toBe(false);
  });
});

describe('canaryVerdict — known-answer trap for the fp-aggregate class', () => {
  it('passes results within benign cross-vendor drift', () => {
    const v = canaryVerdict([1.0, 2.0, 3.0], [1.0001, 1.9998, 3.0002], 1e-2);
    expect(v.pass).toBe(true);
  });

  it('flags tampering above the drift tolerance', () => {
    const v = canaryVerdict([1.0, 2.0, 3.0], [1.0, 2.0, 9.0], 1e-2);
    expect(v.pass).toBe(false);
    expect(v.relativeL2).toBeGreaterThan(1e-2);
  });
});

describe('ReputationBook — adaptive replication toward proven nodes', () => {
  it('starts unproven nodes at a high replication factor', () => {
    const book = new ReputationBook();
    expect(book.replicationFactor('fresh')).toBeGreaterThanOrEqual(2);
  });

  it('drives a consistently-correct node toward ~1.1x', () => {
    const book = new ReputationBook();
    for (let i = 0; i < 50; i++) book.record('good', true);
    expect(book.replicationFactor('good')).toBeLessThanOrEqual(1.3);
  });

  it('penalizes a node that returns a wrong result', () => {
    const book = new ReputationBook();
    for (let i = 0; i < 50; i++) book.record('mixed', true);
    const before = book.reputation('mixed');
    book.record('mixed', false);
    expect(book.reputation('mixed')).toBeLessThan(before);
  });

  it('keeps reputation within [0,1]', () => {
    const book = new ReputationBook();
    for (let i = 0; i < 100; i++) book.record('x', false);
    expect(book.reputation('x')).toBeGreaterThanOrEqual(0);
    for (let i = 0; i < 100; i++) book.record('x', true);
    expect(book.reputation('x')).toBeLessThanOrEqual(1);
  });
});

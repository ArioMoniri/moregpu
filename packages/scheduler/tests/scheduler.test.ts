import { describe, it, expect } from 'vitest';
import type { NodeCapability, NodeLoad, ShardResult } from '@moregpu/protocol';
import {
  shardJob,
  rankNodes,
  assignShards,
  poolResults,
  effectiveCapacity,
} from '../src/index.js';

function cap(id: string, over: Partial<NodeCapability> = {}): NodeCapability {
  return {
    nodeId: id,
    backend: 'webgpu',
    vendor: 'nvidia',
    os: 'linux',
    vramMB: 8192,
    logicalCores: 16,
    maxComputeInvocations: 65535,
    supportsF16: true,
    ...over,
  };
}

describe('shardJob', () => {
  it('splits a shardable job into contiguous, non-overlapping unit ranges covering everything', () => {
    const shards = shardJob({ jobId: 'j1', totalUnits: 100, shardable: true }, 4);
    expect(shards).toHaveLength(4);
    expect(shards[0].unitStart).toBe(0);
    expect(shards[shards.length - 1].unitEnd).toBe(100);
    for (let i = 1; i < shards.length; i++) {
      expect(shards[i].unitStart).toBe(shards[i - 1].unitEnd); // contiguous, no gaps/overlap
    }
    const covered = shards.reduce((n, s) => n + (s.unitEnd - s.unitStart), 0);
    expect(covered).toBe(100);
  });

  it('returns a single whole shard for a non-shardable job', () => {
    const shards = shardJob({ jobId: 'j2', totalUnits: 100, shardable: false }, 8);
    expect(shards).toHaveLength(1);
    expect(shards[0].unitStart).toBe(0);
    expect(shards[0].unitEnd).toBe(100);
  });

  it('never creates more shards than units', () => {
    const shards = shardJob({ jobId: 'j3', totalUnits: 3, shardable: true }, 16);
    expect(shards.length).toBeLessThanOrEqual(3);
  });

  it('distributes remainder units fairly (max-min diff <= 1)', () => {
    const shards = shardJob({ jobId: 'j4', totalUnits: 10, shardable: true }, 3);
    const sizes = shards.map((s) => s.unitEnd - s.unitStart);
    expect(Math.max(...sizes) - Math.min(...sizes)).toBeLessThanOrEqual(1);
  });
});

describe('rankNodes', () => {
  it('orders GPU nodes ahead of CPU nodes', () => {
    const ranked = rankNodes([
      cap('cpu', { backend: 'wasm-cpu', vramMB: 0, maxComputeInvocations: 0, supportsF16: false }),
      cap('gpu'),
    ]);
    expect(ranked[0].nodeId).toBe('gpu');
  });
});

describe('effectiveCapacity — throttle awareness', () => {
  it('scales capacity by the available fraction the node reports', () => {
    const c = cap('n');
    const busy: NodeLoad = { interactiveLoad: 0.9, availableFraction: 0.1 };
    const idle: NodeLoad = { interactiveLoad: 0.0, availableFraction: 1.0 };
    expect(effectiveCapacity(c, busy)).toBeLessThan(effectiveCapacity(c, idle));
  });

  it('is zero when the node has no headroom (protect the interactive user)', () => {
    const c = cap('n');
    expect(effectiveCapacity(c, { interactiveLoad: 1, availableFraction: 0 })).toBe(0);
  });
});

describe('assignShards', () => {
  it('assigns every shard to some available node', () => {
    const shards = shardJob({ jobId: 'j', totalUnits: 100, shardable: true }, 4);
    const nodes = [
      { cap: cap('a'), load: { interactiveLoad: 0.1, availableFraction: 0.9 } },
      { cap: cap('b'), load: { interactiveLoad: 0.2, availableFraction: 0.8 } },
    ];
    const plan = assignShards(shards, nodes);
    expect(plan).toHaveLength(4);
    for (const a of plan) {
      expect(['a', 'b']).toContain(a.nodeId);
    }
  });

  it('skips nodes with zero headroom entirely', () => {
    const shards = shardJob({ jobId: 'j', totalUnits: 10, shardable: true }, 2);
    const nodes = [
      { cap: cap('busy'), load: { interactiveLoad: 1, availableFraction: 0 } },
      { cap: cap('free'), load: { interactiveLoad: 0, availableFraction: 1 } },
    ];
    const plan = assignShards(shards, nodes);
    expect(plan.every((a) => a.nodeId === 'free')).toBe(true);
  });

  it('throws if no node has any headroom', () => {
    const shards = shardJob({ jobId: 'j', totalUnits: 10, shardable: true }, 2);
    const nodes = [{ cap: cap('busy'), load: { interactiveLoad: 1, availableFraction: 0 } }];
    expect(() => assignShards(shards, nodes)).toThrow(/no available/i);
  });
});

describe('poolResults', () => {
  const mk = (id: string, output: number[]): ShardResult & { output: number[] } => ({
    shardId: id,
    jobId: 'j',
    ok: true,
    output,
  });

  it('concatenates in shard order regardless of arrival order', () => {
    const out = poolResults(
      [
        { ...mk('s2', [3, 4]), index: 1 },
        { ...mk('s1', [1, 2]), index: 0 },
      ],
      'concat',
    );
    expect(out).toEqual([1, 2, 3, 4]);
  });

  it('sums elementwise', () => {
    const out = poolResults(
      [
        { ...mk('s1', [1, 2, 3]), index: 0 },
        { ...mk('s2', [10, 20, 30]), index: 1 },
      ],
      'sum',
    );
    expect(out).toEqual([11, 22, 33]);
  });

  it('averages elementwise', () => {
    const out = poolResults(
      [
        { ...mk('s1', [2, 4]), index: 0 },
        { ...mk('s2', [4, 8]), index: 1 },
      ],
      'mean',
    );
    expect(out).toEqual([3, 6]);
  });

  it('throws if any shard failed (no silent partial pooling)', () => {
    expect(() =>
      poolResults([{ shardId: 's1', jobId: 'j', ok: false, index: 0, output: [] }], 'sum'),
    ).toThrow(/failed/i);
  });
});

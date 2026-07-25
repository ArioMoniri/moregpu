import { describe, it, expect } from 'vitest';
import type { NodeCapability, NodeLoad } from '@moregpu/protocol';
import { Orchestrator } from '../src/index.js';

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
const idle: NodeLoad = { interactiveLoad: 0, availableFraction: 1 };

describe('Orchestrator — node registry', () => {
  it('registers nodes and lists them', () => {
    const o = new Orchestrator();
    o.register(cap('n1'), idle);
    o.register(cap('n2'), idle);
    expect(o.nodeCount()).toBe(2);
    expect(o.nodes().map((n) => n.cap.nodeId).sort()).toEqual(['n1', 'n2']);
  });

  it('updates load on heartbeat without duplicating the node', () => {
    const o = new Orchestrator();
    o.register(cap('n1'), idle);
    o.heartbeat('n1', { interactiveLoad: 0.5, availableFraction: 0.5 });
    expect(o.nodeCount()).toBe(1);
    expect(o.nodes()[0].load.availableFraction).toBe(0.5);
  });

  it('evicts nodes that miss the liveness deadline', () => {
    const o = new Orchestrator({ livenessMs: 1000 });
    o.register(cap('n1'), idle, 0);
    o.register(cap('n2'), idle, 0);
    o.heartbeat('n2', idle, 1500);
    o.evictStale(2000); // n1 last seen at 0, deadline 1000 → evicted; n2 seen at 1500 → kept
    expect(o.nodes().map((n) => n.cap.nodeId)).toEqual(['n2']);
  });
});

describe('Orchestrator — job lifecycle', () => {
  it('rejects a job when no nodes are available', () => {
    const o = new Orchestrator();
    expect(() => o.submitJob({ jobId: 'j', tenantId: 't', totalUnits: 10, shardable: true, reduce: 'concat' })).toThrow(
      /no available/i,
    );
  });

  it('shards a job across available nodes and tracks it as pending', () => {
    const o = new Orchestrator();
    o.register(cap('n1'), idle);
    o.register(cap('n2'), idle);
    const plan = o.submitJob({ jobId: 'j', tenantId: 't', totalUnits: 100, shardable: true, reduce: 'sum' }, 4);
    expect(plan.assignments).toHaveLength(4);
    expect(o.jobStatus('j')).toBe('pending');
  });

  it('completes a job and pools results once every shard reports', () => {
    const o = new Orchestrator();
    o.register(cap('n1'), idle);
    const plan = o.submitJob({ jobId: 'j', tenantId: 't', totalUnits: 4, shardable: true, reduce: 'concat' }, 2);
    for (const a of plan.assignments) {
      const idx = Number(a.shardId.split('#')[1]);
      o.reportResult({ shardId: a.shardId, jobId: 'j', ok: true, index: idx, output: [idx * 10, idx * 10 + 1] });
    }
    expect(o.jobStatus('j')).toBe('complete');
    expect(o.pooledOutput('j')).toEqual([0, 1, 10, 11]);
  });

  it('marks a job failed if any shard reports failure', () => {
    const o = new Orchestrator();
    o.register(cap('n1'), idle);
    const plan = o.submitJob({ jobId: 'j', tenantId: 't', totalUnits: 4, shardable: true, reduce: 'concat' }, 2);
    o.reportResult({ shardId: plan.assignments[0].shardId, jobId: 'j', ok: false, index: 0, output: [], error: 'boom' });
    expect(o.jobStatus('j')).toBe('failed');
  });

  it('isolates tenants: a job records its tenant and rejects a cross-tenant result', () => {
    const o = new Orchestrator();
    o.register(cap('n1'), idle);
    const plan = o.submitJob({ jobId: 'j', tenantId: 'tenant-A', totalUnits: 2, shardable: false, reduce: 'none' }, 1);
    expect(() =>
      o.reportResult(
        { shardId: plan.assignments[0].shardId, jobId: 'j', ok: true, index: 0, output: [1, 2] },
        'tenant-B',
      ),
    ).toThrow(/tenant/i);
  });
});

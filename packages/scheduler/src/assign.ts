import type { NodeCapability, NodeLoad } from '@moregpu/protocol';
import { effectiveCapacity, type ShardPlan } from './shard.js';

export interface NodeState {
  cap: NodeCapability;
  load: NodeLoad;
}

export interface Assignment {
  shardId: string;
  jobId: string;
  nodeId: string;
}

/**
 * Assign shards to nodes weighted by current effective capacity (capability × available headroom).
 * Nodes with zero headroom are excluded so we never steal cycles from an active interactive user.
 * Uses a greedy least-loaded fill so bigger/idler nodes take proportionally more shards.
 */
export function assignShards(shards: ShardPlan[], nodes: NodeState[]): Assignment[] {
  const available = nodes
    .map((n) => ({ n, cap: effectiveCapacity(n.cap, n.load) }))
    .filter((x) => x.cap > 0);

  if (available.length === 0) {
    throw new Error('scheduler: no available node has headroom for assignment');
  }

  // Running "cost" per node; each assigned shard adds 1/capacity so higher-capacity nodes fill slower.
  const load = new Map<string, number>(available.map((x) => [x.n.cap.nodeId, 0]));
  const assignments: Assignment[] = [];

  for (const shard of shards) {
    let best = available[0]!;
    let bestCost = Infinity;
    for (const x of available) {
      const projected = (load.get(x.n.cap.nodeId) ?? 0) + 1 / x.cap;
      if (projected < bestCost) {
        bestCost = projected;
        best = x;
      }
    }
    load.set(best.n.cap.nodeId, bestCost);
    assignments.push({ shardId: shard.shardId, jobId: shard.jobId, nodeId: best.n.cap.nodeId });
  }
  return assignments;
}

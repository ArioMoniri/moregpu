import type { NodeCapability, NodeLoad } from '@moregpu/protocol';
import { capabilityScore } from '@moregpu/protocol';

/** Minimal job description the sharder needs. */
export interface ShardableJob {
  jobId: string;
  totalUnits: number;
  shardable: boolean;
}

/** A planned unit range for one shard (sealing/assignment happen later). */
export interface ShardPlan {
  shardId: string;
  jobId: string;
  index: number;
  unitStart: number;
  unitEnd: number;
}

/**
 * Split a job into contiguous, non-overlapping unit ranges that exactly cover [0, totalUnits).
 * Non-shardable jobs become a single whole shard. Remainder units are spread so the largest and
 * smallest shard differ by at most one unit (fair balancing).
 */
export function shardJob(job: ShardableJob, desiredShards: number): ShardPlan[] {
  if (job.totalUnits <= 0) return [];
  if (!job.shardable) {
    return [{ shardId: `${job.jobId}#0`, jobId: job.jobId, index: 0, unitStart: 0, unitEnd: job.totalUnits }];
  }
  const n = Math.max(1, Math.min(desiredShards, job.totalUnits));
  const base = Math.floor(job.totalUnits / n);
  const remainder = job.totalUnits % n;
  const shards: ShardPlan[] = [];
  let cursor = 0;
  for (let i = 0; i < n; i++) {
    const size = base + (i < remainder ? 1 : 0);
    shards.push({ shardId: `${job.jobId}#${i}`, jobId: job.jobId, index: i, unitStart: cursor, unitEnd: cursor + size });
    cursor += size;
  }
  return shards;
}

/** Rank nodes best-first by intrinsic capability (GPU + VRAM + cores). */
export function rankNodes(caps: NodeCapability[]): NodeCapability[] {
  return [...caps].sort((a, b) => capabilityScore(b) - capabilityScore(a));
}

/**
 * Capacity a node can offer *right now* after adaptive throttle: intrinsic score scaled by the
 * fraction the node reports as available. Zero headroom → zero capacity, protecting the local user.
 */
export function effectiveCapacity(cap: NodeCapability, load: NodeLoad): number {
  const frac = Math.max(0, Math.min(1, load.availableFraction));
  return capabilityScore(cap) * frac;
}

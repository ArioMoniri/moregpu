import type { NodeCapability, NodeLoad, ReduceOp } from '@moregpu/protocol';
import { shardJob, assignShards, poolResults, type Assignment, type PoolableResult } from '@moregpu/scheduler';

export interface NodeRecord {
  cap: NodeCapability;
  load: NodeLoad;
  lastSeen: number;
}

export interface JobSpec {
  jobId: string;
  tenantId: string;
  totalUnits: number;
  shardable: boolean;
  reduce: ReduceOp;
}

export type JobStatus = 'pending' | 'complete' | 'failed';

export interface JobPlan {
  jobId: string;
  assignments: Assignment[];
}

export interface OrchestratorConfig {
  /** A node not heard from within this window is eligible for eviction. */
  livenessMs: number;
}

interface JobRecord {
  spec: JobSpec;
  expected: Set<string>;
  results: Map<string, PoolableResult>;
  status: JobStatus;
  pooled?: number[];
}

/**
 * In-memory control-plane brain. Transport (WebSocket/gRPC) and persistence are layered on top by
 * the coordinator app; this class holds the pure orchestration logic so it is fully unit-testable.
 */
export class Orchestrator {
  private readonly registry = new Map<string, NodeRecord>();
  private readonly jobs = new Map<string, JobRecord>();
  private readonly cfg: OrchestratorConfig;

  constructor(cfg: Partial<OrchestratorConfig> = {}) {
    this.cfg = { livenessMs: cfg.livenessMs ?? 15_000 };
  }

  register(cap: NodeCapability, load: NodeLoad, now: number = Date.now()): void {
    this.registry.set(cap.nodeId, { cap, load, lastSeen: now });
  }

  heartbeat(nodeId: string, load: NodeLoad, now: number = Date.now()): void {
    const rec = this.registry.get(nodeId);
    if (!rec) return;
    rec.load = load;
    rec.lastSeen = now;
  }

  /** Remove nodes whose last heartbeat is older than the liveness window. */
  evictStale(now: number = Date.now()): void {
    for (const [id, rec] of this.registry) {
      if (now - rec.lastSeen >= this.cfg.livenessMs) this.registry.delete(id);
    }
  }

  nodes(): NodeRecord[] {
    return [...this.registry.values()];
  }

  nodeCount(): number {
    return this.registry.size;
  }

  /** Shard a job across the current fleet and record it as pending. Throws if nothing can run it. */
  submitJob(spec: JobSpec, desiredShards = 8): JobPlan {
    const shards = shardJob(
      { jobId: spec.jobId, totalUnits: spec.totalUnits, shardable: spec.shardable },
      desiredShards,
    );
    const nodeStates = this.nodes().map((r) => ({ cap: r.cap, load: r.load }));
    const assignments = assignShards(shards, nodeStates); // throws 'no available…' when fleet is empty/busy

    this.jobs.set(spec.jobId, {
      spec,
      expected: new Set(assignments.map((a) => a.shardId)),
      results: new Map(),
      status: 'pending',
    });
    return { jobId: spec.jobId, assignments };
  }

  /** Record a shard result. `assertTenant` enforces tenant isolation when provided. */
  reportResult(result: PoolableResult, assertTenant?: string): void {
    const job = this.jobs.get(result.jobId);
    if (!job) throw new Error(`coordinator: unknown job ${result.jobId}`);
    if (assertTenant !== undefined && assertTenant !== job.spec.tenantId) {
      throw new Error(`coordinator: tenant mismatch — ${assertTenant} may not report on ${job.spec.tenantId}'s job`);
    }
    job.results.set(result.shardId, result);

    if (!result.ok) {
      job.status = 'failed';
      return;
    }
    const allIn = [...job.expected].every((id) => job.results.get(id)?.ok);
    if (allIn) {
      job.pooled = poolResults([...job.results.values()], job.spec.reduce);
      job.status = 'complete';
    }
  }

  jobStatus(jobId: string): JobStatus | undefined {
    return this.jobs.get(jobId)?.status;
  }

  pooledOutput(jobId: string): number[] | undefined {
    return this.jobs.get(jobId)?.pooled;
  }
}

import type { ReduceOp } from '@moregpu/protocol';

/** A completed shard's numeric output plus its position, for ordered pooling. */
export interface PoolableResult {
  shardId: string;
  jobId: string;
  ok: boolean;
  index: number;
  output: number[];
}

/**
 * Pool shard outputs into one result using the job's reduce op. Fails loudly if any shard failed —
 * we never silently pool a partial answer. `concat` reassembles in shard index order regardless of
 * network arrival order; `sum`/`mean` reduce elementwise across equal-length outputs.
 */
export function poolResults(results: PoolableResult[], reduce: ReduceOp): number[] {
  const failed = results.filter((r) => !r.ok);
  if (failed.length > 0) {
    throw new Error(`scheduler: cannot pool — ${failed.length} shard(s) failed: ${failed.map((f) => f.shardId).join(', ')}`);
  }
  const ordered = [...results].sort((a, b) => a.index - b.index);

  switch (reduce) {
    case 'concat':
    case 'none':
      return ordered.flatMap((r) => r.output);
    case 'sum':
      return elementwise(ordered, (acc, v) => acc + v);
    case 'mean': {
      const summed = elementwise(ordered, (acc, v) => acc + v);
      return summed.map((v) => v / ordered.length);
    }
    default: {
      const _exhaustive: never = reduce;
      throw new Error(`scheduler: unknown reduce op ${_exhaustive as string}`);
    }
  }
}

function elementwise(results: PoolableResult[], fn: (acc: number, v: number) => number): number[] {
  if (results.length === 0) return [];
  const width = results[0]!.output.length;
  for (const r of results) {
    if (r.output.length !== width) {
      throw new Error('scheduler: elementwise reduce requires equal-length shard outputs');
    }
  }
  const out = new Array<number>(width).fill(0);
  for (const r of results) {
    for (let i = 0; i < width; i++) {
      out[i] = fn(out[i]!, r.output[i]!);
    }
  }
  return out;
}

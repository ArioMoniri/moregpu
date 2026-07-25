/** One replica's result, identified by a content checksum (e.g. blake3) over its output bytes. */
export interface ReplicaResult {
  nodeId: string;
  checksum: string;
}

export interface QuorumVerdict {
  agreed: boolean;
  /** The agreed checksum when a strict majority exists. */
  value?: string;
  /** Nodes whose checksum differed from the agreed value. */
  dissenters: string[];
}

/**
 * Bit-identical integer quorum: require a strict majority of replicas to report the SAME checksum,
 * and at least two replicas. Sound only for integer/fixed-point work run inside one reproducibility
 * ring (see rings.ts) — never call this on cross-vendor floating-point results.
 */
export function exactQuorum(results: ReplicaResult[]): QuorumVerdict {
  if (results.length < 2) return { agreed: false, dissenters: [] };

  const counts = new Map<string, number>();
  for (const r of results) counts.set(r.checksum, (counts.get(r.checksum) ?? 0) + 1);

  let best: string | undefined;
  let bestCount = 0;
  for (const [checksum, count] of counts) {
    if (count > bestCount) {
      best = checksum;
      bestCount = count;
    }
  }

  const strictMajority = bestCount > results.length / 2;
  if (!strictMajority || best === undefined) return { agreed: false, dissenters: [] };

  const dissenters = results.filter((r) => r.checksum !== best).map((r) => r.nodeId);
  return { agreed: true, value: best, dissenters };
}

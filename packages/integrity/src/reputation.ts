/**
 * Reputation-weighted adaptive replication.
 *
 * PRIOR ART (per council verifier): BOINC adaptive replication — a host that accumulates a track
 * record of validated results earns a lower replication factor, converging toward ~1.1x, while
 * unproven hosts get heavier redundancy. Not a novel mechanism; used as-is.
 */

const ALPHA = 0.1; // EMA responsiveness of the reputation estimate.
const MIN_FACTOR = 1.1;
const MAX_EXTRA = 2.9; // unproven node → up to 1.1 + 2.9 = 4.0x replication.

export class ReputationBook {
  private readonly reps = new Map<string, number>();

  /** Reputation in [0,1]; an unseen node is unproven (0). */
  reputation(nodeId: string): number {
    return this.reps.get(nodeId) ?? 0;
  }

  /** Record whether a node's result matched the quorum/canary. Moves reputation via EMA in [0,1]. */
  record(nodeId: string, correct: boolean): void {
    const prev = this.reps.get(nodeId) ?? 0;
    const target = correct ? 1 : 0;
    const next = prev + ALPHA * (target - prev);
    this.reps.set(nodeId, Math.max(0, Math.min(1, next)));
  }

  /** Higher reputation → fewer replicas. Proven nodes approach 1.1x; unproven start ~4x. */
  replicationFactor(nodeId: string): number {
    const rep = this.reputation(nodeId);
    return MIN_FACTOR + (1 - rep) * MAX_EXTRA;
  }
}

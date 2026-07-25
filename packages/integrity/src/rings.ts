/**
 * Reproducibility rings.
 *
 * PRIOR ART (per council verifier): this is BOINC "Homogeneous Redundancy" (Anderson, ~2007) —
 * routing a work-unit's replicas only to hosts of the same numeric class so their floating-point
 * results are bit-identical and an exact-match quorum is sound again on heterogeneous hardware.
 * We apply it verbatim; the contribution here is wiring it to the admission determinism-class axis,
 * not the mechanism itself.
 */

export interface RingNode {
  nodeId: string;
  vendor: string;
  /** Driver/toolchain generation that fixes rounding behavior (e.g. NVIDIA '550', ROCm '6'). */
  driverClass: string;
}

export type NumericClass = string; // `${vendor}:${driverClass}`

export function numericClassOf(node: RingNode): NumericClass {
  return `${node.vendor}:${node.driverClass}`;
}

/** Partition nodes into rings keyed by numeric class. Bit-exact quorum is only valid WITHIN a ring. */
export function reproducibilityRings(nodes: RingNode[]): Map<NumericClass, string[]> {
  const rings = new Map<NumericClass, string[]>();
  for (const node of nodes) {
    const key = numericClassOf(node);
    const bucket = rings.get(key);
    if (bucket) bucket.push(node.nodeId);
    else rings.set(key, [node.nodeId]);
  }
  return rings;
}

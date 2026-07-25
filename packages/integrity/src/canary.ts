export interface CanaryVerdict {
  pass: boolean;
  relativeL2: number;
}

/**
 * Known-answer trap for the fp-aggregate class, where exact quorum is impossible. We inject a
 * work-unit whose answer we know and compare the returned vector by relative L2 distance. Results
 * within benign cross-vendor floating-point drift pass; anything above the tolerance is flagged.
 *
 * HONEST LIMIT (per council): a smart adversary can bias below the drift tolerance undetected. This
 * catches gross tampering and lazy nodes, not a careful sub-drift attacker. fp work is therefore
 * labeled "no fine-grained integrity" at admission — this is a deterrent, not a proof.
 */
export function canaryVerdict(expected: number[], observed: number[], tolerance: number): CanaryVerdict {
  if (expected.length !== observed.length) {
    return { pass: false, relativeL2: Infinity };
  }
  let num = 0;
  let den = 0;
  for (let i = 0; i < expected.length; i++) {
    const d = observed[i]! - expected[i]!;
    num += d * d;
    den += expected[i]! * expected[i]!;
  }
  const relativeL2 = den === 0 ? Math.sqrt(num) : Math.sqrt(num) / Math.sqrt(den);
  return { pass: relativeL2 <= tolerance, relativeL2 };
}

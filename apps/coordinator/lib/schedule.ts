// Per-round inner LR schedule — identical to apps/worker/moregpu_worker/train/schedule.py (golden: tests/goldens/lr_schedule.json).
export interface LrCfg { lr: number; target_samples?: number; max_rounds?: number; lr_schedule?: { kind: 'constant' | 'cosine'; warmup_frac?: number; min_lr?: number } }

export function progress(cfg: LrCfg, samplesSeen: number, roundSamples: number, round: number): number | null {
  if (cfg.target_samples) return (samplesSeen + roundSamples / 2) / cfg.target_samples;
  if (cfg.max_rounds) return (round + 0.5) / cfg.max_rounds;
  return null;
}

export function lrAt(cfg: LrCfg, p: number | null): number {
  const s = cfg.lr_schedule;
  if (!s || (s.kind ?? 'constant') === 'constant' || p === null) return cfg.lr;
  if (s.kind !== 'cosine') throw new Error(`unknown lr schedule ${s.kind}`);
  const w = s.warmup_frac ?? 0, lo = s.min_lr ?? 0, q0 = Math.min(1, Math.max(0, p));
  if (w > 0 && q0 < w) return (cfg.lr * q0) / w;
  const q = (q0 - w) / Math.max(1e-12, 1 - w);
  return lo + (cfg.lr - lo) * 0.5 * (1 + Math.cos(Math.PI * q));
}

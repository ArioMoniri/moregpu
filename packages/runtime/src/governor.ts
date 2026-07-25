/**
 * ThrottleGovernor — the worker-side control loop that decides, sample by sample, what fraction of
 * this machine MoreGPU may use *right now*. Its prime directive: the person using the donor PC must
 * not feel the background compute. Secondary: never cook the machine or drain a battery.
 *
 * Design: a smoothed (EMA) tracker chases a target derived from interactive load, so we ramp down
 * fast when the user gets busy and ramp back up gently when they idle (hysteresis prevents flapping).
 * Hard safety caps (thermal, battery) override the smoothed value outright.
 */

export interface GovernorConfig {
  /** Absolute ceiling on the offered fraction (e.g. an admin policy of "never exceed 60%"). */
  maxFraction: number;
  /** Floor on the offered fraction under normal conditions. */
  minFraction: number;
  /** Interactive load below this counts as "idle" (headroom fully available). */
  idleThreshold: number;
  /** Above this GPU/CPU temperature, stop entirely and cool down. */
  thermalCeilingC: number;
  /** If true, cap hard while on battery to save energy. */
  respectBattery: boolean;
  /** EMA factor in (0,1]; higher reacts faster. */
  smoothing: number;
  /**
   * Consecutive idle samples required before GPU compute may start (true-idle gate, ADR 0005 / Fix 4).
   * Foreground coexistence is NOT claimed: the governor yields fully until the machine has been quiet
   * for this many samples AND no foreground GPU app is active. Default 3.
   */
  idleGraceSamples?: number;
  /**
   * Hard per-dispatch budget in ms so GPU preemption points recur often and a chunk can never inject a
   * long foreground stall. Advisory to the dispatcher; default 12 (~one frame). See ADR 0005.
   */
  maxDispatchMs?: number;
}

export interface Telemetry {
  /** 0..1 fraction of the machine the interactive user is consuming. */
  interactiveLoad: number;
  temperatureC: number;
  onBattery: boolean;
  /** True if a foreground app is actively using the GPU (game, video, GPU-heavy editor). */
  foregroundGpuActive?: boolean;
}

export interface Decision {
  availableFraction: number;
  reason: string;
}

const BATTERY_CAP = 0.25;

/**
 * Adaptive per-machine duty cycle. Given the machine's current utilization (0..1), returns the fraction
 * of time the pool may compute so that TOTAL system utilization stays under `maxUtil`. As the machine's
 * own user drives utilization up, the returned duty falls toward `minDuty`; when idle it rises to `ceil`.
 * Pure and deterministic — this is the formula the worker agent samples `loadavg` into.
 */
export function adaptiveDutyFromUtil(util: number, opts: { ceil: number; maxUtil: number; minDuty: number }): number {
  const u = Math.max(0, Math.min(1, util));
  const slack = Math.max(0, (opts.maxUtil - u) / opts.maxUtil);
  return Math.max(opts.minDuty, Math.min(opts.ceil, opts.minDuty + (opts.ceil - opts.minDuty) * slack));
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

export class ThrottleGovernor {
  private smoothed = 0;
  private consecutiveIdle = 0;
  private last: Decision | undefined;

  constructor(private readonly cfg: GovernorConfig) {}

  /** Advisory hard per-dispatch budget (ms) — keeps preemption points frequent (ADR 0005). */
  maxDispatchMs(): number {
    return this.cfg.maxDispatchMs ?? 12;
  }

  /** Feed one telemetry sample; returns the fraction MoreGPU may use until the next sample. */
  update(t: Telemetry): number {
    const load = clamp(t.interactiveLoad, 0, 1);
    const graceNeeded = this.cfg.idleGraceSamples ?? 3;

    // True-idle gate: only genuinely-quiet machines with no foreground GPU app may run compute.
    const quiet = load <= this.cfg.idleThreshold && t.foregroundGpuActive !== true;
    this.consecutiveIdle = quiet ? this.consecutiveIdle + 1 : 0;
    const idleGateOpen = this.consecutiveIdle >= graceNeeded;

    let fraction: number;
    let reason: string;

    if (!idleGateOpen) {
      // Not true-idle (user active or foreground GPU busy) → yield fully, decay any residual budget.
      this.smoothed *= 0.5;
      fraction = 0;
      reason = t.foregroundGpuActive === true
        ? 'foreground GPU app active — yielding fully (true-idle-only)'
        : `not true-idle yet (${this.consecutiveIdle}/${graceNeeded}) — yielding to interactive load ${load.toFixed(2)}`;
    } else {
      // Idle: ramp up toward full headroom via EMA for smooth, non-flappy transitions.
      const a = clamp(this.cfg.smoothing, 0.001, 1);
      this.smoothed = this.smoothed + a * (1 - this.smoothed);
      fraction = clamp(this.smoothed, this.cfg.minFraction, this.cfg.maxFraction);
      reason = 'true-idle: full headroom';
    }

    // Hard safety overrides (these win over everything above).
    if (t.temperatureC >= this.cfg.thermalCeilingC) {
      fraction = 0;
      reason = `thermal ceiling ${this.cfg.thermalCeilingC}C reached — cooling down`;
    } else if (this.cfg.respectBattery && t.onBattery) {
      fraction = Math.min(fraction, BATTERY_CAP);
      if (fraction <= BATTERY_CAP) reason = 'on battery — capped to save energy';
    }

    this.last = { availableFraction: fraction, reason };
    return fraction;
  }

  /** The most recent offered fraction. */
  current(): number {
    return this.last?.availableFraction ?? 0;
  }

  /** The most recent full decision, for heartbeat/telemetry reporting. */
  lastDecision(): Decision | undefined {
    return this.last;
  }
}

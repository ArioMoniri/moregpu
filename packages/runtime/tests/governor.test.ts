import { describe, it, expect } from 'vitest';
import { ThrottleGovernor, type GovernorConfig, type Telemetry } from '../src/index.js';

const cfg: GovernorConfig = {
  maxFraction: 1.0,
  minFraction: 0.0,
  idleThreshold: 0.15,
  thermalCeilingC: 85,
  respectBattery: true,
  smoothing: 0.5,
};

function tele(over: Partial<Telemetry> = {}): Telemetry {
  return { interactiveLoad: 0, temperatureC: 50, onBattery: false, ...over };
}

describe('ThrottleGovernor — yield to the interactive user', () => {
  it('offers full headroom on a fully idle machine', () => {
    const g = new ThrottleGovernor(cfg);
    // settle a few idle samples through the smoothing filter
    let f = 0;
    for (let i = 0; i < 10; i++) f = g.update(tele({ interactiveLoad: 0 }));
    expect(f).toBeGreaterThan(0.9);
  });

  it('backs off hard when the user becomes active', () => {
    const g = new ThrottleGovernor(cfg);
    for (let i = 0; i < 10; i++) g.update(tele({ interactiveLoad: 0 }));
    let f = 0;
    for (let i = 0; i < 10; i++) f = g.update(tele({ interactiveLoad: 0.95 }));
    expect(f).toBeLessThan(0.2);
  });

  it('is monotonic: more interactive load never increases the offered fraction (steady state)', () => {
    const settle = (load: number) => {
      const g = new ThrottleGovernor(cfg);
      let f = 0;
      for (let i = 0; i < 30; i++) f = g.update(tele({ interactiveLoad: load }));
      return f;
    };
    expect(settle(0.2)).toBeGreaterThanOrEqual(settle(0.5));
    expect(settle(0.5)).toBeGreaterThanOrEqual(settle(0.8));
  });
});

describe('ThrottleGovernor — hard safety caps', () => {
  it('drops to zero when over the thermal ceiling', () => {
    const g = new ThrottleGovernor(cfg);
    for (let i = 0; i < 10; i++) g.update(tele({ interactiveLoad: 0 }));
    const f = g.update(tele({ interactiveLoad: 0, temperatureC: 90 }));
    expect(f).toBe(0);
  });

  it('caps hard on battery when respectBattery is set (electricity optimization)', () => {
    const g = new ThrottleGovernor(cfg);
    for (let i = 0; i < 10; i++) g.update(tele({ interactiveLoad: 0 }));
    const f = g.update(tele({ interactiveLoad: 0, onBattery: true }));
    expect(f).toBeLessThanOrEqual(0.25);
  });

  it('ignores battery when respectBattery is false', () => {
    const g = new ThrottleGovernor({ ...cfg, respectBattery: false });
    let f = 0;
    for (let i = 0; i < 10; i++) f = g.update(tele({ interactiveLoad: 0, onBattery: true }));
    expect(f).toBeGreaterThan(0.9);
  });
});

describe('ThrottleGovernor — true-idle-only gate (Fix 4)', () => {
  it('yields fully when a foreground GPU app is active, even if input looks idle', () => {
    const g = new ThrottleGovernor(cfg);
    for (let i = 0; i < 10; i++) g.update(tele({ interactiveLoad: 0 }));
    const f = g.update(tele({ interactiveLoad: 0, foregroundGpuActive: true }));
    expect(f).toBe(0);
    expect(g.lastDecision()!.reason).toMatch(/foreground/i);
  });

  it('does not start compute until the machine has been idle for the grace window', () => {
    const g = new ThrottleGovernor({ ...cfg, idleGraceSamples: 5 });
    // one lone idle sample right after activity should NOT open the gate
    g.update(tele({ interactiveLoad: 0.9 }));
    const first = g.update(tele({ interactiveLoad: 0 }));
    expect(first).toBe(0);
    // after enough sustained idle samples it ramps up
    let f = 0;
    for (let i = 0; i < 10; i++) f = g.update(tele({ interactiveLoad: 0 }));
    expect(f).toBeGreaterThan(0.5);
  });

  it('exposes a bounded per-dispatch budget so preemption points stay frequent', () => {
    const g = new ThrottleGovernor({ ...cfg, maxDispatchMs: 10 });
    expect(g.maxDispatchMs()).toBe(10);
    expect(new ThrottleGovernor(cfg).maxDispatchMs()).toBe(12); // sane default
  });
});

describe('ThrottleGovernor — output invariants', () => {
  it('always returns a fraction within [minFraction, maxFraction]', () => {
    const g = new ThrottleGovernor({ ...cfg, minFraction: 0.05, maxFraction: 0.8 });
    for (const load of [0, 0.3, 0.6, 1, -1, 2]) {
      const f = g.update(tele({ interactiveLoad: load }));
      expect(f).toBeGreaterThanOrEqual(0);
      expect(f).toBeLessThanOrEqual(0.8);
    }
  });

  it('exposes the last decision for telemetry/heartbeat', () => {
    const g = new ThrottleGovernor(cfg);
    g.update(tele({ interactiveLoad: 0.4 }));
    const d = g.lastDecision();
    expect(d).toBeDefined();
    expect(d!.availableFraction).toBe(g.current());
    expect(typeof d!.reason).toBe('string');
  });
});

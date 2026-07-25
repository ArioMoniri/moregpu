import { describe, it, expect } from 'vitest';
import { adaptiveDutyFromUtil } from '../src/index.js';

const opts = { ceil: 0.6, maxUtil: 0.85, minDuty: 0.05 };

describe('adaptiveDutyFromUtil — pool backs off as the user gets busier', () => {
  it('offers the full ceiling when the machine is idle', () => {
    expect(adaptiveDutyFromUtil(0, opts)).toBeCloseTo(0.6, 5);
  });

  it('drops to the floor when the machine is at/over the utilization cap', () => {
    expect(adaptiveDutyFromUtil(0.85, opts)).toBeCloseTo(0.05, 5);
    expect(adaptiveDutyFromUtil(1, opts)).toBe(0.05);
  });

  it('is monotonically non-increasing as utilization rises (more user use → less pool)', () => {
    let prev = Infinity;
    for (let u = 0; u <= 1.0001; u += 0.1) {
      const d = adaptiveDutyFromUtil(u, opts);
      expect(d).toBeLessThanOrEqual(prev + 1e-9);
      prev = d;
    }
  });

  it('never exceeds the ceiling nor drops below the floor', () => {
    for (const u of [0, 0.2, 0.5, 0.8, 1, -1, 2]) {
      const d = adaptiveDutyFromUtil(u, opts);
      expect(d).toBeGreaterThanOrEqual(0.05);
      expect(d).toBeLessThanOrEqual(0.6);
    }
  });

  it('reserves more headroom for the user when maxUtil is lower', () => {
    const strict = adaptiveDutyFromUtil(0.5, { ...opts, maxUtil: 0.6 });
    const relaxed = adaptiveDutyFromUtil(0.5, { ...opts, maxUtil: 0.95 });
    expect(strict).toBeLessThan(relaxed);
  });
});

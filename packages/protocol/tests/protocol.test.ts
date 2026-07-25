import { describe, it, expect } from 'vitest';
import {
  PROTOCOL_VERSION,
  makeEnvelope,
  parseEnvelope,
  isEnvelope,
  capabilityScore,
  MessageType,
  type NodeCapability,
  type Envelope,
} from '../src/index.js';

const gpuCap: NodeCapability = {
  nodeId: 'node-1',
  backend: 'webgpu',
  vendor: 'nvidia',
  os: 'linux',
  vramMB: 8192,
  logicalCores: 16,
  maxComputeInvocations: 65535,
  supportsF16: true,
};

const cpuCap: NodeCapability = {
  nodeId: 'node-2',
  backend: 'wasm-cpu',
  vendor: 'unknown',
  os: 'windows',
  vramMB: 0,
  logicalCores: 8,
  maxComputeInvocations: 0,
  supportsF16: false,
};

describe('envelope', () => {
  it('stamps the current protocol version and type', () => {
    const env = makeEnvelope(MessageType.Register, { capability: gpuCap });
    expect(env.v).toBe(PROTOCOL_VERSION);
    expect(env.type).toBe(MessageType.Register);
    expect(env.id).toMatch(/^[0-9a-f-]{36}$/);
    expect(typeof env.ts).toBe('number');
  });

  it('round-trips through JSON', () => {
    const env = makeEnvelope(MessageType.ShardResult, { shardId: 's1', ok: true, output: [1, 2, 3] });
    const wire = JSON.stringify(env);
    const back = parseEnvelope(wire);
    expect(back).toEqual(env);
  });

  it('rejects a payload from a different protocol version', () => {
    const env = makeEnvelope(MessageType.Register, { capability: gpuCap });
    const tampered = JSON.stringify({ ...env, v: 999 });
    expect(() => parseEnvelope(tampered)).toThrow(/version/i);
  });

  it('rejects structurally invalid wire data', () => {
    expect(() => parseEnvelope('{"not":"an envelope"}')).toThrow();
    expect(() => parseEnvelope('not json at all')).toThrow();
  });

  it('type-guards unknown objects', () => {
    expect(isEnvelope(makeEnvelope(MessageType.Heartbeat, { load: 0.2 }))).toBe(true);
    expect(isEnvelope({ foo: 'bar' })).toBe(false);
    expect(isEnvelope(null)).toBe(false);
  });
});

describe('capabilityScore', () => {
  it('ranks a GPU node above a CPU-only node for GPU-preferring work', () => {
    expect(capabilityScore(gpuCap)).toBeGreaterThan(capabilityScore(cpuCap));
  });

  it('never returns a negative score', () => {
    expect(capabilityScore(cpuCap)).toBeGreaterThanOrEqual(0);
  });

  it('rewards more VRAM monotonically for GPU nodes', () => {
    const small = capabilityScore({ ...gpuCap, vramMB: 2048 });
    const big = capabilityScore({ ...gpuCap, vramMB: 24576 });
    expect(big).toBeGreaterThan(small);
  });
});

describe('exhaustive message types', () => {
  it('every MessageType constant is a non-empty string', () => {
    for (const t of Object.values(MessageType)) {
      expect(typeof t).toBe('string');
      expect((t as string).length).toBeGreaterThan(0);
    }
  });

  it('produces a discriminable envelope for each type', () => {
    const seen = new Set<string>();
    for (const t of Object.values(MessageType)) {
      const env: Envelope = makeEnvelope(t, {});
      expect(env.type).toBe(t);
      seen.add(env.id);
    }
    expect(seen.size).toBe(Object.values(MessageType).length);
  });
});

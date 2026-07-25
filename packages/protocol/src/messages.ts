import { randomUUID } from 'node:crypto';
import type { NodeCapability } from './types.js';

/** Bumped on any breaking change to the envelope or payload shapes. */
export const PROTOCOL_VERSION = 1 as const;

/** Every message that can cross the coordinator boundary. */
export enum MessageType {
  Register = 'register',
  RegisterAck = 'register_ack',
  Heartbeat = 'heartbeat',
  ShardAssign = 'shard_assign',
  ShardResult = 'shard_result',
  ThrottleUpdate = 'throttle_update',
  JobStatus = 'job_status',
  Drain = 'drain',
}

/** Uniform framing for everything on the wire. */
export interface Envelope<T = unknown> {
  /** Protocol version. */
  v: typeof PROTOCOL_VERSION;
  /** Unique message id (uuid v4). */
  id: string;
  /** Message type discriminator. */
  type: MessageType;
  /** Epoch ms when framed. */
  ts: number;
  /** Type-specific payload. */
  payload: T;
}

/** Construct a stamped envelope. Uses a real UUID so ids are globally unique. */
export function makeEnvelope<T>(type: MessageType, payload: T): Envelope<T> {
  return { v: PROTOCOL_VERSION, id: randomUUID(), type, ts: Date.now(), payload };
}

/** Narrow an unknown value to an Envelope shape without trusting its version. */
export function isEnvelope(value: unknown): value is Envelope {
  if (typeof value !== 'object' || value === null) return false;
  const e = value as Record<string, unknown>;
  return (
    typeof e.v === 'number' &&
    typeof e.id === 'string' &&
    typeof e.type === 'string' &&
    typeof e.ts === 'number' &&
    'payload' in e &&
    (Object.values(MessageType) as string[]).includes(e.type as string)
  );
}

/** Parse a wire string into a validated Envelope, rejecting version/shape mismatches. */
export function parseEnvelope(wire: string): Envelope {
  let obj: unknown;
  try {
    obj = JSON.parse(wire);
  } catch {
    throw new Error('protocol: invalid JSON on wire');
  }
  if (!isEnvelope(obj)) {
    throw new Error('protocol: payload is not a valid envelope');
  }
  if (obj.v !== PROTOCOL_VERSION) {
    throw new Error(`protocol: unsupported version ${obj.v} (expected ${PROTOCOL_VERSION})`);
  }
  return obj;
}

/**
 * Heuristic desirability score for a node. Higher is better. Used by the scheduler to rank
 * candidates. GPU nodes dominate CPU nodes; more VRAM and cores raise the score monotonically.
 * Never negative so it can be used directly as a weight.
 */
export function capabilityScore(cap: NodeCapability): number {
  const backendWeight = cap.backend === 'native-accel' ? 1500 : cap.backend === 'webgpu' ? 1000 : 100;
  const vramWeight = cap.backend === 'wasm-cpu' ? 0 : cap.vramMB / 16;
  const coreWeight = cap.logicalCores * 4;
  const f16Bonus = cap.supportsF16 ? 50 : 0;
  const wmmaBonus = cap.wmmaSupported ? 200 : 0;
  return Math.max(0, backendWeight + vramWeight + coreWeight + f16Bonus + wmmaBonus);
}

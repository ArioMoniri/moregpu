import { describe, it, expect } from 'vitest';
import {
  deriveTenantKey,
  seal,
  unseal,
  type SealedBlob,
} from '../src/index.js';

const master = new Uint8Array(32).fill(7);

describe('deriveTenantKey', () => {
  it('is deterministic for the same tenant', async () => {
    const a = await deriveTenantKey(master, 'tenant-A');
    const b = await deriveTenantKey(master, 'tenant-A');
    expect(Buffer.from(a).equals(Buffer.from(b))).toBe(true);
  });

  it('differs across tenants (isolation)', async () => {
    const a = await deriveTenantKey(master, 'tenant-A');
    const b = await deriveTenantKey(master, 'tenant-B');
    expect(Buffer.from(a).equals(Buffer.from(b))).toBe(false);
  });

  it('produces a 32-byte key', async () => {
    const k = await deriveTenantKey(master, 'tenant-A');
    expect(k.length).toBe(32);
  });
});

describe('seal / unseal round-trip', () => {
  it('recovers the original plaintext', async () => {
    const key = await deriveTenantKey(master, 'tenant-A');
    const plaintext = new TextEncoder().encode('secret work unit payload');
    const blob = await seal(key, plaintext);
    const out = await unseal(key, blob);
    expect(new TextDecoder().decode(out)).toBe('secret work unit payload');
  });

  it('emits base64 fields, never plaintext', async () => {
    const key = await deriveTenantKey(master, 'tenant-A');
    const blob = await seal(key, new TextEncoder().encode('topsecret-marker'));
    const wire = JSON.stringify(blob);
    expect(wire).not.toContain('topsecret-marker');
    expect(blob.iv).toMatch(/^[A-Za-z0-9+/=]+$/);
    expect(blob.ct).toMatch(/^[A-Za-z0-9+/=]+$/);
  });

  it('uses a fresh IV each time (no nonce reuse)', async () => {
    const key = await deriveTenantKey(master, 'tenant-A');
    const p = new TextEncoder().encode('same input');
    const b1 = await seal(key, p);
    const b2 = await seal(key, p);
    expect(b1.iv).not.toBe(b2.iv);
    expect(b1.ct).not.toBe(b2.ct);
  });
});

describe('tamper + isolation guarantees', () => {
  it('rejects a wrong-tenant key (cannot cross-read)', async () => {
    const keyA = await deriveTenantKey(master, 'tenant-A');
    const keyB = await deriveTenantKey(master, 'tenant-B');
    const blob = await seal(keyA, new TextEncoder().encode('A-only data'));
    await expect(unseal(keyB, blob)).rejects.toThrow();
  });

  it('rejects a tampered ciphertext (authentication)', async () => {
    const key = await deriveTenantKey(master, 'tenant-A');
    const blob = await seal(key, new TextEncoder().encode('integrity matters'));
    const flipped: SealedBlob = { ...blob, ct: flipFirstByte(blob.ct) };
    await expect(unseal(key, flipped)).rejects.toThrow();
  });
});

function flipFirstByte(b64: string): string {
  const buf = Buffer.from(b64, 'base64');
  buf[0] ^= 0xff;
  return buf.toString('base64');
}

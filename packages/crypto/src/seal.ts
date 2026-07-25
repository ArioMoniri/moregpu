/**
 * Sealing primitives for MoreGPU privacy. Built on WebCrypto (crypto.subtle) so the *same*
 * code runs in Node (coordinator/consumer) and in the headless-browser worker.
 *
 * WHAT THIS PROTECTS (and what it does NOT — verifier Fix 2):
 *  ✓ Confidentiality in TRANSIT and at the honest-but-curious COORDINATOR relay: it only ever moves
 *    `SealedBlob`s it cannot open.
 *  ✓ Cross-TENANT compartmentalization: one tenant's key cannot decrypt another tenant's blob, so the
 *    coordinator cannot cross-deliver shards.
 *  ✗ It does NOT hide plaintext from the LOCAL USER/ADMIN of the donor machine that executes a shard.
 *    To compute, the worker must hold the plaintext in VRAM/process memory, and the machine's owner can
 *    dump it. There is no software confidentiality on GeForce/iGPU. Sealing is a transit/relay guarantee,
 *    not a shield against the donor host. Outsourced-secret work must route to a cloud CVM or be rejected
 *    (see @moregpu/oracle confidentiality routing).
 */

const AES_GCM = 'AES-GCM';
const IV_BYTES = 12; // 96-bit nonce, the GCM standard.

/** Authenticated ciphertext safe to relay through an untrusted coordinator. */
export interface SealedBlob {
  /** base64 IV/nonce. */
  iv: string;
  /** base64 ciphertext with appended GCM tag. */
  ct: string;
}

function toB64(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString('base64');
}

function fromB64(b64: string): Uint8Array<ArrayBuffer> {
  return copy(Buffer.from(b64, 'base64'));
}

/**
 * Copy any byte view into a fresh, plain-ArrayBuffer-backed Uint8Array. WebCrypto's `BufferSource`
 * requires an ArrayBuffer (not SharedArrayBuffer/pooled Buffer), which TS 5.7's generic TypedArray
 * types now enforce; this normalizes inputs so the subtle-crypto calls type-check and stay correct.
 */
function copy(u: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(u.byteLength);
  out.set(u);
  return out;
}

/**
 * Derive a stable 32-byte tenant key from a master secret via HKDF-SHA256.
 * Deterministic per (master, tenantId); independent across tenants.
 */
export async function deriveTenantKey(master: Uint8Array, tenantId: string): Promise<Uint8Array> {
  const base = await crypto.subtle.importKey('raw', copy(master), 'HKDF', false, ['deriveBits']);
  const info = new TextEncoder().encode(`moregpu:tenant:${tenantId}`);
  const salt = new TextEncoder().encode('moregpu:hkdf:v1');
  const bits = await crypto.subtle.deriveBits(
    { name: 'HKDF', hash: 'SHA-256', salt, info },
    base,
    256,
  );
  return new Uint8Array(bits);
}

async function importAesKey(raw: Uint8Array): Promise<CryptoKey> {
  return crypto.subtle.importKey('raw', copy(raw), AES_GCM, false, ['encrypt', 'decrypt']);
}

/** Seal plaintext with a tenant key. Produces a fresh random IV every call (no nonce reuse). */
export async function seal(key: Uint8Array, plaintext: Uint8Array): Promise<SealedBlob> {
  const iv = crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const cryptoKey = await importAesKey(key);
  const ct = await crypto.subtle.encrypt({ name: AES_GCM, iv }, cryptoKey, copy(plaintext));
  return { iv: toB64(iv), ct: toB64(new Uint8Array(ct)) };
}

/** Unseal a blob with a tenant key. Throws on wrong key or any tampering (GCM authentication). */
export async function unseal(key: Uint8Array, blob: SealedBlob): Promise<Uint8Array> {
  const iv = fromB64(blob.iv);
  const ct = fromB64(blob.ct);
  const cryptoKey = await importAesKey(key);
  try {
    const pt = await crypto.subtle.decrypt({ name: AES_GCM, iv }, cryptoKey, ct);
    return new Uint8Array(pt);
  } catch {
    throw new Error('crypto: unseal failed — wrong tenant key or tampered ciphertext');
  }
}

// Tensor wire format for DiLoCo sync payloads (ADR-0106). Byte-compatible with
// apps/worker/moregpu_worker/train/tensorwire.py; cross-language goldens in tests/goldens/diloco_tensorwire.json.
// Pure TS + WebCrypto: runs under Deno (coordinator) and Node (vitest).

export type WireDtype = 'f32' | 'bf16' | 'fp16' | 'int8delta';
export interface WireEntry { name: string; shape: number[]; offset: number; nbytes: number; scales_offset?: number; nblocks?: number; block?: number }
export interface WireHeader { v: 1; dtype: WireDtype; sha256: string; tensors: WireEntry[]; error?: { max_abs: number; rel_l2: number } }

export function b64ToBytes(s: string): Uint8Array {
  const F = (Uint8Array as unknown as { fromBase64?: (s: string) => Uint8Array }).fromBase64;
  if (typeof F === 'function') return F(s);   // native (Deno 2.x / modern V8): ~10x faster for MB payloads
  const bin = atob(s); const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
export function bytesToB64(b: Uint8Array): string {
  const f = (b as unknown as { toBase64?: () => string }).toBase64;
  if (typeof f === 'function') return f.call(b);
  let s = ''; const CH = 0x8000;
  for (let i = 0; i < b.length; i += CH) s += String.fromCharCode(...b.subarray(i, i + CH));
  return btoa(s);
}
export async function sha256Hex(b: Uint8Array): Promise<string> {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', b as unknown as ArrayBuffer));
  return Array.from(d, (x) => x.toString(16).padStart(2, '0')).join('');
}
const numel = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

// IEEE half <-> float (round-to-nearest-even, matches numpy/torch)
const f32buf = new Float32Array(1), u32buf = new Uint32Array(f32buf.buffer);
export function halfToFloat(h: number): number {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
export function floatToHalf(v: number): number {
  f32buf[0] = v; const x = u32buf[0]!;
  const sign = (x >>> 16) & 0x8000; let e = (x >>> 23) & 0xff; let m = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  e = e - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    m = m | 0x800000; const shift = 14 - e;
    let hm = m >>> shift; const rem = m & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rem > half || (rem === half && (hm & 1))) hm++;
    return sign | hm;
  }
  let hm = m >>> 13; const rem = m & 0x1fff;
  let he = e;
  if (rem > 0x1000 || (rem === 0x1000 && (hm & 1))) { hm++; if (hm === 0x400) { hm = 0; he++; if (he >= 0x1f) return sign | 0x7c00; } }
  return sign | (he << 10) | hm;
}
export function floatToBf16(v: number): number {
  f32buf[0] = v; const x = u32buf[0]!;
  if ((x & 0x7f800000) === 0x7f800000 && (x & 0x7fffff)) return (x >>> 16) | 0x40;
  const lsb = (x >>> 16) & 1;
  return ((x + 0x7fff + lsb) >>> 16) & 0xffff;
}
export function bf16ToFloat(h: number): number { u32buf[0] = (h & 0xffff) << 16; return f32buf[0]!; }

export async function decodeTensors(hdr: WireHeader, blob: Uint8Array, ref?: Map<string, Float32Array>): Promise<Map<string, Float32Array>> {
  if (await sha256Hex(blob) !== hdr.sha256) throw new Error('tensor payload sha256 mismatch');
  if (hdr.dtype === 'int8delta' && !ref) throw new Error('int8delta decode needs the reference tensors');
  const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const out = new Map<string, Float32Array>();
  for (const e of hdr.tensors) {
    if (!Number.isInteger(e.offset) || !Number.isInteger(e.nbytes) || e.offset < 0 || e.nbytes < 0 || e.offset + e.nbytes > blob.byteLength) throw new Error(`${e.name}: offset/size outside the payload`);
    const n = numel(e.shape), x = new Float32Array(n);
    if (hdr.dtype === 'f32') { if (e.nbytes !== 4 * n) throw new Error(`${e.name}: size mismatch`); for (let i = 0; i < n; i++) x[i] = dv.getFloat32(e.offset + 4 * i, true); }
    else if (hdr.dtype === 'bf16') { if (e.nbytes !== 2 * n) throw new Error(`${e.name}: size mismatch`); for (let i = 0; i < n; i++) x[i] = bf16ToFloat(dv.getUint16(e.offset + 2 * i, true)); }
    else if (hdr.dtype === 'fp16') { if (e.nbytes !== 2 * n) throw new Error(`${e.name}: size mismatch`); for (let i = 0; i < n; i++) x[i] = halfToFloat(dv.getUint16(e.offset + 2 * i, true)); }
    else if (hdr.dtype === 'int8delta') {
      const nb = e.nblocks!, blk = e.block!, r = ref!.get(e.name);
      if (!r || r.length !== n) throw new Error(`${e.name}: missing/mismatched reference`);
      if (e.nbytes !== 4 * nb + n) throw new Error(`${e.name}: size mismatch`);
      const qo = e.offset + 4 * nb;
      for (let i = 0; i < n; i++) {
        const sc = dv.getFloat32(e.offset + 4 * Math.floor(i / blk), true);
        x[i] = Math.fround(Math.fround(dv.getInt8(qo + i) * sc) + r[i]!);
      }
    } else throw new Error(`unknown wire dtype ${hdr.dtype}`);
    out.set(e.name, x);
  }
  return out;
}

/** Encode f32/bf16/fp16 (the coordinator never needs to produce int8delta: it broadcasts the global). */
export async function encodeTensors(t: Map<string, Float32Array>, shapes: Record<string, number[]>, dtype: 'f32' | 'bf16' | 'fp16' = 'f32'): Promise<{ header: WireHeader; blob: Uint8Array }> {
  const width = dtype === 'f32' ? 4 : 2;
  let total = 0; for (const a of t.values()) total += a.length * width;
  const blob = new Uint8Array(total), dv = new DataView(blob.buffer);
  const tensors: WireEntry[] = []; let off = 0;
  for (const [name, a] of t) {
    const shape = shapes[name] ?? [a.length];
    for (let i = 0; i < a.length; i++) {
      if (dtype === 'f32') dv.setFloat32(off + 4 * i, a[i]!, true);
      else dv.setUint16(off + 2 * i, dtype === 'bf16' ? floatToBf16(a[i]!) : floatToHalf(a[i]!), true);
    }
    tensors.push({ name, shape, offset: off, nbytes: a.length * width }); off += a.length * width;
  }
  return { header: { v: 1, dtype, sha256: await sha256Hex(blob), tensors }, blob };
}

export function chunkBytes(b: Uint8Array, size = 4 << 20): Uint8Array[] {
  const out: Uint8Array[] = [];
  for (let i = 0; i < b.length; i += size) out.push(b.subarray(i, i + size));
  return out.length ? out : [new Uint8Array(0)];
}
export function concatBytes(parts: Uint8Array[]): Uint8Array {
  let n = 0; for (const p of parts) n += p.length;
  const out = new Uint8Array(n); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

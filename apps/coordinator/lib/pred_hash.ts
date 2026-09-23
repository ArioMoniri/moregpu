// pred_sha256 (moregpu.pred/1): a deterministic hash of a predicted label volume. Byte-identical to the Python worker's
// moregpu_worker/vision/pred_hash.py; tests/goldens/pred_sha256.json is checked by both sides.
//
// Preimage, as bytes:  "moregpu.pred/1\n" + DTYPE + "\n" + SHAPE + "\n" + LABELS
//   DTYPE   "uint8" if every label <= 255 (also for an empty volume), else "uint16" (> 65535, negative or non-integer
//           labels are refused)
//   SHAPE   the label volume's shape as decimal integers joined by "," with no spaces ("" for a 0-d scalar)
//   LABELS  the labels in C order (last axis fastest), little-endian, 1 or 2 bytes each
// Labels from logits: argmax over the class axis (axis 1 of [B, C, ...]) → [B, ...]; the first maximum wins a tie and a
// NaN counts as the maximum (the first NaN wins), as in torch.argmax. WebGPU workers return logits, never labels, so the
// coordinator derives the labels (and the hash) for torch and WebGPU replies with this one function.
// SHA-256 is WebCrypto (crypto.subtle), available in Deno, browsers and Node 18+.

const VERSION = 'moregpu.pred/1';
const SHA_RE = /^[0-9a-f]{64}$/;

export function labelDtype(labels: ArrayLike<number>): 'uint8' | 'uint16' {
  let hi = 0;
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i]!;
    if (!Number.isInteger(v)) throw new Error('labels must be integer class indices');
    if (v < 0) throw new Error(`labels must not be negative (min ${v})`);
    if (v > hi) hi = v;
  }
  if (hi > 65535) throw new Error(`labels above 65535 are not supported (max ${hi})`);
  return hi <= 255 ? 'uint8' : 'uint16';
}

export function predPreimage(labels: ArrayLike<number>, shape: number[]): Uint8Array {
  const n = shape.reduce((a, b) => a * b, 1);
  if (n !== labels.length) throw new Error(`labels: ${labels.length} values for shape [${shape}]`);
  const dt = labelDtype(labels);
  const head = new TextEncoder().encode(`${VERSION}\n${dt}\n${shape.map((d) => String(Math.trunc(d))).join(',')}\n`);
  const width = dt === 'uint8' ? 1 : 2;
  const out = new Uint8Array(head.length + labels.length * width);
  out.set(head, 0);
  const dv = new DataView(out.buffer, head.length);
  for (let i = 0; i < labels.length; i++) {
    if (width === 1) dv.setUint8(i, labels[i]!); else dv.setUint16(2 * i, labels[i]!, true);
  }
  return out;
}

export async function predSha256(labels: ArrayLike<number>, shape: number[]): Promise<string> {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', predPreimage(labels, shape) as unknown as ArrayBuffer));
  return Array.from(d, (x) => x.toString(16).padStart(2, '0')).join('');
}

/** [B, C, ...] f32 logits (C order) → [B, ...] labels: argmax over axis 1, first max wins, NaN wins. */
export function argmaxLabels(data: Float32Array, shape: number[]): { labels: Uint32Array; shape: number[] } {
  if (shape.length < 2) throw new Error(`logits need a class axis: shape [B, C, ...], got [${shape}]`);
  const [B, C] = shape as [number, number];
  const inner = shape.slice(2).reduce((a, b) => a * b, 1);
  if (B * C * inner !== data.length) throw new Error(`logits: ${data.length} values for shape [${shape}]`);
  const labels = new Uint32Array(B * inner);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < inner; i++) {
      let best = 0, bv = data[b * C * inner + i]!;
      if (!Number.isNaN(bv)) {
        for (let c = 1; c < C; c++) {
          const v = data[(b * C + c) * inner + i]!;
          if (Number.isNaN(v)) { best = c; break; }
          if (v > bv) { best = c; bv = v; }
        }
      }
      labels[b * inner + i] = best;
    }
  }
  return { labels, shape: [B, ...shape.slice(2)] };
}

/** pred_sha256 of a vision_infer reply (base64 little-endian f32 logits of `shape`); null if it is not [B, C>0, ...]. */
export async function predFromLogitsB64(b64: string, shape: number[]): Promise<{ pred_sha256: string; pred_shape: number[]; pred_dtype: string } | null> {
  if (!Array.isArray(shape) || shape.length < 2 || !(Number(shape[1]) > 0)) return null;
  try {
    const bin = atob(b64);
    if (bin.length % 4) return null;
    const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    const dv = new DataView(u.buffer), x = new Float32Array(u.length / 4);
    for (let i = 0; i < x.length; i++) x[i] = dv.getFloat32(4 * i, true);
    const { labels, shape: ls } = argmaxLabels(x, shape.map(Number));
    return { pred_sha256: await predSha256(labels, ls), pred_shape: ls, pred_dtype: labelDtype(labels) };
  } catch { return null; }
}

/** A worker-reported pred_sha256 is kept only if it is a 64-hex digest (worker replies are untrusted). */
export function cleanPredSha(v: unknown): string | undefined {
  return typeof v === 'string' && SHA_RE.test(v) ? v : undefined;
}

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { decodeTensors, encodeTensors, b64ToBytes, bytesToB64 } from './tensorwire.ts';
import { OuterState, weightedAverage, outerStep, dropNonFinite } from './diloco.ts';

const G = JSON.parse(readFileSync(new URL('../../../tests/goldens/diloco_tensorwire.json', import.meta.url), 'utf8'));
const f32 = (a: number[]) => Float32Array.from(a);
const mapOf = (o: Record<string, number[]>) => new Map(Object.entries(o).map(([k, v]) => [k, f32(v)]));

describe('tensorwire (byte-compatible with moregpu_worker.train.tensorwire)', () => {
  for (const dt of ['f32', 'bf16', 'fp16', 'int8delta']) {
    it(`decodes ${dt} goldens exactly`, async () => {
      const w = G.wire[dt];
      const out = await decodeTensors(w.header, b64ToBytes(w.blob_b64), dt === 'int8delta' ? mapOf(G.ref) : undefined);
      for (const [k, v] of Object.entries(w.decoded as Record<string, number[]>)) expect(Array.from(out.get(k)!)).toEqual(Array.from(f32(v)));
    });
  }
  it('rejects a tampered blob', async () => {
    const w = G.wire.f32; const b = b64ToBytes(w.blob_b64); b[0] ^= 1;
    await expect(decodeTensors(w.header, b)).rejects.toThrow(/sha256/);
  });
  it('f32 encode reproduces the Python bytes', async () => {
    const { header, blob } = await encodeTensors(mapOf(G.new), G.shapes, 'f32');
    expect(bytesToB64(blob)).toBe(G.wire.f32.blob_b64);
    expect(header.sha256).toBe(G.wire.f32.header.sha256);
  });
  it('bf16/fp16 encode reproduces the Python bytes', async () => {
    for (const dt of ['bf16', 'fp16'] as const) {
      const { blob } = await encodeTensors(mapOf(G.new), G.shapes, dt);
      expect(bytesToB64(blob)).toBe(G.wire[dt].blob_b64);
    }
  });
});

describe('DiLoCo outer loop (matches moregpu_worker.train.diloco)', () => {
  it('reproduces 3 weighted rounds', () => {
    const st = OuterState.init(mapOf(G.diloco.init));
    for (const r of G.diloco.rounds) {
      const avg = weightedAverage(r.workers.map((w: Record<string, number[]>, i: number) => ({ tensors: mapOf(w), weight: r.samples[i] })));
      outerStep(st, avg, G.diloco.lr, G.diloco.momentum);
      for (const [k, v] of Object.entries(r.global_after as Record<string, number[]>)) {
        const got = st.global.get(k)!;
        v.forEach((x, i) => expect(got[i]).toBeCloseTo(x, 5));
      }
    }
    expect(st.round).toBe(3);
  });
  it('lr=1, momentum=0 is plain averaging; equal weights is the plain mean', () => {
    const st = OuterState.init(new Map([['w', f32([5, 5])]]));
    const avg = weightedAverage([{ tensors: new Map([['w', f32([1, 2])]]), weight: 1 }, { tensors: new Map([['w', f32([3, 4])]]), weight: 1 }]);
    outerStep(st, avg, 1, 0);
    expect(Array.from(st.global.get('w')!)).toEqual([2, 3]);
  });
  it('rejects non-positive total weight and length mismatch', () => {
    expect(() => weightedAverage([{ tensors: new Map([['w', f32([1])]]), weight: 0 }])).toThrow();
    expect(() => weightedAverage([{ tensors: new Map([['w', f32([1])]]), weight: 1 }, { tensors: new Map([['w', f32([1, 2])]]), weight: 1 }])).toThrow();
  });
  it('drops non-finite workers', () => {
    const { kept, dropped } = dropNonFinite([{ id: 'a', tensors: new Map([['w', f32([1])]]), weight: 1 }, { id: 'b', tensors: new Map([['w', f32([NaN])]]), weight: 1 }]);
    expect(kept.map((x) => x.id)).toEqual(['a']); expect(dropped).toEqual(['b']);
  });
});

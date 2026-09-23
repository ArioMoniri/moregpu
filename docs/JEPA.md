# JEPA self-supervised pretraining

The tasks are `ijepa_2d` (images), `jepa_2p5d` (adjacent CT/MR slices as channels) and `jepa_3d` (volumetric patch
tokens). Each is a `TrainTask`, so it runs on one worker or across N workers with DiLoCo ([TRAINING.md](TRAINING.md)).

## Model

- **Encoder.** A ViT with 2D or 3D patches: `micro`, `tiny` (192, 12 blocks, 3 heads), `small` or `base`.
  - Its state dict uses **timm-compatible keys**, so an export loads strictly into `timm`'s VisionTransformer
    (`class_token=False, global_pool=''`). This is tested.
  - Positional embedding is fixed sin-cos.
- **Predictor.** A narrow ViT (`pred_dim`, `pred_depth`, `pred_heads`) that takes context tokens plus mask tokens at the
  target positions.
- **Masking.** I-JEPA multi-block masking:
  - `n_targets` blocks with scale 0.15–0.2 and aspect 0.75–1.5, plus one context block with scale 0.85–1.0;
  - target tokens are removed from the context, and each batch is truncated to rectangular index sets;
  - it is seeded and reproducible.
- **Loss.** Smooth-L1 (default) or L2 between the predictions and layer-normed target features.

## EMA under DiLoCo (ADR-0109)

The EMA target is **only updated after each outer step**, from the freshly synced global, and identically on every
worker. The momentum for a round is the product of the per-step schedule over the steps that round covered:

```
m_k = Π m_j        target ← m_k·target + (1 − m_k)·global
```

- With N=1 and H=1 this is exactly per-step I-JEPA. This is tested.
- Workers report the target's SHA-256 every round. A mismatch raises the "target encoders diverged" alarm, which should
  never fire.

## Monitors and evaluation

- **Collapse monitors every round** (on a fixed probe batch):
  - mean and minimum per-dimension std;
  - RankMe effective rank.

  An alarm fires when either drops below a threshold (`collapse_std_min`, `collapse_rank_min`).
- **Evaluation kinds:**
  - `loss`;
  - `features` (mean-pooled, base64 f32);
  - `monitors`;
  - `knn` (leave-one-out cosine);
  - `linear_probe` (deterministic 2-fold logistic regression). Labels come from `ref.meta.label`.

## Export

| Format | Output | Guarantee |
|---|---|---|
| `safetensors` | `encoder.safetensors` + `encoder_config.json` | reloads bit-exactly |
| `torch_export` | `.pt2` | — |
| `onnx` | `.onnx` | parity probe vs PyTorch ≤ 1e-4 |

An exported encoder can:
- initialise `segment` / `classify` (`encoder: {init: export, path}`);
- be served with `/vision/load {export}` for feature extraction.

## Example

```bash
# on each machine that should help:  moregpu torch-join --server wss://HOST:8787/ws --token … --pin …
moregpu train jepa --task jepa_2p5d --data refs.jsonl --size 224,224 --channels 3 \
   --inner-steps 50 --batch 32 --lr 1e-3 --cosine --target-samples 200000 --export /data/enc
moregpu train segment --encoder /data/enc --data seg_refs.jsonl --num-classes 3 --target-samples 20000
```

Other entry points:
- `examples/jepa_synthetic.py` runs with no data (or `--local` for an in-process run).
- `examples/segment_finetune.py` covers JEPA pretraining, then fine-tuning, then export.

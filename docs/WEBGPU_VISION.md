# WebGPU vision inference (M6, ADR-0114)

`apps/worker/vision_wgsl.ts` runs vision models (3D/2D segmentation nets and ViT encoders) on a worker's WebGPU device.
It works under Deno and in a browser tab, from an **op-graph JSON + safetensors** produced by the Python lowering.
`apps/worker/worker.ts` loads it and routes every `vision_*` op of the sealed `model` RPC to it.

## Scope

- **In scope:** inference, plus JEPA-style feature extraction (e.g. the ViT's pooled patch features are just
  another graph output).
- **Not in scope:** training on WebGPU. There is no autograd, optimiser or gradient kernel. Training stays on
  native torch workers (ADR-0114).
- **Shapes:** static. A graph's input shapes are fixed when it is loaded.
- **Pooling:** the sliding-window driver (below) handles volumes larger than the model's ROI.

## Pipeline

```
op-graph JSON + safetensors ─► VisionModel.compile() ─► launch list + memory plan ─┬─► runCpu()          CPU reference
                                                                                   └─► createGpuRunner()  WGSL on WebGPU
```

- **compile** does the following:
  - validates the graph against the op table and infers every shape;
  - lowers each node to kernel launches and buffer copies;
  - folds constants (views or permutes of weights, batch-norm scale/shift);
  - plans memory.
- **CPU reference.** Every WGSL kernel has a CPU mirror in the same file. The mirror reads the **same uniform
  array**, sees the **same binding windows** and uses the same per-output loop order, reduction tree and fp32
  rounding (`Math.fround` after every op). CPU and GPU execute the **same launch list**, including the tiling. So the
  vitest suite (CPU) and the Deno/Playwright suites (GPU) test one algorithm.
- **What can still differ on the GPU:** FMA contraction, and exp/tanh/sqrt/division within the few ulp WGSL allows.
  Both paths are therefore checked against PyTorch.

### Kernels

| kernel | used for | notes |
|---|---|---|
| `conv` | conv2d/3d, patch-embed | Implicit GEMM: the 16×16 tiled GEMM of `worker.ts` `WGSL.matmul`, with the im2col gather fused into the A-tile load. Supports stride, padding, dilation and groups. |
| `convT` | conv_transpose2d/3d | Gather form. Supports stride, padding, output_padding, dilation and groups. |
| `pool` | max/avg pool 2d/3d | Supports padding and dilation (max). avg supports count_include_pad and divisor_override. |
| `upsample` | nearest, bilinear, trilinear | PyTorch's `nearest_idx` and `area_pixel_compute_source_index`. Supports align_corners, and both scale_factors and output_size. |
| `norm` | group / instance / layer norm | One 256-thread workgroup per row. Two-pass mean/var with a tree reduction. Affine per channel or per element. |
| `chanop` | batch norm (eval), prelu | BN is **folded** at load: `scale = w·rsqrt(var+eps)`, `shift = b − mean·scale`. This is PyTorch's own CPU eval formula. `batchNorm: 'explicit'` is available too. |
| `unary` | relu, leaky_relu, gelu (erf / tanh), sigmoid, tanh, silu, abs, neg, exp, sqrt, rsqrt | erf uses Eigen's fp32 rational approximation, identical in WGSL and in the mirror. |
| `binary` | add/sub (alpha), mul, div | numpy broadcasting up to rank 6. Scalars become 1-element constants. |
| `linered` | softmax, argmax, mean along any dim | argmax returns the first maximal index, as f32. |
| `matmul` | linear, addmm, mm, bmm, matmul | Batched tiled GEMM (from `WGSL.matmul`) with transposed-B, bias, alpha/beta and batch broadcasting. |
| `attention` | scaled_dot_product_attention | Online softmax, extended from `SHARD_WGSL.cachedAttn`. Any batch×heads, optional causal. |
| `copy`, `pad` | permute, transpose, expand, select, slice, pad | Strided gather. Constant, replicate or reflect pad. |

`aten.cat` uses buffer-to-buffer copies, so it has no binding-size limit. Views (`view`, `reshape`, `flatten`,
`unsqueeze`, `squeeze`, `contiguous`, `clone`, `dropout`) alias their source and cost nothing.

2D ops are run as 3D with a trailing `W=1`. This is bit-identical and keeps one kernel per op family.

### Precision

- **fp32** storage and accumulation by default.
- **`f16Weights: true`** (needs `shader-f16`) stores conv/convT/linear weights as f16 and accumulates in f32. The CPU
  mirror rounds the weights the same way (round-to-nearest-even).

Measured in this container:

| case | vs PyTorch fp32 (‖Δ‖∞/‖ref‖∞) |
|---|---|
| 98 per-kernel goldens, CPU reference | ≤ 5e-7 (most ~1e-7) |
| 98 per-kernel goldens, WGSL (lavapipe and SwiftShader) | ≤ 1e-5 (all pass) |
| tiny 3D BasicUNet logits / probs, WGSL | 9.0e-7 / 5.7e-7. The GPU logits equal the CPU mirror bit for bit. |
| ViT-Tiny-width logits / features, WGSL | 6.3e-7 / 7.4e-7 |
| MONAI `sliding_window_inference` (gaussian and constant+pad), CPU and WGSL | ≤ 5.1e-7 |
| f16 weights: ViT / UNet / seg2d logits | 2.7e-4 / 7.3e-4 / 6.5e-5. GPU vs the CPU f16 mirror: ≤ 5e-7. |

## Op-graph contract

The machine-readable source of truth is **`apps/worker/vision_ops.json`**. A test pins its op list to the
executor's `SUPPORTED_OPS`, and the Python lowering (`apps/worker/moregpu_worker/vision/lowering.py`) should read it
to decide whether a model can be lowered.

```json
{"version": 1,
 "inputs":  [{"name": "x", "shape": [1, 1, 96, 96, 96]}],
 "nodes":   [{"op": "aten.conv3d.default", "inputs": ["x", "enc.0.weight", "enc.0.bias"],
              "attrs": {"stride": [1,1,1], "padding": [1,1,1], "dilation": [1,1,1], "groups": 1}, "output": "conv3d"}],
 "outputs": ["logits"],
 "weights": "model.safetensors"}
```

- **`op`:** `aten.<name>`, optionally followed by `.<overload>`. The executor matches on `aten.<name>`.
- **`inputs`:** the tensor arguments in ATen-schema order. Each is the name of a graph input, an earlier node's
  output or a safetensors key. An absent optional tensor is `null`, and trailing nulls may be dropped. A `Tensor[]`
  argument (for `aten.cat`) is flattened in place.
- **`attrs`:** every non-tensor argument, keyed by its ATen **schema argument name**, with defaults filled in. A
  Python number in a Tensor slot (as in `x * 0.5`) goes in `attrs` under that argument's name (`other`).
- **Multi-output ops** (`native_layer_norm`, `native_group_norm`, `_native_batch_norm_legit_no_training`,
  `max_pool*_with_indices`) produce **only output 0**. The lowering maps `getitem(node, 0)` to `output` and must not
  use the other outputs.
- **dtypes:** everything is f32 at the interface. Safetensors weights may be F32, F16, BF16 or integer; they are
  widened on load.

The executor accepts both forms of each op: the high-level training-IR ops that `torch.export.export()` emits
(`conv3d`, `instance_norm`, `layer_norm`, `linear`, `scaled_dot_product_attention`, …) and the core-ATen forms
(`convolution`, `_softmax`, `native_group_norm`, `native_layer_norm`, `addmm`/`permute`/`view`,
`max_pool*_with_indices`, …).

**Recommendation for the lowering:** export with `torch.export.export()` and do **not** run the default core-ATen
decomposition. That decomposition rewrites trilinear/nearest-3d upsampling, replicate padding, instance norm and SDPA
into `index`/`arange`/`where`/`repeat` graphs, which this executor does not take.

`tests/goldens/make_wgsl_goldens.py` contains a ~70-line reference exporter that emits exactly this schema.

### Supported ops (summary; see `vision_ops.json` for attrs and notes)

| family | ops |
|---|---|
| Convolution | `convolution` (incl. transposed), `conv2d`, `conv3d`, `conv_transpose2d`, `conv_transpose3d` |
| Linear algebra | `linear`, `addmm`, `mm`, `bmm`, `matmul`, `scaled_dot_product_attention` |
| Normalisation | `batch_norm` (eval), `_native_batch_norm_legit_no_training`, `instance_norm`, `group_norm`, `native_group_norm`, `layer_norm`, `native_layer_norm` |
| Pooling | `max_pool2d/3d`, `max_pool2d/3d_with_indices`, `avg_pool2d/3d` |
| Upsampling | `upsample_nearest2d/3d`, `upsample_bilinear2d`, `upsample_trilinear3d` |
| Elementwise | `add`, `sub`, `mul`, `div`, `relu(_)`, `leaky_relu(_)`, `gelu`, `sigmoid`, `tanh`, `silu`, `abs`, `neg`, `exp`, `sqrt`, `rsqrt`, `prelu` |
| Reductions | `_softmax`, `softmax`, `argmax`, `mean` |
| Shape / data movement | `view`, `reshape`, `_unsafe_view`, `flatten`, `unsqueeze`, `squeeze`, `contiguous`, `clone`, `alias`, `detach`, `dropout`, `permute`, `transpose`, `t`, `expand`, `select`, `slice`, `pad`, `constant_pad_nd`, `cat` |

## Memory planner and binding-size tiling

- **Buffer reuse.** `VisionModel.plan` holds the plan. The planner computes each value's lifetime over the launch
  list; views share their root's storage. It assigns values to pooled buffers ("slots"), best-fit, allocating a
  value's slot before freeing the dying ones at the same launch, so a kernel's output never aliases its input.
  Graph outputs are never recycled.
  - The plan reports `slots`, `totalBytes`, `naiveBytes`, `peakLiveBytes`, per-value `{slot, def, lastUse}`,
    `tiled` and every binding size.
  - Example: the tiny UNet at 32³ fits its 92 values into 6 slots, 2.1 MiB against 9.8 MiB with one buffer per value.
- **Tiling.** When a single binding would exceed `maxStorageBufferBindingSize`, the compiler tiles. Every strategy
  is also run on the CPU, and the tiled result is **bit-identical** to the untiled one (tested).

  | strategy | when | how |
  |---|---|---|
  | `spatial-slab` | conv, convT, pool, upsample | Split the outermost spatial axis. A copy gathers the input slab plus halo into a scratch buffer, the kernel runs on the slab, and a copy scatters the output slab back. |
  | `flat` | elementwise, copy, pad | Split into 256-byte-aligned windows. |
  | `rows` | norm, matmul (A/out rows per batch), attention (per batch×head), row-contiguous reductions | Windowed row ranges. |
  | `inner-chunk` | softmax/argmax/mean over a middle dim of a big tensor (e.g. channel softmax of a whole volume) | Gather `[outer, S, chunk]` tiles with copies. |

- **Refusals.** The compiler **refuses with an error naming `maxStorageBufferBindingSize`** when one irreducible
  unit cannot fit a binding:
  - a single weight tensor;
  - one norm row, i.e. one (sample, group) for group/instance norm, whose size is a group's channels × the volume;
  - one attention head's q/k/v;
  - a matmul B operand.

  It also refuses when a single value exceeds `maxBufferSize`.

## Sliding-window inference (3D volumes)

`slidingWindowInference(input, roi, predictor, {overlap, mode, sigmaScale, cval})` reproduces
`monai.inferers.sliding_window_inference` in its non-buffered, same-resolution form:

1. Symmetric constant padding when the image is smaller than `roi`.
2. MONAI's `_get_scan_interval` and `dense_patch_slices` window order.
3. `compute_importance_map` (gaussian or constant), in fp32.
4. The count map is accumulated first, then the weighted predictions in window order.
5. Divide by the count map, then crop.

It matches MONAI 1.6 to **≤ 5.1e-7**, on both the CPU reference and WGSL. The goldens use a 3D BasicUNet predictor,
40×36×44 and 24×40×34 volumes, and a 32³ ROI.

`sw_batch_size` is not used: windows run one at a time. Multi-resolution predictor outputs are rejected.

## Worker protocol (`vision_*` ops on the sealed `model` RPC)

Stage `graph.json` and `model.safetensors` with the existing `push_begin` / `push_chunk` / `push_end` path, under the
model id.

| op | payload | reply |
|---|---|---|
| `vision_caps` | `{}` | `{webgpu, backend, f16, ops[], limits, training:false, scope}` |
| `vision_load` | `{id, graph?, f16?, maxBindingBytes?}` (graph inline or staged) | `{id, backend, f16, inputs, outputs, plan}` |
| `vision_plan` | `{id}` | `{plan}` |
| `vision_infer` | `{id, inputs: {name: {shape, b64}}}` (b64 of little-endian f32) | `{outputs: {name: {shape, b64}}, ms, backend}` |
| `vision_sliding_window` | `{id, input:{shape,b64}, roi?, overlap?, mode?, sigma_scale?, cval?, output?, input_name?}` | `{output:{shape,b64}, windows, ms}` |
| `vision_unload` | `{id}` | `{ok}` |

- **Capability.** `worker.ts` advertises the `vision` capability in `register.node.caps` **only** when it has a
  WebGPU device (a `navigator.gpu` adapter was obtained) **and** the vision module loaded.
- **CPU fallback.** Without a device, the ops still work on the CPU reference. That is useful for testing but slow,
  so the coordinator should not route production vision work to such workers.
- **Model limit.** A worker keeps at most 4 loaded models (LRU).
- **Lazy import (distribution).** `worker.ts` imports `vision_wgsl.ts` **lazily** (`import()` with `.catch`).
  - `scripts/install.sh` fetches and signature-verifies **only** `worker.ts`. A static import would make every
    install.sh-provisioned worker fail at startup.
  - With the lazy import, such a worker starts normally and just lacks `vision`.
  - `moregpu join` (raw URL), the repo layout and the browser bundle (`deno bundle`) all include the module.
  - To ship vision through the signed installer, sign `vision_wgsl.ts` as a second artifact and have `install.sh`
    fetch and verify it. **`worker.ts.sig` must be re-signed at release**, because `worker.ts` changed.

## Running the tests

```bash
python3 tests/goldens/make_wgsl_goldens.py        # regenerate goldens (torch 2.x, monai, safetensors)
npx vitest run tests/webgpu                        # CPU reference vs PyTorch/MONAI goldens, planner, dispatch
deno test --unstable-webgpu --allow-read tests/webgpu/deno_webgpu_test.ts             # real WGSL (ignored w/o adapter)
deno test --unstable-webgpu --allow-read --allow-net --allow-run --allow-env tests/webgpu/deno_worker_e2e_test.ts
npx playwright test -c tests/webgpu/playwright.config.ts                              # browser (skipped w/o WebGPU)
```

- **Linux without a GPU:** `apt install mesa-vulkan-drivers` gives Deno/wgpu a Vulkan software adapter (lavapipe,
  with `shader-f16`). Headless Chromium gets SwiftShader WebGPU from the flags in `playwright.config.ts`. WebGPU needs
  a secure context, so the spec serves the page on `http://127.0.0.1`.
- **Skip path:** `MOREGPU_WEBGPU=0` launches Chromium without the WebGPU flags and exercises the skip.

## What was verified in this change (2026-09-23)

- **vitest:** 136 new tests (the suite goes from 137 to 273). They run the CPU reference against 98 per-kernel PyTorch goldens (67 training-IR + 31
  core-ATen variants), check f16 mirrors, UNet/seg2d/ViT parity, MONAI sliding-window parity, and the planner and
  tiling (bit-identical), and cover the dispatch op family and the worker wiring.
- **Deno WebGPU on a real software adapter:** Mesa lavapipe (`llvmpipe (LLVM 20.1.2, 256 bits)`, f16), Deno 2.9.6.
  All 9 tests pass: every kernel and f16 variant compiles, the 98 kernel goldens pass on the GPU, tiled launches
  pass at 1 KiB and 512 B binding limits, UNet/seg2d/ViT parity holds, UNet passes under a 256 KiB limit, f16 weights
  pass, and sliding-window matches MONAI.
- **Deno end-to-end:** the real `worker.ts` against a fake coordinator does push → `vision_load` → `vision_infer`
  over the sealed RPC. It passes both on lavapipe (worker advertises `vision`, serves on `webgpu`) and with
  `MOREGPU_FORCE_CPU=1` (no `vision` cap, CPU reference).
- **Playwright:** headless Chromium 1194 with **SwiftShader** WebGPU (Dawn/Tint), 114/114. It found a real bug that
  naga had accepted: a `-3.40282347e38` literal that Tint rejects as not representable in f32. The runner now wraps
  pipeline creation and submission in WebGPU error scopes, so validation errors throw instead of returning zeros.
- **Not verified here:** discrete GPUs (Metal/D3D12/Vulkan hardware), and performance. The software adapters only
  prove correctness.

## Limits and known gaps

- No `attn_mask` in SDPA; head_dim ≤ 256; `enable_gqa` unsupported.
- `ceil_mode=true` pooling, string conv padding (`'same'`) and `div` rounding modes are unsupported.
- Tensor rank ≤ 6 for broadcast, copy and pad. `mean` needs contiguous reduced dims.
- Static shapes. Dynamic batch means one compiled model per batch size.
- The WGSL is correct first and not tuned. There is no subgroup or cooperative-matrix use, and the conv tile is
  16×16. The CPU reference is a test oracle and is not meant for production throughput.
- Blending in the sliding-window driver runs on the host (JS). Only the predictor runs on the GPU.
- The coordinator side (`/vision/*` routes and mixed-fleet dispatch of `vision_*` to workers with the `vision`
  capability) is **not** part of this change.

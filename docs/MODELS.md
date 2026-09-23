# Running published models

MoreGPU runs vision models the way their authors published them: the published weights, in the published architecture,
loaded strictly. Nothing is retrained or converted in secret. Some formats would need the worker to run code from the
artefact itself. MoreGPU refuses those formats. See [ADR-0113](dev/adr/0113-model-formats.md) and
[ADR-0114](dev/adr/0114-lowering-and-webgpu-vision.md).

## Formats and precedence

When a model is available in several forms, use the first one in this list that fits:

| # | `format` | Loaded with | Trainable | Notes |
|---|----------|-------------|-----------|-------|
| 1 | `state_dict` | `torch.load(path, map_location="cpu", weights_only=True)` | yes | `.pt` / `.pth`. You must name an architecture (`arch`). |
| 1 | `safetensors` | `safetensors.torch.load_file` | yes | Needs `arch`. Preferred over `state_dict` when both exist. |
| 2 | `plugin` | an allowlisted `moregpu.models` entry point | yes | For architectures that no registry covers (see below). |
| 3 | `torch_export` | `torch.export.load` after a pickle pre-check | no | `.pt2` from `torch.export.save`. |
| 3 | `torchscript` | `torch.jit.load` | no | `torch.jit.save` output. |
| 4 | `onnx` | onnxruntime (`CUDA` → `CoreML` → `CPU`, whichever are available) | no | |

The worker builds architectures **by name** from these registries, and never downloads their pretrained weights:

| `arch.registry` | Built with |
|-----------------|------------|
| `torchvision` | `torchvision.models.get_model(name, weights=None, **kwargs)` |
| `timm` | `timm.create_model(name, pretrained=False, **kwargs)` |
| `monai` | `monai.networks.nets.<Name>(**kwargs)` |
| `hf` | `transformers.<AutoClass>.from_config(AutoConfig.for_model(**kwargs.config))`, using an Auto class such as `AutoModelForImageClassification`. The worker never calls `from_pretrained` and never sets `trust_remote_code`. |
| `plugin` | an allowlisted entry point: `build(**kwargs) -> nn.Module` |

### Checkpoint unwrapping

Before a strict load, the worker unwraps the common checkpoint containers:

- the keys `state_dict`, `model_state_dict`, `model`, `module`, `net` and `network`, up to four levels deep;
- the key prefixes `module.` (DataParallel/DDP) and `_orig_mod.` (`torch.compile`).

Each step is reported in `describe()["unwrap"]`, for example `["key:state_dict", "prefix:module."]`.

The load uses `strict=True`. If it fails, the worker raises `KeyMismatch`. The error lists the **missing** keys, the
**unexpected** keys and every **shape mismatch** (checkpoint shape vs model shape). Use this diff to fix
`arch.name` or `arch.kwargs`.

## Model spec v1

The schema is [`docs/model-spec.schema.json`](model-spec.schema.json). It is generated from
`moregpu_worker.vision.spec.SCHEMA` (`python3 -m moregpu_worker.vision.spec`), and a test keeps the two identical.

```json
{
  "version": 1,
  "format": "safetensors",
  "source": "https://example.org/unet.safetensors",
  "sha256": "<64 hex>",
  "arch": {"registry": "monai", "name": "UNet",
           "kwargs": {"spatial_dims": 3, "in_channels": 1, "out_channels": 2,
                      "channels": [16, 32, 64, 128], "strides": [2, 2, 2], "num_res_units": 2}},
  "dtype": "float32",
  "io": {"inputs":  [{"name": "image",  "shape": [null, 1, 96, 96, 96], "dtype": "float32"}],
         "outputs": [{"name": "logits", "shape": [null, 2, 96, 96, 96]}]},
  "preprocess": {},
  "inference": {"mode": "sliding_window", "tta": "flip",
                "sliding_window": {"roi": [96, 96, 96], "overlap": 0.5, "blend": "gaussian", "sw_batch": 4}},
  "postprocess": {"activation": "softmax", "argmax": true},
  "placement": {"device": "auto", "min_vram_gb": 4},
  "licence": "Apache-2.0",
  "citation": "Author et al., Year"
}
```

| Field | Rules |
|-------|-------|
| `format` | One of the six formats above. |
| `source` | One of:<br>• `hf://org/repo[@rev]/file`: downloads one file through `huggingface_hub`.<br>• `https://…`: **requires `sha256`**. The download is cached by content under `MOREGPU_MODEL_CACHE`.<br>• `pushed://<id>`: a blob the coordinator pushed into `MOREGPU_PUSHED_DIR`. **Requires `sha256`**.<br>• `file:///path`: allowed only under `MOREGPU_MODEL_ROOTS` (`os.pathsep`-separated). Symlinks are resolved first.<br>Every format except `plugin` requires a `source`. |
| `sha256` | The worker hashes every artefact and checks it **before opening it**. On a mismatch it refuses with `IntegrityError`. |
| `arch` | Required for `state_dict`, `safetensors` and `plugin`. For `plugin`, `registry` must be `"plugin"`. |
| `dtype` | `float32` (default), `float16` or `bfloat16`. Applies to native modules. |
| `inference` | `mode`: `full` (default) or `sliding_window`. `sliding_window` needs `roi` and uses MONAI's sliding-window inferer, with defaults `overlap` 0.25, `blend` `gaussian` and `sw_batch` 1.<br>`tta`: `none` or `flip`. `flip` averages the prediction on the unflipped input with the prediction for each single-axis spatial flip. |
| `preprocess`, `postprocess` | Free-form objects. The worker keeps them and reports them. They are applied by the `/vision` inference path. |
| `placement` | `device`: `auto`, `cpu`, `cuda` or `mps`. `min_vram_gb` is a scheduling hint. |
| `licence`, `citation` | Reported by `describe()`. Keep the author's licence and citation with their model. |

## Refusal policy

The worker refuses (`RefusedFormat`) any artefact that would require unpickling arbitrary objects. This includes:

- a pickled full model saved with `torch.save(model)`;
- a checkpoint containing any non-tensor object that `weights_only=True` rejects;
- a malicious pickle. It is refused **before** any of its code runs.

The error message says what to do instead. Export a `state_dict` or safetensors file together with a named
architecture, or export with `torch.export` or to ONNX. The other option is to ask the worker admin to install an
allowlisted plugin.

`torch.export.load` is **not** safe on its own for untrusted input. A `.pt2` archive can mark payloads as `use_pickle`,
carry opaque or custom-object constants, or ship sample inputs that PyTorch loads with a silent fallback to
`weights_only=False`. The worker therefore checks every archive with `check_pt2` before PyTorch reads it. The check
refuses any `.pt2` that has:

- pickled payloads;
- non-tensor constants;
- an embedded `torch.save`/pickle member that does not load with `weights_only=True`.

A repo-wide test (`tests/security/pickle_ban_test.py`) bans these patterns from the code:

- `weights_only=False`;
- `torch.load(` without `weights_only=True`;
- bare `pickle.load(s)` / `Unpickler`;
- `trust_remote_code=True`;
- in `apps/worker/**`, a weights-loading `from_pretrained(` without `use_safetensors=True`.

Reviewed exceptions go in `tests/security/pickle_ban_allow.txt`. An exception that no longer matches any code fails
the test.

## Plugins (admin-installed, pinned)

Code never travels over the wire. The coordinator can only **name** a plugin that is already installed on the worker.
To install one:

1. Build or obtain the plugin wheel. It declares
   `[project.entry-points."moregpu.models"] my_net = "my_pkg.models:build"`, where
   `build(**kwargs) -> torch.nn.Module`.
2. Install the wheel with its hash pinned, so that pip records the archive hash in `direct_url.json`:
   ```sh
   echo "my-models @ file:///opt/wheels/my_models-1.2.0-py3-none-any.whl --hash=sha256:<wheel sha256>" > req.txt
   pip install --require-hashes -r req.txt
   ```
3. Add the plugin to the worker allowlist. The allowlist is a JSON file, and the `MOREGPU_MODEL_PLUGIN_ALLOWLIST`
   environment variable points to it. This allowlist is separate from `MOREGPU_PLUGIN_ALLOWLIST`, which covers training
   tasks.
   ```json
   [{"dist": "my-models", "version": "1.2.0", "wheel_sha256": "<wheel sha256>"}]
   ```
4. Restart the worker. `vision_models_describe` lists the plugin under `plugins.available`. If the plugin is refused,
   the same call reports why under `plugins.refused`. A plugin is refused when:
   - it is not in the allowlist;
   - its installed version is not the pinned one;
   - its wheel sha256 is missing or does not match.

A spec can use the plugin either as `{"format": "plugin", "arch": {"registry": "plugin", "name": "my_net"}}`, where
the plugin supplies its own weights, or as a `state_dict`/`safetensors` spec with `arch.registry: "plugin"`, where the
published weights are loaded strictly into the plugin's architecture.

## Lowering and the parity guarantee

`moregpu_worker.vision.lowering.lower(handle, target)` prepares a native model for workers that do not run torch:

1. `torch.export.export` traces the model with an example input, in its **default dialect**. The core-ATen
   decomposition is **not** run (it would turn trilinear upsampling, replicate pad, instance norm and SDPA into
   `index`/`where` graphs). The input comes from `spec.io.inputs[0].shape` or from an explicit `example`.
2. For target `wgsl`: if every op maps onto the WGSL executor's table (`WGSL_OPS`, read from
   `apps/worker/vision_ops.json`), the result is an op-graph JSON plus safetensors weights (`kind: "opgraph"`) in
   **exactly the executor's schema** (docs/WEBGPU_VISION.md): `{version, inputs:[{name,shape}], nodes:[{op, inputs,
   attrs, output}], outputs, weights: "model.safetensors"}`, with tensor arguments in ATen-schema order and every other
   argument in `attrs` under its ATen schema name. The lowering also:
   - maps `getitem(node, 0)` of a multi-output op (`native_layer_norm`, …) onto the node's output, and refuses a use of
     any other output;
   - rewrites `unbind`/`split`/`chunk` + `getitem` into `select`/`slice` nodes;
   - renames an in-place op the executor lacks to its functional twin (`add_` → `add`), but only when nothing that
     shares the mutated tensor's storage is read afterwards;
   - spells non-finite attrs `"Infinity"`, `"-Infinity"`, `"NaN"` (strict JSON).
   `opgraph_ref.py` executes the same schema by calling each ATen overload. It is the parity oracle.
   `tests/webgpu/lowering_parity.test.ts` runs python-lowered graphs (a 3D BasicUNet, a ViT+conv segmenter, a
   ViT-Tiny-width encoder) through the TS executor and requires ≤ 1e-5 rel against PyTorch.
3. Otherwise, or for target `onnx-web`: the model is exported to ONNX in memory for onnxruntime-web's WebGPU execution
   provider (`kind: "onnx"`). With target `wgsl`, `unsupported_ops` lists the ops that forced this fallback.
4. If ONNX export fails as well, the result is `kind: "native"` and `servable: false`, and `unsupported_ops` lists the
   ops that blocked lowering (for example `aten.fft_rfft2.default`).

**Parity guarantee.** Before a lowered artefact can be served, it runs on the example input and its output is compared
with the native forward pass. In fp32 the maximum absolute difference must be ≤ 1e-4. If it is not, the worker refuses
to serve the artefact: `servable: false`, and `run()` raises `ParityRefused`. Results are cached by
`sha256(model sha256 | lowering version | target | example shape)`, both in memory and optionally on disk. The worker
probes a cached artefact again when it loads it from disk, so a tampered cache is never served.

## Support matrix (honest)

| Worker type | What it can run today |
|-------------|-----------------------|
| Native torch worker (`worker_torch.py`, CUDA/MPS/CPU) | **Everything above**: all six formats, sliding-window/TTA inference, training on native handles, and lowering. |
| Deno WebGPU worker / browser tab | **Lowered op-graphs** (inference and JEPA features), via the WGSL vision executor (`vision_wgsl.ts`, M6). The coordinator lowers on a torch worker and pushes the artefact (`/vision/load {fleet: 'webgpu'\|'all'}`, docs/WEBGPU_VISION.md). |
| Browser via onnxruntime-web (WebGPU EP) | Designed, not yet wired. It would run `kind: "onnx"` artefacts (lowered, or ONNX as published) and use the same parity-probed bytes. |

Training stays on native workers. On WebGPU, vision is limited to inference and JEPA feature extraction (ADR-0114).

## Worker RPC

`moregpu_worker.vision.ops.handle(op, payload)` supports these ops:

| Op | Payload | Returns |
|----|---------|---------|
| `vision_models_describe` | none | The worker's formats, registries, plugins, lowering targets, WGSL op table and loaded models. |
| `vision_load` | `{id, spec}` | Loads the model under `id`, replacing any model already loaded there. |
| `vision_describe` | `{id}` | A description of the loaded model. |
| `vision_lower` | `{id, target, example_shape?, include_bytes?}` | The lowering report, including `io` (graph inputs/outputs). With `include_bytes`, also the graph JSON and base64 weights, or the base64 ONNX bytes. It also lowers a model loaded from a MoreGPU export through `vision_infer_load` (via `ops.register_resolver`). |
| `vision_unload` | `{id}` | Unloads the model. |

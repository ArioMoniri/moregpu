# Vision

MoreGPU runs vision models on three kinds of workers:
- **native torch workers** (`apps/worker/worker_torch.py`, CUDA / MPS / CPU);
- **Deno WebGPU workers** (`apps/worker/worker.ts`);
- **browser tabs** (the same `worker.ts`, bundled).

The subsystems are documented separately:

| Document | Covers |
|---|---|
| this file | the matrix of what works where, plus the data plane |
| [MODELS.md](MODELS.md) | running published models exactly as released |
| [TRAINING.md](TRAINING.md) | training sessions and DiLoCo |
| [JEPA.md](JEPA.md) | self-supervised pretraining |
| [WEBGPU_VISION.md](WEBGPU_VISION.md) | the WGSL executor |
| [TELEMETRY.md](TELEMETRY.md) | what is measured |

## Support matrix

Every ✅ cell has an automated test (unit, e2e or real-adapter), named in the last column.

| Capability | Native torch worker | Deno WebGPU worker | Browser tab | Evidence (tests) |
|---|---|---|---|---|
| **Load a published model exactly as released**: state_dict / safetensors + named arch (torchvision, timm, MONAI, HF), allowlisted plugins, torch.export, TorchScript, ONNX | ✅ sha256-verified; pickled full models refused | via lowering only (below) | via lowering only | `tests/py/test_vision_adapters.py`, `tests/e2e/published_model.py` |
| **Automatic lowering** (torch.export graph → MoreGPU op-graph; or ONNX for onnxruntime-web) | ✅ produces it; parity-probed ≤ 1e-4 | consumes it | consumes it | `tests/py/test_vision_lowering.py`, `tests/webgpu/*` |
| **Inference: one tensor** | ✅ | ✅ conv2d/3d, norms, pooling, upsampling, attention, … (see WEBGPU_VISION.md) | ✅ (verified on SwiftShader) | `tests/py/test_vision_infer.py`, `tests/webgpu/deno_webgpu_test.ts`, `browser.spec.ts` |
| **Whole-volume inference** (sliding window, gaussian blend, flip TTA; MONAI-equivalent) | ✅ | ✅ (host-side blend) | ✅ | `tests/py/test_vision_core.py` (MONAI goldens) |
| **Distributed batch inference**: case queue with work stealing, retry on churn | ✅ | via mixed-fleet `/vision/infer_batch` | same | `apps/coordinator/lib/vision_batch.test.ts`, `tests/e2e/vision_pipeline.py` |
| **Tile sharding**: one volume split across workers, merged exactly | ✅ (`split: "tiles"`) | — | — | `tests/py/test_vision_infer.py`, `tests/e2e/vision_pipeline.py` |
| **JEPA pretraining** (`ijepa_2d`, `jepa_2p5d`, `jepa_3d`) | ✅ DiLoCo across N workers | ❌ no autograd in WGSL | ❌ | `tests/py/test_jepa_*.py`, `tests/e2e/jepa_sessions.py` |
| **JEPA feature extraction** (frozen encoder) | ✅ | ✅ (lowered ViT) | ✅ | `tests/py/test_vision_infer.py`, `tests/webgpu/*` |
| **Fine-tune MoreGPU models** (`segment` 2D/2.5D/3D, `classify`; full / frozen / LoRA) | ✅ DiLoCo | ❌ | ❌ | `tests/py/test_vision_tasks.py`, `tests/e2e/vision_pipeline.py` |
| **Fine-tune ANY published native model** (`finetune_model`: torchvision / timm / MONAI / HF / plugin; all / head / LoRA; segment, classify, regress) | ✅ DiLoCo | ❌ | ❌ | `tests/py/test_finetune_model.py` |
| **Data plane**: file:// under roots, https / public buckets with sha256, pushed:// blobs; NumPy, NIfTI, DICOM, PNG/JPEG/TIFF | ✅ | pushed tensors only | pushed tensors only, kept in memory | `tests/py/test_data_*.py`, `tests/e2e/data_plane_jepa.py` |
| **Telemetry** (compute/data/serialise/network/wait, bytes, GPU util, energy, AMP) | ✅ | job-level | job-level | `tests/py/test_telemetry_*.py` |

### Why training is native-only

Training needs autograd plus optimiser state. The WGSL executor only runs the forward pass, which is enough for inference and
feature extraction. Adding WGSL backward kernels is listed as a stretch item in `docs/dev/adr/`. In a mixed fleet, the
WebGPU and browser workers therefore help with inference and evaluation, while the torch workers train.

### How the swarm parallelises vision work

| Kind of work | How it is split | Why |
|---|---|---|
| Training (pretraining and fine-tuning) | **Data-parallel DiLoCo** | Each worker holds the whole model and trains on its own seeded shard for H local steps. The coordinator then averages, weighted by samples, and applies an outer Nesterov step. The only traffic is one state transfer per round. |
| Inference | **By data** | Many studies go through a work-stealing case queue. A single large study can be split tile-by-tile across workers (`split: "tiles"`), which gives exactly the single-node result. |
| — | Not by layers | Pipeline or layer sharding, as used for LLM/MoE models (`/model/shard`), is only needed when a model does not fit on one device. Vision models of the sizes above fit easily. |


## Data plane

The data plane is how a native torch worker gets training arrays. It lives in `apps/worker/moregpu_worker/data/`
([ADR-0110](dev/adr/0110-vision-data-plane.md)). Every access goes through the **donor's** policy. The coordinator
can name data, but it cannot widen what a worker is allowed to read.

### Refs

A ref is one JSON object, and a manifest is a `refs.jsonl` file with one ref per line:

```json
{"uri": "file://scans/0001/volume.npy", "sha256": "…64 hex…", "slice": [10, 13], "meta": {"fmt": "npy"}}
```

| Field | Meaning |
|---|---|
| `uri` | `file://`, `https://`, `s3://`, `gs://` or `pushed://<id>`. Any other scheme, and a bare path, is refused. |
| `sha256` | Optional for `file://` and `pushed://`, where it is verified if given. **Required** for `https://`, `s3://` and `gs://`. |
| `slice` | `[start, stop)` on axis 0, applied after reading. `.npy` is memory-mapped, so only those slices are read. |
| `meta` | Free-form. `meta.fmt` overrides format detection (`npy`, `npz`, `nifti`, `dicom`, `image`, `tiff`). For `.npz` it can also be an array key. |

A manifest's `sha256` is computed over its canonical JSONL: keys sorted, compact separators, defaults omitted, one
`\n` per line. The same refs therefore hash the same whatever the formatting of the source. `DataPlane.open_manifest`
caches manifests per `(uri, sha256)`.

### Policy (environment of the worker)

| Variable | Effect |
|---|---|
| `MOREGPU_DATA_ROOTS` | Local directories that may be read, separated by `os.pathsep` (`:` on Unix). Relative `file://` URIs resolve against the **first** root. With no roots, every `file://` ref is refused. |
| `MOREGPU_DATA_HOSTS` | Comma-separated hostnames (or `host:port`) that `https://` refs may download from. Matching ignores case. |
| `MOREGPU_DATA_BUCKETS=1` | Allows public `s3://` / `gs://` reads, but only when `s5cmd` / `gsutil` is on `PATH`. |
| `MOREGPU_DATA_MAX_DOWNLOAD_BYTES` | Per-download cap. The default is 20 GiB. |
| `MOREGPU_CACHE_DIR`, `MOREGPU_CACHE_BYTES` | Download cache location (default `~/.cache/moregpu/data`) and byte cap (default 20 GiB). |
| `MOREGPU_STAGE_DIR`, `MOREGPU_PUSH_MAX_BYTES` | Staging directory for pushed blobs and their size cap (default 20 GiB). |
| `MOREGPU_BLOB_TOTAL_MAX_BYTES` | Cap on all staged blobs together (default 40 GiB). |

### Readers

Each worker reports the readers it can use as `data_caps.readers`:

| Format | Library | Output |
|---|---|---|
| `.npy` | NumPy (always available) | Read-only memory map |
| `.npz` | NumPy | The first array, or the array named by `meta.fmt` |
| `.nii`, `.nii.gz` | `nibabel` | `(Z, Y, X[, …])`. `read_nifti` also returns the 4×4 affine exactly as stored; it maps `(x, y, z)` voxel indices. |
| DICOM file or series directory | `pydicom` (falls back to `SimpleITK`) | `(Z, Y, X)` sorted by `ImagePositionPatient` along the slice normal, with `RescaleSlope`/`RescaleIntercept` applied. |
| `.png`, `.jpg`, … | `pillow` | `(H, W)` or `(H, W, C)` |
| `.tif`, `.tiff` | `tifffile` | As stored |

All of the optional readers come with `pip install 'moregpu-worker[vision]'`.

`DataPlane.load_batch(manifest, indices, spec)` turns refs into a `(B, …)` tensor:
- **`2d`**: the image becomes `(C, H, W)`. A trailing axis of 4 or fewer is treated as channels-last. A grey image is broadcast when `channels > 1`.
- **`2p5d`**: the ref's slice window gives the `channels` adjacent slices as `(C, H, W)`.
- **`3d`**: the volume becomes `(1, D, H, W)`.

Resizing uses `torch.nn.functional.interpolate` (bilinear or trilinear, `align_corners=False`). After that come the
optional `normalize: {mean, std}` (a scalar or one value per channel) and the output `dtype`. The result is deterministic.

### Cache

`https://` and bucket downloads land in a content-addressed cache: files named by their sha256, with a byte cap and
LRU eviction. Recency is the file mtime, so the LRU order survives restarts. Content is hashed before it enters the
cache, and a mismatch is refused without being stored. A cached file is reused for any ref with the same sha256, so a
file is downloaded once per worker.

### Pushed blobs (`pushed://<id>`)

The coordinator can stream data it holds straight to a worker, so the worker needs no data access of its own.

| Op | Effect |
|---|---|
| `blob_begin{id, sha256, size, suffix?}` | Starts a blob. Beginning an existing id again discards its partial data. |
| `blob_chunk{id, k, data(base64)}` | Adds chunk `k`. Chunks must come strictly in order `k = 0, 1, …`; a skipped or replayed `k` is refused. |
| `blob_end{id}` | Checks the declared size and sha256. Only then is `pushed://<id>` readable. |
| `blob_drop{id}` | Deletes the staged file. |

A blob that goes over its declared size, or fails the check at the end, is deleted.

Staging uses RAM first: `MOREGPU_STAGE_DIR` (alias `MOREGPU_PUSHED_DIR`) if set, else `/dev/shm` when it has at least
2 GiB free and room for the blob, else the OS temp dir. The temp dir is usually on **disk**, so a blob staged there
does touch persistent storage while it is staged. The worker deletes staged files itself: `blob_drop` deletes one, a
coordinator `welcome` and normal process exit delete all of them. A worker that is killed (SIGKILL, crash, power loss)
can leave `moregpu-blob-*` files in a disk staging dir. Set `MOREGPU_STAGE_DIR` to a tmpfs if blobs must never reach
disk.

Caps: one blob is at most `MOREGPU_PUSH_MAX_BYTES` (default 20 GiB), and all staged blobs together are at most
`MOREGPU_BLOB_TOTAL_MAX_BYTES` (default 40 GiB). `blob_begin` is also refused when the staging filesystem has less
free space than the blob's declared size.

The same store also serves `pushed://` **model** sources (docs/MODELS.md), so `/data/push` can deliver a model
artefact as well as data.

### Security properties

- `file://` refs are checked on the **realpath**: symlinks and `..` are resolved before the root check. A symlink or
  `..` that escapes the roots is refused, and so is a sibling directory that only shares a prefix with a root.
- `https://` refs must use an allowlisted host. Every redirect hop is checked against the same allowlist, and
  userinfo tricks such as `allowed@other` are refused. Plain `http://` is accepted only for allowlisted loopback hosts,
  for local mirrors and tests. Integrity always comes from the required sha256.
- Bucket reads are always anonymous. `s5cmd` runs with `--no-sign-request`, `gsutil` gets an empty config, and AWS
  and Google credential variables are removed from the tool's environment. Bucket names are validated, and the tool
  is run without a shell. A ref names exactly one object: URIs with wildcard or glob characters (`*`, `?`, `[`, `]`,
  `{`, `}`) are refused. The object is streamed with `s5cmd cat` / `gsutil cat`, and the tool is killed as soon as the
  download passes `MOREGPU_DATA_MAX_DOWNLOAD_BYTES`.
- Every refusal raises `RefDenied` (a `PermissionError`). A hash or size mismatch raises `IntegrityError`.
- Worker replies (`data_caps`, `data_stats`, `blob_*`) contain counts, hashes and sizes, but never local paths.

### Browser workers

Browser and Deno workers (`worker.ts`) have no data roots, no download cache and no bucket access. Pushed data is kept
in memory only and is never written to persistent storage (IndexedDB, OPFS, Cache API or localStorage). When the job
ends or the tab closes, the data is gone.

### Fast path

`write_shards` writes pre-tiled `.npy` shards (default `float16`) and an `index.json` holding each shard's sha256.
`ShardDataset` memory-maps the shards lazily in each loader process, and can verify the hashes first.
`make_loader` builds a `DataLoader` with a seeded generator, so the sample order depends only on `seed`. It uses
pinned memory when CUDA is present, persistent workers and prefetch. `measure_throughput` reports samples/s.

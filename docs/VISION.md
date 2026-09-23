# Vision

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

Staging uses RAM first: `MOREGPU_STAGE_DIR` if set, else `/dev/shm` when it has at least 2 GB free, else the OS temp
dir. Blobs are never persisted. `blob_drop` deletes the staged file, and all remaining blobs are deleted at process
exit.

### Security properties

- `file://` refs are checked on the **realpath**: symlinks and `..` are resolved before the root check. A symlink or
  `..` that escapes the roots is refused, and so is a sibling directory that only shares a prefix with a root.
- `https://` refs must use an allowlisted host. Every redirect hop is checked against the same allowlist, and
  userinfo tricks such as `allowed@other` are refused. Plain `http://` is accepted only for allowlisted loopback hosts,
  for local mirrors and tests. Integrity always comes from the required sha256.
- Bucket reads are always anonymous. `s5cmd` runs with `--no-sign-request`, `gsutil` gets an empty config, and AWS
  and Google credential variables are removed from the tool's environment. Bucket names are validated, and the tool
  is run without a shell.
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

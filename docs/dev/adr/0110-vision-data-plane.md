# ADR-0110 — Vision data plane

**Status:** Proposed · **Milestone:** M2

## Decision
- **Refs** (`Ref = {uri, sha256?, slice?, meta?}`): `file://` restricted to `MOREGPU_DATA_ROOTS` (realpath check, no
  symlink escape); `https://` only to `MOREGPU_DATA_HOSTS`; `s3://`/`gs://` public buckets via `s5cmd`/`gsutil` if present
  (anonymous only); `pushed://<id>` via the generic blob push (sealed chunks, sha256 verified, RAM-staged).
- **Readers** as optional extras advertised as per-worker capabilities: numpy/memmap (built-in), NIfTI (`nibabel`),
  DICOM (`pydicom`/`SimpleITK`), PNG/JPEG/TIFF (`pillow`/`tifffile`).
- **Cache:** content-addressed (sha256) on disk with byte cap + LRU; browser workers keep pushed data in memory only.
- **Fast path:** pre-tiled `.npy` shards + index; `torch.utils.data.DataLoader` with pinned memory, persistent workers,
  prefetch; throughput reported in telemetry (`data_load` time).
- Per-worker capabilities (`readers`, `data_roots` count, cache size) exposed via a new `/workers/:id/caps` and in the
  dashboard; `/device` gains a union summary.

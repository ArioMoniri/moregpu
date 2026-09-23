#!/usr/bin/env python3
"""Distributed whole-volume segmentation over the pool (work-stealing queue, churn retry, flip TTA).

  python3 examples/vision_batch.py --url ... --token ... --model-dir /path/on/workers/segmenter \
      --volumes vol0.npy vol1.npy ... [--masks gt0.npy ...]
Volume paths are relative to the workers' MOREGPU_DATA_ROOTS; predictions land in MOREGPU_OUTPUT_DIR on the worker."""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clients", "python"))
from moregpu import MoreGPU  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.environ.get("MOREGPU_URL", "http://localhost:8787"))
    p.add_argument("--token", default=os.environ.get("MOREGPU_ADMIN_TOKEN", ""))
    p.add_argument("--model-dir", required=True)
    p.add_argument("--volumes", nargs="+", required=True)
    p.add_argument("--masks", nargs="*", default=[])
    p.add_argument("--tta", default="flip")
    a = p.parse_args(argv)
    pool = MoreGPU(a.url, a.token)
    pool.vision_load("seg", a.model_dir)
    items = [{"ref": {"uri": f"file://{v}"}, "out": f"pred_{os.path.splitext(os.path.basename(v))[0]}",
              **({"mask": {"uri": f"file://{a.masks[i]}"}} if i < len(a.masks) else {})} for i, v in enumerate(a.volumes)]
    job = pool.vision_batch("seg", items, tta=a.tta, steal_after_ms=2000)
    res = pool.vision_wait(job["job"], poll_s=0.5)
    print({k: res[k] for k in ("status", "done", "failed", "retries", "stolen", "per_worker")})
    return res


if __name__ == "__main__":
    main()

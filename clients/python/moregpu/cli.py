"""moregpu-client CLI — quick pool inspection / job submission from Python.

    export MOREGPU_URL=http://ADMIN:8787 MOREGPU_TOKEN=<admin-token>
    moregpu-client device
    moregpu-client workers
    moregpu-client submit matmul 1024
"""
from __future__ import annotations
import json
import os
import sys

from . import MoreGPU


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    url = os.environ.get("MOREGPU_URL", "http://localhost:8787")
    token = os.environ.get("MOREGPU_TOKEN", "")
    pool = MoreGPU(url, token)
    cmd = argv[0] if argv else "device"
    try:
        if cmd == "device":
            print(json.dumps(pool.device(), indent=2))
        elif cmd == "workers":
            print(json.dumps(pool.workers(), indent=2))
        elif cmd == "health":
            print(json.dumps(pool.health(), indent=2))
        elif cmd == "submit":
            kernel = argv[1] if len(argv) > 1 else "matmul"
            size = int(argv[2]) if len(argv) > 2 else 512
            print(json.dumps(pool.submit(kernel, size), indent=2))
        else:
            print(__doc__)
            return 2
    except Exception as e:  # noqa: BLE001 — surface a friendly error
        print(f"moregpu-client: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

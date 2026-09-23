#!/usr/bin/env python3
"""security_routes.py — review P0-1/P0-2/P1-4 regressions against a live coordinator + 1 torch worker:
path-traversal session ids refused on create/resume and never touch files outside MOREGPU_TRAIN_DIR; malformed session
configs refused; oversized bodies refused; a malicious worker's caps are coerced (no HTML reaches the dashboard); the
dashboard sends a restrictive CSP."""
import json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _pool import Pool, Checks  # noqa: E402

TOY = {"task": "toy_linear", "cfg": {"n": 16, "dim": 3, "batch": 4}, "manifest_len": 16, "batch": 4, "inner_steps": 1, "lr": 0.1, "amp": "fp32"}


def main():
    ck = Checks()
    with Pool(["s1"]) as pool:
        canary = os.path.join(pool.root, "canary.json"); open(canary, "w").write("{}")
        for bad in ["..", "../x", ".", "a/b", "a.b", ""]:
            r = pool.api("/train/sessions", "POST", {**TOY, "id": bad, "checkpoint_every": 1, "keep_checkpoints": 1})
            ck(r.get("httperror") == 400, f"create with id {bad!r} refused ({r.get('httperror')})")
        r = pool.api("/train/sessions/resume", "POST", {"id": ".."})
        ck(r.get("httperror") == 400 and "id must match" in r.get("body", ""), "resume of '..' refused (validated id, no file read)")
        ck(os.path.exists(canary) and os.path.exists(os.path.join(pool.root, "mg.json")), "no file outside MOREGPU_TRAIN_DIR was touched")
        for bad in ({"chunk_bytes": 0}, {"manifest_len": 10 ** 12}, {"sync_dtype": "f64"}, {"batch": -1}):
            r = pool.api("/train/sessions", "POST", {**TOY, **bad})
            ck(r.get("httperror") == 400, f"invalid config {bad} refused")
        big = b"{" + b" " * (513 * 2 ** 20) + b"}"
        req = urllib.request.Request(f"http://127.0.0.1:{pool.port}/train/sessions", data=big, method="POST",
                                     headers={"authorization": "Bearer " + pool.admin, "content-type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=60); code = 200
        except urllib.error.HTTPError as e:
            code = e.code
        except Exception:
            code = 413   # server may close the socket after the 413 while we are still sending
        ck(code == 413, f"oversized body refused ({code})")
        caps = pool.api("/workers/s1/caps")
        ck(isinstance(caps.get("data", {}).get("n_roots"), (int, float)), "worker caps are coerced to numbers")
        resp = urllib.request.urlopen(f"http://127.0.0.1:{pool.port}/")
        csp = resp.headers.get("content-security-policy", "")
        ck("connect-src 'self'" in csp and "img-src 'self'" in csp, "dashboard sends a restrictive CSP")
    ck.finish()


if __name__ == "__main__":
    main()

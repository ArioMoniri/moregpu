#!/usr/bin/env bash
# smoke_torch.sh — verify the NATIVE torch worker joins the pool and computes kernels correctly.
# Starts its own coordinator + a CPU-only torch worker (no model download), runs a few verified
# checks through the SDK, and tears down. Safe for CI (torch CPU only; no GPU, no models).
#   bash scripts/smoke_torch.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8791}"; CFG="./.moregpu-torch-smoke.json"
PIDS=(); rm -f "$CFG"
cleanup(){ kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; rm -f "$CFG"; }
trap cleanup EXIT

echo "== starting coordinator + a CPU torch worker on :$PORT =="
PORT="$PORT" MOREGPU_CONFIG="$CFG" deno run --allow-net --allow-env --allow-read --allow-write \
  apps/coordinator/server.ts >/tmp/torch-smoke-srv.log 2>&1 &
PIDS+=($!)
for i in $(seq 1 40); do curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 0.5; done
JOIN=$(python3 -c "import json;print(json.load(open('$CFG'))['joinToken'])")
ADMIN=$(python3 -c "import json;print(json.load(open('$CFG'))['adminToken'])")
python3 apps/worker/worker_torch.py --server "ws://localhost:$PORT/ws" --token "$JOIN" --name torch-smoke --cpu \
  >/tmp/torch-smoke-w.log 2>&1 &
PIDS+=($!)
for i in $(seq 1 60); do
  N=$(curl -sf -H "authorization: Bearer $ADMIN" "http://localhost:$PORT/workers" 2>/dev/null | python3 -c "import sys,json;print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
  [ "$N" -ge 1 ] 2>/dev/null && break; sleep 0.5
done

MOREGPU_BASE="http://localhost:$PORT" MOREGPU_ADMIN_TOKEN="$ADMIN" python3 - <<'PY'
import os, sys, math
sys.path.insert(0, "examples")
from moregpu_client import MoreGPU
import numpy as np
pool = MoreGPU(os.environ["MOREGPU_BASE"], os.environ["MOREGPU_ADMIN_TOKEN"])
ws = pool.workers()
assert any("torch" in (w.get("label") or "") for w in ws), f"no torch worker in fleet: {ws}"
fails = 0
def check(name, got, want, tol):
    global fails
    d = max(abs(a-b) for a, b in zip(got, want)) if got else 9.9
    ok = d <= tol; print(f"  {'PASS' if ok else 'FAIL'} {name} (maxD={d:.2e})"); fails += 0 if ok else 1
check("matmul", pool.matmul([1,2,3,4,5,6],[7,8,9,10,11,12],2,2,3), [58,64,139,154], 1e-3)
M,K,N = 3,8,5
A = np.random.RandomState(0).randn(M,K).astype("f"); B = np.random.RandomState(1).randn(K,N).astype("f")
ref = (A@B).flatten().tolist()
pool.upload_weight("w", B.flatten().tolist(), rows=K, cols=N)
check("matmul_resident", pool.matmul_resident(A.flatten().tolist(), "w", M), ref, 1e-3)
Q=[0.1,0.2,0.3,0.4]; Kk=[0.5,0.6,0.7,0.8]; V=[1,0,0,1]
Qm,Km,Vm = (np.array(x).reshape(2,2) for x in (Q,Kk,V))
sc=(Qm@Km.T)/math.sqrt(2); sm=np.exp(sc-sc.max(1,keepdims=True)); sm/=sm.sum(1,keepdims=True)
check("attention", pool.attention(Q,Kk,V,seq=2,d=2), (sm@Vm).flatten().tolist(), 1e-3)
print("TORCH-SMOKE:", "ALL PASS" if fails==0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
PY

#!/usr/bin/env bash
# smoke.sh — smoke-test every server endpoint + every kernel against a live pool.
# Spins up its own server (with a built-in worker slot) + a CPU worker, exercises everything, tears down.
#   bash scripts/smoke.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8799}"; CFG="./.moregpu-smoke.json"
PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "  ✅ $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  ❌ $1"; }
check(){ if [ "$1" = "$2" ]; then ok "$3"; else bad "$3 (got: $1, want: $2)"; fi; }

cleanup(){ kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; rm -f "$CFG"; }
trap cleanup EXIT
PIDS=(); rm -f "$CFG"

echo "== starting server (+built-in worker slot) + a CPU worker on :$PORT =="
# MOREGPU_INSECURE=1: loopback smoke over plaintext ws:// (default transport is wss://; TLS is covered by
# tests/security/tls_default.py + tls_install_path.py). The built-in --worker slot then connects ws://127.0.0.1.
PORT="$PORT" MOREGPU_CONFIG="$CFG" MOREGPU_INSECURE=1 deno run --allow-net --allow-env --allow-read --allow-write --allow-run --allow-sys \
  apps/coordinator/server.ts --worker >/tmp/smoke-srv.log 2>&1 &
PIDS+=($!); sleep 5
JOIN=$(deno eval "console.log(JSON.parse(Deno.readTextFileSync('$CFG')).joinToken)")
ADMIN=$(deno eval "console.log(JSON.parse(Deno.readTextFileSync('$CFG')).adminToken)")
deno run --allow-net --allow-env --allow-sys apps/worker/worker.ts \
  --server ws://localhost:$PORT/ws --token "$JOIN" --name smoke-cpu --cpu >/tmp/smoke-w.log 2>&1 &
PIDS+=($!); sleep 4
B="http://localhost:$PORT"; A="authorization: Bearer $ADMIN"
jq() { python3 -c "import sys,json;d=json.load(sys.stdin);print($1)"; }

echo "== endpoints =="
check "$(curl -s "$B/health" | jq 'd["ok"]')" "True" "GET /health"
check "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$B/submit" -d '{}')" "401" "POST /submit without token → 401"
check "$(curl -s "$B/device" -H "$A" | jq 'd["kind"]')" "virtual-gpu" "GET /device"
check "$(curl -s "$B/device" -H "$A" | jq 'd["capabilities"]["signedResults"]')" "True" "/device signedResults capability"
check "$(curl -s "$B/gpu" -H "$A" | jq '"slots" in d')" "True" "GET /gpu"
check "$(curl -s "$B/workers" -H "$A" | jq 'len(d)>=1')" "True" "GET /workers (>=1 worker)"
check "$(curl -s "$B/metrics" -H "$A" | grep -c '^moregpu_fleet ')" "1" "GET /metrics has moregpu_fleet"

echo "== every kernel (benchmark mode; expect verified + signed) =="
for k in matmul vector_add vector_mul saxpy relu scale gelu softmax layernorm; do
  sz=$([ "$k" = matmul ] && echo 256 || echo 100000)
  r=$(curl -s -X POST "$B/submit" -H "$A" -d "{\"kernel\":\"$k\",\"size\":$sz}")
  v=$(echo "$r" | jq 'str(d.get("verified")).lower()'); s=$(echo "$r" | jq 'str(d.get("signed")).lower()')
  if [ "$v" = "true" ] && [ "$s" = "true" ]; then ok "kernel $k (verified+signed)"; else bad "kernel $k (verified=$v signed=$s)"; fi
done

echo "== data mode (real tensors → correct output) =="
C=$(MOREGPU_URL="$B" MOREGPU_TOKEN="$ADMIN" python3 -c '
import os,sys; sys.path.insert(0,"examples"); from moregpu_client import MoreGPU
print(MoreGPU(os.environ["MOREGPU_URL"],os.environ["MOREGPU_TOKEN"]).matmul([1,2,3,4,5,6],[7,8,9,10,11,12],M=2,N=2,K=3))')
check "$C" "[58.0, 64.0, 139.0, 154.0]" "data-mode matmul → correct product"

echo "== async submit + poll =="
JID=$(curl -s -X POST "$B/submit?async=1" -H "$A" -d '{"kernel":"relu","size":50000}' | jq 'd["id"]')
sleep 2
check "$(curl -s "$B/jobs/$JID" -H "$A" | jq 'd["status"]')" "done" "async job $JID reaches done"

echo
echo "==================== $PASS passed, $FAIL failed ===================="
[ "$FAIL" -eq 0 ]

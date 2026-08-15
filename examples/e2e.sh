#!/usr/bin/env bash
# End-to-end MoreGPU: admin server (wizard tokens) + a GPU worker + a throttled CPU worker + a sealed job.
set -uo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8787}"
CFG="./.moregpu-e2e.json"
rm -f "$CFG"

cleanup() { kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; rm -f "$CFG"; }
trap cleanup EXIT
PIDS=()

echo "== starting admin server (:$PORT) — first run prints the setup wizard =="
# This local demo uses plaintext ws:// (MOREGPU_INSECURE=1) to stay dependency-free. The DEFAULT transport is
# now wss:// with a self-signed cert: for a real pool, drop MOREGPU_INSECURE and add each worker with the
# printed MOREGPU_PIN (curl scripts/install.sh | … MOREGPU_PIN=<pin> sh). See SECURITY.md.
PORT="$PORT" MOREGPU_CONFIG="$CFG" MOREGPU_INSECURE=1 deno run --allow-net --allow-env --allow-read --allow-write apps/coordinator/server.ts >/tmp/moregpu-server.log 2>&1 &
PIDS+=($!); sleep 2
JOIN=$(deno eval "console.log(JSON.parse(Deno.readTextFileSync('$CFG')).joinToken)")
ADMIN=$(deno eval "console.log(JSON.parse(Deno.readTextFileSync('$CFG')).adminToken)")

echo "== joining a GPU worker and a throttled CPU worker (outbound WebSocket, join token) =="
deno run --unstable-webgpu --allow-net --allow-env --allow-sys apps/worker/worker.ts --server "ws://localhost:$PORT/ws" --token "$JOIN" --name gpu-node >/tmp/moregpu-gpu.log 2>&1 &
PIDS+=($!)
deno run --allow-net --allow-env --allow-sys apps/worker/worker.ts --server "ws://localhost:$PORT/ws" --token "$JOIN" --name cpu-node --cpu --throttle 0.5 >/tmp/moregpu-cpu.log 2>&1 &
PIDS+=($!); sleep 5

echo "== fleet =="; curl -s "localhost:$PORT/workers" -H "authorization: Bearer $ADMIN"; echo
echo "== submit a 512x512 matmul (sharded across GPU+CPU, sealed, pooled, verified) =="
curl -s -X POST "localhost:$PORT/submit" -H "authorization: Bearer $ADMIN" -H 'content-type: application/json' -d '{"size":512}'; echo
echo "== server log =="; tail -n 5 /tmp/moregpu-server.log

#!/usr/bin/env bash
# scripts/isolate-linux.sh — dedicate BOUNDED, ISOLATED hardware to a MoreGPU worker on Linux.
#
# Wraps the given command in a cgroup v2 transient scope (systemd-run) with a CPU quota,
# cpuset pin, memory cap, low CPU/IO weight, and idle nice/ionice — so the pool can never
# disturb the machine's primary user. Degrades gracefully, in this order:
#   cgroup v2 scope (full)  ->  cgroup v2 scope (memory-only)  ->  nice/ionice only  ->  plain exec
#
# Usage:  isolate-linux.sh <command> [args...]
# Example:
#   MOREGPU_CPU_QUOTA=40% MOREGPU_CPUS=2-3 MOREGPU_MEM_MAX=2G \
#     scripts/isolate-linux.sh deno run --unstable-webgpu --allow-net --allow-env --allow-sys \
#       ~/.moregpu/worker.ts --server wss://ADMIN:8787/ws --token TOK
#
# All limits are env-overridable; defaults are deliberately polite.
set -euo pipefail

# ---- resource tunables -------------------------------------------------------
CPU_QUOTA="${MOREGPU_CPU_QUOTA:-40%}"      # CPUQuota= : max CPU time as % of ONE cpu; >100% spans cores
CPUS="${MOREGPU_CPUS:-}"                    # AllowedCPUs= : cpuset pin, e.g. "2-3" or "0,2,4" ("" = none)
MEM_MAX="${MOREGPU_MEM_MAX:-2G}"           # MemoryMax= : hard cap; kernel OOM-kills the scope above it
MEM_HIGH="${MOREGPU_MEM_HIGH:-}"           # MemoryHigh= : soft throttle ("" -> auto ~75% of MEM_MAX)
MEM_SWAP_MAX="${MOREGPU_MEM_SWAP_MAX:-0}"  # MemorySwapMax= : 0 keeps the worker from thrashing swap
CPU_WEIGHT="${MOREGPU_CPU_WEIGHT:-20}"     # CPUWeight= 1..10000 (default 100); low = yields under contention
IO_WEIGHT="${MOREGPU_IO_WEIGHT:-10}"       # IOWeight= 1..10000 (default 100); low = yields disk I/O
NICE="${MOREGPU_NICE:-19}"                 # nice -n : 19 = lowest CPU priority
IONICE_CLASS="${MOREGPU_IONICE_CLASS:-3}"  # ionice -c : 3 = idle I/O class (only runs when disk idle)
SCOPE_NAME="${MOREGPU_SCOPE_NAME:-moregpu-worker}"
ISOLATE="${MOREGPU_ISOLATE:-1}"            # 0 = bypass all isolation (debugging)

log() { printf '[moregpu-isolate] %s\n' "$*" >&2; }
[ "$#" -ge 1 ] || { log "usage: isolate-linux.sh <command> [args...]"; exit 2; }

# ---- GPU steering (exported into the worker's env; harmless when unused) ------
# CUDA backend (NVIDIA): pick a MIG instance or MPS client and cap its slice.
gpu_sel="${MOREGPU_CUDA_VISIBLE_DEVICES:-${MOREGPU_GPU_UUID:-}}"
[ -n "$gpu_sel" ] && export CUDA_VISIBLE_DEVICES="$gpu_sel"
[ -n "${MOREGPU_CUDA_MPS_THREAD_PCT:-}" ] && export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$MOREGPU_CUDA_MPS_THREAD_PCT"
[ -n "${MOREGPU_CUDA_MEM_MB:-}" ] && export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=${MOREGPU_CUDA_MEM_MB}M"
# Vulkan/WebGPU backend on Mesa (AMD RADV / Intel ANV): pin a specific device "vendorID:deviceID".
[ -n "${MOREGPU_VK_DEVICE_SELECT:-}" ] && export MESA_VK_DEVICE_SELECT="$MOREGPU_VK_DEVICE_SELECT"

# ---- helpers -----------------------------------------------------------------
pct() { case "$1" in *%) printf '%s' "$1" ;; *) printf '%s%%' "$1" ;; esac; }  # 40 -> 40%
derive_high() {  # ~75% of a size; always emit an INTEGER (awk %g would turn 6G into "4.5e+09", which systemd rejects)
  case "$1" in
    *[Gg]) local n="${1%[GgIiBb]}"; awk -v n="$n" 'BEGIN{printf "%.0fM", n*768}' ;;   # 0.75 * 1024 M per G
    *[Mm]) local n="${1%[MmIiBb]}"; awk -v n="$n" 'BEGIN{printf "%.0fK", n*768}' ;;   # 0.75 * 1024 K per M
    *[Kk]) local n="${1%[KkIiBb]}"; awk -v n="$n" 'BEGIN{printf "%.0f",  n*768}' ;;
    *[0-9]) awk -v n="$1" 'BEGIN{printf "%.0f", n*0.75}' ;;                            # plain bytes
    *) printf '' ;;  # non-numeric (e.g. a percentage) -> skip auto-derive
  esac
}

# Low-priority scheduler wrapper (Linux nice/ionice). Prepended INSIDE the scope.
prio=()
command -v nice   >/dev/null 2>&1 && prio+=(nice -n "$NICE")
command -v ionice >/dev/null 2>&1 && prio+=(ionice -c "$IONICE_CLASS")

exec_prio()  { log "cgroup unavailable — nice/ionice only"; exec "${prio[@]}" "$@"; }

# ---- graceful degradation gates ---------------------------------------------
[ "$ISOLATE" = "1" ]            || { log "MOREGPU_ISOLATE=0 — no isolation"; exec "$@"; }
[ "$(uname -s)" = "Linux" ]    || { log "not Linux — exec directly"; exec "$@"; }
[ -f /sys/fs/cgroup/cgroup.controllers ] || exec_prio "$@"      # need cgroup v2 unified hierarchy
command -v systemd-run >/dev/null 2>&1   || exec_prio "$@"      # need systemd

# ---- which controllers are actually available to us? ------------------------
# root -> system scope (all mounted controllers). unprivileged -> user scope (delegated set only).
uid="$(id -u)"
if [ "$uid" = "0" ]; then
  SCOPE=(--scope)
  CTRL_FILE=/sys/fs/cgroup/cgroup.controllers
else
  SCOPE=(--user --scope)
  CTRL_FILE="/sys/fs/cgroup/user.slice/user-${uid}.slice/user@${uid}.service/cgroup.controllers"
fi
CONTROLLERS=""
[ -r "$CTRL_FILE" ] && CONTROLLERS="$(cat "$CTRL_FILE" 2>/dev/null || true)"
have() { [ -z "$CONTROLLERS" ] && return 0; case " $CONTROLLERS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ---- assemble -p properties for the controllers we hold ---------------------
props=()      # full set
memprops=()   # memory-only fallback (delegated to every user by default)
if have memory; then
  [ -n "$MEM_MAX" ]      && { props+=(-p "MemoryMax=$MEM_MAX");   memprops+=(-p "MemoryMax=$MEM_MAX"); }
  [ -n "$MEM_MAX" ] && [ -z "$MEM_HIGH" ] && MEM_HIGH="$(derive_high "$MEM_MAX")"
  [ -n "$MEM_HIGH" ]     && { props+=(-p "MemoryHigh=$MEM_HIGH"); memprops+=(-p "MemoryHigh=$MEM_HIGH"); }
  [ -n "$MEM_SWAP_MAX" ] && { props+=(-p "MemorySwapMax=$MEM_SWAP_MAX"); memprops+=(-p "MemorySwapMax=$MEM_SWAP_MAX"); }
fi
if have cpu; then
  [ -n "$CPU_QUOTA" ]  && props+=(-p "CPUQuota=$(pct "$CPU_QUOTA")")
  [ -n "$CPU_WEIGHT" ] && props+=(-p "CPUWeight=$CPU_WEIGHT")
fi
have cpuset && [ -n "$CPUS" ]      && props+=(-p "AllowedCPUs=$CPUS")
have io     && [ -n "$IO_WEIGHT" ] && props+=(-p "IOWeight=$IO_WEIGHT")

[ "${#props[@]}" -gt 0 ] || exec_prio "$@"   # nothing delegated at all

# ---- validate the property set on a no-op, then exec the real worker --------
# A --scope runs synchronously and forwards the child's exit code to the caller (so systemd
# Restart=always sees the worker's real status). We probe with `true` first so we can still
# fall back cleanly; on success we exec the scope and let it replace this shell.
probe() { systemd-run "${SCOPE[@]}" --quiet --collect "$@" -- true >/dev/null 2>&1; }

if probe "${props[@]}"; then
  log "isolating: [${props[*]}] nice=${NICE} ionice=${IONICE_CLASS} uid=${uid}"
  exec systemd-run "${SCOPE[@]}" --quiet --collect --unit="${SCOPE_NAME}-$$" "${props[@]}" -- "${prio[@]}" "$@"
elif [ "${#memprops[@]}" -gt 0 ] && probe "${memprops[@]}"; then
  log "partial isolation (memory only): [${memprops[*]}] nice=${NICE} ionice=${IONICE_CLASS}"
  exec systemd-run "${SCOPE[@]}" --quiet --collect --unit="${SCOPE_NAME}-$$" "${memprops[@]}" -- "${prio[@]}" "$@"
else
  exec_prio "$@"
fi

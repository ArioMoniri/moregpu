#!/bin/sh
# MoreGPU worker — one-liner install for Linux & macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/ArioMoniri/moregpu/main/scripts/install.sh \
#     | MOREGPU_SERVER=wss://ADMIN:8787/ws MOREGPU_TOKEN=<join-token> sh
#
# Env:
#   MOREGPU_SERVER   ws:// or wss:// admin URL (required for a real pool)
#   MOREGPU_TOKEN    the pool's join token (from the server's setup wizard)
#   MOREGPU_NAME     optional worker name
#   MOREGPU_SERVICE  =1 to install a reboot-surviving background service (else runs in foreground)
#   MOREGPU_THROTTLE optional duty cycle 0.05..1 (lower = gentler on the user, less power)
set -eu

REPO="${MOREGPU_REPO:-ArioMoniri/moregpu}"; BRANCH="${MOREGPU_BRANCH:-main}"
SERVER="${MOREGPU_SERVER:-ws://localhost:8787/ws}"; TOKEN="${MOREGPU_TOKEN:-}"
WORKER_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/apps/worker/worker.ts"
HOME_DIR="${HOME}"; MG_DIR="${HOME_DIR}/.moregpu"
mkdir -p "$MG_DIR"

# --- resilient Deno install (retries; self-heals a broken install) ---
ensure_deno() {
  if command -v deno >/dev/null 2>&1 && deno --version >/dev/null 2>&1; then return 0; fi
  export DENO_INSTALL="${DENO_INSTALL:-$HOME_DIR/.deno}"; export PATH="$DENO_INSTALL/bin:$PATH"
  if command -v deno >/dev/null 2>&1 && deno --version >/dev/null 2>&1; then return 0; fi
  i=0; while [ $i -lt 3 ]; do
    echo "[moregpu] installing Deno runtime (try $((i+1)))…"
    if curl -fsSL https://deno.land/install.sh | sh >/dev/null 2>&1; then
      export PATH="$DENO_INSTALL/bin:$PATH"
      command -v deno >/dev/null 2>&1 && deno --version >/dev/null 2>&1 && return 0
    fi
    i=$((i+1)); sleep 3
  done
  echo "[moregpu] ERROR: could not install Deno. See https://deno.land/#installation"; exit 1
}
ensure_deno
DENO_BIN="$(command -v deno)"

# Cache the worker locally so it runs at boot / offline; re-fetch if the network is up.
echo "[moregpu] fetching worker…"
curl -fsSL "$WORKER_URL" -o "$MG_DIR/worker.ts.new" 2>/dev/null && mv "$MG_DIR/worker.ts.new" "$MG_DIR/worker.ts" || true
[ -f "$MG_DIR/worker.ts" ] || { echo "[moregpu] ERROR: could not fetch worker and no cached copy exists"; exit 1; }

RUN_ARGS="run --unstable-webgpu --allow-net --allow-env $MG_DIR/worker.ts --server $SERVER"
[ -n "$TOKEN" ] && RUN_ARGS="$RUN_ARGS --token $TOKEN"
[ -n "${MOREGPU_NAME:-}" ] && RUN_ARGS="$RUN_ARGS --name $MOREGPU_NAME"
[ -n "${MOREGPU_THROTTLE:-}" ] && RUN_ARGS="$RUN_ARGS --throttle $MOREGPU_THROTTLE"

if [ "${MOREGPU_SERVICE:-0}" = "1" ]; then
  OS="$(uname -s)"
  if [ "$OS" = "Linux" ] && command -v systemctl >/dev/null 2>&1; then
    UNIT_DIR="$HOME_DIR/.config/systemd/user"; mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/moregpu-worker.service" <<EOF
[Unit]
Description=MoreGPU worker
After=network-online.target
[Service]
ExecStart=$DENO_BIN $RUN_ARGS
Restart=always
RestartSec=5
Nice=15
[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now moregpu-worker.service
    loginctl enable-linger "$(whoami)" 2>/dev/null || echo "[moregpu] (run 'sudo loginctl enable-linger $(whoami)' so it runs before login)"
    echo "[moregpu] installed as a systemd user service. Logs: journalctl --user -u moregpu-worker -f"
  elif [ "$OS" = "Darwin" ]; then
    PLIST="$HOME_DIR/Library/LaunchAgents/dev.moregpu.worker.plist"
    ARGS_XML=""; for a in $RUN_ARGS; do ARGS_XML="$ARGS_XML<string>$a</string>"; done
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.moregpu.worker</string>
  <key>ProgramArguments</key><array><string>$DENO_BIN</string>$ARGS_XML</array>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Background</string>
</dict></plist>
EOF
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "[moregpu] installed as a launchd agent (dev.moregpu.worker). It starts at login and survives reboot."
  else
    echo "[moregpu] no supported service manager found; running in foreground instead."
    exec "$DENO_BIN" $RUN_ARGS
  fi
else
  # Self-healing supervised loop: if the worker ever exits, it restarts with backoff.
  # Optionally run detached inside tmux so it survives your SSH session closing.
  SUP="while true; do echo '[moregpu] starting worker'; \"$DENO_BIN\" $RUN_ARGS; code=\$?; echo \"[moregpu] worker exited (\$code); restarting in 5s\"; sleep 5; done"
  if [ "${MOREGPU_TMUX:-0}" = "1" ] && command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t moregpu 2>/dev/null || true
    tmux new-session -d -s moregpu "$SUP"
    echo "[moregpu] running in detached tmux session 'moregpu'. Attach: tmux attach -t moregpu"
  else
    echo "[moregpu] joining pool at ${SERVER} (GPU if available, else CPU). Self-healing; Ctrl-C to stop."
    sh -c "$SUP"
  fi
fi

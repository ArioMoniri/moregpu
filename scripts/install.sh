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
# Coordinator TLS cert pin (from the join banner). Normalize: strip a leading sha256:/colons, lowercase.
PIN="${MOREGPU_PIN:-}"; PIN="${PIN#sha256:}"; PIN="${PIN#SHA256:}"
PIN="$(printf '%s' "$PIN" | tr 'A-Z' 'a-z' | tr -d ':')"
WORKER_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/apps/worker/worker.ts"
SIG_URL="${WORKER_URL}.sig"

# --- PINNED RELEASE TRUST ROOT ------------------------------------------------------------
# This installer is the ROOT OF TRUST. Instead of running whatever `main` serves at fetch
# time, it refuses any worker.ts whose sha256 != WORKER_TS_SHA256, or whose detached Ed25519
# signature does not verify against RELEASE_PUBKEY_B64 (the maintainer's release key — PUBLIC,
# safe to pin here). Regenerate both on each signed release:
#     python3 scripts/release_sign.py sign --key <priv> apps/worker/worker.ts
RELEASE_PUBKEY_B64="${MOREGPU_RELEASE_PUBKEY:-oL3CUAld59+vdmrXYlLGl3RcvMAr3ZzBT8ib+Hpx7go=}"
WORKER_TS_SHA256="${MOREGPU_WORKER_SHA256:-5a02ba4dd9bd8f86a338f75cc8e427b396f0ad030b90033a6dc627697eafe51a}"
# DEV / UNPINNED escape hatch — LOCAL RUNS ONLY. Set MOREGPU_DEV_UNPINNED=1 to skip the gate
# when hacking on your own checkout. NEVER set it to join a real pool.
DEV_UNPINNED="${MOREGPU_DEV_UNPINNED:-0}"
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
curl -fsSL "$SIG_URL"    -o "$MG_DIR/worker.ts.sig.new" 2>/dev/null && mv "$MG_DIR/worker.ts.sig.new" "$MG_DIR/worker.ts.sig" || true
[ -f "$MG_DIR/worker.ts" ] || { echo "[moregpu] ERROR: could not fetch worker and no cached copy exists"; exit 1; }

# --- SUPPLY-CHAIN GATE: verify the LOCAL worker.ts (fresh OR cached) before it is ever run ---
# Runs on BOTH the MOREGPU_SERVICE install path and the foreground path, because both execute
# this same $MG_DIR/worker.ts. Fails CLOSED: any hash/signature mismatch aborts before exec.
if [ "$DEV_UNPINNED" = "1" ]; then
  echo "[moregpu] !! DEV/UNPINNED: skipping release verification (MOREGPU_DEV_UNPINNED=1) — never use this to join a real pool."
else
  [ -f "$MG_DIR/worker.ts.sig" ] || { echo "[moregpu] ERROR: no signature for worker.ts — refusing to run (set MOREGPU_DEV_UNPINNED=1 only for local hacking)"; exit 1; }
  # Embed the verifier locally (no second unverified fetch), then run it under the Deno we just ensured.
  cat > "$MG_DIR/verify_release.ts" <<'MOREGPU_VERIFY_EOF'
#!/usr/bin/env -S deno run --allow-read
// verify_release.ts — the install-time supply-chain GATE.
//
// Refuses to let the worker run unless the fetched artifact matches BOTH:
//   (1) a pinned sha256  — recomputed over the bytes on disk, constant-length-compared, and
//   (2) a detached Ed25519 signature that verifies against a pinned release PUBLIC key, over
//       the domain-separated message  `moregpu-release/v1\n<name>\n<sha256-hex>`.
//
// It reuses the coordinator's own verify path VERBATIM (apps/coordinator/server.ts:35,205,251):
//   b64d(...) + crypto.subtle.importKey('raw', pub, {name:'Ed25519'}, ...) +
//   crypto.subtle.verify({name:'Ed25519'}, pub, sig, msg).
// The message format is kept in lockstep with scripts/release_sign.py :: release_message().
//
// Exit codes:  0 = trusted (run the worker)   2 = usage   3 = sha256 mismatch   4 = bad signature
//
// Usage:
//   deno run --allow-read scripts/verify_release.ts \
//     --artifact <path> --sig <path> --sha256 <hex> --pubkey <b64> --name <basename>

// server.ts:35 — base64 decode (native fromBase64 when present, else atob fallback).
function b64d(s: string): Uint8Array {
  const F = (Uint8Array as unknown as { fromBase64?: (s: string) => Uint8Array }).fromBase64;
  if (typeof F === "function") return F(s);
  const bin = atob(s);
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}

function hex(u8: Uint8Array): string {
  let s = "";
  for (let i = 0; i < u8.length; i++) s += u8[i].toString(16).padStart(2, "0");
  return s;
}

// Constant-time-ish, length-checked string equality (avoid early-exit leaks on the pin compare).
function eq(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}

function argOf(flag: string): string | undefined {
  const i = Deno.args.indexOf(flag);
  return i >= 0 && i + 1 < Deno.args.length ? Deno.args[i + 1] : undefined;
}

const artifactPath = argOf("--artifact");
const sigPath = argOf("--sig");
const pinnedSha = (argOf("--sha256") ?? "").toLowerCase();
const pubkeyB64 = argOf("--pubkey") ?? "";
const name = argOf("--name") ?? (artifactPath ? artifactPath.split("/").pop()! : "");

if (!artifactPath || !sigPath || !pinnedSha || !pubkeyB64 || !name) {
  console.error(
    "[verify] usage: verify_release.ts --artifact P --sig P --sha256 HEX --pubkey B64 --name NAME",
  );
  Deno.exit(2);
}

// (1) hash pin — recompute sha256 of the exact bytes that would be executed.
const data = await Deno.readFile(artifactPath);
const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", data));
const got = hex(digest);
if (!eq(got, pinnedSha)) {
  console.error(`[verify] REJECT ${name}: sha256 mismatch`);
  console.error(`         pinned=${pinnedSha}`);
  console.error(`         actual=${got}`);
  Deno.exit(3);
}

// (2) authenticity — detached Ed25519 signature over the domain-separated message.
let sigB64: string;
try {
  sigB64 = (await Deno.readTextFile(sigPath)).trim();
} catch {
  console.error(`[verify] REJECT ${name}: signature file missing (${sigPath})`);
  Deno.exit(4);
}
let ok = false;
try {
  const pub = await crypto.subtle.importKey("raw", b64d(pubkeyB64) as BufferSource, { name: "Ed25519" }, false, ["verify"]);
  const msg = new TextEncoder().encode(`moregpu-release/v1\n${name}\n${got}`);
  ok = await crypto.subtle.verify({ name: "Ed25519" }, pub, b64d(sigB64) as BufferSource, msg);
} catch (e) {
  console.error(`[verify] REJECT ${name}: signature check errored (${e instanceof Error ? e.message : e})`);
  Deno.exit(4);
}
if (!ok) {
  console.error(`[verify] REJECT ${name}: signature does not verify against the pinned release key`);
  Deno.exit(4);
}

console.log(`[verify] OK ${name}: sha256=${got} · signed by pinned release key`);
Deno.exit(0);
MOREGPU_VERIFY_EOF
  if "$DENO_BIN" run --allow-read "$MG_DIR/verify_release.ts" \
      --artifact "$MG_DIR/worker.ts" --sig "$MG_DIR/worker.ts.sig" \
      --sha256 "$WORKER_TS_SHA256" --pubkey "$RELEASE_PUBKEY_B64" --name worker.ts; then
    :
  else
    echo "[moregpu] ERROR: worker.ts FAILED release verification — REFUSING to run (possible tampering, or a stale/unsigned cache)."
    exit 1
  fi
fi

# --- TLS: pin + trust the coordinator's self-signed cert (default transport is wss://) ---------------
# The coordinator serves wss:// with a SELF-SIGNED cert whose SHA-256 fingerprint it prints as MOREGPU_PIN.
# Deno's WebSocket has no per-connection "trust this self-signed cert" switch, so we can't fingerprint-pin
# the live socket the way the Python worker does. Instead we do a MITM-safe trust-on-fetch: pull the public
# leaf cert from the coordinator's /cert.pem, verify its sha256(DER) == the out-of-band MOREGPU_PIN, and only
# then hand it to Deno via DENO_CERT so the live wss handshake is validated against exactly that cert. A wrong
# cert (attacker's) has a different fingerprint → fails the pin check → we abort before the token is ever sent.
CERT_FILE=""
case "$SERVER" in
  wss://*)
    if [ -z "$PIN" ]; then
      # No pin → trust the SYSTEM CA store (parity with the torch worker's ssl.create_default_context). Leaving
      # CERT_FILE empty means no DENO_CERT, so Deno's WebSocket does normal CA verification: a real-cert coordinator
      # (a tunnel like trycloudflare, or Let's Encrypt behind a reverse proxy) validates and connects; a SELF-SIGNED
      # coordinator FAILS the handshake (fail-closed) — copy MOREGPU_PIN from its join banner to pin that.
      echo "[moregpu] wss:// with no MOREGPU_PIN — trusting the system CA store (works for a real-cert tunnel;"
      echo "          a self-signed coordinator will fail the TLS handshake — copy MOREGPU_PIN from its banner for that)."
    else
      base="${SERVER#wss://}"; base="${base%/ws}"; base="${base%/}"
      CERT_URL="https://${base}/cert.pem"
      echo "[moregpu] fetching coordinator TLS cert ($CERT_URL) to pin against sha256:${PIN}…"
      # -k: the fetch itself is unauthenticated; the sha256==PIN check below is what makes it trustworthy.
      if ! curl -fsSLk "$CERT_URL" -o "$MG_DIR/moregpu-cert.pem.new"; then
        echo "[moregpu] ERROR: could not fetch $CERT_URL — is the coordinator reachable and serving TLS?"; exit 1
      fi
      GOT_FP="$(CERTF="$MG_DIR/moregpu-cert.pem.new" "$DENO_BIN" eval '
        const pem = await Deno.readTextFile(Deno.env.get("CERTF"));
        const m = pem.match(/-----BEGIN CERTIFICATE-----([\s\S]*?)-----END CERTIFICATE-----/);
        if (!m) { console.error("no CERTIFICATE block in fetched PEM"); Deno.exit(2); }
        const der = Uint8Array.from(atob(m[1].replace(/\s+/g, "")), (c) => c.charCodeAt(0));
        const dig = new Uint8Array(await crypto.subtle.digest("SHA-256", der));
        console.log([...dig].map((x) => x.toString(16).padStart(2, "0")).join(""));
      ' 2>/dev/null)"
      if [ "$GOT_FP" != "$PIN" ]; then
        echo "[moregpu] ERROR: coordinator cert fingerprint sha256:${GOT_FP:-<none>} != pinned sha256:${PIN}"
        echo "          REFUSING to join (possible MITM, or the coordinator cert rotated — re-copy MOREGPU_PIN)."
        rm -f "$MG_DIR/moregpu-cert.pem.new"; exit 1
      fi
      mv "$MG_DIR/moregpu-cert.pem.new" "$MG_DIR/moregpu-cert.pem"
      CERT_FILE="$MG_DIR/moregpu-cert.pem"
      echo "[moregpu] coordinator TLS cert pinned OK · sha256:$(printf '%s' "$PIN" | cut -c1-16)… — trusting via DENO_CERT"
    fi
    ;;
esac

RUN_ARGS="run --unstable-webgpu --allow-net --allow-env --allow-sys $MG_DIR/worker.ts --server $SERVER"
[ -n "$TOKEN" ] && RUN_ARGS="$RUN_ARGS --token $TOKEN"
[ -n "${MOREGPU_NAME:-}" ] && RUN_ARGS="$RUN_ARGS --name $MOREGPU_NAME"
[ -n "${MOREGPU_THROTTLE:-}" ] && RUN_ARGS="$RUN_ARGS --throttle $MOREGPU_THROTTLE"

# If we pinned a self-signed coordinator cert above, trust it on every launch path. `export` covers the
# foreground / tmux / supervised loop (child inherits it); the service managers don't inherit the installer's
# env, so we also bake DENO_CERT into the systemd unit / launchd plist below.
SYSTEMD_CERT_ENV=""; PLIST_ENV=""
if [ -n "$CERT_FILE" ]; then
  export DENO_CERT="$CERT_FILE"
  SYSTEMD_CERT_ENV="Environment=DENO_CERT=$CERT_FILE
"
  PLIST_ENV="  <key>EnvironmentVariables</key><dict><key>DENO_CERT</key><string>$CERT_FILE</string></dict>
"
fi

if [ "${MOREGPU_SERVICE:-0}" = "1" ]; then
  OS="$(uname -s)"
  if [ "$OS" = "Linux" ] && command -v systemctl >/dev/null 2>&1; then
    UNIT_DIR="$HOME_DIR/.config/systemd/user"; mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/moregpu-worker.service" <<EOF
[Unit]
Description=MoreGPU worker
After=network-online.target
[Service]
${SYSTEMD_CERT_ENV}ExecStart=$DENO_BIN $RUN_ARGS
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
${PLIST_ENV}  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
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

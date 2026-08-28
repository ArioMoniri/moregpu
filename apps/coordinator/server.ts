#!/usr/bin/env -S deno run --allow-net --allow-env --allow-read --allow-write
/**
 * MoreGPU coordinator / admin server — self-contained, multi-tenant-isolated, token-secured.
 *
 * Presents your worker fleet as ONE virtual GPU: a Slurm-like job queue accepts tensor jobs, shards
 * them across connected workers, AES-GCM seals each work unit on the wire, pools + verifies results,
 * and exposes a modern web dashboard (fleet, GPU slot, queue, live per-worker load/throttle, errors,
 * and Prometheus /metrics for Grafana). First run is an automated wizard that mints this pool's tokens.
 *
 *   deno run --allow-net --allow-env --allow-read --allow-write \
 *     https://raw.githubusercontent.com/ArioMoniri/moregpu/main/apps/coordinator/server.ts
 *
 * Flags: --help
 * Env: PORT(8787) MOREGPU_BIND(0.0.0.0) MOREGPU_HOST MOREGPU_CONFIG MOREGPU_TLS_CERT+MOREGPU_TLS_KEY MOREGPU_DUTY
 *
 * TLS IS THE DEFAULT: with no MOREGPU_TLS_CERT/_KEY the coordinator mints a self-signed cert on first run
 * (persisted beside MOREGPU_CONFIG) and serves wss:// + https, printing the cert's SHA-256 fingerprint as the
 * worker pin (MOREGPU_PIN). Set MOREGPU_INSECURE=1 to opt back into plaintext ws:// for a simple local/CI run.
 */
if (Deno.args.includes('--help') || Deno.args.includes('-h')) { printHelp(); Deno.exit(0); }

const PORT = Number(Deno.env.get('PORT') ?? 8787);
const BIND = Deno.env.get('MOREGPU_BIND') ?? '0.0.0.0';
const CONFIG_PATH = Deno.env.get('MOREGPU_CONFIG') ?? './.moregpu-server.json';
const ADVERTISE_HOST = Deno.env.get('MOREGPU_HOST') ?? 'localhost';
const DUTY_HINT = Number(Deno.env.get('MOREGPU_DUTY') ?? 0.6);
const CERT_PATH = Deno.env.get('MOREGPU_TLS_CERT');
const KEY_PATH = Deno.env.get('MOREGPU_TLS_KEY');
const C = { reset: '\x1b[0m', dim: '\x1b[2m', b: '\x1b[1m', cyan: '\x1b[36m', grn: '\x1b[32m', yel: '\x1b[33m', mag: '\x1b[35m', red: '\x1b[31m' };

// ---------- base64 + sealing ----------
// Prefer the native TC39 Uint8Array<->base64 (Deno 2.x) — the manual String.fromCharCode loop is an order of
// magnitude slower and dominates a weight-push (GBs of base64). Feature-detected so older runtimes still work.
function b64e(u8: Uint8Array): string {
  const f = (u8 as unknown as { toBase64?: () => string }).toBase64;
  if (typeof f === 'function') return f.call(u8);
  let s = ''; const K = 0x8000; for (let i = 0; i < u8.length; i += K) s += String.fromCharCode(...u8.subarray(i, i + K)); return btoa(s);
}
function b64d(s: string): Uint8Array {
  const F = (Uint8Array as unknown as { fromBase64?: (s: string) => Uint8Array }).fromBase64;
  if (typeof F === 'function') return F(s);
  const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u;
}
// base64url, but never start with '-'/'_': a leading dash makes `worker_torch.py --token <tok>` unparseable
// (argparse reads it as a flag), and a leading dash is awkward in shells/URLs generally. Trim any from the front.
function tokenB64url(n = 24): string { return b64e(crypto.getRandomValues(new Uint8Array(n))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '').replace(/^[-_]+/, ''); }
async function importKey(raw: Uint8Array) { return crypto.subtle.importKey('raw', raw as BufferSource, 'AES-GCM', false, ['encrypt', 'decrypt']); }
async function seal(key: Uint8Array, plain: Uint8Array) { const iv = crypto.getRandomValues(new Uint8Array(12)); const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv: iv as BufferSource }, await importKey(key), plain as BufferSource); return { iv: b64e(iv), ct: b64e(new Uint8Array(ct)) }; }
async function unseal(key: Uint8Array, blob: { iv: string; ct: string }) { return new Uint8Array(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: b64d(blob.iv) as BufferSource }, await importKey(key), b64d(blob.ct) as BufferSource)); }
const f32ToB64 = (a: Float32Array) => b64e(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
const b64ToF32 = (s: string) => new Float32Array(b64d(s).buffer);
const constEq = (a: string, b: string) => { if (a.length !== b.length) return false; let d = 0; for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i); return d === 0; };

// ---------- structured log ring buffer (errors + debug) ----------
type Level = 'info' | 'warn' | 'error' | 'debug';
interface LogEntry { ts: number; level: Level; msg: string; ctx?: string }
const LOG: LogEntry[] = [];
function log(level: Level, msg: string, ctx?: string) {
  const e: LogEntry = { ts: Date.now(), level, msg, ctx };
  LOG.push(e); if (LOG.length > 500) LOG.shift();
  const col = level === 'error' ? C.red : level === 'warn' ? C.yel : level === 'debug' ? C.dim : C.cyan;
  console.log(`${col}[coord:${level}]${C.reset} ${msg}${ctx ? C.dim + ' · ' + ctx + C.reset : ''}`);
}

// ---------- config / wizard ----------
// keyEpoch + removedPubkeys are PERSISTED so a revocation survives a coordinator restart: without this a restart
// resets the epoch to 0 (re-deriving pre-revocation keys) and empties the ban list, resurrecting a removed worker.
interface Config { adminToken: string; joinToken: string; tenantKeyB64: string; created: string; keyEpoch?: number; removedPubkeys?: string[]; }
async function loadOrInitConfig(): Promise<{ cfg: Config; fresh: boolean }> {
  try { return { cfg: JSON.parse(await Deno.readTextFile(CONFIG_PATH)), fresh: false }; }
  catch {
    const cfg: Config = { adminToken: tokenB64url(24), joinToken: tokenB64url(18), tenantKeyB64: b64e(crypto.getRandomValues(new Uint8Array(32))), created: new Date().toISOString(), keyEpoch: 0, removedPubkeys: [] };
    await Deno.writeTextFile(CONFIG_PATH, JSON.stringify(cfg, null, 2));
    return { cfg, fresh: true };
  }
}
const { cfg, fresh } = await loadOrInitConfig();
// ── PER-WORKER KEYS + KEY-EPOCH ROTATION ─────────────────────────────────────────────────────────────────
// The pool secret in config (`tenantKeyB64`) is NO LONGER broadcast to every worker. It is now the MASTER
// secret from which each worker's OWN sealing key is DERIVED via HKDF-SHA256 over (worker id + key epoch):
// `welcome` hands a per-worker key, and every coordinator<->worker sealed frame (assign / model+train relay /
// weight cache) uses THAT worker's key — so one worker cannot open another worker's coordinator traffic even if
// it captures the wire. `keyEpoch` is a monotonic revocation counter: an admin `remove` bumps it, so a key
// derived afterwards differs, killing a captured pre-revocation key (rotation). keyEpoch + the ban list are
// PERSISTED to config (saveConfig), so a revocation survives a coordinator restart instead of resetting to 0.
const MASTER = b64d(cfg.tenantKeyB64);
let keyEpoch = cfg.keyEpoch ?? 0;
// Persist mutable trust state (epoch + ban list) back to the config file so revocation is durable across restarts.
async function saveConfig(): Promise<void> {
  const out: Config = { adminToken: cfg.adminToken, joinToken: cfg.joinToken, tenantKeyB64: cfg.tenantKeyB64, created: cfg.created, keyEpoch, removedPubkeys: [...removedPubkeys] };
  try { await Deno.writeTextFile(CONFIG_PATH, JSON.stringify(out, null, 2)); } catch (e) { log('warn', `could not persist config: ${e instanceof Error ? e.message : e}`); }
}
const HKDF_SALT = new TextEncoder().encode('moregpu:hkdf:v1');
async function hkdf(info: string, bytes = 32): Promise<Uint8Array> {
  const base = await crypto.subtle.importKey('raw', MASTER as BufferSource, 'HKDF', false, ['deriveBits']);
  const bits = await crypto.subtle.deriveBits(
    { name: 'HKDF', hash: 'SHA-256', salt: HKDF_SALT as BufferSource, info: new TextEncoder().encode(info) as BufferSource },
    base, bytes * 8,
  );
  return new Uint8Array(bits);
}
// Per-worker coordinator<->worker key: HKDF(master, "worker:<id>:epoch:<n>"). Deterministic per (id, epoch),
// independent across ids, rotated by an epoch bump. A reconnecting worker (same id, same epoch) re-derives the
// SAME key, so mid-stream resume/failover is unaffected.
const deriveWorkerKey = (workerId: string, epoch: number) => hkdf(`moregpu:worker:${workerId}:epoch:${epoch}`);
// Per-PAIR peer-edge session key: HKDF(master, "peeredge:<sid>:ke<keyEpoch>:re<ringEpoch>:<from>-><to>"). The
// coordinator BROKERS it — it hands `from` the seal key and `to` the open key — so a peer act/expert frame on
// edge A->B is sealed with a key ONLY that pair holds: not the master, not either end's per-worker key. Fresh
// per wiring (ring epoch) AND per revocation (key epoch).
const deriveEdgeKey = (sid: string, ringEpoch: number, from: string, to: string) =>
  hkdf(`moregpu:peeredge:${sid}:ke${keyEpoch}:re${ringEpoch}:${from}->${to}`);
const RAW = 'https://raw.githubusercontent.com/ArioMoniri/moregpu/main';

// ---------- TLS: the DEFAULT transport ----------
// MoreGPU serves wss:// (+ https) by DEFAULT. If the operator supplies MOREGPU_TLS_CERT+_KEY we use those;
// otherwise, on first run, the coordinator MINTS its own self-signed P-256 cert and persists it beside the
// pool config (the MOREGPU_CONFIG dir), so the fingerprint is STABLE across restarts. The cert's SHA-256
// fingerprint is printed in the wizard and handed to workers as the pin (MOREGPU_PIN) — a worker refuses any
// coordinator whose cert doesn't match. Set MOREGPU_INSECURE=1 to opt back into plaintext ws:// (the simple
// local/CI path, unchanged). The generated private key never leaves the admin box and is written mode 0600.
const INSECURE = ['1', 'true', 'yes', 'on'].includes((Deno.env.get('MOREGPU_INSECURE') ?? '').toLowerCase());
const CONFIG_DIR = (() => { const i = Math.max(CONFIG_PATH.lastIndexOf('/'), CONFIG_PATH.lastIndexOf('\\')); return i >= 0 ? CONFIG_PATH.slice(0, i) : '.'; })();
const CERT_STORE = `${CONFIG_DIR}/moregpu-cert.pem`;
const KEY_STORE = `${CONFIG_DIR}/moregpu-key.pem`;
// A minimal DER encoder + a self-signed X.509v3 ECDSA-P256/SHA-256 cert, built with WebCrypto ONLY — no
// external deps and no --allow-run, so the documented one-liner `deno run` keeps its exact permission set.
const _dLen = (n: number): number[] => { if (n < 0x80) return [n]; const b: number[] = []; let x = n; while (x > 0) { b.unshift(x & 0xff); x = Math.floor(x / 256); } return [0x80 | b.length, ...b]; };
const _der = (tag: number, c: number[]): number[] => [tag, ..._dLen(c.length), ...c];
const _dInt = (bytes: number[]): number[] => { let b = bytes.slice(); while (b.length > 1 && b[0] === 0 && (b[1] & 0x80) === 0) b.shift(); if (b[0] & 0x80) b = [0, ...b]; return _der(0x02, b); };
const _ALG_ES256 = _der(0x30, _der(0x06, [0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02])); // AlgorithmIdentifier ecdsa-with-SHA256 (no params)
function _toPem(label: string, der: Uint8Array): string { const b64 = b64e(der).replace(/(.{64})/g, '$1\n').replace(/\n$/, ''); return `-----BEGIN ${label}-----\n${b64}\n-----END ${label}-----\n`; }
function _firstCertDer(pem: string): Uint8Array { const m = pem.match(/-----BEGIN CERTIFICATE-----([\s\S]*?)-----END CERTIFICATE-----/); if (!m) throw new Error('no CERTIFICATE block in PEM'); return b64d(m[1].replace(/\s+/g, '')); }
// The pin is the SHA-256 of the DER of the LEAF certificate — byte-identical to `openssl x509 -fingerprint -sha256`.
async function certFingerprint(certPem: string): Promise<string> { return [...new Uint8Array(await crypto.subtle.digest('SHA-256', _firstCertDer(certPem) as BufferSource))].map((x) => x.toString(16).padStart(2, '0')).join(''); }
async function generateSelfSigned(): Promise<{ certPem: string; keyPem: string }> {
  const kp = await crypto.subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign', 'verify']) as CryptoKeyPair;
  const spki = [...new Uint8Array(await crypto.subtle.exportKey('spki', kp.publicKey))]; // DER SubjectPublicKeyInfo, as-is
  const serial = [...crypto.getRandomValues(new Uint8Array(16))]; serial[0] &= 0x7f; if (serial[0] === 0) serial[0] = 1; // positive
  const utc = (d: Date) => { const p = (n: number) => String(n).padStart(2, '0'); return _der(0x17, [...new TextEncoder().encode(`${p(d.getUTCFullYear() % 100)}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}${p(d.getUTCHours())}${p(d.getUTCMinutes())}${p(d.getUTCSeconds())}Z`)]); };
  const now = Date.now();
  const validity = _der(0x30, [...utc(new Date(now - 3600e3)), ...utc(new Date(now + 3650 * 864e5))]); // ~10y, 1h backdated for skew
  const nm = _der(0x30, _der(0x31, _der(0x30, [..._der(0x06, [0x55, 0x04, 0x03]), ..._der(0x0c, [...new TextEncoder().encode('moregpu')])]))); // CN=moregpu
  // subjectAltName: localhost + 127.0.0.1 (+ the advertised host), so a verifying LAN client also matches.
  const gn: number[] = []; const dns = (h: string) => { const b = [...new TextEncoder().encode(h)]; gn.push(0x82, ..._dLen(b.length), ...b); }; const ip4 = (a: number[]) => gn.push(0x87, 0x04, ...a);
  dns('localhost');
  const asIp = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(ADVERTISE_HOST || '');
  if (ADVERTISE_HOST && ADVERTISE_HOST !== 'localhost' && !asIp && /^[A-Za-z0-9.-]+$/.test(ADVERTISE_HOST)) dns(ADVERTISE_HOST);
  ip4([127, 0, 0, 1]);
  if (asIp) ip4([+asIp[1] & 0xff, +asIp[2] & 0xff, +asIp[3] & 0xff, +asIp[4] & 0xff]);
  const san = _der(0x30, [..._der(0x06, [0x55, 0x1d, 0x11]), ..._der(0x04, _der(0x30, gn))]);
  const bc = _der(0x30, [..._der(0x06, [0x55, 0x1d, 0x13]), 0x01, 0x01, 0xff, ..._der(0x04, _der(0x30, []))]); // basicConstraints CA:FALSE, critical
  const exts = _der(0xa3, _der(0x30, [...bc, ...san])); // [3] EXPLICIT Extensions
  const tbs = _der(0x30, [..._der(0xa0, _dInt([2])), ..._dInt(serial), ..._ALG_ES256, ...nm, ...validity, ...nm, ...spki, ...exts]);
  const rawSig = new Uint8Array(await crypto.subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, kp.privateKey, Uint8Array.from(tbs) as BufferSource)); // IEEE-P1363 r||s
  const sig = _der(0x30, [..._dInt([...rawSig.slice(0, 32)]), ..._dInt([...rawSig.slice(32, 64)])]); // → DER Ecdsa-Sig-Value
  const cert = Uint8Array.from(_der(0x30, [...tbs, ..._ALG_ES256, ..._der(0x03, [0x00, ...sig])]));
  const pkcs8 = new Uint8Array(await crypto.subtle.exportKey('pkcs8', kp.privateKey));
  return { certPem: _toPem('CERTIFICATE', cert), keyPem: _toPem('PRIVATE KEY', pkcs8) };
}
let TLS_CERT: string | undefined, TLS_KEY: string | undefined, TLS_FP: string | undefined, TLS_CERT_FILE: string | undefined;
if (!INSECURE) {
  if (CERT_PATH && KEY_PATH) {
    TLS_CERT = await Deno.readTextFile(CERT_PATH); TLS_KEY = await Deno.readTextFile(KEY_PATH); TLS_CERT_FILE = CERT_PATH;
  } else {
    try { TLS_CERT = await Deno.readTextFile(CERT_STORE); TLS_KEY = await Deno.readTextFile(KEY_STORE); } // reuse a persisted cert → stable pin
    catch {
      const g = await generateSelfSigned(); TLS_CERT = g.certPem; TLS_KEY = g.keyPem;
      await Deno.writeTextFile(CERT_STORE, TLS_CERT);
      await Deno.writeTextFile(KEY_STORE, TLS_KEY, { mode: 0o600 });
      try { await Deno.chmod(KEY_STORE, 0o600); } catch { /* windows / best-effort: keep the key non-world-readable */ }
      log('info', `minted self-signed TLS cert → ${CERT_STORE} (private key ${KEY_STORE}, mode 0600)`);
    }
    TLS_CERT_FILE = CERT_STORE;
  }
  TLS_FP = await certFingerprint(TLS_CERT);
}
const scheme = TLS_CERT ? 'wss' : 'ws';
const httpScheme = TLS_CERT ? 'https' : 'http';

function art() {
  // ANSI Shadow "MOREGPU" with an indigo→pink→red 256-color gradient (matches the `moregpu` CLI).
  const rows = [
    '███╗   ███╗ ██████╗ ██████╗ ███████╗ ██████╗ ██████╗ ██╗   ██╗',
    '████╗ ████║██╔═══██╗██╔══██╗██╔════╝██╔════╝ ██╔══██╗██║   ██║',
    '██╔████╔██║██║   ██║██████╔╝█████╗  ██║  ███╗██████╔╝██║   ██║',
    '██║╚██╔╝██║██║   ██║██╔══██╗██╔══╝  ██║   ██║██╔═══╝ ██║   ██║',
    '██║ ╚═╝ ██║╚██████╔╝██║  ██║███████╗╚██████╔╝██║     ╚██████╔╝',
    '╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝      ╚═════╝ ',
  ];
  const cols = [63, 99, 135, 171, 205, 203];
  return '\n' + rows.map((r, i) => `  \x1b[38;5;${cols[i]}m${r}${C.reset}`).join('\n');
}
function wizardBanner() {
  const wsUrl = `${scheme}://${ADVERTISE_HOST}:${PORT}/ws`;
  const pinEnv = TLS_FP ? ` MOREGPU_PIN=${TLS_FP}` : '';
  console.log(art());
  console.log(`\n  ${C.b}${fresh ? C.grn + 'NEW POOL CREATED' : 'pool ready'}${C.reset}`);
  console.log(`  ${C.dim}dashboard${C.reset}  ${httpScheme}://${ADVERTISE_HOST}:${PORT}`);
  console.log(`  ${C.dim}admin token${C.reset}  ${C.yel}${cfg.adminToken}${C.reset}   ${C.dim}(controls the pool — keep secret)${C.reset}`);
  console.log(`  ${C.dim}join token${C.reset}   ${C.mag}${cfg.joinToken}${C.reset}   ${C.dim}(lets machines enroll)${C.reset}`);
  if (TLS_FP) console.log(`  ${C.dim}cert pin${C.reset}     ${C.grn}sha256:${TLS_FP}${C.reset}   ${C.dim}(workers verify this — MOREGPU_PIN)${C.reset}`);
  console.log(`  ${C.dim}config${C.reset}       ${CONFIG_PATH}`);
  console.log(`\n  ${C.b}Add a machine to this pool${C.reset} ${C.dim}(Linux/macOS)${C.reset}:`);
  console.log(`    ${C.grn}curl -fsSL ${RAW}/scripts/install.sh \\`);
  console.log(`      | MOREGPU_SERVER=${wsUrl} MOREGPU_TOKEN=${cfg.joinToken}${pinEnv} sh${C.reset}`);
  console.log(`  ${C.dim}Windows:  $env:MOREGPU_SERVER="${wsUrl}"; $env:MOREGPU_TOKEN="${cfg.joinToken}";${TLS_FP ? ` $env:MOREGPU_PIN="${TLS_FP}";` : ''} irm ${RAW}/scripts/install.ps1 | iex${C.reset}`);
  console.log(`  ${C.dim}reboot-surviving service: add MOREGPU_SERVICE=1 · type /help in this console's HTTP: ${httpScheme}://${ADVERTISE_HOST}:${PORT}/help${C.reset}`);
  if (TLS_FP) {
    console.log(`\n  ${C.grn}${C.b}✔ TLS on${C.reset}${C.dim} — serving ${scheme}:// with a ${CERT_PATH && KEY_PATH ? 'provided' : 'self-signed'} cert. Workers pin the fingerprint above; a mismatched cert is refused.${C.reset}`);
    console.log(`  ${C.dim}Simple local/CI plaintext: set MOREGPU_INSECURE=1 to serve ws:// instead.${C.reset}`);
  } else if (BIND !== '127.0.0.1' && BIND !== 'localhost') {
    console.log(`\n  ${C.red}${C.b}⚠ MOREGPU_INSECURE=1 and bound to ${BIND}${C.reset}${C.red} — the join handshake (including the tenant key) travels in plaintext.${C.reset}`);
    console.log(`  ${C.dim}Drop MOREGPU_INSECURE for the default wss://, or bind MOREGPU_BIND=127.0.0.1 behind a VPN/tunnel.${C.reset}`);
  }
  // The install line advertises ADVERTISE_HOST. If that is loopback but the server is bound wide, a worker on
  // ANOTHER machine that pastes it would connect to its OWN localhost — flag the mismatch and point at the fix.
  if ((ADVERTISE_HOST === 'localhost' || ADVERTISE_HOST === '127.0.0.1') && BIND !== '127.0.0.1' && BIND !== 'localhost') {
    console.log(`\n  ${C.dim}The add-a-machine command above uses ${C.reset}localhost${C.dim} — that only resolves on THIS machine.`);
    console.log(`  For workers elsewhere, set ${C.reset}MOREGPU_HOST${C.dim}=<a hostname/IP they can reach>, or expose a tunnel (${C.reset}moregpu serve --tunnel${C.dim}).${C.reset}`);
  }
  // No built-in worker → the fleet is empty until something joins; nudge toward the one-flag fix.
  if (!(Deno.args.includes('--worker') || Deno.env.get('MOREGPU_SELF_WORKER') === '1')) {
    console.log(`\n  ${C.dim}No worker in this pool yet — the fleet stays empty until one joins. Add ${C.reset}--worker${C.dim} to contribute THIS machine, or run the command above on another machine.${C.reset}`);
  }
  console.log('');
}
function printHelp() {
  console.log(art());
  console.log(`
  ${C.b}MoreGPU admin server${C.reset} — presents your worker fleet as one virtual GPU.

  ${C.b}Run${C.reset}
    deno run --allow-net --allow-env --allow-read --allow-write ${RAW}/apps/coordinator/server.ts

  ${C.b}Environment${C.reset}
    PORT=8787              HTTP/WebSocket port
    MOREGPU_BIND=0.0.0.0   bind address (0.0.0.0 = reachable from other machines)
    MOREGPU_HOST=localhost hostname advertised in the wizard/worker command
    MOREGPU_CONFIG=./.moregpu-server.json   pool identity (tokens + key); its dir also holds the self-signed cert
    MOREGPU_TLS_CERT / MOREGPU_TLS_KEY      PEM paths → serve https + wss (else a self-signed cert is minted)
    MOREGPU_INSECURE=1     opt out of TLS → plaintext ws:// (simple local/CI); default is wss://
    MOREGPU_DUTY=0.6       duty-cycle ceiling hint sent to workers

  ${C.b}HTTP API${C.reset} ${C.dim}(admin endpoints need: Authorization: Bearer <admin token>)${C.reset}
    GET  /                 dashboard (fleet, GPU slot, queue, logs)
    GET  /help             this help as JSON/text
    GET  /health           liveness + fleet size (public)
    GET  /gpu              the pool as one virtual GPU device (admin)
    GET  /workers          connected workers + live load/throttle (admin)
    POST /submit           {"kernel":"matmul","size":1024}  run a job (admin)
    GET  /jobs             job queue + history (admin)
    GET  /jobs/:id         one job (admin)
    GET  /logs             recent errors/debug (admin)
    GET  /metrics          Prometheus metrics for Grafana (admin)
    POST /workers/:id/control  pause · resume · set duty (ceil) · schedule · nick · remove (admin)

  ${C.b}Kernels${C.reset}  matmul · vector_add · vector_mul · saxpy · relu · scale · softmax · layernorm   ${C.dim}(extensible)${C.reset}
  ${C.b}Workers${C.reset}  a worker sets when it contributes with MOREGPU_SCHEDULE=always|idle-only|HH:MM-HH:MM
`);
}

// ---------- metrics ----------
const M = { jobsTotal: 0, jobsDone: 0, jobsFailed: 0, shardsDone: 0, shardsFailed: 0, gpuShards: 0, cpuShards: 0 };
const KM: Record<string, number> = {}; // jobs per kernel

// ---------- worker registry ----------
interface Worker {
  id: string; backend: string; label: string; os: string; ws: WebSocket;
  load1: number; util: number; duty: number; busy: boolean; joinedAt: number;
  // contribution accounting. `ops` is the CONSISTENT activity currency (one kernel shard OR one serving call
  // OR one training round = 1 op) — the trend + share are built from it so they never conflate LLM tokens
  // with kernel matrix-elements. `units` = kernel output-elements (detail); `tokens` = LLM tokens served (detail).
  shards: number; units: number; ops: number; tokens: number; errors: number; consecErrors: number; healthyBeats: number; busyCount: number; totalMs: number;
  lastOps: number; history: number[]; // per-sample ops completed → sparkline trend (consistent unit)
  pubkey?: CryptoKey; // Ed25519 public key for verifying this worker's result signatures
  pubkeyB64?: string; // raw public key (for the removed-worker denylist)
  key: Uint8Array; // this worker's OWN AES key = HKDF(master, id, keyEpoch) — every coordinator<->worker seal uses THIS, never a shared/fleet key
  keyEpoch: number; // the key epoch this worker's key was minted at (bumps on an admin revocation)
  paused: boolean; // scheduled-off or admin-paused → coordinator assigns it no new work
  pausedReason?: string | null; // 'admin' | 'schedule' | null — why it's paused
  ceil: number; // the worker's reported/administered duty CEILING (distinct from the live effective duty)
  schedule?: string; // the machine's own contribution schedule ("always" / "idle-only" / "HH:MM-HH:MM")
  nick?: string; // optional admin-set display label
  peer?: { url: string; pub: string; candidates?: string[] }; // PEER TRANSPORT (opt-in): worker's activation endpoint(s) + pubkey (candidates: reachable URLs in preference order — public/forwarded first, LAN last)
  reflexiveIp?: string; // the source IP the coordinator observed for this worker's control connection (a STUN-like hint for MOREGPU_PEER_PUBLIC)
  caps: Set<string>; // what this worker can host — 'kernel' (offloaded matmul/elementwise), 'shard' (a pipeline decoder-block slice), 'shardEnds' (first/last stage: embeddings/tokenizer/head), 'resident' (a whole model), 'train' (autograd). A WebGPU worker advertises {kernel,shard}; a torch worker gets them all.
}
// Capabilities a worker gets when it does NOT advertise a `caps` list (back-compat): a torch worker can do
// everything (autograd + whole-model residency); anything else (WebGPU/CPU kernel worker) is kernel-only.
function deriveCaps(label: string): string[] { return label.includes('torch') ? ['kernel', 'shard', 'shardEnds', 'resident', 'train'] : ['kernel']; }
const workers = new Map<string, Worker>();
const removedPubkeys = new Set<string>(cfg.removedPubkeys ?? []); // workers an admin removed — refuse their re-registration (ban by key); restored from config so a ban survives restart
// Weight RESIDENCY: a named weight is cached on ONE worker; a resident matmul (bRef) runs there without
// re-sending the weight. Place different layers' weights on different workers to split a model (pipeline).
const weightHome = new Map<string, { worker: string; rows: number; cols: number; dtype: 'f32' | 'f16' }>(); // weightId → home worker + dims
const weightStore = new Map<string, Float32Array>(); // coordinator's own copy (DEQUANTIZED for f16), used to verify resident results
const REGISTER_TIMEOUT_MS = Number(Deno.env.get('MOREGPU_REGISTER_TIMEOUT_MS') ?? 8000);
const MAX_JOBS = Number(Deno.env.get('MOREGPU_MAX_JOBS') ?? 500); // cap retained job records (+ their output blobs)
const safeId = (s: string) => (String(s ?? '').replace(/[^A-Za-z0-9._:@-]/g, '').slice(0, 64) || 'worker');
/** Workers eligible for new work right now (not scheduled-off / admin-paused). */
function activeFleet(): Worker[] { return [...workers.values()].filter((w) => !w.paused); }
type ResultMsg = { ok: boolean; sealedOut?: { iv: string; ct: string }; error?: string; backend?: string; ms?: number; signed?: boolean };
interface Pending { resolve: (r: ResultMsg) => void; reject: (e: Error) => void; workerId: string; }
const pending = new Map<string, Pending>();
const SHARD_TIMEOUT_MS = Number(Deno.env.get('MOREGPU_SHARD_TIMEOUT_MS') ?? 60_000);
// PEER TRANSPORT (opt-in): when on, the coordinator wires a worker->worker activation ring after a shard loads
// and routes an uncached shard_forward through it (coordinator off the per-token data path). Default OFF → the
// relay path (shardPipe) is used unchanged. See docs/ROADMAP.md "Peer transport".
const PEER_TRANSPORT = ['1', 'true', 'yes', 'on'].includes((Deno.env.get('MOREGPU_PEER_TRANSPORT') ?? '').toLowerCase());
const MAX_SHARD_ATTEMPTS = Number(Deno.env.get('MOREGPU_MAX_SHARD_ATTEMPTS') ?? 3); // reassign a failed/timed-out shard to other workers
const AUTO_PAUSE_ERRORS = Number(Deno.env.get('MOREGPU_AUTO_PAUSE_ERRORS') ?? 4); // consecutive HARD failures before a worker is auto-paused
const AUTO_RESUME_BEATS = Number(Deno.env.get('MOREGPU_AUTO_RESUME_BEATS') ?? 3); // healthy heartbeats that auto-un-pause an errored worker
const MAX_CONCURRENT_JOBS = Number(Deno.env.get('MOREGPU_MAX_CONCURRENT_JOBS') ?? 4); // jobs the queue runs at once
const STALE_JOB_MS = Number(Deno.env.get('MOREGPU_STALE_JOB_MS') ?? 300_000); // fail a job that can't be scheduled within this

function wireWorker(ws: WebSocket, reflexiveIp?: string) {
  let id = '';
  let registered = false;
  // Close a socket that connects but never authenticates, so an unauthenticated peer can't hold FDs/RAM open.
  const authTimer = setTimeout(() => { if (!registered) { try { ws.send(JSON.stringify({ t: 'denied', reason: 'no register within timeout' })); } catch { /* */ } try { ws.close(); } catch { /* */ } } }, REGISTER_TIMEOUT_MS);
  ws.onmessage = async (ev) => {
    let m: Record<string, unknown>;
    try { m = JSON.parse(ev.data as string); } catch { log('warn', 'bad frame from worker'); return; }
    if (m.t === 'register') {
      if (registered) return; // one register per socket
      if (!constEq(String(m.joinToken ?? ''), cfg.joinToken)) { ws.send(JSON.stringify({ t: 'denied', reason: 'bad join token' })); log('warn', 'worker rejected: bad join token'); ws.close(); return; }
      const node = m.node as { id: string; backend: string; label: string; os: string; caps?: unknown; peer?: { url: string; pub: string; candidates?: unknown } };
      // CAPABILITY NEGOTIATION: a worker MAY advertise a `caps` list; otherwise derive from its label so old
      // torch/WebGPU workers keep working. Sanitized + capped. This is the authority that replaces label-sniffing.
      const advertised = Array.isArray(node.caps) ? (node.caps as unknown[]).filter((c): c is string => typeof c === 'string').map((c) => safeId(c)).slice(0, 16) : null;
      const caps = new Set<string>(advertised && advertised.length ? advertised : deriveCaps(safeId(node.label)));
      const pubkeyB64 = typeof m.pubkey === 'string' ? m.pubkey : undefined;
      // PEER TRANSPORT (opt-in): a peer-enabled worker advertises its LAN activation endpoint; only honoured when the
      // coordinator itself has the flag on, and NEVER exposed publicly — it is handed only to same-tenant ring peers.
      const peer = (PEER_TRANSPORT && node.peer && typeof node.peer.url === 'string' && typeof node.peer.pub === 'string')
        ? { url: node.peer.url, pub: node.peer.pub,
            candidates: Array.isArray(node.peer.candidates) ? (node.peer.candidates as unknown[]).filter((c): c is string => typeof c === 'string' && /^wss?:\/\//.test(c)).slice(0, 8) : undefined }
        : undefined;
      // Ban list: an admin-removed worker (by key) may not re-enroll.
      if (pubkeyB64 && removedPubkeys.has(pubkeyB64)) { ws.send(JSON.stringify({ t: 'denied', reason: 'removed by admin' })); ws.close(); return; }
      const wantId = safeId(node.id); // sanitize (also prevents /metrics label injection)
      // Reject an id that's already live so a token-holder can't hijack another worker's identity/shards.
      if (workers.has(wantId)) { ws.send(JSON.stringify({ t: 'denied', reason: 'worker id already registered' })); log('warn', `worker rejected: duplicate id ${wantId}`); ws.close(); return; }
      id = wantId;
      let pubkey: CryptoKey | undefined;
      try { if (pubkeyB64) pubkey = await crypto.subtle.importKey('raw', b64d(pubkeyB64) as BufferSource, { name: 'Ed25519' }, false, ['verify']); } catch { /* worker without a valid key runs unsigned */ }
      registered = true; clearTimeout(authTimer);
      const wkey = await deriveWorkerKey(id, keyEpoch); // this worker's OWN sealing key — derived, never the master, never shared
      workers.set(id, { id, backend: safeId(node.backend), label: safeId(node.label), os: safeId(node.os), ws, load1: 0, util: 0, duty: DUTY_HINT, ceil: DUTY_HINT, busy: false, joinedAt: Date.now(), shards: 0, units: 0, ops: 0, tokens: 0, errors: 0, consecErrors: 0, healthyBeats: 0, busyCount: 0, totalMs: 0, lastOps: 0, history: [], pubkey, pubkeyB64, key: wkey, keyEpoch, paused: false, pausedReason: null, peer, reflexiveIp, caps });
      // welcome still carries the key under `tenantKeyB64` (wire-compatible with existing workers + the fake fleet),
      // but it is now this worker's PER-WORKER derived key, plus the key epoch it was minted at.
      ws.send(JSON.stringify({ t: 'welcome', tenantKeyB64: b64e(wkey), epoch: keyEpoch, duty: DUTY_HINT }));
      log('info', `worker joined: ${id} (${safeId(node.label)}, ${safeId(node.os)})${pubkey ? ' · signed' : ''} · fleet=${workers.size}`);
      pumpQueue();
    } else if (m.t === 'heartbeat') {
      if (!registered || !id) return; // ignore heartbeats before auth; trust only the socket's own id, not m.id
      const w = workers.get(id);
      if (w) {
        w.load1 = Number(m.load1) || 0; w.util = Number(m.util) || 0; w.duty = Number(m.duty) || w.duty;
        if (typeof m.ceil === 'number') w.ceil = m.ceil;
        // A coordinator-owned time-window schedule must NOT be clobbered by the worker's self-report (the torch
        // worker always reports "always"); non-window schedules ("always"/"idle-only") stay worker-reported.
        if (typeof m.schedule === 'string' && !isWindow(w.schedule)) w.schedule = m.schedule;
        if (w.pausedReason === 'errors') {
          // auto-recover: a worker that heartbeats healthy for a few beats is un-paused automatically (no operator needed)
          if (m.paused === false) {
            w.healthyBeats++;
            if (w.healthyBeats >= AUTO_RESUME_BEATS) { w.paused = false; w.pausedReason = null; w.consecErrors = 0; w.healthyBeats = 0; log('info', `auto-resumed ${w.id} after recovery`); pumpQueue(); }
          } else w.healthyBeats = 0;
        } else if (isWindow(w.schedule) || w.pausedReason === 'schedule') {
          // the COORDINATOR owns the pause state for a time-window schedule → ignore the worker's paused report here
        } else { // the worker's own schedule/admin pause state (always / idle-only)
          if (typeof m.paused === 'boolean') w.paused = m.paused;
          w.pausedReason = (m.pausedReason as string | null) ?? null;
          if (m.paused === false) pumpQueue();
        }
      }
    } else if (m.t === 'cached') {
      if (!registered) return;
      const e = pendingCache.get(String(m.id)); if (e && e.workerId === id) { pendingCache.delete(String(m.id)); e.cb({ ok: !!m.ok, error: m.error as string | undefined }); } // ignore a cached ack from a worker other than the home

    } else if (m.t === 'train_reply' || m.t === 'model_reply') {
      if (!registered) return;
      const pm = m.t === 'train_reply' ? pendingTrain : pendingModel;
      const e = pm.get(String(m.reqId));
      if (e && e.workerId === id) { pm.delete(String(m.reqId)); e.cb({ ok: !!m.ok, sealed: m.sealed as { iv: string; ct: string } | undefined, error: m.error as string | undefined }); } // reject a reply for another worker's RPC
    } else if (m.t === 'result') {
      if (!registered) return;
      const p = pending.get(String(m.shardId));
      if (!p || p.workerId !== id) return; // forged/foreign result for another worker's shard → ignored
      const w = workers.get(id);
      let signed = false;
      if (m.ok && m.sealedOut && w?.pubkey && typeof m.sig === 'string') {
        const blob = m.sealedOut as { iv: string; ct: string };
        const okSig = await crypto.subtle.verify({ name: 'Ed25519' }, w.pubkey, b64d(m.sig) as BufferSource, new TextEncoder().encode(`${m.shardId}|${blob.iv}|${blob.ct}`));
        if (!okSig) { log('warn', `result signature INVALID from ${id} — rejecting shard`); p.reject(new Error('invalid result signature')); return; }
        signed = true;
      }
      p.resolve({ ok: m.ok as boolean, sealedOut: m.sealedOut as { iv: string; ct: string } | undefined, error: m.error as string | undefined, backend: m.backend as string | undefined, ms: m.ms as number | undefined, signed });
    } else if (m.t === 'ring_ack') {
      // PEER TRANSPORT: a stage answered ring_wire with its successor's reachability (direct vs relay for that edge).
      if (!registered) return;
      const e = pendingRingWire.get(String(m.reqId));
      if (e && e.workerId === id) { pendingRingWire.delete(String(m.reqId)); e.resolve({ ok: true, reachable: !!m.reachable }); }
    } else if (m.t === 'complete') {
      // PEER TRANSPORT: the LAST stage delivered a finished token (sid,seq) directly — resolve the pending ring token.
      if (!registered) return;
      const ring = rings.get(String(m.sid));
      if (!ring || ring.epoch !== m.epoch) return; // stale epoch (a torn/re-wired pipe) → ignore
      const k = `${m.sid}|${m.seq}`, p = pendingRing.get(k);
      if (!p) return;
      pendingRing.delete(k);
      const st = ringStats.get(String(m.sid)); if (st) st.completes++;
      if (m.ok === false) p.reject(new Error(String(m.error ?? 'ring stage error')));
      else p.resolve({ argmax: m.argmax as number, ...(m.logits != null ? { logits: m.logits } : {}) });
    } else if (m.t === 'edge_fault') {
      // PEER TRANSPORT: a direct edge died mid-token → flip it to relay and tell the `from` stage to BRIDGE that
      // edge through the coordinator from now on (mixed-pipe), so the pipe keeps running instead of whole-pipe relay.
      if (!registered) return;
      const ring = rings.get(String(m.sid));
      if (ring && m.epoch === ring.epoch) {
        const e = ring.edges.find((x) => x.from === id);
        if (e) e.mode = 'relay';
        ring.allDirect = false;
        try { workers.get(id)?.ws.send(JSON.stringify({ t: 'ring_mode', sid: m.sid, epoch: ring.epoch, relay: true })); } catch { /* dropped socket → next token's pre-check falls back */ }
        log('warn', `ring ${m.sid}: edge ${id}→${m.succ} faulted (${m.error}) → relay; subsequent tokens BRIDGE this edge through the coordinator`);
      }
    } else if (m.t === 'bridge') {
      // PEER TRANSPORT (mixed pipe): a RELAY edge — the `from` stage handed us its signed+sealed activation; route
      // it into ITS successor as `bridge_in`. The coordinator sits on the data path for THIS edge only (the direct
      // edges stay worker->worker). We forward the predecessor's frame UNCHANGED, so the successor's Ed25519 origin
      // check against its predecessor's pubkey still holds — the coordinator can relay it but cannot forge it.
      if (!registered) return;
      const ring = rings.get(String(m.sid));
      if (!ring || ring.epoch !== m.epoch) return; // stale epoch (torn/re-wired pipe) → drop; token times out → falls back
      const edge = ring.edges.find((x) => x.from === id);
      if (!edge) return;
      const sw = workers.get(edge.to);
      const k = `${m.sid}|${m.seq}`;
      if (!sw) { const p = pendingRing.get(k); if (p) { pendingRing.delete(k); p.reject(new Error(`bridge: successor ${edge.to} gone`)); } return; }
      ringStat(String(m.sid)).bridged++; // PEER-TRANSPORT proof counter: one per-token activation BRIDGED (relay edge)
      try {
        sw.ws.send(JSON.stringify({ t: 'bridge_in', sid: m.sid, seq: m.seq, epoch: m.epoch, token: m.token,
          sealed: m.sealed, sig: m.sig, return_logits: m.return_logits,
          ...(m.cached ? { cached: true, session: m.session, pos: m.pos } : {}) }));
      } catch (e) { const p = pendingRing.get(k); if (p) { pendingRing.delete(k); p.reject(new Error(`bridge send to ${edge.to} failed: ${e}`)); } }
    } else if (m.t === 'moe_wire_ack') {
      // PEER TRANSPORT (MoE): the backbone answered moe_wire with whether every expert holder is reachable from it.
      if (!registered) return;
      const e = pendingMoeWire.get(String(m.reqId));
      if (e && e.workerId === id) { pendingMoeWire.delete(String(m.reqId)); e.resolve({ ok: true, reachable: !!m.reachable }); }
    } else if (m.t === 'moe_complete') {
      // PEER TRANSPORT (MoE): the backbone finished a whole forward — dispatching each layer's experts worker->worker
      // over the peer mesh — and delivered the final logits directly. The coordinator only injected + collected this.
      if (!registered) return;
      const mring = moeRings.get(String(m.sid));
      if (!mring || mring.epoch !== m.epoch) return; // stale epoch (a re-wired MoE ring) → ignore
      const k = `${m.sid}|${m.seq}`, p = pendingMoeRing.get(k);
      if (!p) return;
      pendingMoeRing.delete(k);
      const st = moeRingStats.get(String(m.sid)); if (st) st.completes++;
      if (m.ok === false) p.reject(new Error(String(m.error ?? 'moe ring stage error')));
      else p.resolve({ argmax: m.argmax as number, ...(m.logits != null ? { logits: m.logits } : {}) });
    }
  };
  ws.onclose = () => {
    clearTimeout(authTimer);
    if (!id) return;
    workers.delete(id);
    // reject any shard still in flight on this worker so the job fails fast instead of hanging the queue
    for (const [sid, p] of pending) if (p.workerId === id) { pending.delete(sid); p.reject(new Error(`worker ${id} disconnected mid-shard`)); }
    // same for in-flight train/model relay RPCs (else the awaiting HTTP handler hangs until the 120s timeout)
    for (const [rid, e] of pendingTrain) if (e.workerId === id) { pendingTrain.delete(rid); e.cb({ ok: false, error: `worker ${id} disconnected mid-rpc` }); }
    for (const [rid, e] of pendingModel) if (e.workerId === id) { pendingModel.delete(rid); e.cb({ ok: false, error: `worker ${id} disconnected mid-rpc` }); }
    // and any in-flight /weights cache ack destined for this worker (else that POST hangs for the full 30s)
    for (const [cid, e] of pendingCache) if (e.workerId === id) { pendingCache.delete(cid); e.cb({ ok: false, error: `worker ${id} disconnected before cache ack` }); }
    // drop weights that lived on this worker — they must be re-uploaded (the coordinator forgets its home)
    for (const [wid, home] of weightHome) if (home.worker === id) { weightHome.delete(wid); weightStore.delete(wid); }
    // forget any resident model / training session that lived here
    if (trainingHome === id) trainingHome = null;
    for (const [mid, wid] of modelHome) if (wid === id) modelHome.delete(mid);
    // A pipeline stage that lived here breaks the LIVE request, but a POST-LOAD shard can be HEALED instead of nuked:
    // KEEP the plan so the next shardPipe re-places this stage onto another torch worker (or waits for a reconnect).
    // A still-LOADING plan (empty stages) is handled by the load path's own reconnect/resume, so this only affects
    // READY plans — exactly the post-load failover case.
    for (const [sid, plan] of shardPlans) if (plan.stages.some((s) => s.worker === id)) log('warn', `worker ${id} held a stage of ready shard ${sid} — will re-place on next forward`);
    log('info', `worker left: ${id} · fleet=${workers.size}`);
  };
  ws.onerror = () => log('warn', `socket error${id ? ' from ' + id : ''}`);
}

// ---------- contribution trend sampler (per-worker + pool throughput sparklines) ----------
const poolHistory: number[] = [];
let lastPoolOps = 0;
setInterval(() => {
  let totalOps = 0;
  for (const w of workers.values()) {
    const delta = Math.max(0, w.ops - w.lastOps); w.lastOps = w.ops; // ops/interval — consistent across kernel + serving
    w.history.push(delta); if (w.history.length > 30) w.history.shift();
    totalOps += w.ops;
  }
  const pd = Math.max(0, totalOps - lastPoolOps); lastPoolOps = totalOps;
  poolHistory.push(pd); if (poolHistory.length > 30) poolHistory.shift();
}, 3000);

// ---------- coordinator-side schedule (time windows) ----------
// "always"/"idle-only" are honoured by the worker itself (self-reported via heartbeat). A time window
// "HH:MM-HH:MM" (the ON hours, may wrap past midnight, e.g. 22:00-07:00 = nights only) is evaluated HERE so it
// works for ANY worker type — the torch worker doesn't self-schedule — pausing/resuming the node as the wall
// clock enters/leaves the window. Set live from the admin panel; the coordinator owns the pause state for it.
const isWindow = (s?: string) => /^\d{1,2}:\d{2}-\d{1,2}:\d{2}$/.test((s ?? '').trim());
function scheduleOff(s: string): boolean {
  const [a, b] = s.split('-'); const [ah, am] = a.split(':').map(Number); const [bh, bm] = b.split(':').map(Number);
  if ([ah, am, bh, bm].some((n) => !Number.isFinite(n))) return false;
  const start = ah * 60 + am, end = bh * 60 + bm, now = new Date(), cur = now.getHours() * 60 + now.getMinutes();
  const active = start <= end ? (cur >= start && cur < end) : (cur >= start || cur < end); // wraps past midnight
  return !active;
}
function reconcileSchedules() {
  for (const w of workers.values()) {
    if (w.pausedReason === 'admin' || w.pausedReason === 'errors') continue; // admin/error pause outranks schedule
    if (isWindow(w.schedule)) {
      const off = scheduleOff(w.schedule!.trim());
      if (off && !(w.paused && w.pausedReason === 'schedule')) { w.paused = true; w.pausedReason = 'schedule'; log('info', `${w.id} scheduled-off (outside ${w.schedule})`); }
      else if (!off && w.pausedReason === 'schedule') { w.paused = false; w.pausedReason = null; log('info', `${w.id} scheduled-on (inside ${w.schedule})`); pumpQueue(); }
    } else if (w.pausedReason === 'schedule') { // schedule cleared back to always/idle-only → resume
      w.paused = false; w.pausedReason = null; pumpQueue();
    }
  }
}
setInterval(reconcileSchedules, 30_000);

// Backstop: fail any job that has sat un-schedulable in the queue too long, so a caller never hangs forever.
setInterval(() => {
  const now = Date.now();
  for (let i = queue.length - 1; i >= 0; i--) {
    const rec = queue[i];
    if (now - rec.submittedAt > STALE_JOB_MS) {
      queue.splice(i, 1); rec.status = 'failed'; rec.error = `stale: no worker available within ${Math.round(STALE_JOB_MS / 1000)}s`;
      jobInputs.delete(rec.id); jobSigned.delete(rec.id); // free the retained input buffers (runJob never ran for this job)
      M.jobsFailed++; log('warn', `${rec.id} failed: stale in queue (no active worker)`);
    }
  }
}, 15_000);

// ---------- kernels ----------
const ELEMENTWISE = new Set(['vector_add', 'vector_mul', 'saxpy', 'relu', 'scale', 'gelu']);
const geluF = (x: number) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
const ROWWISE = new Set(['softmax', 'layernorm']); // per-row reductions; sharded by whole rows
function cpuRowwise(kernel: string, a: Float32Array, cols: number): Float32Array {
  const rows = Math.floor(a.length / cols), o = new Float32Array(a.length);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    if (kernel === 'softmax') {
      let mx = -Infinity; for (let j = 0; j < cols; j++) mx = Math.max(mx, a[off + j]);
      let s = 0; for (let j = 0; j < cols; j++) { const e = Math.exp(a[off + j] - mx); o[off + j] = e; s += e; }
      for (let j = 0; j < cols; j++) o[off + j] /= s;
    } else {
      let m = 0; for (let j = 0; j < cols; j++) m += a[off + j]; m /= cols;
      let v = 0; for (let j = 0; j < cols; j++) { const d = a[off + j] - m; v += d * d; } v /= cols;
      const inv = 1 / Math.sqrt(v + 1e-5);
      for (let j = 0; j < cols; j++) o[off + j] = (a[off + j] - m) * inv;
    }
  }
  return o;
}
function cpuKernel(kernel: string, a: Float32Array, b: Float32Array | null, scalar: number): Float32Array {
  const n = a.length, o = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    switch (kernel) {
      case 'vector_add': o[i] = a[i] + (b as Float32Array)[i]; break;
      case 'vector_mul': o[i] = a[i] * (b as Float32Array)[i]; break;
      case 'saxpy': o[i] = scalar * a[i] + (b as Float32Array)[i]; break;
      case 'relu': o[i] = a[i] > 0 ? a[i] : 0; break;
      case 'scale': o[i] = a[i] * scalar; break;
      case 'gelu': o[i] = geluF(a[i]); break;
    }
  }
  return o;
}
function cpuMatmul(a: Float32Array, b: Float32Array, Mm: number, N: number, K: number): Float32Array {
  const o = new Float32Array(Mm * N);
  for (let i = 0; i < Mm; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < K; k++) s += a[i * K + k] * b[k * N + j]; o[i * N + j] = s; }
  return o;
}

// ---------- jobs + Slurm-like queue ----------
type JobStatus = 'queued' | 'running' | 'done' | 'failed';
interface JobRec { id: string; status: JobStatus; kernel: string; size: number; sealed: boolean; submittedAt: number; ms?: number; gflops?: number; verified?: boolean; signed?: boolean; shards?: { worker: string; backend: string; work: number; ms: number }[]; error?: string; dataMode?: boolean; output?: string; outLen?: number; }
/** Real input a caller submitted (data mode); the pool computes on THIS data and returns the output. */
interface JobInput { a?: Float32Array; b?: Float32Array; scalar?: number; M?: number; N?: number; K?: number; bRef?: string; }
const residentCount = (wid: string) => [...weightHome.values()].filter((h) => h.worker === wid).length;
const pendingCache = new Map<string, { workerId: string; cb: (r: { ok: boolean; error?: string }) => void }>(); // /weights waits for the worker's ack (tagged with the home worker so a disconnect can reject it)
// On-pool fine-tuning is relayed to ONE native (torch) worker that runs the whole train step locally.
// The coordinator seals/accounts the RPC like a shard but can't verify a stochastic loss (it has no
// torch) — correctness is checked out-of-band against a seeded reference (see examples/lora_finetune.py).
// Relayed RPCs to a native (torch) worker: on-pool fine-tuning ('train') and resident-model serving
// ('model'). The coordinator seals/accounts them like a shard but can't verify the result (it has no
// torch) — training is checked out-of-band vs a seeded reference; serving is checked vs transformers
// exact-match by the client (examples/llm_serve.py).
type RelayReply = { ok: boolean; sealed?: { iv: string; ct: string }; error?: string };
type RelayPending = { workerId: string; cb: (r: RelayReply) => void };
const pendingTrain = new Map<string, RelayPending>();
const pendingModel = new Map<string, RelayPending>();
let relaySeq = 0;
let trainingHome: string | null = null; // worker id hosting the single (non-DiLoCo) training session
// TODO(review): a worker LRU-evicts its own resident models/shards past MAX_RESIDENT_MODELS, but the
// coordinator is never told, so modelHome/shardPlans keep pointing at an id the worker no longer holds
// (a later /model/forward then errors "not loaded"). Fix: have the worker report evicted ids on load
// replies so the coordinator can drop the stale bookkeeping, or mirror the worker's cap in lockstep.
const modelHome = new Map<string, string>(); // resident model id → worker id
// Async model loads: a download-free push can take longer than a public tunnel will hold one HTTP request
// open (trycloudflare 502s a long request), so /model/load?async streams in the BACKGROUND and the caller
// polls GET /model/status. This is also the answer to "heavy models over a slow link" — the request is short.
type LoadState = { status: 'loading' | 'ready' | 'error'; worker: string; model: string; started: number; info?: Record<string, unknown>; error?: string };
const modelLoads = new Map<string, LoadState>(); // model id → last load's progress/outcome
// Same idea for a download-free SHARD load: streaming each stage's slice across the fleet can outlast a tunnel's
// request timeout, so /model/shard?async loads the stages in the background and the caller polls /model/shard_status.
type ShardLoadState = { status: 'loading' | 'ready' | 'error'; model: string; started: number; stagesDone: number; stagesTotal: number; info?: unknown; error?: string; aborted?: boolean };
const shardLoads = new Map<string, ShardLoadState>();
// Pipeline-parallel sharding (GPT-2 family): a model's transformer layers are split into contiguous
// STAGES, one per worker — each worker holds ONLY its stage resident. The plan records, per shard id,
// the ordered stages (which worker owns which layer range + which is first/last); /model/shard_forward
// pipes the hidden state stage→stage (only [seq×hidden] activations cross the wire, never the weights).
interface ShardStage { worker: string; start: number; end: number; first: boolean; last: boolean }
const shardPlans = new Map<string, { model: string; stages: ShardStage[]; fp16?: boolean; quant?: string }>();
// POST-LOAD failover keeps the download-free streaming inputs beside each ready plan, so ONE stage can be re-streamed
// to a fresh worker without redoing the /model/shard preflight. push:false shards carry nulls. Lifecycle = shardPlans.
type ShardStream = { model: string; push: boolean; configText: string | null; stPlan: STPlan | null };
const shardStreams = new Map<string, ShardStream>();
// EXPERT PARALLELISM (MoE) hybrid plan — see docs/ROADMAP.md "MoE expert parallelism (the Kimi path)". The DENSE
// backbone (embed / attention / router / final norm+head) lives on ONE worker; the routed experts are SPLIT across
// holder workers (each resident for a SUBSET of expert indices, all layers). Sibling of shardPlans. /model/moe_forward
// drives one forward layer-by-layer, RELAYING the router dispatch/combine THROUGH THE COORDINATOR — no peer mesh yet
// (that all-to-all is the SPOF the mesh removes; a LATER increment, roadmap §4). Only [seq×hidden] activations + the
// tiny routing table cross the wire, never the weights.
interface MoEHolder { worker: string; experts: number[] }
interface MoEPlan { model: string; backbone: string; holders: MoEHolder[]; nLayer: number; nExperts: number; topk: number }
const moePlans = new Map<string, MoEPlan>();
// Generous by default: model_load/train_load download & instantiate a model on the worker, which for a
// heavy model or a slow link can far exceed 2min. Fine-tuning + big-model inference need this headroom.
const RELAY_TIMEOUT_MS = Number(Deno.env.get('MOREGPU_TRAIN_TIMEOUT_MS') ?? 600_000);
/** Native torch workers can fine-tune (autograd) and hold a whole model resident; WebGPU workers cannot. */
const torchWorkers = () => activeFleet().filter((w) => w.label.includes('torch'));
// Workers that can host a pipeline SHARD stage (a decoder-block slice) — torch OR a WebGPU worker with a WGSL
// middle-stage runtime. The end stages (embeddings/tokenizer/head) additionally need 'shardEnds'. Used to place
// shards; train/resident-serve/MoE stay on torchWorkers() (autograd / whole-model residency).
const shardWorkers = () => activeFleet().filter((w) => w.caps.has('shard'));
async function relayRPC(w: Worker, kind: 'train' | 'model', pend: Map<string, RelayPending>, op: string, payload: unknown): Promise<{ ok: boolean; data?: Record<string, unknown>; error?: string }> {
  // Bail fast if the socket is already closed: a chunk sent AFTER onclose ran (e.g. the next push in a multi-chunk
  // stream, once onclose rejected the in-flight one) would otherwise sit in `pend` for the full RELAY_TIMEOUT_MS —
  // onclose won't fire again to reject it. Surfacing the disconnect immediately is what lets a stage retry promptly.
  if (w.ws.readyState !== WebSocket.OPEN) return { ok: false, error: `worker ${w.id} disconnected` };
  const reqId = `${kind}-${++relaySeq}`;
  const sealed = await seal(w.key, new TextEncoder().encode(JSON.stringify(payload))); // this worker's per-worker key
  // Bind the reply to THIS worker's id, so another worker can't resolve someone else's RPC (see train_reply handler).
  // TODO(review): a timeout here only abandons the coordinator's wait — the worker keeps running the compute
  // (and, for a train op, may still commit optimizer/step state). Needs a cancel/epoch protocol (send an
  // abort frame, or a generation counter the worker checks between steps) so a timed-out round is truly voided.
  const ack = new Promise<RelayReply>((res) => { const t = setTimeout(() => { pend.delete(reqId); res({ ok: false, error: `${kind} rpc timeout` }); }, RELAY_TIMEOUT_MS); pend.set(reqId, { workerId: w.id, cb: (r) => { clearTimeout(t); res(r); } }); });
  // Resident serving + training run through this relay, NOT the kernel-shard path — so without accounting here
  // they'd never show in the per-worker contribution / trend (flat while you chat). Mark the node busy for the
  // op's duration; on success credit ONE op (the consistent activity currency) for actual COMPUTE ops only —
  // never for weight-transfer/control ops (push_*/load/unload/arch/shard_unload), which the worker only
  // receives, not computes (crediting those would make a downloading node look like a top contributor).
  w.busyCount++; w.busy = true;
  try {
    w.ws.send(JSON.stringify({ t: kind, reqId, op, sealed }));
  } catch (e) { w.busyCount = Math.max(0, w.busyCount - 1); w.busy = w.busyCount > 0; return { ok: false, error: `send failed: ${e}` }; }
  try {
    const r = await ack;
    if (!r.ok) return { ok: false, error: r.error };
    if (!r.sealed) return { ok: true, data: {} };
    let data: Record<string, unknown>;
    try { data = JSON.parse(new TextDecoder().decode(await unseal(w.key, r.sealed))) as Record<string, unknown>; }
    catch (e) { return { ok: false, error: `bad ${kind} reply: ${e}` }; }
    // NB: keep w.totalMs (→ avgMs) shard-only — mixing serve/train ms into a kernel-shard average corrupts it.
    if (SERVE_OPS.has(op)) { w.ops++; w.tokens += Math.max(0, Math.round(Number(data.n) || 0)); }
    else if (TRAIN_OPS.has(op)) { w.ops++; }
    return { ok: true, data };
  } finally { w.busyCount = Math.max(0, w.busyCount - 1); w.busy = w.busyCount > 0; }
}
// COMPUTE relay ops that count toward a node's contribution (one call = one op). Serving forwards/decodes and
// pipeline-shard stages, plus training inner-loops/steps. Everything else on the relay is transfer/setup.
const SERVE_OPS = new Set(['forward', 'generate', 'chat', 'shard_forward']);
const TRAIN_OPS = new Set(['inner', 'step']);
const trainRPC = (w: Worker, op: string, payload: unknown) => relayRPC(w, 'train', pendingTrain, op, payload);
const modelRPC = (w: Worker, op: string, payload: unknown) => relayRPC(w, 'model', pendingModel, op, payload);

// ── DOWNLOAD-FREE resident load: the COORDINATOR (admin box) fetches the weights from HF over HTTPS and
// streams the raw file bytes to the worker; the worker stages them (in RAM when it has /dev/shm — Linux/Colab
// — else a transient temp dir) and loads with the hub disabled, so it never downloads anything. This keeps
// "all weights on the admin, fleet = pure GPU compute". It's a ONE-TIME transfer (unlike the per-token
// pipeline pipe), so it tolerates a slow/flaky tunnel far better.
const HF_BASE = Deno.env.get('MOREGPU_HF_BASE') ?? 'https://huggingface.co';
// 4 MiB raw / chunk. Smaller than you'd pick for pure throughput on purpose: each acked chunk is PERSISTED on the
// worker, so on a churning link this is the RESUME granularity (a dropped stage restarts from the last flushed
// chunk, not from zero) and a smaller frame is likelier to cross a flaky tunnel intact. Raise on a fast/LAN link.
const PUSH_CHUNK = Math.max(1 << 18, Number(Deno.env.get('MOREGPU_PUSH_CHUNK') ?? (4 << 20)));
// Safety rail against a runaway/oversized repo OOM-ing the coordinator or a donated worker (default 20 GiB,
// env-tunable). The index.json is metadata (KB) so it gets a much tighter cap of its own.
const PUSH_MAX_BYTES = Math.max(1 << 20, Number(Deno.env.get('MOREGPU_PUSH_MAX_BYTES') ?? (20 * (1 << 30))));
const PUSH_INDEX_MAX = 64 << 20; // a safetensors index is normally KB; refuse a multi-GB one before .text()
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
// Churn tolerance for a download-free shard: a stage worker that drops mid-stream (slow/flaky tunnel, Colab blip)
// reconnects under the same id via the keepalive, so we WAIT for it and re-stream just that stage instead of
// nuking a 10-minute load. Bounded by tries-per-stage and a per-reconnect wait (both under the overall deadline).
const SHARD_STAGE_TRIES = Math.max(1, Number(Deno.env.get('MOREGPU_SHARD_STAGE_TRIES') ?? 6));
const SHARD_RECONNECT_WAIT_MS = Math.max(0, Number(Deno.env.get('MOREGPU_SHARD_RECONNECT_WAIT_MS') ?? 180_000));
// model id must be a plain HF repo ref (owner/name or a canonical alias) — it goes straight into a URL.
const HF_REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*(\/[A-Za-z0-9][A-Za-z0-9._-]*)?$/;
// the finite set of hub files a causal-LM load may need; each is fetched, and a 404 is simply skipped.
const HF_AUX_FILES = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
  'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json', 'tokenizer.model', 'chat_template.jinja'];
const pushInFlight = new Set<string>(); // `${workerId}:${id}` currently being pushed — blocks a concurrent same-id race

async function hfFetch(model: string, file: string): Promise<Response | null> {
  const headers: Record<string, string> = {};
  const tok = Deno.env.get('HF_TOKEN'); if (tok) headers['authorization'] = `Bearer ${tok}`;
  const r = await fetch(`${HF_BASE}/${model}/resolve/main/${encodeURIComponent(file)}?download=true`, { headers, redirect: 'follow' });
  if (r.status === 404 || r.status === 403) { await r.body?.cancel(); return null; }
  if (!r.ok) { await r.body?.cancel(); throw new Error(`HF ${file} → HTTP ${r.status}`); }
  return r;
}

// Read a small metadata file into a string, but refuse (before buffering) if it declares more than `max` bytes.
async function hfFetchText(model: string, file: string, max: number): Promise<string | null> {
  const r = await hfFetch(model, file);
  if (!r) return null;
  const len = Number(r.headers.get('content-length') ?? 0);
  if (len > max) { await r.body?.cancel(); throw new Error(`${file} is ${len} bytes (> ${max} cap) — refusing to buffer`); }
  const buf = new Uint8Array(await r.arrayBuffer());
  if (buf.length > max) throw new Error(`${file} exceeded ${max} bytes — refusing`);
  return new TextDecoder().decode(buf);
}

async function streamFileToWorker(w: Worker, id: string, name: string, resp: Response, budget: { left: number }): Promise<number> {
  const reader = resp.body!.getReader();
  let pending = new Uint8Array(0), seq = 0, total = 0;
  const send = async (data: Uint8Array, last: boolean) => {
    const r = await modelRPC(w, 'push_chunk', { id, name, seq: seq++, data: b64e(data), last });
    if (!r.ok) throw new Error(`push_chunk ${name}#${seq}: ${r.error}`);
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length; budget.left -= value.length;
    if (budget.left < 0) { await reader.cancel(); throw new Error(`weight push exceeded ${PUSH_MAX_BYTES}-byte cap (raise MOREGPU_PUSH_MAX_BYTES) — aborted`); }
    const merged = new Uint8Array(pending.length + value.length); merged.set(pending); merged.set(value, pending.length);
    pending = merged;
    while (pending.length >= PUSH_CHUNK) { await send(pending.slice(0, PUSH_CHUNK), false); pending = pending.slice(PUSH_CHUNK); }
  }
  await send(pending, true); // final chunk (may be empty) → guarantees the file is created even if 0 bytes
  return total;
}

// finalize=true (default): push_end assembles a RESIDENT serving model + drops staging. finalize=false: stage
// the files ONLY and leave them for the caller's own loader (download-free TRAINING reads the staged dir in
// train_load, then cleans up) — the difference between "serve this" and "fine-tune this without downloading".
async function pushModelToWorker(w: Worker, model: string, id: string, fp16: boolean, finalize = true): Promise<Record<string, unknown>> {
  if (!HF_REPO_RE.test(model)) throw new Error(`bad model ref "${model}" — expected an HF repo id like "gpt2" or "Qwen/Qwen2.5-0.5B"`);
  const guard = `${w.id}:${id}`;
  if (pushInFlight.has(guard)) throw new Error(`a download-free push for "${id}" is already in progress on ${w.id} — wait for it to finish`);
  pushInFlight.add(guard);
  try {
    // resolve the weight files: a sharded checkpoint lists them in an index; otherwise it's a single safetensors.
    // The index itself MUST also be staged — transformers discovers shards only through model.safetensors.index.json.
    let weightFiles: string[];
    const indexText = await hfFetchText(model, 'model.safetensors.index.json', PUSH_INDEX_MAX);
    if (indexText !== null) {
      const idx = JSON.parse(indexText) as { weight_map?: Record<string, string> };
      weightFiles = [...new Set(Object.values(idx.weight_map ?? {}))];
      if (weightFiles.length === 0) throw new Error(`${model}: empty safetensors index`);
    } else {
      weightFiles = ['model.safetensors'];
    }
    const begin = await modelRPC(w, 'push_begin', { id, model });
    if (!begin.ok) throw new Error(`push_begin failed: ${begin.error}`);
    const budget = { left: PUSH_MAX_BYTES };
    try {
      let bytes = 0, sentWeights = 0;
      for (const f of HF_AUX_FILES) {            // config + tokenizer/generation files (skip any that 404)
        const resp = await hfFetch(model, f);
        if (resp) bytes += await streamFileToWorker(w, id, f, resp, budget);
      }
      if (indexText !== null) {                  // stage the shard index too, so from_pretrained can find the shards
        const b = new TextEncoder().encode(indexText);
        const r = await modelRPC(w, 'push_chunk', { id, name: 'model.safetensors.index.json', seq: 0, data: b64e(b), last: true });
        if (!r.ok) throw new Error(`push_chunk index: ${r.error}`); bytes += b.length;
      }
      for (const f of weightFiles) {             // the actual weights (single or sharded safetensors)
        const resp = await hfFetch(model, f);
        if (!resp) throw new Error(`weight file ${f} missing on HF (no safetensors?) — this model can't be pushed download-free`);
        bytes += await streamFileToWorker(w, id, f, resp, budget); sentWeights++;
      }
      if (sentWeights === 0) throw new Error(`${model}: no weight files found`);
      log('info', `weight-push ${model} → ${w.id}: streamed ${(bytes / 1e6).toFixed(1)} MB (${weightFiles.length} shard(s))${finalize ? ', assembling…' : ' (staged for training)'}`);
      if (!finalize) return { staged: true, id, bytes };  // stage-only: the caller (train_load push) loads the staged dir
      const end = await modelRPC(w, 'push_end', { id, model, fp16 });
      if (!end.ok) throw new Error(`push_end failed: ${end.error}`);
      return end.data ?? {};
    } catch (e) {
      await modelRPC(w, 'unload', { id }).catch(() => {}); // best-effort: drop any partial staging on the worker
      throw e;
    }
  } finally {
    pushInFlight.delete(guard);
  }
}

// ── DOWNLOAD-FREE PIPELINE SHARDING: the coordinator fetches the model ONCE and streams each worker only the
// tensors for ITS stage (its contiguous layer slice + embeddings on the first stage + final norm/head on the
// last), rebuilt as a small per-stage safetensors. The fleet never touches the HF hub. A layer tensor is any
// weight whose name contains ".h.<i>." (GPT-2) or ".layers.<i>." (Llama-family); everything else is non-layer
// (embeddings / final norm / lm_head) and goes to the end stages only.
type STHeader = Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
// A tensor's RESOLVED source: which safetensors file it lives in, that file's data-region start (8 + its
// own header length), and its byte offsets WITHIN that file. A single-file model and a multi-file (sharded)
// model both resolve to one of these maps, so the staging code below is agnostic to how many files back it.
type TensorLoc = { dtype: string; shape: number[]; data_offsets: [number, number]; file: string; dataStart: number };
type STPlan = Record<string, TensorLoc>;
const LAYER_RE = /(?:^|\.)(?:h|layers)\.(\d+)\./;

async function hfFetchRange(model: string, file: string, start: number, end: number): Promise<Uint8Array> {
  const headers: Record<string, string> = { range: `bytes=${start}-${end}` };
  const tok = Deno.env.get('HF_TOKEN'); if (tok) headers['authorization'] = `Bearer ${tok}`;
  const r = await fetch(`${HF_BASE}/${model}/resolve/main/${encodeURIComponent(file)}?download=true`, { headers, redirect: 'follow' });
  if (!r.ok && r.status !== 206) { await r.body?.cancel(); throw new Error(`HF range ${file} [${start}-${end}] → HTTP ${r.status}`); }
  let buf = new Uint8Array(await r.arrayBuffer());
  if (r.status === 200 && buf.length > end - start + 1) buf = buf.slice(start, end + 1); // server ignored Range → slice locally
  return buf;
}

async function hfSafetensorsHeader(model: string, file: string): Promise<{ header: STHeader; headerLen: number }> {
  const lenBuf = await hfFetchRange(model, file, 0, 7);
  if (lenBuf.length < 8) throw new Error(`${file}: could not read safetensors header length`);
  const headerLen = Number(new DataView(lenBuf.buffer, lenBuf.byteOffset, 8).getBigUint64(0, true));
  if (!(headerLen > 0 && headerLen < (256 << 20))) throw new Error(`${file}: implausible safetensors header length ${headerLen}`);
  const hb = await hfFetchRange(model, file, 8, 8 + headerLen - 1);
  const parsed = JSON.parse(new TextDecoder().decode(hb)) as Record<string, unknown>;
  delete parsed['__metadata__'];
  return { header: parsed as STHeader, headerLen };
}

// Resolve a model's weights into ONE tensor→location plan. Fast path: a single model.safetensors (one header
// fetch). Sharded path: model.safetensors.index.json's weight_map (tensorName→fileName) — fetch EACH referenced
// file's header via Range and merge, tagging every tensor with its file + that file's data-region start. Key
// order follows weight_map insertion order, so the per-stage byte layout stays deterministic across resumes.
async function hfSafetensorsPlan(model: string, indexText: string | null): Promise<STPlan> {
  const plan: STPlan = {};
  if (indexText === null) { // single-file fast path — same tensors, same order as before
    const { header, headerLen } = await hfSafetensorsHeader(model, 'model.safetensors');
    const dataStart = 8 + headerLen;
    for (const [name, t] of Object.entries(header)) plan[name] = { dtype: t.dtype, shape: t.shape, data_offsets: t.data_offsets, file: 'model.safetensors', dataStart };
    return plan;
  }
  const idx = JSON.parse(indexText) as { weight_map?: Record<string, string> };
  const weightMap = idx.weight_map ?? {};
  const files = [...new Set(Object.values(weightMap))];
  if (files.length === 0) throw new Error(`${model}: empty safetensors index (no weight_map)`);
  const headers = new Map<string, { header: STHeader; dataStart: number }>();
  for (const f of files) { const { header, headerLen } = await hfSafetensorsHeader(model, f); headers.set(f, { header, dataStart: 8 + headerLen }); }
  for (const [name, f] of Object.entries(weightMap)) {
    const h = headers.get(f); if (!h) throw new Error(`${model}: index maps ${name}→${f} but that file has no header`);
    const t = h.header[name]; if (!t) throw new Error(`${model}: ${name} is in the index but absent from ${f}'s header`);
    plan[name] = { dtype: t.dtype, shape: t.shape, data_offsets: t.data_offsets, file: f, dataStart: h.dataStart };
  }
  return plan;
}

function stageTensors(header: STPlan, start: number, end: number, first: boolean, last: boolean): string[] {
  const names: string[] = [];
  for (const name of Object.keys(header)) {
    const m = LAYER_RE.exec(name);
    if (m) { const i = Number(m[1]); if (i >= start && i < end) names.push(name); }   // this stage's decoder blocks
    else if (first || last) names.push(name);                                          // embeddings / final norm / head
  }
  return names;
}

// ── EXPERT PARALLELISM (MoE) tensor selection — see docs/ROADMAP.md "MoE expert parallelism (the Kimi path)".
// stageTensors is LAYER-only (it has no notion of an expert index); these add EXPERT-index selection so a stage
// can be assigned a SUBSET of experts. The LOADER is untouched: hfSafetensorsPlan already tagged every tensor
// with its {file, offsets}, so selecting one expert's (possibly cross-file) tensors is pure selection — the
// UNCHANGED streamStageSafetensors Range-fetches each from its own file and merges them into one per-node file.
// Grounded in real routed-MoE naming (OLMoE / Qwen2-MoE / DeepSeek-V3-Kimi):
//   routed expert  = "….mlp.experts.<E>.{gate,up,down}_proj.weight"   → EXPERT plane (split across holders)
//   router         = "….mlp.gate.weight"     ┐
//   attention/norms= "….self_attn.*" / "….*layernorm.*"              ├ DENSE backbone (one worker, EP×pipe hybrid)
//   embed/head/norm= "model.embed_tokens.*" / "…norm.*" / "lm_head.*" ┘
const EXPERT_RE = /(?:^|\.)experts\.(\d+)\./;              // captures the routed-expert index E
function isExpertTensor(name: string): boolean { return EXPERT_RE.test(name); }
// Backbone = every tensor that is NOT a routed expert (the router mlp.gate and any shared expert STAY dense).
function stageBackboneTensors(plan: STPlan): string[] { return Object.keys(plan).filter((n) => !isExpertTensor(n)); }
// Expert selection = the exact routed-expert tensors whose index is in `experts`, across ALL layers (a holder
// keeps the same expert-index subset for every layer). One expert's 3 tensors may live in DIFFERENT source files.
function stageExpertTensors(plan: STPlan, experts: Set<number>): string[] {
  return Object.keys(plan).filter((n) => { const m = EXPERT_RE.exec(n); return !!m && experts.has(Number(m[1])); });
}

// Build a valid per-stage safetensors (recomputed contiguous offsets) and stream it as "model.safetensors".
// The selected tensors may come from ONE source file or from MANY (a sharded index) — each is Range-fetched
// from its own file via the plan — but the output is always a single merged per-stage file for the worker.
// `resumeFrom` = bytes already staged on the worker (from a dropped attempt): the byte layout is deterministic
// (same tensor order → same header), so we skip whole segments before the offset (no HF re-fetch) and append only
// the tail — a churned stage makes forward progress across reconnects instead of restarting from zero.
async function streamStageSafetensors(w: Worker, id: string, model: string, plan: STPlan, names: string[], budget: { left: number }, resumeFrom = 0): Promise<number> {
  const newHeader: STHeader = {}; const parts: { file: string; srcStart: number; len: number }[] = []; let off = 0;
  for (const name of names) {
    const t = plan[name]; const len = t.data_offsets[1] - t.data_offsets[0];
    newHeader[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + len] };
    // each tensor's bytes come from ITS OWN file (single-file → all the same file; sharded → mixed)
    parts.push({ file: t.file, srcStart: t.dataStart + t.data_offsets[0], len }); off += len;
  }
  let hstr = JSON.stringify(newHeader);
  while ((8 + new TextEncoder().encode(hstr).length) % 8 !== 0) hstr += ' '; // safetensors: header padded so data is 8-byte aligned
  const hbytes = new TextEncoder().encode(hstr);
  const prefix = new Uint8Array(8); new DataView(prefix.buffer).setBigUint64(0, BigInt(hbytes.length), true);
  // chunked push of: [8-byte len][header][each tensor's bytes, in order] → appended into the worker's model.safetensors
  let pending = new Uint8Array(0), seq = 0, total = 0, pos = 0; // pos = absolute offset in the full stream
  const push = async (data: Uint8Array, last: boolean) => { const r = await modelRPC(w, 'push_chunk', { id, name: 'model.safetensors', seq: seq++, data: b64e(data), last }); if (!r.ok) throw new Error(`push_chunk safetensors: ${r.error}`); };
  const feed = async (b: Uint8Array) => {
    total += b.length; budget.left -= b.length;
    if (budget.left < 0) throw new Error(`shard push exceeded ${PUSH_MAX_BYTES}-byte cap (raise MOREGPU_PUSH_MAX_BYTES)`);
    const merged = new Uint8Array(pending.length + b.length); merged.set(pending); merged.set(b, pending.length); pending = merged;
    while (pending.length >= PUSH_CHUNK) { await push(pending.slice(0, PUSH_CHUNK), false); pending = pending.slice(PUSH_CHUNK); }
  };
  // Emit one stream segment [pos, pos+len). Fully before resumeFrom → skip (and skip the HF fetch); straddling →
  // fetch and append only the tail; fully after → append whole.
  const emit = async (len: number, get: () => Promise<Uint8Array>) => {
    const segStart = pos; pos += len;
    if (pos <= resumeFrom) return;
    let bytes = await get();
    if (segStart < resumeFrom) bytes = bytes.slice(resumeFrom - segStart);
    await feed(bytes);
  };
  await emit(prefix.length, () => Promise.resolve(prefix));
  await emit(hbytes.length, () => Promise.resolve(hbytes));
  for (const p of parts) await emit(p.len, () => hfFetchRange(model, p.file, p.srcStart, p.srcStart + p.len - 1)); // Range-fetch just this tensor from its file
  await push(pending, true);
  return resumeFrom + total; // full staged size (already-there + this attempt's tail)
}

// Run ONE forward across a shard plan: stage 0 embeds input_ids → hidden; each next stage runs its blocks on
// the piped hidden; the last returns {argmax, logits?}. Only activations cross the wire. Re-checks each stage's
// worker is still connected (so a node churning mid-generation surfaces cleanly instead of hanging).
//
// KV CACHE (`cache`): pass {session,pos} to run in cached mode — each stage keeps its own layers' past_key_values
// keyed by (sid,session). A pos-0 call PREFILLS the whole prompt; a pos>0 call is a DECODE step where `input_ids`
// is just the ONE new token (first stage) and the piped hidden is a single position — so decode stops re-running
// the growing prefix (O(n) not O(n^2)). `pos` = tokens already cached; the stage rejects a pos mismatch (its KV was
// lost to a reconnect → re-prefill needed). Uncached (no `cache`) is the original stateless full-sequence pipe.
// `seq` is the FULL token sequence so far — carried so a POST-LOAD failover (a re-placed stage has no KV) can reset
// the session and RE-PREFILL the whole sequence instead of feeding one token into an empty cache.
type ShardCache = { session: string; pos: number; seq?: number[] };

// (Re)stream ONE stage's slice to a worker and shard_load it. Shared by the INITIAL /model/shard load and the
// POST-LOAD failover re-placement, so a single stage can be (re)streamed to a NEW worker through the same path.
// `resume` keeps a worker's partial staging (a mid-stream drop on the SAME worker → re-stream only the tail); a fresh
// replacement worker passes resume=false. Any error → {ok:false}. (Formerly the load path's `loadStage` closure.)
async function streamStageToWorker(sid: string, model: string, push: boolean, configText: string | null, stPlan: STPlan | null, st: ShardStage, w: Worker, resume: boolean): Promise<{ ok: boolean; data?: Record<string, unknown>; error?: string; bytes: number }> {
  const fp16 = !!shardPlans.get(sid)?.fp16;  // fp16 halves each stage's footprint on a GPU worker (plan-wide; failover reads it too)
  const quant = shardPlans.get(sid)?.quant;   // int8/nf4 (non-push only) — the worker quantizes on load; CUDA-gated worker-side
  if (!push) { const r = await modelRPC(w, 'shard_load', { model, id: sid, start: st.start, end: st.end, first: st.first, last: st.last, fp16, quant }); return { ...r, bytes: 0 }; }
  try {
    const begin = await modelRPC(w, 'push_begin', { id: sid, model, resume }); if (!begin.ok) throw new Error(`push_begin: ${begin.error}`);
    const sizes = (begin.data?.sizes ?? {}) as Record<string, number>;
    if (resume && Number(sizes['model.safetensors']) > 0) log('info', `shard ${sid} stage ${st.worker}: resuming — ${(Number(sizes['model.safetensors']) / 1e6).toFixed(1)}MB already staged, streaming the rest`);
    const budget = { left: PUSH_MAX_BYTES };
    if (!(Number(sizes['config.json']) > 0)) {
      const cb = new TextEncoder().encode(configText!);
      const cr = await modelRPC(w, 'push_chunk', { id: sid, name: 'config.json', seq: 0, data: b64e(cb), last: true }); if (!cr.ok) throw new Error(`push config: ${cr.error}`);
    }
    if (st.first) {
      for (const f of ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json', 'tokenizer.model', 'chat_template.jinja', 'generation_config.json']) {
        if (Number(sizes[f]) > 0) continue;
        const tr = await hfFetch(model, f); if (tr) await streamFileToWorker(w, sid, f, tr, budget);
      }
    }
    const names = stageTensors(stPlan!, st.start, st.end, st.first, st.last);
    const bytes = await streamStageSafetensors(w, sid, model, stPlan!, names, budget, Number(sizes['model.safetensors']) || 0);
    const r = await modelRPC(w, 'shard_load', { model, id: sid, start: st.start, end: st.end, first: st.first, last: st.last, push: true, fp16 });
    return { ...r, bytes };
  } catch (e) { return { ok: false, error: e instanceof Error ? e.message : String(e), bytes: 0 }; }
}

// POST-LOAD failover for ONE stage whose worker dropped after the shard was ready: re-place its layer range onto
// another connected torch worker not already carrying a stage of this plan (or, if none free, bounded-wait for a
// spare / the same id to reconnect), (re)stream it download-free via streamStageToWorker, and swap the worker in the
// live plan. The caller (shardPipe) then restarts the forward on the healed plan.
async function replaceStage(sid: string, failedIdx: number): Promise<{ ok: boolean; error?: string }> {
  const plan = shardPlans.get(sid);
  if (!plan || plan.stages.length === 0) return { ok: false, error: `shard ${sid} not loaded/ready` };
  const meta = shardStreams.get(sid);
  if (!meta) return { ok: false, error: `shard ${sid}: no stream metadata — cannot re-place a stage (re-shard)` };
  const st = plan.stages[failedIdx];
  const others = new Set(plan.stages.filter((_, i) => i !== failedIdx).map((s) => s.worker));
  // Re-place onto a shard-capable spare; a first/last stage additionally needs 'shardEnds' (embeddings/head).
  const needEnds = st.first || st.last;
  const pick = () => shardWorkers().find((w) => !others.has(w.id) && (!needEnds || w.caps.has('shardEnds')));
  let target = pick();
  if (!target) {
    log('info', `shard ${sid}: no spare torch worker for stage ${failedIdx} (layers ${st.start}-${st.end}) — waiting up to ${SHARD_RECONNECT_WAIT_MS}ms`);
    const until = Date.now() + SHARD_RECONNECT_WAIT_MS;
    while (!target && Date.now() < until && shardPlans.has(sid)) { await sleep(1500); target = pick(); }
  }
  if (!target) return { ok: false, error: `stage worker ${st.worker} (layers ${st.start}-${st.end}) gone and no torch worker to re-place it within ${SHARD_RECONNECT_WAIT_MS}ms` };
  const old = workers.get(st.worker);
  if (old && old.id !== target.id) await modelRPC(old, 'shard_unload', { id: sid }).catch(() => {});
  const newStage: ShardStage = { worker: target.id, start: st.start, end: st.end, first: st.first, last: st.last };
  const r = await streamStageToWorker(sid, meta.model, meta.push, meta.configText, meta.stPlan, newStage, target, false);
  if (!r.ok) return { ok: false, error: `re-place stage ${failedIdx} (layers ${st.start}-${st.end}) onto ${target.id} failed: ${r.error}` };
  const cur = shardPlans.get(sid); if (!cur) return { ok: false, error: `shard ${sid} unloaded during re-placement` };
  cur.stages[failedIdx] = newStage;
  log('info', `shard ${sid}: re-placed stage ${failedIdx} (layers ${st.start}-${st.end}) ${st.worker} → ${target.id} after post-load disconnect`);
  return { ok: true };
}

// ============================================================================
// PEER TRANSPORT (opt-in via MOREGPU_PEER_TRANSPORT=1): take the coordinator OFF the per-token data path.
// After a shard loads, wireRing hands each stage its successor's LAN endpoint + the predecessor's pubkey and
// probes reachability. When every edge is `direct`, ringPipe injects stage 0 and awaits the last stage's
// `complete` — the hidden state travels worker->worker (sealed with the shared tenant key, Ed25519-signed by the
// predecessor), NEVER re-crossing the coordinator between stages. Any non-reachable edge (or a cached/KV request)
// falls back to the unchanged relay path (shardPipe). Reuses shardPlans/ShardStage, per-edge session keys (via the workers'
// own seal), each worker's registered pubkey, tokenB64url, and the existing control WS.
interface RingEdge { from: string; to: string; mode: 'direct' | 'relay' }
interface Ring { sid: string; epoch: number; edges: RingEdge[]; token: string; allDirect: boolean; torn?: boolean }
const rings = new Map<string, Ring>();
const pendingRing = new Map<string, { resolve: (r: unknown) => void; reject: (e: Error) => void }>(); // `${sid}|${seq}` → token promise
const pendingRingWire = new Map<string, { workerId: string; resolve: (r: { ok: boolean; reachable?: boolean }) => void }>();
// per-sid proof counters. `shardForwards` = per-token activations RELAYED whole-pipe through the coordinator (shardPipe);
// `bridged` = per-token activations the coordinator relayed for a single RELAY edge of a MIXED peer pipe (the direct
// edges of that same pipe stay worker->worker). On an all-direct peer path both stay 0 while injects/completes climb.
const ringStats = new Map<string, { injects: number; completes: number; shardForwards: number; bridged: number }>();
let ringSeq = 0;
const ringStat = (sid: string) => { let s = ringStats.get(sid); if (!s) { s = { injects: 0, completes: 0, shardForwards: 0, bridged: 0 }; ringStats.set(sid, s); } return s; };
// TEST/OPS knob: worker ids whose OUTGOING ring edge is forced onto the relay-bridge path even when the peer probe
// succeeds — lets a test deterministically exercise a MIXED pipe (some direct, some bridged) without racing on real
// network reachability. Empty by default → mode is decided purely by the reachability probe.
const FORCE_RELAY = new Set((Deno.env.get('MOREGPU_PEER_FORCE_RELAY') ?? '').split(',').map((s) => s.trim()).filter(Boolean));

// Send a stage its successor endpoint + ring token and await its reachability ack (a control-WS RPC, NOT a relay).
function ringWireRPC(w: Worker, payload: Record<string, unknown>): Promise<{ ok: boolean; reachable?: boolean }> {
  const reqId = tokenB64url(12);
  return new Promise((resolve) => {
    const to = setTimeout(() => { if (pendingRingWire.delete(reqId)) resolve({ ok: false }); }, 8000);
    pendingRingWire.set(reqId, { workerId: w.id, resolve: (r) => { clearTimeout(to); resolve(r); } });
    try { w.ws.send(JSON.stringify({ ...payload, t: 'ring_wire', reqId })); }
    catch { clearTimeout(to); pendingRingWire.delete(reqId); resolve({ ok: false }); }
  });
}

// Wire (or re-wire) the peer ring for a ready shard. Called after a load finalizes (and after an edge_fault re-wire).
// Every consecutive stage pair gets an edge. An edge is `direct` (worker->worker) when both ends advertise a peer
// endpoint, the successor is reachable from the predecessor (probe), and the predecessor isn't force-relayed; else
// it is `relay` — the predecessor BRIDGES that hop through the coordinator (mixed pipe). Each non-first stage is
// handed its predecessor's pubkey (Ed25519 origin auth), and each non-last stage its final edge mode via ring_mode.
async function wireRing(sid: string): Promise<Ring | null> {
  const plan = shardPlans.get(sid);
  if (!plan || plan.stages.length === 0) return null;
  const ws = plan.stages.map((s) => workers.get(s.worker));
  if (ws.some((w) => !w)) return null; // a stage worker is gone → leave it to shardPipe's re-placement
  const epoch = Date.now(), token = tokenB64url(18), edges: RingEdge[] = [];
  for (let i = 0; i < plan.stages.length; i++) {
    const cur = ws[i]!, suc = i + 1 < plan.stages.length ? ws[i + 1]! : null, pre = i > 0 ? ws[i - 1]! : null;
    // hand this stage its predecessor's raw pubkey so it can origin-verify an inbound activation (direct OR bridged)
    if (pre) { const predPub = pre.peer?.pub ?? pre.pubkeyB64; const predKey = b64e(await deriveEdgeKey(sid, epoch, pre.id, cur.id)); if (predPub) { try { cur.ws.send(JSON.stringify({ t: 'ring_pred', sid, epoch, pred_pub: predPub, pred_key: predKey })); } catch { /* dropped socket → shardPipe covers it */ } } }
    // a direct edge needs BOTH ends peer-capable + reachable + not force-relayed; otherwise the successor endpoint
    // is withheld and the predecessor bridges. Always send ring_wire (sets epoch/token even for the last stage).
    const canDirect = i + 1 < plan.stages.length && !!cur.peer && !!suc!.peer && !FORCE_RELAY.has(cur.id);
    const succEndpoint = canDirect ? { id: suc!.id, url: suc!.peer!.url, pub: suc!.peer!.pub, candidates: suc!.peer!.candidates } : null;
    // Per-edge seal key for cur->suc (needed for BOTH a direct hand-off and a bridged/relay hop — the coordinator
    // only relays ciphertext on a bridge, it does not USE this key to open the frame — though as the broker it can
    // derive every edge key from MASTER; the coordinator is fully trusted). `suc` gets the matching key as pred_key.
    const succKey = suc ? b64e(await deriveEdgeKey(sid, epoch, cur.id, suc.id)) : undefined;
    const ack = await ringWireRPC(cur, { sid, epoch, token, succ: succEndpoint, ...(succKey ? { succ_key: succKey } : {}) });
    if (suc) {
      const mode: RingEdge['mode'] = (canDirect && ack.ok && !!ack.reachable) ? 'direct' : 'relay';
      edges.push({ from: cur.id, to: suc.id, mode });
      try { cur.ws.send(JSON.stringify({ t: 'ring_mode', sid, epoch, relay: mode !== 'direct' })); } catch { /* dropped socket → shardPipe covers it */ }
    }
  }
  const allDirect = plan.stages.length > 1 && edges.length > 0 && edges.every((e) => e.mode === 'direct');
  const ring: Ring = { sid, epoch, edges, token, allDirect };
  rings.set(sid, ring); ringStat(sid);
  log('info', `ring ${sid}: ${edges.map((e) => `${e.from}→${e.to}[${e.mode}]`).join(' ') || '(single stage)'} · ${allDirect ? 'ALL-DIRECT peer path' : edges.some((e) => e.mode === 'direct') ? 'MIXED (some edges bridged)' : 'relay fallback'}`);
  return ring;
}

// Run ONE forward over the peer ring: inject stage 0, await the last stage's `complete`. The hidden state travels
// worker->worker on every DIRECT edge and is bridged through the coordinator on a RELAY edge (mixed pipe). Works for
// an uncached forward AND a cached/KV decode step (session/pos ride the inject + act/bridge frames, each stage runs
// its cached shard_forward). Falls back to the unchanged relay path (shardPipe) when peer transport is off, no ring
// has a direct edge (nothing to win), the ring was torn by a drop, or a stage worker has dropped (shardPipe heals it).
async function ringPipe(sid: string, input_ids: number[], returnLogits: boolean, cache?: ShardCache): Promise<{ ok: true; data: Record<string, unknown> } | { ok: false; error: string; disconnected?: boolean }> {
  const plan = shardPlans.get(sid), ring = rings.get(sid);
  if (!PEER_TRANSPORT || !plan || !ring || ring.torn || !ring.edges.some((e) => e.mode === 'direct')) return shardPipe(sid, input_ids, returnLogits, cache);
  // A stage worker dropped after wiring → hand THIS token to shardPipe, which re-places the stage (post-load
  // failover) + (cached) resets & re-prefills. Tear the ring so later tokens relay until the shard is re-loaded/re-wired.
  for (const st of plan.stages) if (!workers.has(st.worker)) { ring.torn = true; return shardPipe(sid, input_ids, returnLogits, cache); }
  const st0 = workers.get(plan.stages[0].worker)!;
  const seq = ++ringSeq, key = `${sid}|${seq}`, stats = ringStat(sid);
  const done = new Promise((resolve, reject) => pendingRing.set(key, { resolve, reject }));
  const to = setTimeout(() => { const p = pendingRing.get(key); if (p) { pendingRing.delete(key); p.reject(new Error('ring token timeout — re-issue')); } }, SHARD_TIMEOUT_MS);
  const injectFrame: Record<string, unknown> = { t: 'inject', sid, seq, epoch: ring.epoch, token: ring.token, input_ids, return_logits: returnLogits };
  if (cache) { injectFrame.cached = true; injectFrame.session = cache.session; injectFrame.pos = cache.pos; }
  try { st0.ws.send(JSON.stringify(injectFrame)); stats.injects++; }
  catch (e) { clearTimeout(to); pendingRing.delete(key); return { ok: false, error: `inject send failed on ${st0.id}: ${e instanceof Error ? e.message : e}` }; }
  try { const data = await done as Record<string, unknown>; clearTimeout(to); return { ok: true, data }; }
  catch (e) {
    clearTimeout(to);
    log('warn', `ring ${sid}: ${(e as Error).message} — relaying this token`);
    // A mid-token ring fault may have advanced SOME stages' KV but not others; reset + re-prefill the full sequence
    // so the relay retry is consistent (shard_generate/shard_chat carry cache.seq; a bare cached forward may not).
    if (cache) { await shardReset(sid, cache.session); return shardPipe(sid, cache.seq ?? input_ids, returnLogits, { session: cache.session, pos: 0, seq: cache.seq }); }
    return shardPipe(sid, input_ids, returnLogits, cache);
  }
}
// Route a shard forward through the peer ring when peer transport is on (ringPipe self-selects direct/mixed/relay),
// else straight to the relay path. One helper so shard_forward, shard_generate, and shard_chat all get peer transport.
function pipeForward(sid: string, input_ids: number[], returnLogits: boolean, cache?: ShardCache) {
  return PEER_TRANSPORT ? ringPipe(sid, input_ids, returnLogits, cache) : shardPipe(sid, input_ids, returnLogits, cache);
}

async function shardPipe(sid: string, input_ids: number[], returnLogits: boolean, cache?: ShardCache): Promise<{ ok: true; data: Record<string, unknown> } | { ok: false; error: string; disconnected?: boolean }> {
  let replaces = 0, healed = false;
  for (;;) {
    const plan = shardPlans.get(sid);
    if (!plan || plan.stages.length === 0) return { ok: false, error: `shard ${sid} not loaded/ready` };
    // 1) heal any stage whose worker already left, BEFORE firing an RPC into the void
    const goneIdx = plan.stages.findIndex((s) => !workers.has(s.worker));
    if (goneIdx >= 0) {
      if (replaces++ >= SHARD_STAGE_TRIES) return { ok: false, error: `stage worker ${plan.stages[goneIdx].worker} disconnected; re-placement exhausted after ${replaces} tries`, disconnected: true };
      const rr = await replaceStage(sid, goneIdx);
      if (!rr.ok) return { ok: false, error: rr.error!, disconnected: true };
      healed = true; continue; // plan changed → re-run from stage 0
    }
    // after a re-placement in CACHED mode the new stage has no KV → reset the session and re-prefill the FULL seq
    // (a stateless pipe re-runs input_ids anyway; nothing to reset).
    let ids = input_ids, epos = cache?.pos ?? 0;
    if (healed && cache) { await shardReset(sid, cache.session); ids = cache.seq ?? input_ids; epos = 0; }
    healed = false;
    // 2) run the pipe stage → stage
    let carry: Record<string, unknown> = {}; let failedIdx = -1, failErr = '';
    for (let i = 0; i < plan.stages.length; i++) {
      const st = plan.stages[i]; const w = workers.get(st.worker);
      if (!w) { failedIdx = i; failErr = `stage worker ${st.worker} disconnected`; break; }
      const payload: Record<string, unknown> = { id: sid, first: st.first, last: st.last };
      if (st.first) payload.input_ids = ids; else { payload.hidden = carry.hidden; payload.seq = carry.seq; }
      if (cache) { payload.cached = true; payload.session = cache.session; payload.pos = epos; }
      if (st.last && returnLogits) payload.return_logits = true;
      ringStat(sid).shardForwards++; // PEER-TRANSPORT proof counter: one per-token activation RELAYED through the coordinator
      const r = await modelRPC(w, 'shard_forward', payload);
      if (!r.ok) {
        // RPC failed: if the worker is GONE now (post-load churn) re-place + restart; a live worker's error is real.
        if (!workers.has(st.worker)) { failedIdx = i; failErr = `shard_forward on ${st.worker} (stage ${i}) failed after disconnect: ${r.error}`; break; }
        return { ok: false, error: `shard_forward failed on ${st.worker} (stage ${i}, layers ${st.start}-${st.end}): ${r.error}` };
      }
      carry = r.data!;
    }
    if (failedIdx < 0) return { ok: true, data: carry };
    if (replaces++ >= SHARD_STAGE_TRIES) return { ok: false, error: `${failErr}; re-placement exhausted after ${replaces} tries`, disconnected: true };
    log('warn', `shard ${sid}: ${failErr} — re-placing stage ${failedIdx} onto a spare and resuming generation`);
    const rr = await replaceStage(sid, failedIdx);
    if (!rr.ok) return { ok: false, error: rr.error!, disconnected: true };
    healed = true; // loop → cached: reset+re-prefill; stateless: re-run input_ids
  }
}

// Evict a shard's live KV on every stage (one session, or all sessions when `session` is omitted). Best-effort:
// a disconnected/racing stage just skips. Called when a cached chat/generate finishes and on explicit reset.
async function shardReset(sid: string, session?: string): Promise<void> {
  const plan = shardPlans.get(sid);
  if (!plan) return;
  for (const st of plan.stages) { const w = workers.get(st.worker); if (w) await modelRPC(w, 'shard_reset', { id: sid, ...(session != null ? { session } : {}) }).catch(() => {}); }
}

// ── EXPERT PARALLELISM (MoE): stream ONE role's tensors download-free to a worker and load it. Reuses the SAME
// push staging + the UNCHANGED streamStageSafetensors (Range-fetch each selected tensor from its own file, merge
// into one per-node model.safetensors), then calls `loadOp` (moe_backbone_load | expert_load). `names` is the
// role's tensor selection (stageBackboneTensors | stageExpertTensors) — a SUBSET of the checkpoint.
async function streamMoERole(sid: string, model: string, configText: string, stPlan: STPlan, names: string[], w: Worker, loadOp: string, loadCfg: Record<string, unknown>): Promise<{ ok: boolean; data?: Record<string, unknown>; error?: string; bytes: number }> {
  try {
    const begin = await modelRPC(w, 'push_begin', { id: sid, model, resume: false }); if (!begin.ok) throw new Error(`push_begin: ${begin.error}`);
    const budget = { left: PUSH_MAX_BYTES };
    const cb = new TextEncoder().encode(configText);
    const cr = await modelRPC(w, 'push_chunk', { id: sid, name: 'config.json', seq: 0, data: b64e(cb), last: true }); if (!cr.ok) throw new Error(`push config: ${cr.error}`);
    const bytes = await streamStageSafetensors(w, sid, model, stPlan, names, budget, 0);
    const r = await modelRPC(w, loadOp, { model, id: sid, push: true, ...loadCfg });
    return { ...r, bytes };
  } catch (e) { return { ok: false, error: e instanceof Error ? e.message : String(e), bytes: 0 }; }
}

// Drive ONE MoE forward across the hybrid plan, RELAYING the router dispatch/combine THROUGH THE COORDINATOR
// (correctness-first; the peer-mesh all-to-all is a LATER increment — roadmap §4). Per MoE layer: the backbone
// runs attention + router (moe_route) → the coordinator groups the routed experts by holder and DISPATCHES to
// each holder that owns ≥1 routed expert (expert_forward on its resident subset) → the backbone COMBINES the
// partials + residual (moe_apply). Only activations + the tiny routing table cross the wire, never the weights.
async function moePipe(sid: string, input_ids: number[], returnLogits: boolean): Promise<{ ok: true; data: Record<string, unknown> } | { ok: false; error: string; disconnected?: boolean }> {
  const plan = moePlans.get(sid);
  if (!plan) return { ok: false, error: `moe ${sid} not loaded/ready` };
  const bw = workers.get(plan.backbone);
  if (!bw) return { ok: false, error: `moe backbone worker ${plan.backbone} disconnected`, disconnected: true };
  for (const h of plan.holders) if (!workers.has(h.worker)) return { ok: false, error: `expert holder ${h.worker} disconnected`, disconnected: true };
  const emb = await modelRPC(bw, 'moe_embed', { id: sid, input_ids });
  if (!emb.ok) return { ok: false, error: `moe_embed: ${emb.error}` };
  let hidden = emb.data!.hidden as string; const seq = emb.data!.seq as number;
  for (let L = 0; L < plan.nLayer; L++) {
    const rt = await modelRPC(bw, 'moe_route', { id: sid, layer: L, hidden, seq });
    if (!rt.ok) return { ok: false, error: `moe_route L${L}: ${rt.error}` };
    const moeIn = rt.data!.moe_in as string, topkI = rt.data!.topk_i as number[], topkW = rt.data!.topk_w as string, k = rt.data!.k as number;
    const used = new Set<number>(topkI);                                   // routed experts this layer
    const partials: string[] = [];
    for (const holder of plan.holders) {
      const residentUsed = holder.experts.filter((e) => used.has(e));      // DISPATCH: only this holder's routed experts
      if (residentUsed.length === 0) continue;                             // skip a holder with nothing routed to it
      const hw = workers.get(holder.worker)!;
      moeRingStat(sid).dispatches++;                                        // proof counter: an expert activation RELAYED through the coordinator
      const ef = await modelRPC(hw, 'expert_forward', { id: sid, layer: L, seq, hidden: moeIn, topk_i: topkI, topk_w: topkW, k, experts: residentUsed });
      if (!ef.ok) return { ok: false, error: `expert_forward L${L} holder ${holder.worker}: ${ef.error}` };
      partials.push(ef.data!.partial as string);
    }
    const ap = await modelRPC(bw, 'moe_apply', { id: sid, layer: L, attn_hidden: rt.data!.attn_hidden, partials, seq });
    if (!ap.ok) return { ok: false, error: `moe_apply L${L}: ${ap.error}` };
    hidden = ap.data!.hidden as string;
  }
  const hd = await modelRPC(bw, 'moe_head', { id: sid, hidden, seq, return_logits: returnLogits });
  if (!hd.ok) return { ok: false, error: `moe_head: ${hd.error}` };
  return { ok: true, data: hd.data! };
}

// ============================================================================
// PEER TRANSPORT (MoE all-to-all): take the coordinator OFF the per-token EXPERT data path. wireMoERing hands the
// BACKBONE every expert holder's LAN endpoint (+ pubkey) and probes reachability; each holder gets the backbone's
// pubkey (origin auth). When all holders are reachable, moeRingPipe injects ONE forward at the backbone and awaits
// its `moe_complete` — the backbone runs embed → per-layer (route → DISPATCH each routed expert to its holder
// worker->worker over a sealed+signed peer channel → COMBINE) → head locally, so the coordinator relays ZERO expert
// activations (only the kickoff + the final logits cross it). Any unreachable holder / mid-forward peer fault falls
// back to the unchanged coordinator-relayed moePipe. Reuses the workers' peer listeners, per-edge session keys, pubkeys.
interface MoERing { sid: string; epoch: number; token: string; allDirect: boolean }
const moeRings = new Map<string, MoERing>();
const pendingMoeRing = new Map<string, { resolve: (r: unknown) => void; reject: (e: Error) => void }>(); // `${sid}|${seq}`
const pendingMoeWire = new Map<string, { workerId: string; resolve: (r: { ok: boolean; reachable?: boolean }) => void }>();
// per-sid MoE proof counters. `dispatches` = expert activations RELAYED through the coordinator (moePipe); it stays
// 0 on the peer path while injects/completes climb — the observable evidence the mesh carries the all-to-all.
const moeRingStats = new Map<string, { injects: number; completes: number; dispatches: number }>();
let moeRingSeq = 0;
const moeRingStat = (sid: string) => { let s = moeRingStats.get(sid); if (!s) { s = { injects: 0, completes: 0, dispatches: 0 }; moeRingStats.set(sid, s); } return s; };

// Hand the backbone all holder endpoints + await its reachability ack (a control-WS RPC, NOT a relay).
function moeWireRPC(w: Worker, payload: Record<string, unknown>): Promise<{ ok: boolean; reachable?: boolean }> {
  const reqId = tokenB64url(12);
  return new Promise((resolve) => {
    const to = setTimeout(() => { if (pendingMoeWire.delete(reqId)) resolve({ ok: false }); }, 10_000);
    pendingMoeWire.set(reqId, { workerId: w.id, resolve: (r) => { clearTimeout(to); resolve(r); } });
    try { w.ws.send(JSON.stringify({ ...payload, t: 'moe_wire', reqId })); }
    catch { clearTimeout(to); pendingMoeWire.delete(reqId); resolve({ ok: false }); }
  });
}

// Wire (or re-wire) the MoE peer mesh for a ready moe plan. Every holder must advertise a peer endpoint and be
// reachable from the backbone, else the mesh stays on the coordinator relay (allDirect=false → moeRingPipe defers).
async function wireMoERing(sid: string): Promise<MoERing | null> {
  const plan = moePlans.get(sid);
  if (!plan) return null;
  const bw = workers.get(plan.backbone);
  if (!bw || !bw.peer) { log('info', `moe-ring ${sid}: backbone ${plan.backbone} has no peer endpoint → relay path`); return null; }
  const epoch = Date.now(), token = tokenB64url(18);
  const holderEps: { id: string; url: string; pub: string; experts: number[]; candidates?: string[] }[] = [];
  // Per-PAIR edge key for each backbone<->holder pair: the backbone seals its expert dispatch AND opens the
  // returned partial with it; the holder does the mirror. Neither uses the master or a per-worker key.
  const holderKeys: Record<string, string> = {};
  for (const h of plan.holders) {
    const hw = workers.get(h.worker);
    if (!hw || !hw.peer) { log('info', `moe-ring ${sid}: holder ${h.worker} has no peer endpoint → relay path`); return null; }
    holderEps.push({ id: hw.id, url: hw.peer.url, pub: hw.peer.pub, experts: h.experts, candidates: hw.peer.candidates });
    holderKeys[hw.id] = b64e(await deriveEdgeKey(sid, epoch, bw.id, hw.id));
  }
  // give each holder the backbone's pubkey (origin-verify a dispatch frame) + its per-edge key (open the dispatch, seal the partial)
  for (const h of plan.holders) { const hw = workers.get(h.worker); try { hw!.ws.send(JSON.stringify({ t: 'moe_wire_holder', sid, epoch, token, backbone_pub: bw.peer.pub, edge_key: holderKeys[hw!.id] })); } catch { /* dropped → relay covers it */ } }
  const ack = await moeWireRPC(bw, { sid, epoch, token, holders: holderEps, holder_keys: holderKeys });   // backbone probes every holder, acks reachability
  const allDirect = ack.ok && !!ack.reachable;
  const mring: MoERing = { sid, epoch, token, allDirect };
  moeRings.set(sid, mring); moeRingStat(sid);
  log('info', `moe-ring ${sid}: backbone=${bw.id} holders=${holderEps.map((h) => h.id).join(',')} · ${allDirect ? 'DIRECT peer all-to-all' : 'relay fallback'}`);
  return mring;
}

// Run ONE MoE forward over the peer mesh: inject the backbone, await `moe_complete`. Falls back to the unchanged
// coordinator-relayed moePipe when peer transport is off, the mesh isn't all-direct, or a worker has dropped.
async function moeRingPipe(sid: string, input_ids: number[], returnLogits: boolean): Promise<{ ok: true; data: Record<string, unknown> } | { ok: false; error: string; disconnected?: boolean }> {
  const plan = moePlans.get(sid), mring = moeRings.get(sid);
  if (!PEER_TRANSPORT || !plan || !mring || !mring.allDirect) return moePipe(sid, input_ids, returnLogits);
  if (!workers.has(plan.backbone)) return moePipe(sid, input_ids, returnLogits);   // backbone gone → moePipe surfaces the disconnect
  for (const h of plan.holders) if (!workers.has(h.worker)) { mring.allDirect = false; return moePipe(sid, input_ids, returnLogits); }
  const bw = workers.get(plan.backbone)!;
  const seq = ++moeRingSeq, key = `${sid}|${seq}`, stats = moeRingStat(sid);
  const done = new Promise((resolve, reject) => pendingMoeRing.set(key, { resolve, reject }));
  const to = setTimeout(() => { const p = pendingMoeRing.get(key); if (p) { pendingMoeRing.delete(key); p.reject(new Error('moe ring token timeout — re-issue')); } }, SHARD_TIMEOUT_MS);
  try { bw.ws.send(JSON.stringify({ t: 'moe_inject', sid, seq, epoch: mring.epoch, token: mring.token, input_ids, return_logits: returnLogits })); stats.injects++; }
  catch (e) { clearTimeout(to); pendingMoeRing.delete(key); return { ok: false, error: `moe inject send failed on ${bw.id}: ${e instanceof Error ? e.message : e}` }; }
  try { const data = await done as Record<string, unknown>; clearTimeout(to); return { ok: true, data }; }
  catch (e) {
    clearTimeout(to); mring.allDirect = false;
    log('warn', `moe-ring ${sid}: ${(e as Error).message} — falling back to the relayed moePipe for this token`);
    return moePipe(sid, input_ids, returnLogits);
  }
}

// --- DiLoCo: the coordinator is the parameter server. It holds the GLOBAL adapter + an outer Nesterov
// momentum buffer; each round it averages the workers' post-inner-step adapters, applies the outer step,
// and broadcasts the new global. This is the genuine reduce path (matmul pooling only concatenates). ---
interface Diloco { workers: string[]; model: string; global: Map<string, Float32Array>; momentum: Map<string, Float32Array>; shape: Map<string, number[]>; round: number; busy: boolean; }
let diloco: Diloco | null = null;
function tensorsFromReply(data: Record<string, unknown>): Map<string, Float32Array> {
  const m = new Map<string, Float32Array>();
  const t = (data.tensors ?? {}) as Record<string, { data: string }>;
  for (const [n, v] of Object.entries(t)) m.set(n, b64ToF32(v.data));
  return m;
}
const jobs = new Map<string, JobRec>();
const jobInputs = new Map<string, JobInput>();
const jobSigned = new Map<string, { signed: number; total: number }>(); // Ed25519 verification tally per job
const queue: JobRec[] = [];
let jobSeq = 0, shardSeq = 0, inflight = 0;

function submit(kernel: string, size: number, input?: JobInput): JobRec {
  const id = `job-${++jobSeq}`;
  const rec: JobRec = { id, status: 'queued', kernel, size, sealed: true, submittedAt: Date.now(), dataMode: !!input?.a };
  jobs.set(id, rec); queue.push(rec); M.jobsTotal++; KM[kernel] = (KM[kernel] ?? 0) + 1;
  // cap retained job records (and their data-mode output blobs) — drop the oldest terminal jobs first
  if (jobs.size > MAX_JOBS) for (const [jid, j] of jobs) { if (jobs.size <= MAX_JOBS) break; if (j.status === 'done' || j.status === 'failed') jobs.delete(jid); }
  if (input?.a) jobInputs.set(id, input);
  log('info', `queued ${id}: ${kernel} ${input?.a ? 'data' : 'size=' + size}`, `queue depth ${queue.length}`);
  pumpQueue();
  return rec;
}

// Run up to MAX_CONCURRENT_JOBS jobs at once; each finished job re-pumps so the queue keeps flowing and
// one slow/large job never blocks the rest. Fire-and-forget: callers poll rec.status (never await this).
function pumpQueue() {
  while (inflight < MAX_CONCURRENT_JOBS && queue.length && activeFleet().length > 0) {
    const rec = queue.shift()!;
    rec.status = 'running';
    inflight++;
    runJob(rec)
      .then(() => { rec.status = 'done'; M.jobsDone++; })
      .catch((e) => { rec.status = 'failed'; rec.error = String(e); M.jobsFailed++; jobInputs.delete(rec.id); jobSigned.delete(rec.id); log('error', `${rec.id} failed: ${rec.error}`); })
      .finally(() => { inflight--; pumpQueue(); });
  }
}

/** Auto-pause a worker that keeps HARD-failing so the fleet stops handing it work (admin/auto resume).
 *  Never pauses the last active worker — that would strand the queue with nothing to run. */
function maybeAutoPause(w: Worker) {
  if (w.consecErrors >= AUTO_PAUSE_ERRORS && !w.paused && activeFleet().length > 1) {
    w.paused = true; w.pausedReason = 'errors'; w.healthyBeats = 0;
    log('warn', `auto-paused ${w.id} after ${w.consecErrors} consecutive hard failures`);
  }
}

/** Dispatch one shard, retrying on OTHER active workers if it fails/times out (so a dead worker doesn't
 *  fail the whole job). Prefers `preferred`, then the HEALTHIEST untried active worker (fewest recent errors). */
async function dispatchResilient(jobId: string, payload: Record<string, unknown>, preferred: Worker): Promise<{ out: Float32Array; worker: Worker }> {
  const tried = new Set<string>();
  let lastErr: unknown;
  for (let attempt = 0; attempt < MAX_SHARD_ATTEMPTS; attempt++) {
    const pool = activeFleet().filter((w) => !tried.has(w.id)).sort((a, b) => a.consecErrors - b.consecErrors);
    const candidate = (attempt === 0 && !preferred.paused && !tried.has(preferred.id)) ? preferred : pool[0];
    if (!candidate) break;
    tried.add(candidate.id);
    try {
      const out = await dispatchShard(candidate, jobId, payload);
      candidate.consecErrors = 0;
      return { out, worker: candidate };
    } catch (e) {
      lastErr = e; candidate.errors++;
      // Only a HARD compute failure counts toward auto-pause; a timeout/disconnect is transient backpressure,
      // not a reason to attrit a healthy-but-backlogged worker (a burst of concurrent shards must not pause it).
      if (String(e).includes('failed:')) { candidate.consecErrors++; maybeAutoPause(candidate); }
      log('warn', `shard reassign: ${candidate.id} failed (${String(e)}); attempt ${attempt + 1}/${MAX_SHARD_ATTEMPTS}`);
    }
  }
  throw lastErr ?? new Error('no active worker could complete the shard');
}

async function dispatchShard(w: Worker, jobId: string, payload: Record<string, unknown>): Promise<Float32Array> {
  const shardId = `s-${++shardSeq}-${tokenB64url(6)}`; // unguessable so results can't be forged by id
  const sealedIn = await seal(w.key, new TextEncoder().encode(JSON.stringify(payload))); // this worker's per-worker key
  const done = new Promise<ResultMsg>((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(shardId); reject(new Error(`shard timeout (${SHARD_TIMEOUT_MS}ms) on ${w.id}`)); }, SHARD_TIMEOUT_MS);
    pending.set(shardId, { resolve: (r) => { clearTimeout(timer); resolve(r); }, reject: (e) => { clearTimeout(timer); reject(e); }, workerId: w.id });
  });
  w.busyCount++; w.busy = true; // a worker may run several concurrent shards under MAX_CONCURRENT_JOBS
  w.ws.send(JSON.stringify({ t: 'assign', shardId, jobId, sealedIn }));
  let r: ResultMsg;
  try { r = await done; } finally { pending.delete(shardId); w.busyCount = Math.max(0, w.busyCount - 1); w.busy = w.busyCount > 0; }
  if (!r.ok || !r.sealedOut) { M.shardsFailed++; throw new Error(`shard on ${w.id} failed: ${r.error}`); }
  M.shardsDone++; (r.backend?.startsWith('gpu') ? M.gpuShards++ : M.cpuShards++);
  const outObj = JSON.parse(new TextDecoder().decode(await unseal(w.key, r.sealedOut)));
  const out = b64ToF32(outObj.out);
  w.shards++; w.ops++; w.units += out.length; w.totalMs += r.ms ?? 0; // contribution accounting (one shard = one op)
  const js = jobSigned.get(jobId) ?? { signed: 0, total: 0 }; js.total++; if (r.signed) js.signed++; jobSigned.set(jobId, js);
  return out;
}

async function runJob(rec: JobRec) {
  const fleet = activeFleet();
  if (fleet.length === 0) throw new Error('no active workers (all paused/scheduled-off)');
  const input = jobInputs.get(rec.id); jobInputs.delete(rec.id);
  const data = !!input?.a; // data mode: compute on the caller's real tensors and return the output
  const t0 = performance.now();
  // Resident matmul: A · (a weight that lives on its home worker). Runs whole on that worker — the weight
  // is never re-sent. This is the building block for splitting a model across workers (pipeline parallel).
  if (rec.kernel === 'matmul' && input?.bRef) {
    const home = weightHome.get(input.bRef);
    if (!home) throw new Error(`weight ${input.bRef} is not resident — upload it via POST /weights`);
    const w = workers.get(home.worker);
    if (!w || w.paused) throw new Error(`home worker for ${input.bRef} unavailable`);
    const A = input.a!, K = home.rows, N = home.cols;
    const M = input.M ?? Math.round(A.length / K);
    if (A.length !== M * K) throw new Error(`resident matmul shape mismatch (A=${A.length}, expected ${M}x${K})`);
    const out = await dispatchShard(w, rec.id, { kernel: 'matmul', a: f32ToB64(A), bRef: input.bRef, rows: M, N, K });
    const wall = performance.now() - t0;
    const B = weightStore.get(input.bRef);
    const tol = home.dtype === 'f16' ? Math.max(3e-2, K * 3e-3) : Math.max(1e-2, K * 1e-4); // f16 keeps ~3 sig digits
    if (B && M * N <= 640 * 640) { const ref = cpuMatmul(A, B, M, N, K); let md = 0; for (let i = 0; i < out.length; i++) md = Math.max(md, Math.abs(out[i] - ref[i])); rec.verified = md < tol; }
    rec.gflops = (2 * M * N * K) / (wall / 1000) / 1e9; rec.ms = wall;
    rec.shards = [{ worker: w.id, backend: w.label, work: out.length, ms: 0 }];
    rec.output = f32ToB64(out); rec.outLen = out.length;
    const js = jobSigned.get(rec.id); rec.signed = !!js && js.total > 0 && js.signed === js.total; jobSigned.delete(rec.id);
    log('info', `${rec.id} done: resident matmul ${input.bRef}@${w.id} · ${wall.toFixed(0)}ms · verified=${rec.verified}`);
    return;
  }
  if (rec.kernel === 'matmul') {
    // dims: data mode uses input.M/N/K (default square from array); else a random square of rec.size
    const M = data ? (input!.M ?? Math.round(Math.sqrt(input!.a!.length))) : rec.size;
    const K = data ? (input!.K ?? M) : rec.size;
    const N = data ? (input!.N ?? K) : rec.size;
    const A = data ? input!.a! : new Float32Array(M * K).map(() => Math.random());
    const B = data ? input!.b! : new Float32Array(K * N).map(() => Math.random());
    if (A.length !== M * K || B.length !== K * N) throw new Error(`matmul input shape mismatch (A=${A.length} expected ${M * K}, B=${B.length} expected ${K * N})`);
    const rowsPer = Math.ceil(M / fleet.length);
    const parts = await Promise.all(fleet.map(async (w, i) => {
      const r0 = i * rowsPer, rows = Math.min(M, r0 + rowsPer) - r0; if (rows <= 0) return null;
      const { out, worker } = await dispatchResilient(rec.id, { kernel: 'matmul', a: f32ToB64(A.slice(r0 * K, (r0 + rows) * K)), b: f32ToB64(B), rows, N, K }, w);
      return { r0, out, w: worker };
    }));
    const Cm = new Float32Array(M * N); const sh: JobRec['shards'] = [];
    for (const p of parts) if (p) { Cm.set(p.out, p.r0 * N); sh.push({ worker: p.w.id, backend: p.w.label, work: p.out.length, ms: 0 }); }
    const wall = performance.now() - t0;
    if (M * N <= 640 * 640) { let md = 0; const ref = cpuMatmul(A, B, M, N, K); for (let i = 0; i < Cm.length; i++) md = Math.max(md, Math.abs(Cm[i] - ref[i])); rec.verified = md < Math.max(1e-2, K * 1e-4); }
    rec.gflops = (2 * M * N * K) / (wall / 1000) / 1e9; rec.ms = wall; rec.shards = sh;
    if (data) { rec.output = f32ToB64(Cm); rec.outLen = Cm.length; }
  } else if (ELEMENTWISE.has(rec.kernel)) {
    const binary = rec.kernel !== 'relu' && rec.kernel !== 'scale' && rec.kernel !== 'gelu';
    const a = data ? input!.a! : new Float32Array(rec.size).map(() => Math.random() * 2 - 1);
    const b = data ? (input!.b ?? new Float32Array(a.length)) : new Float32Array(a.length).map(() => Math.random());
    const scalar = data ? (input!.scalar ?? 1) : 2;
    const n = a.length;
    if (binary && b.length !== n) throw new Error(`elementwise input length mismatch (a=${n}, b=${b.length})`);
    const per = Math.ceil(n / fleet.length);
    const parts = await Promise.all(fleet.map(async (w, i) => {
      const s0 = i * per, len = Math.min(n, s0 + per) - s0; if (len <= 0) return null;
      const { out, worker } = await dispatchResilient(rec.id, { kernel: rec.kernel, a: f32ToB64(a.subarray(s0, s0 + len)), b: binary ? f32ToB64(b.subarray(s0, s0 + len)) : undefined, scalar, len }, w);
      return { s0, out, w: worker };
    }));
    const O = new Float32Array(n); const sh: JobRec['shards'] = [];
    for (const p of parts) if (p) { O.set(p.out, p.s0); sh.push({ worker: p.w.id, backend: p.w.label, work: p.out.length, ms: 0 }); }
    const ref = cpuKernel(rec.kernel, a, b, scalar); let md = 0; for (let i = 0; i < n; i++) md = Math.max(md, Math.abs(O[i] - ref[i]));
    rec.verified = md < 1e-3; rec.ms = performance.now() - t0; rec.shards = sh;
    if (data) { rec.output = f32ToB64(O); rec.outLen = O.length; }
  } else if (ROWWISE.has(rec.kernel)) {
    // per-row op (softmax/layernorm): N = columns per row; shard whole rows across the fleet.
    const cols = Math.max(1, data ? (input!.N ?? 128) : 128);
    const a = data ? input!.a! : new Float32Array(rec.size * cols).map(() => Math.random() * 4 - 2);
    const rows = Math.floor(a.length / cols);
    if (rows < 1) throw new Error(`${rec.kernel}: need at least one full row of ${cols} columns`);
    const rowsPer = Math.ceil(rows / fleet.length);
    const parts = await Promise.all(fleet.map(async (w, i) => {
      const r0 = i * rowsPer, rn = Math.min(rows, r0 + rowsPer) - r0; if (rn <= 0) return null;
      const { out, worker } = await dispatchResilient(rec.id, { kernel: rec.kernel, a: f32ToB64(a.slice(r0 * cols, (r0 + rn) * cols)), cols }, w);
      return { r0, out, w: worker };
    }));
    const O = new Float32Array(rows * cols); const sh: JobRec['shards'] = [];
    for (const p of parts) if (p) { O.set(p.out, p.r0 * cols); sh.push({ worker: p.w.id, backend: p.w.label, work: p.out.length, ms: 0 }); }
    const ref = cpuRowwise(rec.kernel, a, cols); let md = 0; for (let i = 0; i < O.length; i++) md = Math.max(md, Math.abs(O[i] - ref[i]));
    rec.verified = md < 1e-4; rec.ms = performance.now() - t0; rec.shards = sh;
    if (data) { rec.output = f32ToB64(O); rec.outLen = O.length; }
  } else { throw new Error(`unknown kernel ${rec.kernel}`); }
  const js = jobSigned.get(rec.id); rec.signed = !!js && js.total > 0 && js.signed === js.total; jobSigned.delete(rec.id);
  log('info', `${rec.id} done: ${rec.kernel} · ${(rec.ms ?? 0).toFixed(0)}ms · verified=${rec.verified} · signed=${rec.signed}${data ? ' · data' : ''}`);
}

// ---------- device descriptor (the pool presented as a real GPU slot) ----------
const LIMITS = { maxMatmulDim: 2048, maxElements: 8_000_000, maxInputElements: 16_000_000 };
function deviceDescriptor() {
  const g = virtualGpu();
  return {
    name: 'MoreGPU-Pool',
    kind: 'virtual-gpu',
    backends: [...new Set([...workers.values()].map((w) => w.backend))],
    vendors: [...new Set([...workers.values()].map((w) => w.label.split(':')[1]?.split('/')[0] ?? w.backend))],
    slots: g.slots, gpuSlots: g.gpuSlots, cpuSlots: g.cpuSlots, busy: g.busy,
    kernels: KERNELS,
    limits: LIMITS,
    queue: { depth: g.queueDepth, running: g.busy },
    throughput: { totalOps: g.totalOps, totalUnits: g.totalUnits, totalShards: g.totalShards, tokensServed: g.totalTokens, trend: g.poolTrend },
    seal: 'AES-256-GCM',
    capabilities: { dataMode: true, verifiedResults: true, signedResults: true, adaptiveThrottle: true, asyncSubmit: true, sealedWire: true, tokenIsolated: true },
  };
}

// ---------- virtual GPU view ----------
function virtualGpu() {
  const fleet = [...workers.values()];
  const slots = fleet.length;
  const gpu = fleet.filter((w) => w.backend === 'gpu').length;
  const avgUtil = slots ? fleet.reduce((s, w) => s + w.util, 0) / slots : 0;
  const avgDuty = slots ? fleet.reduce((s, w) => s + w.duty, 0) / slots : 0;
  const totalUnits = fleet.reduce((s, w) => s + w.units, 0);
  const totalShards = fleet.reduce((s, w) => s + w.shards, 0);
  const totalOps = fleet.reduce((s, w) => s + w.ops, 0);
  const totalTokens = fleet.reduce((s, w) => s + w.tokens, 0);
  return { device: 'MoreGPU-Pool', slots, gpuSlots: gpu, cpuSlots: slots - gpu, busy: fleet.filter((w) => w.busy).length, avgUserUtil: +avgUtil.toFixed(3), avgPoolDuty: +avgDuty.toFixed(3), queueDepth: queue.length, totalUnits, totalShards, totalOps, totalTokens, poolTrend: poolHistory, perKernel: KM, sealed: 'AES-256-GCM' };
}

// ---------- HTTP + WS ----------
const json = (o: unknown, status = 200) => new Response(JSON.stringify(o, null, 2), { status, headers: { 'content-type': 'application/json' } });
const authOk = (req: Request) => constEq((req.headers.get('authorization') ?? '').replace(/^Bearer\s+/i, '') || (req.headers.get('x-admin-token') ?? ''), cfg.adminToken);
const KERNELS = ['matmul', 'vector_add', 'vector_mul', 'saxpy', 'relu', 'scale', 'gelu', 'softmax', 'layernorm'];

async function handler(req: Request, info?: Deno.ServeHandlerInfo): Promise<Response> {
  const reflexiveIp = (info && 'remoteAddr' in info && info.remoteAddr && 'hostname' in info.remoteAddr) ? (info.remoteAddr as Deno.NetAddr).hostname : undefined;
  const url = new URL(req.url);
  if (url.pathname === '/ws') { const { socket, response } = Deno.upgradeWebSocket(req); wireWorker(socket, reflexiveIp); return response; }
  // STUN-like reflexive-address hint: report the caller's observed public IP so an operator can decide whether to
  // port-forward the peer port and advertise it via MOREGPU_PEER_PUBLIC. Public + read-only (no secrets).
  if (url.pathname === '/whoami') return json({ ip: reflexiveIp ?? null });
  if (url.pathname === '/health') return json({ ok: true, fleet: workers.size, queue: queue.length });
  // Public leaf cert (PEM) so a joining worker's installer can fetch it and DENO_CERT-trust it AFTER checking
  // its sha256 fingerprint == the out-of-band MOREGPU_PIN (see scripts/install.sh). The cert is public; the
  // private key never leaves the admin box. 404 under MOREGPU_INSECURE=1 (plaintext ws:// — no cert exists).
  if (url.pathname === '/cert.pem') {
    return TLS_CERT
      ? new Response(TLS_CERT, { status: 200, headers: { 'content-type': 'application/x-pem-file', 'x-moregpu-cert-sha256': TLS_FP ?? '' } })
      : new Response('no TLS cert: coordinator is running MOREGPU_INSECURE=1 (plaintext ws://)\n', { status: 404, headers: { 'content-type': 'text/plain' } });
  }
  if (url.pathname === '/help') return json({ kernels: KERNELS, endpoints: ['/ (dashboard)', '/health', '/device', '/gpu', '/workers', 'POST /workers/:id/control', 'POST /submit (?async=1)', '/jobs', '/jobs/:id', '/logs', '/metrics'], workerSchedule: 'MOREGPU_SCHEDULE=always|idle-only|HH:MM-HH:MM', auth: 'admin endpoints require Authorization: Bearer <admin token>' });
  // admin-gated
  if (['/gpu', '/device', '/workers', '/jobs', '/logs', '/metrics', '/weights', '/net'].some((p) => url.pathname === p || url.pathname.startsWith('/jobs/')) || url.pathname.startsWith('/workers/') || url.pathname.startsWith('/train/') || url.pathname.startsWith('/model/') || (req.method === 'POST' && url.pathname === '/submit')) {
    if (!authOk(req)) return json({ error: 'unauthorized — send Authorization: Bearer <admin token>' }, 401);
  }
  if (url.pathname === '/gpu') return json(virtualGpu());
  // Fleet network self-test: MoreGPU's viable workloads depend entirely on round-trip latency. Sub-ms (a LAN of
  // idle desktops) makes pipeline/expert sharding of big models viable; 100s of ms (a public tunnel) means only
  // single-node serving + coarse DiLoCo make sense. Measure each torch worker's RTT (min of a few timed pings)
  // and give an HONEST, latency-aware recommendation — no guessing whether your fleet can host a big model.
  if (url.pathname === '/net') {
    const cands = torchWorkers();
    const BW_BYTES = 512 << 10; // 512 KiB probe — enough to gauge throughput, small enough not to stall a slow link
    const bwBuf = new Uint8Array(BW_BYTES); // getRandomValues caps at 64 KiB/call → fill in chunks (entropy avoids any deflate skew)
    for (let o = 0; o < BW_BYTES; o += 65536) crypto.getRandomValues(bwBuf.subarray(o, Math.min(o + 65536, BW_BYTES)));
    const blob = b64e(bwBuf);
    const BW_CAP_MS = 10_000; // one slow/dead node must never hang the self-test → cap each probe and report what we got
    const cap = <T>(p: Promise<{ ok: boolean } & T>, ms: number) => Promise.race([p, new Promise<{ ok: false }>((res) => setTimeout(() => res({ ok: false }), ms))]);
    const rows = await Promise.all(cands.map(async (w) => {
      let rtt = Infinity;
      for (let i = 0; i < 4; i++) { const t0 = performance.now(); const r = await cap(modelRPC(w, 'ping', {}), 4000); if (r.ok) rtt = Math.min(rtt, performance.now() - t0); else break; }
      let mbps: number | null = null;
      const t1 = performance.now(); const br = await cap(modelRPC(w, 'ping', { blob }), BW_CAP_MS);
      if (br.ok) { const secs = Math.max(1e-3, (performance.now() - t1 - (Number.isFinite(rtt) ? rtt : 0)) / 1000); mbps = +((BW_BYTES / 1e6) / secs).toFixed(1); }
      // if the probe didn't finish in the cap, the link is slower than 512KB/10s ≈ 0.4 Mbps → report that honestly
      const slowNote = (mbps == null && Number.isFinite(rtt)) ? `< ${((BW_BYTES * 8 / 1e6) / (BW_CAP_MS / 1000)).toFixed(1)} Mbps (probe capped at ${BW_CAP_MS / 1000}s)` : undefined;
      return { id: w.id, backend: w.label, rtt_ms: Number.isFinite(rtt) ? +rtt.toFixed(2) : null, up_mbps: mbps, up_note: slowNote };
    }));
    const rtts = rows.map((r) => r.rtt_ms).filter((x): x is number => x != null).sort((a, b) => a - b);
    const median = rtts.length ? rtts[Math.floor(rtts.length / 2)] : null;
    const bws = rows.map((r) => r.up_mbps).filter((x): x is number => x != null).sort((a, b) => a - b);
    const medBw = bws.length ? bws[Math.floor(bws.length / 2)] : null;
    // Two independent limits: LATENCY (round-trips) gates single-request sharded DECODE; BANDWIDTH gates weight
    // transfer + throughput. A fat internet pipe does NOT lower RTT — so distinguish honestly.
    const latencyBound = median == null ? [] :
      median < 2 ? ['single-request tensor / expert parallelism (2·layers syncs/token) — needs a LAN, and you have one']
      : median < 20 ? ['single-request pipeline-sharded decode (a few hops/token) — workable; per-token tensor/expert parallelism is marginal']
      : ['single-request sharded decode is latency-bound at this RTT — it will be slow per token no matter your bandwidth (round-trips can\'t be bought with Mbps)'];
    const alwaysWorks = ['single-node resident serving (whole model on one node — any RTT)',
      'DiLoCo LoRA fine-tuning (syncs rarely + small adapters — latency-tolerant, fine over the internet if bandwidth allows)',
      'download-free model loading (bandwidth-bound — "if the internet allows" applies here: fast pipe ⇒ fast load)',
      'throughput serving for many users (batching hides per-request latency — a WAN with good bandwidth still gives high aggregate tok/s)'];
    const note = median == null ? 'no torch workers to probe' :
      median < 2 ? 'LAN-class RTT — the full menu is viable here, including single-request sharded/expert-parallel hosting of big models.' :
      median < 20 ? 'fast link — single-request pipeline sharding is viable; the internet-tolerant workloads all work.' :
      'WAN-class RTT — over the internet, use the latency-TOLERANT workloads (single-node serving, DiLoCo fine-tuning, download-free loading, throughput-batched serving). Single-request sharded DECODE stays slow: that is round-trip latency, which more bandwidth cannot fix.';
    return json({ ok: true, workers: rows, median_rtt_ms: median, median_up_mbps: medBw, latency_tolerant_anywhere: alwaysWorks, latency_bound_needs_low_rtt: latencyBound, note });
  }
  if (url.pathname === '/device') return json(deviceDescriptor());
  if (url.pathname === '/workers') {
    const totalOps = [...workers.values()].reduce((s, w) => s + w.ops, 0) || 1; // share of compute OPERATIONS (consistent unit)
    // which workers currently HOLD a resident model or a pipeline stage → they read "serving" even while idle between requests
    const serveWorkers = new Set<string>([...modelHome.values(), ...[...shardPlans.values()].flatMap((p) => p.stages.map((s) => s.worker))]);
    return json([...workers.values()].map((w) => ({
      id: w.id, nick: w.nick, backend: w.backend, label: w.label, os: w.os, userUtil: w.util, poolDuty: w.duty, ceil: w.ceil, busy: w.busy,
      paused: w.paused, pausedReason: w.pausedReason ?? null, schedule: w.schedule ?? 'always', serving: serveWorkers.has(w.id),
      shards: w.shards, ops: w.ops, units: w.units, tokens: w.tokens, share: +(w.ops / totalOps).toFixed(3), errors: w.errors,
      avgMs: w.shards ? +(w.totalMs / w.shards).toFixed(1) : 0, uptimeS: Math.round((Date.now() - w.joinedAt) / 1000), trend: w.history,
      peer: w.peer ? { candidates: w.peer.candidates ?? [w.peer.url] } : null, reflexiveIp: w.reflexiveIp ?? null, // NAT diagnostics: advertised endpoints + observed public IP
    })));
  }
  // Admin control of a single worker: pause/resume, cap its duty, set its schedule, relabel, or remove it.
  //   POST /workers/:id/control  { "action": "pause"|"resume"|"remove", "ceil": 0.4, "schedule": "22:00-07:00", "nick": "lab-3" }
  if (req.method === 'POST' && url.pathname.startsWith('/workers/') && url.pathname.endsWith('/control')) {
    const id = decodeURIComponent(url.pathname.slice('/workers/'.length, -'/control'.length));
    const w = workers.get(id);
    if (!w) return json({ error: 'worker not found' }, 404);
    const body = await req.json().catch(() => ({})) as { action?: string; ceil?: number; schedule?: string; nick?: string };
    if (body.action === 'remove') { if (w.pubkeyB64) removedPubkeys.add(w.pubkeyB64); try { w.ws.close(); } catch { /* */ } workers.delete(id); keyEpoch++; /* rotate: every key derived after this differs, so a captured pre-revocation key is dead */ await saveConfig(); /* persist so the bump + ban survive a restart */ log('warn', `admin removed worker ${id}${w.pubkeyB64 ? ' (banned by key)' : ' (unsigned — could rejoin under a new id)'} · key epoch → ${keyEpoch}`); return json({ ok: true, removed: id, banned: !!w.pubkeyB64, epoch: keyEpoch }); }
    const ctl: Record<string, unknown> = { t: 'control' };
    if (body.action === 'pause') { w.paused = true; w.pausedReason = 'admin'; ctl.pause = true; }
    if (body.action === 'resume') { w.paused = false; w.pausedReason = null; w.consecErrors = 0; ctl.pause = false; }
    if (typeof body.ceil === 'number' && isFinite(body.ceil)) { const c = Math.max(0.05, Math.min(1, body.ceil)); w.duty = c; w.ceil = c; ctl.ceil = c; }
    if (typeof body.schedule === 'string') { w.schedule = body.schedule.trim().toLowerCase().slice(0, 40); ctl.schedule = w.schedule; if (w.pausedReason === 'schedule') { w.paused = false; w.pausedReason = null; } reconcileSchedules(); } // apply the window now (coordinator-owned)
    if (typeof body.nick === 'string') w.nick = body.nick.slice(0, 40);
    try { if (Object.keys(ctl).length > 1) w.ws.send(JSON.stringify(ctl)); } catch { /* */ }
    log('info', `admin control → ${id}: ${JSON.stringify(body)}`);
    if (w.paused === false) pumpQueue(); // resuming may unblock the queue
    return json({ ok: true, id, paused: w.paused, ceil: w.duty, schedule: w.schedule ?? 'always', nick: w.nick });
  }
  // Weight residency: upload a named weight (pinned resident on one worker), or list what's cached.
  if (url.pathname === '/weights') {
    if (req.method === 'POST') {
      const body = await req.json().catch(() => ({})) as { id?: string; data?: string; rows?: number; cols?: number; worker?: string; dtype?: string };
      if (!body.id || typeof body.data !== 'string' || !body.rows || !body.cols) return json({ error: 'need {id, data (base64), rows, cols, dtype?, worker?}' }, 400);
      const dtype: 'f32' | 'f16' = body.dtype === 'f16' ? 'f16' : 'f32';
      const bytes = b64d(body.data);
      // Coordinator keeps a DEQUANTIZED f32 copy so its cpuMatmul verify matches the worker's f16 result.
      const arr = dtype === 'f16' ? Float32Array.from(new Float16Array(bytes.buffer)) : new Float32Array(bytes.buffer);
      if (arr.length !== body.rows * body.cols) return json({ error: `data length ${arr.length} != rows*cols ${body.rows * body.cols}` }, 400);
      // Cap per-weight size so a huge upload gets a clean 413 instead of OOM-ing the coordinator (a sealed
      // WS frame carries the base64 payload). Big projections (e.g. an LM head) should be tiled or host-side.
      const WCAP = Number(Deno.env.get('MOREGPU_MAX_WEIGHT_ELEMENTS') ?? 16_000_000);
      if (arr.length > WCAP) return json({ error: `weight too large (${arr.length} > ${WCAP} elements) — tile it or keep it host-side` }, 413);
      const active = activeFleet();
      if (active.length === 0) return json({ error: 'no active worker to hold the weight' }, 503);
      // home = explicit worker, else the active worker holding the fewest weights (spread the model across GPUs)
      const home = body.worker ? active.find((w) => w.id === body.worker) : active.slice().sort((a, b) => residentCount(a.id) - residentCount(b.id))[0];
      if (!home) return json({ error: `worker ${body.worker} not active` }, 404);
      const sealed = await seal(home.key, new TextEncoder().encode(JSON.stringify({ data: body.data, rows: body.rows, cols: body.cols, dtype }))); // seal to the home worker's per-worker key
      const ack = new Promise<{ ok: boolean; error?: string }>((res) => { const t = setTimeout(() => { pendingCache.delete(body.id!); res({ ok: false, error: 'cache ack timeout' }); }, 30_000); pendingCache.set(body.id!, { workerId: home.id, cb: (r) => { clearTimeout(t); res(r); } }); });
      home.ws.send(JSON.stringify({ t: 'cache', id: body.id, sealed }));
      const r = await ack;
      if (!r.ok) return json({ error: `cache failed on ${home.id}: ${r.error}` }, 502);
      weightHome.set(body.id, { worker: home.id, rows: body.rows, cols: body.cols, dtype });
      weightStore.set(body.id, arr);
      log('info', `weight ${body.id} (${body.rows}x${body.cols} ${dtype}) → resident on ${home.id}`);
      return json({ ok: true, id: body.id, worker: home.id, rows: body.rows, cols: body.cols, dtype });
    }
    return json([...weightHome.entries()].map(([id, h]) => ({ id, worker: h.worker, rows: h.rows, cols: h.cols })));
  }
  // On-pool fine-tuning (single native torch worker). load → repeated step → adapter pull.
  //   POST /train/load  { model, rank?, alpha?, lr?, seed?, targets?, worker? }
  //   POST /train/step  { input_ids:[...], labels?:[...], lr? }  → { loss, step }
  //   POST /train/adapter                                        → { step, tensors:{name:{data,shape}} }
  if (req.method === 'POST' && url.pathname === '/train/load') {
    const body = await req.json().catch(() => ({})) as { model?: string; worker?: string; rank?: number; alpha?: number; lr?: number; seed?: number; targets?: string[]; force?: boolean; push?: boolean };
    if (!body.model) return json({ error: 'need {model}' }, 400);
    if (body.push && !HF_REPO_RE.test(body.model)) return json({ error: `bad model ref "${body.model}" for a download-free training push` }, 400);
    // refuse to clobber a live single-worker session unless {force:true} (would silently reset its adapter/optimizer/step)
    if (trainingHome && workers.has(trainingHome) && !body.force) return json({ error: `a training session is already live on ${trainingHome} — pass {force:true} to replace it`, worker: trainingHome }, 409);
    const cands = torchWorkers();
    if (cands.length === 0) return json({ error: 'no native (torch) worker connected — start apps/worker/worker_torch.py; the WebGPU workers cannot train (no autograd)' }, 503);
    const w = body.worker ? cands.find((x) => x.id === body.worker) : cands[0];
    if (!w) return json({ error: `torch worker ${body.worker} not active` }, 404);
    // A worker holds ONE global TRAIN slot (shared by /train and /train/diloco), so it can't host both at
    // once. TODO(review): full fix is to key training sessions by id on the worker; this guard prevents the
    // silent cross-session corruption until then.
    if (diloco && diloco.workers.includes(w.id)) return json({ error: `worker ${w.id} is part of a live DiLoCo group — it shares the single global TRAIN slot, so it can't also host a /train session; unload the DiLoCo group or pick another worker`, worker: w.id }, 409);
    // RESERVE the single TRAIN slot BEFORE the (possibly multi-second) push, so a concurrent /train/load 409s
    // at the guard above instead of both passing check-then-set and clobbering the slot. Roll back on failure.
    const prevHome = trainingHome; trainingHome = w.id;
    // DOWNLOAD-FREE: stage the base (fp32 for LoRA) WITHOUT push_end under a TRAINING-SCOPED id — never the bare
    // model name, so a failed push's cleanup (pushModelToWorker's catch → unload {id}) can't evict a same-named
    // resident SERVING model (MODELS[model]); train_load reads this staged id + drops it.
    const trainStageId = `train::${body.model}`;
    if (body.push) {
      try { await pushModelToWorker(w, body.model, trainStageId, false, false); }
      catch (e) { trainingHome = prevHome; return json({ error: `download-free training push failed: ${e instanceof Error ? e.message : e}` }, 502); }
    }
    const r = await trainRPC(w, 'load', { model: body.model, rank: body.rank ?? 8, alpha: body.alpha ?? 16, lr: body.lr ?? 1e-3, seed: body.seed ?? 0, targets: body.targets, push: body.push, id: trainStageId });
    if (!r.ok) { trainingHome = prevHome; return json({ error: r.error }, 502); }
    log('info', `train session loaded ${body.model} on ${w.id} · trainable=${r.data?.trainable_params}`);
    return json({ ok: true, worker: w.id, ...r.data });
  }
  if (req.method === 'POST' && (url.pathname === '/train/step' || url.pathname === '/train/adapter')) {
    if (!trainingHome) return json({ error: 'no training session — POST /train/load first' }, 409);
    const w = workers.get(trainingHome);
    if (!w) { trainingHome = null; return json({ error: 'training worker disconnected — reload the session' }, 503); }
    if (url.pathname === '/train/adapter') {
      const r = await trainRPC(w, 'adapter', {});
      return r.ok ? json({ ok: true, ...r.data }) : json({ error: r.error }, 502);
    }
    const body = await req.json().catch(() => ({})) as { input_ids?: number[]; labels?: number[]; lr?: number };
    if (!Array.isArray(body.input_ids) || body.input_ids.length === 0) return json({ error: 'need {input_ids:[...]}' }, 400);
    const okInts = (a?: number[]) => !a || (a.length <= 100_000 && a.every((x) => Number.isInteger(x) && x >= 0));
    // labels may also carry HF's ignore index -100 to mask a position out of the loss (input_ids may not)
    const okLabels = (a?: number[]) => !a || (a.length <= 100_000 && a.every((x) => Number.isInteger(x) && (x === -100 || x >= 0)));
    if (!okInts(body.input_ids) || !okLabels(body.labels)) return json({ error: 'input_ids must be non-negative ints ≤100000; labels may also be -100 (ignore index)' }, 400);
    const r = await trainRPC(w, 'step', { input_ids: body.input_ids, labels: body.labels, lr: body.lr });
    if (!r.ok) return json({ error: r.error }, 502);
    return json({ ok: true, ...r.data });
  }
  // Generate from the LIVE fine-tuned model (base + adapter-so-far) — "chat with what you just trained",
  // no separate load. Read-only on the worker (eval for the decode, then restores train()).
  if (req.method === 'POST' && url.pathname === '/train/generate') {
    if (!trainingHome) return json({ error: 'no training session — POST /train/load first' }, 409);
    const w = workers.get(trainingHome);
    if (!w) { trainingHome = null; return json({ error: 'training worker disconnected — reload the session' }, 503); }
    const body = await req.json().catch(() => ({})) as { input_ids?: number[]; max_new_tokens?: number };
    if (!Array.isArray(body.input_ids) || body.input_ids.length === 0) return json({ error: 'need {input_ids:[...]}' }, 400);
    if (body.input_ids.length > 100_000 || !body.input_ids.every((x) => Number.isInteger(x) && x >= 0)) return json({ error: 'input_ids must be non-negative ints ≤100000' }, 400);
    const r = await trainRPC(w, 'generate', { input_ids: body.input_ids, max_new_tokens: body.max_new_tokens });
    if (!r.ok) return json({ error: r.error }, 502);
    return json({ ok: true, ...r.data });
  }
  // DiLoCo distributed LoRA: load the same seeded adapter on N workers, then run rounds (each worker does
  // H local steps on its shard; the coordinator averages + outer-Nesterov-steps + broadcasts the global).
  if (req.method === 'POST' && url.pathname === '/train/diloco/load') {
    const body = await req.json().catch(() => ({})) as { model?: string; rank?: number; alpha?: number; lr?: number; seed?: number; targets?: string[]; workers?: string[] };
    if (!body.model) return json({ error: 'need {model}' }, 400);
    const cands = torchWorkers();
    const group = (Array.isArray(body.workers) && body.workers.length) ? cands.filter((w) => body.workers!.includes(w.id)) : cands;
    if (group.length === 0) return json({ error: 'no native (torch) workers connected for DiLoCo' }, 503);
    // Each worker holds ONE global TRAIN slot shared by /train and /train/diloco. TODO(review): full fix is
    // per-session keying on the worker. Loading a DiLoCo group re-loads (and resets) every member's TRAIN slot
    // below, so it SUPERSEDES any single-worker /train session on those workers — end that session cleanly
    // rather than refusing (the old session's next /train/step then gets a clean 409). Only a busy round blocks.
    if (diloco && diloco.busy) return json({ error: 'a DiLoCo round is in progress — reload the group after it finishes' }, 409);
    if (trainingHome && group.some((w) => w.id === trainingHome)) { log('info', `DiLoCo supersedes the single-worker /train session on ${trainingHome}`); trainingHome = null; }
    const loads = await Promise.all(group.map((w) => trainRPC(w, 'load', { model: body.model, rank: body.rank ?? 8, alpha: body.alpha ?? 16, lr: body.lr ?? 1e-3, seed: body.seed ?? 0, targets: body.targets, no_dropout: true })));
    const okWorkers = group.filter((_, i) => loads[i].ok);
    if (okWorkers.length === 0) return json({ error: `train_load failed on all workers: ${loads[0]?.error}` }, 502);
    const init = await trainRPC(okWorkers[0], 'adapter', {});
    if (!init.ok) return json({ error: `adapter pull failed: ${init.error}` }, 502);
    const global = tensorsFromReply(init.data!);
    const shape = new Map<string, number[]>();
    for (const [n, v] of Object.entries((init.data!.tensors ?? {}) as Record<string, { shape: number[] }>)) shape.set(n, v.shape);
    const momentum = new Map<string, Float32Array>();
    for (const [n, a] of global) momentum.set(n, new Float32Array(a.length));
    diloco = { workers: okWorkers.map((w) => w.id), model: body.model, global, momentum, shape, round: 0, busy: false };
    const trainable = [...global.values()].reduce((s, a) => s + a.length, 0);
    log('info', `DiLoCo group loaded ${body.model} on ${diloco.workers.length} workers · adapter params=${trainable}`);
    return json({ ok: true, workers: diloco.workers, model: body.model, adapter_params: trainable });
  }
  if (req.method === 'POST' && url.pathname === '/train/diloco/round') {
    if (!diloco) return json({ error: 'no DiLoCo group — POST /train/diloco/load first' }, 409);
    if (diloco.busy) return json({ error: 'a DiLoCo round is already in progress — rounds are serialized' }, 409);
    const body = await req.json().catch(() => ({})) as { inner_steps?: number; lr?: number; outer_lr?: number; outer_momentum?: number; batches?: Record<string, number[][]> };
    const H = Math.max(1, Math.min(Number(body.inner_steps ?? 4), 1000));
    const lr = Number(body.lr ?? 1e-3), outerLr = Number(body.outer_lr ?? 0.7), mom = Number(body.outer_momentum ?? 0.9);
    if (![lr, outerLr, mom].every((x) => Number.isFinite(x))) return json({ error: 'lr/outer_lr/outer_momentum must be finite numbers' }, 400);
    const batches = (body.batches ?? {}) as Record<string, number[][]>;
    // Every DiLoCo batch bypasses the /train/step path, so apply the same id/size validation here: each
    // window is non-negative int token ids capped at 100000 tokens, capped at 100000 windows per worker.
    // (train_inner also applies the context-window guard on the worker.)
    const okInts = (a?: number[]) => !a || (a.length <= 100_000 && a.every((x) => Number.isInteger(x) && x >= 0));
    const okBatch = (rows?: number[][]) => !rows || (Array.isArray(rows) && rows.length <= 100_000 && rows.every((r) => Array.isArray(r) && okInts(r)));
    for (const [wid, rows] of Object.entries(batches)) if (!okBatch(rows)) return json({ error: `batches[${wid}] must be arrays of non-negative int token ids (each window ≤100000 tokens, ≤100000 windows)` }, 400);
    const live = diloco.workers.map((id) => workers.get(id)).filter((w): w is Worker => !!w);
    if (live.length === 0) { diloco = null; return json({ error: 'all DiLoCo workers disconnected' }, 503); }
    diloco.busy = true;
    try {
    const results = await Promise.all(live.map((w) => trainRPC(w, 'inner', { steps: H, lr, batches: batches[w.id] ?? batches['*'] ?? [] })));
    const okWorkers = live.map((w, i) => ({ w, r: results[i] })).filter((x) => x.r.ok && x.r.data?.tensors);
    // Reject a worker whose adapter carries ANY non-finite value (a diverged/exploded local step, or a poisoned
    // node) BEFORE it enters the average — one NaN/Inf would corrupt the whole global adapter for every worker.
    const nonFinite: string[] = [];
    const good = okWorkers.filter(({ w, r }) => {
      for (const arr of tensorsFromReply(r.data!).values()) for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) { nonFinite.push(w.id); return false; }
      return true;
    });
    if (nonFinite.length) log('warn', `DiLoCo: dropped ${nonFinite.join(', ')} from the average — non-finite adapter (diverged/poisoned)`);
    if (good.length === 0) return json({ error: `inner step produced no usable adapter${nonFinite.length ? ` (all non-finite: ${nonFinite.join(', ')})` : `: ${results[0]?.error}`}` }, 502);
    // average the surviving workers' post-inner-step adapters
    const avg = new Map<string, Float32Array>();
    for (const n of diloco.global.keys()) {
      const acc = new Float32Array(diloco.global.get(n)!.length); let cnt = 0;
      for (const { r } of good) { const a = tensorsFromReply(r.data!).get(n); if (a && a.length === acc.length) { for (let i = 0; i < acc.length; i++) acc[i] += a[i]; cnt++; } }
      if (cnt > 0) for (let i = 0; i < acc.length; i++) acc[i] /= cnt;
      avg.set(n, acc);
    }
    // outer Nesterov step on the pseudo-gradient Δ = global − avg
    for (const n of diloco.global.keys()) {
      const g = diloco.global.get(n)!, v = diloco.momentum.get(n)!, a = avg.get(n)!;
      for (let i = 0; i < g.length; i++) { const d = g[i] - a[i]; v[i] = mom * v[i] + d; g[i] = g[i] - outerLr * (d + mom * v[i]); }
    }
    // broadcast the new global to every worker, and VERIFY each accepted it: a worker that fails set_adapter
    // is now out of sync with the global adapter, so drop it from the group (its next round would otherwise
    // train from a stale adapter and silently corrupt the reduce). If ALL fail, dissolve the group.
    const tensors: Record<string, { data: string; shape: number[] }> = {};
    for (const [n, a] of diloco.global) tensors[n] = { data: f32ToB64(a), shape: diloco.shape.get(n)! };
    const bcast = await Promise.all(live.map((w) => trainRPC(w, 'set_adapter', { tensors })));
    const dropped = live.filter((_, i) => !bcast[i].ok).map((w) => w.id);
    if (dropped.length) {
      const bad = new Set(dropped);
      diloco.workers = diloco.workers.filter((wid) => !bad.has(wid));
      log('warn', `DiLoCo: set_adapter broadcast failed on ${dropped.join(', ')} — dropped from the group (out of sync)`);
    }
    if (diloco.workers.length === 0) { diloco = null; return json({ error: 'all DiLoCo workers fell out of sync (set_adapter broadcast failed) — group dissolved', dropped }, 502); }
    diloco.round++;
    const worker_losses = good.map(({ w, r }) => ({ worker: w.id, losses: r.data!.losses as number[] }));
    const avg_last_loss = worker_losses.reduce((s, x) => s + (x.losses[x.losses.length - 1] ?? 0), 0) / worker_losses.length;
    log('info', `DiLoCo round ${diloco.round}: ${good.length} workers · avg last loss ${avg_last_loss.toFixed(4)}${dropped.length ? ` · dropped ${dropped.length}` : ''}`);
    return json({ ok: true, round: diloco.round, workers: good.length, avg_last_loss, worker_losses, dropped: dropped.length ? dropped : undefined, dropped_nonfinite: nonFinite.length ? nonFinite : undefined });
    } finally { if (diloco) diloco.busy = false; }
  }
  if (req.method === 'POST' && url.pathname === '/train/diloco/adapter') {
    if (!diloco) return json({ error: 'no DiLoCo group' }, 409);
    const tensors: Record<string, { data: string; shape: number[] }> = {};
    for (const [n, a] of diloco.global) tensors[n] = { data: f32ToB64(a), shape: diloco.shape.get(n)! };
    return json({ ok: true, round: diloco.round, workers: diloco.workers, tensors });
  }
  // Resident-model serving: hold a whole model on a torch worker and run the ENTIRE forward per call —
  // ONE round-trip per token instead of the ~500 the fine-grained kernel path needs. THIS is fast serving.
  //   POST /model/load     { model, id?, fp16?, worker? }
  //   POST /model/forward  { id, input_ids:[...], return_logits?, topk? }  → { argmax, logits?, top? }
  if (req.method === 'POST' && url.pathname === '/model/load') {
    const body = await req.json().catch(() => ({})) as { model?: string; id?: string; fp16?: boolean; worker?: string; push?: boolean; async?: boolean };
    if (!body.model) return json({ error: 'need {model}' }, 400);
    const cands = torchWorkers();
    if (cands.length === 0) return json({ error: 'no native (torch) worker connected — start apps/worker/worker_torch.py; resident-model serving needs one' }, 503);
    const w = body.worker ? cands.find((x) => x.id === body.worker) : cands[0];
    if (!w) return json({ error: `torch worker ${body.worker} not active` }, 404);
    const mid = body.id ?? body.model;
    // Already resident on the target worker? reuse it instantly — a re-Load must not re-stream a live model.
    if (modelHome.get(mid) === w.id) {
      const s = modelLoads.get(mid);
      log('info', `resident model ${mid} already on ${w.id} — reusing (no re-load)`);
      return json({ ok: true, worker: w.id, id: mid, status: 'ready', reused: true, ...(s?.info ?? {}) });
    }
    if (body.push) {
      // download-free: coordinator fetches the weights and streams them to the worker (no hub access on the worker)
      if (body.async) {
        // return immediately (short request → survives a public tunnel); stream in the background, poll /model/status.
        if (modelLoads.get(mid)?.status === 'loading') return json({ error: `a load for "${mid}" is already in progress`, status: 'loading' }, 409);
        modelLoads.set(mid, { status: 'loading', worker: w.id, model: body.model!, started: Date.now() });
        pushModelToWorker(w, body.model!, mid, !!body.fp16)
          .then((info) => { modelHome.set(String(info.id ?? mid), w.id); modelLoads.set(mid, { status: 'ready', worker: w.id, model: body.model!, started: modelLoads.get(mid)!.started, info }); log('info', `resident model ${mid} weight-pushed to ${w.id} · ${info.n_params} params · staging=${info.staging}`); })
          .catch((e) => { modelLoads.set(mid, { status: 'error', worker: w.id, model: body.model!, started: modelLoads.get(mid)!.started, error: e instanceof Error ? e.message : String(e) }); log('warn', `weight-push ${mid} → ${w.id} failed: ${e instanceof Error ? e.message : e}`); });
        return json({ ok: true, worker: w.id, id: mid, status: 'loading', poll: `/model/status?id=${encodeURIComponent(mid)}` });
      }
      try {
        const info = await pushModelToWorker(w, body.model, mid, !!body.fp16);
        modelHome.set(String(info.id ?? mid), w.id);
        modelLoads.set(mid, { status: 'ready', worker: w.id, model: body.model!, started: Date.now(), info });
        log('info', `resident model ${info.id ?? mid} weight-pushed to ${w.id} · ${info.n_params} params · staging=${info.staging}`);
        return json({ ok: true, worker: w.id, ...info });
      } catch (e) { return json({ error: `weight push failed: ${e instanceof Error ? e.message : e}` }, 502); }
    }
    const r = await modelRPC(w, 'load', { model: body.model, id: mid, fp16: !!body.fp16 });
    if (!r.ok) return json({ error: r.error }, 502);
    const rid = String(r.data?.id ?? mid);
    modelHome.set(rid, w.id);
    modelLoads.set(rid, { status: 'ready', worker: w.id, model: body.model!, started: Date.now(), info: r.data }); // so /model/status is complete
    log('info', `resident model ${r.data?.id} loaded on ${w.id} · ${r.data?.n_params} params`);
    return json({ ok: true, worker: w.id, ...r.data });
  }
  // Poll an async download-free load: GET /model/status?id=<model id> → { status: loading|ready|error, … }
  if (req.method === 'GET' && url.pathname === '/model/status') {
    const id = url.searchParams.get('id') ?? [...modelLoads.keys()].pop();
    if (!id) return json({ status: 'unknown', models: [...modelHome.keys()] });
    if (modelHome.has(id)) { const s = modelLoads.get(id); return json({ status: 'ready', id, worker: modelHome.get(id), ...(s?.info ?? {}) }); }
    const s = modelLoads.get(id);
    if (!s) return json({ status: 'unknown', id });
    return json({ status: s.status, id, worker: s.worker, elapsed_ms: Date.now() - s.started, ...(s.error ? { error: s.error } : {}), ...(s.info ?? {}) });
  }
  if (req.method === 'POST' && (url.pathname === '/model/forward' || url.pathname === '/model/generate')) {
    const body = await req.json().catch(() => ({})) as { id?: string; input_ids?: number[]; return_logits?: boolean; topk?: number; max_new_tokens?: number };
    if (!body.id && modelHome.size > 1) return json({ error: `${modelHome.size} models resident — specify {id} (one of ${[...modelHome.keys()].join(', ')})` }, 400);
    const mid = body.id ?? [...modelHome.keys()][0];
    if (!mid || !modelHome.has(mid)) return json({ error: 'no resident model — POST /model/load first' }, 409);
    if (!Array.isArray(body.input_ids) || body.input_ids.length === 0) return json({ error: 'need {input_ids:[...]}' }, 400);
    if (body.input_ids.length > 100_000 || !body.input_ids.every((x) => Number.isInteger(x) && x >= 0)) return json({ error: 'input_ids must be non-negative ints, ≤100000' }, 400);
    const w = workers.get(modelHome.get(mid)!);
    if (!w) { modelHome.delete(mid); return json({ error: 'model worker disconnected — reload it' }, 503); }
    if (url.pathname === '/model/generate') {
      const n = Math.max(1, Math.min(Number(body.max_new_tokens ?? 16), 1024));
      const r = await modelRPC(w, 'generate', { id: mid, input_ids: body.input_ids, max_new_tokens: n });
      return r.ok ? json({ ok: true, ...r.data }) : json({ error: r.error }, 502);
    }
    const r = await modelRPC(w, 'forward', { id: mid, input_ids: body.input_ids, return_logits: !!body.return_logits, topk: body.topk });
    return r.ok ? json({ ok: true, ...r.data }) : json({ error: r.error }, 502);
  }
  // Text in → text out (the worker tokenizes with the model's own tokenizer). Powers the /chat page.
  if (req.method === 'POST' && url.pathname === '/model/chat') {
    const body = await req.json().catch(() => ({})) as { id?: string; prompt?: string; max_new_tokens?: number; do_sample?: boolean; temperature?: number };
    const mid = body.id ?? [...modelHome.keys()][0];
    if (!mid || !modelHome.has(mid)) return json({ error: 'no resident model — POST /model/load first', models: [...modelHome.keys()] }, 409);
    if (typeof body.prompt !== 'string' || !body.prompt.trim()) return json({ error: 'need {prompt}' }, 400);
    const w = workers.get(modelHome.get(mid)!);
    if (!w) { modelHome.delete(mid); return json({ error: 'model worker disconnected — reload it' }, 503); }
    const r = await modelRPC(w, 'chat', { id: mid, prompt: body.prompt.slice(0, 8000), max_new_tokens: Math.max(1, Math.min(Number(body.max_new_tokens ?? 96), 1024)), do_sample: !!body.do_sample, temperature: body.temperature });
    return r.ok ? json({ ok: true, worker: w.id, ...r.data }) : json({ error: r.error }, 502);
  }
  if (req.method === 'POST' && url.pathname === '/model/unload') {
    const body = await req.json().catch(() => ({})) as { id?: string };
    const mid = body.id ?? [...modelHome.keys()][0];
    if (!mid || !modelHome.has(mid)) return json({ error: 'no such resident model' }, 404);
    const w = workers.get(modelHome.get(mid)!);
    modelHome.delete(mid);
    if (w) await modelRPC(w, 'unload', { id: mid });
    log('info', `resident model ${mid} unloaded`);
    return json({ ok: true, id: mid });
  }
  // Pipeline-parallel sharding (GPT-2 family only). Split the layer range into contiguous stages across
  // the torch workers, load each stage on its worker, and record the plan.
  //   POST /model/shard          { model, id?, layers?, workers? }              → { id, stages:[{worker,start,end,first,last,params_held}] }
  //   POST /model/shard_forward  { id?, input_ids:[...], return_logits?,        → { argmax, logits?, past? }  (pipes hidden state stage→stage;
  //                                session?, cached?, pos? }                            cached mode keeps per-stage KV — pos-0 prefills, pos>0 decodes one token)
  //   POST /model/shard_reset    { id?, session? }                              → evict live KV (one session or all) — weights stay loaded
  //   POST /model/shard_unload   { id? }                                        → unload every stage + drop the plan
  if (req.method === 'GET' && url.pathname === '/model/shard_status') {
    const id = url.searchParams.get('id') ?? [...shardLoads.keys()].pop();
    if (!id) return json({ status: 'unknown', shards: [...shardPlans.keys()] });
    const s = shardLoads.get(id);
    if (!s) return json({ status: shardPlans.get(id)?.stages.length ? 'ready' : 'unknown', id });
    return json({ status: s.status, id, model: s.model, stages_done: s.stagesDone, stages_total: s.stagesTotal, elapsed_ms: Date.now() - s.started, ...(s.error ? { error: s.error } : {}), ...(s.status === 'ready' && s.info ? { stages: s.info } : {}) });
  }
  // PEER TRANSPORT introspection: the ring wiring + the proof counters. `shard_forwards` is how many per-token
  // activations were RELAYED through the coordinator for this shard; on the direct peer path it stays flat while
  // `injects`/`completes` climb — the observable evidence that the coordinator is off the per-token data path.
  if (req.method === 'GET' && url.pathname === '/model/ring_stats') {
    const id = url.searchParams.get('id') ?? [...rings.keys()].pop() ?? [...shardPlans.keys()][0];
    if (!id) return json({ enabled: PEER_TRANSPORT, wired: false });
    const ring = rings.get(id), st = ringStats.get(id) ?? { injects: 0, completes: 0, shardForwards: 0, bridged: 0 };
    return json({ enabled: PEER_TRANSPORT, id, wired: !!ring, all_direct: ring?.allDirect ?? false, epoch: ring?.epoch,
      edges: ring?.edges ?? [], injects: st.injects, completes: st.completes, shard_forwards: st.shardForwards, bridged: st.bridged });
  }
  // PEER TRANSPORT (MoE) introspection: the mesh wiring + proof counters. `dispatches` is how many expert activations
  // were RELAYED through the coordinator; on the peer all-to-all it stays flat while injects/completes climb — the
  // observable evidence the backbone<->holder dispatch/combine runs worker->worker and the coordinator is off it.
  if (req.method === 'GET' && url.pathname === '/model/moe_ring_stats') {
    const id = url.searchParams.get('id') ?? [...moeRings.keys()].pop() ?? [...moePlans.keys()][0];
    if (!id) return json({ enabled: PEER_TRANSPORT, wired: false });
    const mring = moeRings.get(id), st = moeRingStats.get(id) ?? { injects: 0, completes: 0, dispatches: 0 }, plan = moePlans.get(id);
    return json({ enabled: PEER_TRANSPORT, id, wired: !!mring, all_direct: mring?.allDirect ?? false, epoch: mring?.epoch,
      backbone: plan?.backbone, holders: plan?.holders.map((h) => h.worker) ?? [], injects: st.injects, completes: st.completes, dispatches: st.dispatches });
  }
  if (req.method === 'POST' && url.pathname === '/model/shard') {
    const body = await req.json().catch(() => ({})) as { model?: string; id?: string; layers?: number; workers?: string[]; split?: number[]; push?: boolean; async?: boolean; fp16?: boolean; quant?: string };
    if (!body.model) return json({ error: 'need {model} (GPT-2 / Llama-family)' }, 400);
    if (body.push && !HF_REPO_RE.test(body.model)) return json({ error: `bad model ref "${body.model}" for download-free shard` }, 400);
    // int8 / nf4 (bitsandbytes) — CUDA-only, and NON-PUSH only: a quantized stage is built by the worker's own
    // from_pretrained, so it can't be assembled from a streamed per-stage safetensors. Reject bad combos up front.
    if (body.quant !== undefined) {
      if (body.quant !== 'int8' && body.quant !== 'nf4' && body.quant !== 'auto') return json({ error: `bad {quant} "${body.quant}" — use "int8"/"nf4" (quantize an fp16 checkpoint on load) or "auto" (an already-quantized bnb-4bit/AWQ/GPTQ checkpoint, loaded as-is with no fp16 peak)` }, 400);
      if (body.push) return json({ error: 'quant is non-push only — omit push:true (the worker self-loads + quantizes from HF; quant + download-free streaming is not supported)' }, 400);
    }
    // explicit `workers` picks the stage order (stage i = workers[i]); otherwise use the torch fleet order
    const cands = (Array.isArray(body.workers) && body.workers.length)
      ? body.workers.map((wid) => shardWorkers().find((w) => w.id === wid)).filter((w): w is Worker => !!w)
      : shardWorkers();
    if (cands.length === 0) return json({ error: 'no shard-capable worker connected — start apps/worker/worker_torch.py (or a WebGPU worker advertising the "shard" cap); pipeline sharding needs ≥1 (≥2 for a real split)' }, 503);
    const sid = body.id ?? body.model;
    if (shardPlans.has(sid)) return json({ error: `shard ${sid} already loaded — POST /model/shard_unload first`, id: sid }, 409);
    shardPlans.set(sid, { model: body.model, stages: [], fp16: !!body.fp16, quant: body.quant }); // RESERVE synchronously so a concurrent same-id shard 409s (TOCTOU)
    const fail = (msg: string, code: number) => { shardPlans.delete(sid); shardStreams.delete(sid); return json({ error: msg, id: sid }, code); };
    // DOWNLOAD-FREE: fetch config + the safetensors header ON THE COORDINATOR (no worker download), so the
    // fleet gets only its per-stage slice. Also gives the real layer count without a worker-side model_load.
    let configText: string | null = null, stPlan: STPlan | null = null, nLayer = 0;
    if (body.push) {
      try {
        configText = await hfFetchText(body.model, 'config.json', PUSH_INDEX_MAX);
        if (!configText) return fail(`${body.model}: no config.json on HF`, 502);
        const cfg = JSON.parse(configText) as Record<string, unknown>;
        nLayer = Math.floor(Number(cfg.num_hidden_layers ?? cfg.n_layer ?? 0));
        // Weights may be a single model.safetensors OR split across many files (model.safetensors.index.json's
        // weight_map). Resolve either into one tensor→{file,offsets} plan; each stage then Range-fetches only its
        // tensors from the CORRECT file. The fleet still downloads nothing — the coordinator merges the selected
        // tensors into one per-stage safetensors before streaming it (a 404 index → single-file fast path).
        const indexText = await hfFetchText(body.model, 'model.safetensors.index.json', PUSH_INDEX_MAX);
        stPlan = await hfSafetensorsPlan(body.model, indexText);
      } catch (e) { return fail(`download-free shard preflight failed: ${e instanceof Error ? e.message : e}`, 502); }
    } else {
      // Resolve the model's REAL layer count (config-only, no weights) so gpt2-medium/large/xl shard correctly.
      // Probe a RESIDENT-capable (torch) worker — a WebGPU shard worker may not implement the 'arch' RPC.
      const archW = cands.find((w) => w.caps.has('resident')) ?? cands[0];
      const arch = await modelRPC(archW, 'arch', { model: body.model });
      if (arch.ok && Number(arch.data?.n_layer) > 0) nLayer = Math.floor(Number(arch.data!.n_layer));
    }
    if (!(nLayer > 0)) nLayer = Number(body.layers) > 0 ? Math.max(1, Math.floor(Number(body.layers))) : 0;
    if (!(nLayer > 0)) return fail(`could not determine layer count for ${body.model} — pass {layers:N} to shard explicitly`, 502);
    // Stage layout. Default: split [0, nLayer) into contiguous ranges, as even as possible. But an even split
    // starves a heterogeneous fleet — a node on a 0.3 Mbps tunnel can't stream half a model in reasonable time,
    // while its GPU could hold far more. An explicit {split:[n0,n1,...]} (per-stage layer counts, one per leading
    // worker, summing to nLayer) lets the caller give bandwidth/VRAM-poor nodes fewer layers and fast-link nodes more.
    const stages: ShardStage[] = [];
    let cursor = 0;
    const validSplit = Array.isArray(body.split) && body.split.length >= 1 && body.split.length <= cands.length
      && body.split.every((n) => Number.isInteger(n) && n >= 1) && body.split.reduce((a, b) => a + b, 0) === nLayer;
    if (Array.isArray(body.split) && !validSplit) return fail(`bad {split} for ${body.model}: need ${cands.length >= 1 ? '≤' + cands.length : ''} positive integers summing to nLayer=${nLayer} (got ${JSON.stringify(body.split)})`, 400);
    if (validSplit) {
      const ns = body.split!.length;
      for (let i = 0; i < ns; i++) { const size = body.split![i]; stages.push({ worker: cands[i].id, start: cursor, end: cursor + size, first: i === 0, last: i === ns - 1 }); cursor += size; }
    } else {
      const nStages0 = Math.min(cands.length, nLayer);
      const per = Math.floor(nLayer / nStages0), extra = nLayer % nStages0;
      for (let i = 0; i < nStages0; i++) { const size = per + (i < extra ? 1 : 0); stages.push({ worker: cands[i].id, start: cursor, end: cursor + size, first: i === 0, last: i === nStages0 - 1 }); cursor += size; }
    }
    const nStages = stages.length;
    // END-STAGE + QUANT capability guards. The first stage embeds token ids (needs the tokenizer + embedding)
    // and the last applies the final norm + LM head — so both need 'shardEnds'; a WebGPU worker (shard-but-not-
    // shardEnds) can only host a MIDDLE stage. int8/nf4 quant is bitsandbytes/CUDA-only → every stage must be a
    // torch/'resident' worker. Fail closed with a clear message instead of dispatching an op the worker can't run.
    for (const st of stages) {
      const w = workers.get(st.worker);
      if ((st.first || st.last) && !w?.caps.has('shardEnds')) return fail(`stage worker ${st.worker} cannot host the ${st.first ? 'first' : 'last'} stage (needs embeddings/tokenizer/head — the 'shardEnds' cap); a WebGPU worker can only take a MIDDLE stage`, 400);
      if (body.quant && !w?.caps.has('resident')) return fail(`quant (${body.quant}) is bitsandbytes/CUDA-only but stage worker ${st.worker} is not a torch/resident worker`, 400);
    }
    // Load every stage (streaming its slice for a push shard). Returns the stage info or throws after cleaning
    // up any stages already loaded — so a half-loaded pipe never lingers. Runnable in the background for async.
    const model = body.model, push = !!body.push;
    const runLoad = async (): Promise<(ShardStage & { params_held?: number; bytes?: number })[]> => {
      const info: (ShardStage & { params_held?: number; bytes?: number })[] = [];
      const unloadAll = async () => { for (const d of info) { const dw = workers.get(d.worker); if (dw) await modelRPC(dw, 'shard_unload', { id: sid }).catch(() => {}); } };
      // Stream ONE stage's slice to a worker (resume keeps a same-worker mid-stream drop's partial staging). The body
      // now lives in the module-level streamStageToWorker, shared with the POST-LOAD failover re-placement path; this
      // closure just binds this load's model/push/config/stPlan.
      const loadStage = (st: ShardStage, w: Worker, resume: boolean) => streamStageToWorker(sid, model, push, configText, stPlan, st, w, resume);
      // Wait (bounded) for a churned-out stage worker to reconnect under the SAME id — the keepalive reconnect means
      // a transient tunnel/Colab blip no longer nukes a long load; we just re-stream that one stage.
      const waitForWorker = async (id: string): Promise<Worker | undefined> => {
        const until = Date.now() + SHARD_RECONNECT_WAIT_MS; let w = workers.get(id);
        while (!w && Date.now() < until && !shardLoads.get(sid)?.aborted) { await sleep(1500); w = workers.get(id); }
        return w;
      };
      for (const st of stages) {
        let r: { ok: boolean; data?: Record<string, unknown>; error?: string; bytes: number } | null = null;
        for (let attempt = 1; ; attempt++) {
          if (shardLoads.get(sid)?.aborted) { await unloadAll(); throw new Error('shard load aborted (deadline)'); }
          const w = workers.get(st.worker) ?? await waitForWorker(st.worker);
          if (!w) { await unloadAll(); throw new Error(`stage worker ${st.worker} never (re)connected within ${SHARD_RECONNECT_WAIT_MS}ms (layers ${st.start}-${st.end})`); }
          r = await loadStage(st, w, attempt > 1); // retry ⇒ resume: keep the partial staging, re-stream only the tail
          if (r.ok) break;
          // NB: do NOT unload here — keeping the partial staging is what lets the next attempt resume. Only the
          // terminal-failure path below drops it (a worker that never streams enough between drops).
          if (attempt >= SHARD_STAGE_TRIES) { const cw = workers.get(st.worker); if (cw) await modelRPC(cw, 'shard_unload', { id: sid }).catch(() => {}); await unloadAll(); throw new Error(`shard_load failed on ${st.worker} (layers ${st.start}-${st.end}) after ${attempt} tries: ${r.error}`); }
          log('warn', `shard ${sid} stage ${st.worker} (layers ${st.start}-${st.end}) attempt ${attempt}/${SHARD_STAGE_TRIES} failed (${r.error}) — reconnect + resume from staged bytes`);
          await sleep(1500);
        }
        info.push({ ...st, params_held: r.data?.params_held as number | undefined, bytes: r.bytes || undefined });
        const ls = shardLoads.get(sid); if (ls) ls.stagesDone = info.length; // progress for the poller
        if (shardLoads.get(sid)?.aborted) { await unloadAll(); throw new Error('shard load aborted (deadline)'); } // deadline fired → don't resurrect
      }
      if (shardLoads.get(sid)?.aborted) { await unloadAll(); throw new Error('shard load aborted (deadline)'); }
      shardStreams.set(sid, { model, push, configText, stPlan }); // keep streaming inputs so a POST-LOAD failover can re-stream one stage to a fresh worker
      shardPlans.set(sid, { model, stages, fp16: !!body.fp16, quant: body.quant }); // finalize the reservation with the real plan (unblocks shard_forward)
      log('info', `sharded ${model} (${nLayer} layers) → ${nStages} stages${push ? ' [download-free]' : ''}: ${stages.map((s) => `${s.worker}[${s.start}-${s.end})`).join(' → ')}`);
      if (PEER_TRANSPORT) { try { await wireRing(sid); } catch (e) { log('warn', `ring wiring failed for ${sid}: ${e instanceof Error ? e.message : e}`); } } // opt-in: wire the worker->worker ring (relay stays the fallback)
      return info;
    };
    if (body.async) {
      // return immediately (short request → survives a slow/tunneled link); stream stages in the background.
      const started = Date.now();
      shardLoads.set(sid, { status: 'loading', model, started, stagesDone: 0, stagesTotal: nStages });
      // Deadline: a stage stalling on a half-dead node would otherwise leave the load 'loading' forever (each
      // push_chunk only errors after RELAY_TIMEOUT_MS). On expiry flag aborted + unload finished stages. NB: this
      // can't cancel an in-flight push_chunk on the worker (no worker-side cancel yet) — it stops the bookkeeping
      // hang and blocks resurrection; runLoad checks `aborted` before finalizing.
      const DEADLINE_MS = Number(Deno.env.get('MOREGPU_SHARD_LOAD_DEADLINE_MS') ?? 7_200_000); // 2h default — a churning tunnel resumes a stage across many reconnects; the retry/reconnect caps bound a truly stuck load
      const deadline = setTimeout(async () => {
        const s = shardLoads.get(sid);
        if (s && s.status === 'loading') {
          shardLoads.set(sid, { ...s, aborted: true, status: 'error', error: `shard load exceeded ${DEADLINE_MS}ms deadline (aborted)` });
          shardPlans.delete(sid); shardStreams.delete(sid);
          for (const st of stages) { const w = workers.get(st.worker); if (w) await modelRPC(w, 'shard_unload', { id: sid }).catch(() => {}); }
          log('warn', `shard ${sid} load timed out after ${DEADLINE_MS}ms — aborted`);
        }
      }, DEADLINE_MS);
      runLoad()
        .then((info) => { clearTimeout(deadline); if (shardLoads.get(sid)?.aborted) return; shardLoads.set(sid, { status: 'ready', model, started, stagesDone: nStages, stagesTotal: nStages, info }); })
        .catch((e) => { clearTimeout(deadline); if (shardLoads.get(sid)?.aborted) return; shardPlans.delete(sid); shardStreams.delete(sid); shardLoads.set(sid, { status: 'error', model, started, stagesDone: shardLoads.get(sid)?.stagesDone ?? 0, stagesTotal: nStages, error: e instanceof Error ? e.message : String(e) }); log('warn', `shard ${sid} failed: ${e instanceof Error ? e.message : e}`); });
      return json({ ok: true, id: sid, model, layers: nLayer, mode: push ? 'download-free' : 'download', stages_total: nStages, status: 'loading', poll: `/model/shard_status?id=${encodeURIComponent(sid)}` });
    }
    try {
      const started = Date.now();
      shardLoads.set(sid, { status: 'loading', model, started, stagesDone: 0, stagesTotal: nStages }); // fresh state — never inherit a prior load's `aborted`
      const info = await runLoad();
      shardLoads.set(sid, { status: 'ready', model, started, stagesDone: nStages, stagesTotal: nStages, info });
      return json({ ok: true, id: sid, model, layers: nLayer, mode: push ? 'download-free' : 'download', stages: info });
    } catch (e) { return fail(e instanceof Error ? e.message : String(e), 502); }
  }
  if (req.method === 'POST' && url.pathname === '/model/shard_forward') {
    const body = await req.json().catch(() => ({})) as { id?: string; input_ids?: number[]; return_logits?: boolean; session?: string; cached?: boolean; pos?: number };
    const sid = body.id ?? [...shardPlans.keys()][0];
    if (!sid || !shardPlans.has(sid)) return json({ error: 'no sharded model — POST /model/shard first' }, 409);
    if (shardPlans.get(sid)!.stages.length === 0) return json({ error: `shard ${sid} is still loading`, id: sid }, 409);
    if (!Array.isArray(body.input_ids) || body.input_ids.length === 0) return json({ error: 'need {input_ids:[...]}' }, 400);
    if (body.input_ids.length > 100_000 || !body.input_ids.every((x) => Number.isInteger(x) && x >= 0)) return json({ error: 'input_ids must be non-negative ints, ≤100000' }, 400);
    // Optional cached mode: {session, cached:true, pos} runs the incremental KV path (pos-0 prefill, then a
    // decode step feeds only the ONE new token in input_ids). Lets a client (or the parity test) drive the
    // cache step-by-step; omit it for the original stateless full-sequence forward.
    const cache = (body.cached && typeof body.session === 'string') ? { session: body.session, pos: Math.max(0, Math.floor(Number(body.pos ?? 0))) } : undefined;
    // PEER TRANSPORT (opt-in): an uncached forward goes worker->worker via ringPipe (coordinator off the data path);
    // ringPipe itself falls back to shardPipe when no direct ring is wired or a cache step is requested.
    const out = await pipeForward(sid, body.input_ids, !!body.return_logits, cache);
    if (!out.ok) { if (out.disconnected) shardPlans.delete(sid); shardStreams.delete(sid); return json({ error: out.error, id: sid }, out.disconnected ? 503 : 502); }
    return json({ ok: true, id: sid, ...out.data });
  }
  // Coordinator-side pipelined GENERATE: run the whole greedy loop HERE (one client request), streaming one
  // NDJSON line per token. On the target LAN only the fast coordinator↔fleet hops run per token; over a slow
  // tunnel the client crosses it ONCE (steady bytes also stop an idle-proxy from 502-ing a long request).
  // This is the "possible, not fast" inference story: N tokens = 1 streamed request, not N slow round-trips.
  if (req.method === 'POST' && url.pathname === '/model/shard_generate') {
    const body = await req.json().catch(() => ({})) as { id?: string; input_ids?: number[]; max_new_tokens?: number };
    const sid = body.id ?? [...shardPlans.keys()][0];
    if (!sid || !shardPlans.has(sid)) return json({ error: 'no sharded model — POST /model/shard first' }, 409);
    if (shardPlans.get(sid)!.stages.length === 0) return json({ error: `shard ${sid} is still loading`, id: sid }, 409);
    if (!Array.isArray(body.input_ids) || body.input_ids.length === 0) return json({ error: 'need {input_ids:[...]}' }, 400);
    if (body.input_ids.length > 100_000 || !body.input_ids.every((x) => Number.isInteger(x) && x >= 0)) return json({ error: 'input_ids must be non-negative ints, ≤100000' }, 400);
    const n = Math.max(1, Math.min(Number(body.max_new_tokens ?? 32), 1024));
    const prompt = body.input_ids.slice();
    const seq = body.input_ids.slice();
    const session = `${sid}::${crypto.randomUUID()}`; // fresh per request → each stage's KV starts clean, no stale carry
    const enc = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        const send = (o: unknown) => { try { controller.enqueue(enc.encode(JSON.stringify(o) + '\n')); } catch { /* client gone */ } };
        const t0 = Date.now();
        try {
          for (let k = 0; k < n; k++) {
            // KV CACHE: k==0 PREFILLS the whole prompt (pos 0); every later step feeds ONLY the new token, and each
            // stage attends to its cached prefix — so decode no longer re-runs the growing sequence.
            const step = k === 0 ? seq.slice() : [seq[seq.length - 1]];
            const out = await pipeForward(sid, step, false, { session, pos: k === 0 ? 0 : seq.length - 1, seq });
            if (!out.ok) { send({ error: out.error, at: k }); break; } // in-band terminal error (headers already sent → can't 502)
            const tok = Number(out.data.argmax); seq.push(tok);
            send({ token: tok, i: k, ms: Date.now() - t0 });
          }
          send({ done: true, tokens: seq.slice(prompt.length), n: seq.length - prompt.length, ms: Date.now() - t0 });
        } catch (e) { send({ error: e instanceof Error ? e.message : String(e) }); }
        finally { await shardReset(sid, session); controller.close(); } // evict this request's live KV on every stage
      },
    });
    return new Response(stream, { headers: { 'content-type': 'application/x-ndjson', 'cache-control': 'no-cache' } });
  }
  // Text-in/text-out chat over a SHARDED model — so all nodes contribute to a chat. The FIRST stage holds the
  // tokenizer: tokenize the prompt there, pipe the ids through every stage per token (shardPipe), decode there.
  if (req.method === 'POST' && url.pathname === '/model/shard_chat') {
    const body = await req.json().catch(() => ({})) as { id?: string; prompt?: string; max_new_tokens?: number };
    const sid = body.id ?? [...shardPlans.keys()][0];
    if (!sid || !shardPlans.has(sid) || shardPlans.get(sid)!.stages.length === 0) return json({ error: 'no ready sharded model — POST /model/shard first', id: sid }, 409);
    if (typeof body.prompt !== 'string' || !body.prompt.trim()) return json({ error: 'need {prompt}' }, 400);
    const plan = shardPlans.get(sid)!;
    const first = workers.get(plan.stages[0].worker);
    if (!first) { shardPlans.delete(sid); shardStreams.delete(sid); return json({ error: 'first-stage worker disconnected — re-shard', id: sid }, 503); }
    const tk = await modelRPC(first, 'shard_tok', { id: sid, prompt: body.prompt.slice(0, 8000) });
    if (!tk.ok) return json({ error: `tokenize failed (re-shard — the tokenizer streams to the first stage): ${tk.error}` }, 502);
    const seq = (tk.data!.input_ids as number[]).slice();
    const promptLen = seq.length, eos = Number(tk.data!.eos);
    const n = Math.max(1, Math.min(Number(body.max_new_tokens ?? 64), 512));
    const session = `${sid}::${crypto.randomUUID()}`; // fresh KV session per chat request
    const t0 = Date.now();
    let piped: { ok: true; data: Record<string, unknown> } | { ok: false; error: string; disconnected?: boolean } | null = null;
    try {
      for (let k = 0; k < n; k++) {
        // KV CACHE: prefill the whole prompt on step 0, then send ONLY the new token each step — the stages
        // attend to their cached prefix, so a sharded chat no longer re-runs the growing sequence per token.
        const step = k === 0 ? seq.slice() : [seq[seq.length - 1]];
        const out = await pipeForward(sid, step, false, { session, pos: k === 0 ? 0 : seq.length - 1, seq });
        if (!out.ok) { piped = out; break; }
        const tok = Number(out.data.argmax); seq.push(tok);
        if (Number.isFinite(eos) && tok === eos) break;
      }
    } finally { await shardReset(sid, session); } // evict this request's live KV on every stage (also on error/return)
    if (piped && !piped.ok) { if (piped.disconnected) shardPlans.delete(sid); shardStreams.delete(sid); return json({ error: piped.error, id: sid }, piped.disconnected ? 503 : 502); }
    const newTokens = seq.slice(promptLen);
    const dt = await modelRPC(first, 'shard_detok', { id: sid, tokens: newTokens });
    if (!dt.ok) return json({ error: `decode failed: ${dt.error}` }, 502);
    return json({ ok: true, id: sid, text: dt.data!.text, n: newTokens.length, ms: Date.now() - t0, workers: plan.stages.map((s) => s.worker) });
  }
  // Evict a shard's live KV cache without unloading the weights: {id?, session?} — one session, or all sessions
  // when session is omitted. Frees the incremental-decode caches; the sharded model stays loaded and ready.
  if (req.method === 'POST' && url.pathname === '/model/shard_reset') {
    const body = await req.json().catch(() => ({})) as { id?: string; session?: string };
    const sid = body.id ?? [...shardPlans.keys()][0];
    if (!sid || !shardPlans.has(sid)) return json({ error: 'no such sharded model' }, 404);
    await shardReset(sid, body.session);
    return json({ ok: true, id: sid, session: body.session ?? null });
  }
  if (req.method === 'POST' && url.pathname === '/model/shard_unload') {
    const body = await req.json().catch(() => ({})) as { id?: string };
    const sid = body.id ?? [...shardPlans.keys()][0];
    if (!sid || !shardPlans.has(sid)) return json({ error: 'no such sharded model' }, 404);
    const plan = shardPlans.get(sid)!;
    shardPlans.delete(sid); shardStreams.delete(sid); rings.delete(sid); ringStats.delete(sid); // drop any peer ring too
    for (const st of plan.stages) { const w = workers.get(st.worker); if (w) await modelRPC(w, 'shard_unload', { id: sid }); }
    log('info', `sharded model ${sid} unloaded (${plan.stages.length} stages)`);
    return json({ ok: true, id: sid, stages: plan.stages.length });
  }
  // EXPERT PARALLELISM (MoE) — see docs/ROADMAP.md. FIRST verifiable increment: a routed-MoE placed as an EP×pipe
  // hybrid — the DENSE backbone on one worker, the routed experts SPLIT across holder workers (each resident for a
  // SUBSET), and one forward driven layer-by-layer with the router dispatch/combine RELAYED THROUGH THE COORDINATOR.
  //   POST /model/moe_shard    { model, id?, push? }                 → place backbone + split experts (download-free)
  //   POST /model/moe_forward  { id?, input_ids:[...], return_logits?} → { argmax, logits? }  (coordinator-relayed EP)
  //   POST /model/moe_unload   { id? }                               → free backbone + every holder
  if (req.method === 'POST' && url.pathname === '/model/moe_shard') {
    const body = await req.json().catch(() => ({})) as { model?: string; id?: string; push?: boolean; fp16?: boolean };
    if (!body.model) return json({ error: 'need {model} (a pure routed-MoE — OLMoE / Qwen3-MoE; shared-expert families like Qwen2-MoE / DeepSeek are not supported yet)' }, 400);
    const push = body.push !== false;   // download-free by default (the coordinator is the one-time weight source)
    if (push && !HF_REPO_RE.test(body.model)) return json({ error: `bad model ref "${body.model}" for download-free MoE shard` }, 400);
    const cands = torchWorkers();
    if (cands.length < 2) return json({ error: 'MoE expert parallelism needs ≥2 native (torch) workers — 1 dense backbone + ≥1 expert holder' }, 503);
    const sid = body.id ?? body.model;
    if (moePlans.has(sid)) return json({ error: `moe ${sid} already loaded — POST /model/moe_unload first`, id: sid }, 409);
    // Preflight ON THE COORDINATOR (no worker download): config → layer/expert counts + top-k; safetensors plan
    // (single OR multi-file). hfSafetensorsPlan already resolves every routed-expert tensor to its {file,offsets},
    // so an expert's 3 tensors can live in different shards and still be selected + merged download-free.
    let configText: string, stPlan: STPlan, nLayer: number, nExperts: number, topk: number;
    try {
      const ct = await hfFetchText(body.model, 'config.json', PUSH_INDEX_MAX);
      if (!ct) return json({ error: `${body.model}: no config.json on HF` }, 502);
      configText = ct;
      const cfg = JSON.parse(configText) as Record<string, unknown>;
      nLayer = Math.floor(Number(cfg.num_hidden_layers ?? 0));
      nExperts = Math.floor(Number(cfg.num_experts ?? cfg.n_routed_experts ?? cfg.num_local_experts ?? 0));
      topk = Math.floor(Number(cfg.num_experts_per_tok ?? cfg.num_experts_per_token ?? 0));
      if (!(nLayer > 0 && nExperts > 0 && topk > 0)) return json({ error: `${body.model}: not a routed-MoE config (num_hidden_layers=${nLayer}, num_experts=${nExperts}, num_experts_per_tok=${topk})` }, 400);
      const indexText = await hfFetchText(body.model, 'model.safetensors.index.json', PUSH_INDEX_MAX);
      stPlan = await hfSafetensorsPlan(body.model, indexText);
    } catch (e) { return json({ error: `MoE preflight failed: ${e instanceof Error ? e.message : e}` }, 502); }
    // PLACEMENT: dense backbone on cands[0]; routed experts split as evenly as possible across cands[1..] (holders
    // ≤ nExperts). Contiguity is irrelevant for experts (unlike pipeline layers), so this is a simple even split.
    const backboneW = cands[0], holderWs = cands.slice(1);
    const nH = Math.min(holderWs.length, nExperts);
    const holders: MoEHolder[] = []; const per = Math.floor(nExperts / nH), extra = nExperts % nH; let curE = 0;
    for (let i = 0; i < nH; i++) { const size = per + (i < extra ? 1 : 0); holders.push({ worker: holderWs[i].id, experts: Array.from({ length: size }, (_, j) => curE + j) }); curE += size; }
    moePlans.set(sid, { model: body.model, backbone: backboneW.id, holders, nLayer, nExperts, topk }); // RESERVE (TOCTOU)
    const cleanup = async () => { const bw = workers.get(backboneW.id); if (bw) await modelRPC(bw, 'moe_unload', { id: sid }).catch(() => {}); for (const h of holders) { const w = workers.get(h.worker); if (w) await modelRPC(w, 'moe_unload', { id: sid }).catch(() => {}); } };
    const fail = async (msg: string, code: number) => { moePlans.delete(sid); await cleanup(); return json({ error: msg, id: sid }, code); };
    try {
      // stream the DENSE backbone (every tensor that is NOT a routed expert — attn, router gate, embed, norm, head)
      const bbNames = stageBackboneTensors(stPlan);
      const br = await streamMoERole(sid, body.model, configText, stPlan, bbNames, backboneW, 'moe_backbone_load', { fp16: !!body.fp16 });
      if (!br.ok) return await fail(`backbone load on ${backboneW.id} failed: ${br.error}`, 502);
      // stream each holder ONLY its routed-expert subset (selected across all layers, merged from the right files)
      const holderInfo: Record<string, unknown>[] = [];
      for (const h of holders) {
        const hw = workers.get(h.worker); if (!hw) return await fail(`holder worker ${h.worker} disconnected during load`, 503);
        const enames = stageExpertTensors(stPlan, new Set(h.experts));
        const hr = await streamMoERole(sid, body.model, configText, stPlan, enames, hw, 'expert_load', { experts: h.experts, fp16: !!body.fp16 });
        if (!hr.ok) return await fail(`expert holder ${h.worker} (experts ${h.experts.join(',')}) load failed: ${hr.error}`, 502);
        holderInfo.push({ worker: h.worker, experts: h.experts, params_held: hr.data?.params_held, bytes: hr.bytes });
      }
      log('info', `moe-sharded ${body.model} (${nLayer} layers · ${nExperts} experts · top-${topk}): backbone=${backboneW.id} · holders ${holders.map((h) => `${h.worker}{${h.experts.join(',')}}`).join(' ')}`);
      if (PEER_TRANSPORT) { try { await wireMoERing(sid); } catch (e) { log('warn', `moe-ring wiring failed for ${sid}: ${e instanceof Error ? e.message : e}`); } } // opt-in: wire the backbone<->holders peer mesh (relay stays the fallback)
      return json({ ok: true, id: sid, model: body.model, layers: nLayer, n_experts: nExperts, topk, mode: push ? 'download-free' : 'download', backbone: { worker: backboneW.id, params_held: br.data?.params_held, bytes: br.bytes }, holders: holderInfo });
    } catch (e) { return await fail(e instanceof Error ? e.message : String(e), 502); }
  }
  if (req.method === 'POST' && url.pathname === '/model/moe_forward') {
    const body = await req.json().catch(() => ({})) as { id?: string; input_ids?: number[]; return_logits?: boolean };
    const sid = body.id ?? [...moePlans.keys()][0];
    if (!sid || !moePlans.has(sid)) return json({ error: 'no MoE model — POST /model/moe_shard first' }, 409);
    if (!Array.isArray(body.input_ids) || body.input_ids.length === 0) return json({ error: 'need {input_ids:[...]}' }, 400);
    if (body.input_ids.length > 100_000 || !body.input_ids.every((x) => Number.isInteger(x) && x >= 0)) return json({ error: 'input_ids must be non-negative ints, ≤100000' }, 400);
    const out = await moeRingPipe(sid, body.input_ids, !!body.return_logits);
    if (!out.ok) { if (out.disconnected) moePlans.delete(sid); return json({ error: out.error, id: sid }, out.disconnected ? 503 : 502); }
    return json({ ok: true, id: sid, ...out.data });
  }
  if (req.method === 'POST' && url.pathname === '/model/moe_unload') {
    const body = await req.json().catch(() => ({})) as { id?: string };
    const sid = body.id ?? [...moePlans.keys()][0];
    if (!sid || !moePlans.has(sid)) return json({ error: 'no such MoE model' }, 404);
    const plan = moePlans.get(sid)!; moePlans.delete(sid); moeRings.delete(sid); moeRingStats.delete(sid); // drop any peer mesh too
    const bw = workers.get(plan.backbone); if (bw) await modelRPC(bw, 'moe_unload', { id: sid }).catch(() => {});
    for (const h of plan.holders) { const w = workers.get(h.worker); if (w) await modelRPC(w, 'moe_unload', { id: sid }).catch(() => {}); }
    log('info', `moe model ${sid} unloaded (backbone ${plan.backbone} + ${plan.holders.length} holders)`);
    return json({ ok: true, id: sid, backbone: plan.backbone, holders: plan.holders.length });
  }
  if (url.pathname === '/logs') return json(LOG.slice(-200).reverse());
  if (url.pathname === '/metrics') return new Response(prometheus(), { headers: { 'content-type': 'text/plain; version=0.0.4' } });
  if (req.method === 'POST' && url.pathname === '/submit') {
    const body = await req.json().catch(() => ({})) as { kernel?: string; size?: number; a?: string; b?: string; scalar?: number; M?: number; N?: number; K?: number; bRef?: string };
    const kernel = KERNELS.includes(body.kernel ?? '') ? body.kernel! : 'matmul';
    // Data mode: caller supplies real tensors (base64 Float32) → the pool computes on THEM and returns the output.
    let input: JobInput | undefined;
    if (typeof body.a === 'string') {
      const a = b64ToF32(body.a);
      const b = typeof body.b === 'string' ? b64ToF32(body.b) : undefined;
      const CAP = 16_000_000;
      if (a.length > CAP || (b && b.length > CAP)) return json({ error: 'input too large (>16M elements per buffer)' }, 413);
      input = { a, b, scalar: body.scalar, M: body.M, N: body.N, K: body.K, bRef: body.bRef };
    }
    const size = Math.max(16, Math.min(kernel === 'matmul' ? 2048 : 8_000_000, Number(body.size ?? (kernel === 'matmul' ? 512 : 1_000_000))));
    // Async mode (GPU-style submit-and-poll): return the job handle immediately; caller polls GET /jobs/:id.
    const isAsync = (body as { async?: boolean }).async === true || url.searchParams.get('async') === '1';
    if (activeFleet().length === 0) { const r = submit(kernel, size, input); return json({ ...r, poll: `/jobs/${r.id}`, note: 'queued — no active workers yet (none connected, or all paused/scheduled-off); will run when one is available' }, 202); }
    const rec = submit(kernel, size, input);
    if (isAsync) return json({ id: rec.id, status: rec.status, kernel: rec.kernel, poll: `/jobs/${rec.id}` }, 202);
    // sync: wait briefly for a result; clients must still check status (may still be queued/running)
    for (let i = 0; i < 600 && rec.status !== 'done' && rec.status !== 'failed'; i++) await new Promise((r) => setTimeout(r, 50));
    return json(rec, rec.status === 'failed' ? 503 : 200);
  }
  if (url.pathname === '/jobs') return json([...jobs.values()].slice(-50).reverse().map(({ output: _o, ...r }) => ({ ...r, hasOutput: !!_o })));
  if (url.pathname.startsWith('/jobs/')) { const r = jobs.get(url.pathname.slice(6)); return r ? json(r) : json({ error: 'not found' }, 404); }
  if (url.pathname === '/chat') return new Response(CHAT_HTML, { headers: { 'content-type': 'text/html' } });
  return new Response(dashboard(), { headers: { 'content-type': 'text/html' } });
}

function prometheus(): string {
  const g = virtualGpu();
  const totalOps = g.totalOps || 1; // share is a fraction of compute OPERATIONS (never tokens-vs-elements)
  const lines = [
    '# HELP moregpu_fleet Connected workers', '# TYPE moregpu_fleet gauge', `moregpu_fleet ${g.slots}`,
    '# TYPE moregpu_gpu_slots gauge', `moregpu_gpu_slots ${g.gpuSlots}`,
    '# TYPE moregpu_cpu_slots gauge', `moregpu_cpu_slots ${g.cpuSlots}`,
    '# TYPE moregpu_busy gauge', `moregpu_busy ${g.busy}`,
    '# TYPE moregpu_queue_depth gauge', `moregpu_queue_depth ${g.queueDepth}`,
    '# TYPE moregpu_avg_user_util gauge', `moregpu_avg_user_util ${g.avgUserUtil}`,
    '# TYPE moregpu_avg_pool_duty gauge', `moregpu_avg_pool_duty ${g.avgPoolDuty}`,
    '# TYPE moregpu_total_units counter', `moregpu_total_units ${g.totalUnits}`,
    '# TYPE moregpu_total_shards counter', `moregpu_total_shards ${g.totalShards}`,
    '# TYPE moregpu_total_ops counter', `moregpu_total_ops ${g.totalOps}`,
    '# HELP moregpu_tokens_served LLM tokens generated across the pool', '# TYPE moregpu_tokens_served counter', `moregpu_tokens_served ${g.totalTokens}`,
    '# TYPE moregpu_jobs_total counter', `moregpu_jobs_total ${M.jobsTotal}`, `moregpu_jobs_done ${M.jobsDone}`, `moregpu_jobs_failed ${M.jobsFailed}`,
    `moregpu_shards_done ${M.shardsDone}`, `moregpu_shards_failed ${M.shardsFailed}`, `moregpu_gpu_shards ${M.gpuShards}`, `moregpu_cpu_shards ${M.cpuShards}`,
    '# HELP moregpu_worker_units Kernel output-elements completed per worker (detail)', '# TYPE moregpu_worker_units counter',
  ];
  for (const w of workers.values()) {
    const lbl = `{worker="${w.id.replace(/"/g, '')}",backend="${w.backend}"}`;
    lines.push(`moregpu_worker_units${lbl} ${w.units}`);
    lines.push(`moregpu_worker_ops${lbl} ${w.ops}`);
    lines.push(`moregpu_worker_tokens${lbl} ${w.tokens}`);
    lines.push(`moregpu_worker_share${lbl} ${(w.ops / totalOps).toFixed(4)}`);
    lines.push(`moregpu_worker_shards${lbl} ${w.shards}`);
    lines.push(`moregpu_worker_errors${lbl} ${w.errors}`);
    lines.push(`moregpu_worker_user_util${lbl} ${w.util}`);
    lines.push(`moregpu_worker_pool_duty${lbl} ${w.duty}`);
  }
  return lines.join('\n') + '\n';
}

const serveOpts: Deno.ServeTcpOptions & { cert?: string; key?: string } = { port: PORT, hostname: BIND, onListen: () => wizardBanner() };
if (TLS_CERT && TLS_KEY) { serveOpts.cert = TLS_CERT; serveOpts.key = TLS_KEY; } // resolved above: provided cert, persisted mint, or (INSECURE) none
Deno.serve(serveOpts, handler);

// Built-in worker: with --worker (or MOREGPU_SELF_WORKER=1) the admin's OWN machine joins its own pool
// as a compute slot — so a solo admin needs nothing else; submitting a job "just works" like a local GPU.
const SELF_WORKER = Deno.args.includes('--worker') || Deno.env.get('MOREGPU_SELF_WORKER') === '1';
if (SELF_WORKER) {
  try {
    const workerUrl = new URL('../worker/worker.ts', import.meta.url).href;
    // Under the default TLS the built-in worker connects over wss:// to `localhost` (a cert SAN) and trusts the
    // freshly minted cert via DENO_CERT (Deno's CA-file env) — no separate pin needed on this loopback hop.
    const wsUrl = TLS_CERT ? `wss://localhost:${PORT}/ws` : `ws://127.0.0.1:${PORT}/ws`;
    // When the coordinator itself was launched from a URL (the `moregpu serve` / raw-URL path), the built-in
    // worker resolves to the REMOTE worker.ts, which Deno serves from its module cache. A cache holding an
    // OLDER worker.ts (e.g. from before a registration-protocol change) would desync from THIS coordinator and
    // self-reject with "bad join token". Force-refresh just that one module in the remote case so the built-in
    // worker always matches the coordinator; a local file:// worker is always read fresh, so skip it there.
    const reload = workerUrl.startsWith('http') ? [`--reload=${workerUrl}`] : [];
    new Deno.Command(Deno.execPath(), {
      args: ['run', '--unstable-webgpu', ...reload, '--allow-net', '--allow-env', '--allow-sys', workerUrl,
        '--server', wsUrl, '--token', cfg.joinToken, '--name', Deno.env.get('MOREGPU_NAME') ?? 'admin-slot'],
      env: TLS_CERT_FILE ? { DENO_CERT: TLS_CERT_FILE } : {}, // merged onto the inherited env; trusts our self-signed cert
      stdout: 'inherit', stderr: 'inherit',
    }).spawn();
    log('info', 'built-in worker started — this machine is a pool slot (admin-slot). Submit a job; it just runs.');
  } catch (e) {
    log('warn', `could not start the built-in worker (the server needs --allow-run for --worker): ${e}`);
  }
}

function dashboard(): string {
  return DASHBOARD_HTML;
}
const DASHBOARD_HTML = `<!doctype html><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1"><title>MoreGPU · admin</title>
<style>
:root{
  --bg:#070a12;--bg2:#0b0f1a;--card:#0f1523;--card2:#131a2b;--line:#1e2740;--line2:#28324f;
  --ink:#e8ecf5;--mut:#8a96b0;--dim:#5c6885;
  --acc:#6366f1;--acc2:#a855f7;--pink:#ec4899;--red:#f43f5e;
  --grn:#34d399;--yel:#fbbf24;--blu:#60a5fa;
  --grad:linear-gradient(100deg,#6366f1,#a855f7 42%,#ec4899 72%,#f43f5e);
  --shadow:0 10px 40px -12px rgba(0,0,0,.6);
}
*{box-sizing:border-box}
html,body{margin:0}
body{
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  color:var(--ink);background:var(--bg);
  background-image:radial-gradient(1200px 600px at 12% -8%,rgba(99,102,241,.16),transparent 60%),radial-gradient(1000px 560px at 92% -4%,rgba(236,72,153,.13),transparent 62%);
  -webkit-font-smoothing:antialiased;min-height:100vh;
}
.wrap{max-width:1180px;margin:0 auto;padding:20px 18px 60px}
a{color:#a5b4fc;text-decoration:none}a:hover{text-decoration:underline}
.mut{color:var(--mut)}.dim{color:var(--dim)}
.mono{font-family:ui-monospace,"SF Mono",Menlo,monospace}

/* ---- top bar ---- */
.top{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:22px}
.brand{display:flex;align-items:center;gap:11px;min-width:0}
.mark{width:38px;height:38px;flex:0 0 auto;border-radius:11px;background:var(--grad);display:grid;place-items:center;box-shadow:0 6px 22px -6px rgba(124,58,237,.6)}
.mark svg{width:22px;height:22px;display:block}
.brand .name{font-size:19px;font-weight:750;letter-spacing:-.4px;line-height:1;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.brand .tag{font-size:11.5px;color:var(--mut);margin-top:3px}
.top .spacer{flex:1}
.hstat{display:flex;gap:9px;align-items:stretch;flex-wrap:wrap}
.chip{display:flex;flex-direction:column;justify-content:center;padding:7px 13px;border:1px solid var(--line);border-radius:11px;background:rgba(15,21,35,.7);min-width:76px}
.chip .n{font-size:17px;font-weight:700;letter-spacing:-.4px;line-height:1.1}
.chip .l{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-top:2px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--dim);margin-right:6px;vertical-align:middle}
.dot.on{background:var(--grn);box-shadow:0 0 0 3px rgba(52,211,153,.18)}
.dot.off{background:var(--red);box-shadow:0 0 0 3px rgba(244,63,94,.18)}

/* ---- generic ---- */
.card{background:linear-gradient(180deg,rgba(19,26,43,.55),rgba(15,21,35,.85));border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:var(--shadow)}
.card+.card,.sp{margin-top:16px}
.card h2{margin:0 0 3px;font-size:15px;font-weight:700;letter-spacing:-.2px;display:flex;align-items:center;gap:8px}
.card .hint{font-size:12px;color:var(--mut);margin:0 0 14px}
.card h2 .badge{font-size:10px;font-weight:600;color:var(--mut)}
.grid{display:grid;gap:14px}
.g-kpi{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
.g-fleet{grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}
.section-title{display:flex;align-items:center;gap:10px;margin:26px 2px 12px}
.section-title h2{margin:0;font-size:16px;font-weight:750;letter-spacing:-.3px}
.section-title .ln{flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent)}

.kpi{position:relative;overflow:hidden;padding:16px 17px}
.kpi.accent{background:linear-gradient(150deg,rgba(99,102,241,.16),rgba(168,85,247,.06) 55%,transparent),var(--card)}
.kpi .kl{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--mut);font-weight:600}
.kpi .kv{font-size:30px;font-weight:750;letter-spacing:-1.2px;margin-top:5px;line-height:1}
.kpi .ks{font-size:12px;color:var(--mut);margin-top:4px}
.bar{height:7px;border-radius:6px;background:#0b1120;border:1px solid var(--line);overflow:hidden;margin-top:9px}
.bar>i{display:block;height:100%;background:var(--grad);transition:width .5s ease}
.bar>i.grn{background:linear-gradient(90deg,#10b981,#34d399)}

button{font:inherit;background:var(--acc);color:#fff;border:0;border-radius:10px;padding:9px 15px;font-weight:600;cursor:pointer;transition:filter .15s,opacity .15s}
button:hover{filter:brightness(1.08)}button:active{filter:brightness(.94)}button:disabled{opacity:.5;cursor:not-allowed;filter:none}
button.grad{background:var(--grad)}
button.ghost{background:rgba(99,102,241,.12);color:#c7d2fe;border:1px solid var(--line2)}
button.ghost:hover{background:rgba(99,102,241,.2)}
button.mini{padding:5px 9px;font-size:12px;border-radius:8px;background:rgba(99,102,241,.15);color:#c7d2fe;border:1px solid var(--line2)}
button.mini.danger{background:rgba(244,63,94,.12);color:#fca5a5;border-color:rgba(244,63,94,.3)}
button.mini.danger:hover{background:rgba(244,63,94,.22)}
input,select,textarea{font:inherit;background:#0a0f1b;border:1px solid var(--line);color:var(--ink);border-radius:10px;padding:9px 11px;outline:none;transition:border-color .15s,box-shadow .15s}
input:focus,select:focus,textarea:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(99,102,241,.18)}
input::placeholder{color:var(--dim)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
label.f{font-size:12px;color:var(--mut);display:inline-flex;align-items:center;gap:6px}

.pill{display:inline-flex;align-items:center;gap:5px;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600;border:1px solid transparent}
.pill.gpu{background:rgba(52,211,153,.13);color:#6ee7b7;border-color:rgba(52,211,153,.25)}
.pill.cpu{background:rgba(251,191,36,.12);color:#fcd34d;border-color:rgba(251,191,36,.25)}
.state{font-size:11px;font-weight:600;display:inline-flex;align-items:center;gap:5px}
.state .sd{width:7px;height:7px;border-radius:50%;background:currentColor}
.s-work{color:var(--blu)}.s-serve{color:var(--grn)}.s-idle{color:var(--dim)}.s-pause{color:var(--yel)}.s-auto{color:var(--red)}.s-sched{color:var(--mut)}

/* ---- worker card ---- */
.wk{padding:15px 16px;display:flex;flex-direction:column;gap:11px}
.wk .wkh{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}
.wk .wkname{font-weight:700;font-size:14.5px;letter-spacing:-.2px;word-break:break-word;line-height:1.25}
.wk .wksub{font-size:11px;color:var(--dim);margin-top:2px;display:flex;gap:7px;flex-wrap:wrap;align-items:center}
.wk .spark{width:100%;height:34px;display:block}
.wk .grid3{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.wk .cellv{font-size:15px;font-weight:700;letter-spacing:-.3px}
.wk .celll{font-size:9.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin-top:1px}
.wk .util{display:flex;gap:12px;font-size:11px;color:var(--mut)}
.wk .util b{color:var(--ink);font-weight:600}
.wk .ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;border-top:1px solid var(--line);padding-top:11px}
.wk .duty{display:flex;align-items:center;gap:7px;flex:1;min-width:150px}
.duty input[type=range]{flex:1;min-width:70px;accent-color:#a855f7;cursor:pointer;height:4px}
.duty .dv{font-size:11px;color:var(--mut);width:38px;text-align:right;font-variant-numeric:tabular-nums}
.wk .schedin,.wk .nickin{font-size:11px;padding:5px 8px;border-radius:8px}
.wk .schedin{width:118px}.wk .nickin{flex:1;min-width:90px}
.err{color:var(--red);font-size:11px;font-weight:600}

/* ---- net + model ---- */
.nettbl{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}
.nettbl th,.nettbl td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
.nettbl th{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);font-weight:600}
.nettbl td.mono{font-variant-numeric:tabular-nums}
.verdict{margin-top:14px;display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}
.vbox{border:1px solid var(--line);border-radius:12px;padding:12px 14px;background:rgba(10,15,27,.5)}
.vbox h4{margin:0 0 8px;font-size:12px;display:flex;align-items:center;gap:7px}
.vbox ul{margin:0;padding-left:17px;font-size:12px;color:var(--mut);line-height:1.55}
.vbox.ok h4{color:#6ee7b7}.vbox.warn h4{color:#fcd34d}
.note{margin-top:12px;font-size:12.5px;color:var(--mut);border-left:2px solid var(--acc2);padding:2px 0 2px 12px}

.chatbox{display:flex;flex-direction:column;gap:10px;background:#0a0f1b;border:1px solid var(--line);border-radius:12px;padding:12px;min-height:120px;max-height:44vh;overflow:auto;margin-top:12px}
.msg{padding:9px 12px;border-radius:12px;max-width:86%;white-space:pre-wrap;word-wrap:break-word;font-size:13.5px;line-height:1.5}
.msg.u{align-self:flex-end;background:linear-gradient(135deg,#4338ca,#6d28d9);color:#fff;border-bottom-right-radius:4px}
.msg.b{align-self:flex-start;background:#131a2b;border:1px solid var(--line2);border-bottom-left-radius:4px}
.msg .meta{display:block;font-size:10.5px;color:var(--dim);margin-top:5px}
.mstatus{font-size:12.5px;color:var(--mut);display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.mstatus.ok{color:#6ee7b7}.mstatus.err{color:#fca5a5}

pre.logs{background:#0a0f1b;border:1px solid var(--line);border-radius:12px;padding:12px;overflow:auto;max-height:280px;font-size:12px;margin:12px 0 0;line-height:1.55}
.lg-error{color:var(--red)}.lg-warn{color:var(--yel)}.lg-info{color:#93c5fd}.lg-debug{color:var(--dim)}
.k{display:inline-block;background:rgba(99,102,241,.12);color:#c7d2fe;border:1px solid var(--line2);border-radius:8px;padding:3px 10px;margin:0 5px 5px 0;font-size:12px;font-family:ui-monospace,monospace}
.empty{color:var(--dim);font-size:13px;padding:8px 2px}

/* ---- toast + gate ---- */
.toasts{position:fixed;right:16px;bottom:16px;display:flex;flex-direction:column;gap:9px;z-index:60;max-width:min(360px,92vw)}
.toast{background:#131a2b;border:1px solid var(--line2);border-left:3px solid var(--acc);border-radius:11px;padding:11px 14px;font-size:13px;box-shadow:var(--shadow);animation:tin .25s ease}
.toast.err{border-left-color:var(--red)}.toast.ok{border-left-color:var(--grn)}
@keyframes tin{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.gate{position:fixed;inset:0;background:rgba(5,8,14,.82);backdrop-filter:blur(6px);display:none;align-items:center;justify-content:center;z-index:80;padding:20px}
.gate.show{display:flex}
.gate .box{background:var(--card);border:1px solid var(--line2);border-radius:18px;padding:26px;max-width:420px;width:100%;box-shadow:var(--shadow)}
.gate .box .mark{margin:0 auto 14px}
.gate h3{margin:0 0 6px;text-align:center;font-size:18px}
.gate p{margin:0 0 16px;text-align:center;color:var(--mut);font-size:13px}
.gate .row2{display:flex;gap:9px}
.gate input{flex:1}
.tokbar{display:flex;align-items:center;gap:8px}
.tokbar input{width:150px}
@media(max-width:640px){
  .wrap{padding:16px 13px 50px}
  .top .spacer{display:none}
  .hstat{width:100%}
  .tokbar input{width:120px}
  .g-fleet{grid-template-columns:1fr}
}
</style>

<div class="wrap">
  <div class="top">
    <div class="brand">
      <div class="mark" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none"><rect x="4" y="4" width="16" height="16" rx="3" stroke="#fff" stroke-width="1.7"/><rect x="8.5" y="8.5" width="7" height="7" rx="1.3" fill="#fff"/><path d="M9 1.5v3M12 1.5v3M15 1.5v3M9 19.5v3M12 19.5v3M15 19.5v3M1.5 9h3M1.5 12h3M1.5 15h3M19.5 9h3M19.5 12h3M19.5 15h3" stroke="#fff" stroke-width="1.5" stroke-linecap="round"/></svg>
      </div>
      <div>
        <div class="name">MoreGPU</div>
        <div class="tag">every worker, one virtual GPU</div>
      </div>
    </div>
    <div class="spacer"></div>
    <div class="hstat" id="hstat">
      <div class="chip"><span class="n" id="hFleet">—</span><span class="l"><span class="dot" id="hDot"></span>fleet</span></div>
      <div class="chip"><span class="n" id="hQueue">—</span><span class="l">queue</span></div>
      <div class="chip tokbar" style="flex-direction:row;align-items:center">
        <input id="tokTop" type="password" placeholder="admin token" autocomplete="off">
        <button class="mini" id="signout" title="clear the stored token">sign out</button>
      </div>
    </div>
  </div>

  <!-- KPI overview -->
  <div class="grid g-kpi">
    <div class="card kpi accent">
      <div class="kl">Virtual GPU</div>
      <div class="kv" id="kSlots">—</div>
      <div class="ks" id="kSlotsub">connect a worker…</div>
      <div class="ks" style="margin-top:9px">user load <span id="kUU" class="mut">–</span></div>
      <div class="bar"><i id="kUUbar" style="width:0"></i></div>
      <div class="ks" style="margin-top:7px">pool duty <span id="kPD" class="mut">–</span></div>
      <div class="bar"><i id="kPDbar" class="grn" style="width:0"></i></div>
    </div>
    <div class="card kpi">
      <div class="kl">Throughput</div>
      <div class="kv" id="kOps">0</div>
      <div class="ks"><span id="kToks">0</span> tokens · <span id="kUnits">0</span> kernel-elts</div>
      <div id="kSpark" style="margin-top:12px"></div>
    </div>
    <div class="card kpi">
      <div class="kl">Queue</div>
      <div class="kv" id="kQueue">0</div>
      <div class="ks">waiting jobs</div>
      <div class="ks" style="margin-top:10px"><span class="mut">done</span> <b id="kDone">0</b> &nbsp;·&nbsp; <span class="mut">failed</span> <b id="kFail" class="err" style="font-weight:700">0</b></div>
    </div>
    <div class="card kpi">
      <div class="kl">On-wire sealing</div>
      <div class="kv" style="font-size:21px;letter-spacing:-.5px">AES-256-GCM</div>
      <div class="ks">every work unit, encrypted</div>
      <div class="ks" style="margin-top:12px"><a href="/metrics" target="_blank" rel="noopener">/metrics</a> <span class="mut">· wire to Grafana</span></div>
    </div>
  </div>

  <!-- FLEET -->
  <div class="section-title">
    <h2>Fleet</h2><span class="badge mut" id="fCount"></span><div class="ln"></div>
    <input id="fSearch" placeholder="filter name / type / OS" style="width:200px;max-width:44vw">
    <button class="ghost mini" id="pauseAll">Pause all</button>
    <button class="ghost mini" id="resumeAll">Resume all</button>
  </div>
  <div class="grid g-fleet" id="fleet"><div class="empty">connect a worker to see it here…</div></div>

  <!-- MODEL SERVING -->
  <div class="section-title"><h2>Model serving</h2><div class="ln"></div></div>
  <div class="card">
    <div class="row">
      <input id="mModel" value="gpt2" placeholder="HF model id (gpt2 · Qwen/Qwen2.5-0.5B …)" style="flex:1;min-width:200px">
      <input id="mWorker" placeholder="worker (optional)" style="width:150px">
      <label class="f"><input id="mPush" type="checkbox">download-free</label>
      <button class="grad" id="mLoad">Load</button>
    </div>
    <div class="mstatus" id="mStatus" style="margin-top:12px">No model loaded. Enter an id and press Load.</div>
    <div class="chatbox" id="mChat" style="display:none"></div>
    <div class="row" id="mChatBar" style="margin-top:12px;display:none">
      <input id="mPrompt" placeholder="message…  (Enter to send)" style="flex:1;min-width:200px">
      <label class="f">max <input id="mMax" type="number" value="96" style="width:66px"></label>
      <label class="f"><input id="mSamp" type="checkbox">sample</label>
      <button id="mSend" disabled>Send</button>
    </div>
  </div>

  <!-- NETWORK SELF-TEST -->
  <div class="section-title"><h2>Network self-test</h2><div class="ln"></div></div>
  <div class="card">
    <p class="hint">RTT gates single-request sharded decode (needs a LAN); bandwidth gates weight transfer &amp; throughput. Run it to see which workloads THIS fleet can actually host.</p>
    <div class="row">
      <button class="grad" id="netBtn">Run self-test</button>
      <span class="mstatus" id="netNote"></span>
    </div>
    <div id="netOut"></div>
  </div>

  <!-- KERNELS + LOGS -->
  <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr));margin-top:26px">
    <div class="card">
      <h2>Per-kernel jobs</h2>
      <div id="kernels" style="margin-top:10px"><span class="empty">no jobs yet</span></div>
    </div>
    <div class="card">
      <h2>Errors &amp; debug log</h2>
      <pre class="logs" id="logs">—</pre>
    </div>
  </div>
</div>

<div class="toasts" id="toasts"></div>

<div class="gate" id="gate">
  <div class="box">
    <div class="mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none"><rect x="4" y="4" width="16" height="16" rx="3" stroke="#fff" stroke-width="1.7"/><rect x="8.5" y="8.5" width="7" height="7" rx="1.3" fill="#fff"/></svg></div>
    <h3>Admin token required</h3>
    <p>Paste the token minted by the coordinator's setup wizard. It is stored only in this browser.</p>
    <div class="row2">
      <input id="tokGate" type="password" placeholder="paste admin token" autocomplete="off">
      <button class="grad" id="tokGateBtn">Unlock</button>
    </div>
  </div>
</div>

<script>
"use strict";
var K="moregpu_admin_token";

/* ---------- helpers ---------- */
function $(id){return document.getElementById(id);}
function tok(){return (localStorage.getItem(K)||"").trim();}
function H(){return {"content-type":"application/json","authorization":"Bearer "+tok()};}
function pct(x){return Math.round((x||0)*100)+"%";}
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];});}
function fmt(n){n=n||0;return n>=1e9?(n/1e9).toFixed(1)+"G":n>=1e6?(n/1e6).toFixed(1)+"M":n>=1e3?(n/1e3).toFixed(1)+"k":""+Math.round(n);}
function updn(s){if(s<60)return s+"s";if(s<3600)return Math.floor(s/60)+"m";if(s<86400)return Math.floor(s/3600)+"h";return Math.floor(s/86400)+"d";}

var TOAST_SEEN=0;
function toast(msg,kind){
  var wrap=$("toasts"),t=document.createElement("div");
  t.className="toast"+(kind?" "+kind:"");t.textContent=msg;wrap.appendChild(t);
  setTimeout(function(){t.style.transition="opacity .4s";t.style.opacity="0";setTimeout(function(){t.remove();},400);},4200);
}
function showGate(show){$("gate").classList.toggle("show",!!show);if(show){var g=$("tokGate");g.value=tok();setTimeout(function(){g.focus();},60);}}

/* ---------- fetch wrappers ---------- */
function jget(path){
  return fetch(path,{headers:H()}).then(function(r){
    if(r.status===401){throw {auth:true};}
    return r.json();
  });
}
function jpost(path,body){
  return fetch(path,{method:"POST",headers:H(),body:JSON.stringify(body||{})}).then(function(r){
    if(r.status===401){throw {auth:true};}
    return r.json();
  });
}
var AUTH_BAD=false;
function onAuthErr(){
  if(AUTH_BAD)return;AUTH_BAD=true;
  toast("Unauthorized — the admin token is missing or wrong.","err");
  showGate(true);
}

/* ---------- sparkline ---------- */
function spark(arr,w,h,col){
  arr=arr||[];
  if(!arr.length)return '<svg width="'+w+'" height="'+h+'"></svg>';
  var mx=1;for(var i=0;i<arr.length;i++)if(arr[i]>mx)mx=arr[i];
  var step=w/Math.max(1,arr.length-1),pts=[],area=[];
  for(var j=0;j<arr.length;j++){
    var x=(j*step).toFixed(1),y=(h-2-((arr[j]||0)/mx)*(h-4)).toFixed(1);
    pts.push(x+","+y);area.push(x+","+y);
  }
  var uid="g"+Math.random().toString(36).slice(2,8);
  var poly=pts.join(" ");
  var fill="0,"+h+" "+area.join(" ")+" "+w+","+h;
  return '<svg class="spark" viewBox="0 0 '+w+' '+h+'" preserveAspectRatio="none">'
    +'<defs><linearGradient id="'+uid+'" x1="0" y1="0" x2="0" y2="1">'
    +'<stop offset="0" stop-color="'+col+'" stop-opacity="0.32"/><stop offset="1" stop-color="'+col+'" stop-opacity="0"/></linearGradient></defs>'
    +'<polygon points="'+fill+'" fill="url(#'+uid+')"/>'
    +'<polyline points="'+poly+'" fill="none" stroke="'+col+'" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/></svg>';
}

/* ---------- worker control ---------- */
function ctl(id,body,silent){
  return fetch("/workers/"+encodeURIComponent(id)+"/control",{method:"POST",headers:H(),body:JSON.stringify(body)})
    .then(function(r){if(r.status===401){onAuthErr();return;}if(!silent)refresh();})
    .catch(function(){toast("Control request failed for "+id,"err");});
}
var FLEET=[];
function pauseAll(p){
  Promise.all((FLEET||[]).map(function(x){return ctl(x.id,{action:p?"pause":"resume"},true);}))
    .then(function(){toast((p?"Paused":"Resumed")+" "+((FLEET||[]).length)+" workers","ok");refresh();});
}

/* ---------- fleet render ---------- */
var DRAGGING=false;
function stateChip(x){
  if(x.paused){
    if(x.pausedReason==="schedule")return '<span class="state s-sched"><span class="sd"></span>scheduled-off</span>';
    if(x.pausedReason==="errors")return '<span class="state s-auto"><span class="sd"></span>auto-paused</span>';
    return '<span class="state s-pause"><span class="sd"></span>paused</span>';
  }
  if(x.busy)return '<span class="state s-work"><span class="sd"></span>working</span>';
  if(x.serving)return '<span class="state s-serve"><span class="sd"></span>serving</span>';
  return '<span class="state s-idle"><span class="sd"></span>idle</span>';
}
function renderFleet(){
  var el=$("fleet");if(!el)return;
  var ae=document.activeElement;
  if(DRAGGING)return;
  /* don't clobber an in-flight edit: a slider being dragged, or a schedule/nick field being typed into */
  if(ae&&ae.classList&&(ae.classList.contains("dutyslider")||ae.classList.contains("schedin")||ae.classList.contains("nickin")))return;
  var q=($("fSearch").value||"").toLowerCase();
  var list=(FLEET||[]).filter(function(x){
    return !q||((x.nick||"")+" "+x.id+" "+x.backend+" "+(x.os||"")+" "+(x.label||"")).toLowerCase().indexOf(q)>=0;
  });
  list.sort(function(a,b){return (b.share||0)-(a.share||0);});
  var shown=list.slice(0,200);
  $("fCount").textContent=(FLEET||[]).length?("showing "+shown.length+" of "+FLEET.length+(q?" matched":"")):"";
  if(!shown.length){el.innerHTML='<div class="empty">'+(FLEET.length?"no workers match that filter":"connect a worker to see it here…")+'</div>';return;}
  el.innerHTML=shown.map(function(x){
    var isGpu=x.backend==="gpu";
    var col=isGpu?"#34d399":"#fbbf24";
    var ceil=(x.ceil!=null?x.ceil:(x.poolDuty||0.6));
    var label=x.label?esc(x.label):esc(x.backend);
    var os=x.os?'<span>'+esc(x.os)+'</span>':"";
    var sched=(x.schedule&&x.schedule!=="always")?'<span>· '+esc(x.schedule)+'</span>':"";
    return '<div class="card wk">'
      +'<div class="wkh"><div style="min-width:0">'
        +'<div class="wkname">'+esc(x.nick||x.id)+(x.errors?' <span class="err">('+(x.errors|0)+' err)</span>':"")+'</div>'
        +'<div class="wksub"><span class="pill '+(isGpu?"gpu":"cpu")+'">'+(isGpu?"GPU":"CPU")+' · '+label+'</span>'+os+sched+'<span>up '+updn(x.uptimeS||0)+'</span></div>'
      +'</div>'+stateChip(x)+'</div>'
      +spark(x.trend,300,34,col)
      +'<div class="grid3">'
        +'<div><div class="cellv">'+pct(x.share)+'</div><div class="celll">share</div></div>'
        +'<div><div class="cellv">'+fmt(x.ops||0)+'</div><div class="celll">ops</div></div>'
        +'<div><div class="cellv">'+fmt(x.tokens||0)+'</div><div class="celll">tokens</div></div>'
        +'<div><div class="cellv">'+fmt(x.units||0)+'</div><div class="celll">kernel-elts</div></div>'
      +'</div>'
      +'<div class="util"><span>shards <b>'+(x.shards|0)+'</b></span><span>avg <b>'+(x.avgMs||0)+'ms</b></span><span>user <b>'+pct(x.userUtil)+'</b></span><span>duty <b>'+pct(x.poolDuty)+'</b></span></div>'
      +'<div class="ctl">'
        +'<button class="mini" data-act="'+(x.paused?"resume":"pause")+'" data-id="'+esc(x.id)+'" title="'+(x.paused?"resume":"pause")+'">'+(x.paused?"▶ resume":"⏸ pause")+'</button>'
        +'<div class="duty" title="usage ceiling — drag to change live"><input type="range" class="dutyslider" min="0.05" max="1" step="0.05" value="'+ceil+'" data-id="'+esc(x.id)+'"><span class="dv">'+Math.round(ceil*100)+'%</span></div>'
        +'<input class="schedin" value="'+esc(x.schedule||"always")+'" data-id="'+esc(x.id)+'" title="always · idle-only · a window like 22:00-07:00. Applied on Enter/blur.">'
        +'<input class="nickin" value="'+esc(x.nick||"")+'" placeholder="'+esc(x.id)+'" data-id="'+esc(x.id)+'" title="rename this worker — applied on Enter/blur">'
        +'<button class="mini danger" data-act="remove" data-id="'+esc(x.id)+'" title="remove worker">✕</button>'
      +'</div>'
    +'</div>';
  }).join("");
}

/* ---------- main refresh ---------- */
function refresh(){
  /* health is public — always try it so the header stays alive even without a token */
  fetch("/health").then(function(r){return r.json();}).then(function(h){
    $("hFleet").textContent=(h.fleet==null?"—":h.fleet);
    $("hQueue").textContent=(h.queue==null?"—":h.queue);
    $("hDot").className="dot "+(h.ok?"on":"off");
  }).catch(function(){$("hDot").className="dot off";});

  if(!tok()){showGate(true);return;}

  jget("/gpu").then(function(g){
    AUTH_BAD=false;$("gate").classList.remove("show");
    $("kSlots").textContent=(g.slots==null?"—":g.slots);
    $("kSlotsub").textContent=(g.gpuSlots||0)+" GPU · "+(g.cpuSlots||0)+" CPU · "+(g.busy||0)+" busy";
    $("kUU").textContent=pct(g.avgUserUtil);$("kUUbar").style.width=pct(g.avgUserUtil);
    $("kPD").textContent=pct(g.avgPoolDuty);$("kPDbar").style.width=pct(g.avgPoolDuty);
    $("kOps").textContent=fmt(g.totalOps||0);
    $("kToks").textContent=fmt(g.totalTokens||0);
    $("kUnits").textContent=fmt(g.totalUnits||0);
    $("kQueue").textContent=(g.queueDepth||0);
    $("kSpark").innerHTML=spark(g.poolTrend,300,40,"#818cf8");
    var pk=g.perKernel||{},ks=Object.keys(pk);
    $("kernels").innerHTML=ks.length?ks.map(function(k){return '<span class="k">'+esc(k)+' · '+(pk[k]|0)+'</span>';}).join(""):'<span class="empty">no jobs yet</span>';
  }).catch(function(e){if(e&&e.auth)onAuthErr();});

  jget("/workers").then(function(w){
    FLEET=Array.isArray(w)?w:[];renderFleet();
  }).catch(function(e){if(e&&e.auth)onAuthErr();});

  jget("/jobs").then(function(j){
    if(!Array.isArray(j))return;
    $("kDone").textContent=j.filter(function(x){return x.status==="done";}).length;
    $("kFail").textContent=j.filter(function(x){return x.status==="failed";}).length;
  }).catch(function(){});

  jget("/logs").then(function(L){
    if(!Array.isArray(L)){$("logs").textContent="—";return;}
    $("logs").innerHTML=L.slice(0,80).map(function(e){
      return '<span class="lg-'+esc(e.level)+'">'+new Date(e.ts).toLocaleTimeString()+" ["+esc(e.level)+"] "+esc(e.msg)+(e.ctx?" · "+esc(e.ctx):"")+'</span>';
    }).join("\\n")||"—";
  }).catch(function(){});
}

/* ---------- fleet events (delegated) ---------- */
$("fleet").addEventListener("click",function(ev){
  var b=ev.target.closest&&ev.target.closest("button[data-act]");if(!b)return;
  var id=b.getAttribute("data-id"),act=b.getAttribute("data-act");
  if(act==="remove"){if(!confirm("Remove worker "+id+"? It will be banned by key if signed."))return;ctl(id,{action:"remove"});}
  else ctl(id,{action:act});
});
$("fleet").addEventListener("pointerdown",function(ev){if(ev.target.classList&&ev.target.classList.contains("dutyslider"))DRAGGING=true;});
window.addEventListener("pointerup",function(){if(DRAGGING){DRAGGING=false;}});
$("fleet").addEventListener("input",function(ev){
  var s=ev.target;if(!s.classList||!s.classList.contains("dutyslider"))return;
  DRAGGING=true;
  var v=s.parentNode.querySelector(".dv");if(v)v.textContent=Math.round(s.value*100)+"%";
});
$("fleet").addEventListener("change",function(ev){
  var s=ev.target;if(!s.classList)return;
  if(s.classList.contains("dutyslider")){DRAGGING=false;ctl(s.getAttribute("data-id"),{ceil:parseFloat(s.value)},true);}
  else if(s.classList.contains("schedin")){ctl(s.getAttribute("data-id"),{schedule:(s.value||"always").trim()});}
  else if(s.classList.contains("nickin")){ctl(s.getAttribute("data-id"),{nick:s.value.trim()});}
});
$("fleet").addEventListener("keydown",function(ev){
  if(ev.key==="Enter"&&ev.target.classList&&(ev.target.classList.contains("schedin")||ev.target.classList.contains("nickin")))ev.target.blur();
});
$("fSearch").addEventListener("input",renderFleet);
$("pauseAll").addEventListener("click",function(){pauseAll(true);});
$("resumeAll").addEventListener("click",function(){pauseAll(false);});

/* ---------- token bar / gate ---------- */
function setToken(v){localStorage.setItem(K,(v||"").trim());$("tokTop").value=tok();AUTH_BAD=false;}
$("tokTop").value=tok();
$("tokTop").addEventListener("change",function(){setToken($("tokTop").value);toast("Token saved","ok");refresh();});
$("signout").addEventListener("click",function(){localStorage.removeItem(K);$("tokTop").value="";toast("Signed out");showGate(true);});
$("tokGateBtn").addEventListener("click",function(){setToken($("tokGate").value);showGate(false);toast("Token saved","ok");refresh();});
$("tokGate").addEventListener("keydown",function(ev){if(ev.key==="Enter")$("tokGateBtn").click();});

/* ---------- model serving ---------- */
var MID=null,POLLING=false;
function mstat(text,cls){var e=$("mStatus");e.className="mstatus"+(cls?" "+cls:"");e.innerHTML=text;}
function addMsg(role,text){
  var box=$("mChat");box.style.display="flex";
  var el=document.createElement("div");el.className="msg "+role;el.textContent=text;box.appendChild(el);box.scrollTop=box.scrollHeight;return el;
}
function modelReady(r){
  MID=(r&&r.id)||$("mModel").value.trim();
  $("mChatBar").style.display="flex";$("mChat").style.display="flex";$("mSend").disabled=false;
  var parts=[];
  if(r&&r.worker)parts.push("on "+esc(r.worker));
  if(r&&r.device)parts.push(esc(r.device));
  if(r&&r.n_params)parts.push(Number(r.n_params).toLocaleString()+" params");
  if(r&&r.mode==="download-free")parts.push("download-free"+(r.staging?" ("+esc(r.staging)+")":""));
  mstat("✓ ready — <b>"+esc(MID)+"</b>"+(parts.length?" · "+parts.join(" · "):"")+" — type a message below.","ok");
}
function pollStatus(id,t0){
  if(POLLING)return;POLLING=true;
  var tick=function(){
    jget("/model/status?id="+encodeURIComponent(id)).then(function(r){
      if(r.status==="ready"){POLLING=false;modelReady(r);return;}
      if(r.status==="error"){POLLING=false;mstat("✗ "+esc(r.error||"load failed"),"err");return;}
      if(r.status==="unknown"){POLLING=false;mstat("✗ model not found — try loading again","err");return;}
      mstat("streaming <b>"+esc(id)+"</b>… "+Math.round((Date.now()-t0)/1000)+"s (weights → worker)");
      setTimeout(tick,2000);
    }).catch(function(e){if(e&&e.auth){POLLING=false;onAuthErr();return;}setTimeout(tick,2500);});
  };
  tick();
}
$("mLoad").addEventListener("click",function(){
  if(!tok()){onAuthErr();return;}
  MID=null;$("mSend").disabled=true;
  var m=$("mModel").value.trim();if(!m){toast("Enter a model id","err");return;}
  var wk=$("mWorker").value.trim(),push=$("mPush").checked;
  mstat((push?"streaming ":"loading ")+"<b>"+esc(m)+"</b>… (this may take a moment)");
  var body={model:m,id:m,push:push};if(wk)body.worker=wk;if(push)body.async=true;
  jpost("/model/load",body).then(function(r){
    if(r.error){mstat("✗ "+esc(r.error),"err");return;}
    if(r.status==="loading"){pollStatus(m,Date.now());return;}
    modelReady(r);
  }).catch(function(e){if(e&&e.auth){onAuthErr();return;}mstat("✗ "+esc(e&&e.message||e),"err");});
});
function sendChat(){
  var p=$("mPrompt"),text=p.value.trim();if(!text)return;
  if(!MID){toast("Load a model first","err");return;}
  addMsg("u",text);p.value="";
  var rep=addMsg("b","…"),sb=$("mSend");sb.disabled=true;
  var max=(+$("mMax").value)||96;
  jpost("/model/chat",{id:MID,prompt:text,max_new_tokens:max,do_sample:$("mSamp").checked}).then(function(r){
    if(r.error){rep.textContent="✗ "+r.error;rep.className="msg b";sb.disabled=false;return;}
    rep.textContent=r.text||"(empty)";
    if(r.ms||r.worker){var m=document.createElement("span");m.className="meta";m.textContent=(r.ms?(r.ms/1000).toFixed(1)+"s":"")+(r.worker?" · "+r.worker:"");rep.appendChild(m);}
    sb.disabled=false;$("mChat").scrollTop=$("mChat").scrollHeight;
  }).catch(function(e){if(e&&e.auth){onAuthErr();}rep.textContent="✗ "+(e&&e.message||e);sb.disabled=false;});
}
$("mSend").addEventListener("click",sendChat);
$("mPrompt").addEventListener("keydown",function(ev){if(ev.key==="Enter")sendChat();});

/* ---------- network self-test ---------- */
$("netBtn").addEventListener("click",function(){
  if(!tok()){onAuthErr();return;}
  var b=$("netBtn");b.disabled=true;b.textContent="probing…";$("netNote").textContent="";$("netNote").className="mstatus";
  jget("/net").then(function(d){
    b.disabled=false;b.textContent="Run self-test";
    if(d.error){$("netNote").textContent="✗ "+d.error;$("netNote").className="mstatus err";$("netOut").innerHTML="";return;}
    $("netNote").className="mstatus ok";
    $("netNote").textContent="median RTT "+(d.median_rtt_ms==null?"–":d.median_rtt_ms+" ms")+(d.median_up_mbps!=null?" · "+d.median_up_mbps+" Mbps up":"");
    var rows=(d.workers||[]).map(function(w){
      return '<tr><td>'+esc(w.id)+'</td><td class="mono">'+esc(w.backend||"")+'</td><td class="mono">'+(w.rtt_ms==null?"–":w.rtt_ms+" ms")+'</td><td class="mono">'+(w.up_mbps==null?esc(w.up_note||"–"):w.up_mbps+" Mbps")+'</td></tr>';
    }).join("");
    var li=function(a){return (a||[]).map(function(x){return "<li>"+esc(x)+"</li>";}).join("");};
    var out='<table class="nettbl"><thead><tr><th>worker</th><th>type</th><th>RTT</th><th>up</th></tr></thead><tbody>'+(rows||'<tr><td colspan="4" class="empty">no torch workers to probe</td></tr>')+'</tbody></table>';
    out+='<div class="verdict">'
      +'<div class="vbox ok"><h4>✓ works anywhere (latency-tolerant)</h4><ul>'+li(d.latency_tolerant_anywhere)+'</ul></div>'
      +'<div class="vbox warn"><h4>needs low RTT (single-request sharded decode)</h4><ul>'+li(d.latency_bound_needs_low_rtt)+'</ul></div>'
      +'</div>';
    if(d.note)out+='<div class="note">'+esc(d.note)+'</div>';
    $("netOut").innerHTML=out;
  }).catch(function(e){
    b.disabled=false;b.textContent="Run self-test";
    if(e&&e.auth){onAuthErr();return;}
    $("netNote").className="mstatus err";$("netNote").textContent="✗ "+(e&&e.message||e);
  });
});

/* ---------- boot ---------- */
if(!tok())showGate(true);
refresh();
setInterval(refresh,2000);
</script>`;

// ---- /chat : a minimal chatbot page to test a served model (text↔text via the worker's tokenizer) ----
const CHAT_HTML = `<!doctype html><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1"><title>MoreGPU · chat</title>
<style>
:root{--bg:#0b0f17;--card:#111827;--line:#1f2937;--fg:#e5e7eb;--mut:#8792a8;--acc:#818cf8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:20px}
.logo{font:9px/1.05 ui-monospace,Menlo,monospace;white-space:pre;background:linear-gradient(90deg,#5f5fff,#875fff,#af5fff,#d75fff,#ff5faf,#ff5f5f);-webkit-background-clip:text;background-clip:text;color:transparent;margin:0 0 6px}
h1{font-size:18px;margin:0 0 12px}a{color:#a5b4fc}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
input,button{font:inherit;background:#0e1420;color:var(--fg);border:1px solid var(--line);border-radius:9px;padding:8px 11px}
input[type=password],#model{min-width:150px}#prompt{flex:1;min-width:220px}
button{background:var(--acc);border-color:var(--acc);color:#0b1020;font-weight:600;cursor:pointer}
button.g{background:#0e1420;color:var(--fg);border-color:var(--line);font-weight:500}
.mut{color:var(--mut);font-size:13px}
.chat{min-height:260px;max-height:56vh;overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:10px;margin:10px 0}
.msg{padding:9px 12px;border-radius:12px;max-width:88%;white-space:pre-wrap;word-wrap:break-word}
.msg.user{align-self:flex-end;background:#312e81;border:1px solid #4338ca}
.msg.bot{align-self:flex-start;background:#0e1420;border:1px solid var(--line)}
</style>
<div class=wrap>
<pre class=logo>███╗   ███╗ ██████╗ ██████╗ ███████╗ ██████╗ ██████╗ ██╗   ██╗
██║ ╚═╝ ██║╚██████╔╝██║  ██║███████╗╚██████╔╝██║     ╚██████╔╝</pre>
<h1>MoreGPU · chat <span class=mut>— test a model served on your pool · <a href="/">admin panel</a></span></h1>
<div class=bar>
<input id=tok type=password placeholder="admin token">
<input id=model value="gpt2" placeholder="HF model (gpt2 · Qwen/Qwen2.5-0.5B …)">
<input id=worker placeholder="worker (optional, e.g. colab-cuda)" style="min-width:150px">
<label class=mut title="OFF (default, fast): the worker downloads the model itself — quick on a machine with good internet (e.g. a Colab GPU). ON: the coordinator streams the weights so the worker never touches the HF hub and keeps nothing on its disk — but it's SLOWER (sealed streaming, ~minutes on Colab)."><input id=pushfree type=checkbox> download-free</label>
<label class=mut title="Split the model across ALL torch workers (pipeline parallel) so EVERY node contributes to each token — for a model too big for one node. Slower per token (each token hops between nodes), always download-free. Ignores the worker field."><input id=shardmode type=checkbox> shard across fleet</label>
<button onclick=load()>Load</button>
<span id=status class=mut></span>
</div>
<div id=chat class=chat></div>
<div class=bar>
<input id=prompt placeholder="message…  (Enter to send)" onkeydown="if(event.key==='Enter')send()">
<button onclick=send() id=sendb disabled title="Load a model first — Send enables when it's ready">Send</button>
<label class=mut>max <input id=mnt type=number value=96 style="width:64px"></label>
<label class=mut><input id=samp type=checkbox> sample</label>
</div>
<p class=mut><b>Load</b> a model, wait for the status to say <b>✓ ready</b> (Send stays disabled until then), then chat. Leave <b>download-free</b> <b>off</b> for quick interactive chat on a Colab GPU (the worker downloads the model itself — fast). Turn it <b>on</b> only when you want the fleet to touch <em>nothing</em> on disk — the coordinator streams the sealed weights (no HF hub on the worker, staged in RAM), which is slower (~minutes on Colab). Pin a specific <b>worker</b> to place the model.</p>
</div>
<script>
var K='moregpu_admin_token',tokEl=document.getElementById('tok');tokEl.value=localStorage.getItem(K)||'';
tokEl.onchange=function(){localStorage.setItem(K,tokEl.value.trim());};
function H(){return {'content-type':'application/json','authorization':'Bearer '+(localStorage.getItem(K)||tokEl.value.trim())};}
var MID=null,SHARDED=false;
function add(role,text){var d=document.getElementById('chat'),el=document.createElement('div');el.className='msg '+role;el.textContent=text;d.appendChild(el);d.scrollTop=d.scrollHeight;return el;}
function enableSend(){var sb=document.getElementById('sendb');sb.disabled=false;sb.title='';}
function ready(m,r,s){MID=r.id||m;SHARDED=false;enableSend();s.textContent='✓ ready — '+MID+' on '+(r.worker||'?')+(r.device?' · '+r.device:'')+' · '+((r.n_params||0)).toLocaleString()+' params'+(r.mode==='download-free'?' · download-free ('+(r.staging||'?')+')':'')+' — type a message and Send';}
function readyShard(m,r,s){MID=m;SHARDED=true;enableSend();var st=(r.stages||[]).map(function(x){return x.worker+'['+x.start+'-'+x.end+')';}).join(' → ');s.textContent='✓ ready — '+m+' SHARDED across '+((r.stages||[]).length)+' nodes: '+st+' — every node contributes per token';}
function pollShard(m,s,t0){fetch('/model/shard_status?id='+encodeURIComponent(m),{headers:H()}).then(function(r){return r.json();}).then(function(r){
  if(r.status==='ready'){readyShard(m,r,s);return;}
  if(r.status==='error'){s.textContent='✗ '+r.error;return;}
  s.textContent='sharding '+m+' across the fleet… '+Math.round((Date.now()-t0)/1000)+'s ('+(r.stages_done||0)+'/'+(r.stages_total||'?')+' stages streamed)';setTimeout(function(){pollShard(m,s,t0);},2000);
 }).catch(function(){setTimeout(function(){pollShard(m,s,t0);},2500);});}
function poll(m,s,t0){fetch('/model/status?id='+encodeURIComponent(m),{headers:H()}).then(function(r){return r.json();}).then(function(r){
  if(r.status==='ready'){ready(m,r,s);return;}
  if(r.status==='error'){s.textContent='✗ '+r.error;return;}
  s.textContent='streaming '+m+'… '+Math.round((Date.now()-t0)/1000)+'s (weights → worker)';setTimeout(function(){poll(m,s,t0);},2000);
 }).catch(function(){setTimeout(function(){poll(m,s,t0);},2500);});}
function load(){localStorage.setItem(K,tokEl.value.trim());MID=null;var sb=document.getElementById('sendb');sb.disabled=true;sb.title='Loading… Send enables when the model is ready';var m=document.getElementById('model').value.trim(),wk=document.getElementById('worker').value.trim(),pf=document.getElementById('pushfree').checked,sh=document.getElementById('shardmode').checked,s=document.getElementById('status');
 if(sh){ // SHARD across all nodes — always download-free + async; every node contributes per token
  s.textContent='sharding '+m+' across the fleet… (Send disabled until ready)';
  fetch('/model/shard',{method:'POST',headers:H(),body:JSON.stringify({model:m,id:m,push:true,async:true})}).then(function(r){return r.json();}).then(function(r){
   if(r.error){if(/already loaded/i.test(r.error)){pollShard(m,s,Date.now());return;}s.textContent='✗ '+r.error;return;} // attach to an in-flight/ready shard instead of erroring
   if(r.status==='loading'){pollShard(m,s,Date.now());return;}
   readyShard(m,r,s);}).catch(function(e){s.textContent='✗ '+e;});
  return;
 }
 s.textContent=(pf?'streaming ':'loading ')+m+'… (Send disabled until ready)';
 // async for download-free (a big push outlives a public tunnel's request timeout) — return fast, then poll status
 var b={model:m,id:m,push:pf};if(wk)b.worker=wk;if(pf)b.async=true;
 fetch('/model/load',{method:'POST',headers:H(),body:JSON.stringify(b)}).then(function(r){return r.json();}).then(function(r){
  if(r.error){s.textContent='✗ '+r.error;return;}
  if(r.status==='loading'){poll(m,s,Date.now());return;}
  ready(m,r,s);}).catch(function(e){s.textContent='✗ '+e;});}
function send(){var p=document.getElementById('prompt'),text=p.value.trim();if(!text)return;if(!MID){alert('Load a model first (top bar).');return;}
 add('user',text);p.value='';var rep=add('bot','…'),sb=document.getElementById('sendb');sb.disabled=true;
 var mnt=(+document.getElementById('mnt').value)||96;
 var url=SHARDED?'/model/shard_chat':'/model/chat';
 var body=SHARDED?{id:MID,prompt:text,max_new_tokens:mnt}:{id:MID,prompt:text,max_new_tokens:mnt,do_sample:document.getElementById('samp').checked};
 fetch(url,{method:'POST',headers:H(),body:JSON.stringify(body)}).then(function(r){return r.json();}).then(function(r){
  rep.textContent=r.error?('✗ '+r.error):((r.text||'(empty)')+(r.ms?'   ·  '+(r.ms/1000).toFixed(1)+'s '+(SHARDED?('across '+((r.workers||[]).length)+' nodes'):('on '+r.worker)):''));sb.disabled=false;}).catch(function(e){rep.textContent='✗ '+e;sb.disabled=false;});}
// deep link: /chat?model=NAME&shard=1 → prefill the model, tick "shard across fleet", and auto-Load once a token is present.
// (token is never taken from the URL — it stays in localStorage; the model auto-shards across every node into one pipe.)
(function(){var q=new URLSearchParams(location.search),m=q.get('model'),sh=q.get('shard');
 if(m)document.getElementById('model').value=m;
 if(sh&&sh!=='0')document.getElementById('shardmode').checked=true;
 if(!(m||sh))return;
 if((localStorage.getItem(K)||'').trim()){load();return;}
 document.getElementById('status').textContent='enter your admin token above — this '+(sh&&sh!=='0'?'sharded ':'')+'model then loads automatically';
 tokEl.addEventListener('change',function once(){if((tokEl.value||'').trim()){tokEl.removeEventListener('change',once);load();}});
})();
</script>`;

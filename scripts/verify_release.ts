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
//              5 = manifest mismatch (a covered file tampered / missing / added, a __pycache__ dir, a symlinked dir)
//              6 = manifest version header missing / != <root>/pyproject.toml version / != --expect-version
//
// Usage:
//   deno run --allow-read scripts/verify_release.ts \
//     --artifact <path> --sig <path> --sha256 <hex> --pubkey <b64> --name <basename>
//
// MANIFEST mode (ADR-0103, the torch worker tree): the artifact is a signed MANIFEST.sha256 (a
// `# moregpu-worker-version: <v>` header, then `<sha256>  <path>` lines, written by `release_sign.py manifest`). After its
// signature verifies (a --sha256 pin is optional here: the signature binds its content), every listed file under
// --manifest-root must hash to its line, no unlisted file may exist ANYWHERE under the root (only the manifest, its .sig
// and other `*.sig` files are exempt), __pycache__ dirs are refused (a planted .pyc can shadow a verified .py), and the
// header must equal <root>/pyproject.toml's version (and --expect-version when given):
//   deno run --allow-read scripts/verify_release.ts --manifest-root apps/worker \
//     --artifact apps/worker/MANIFEST.sha256 --sig apps/worker/MANIFEST.sha256.sig --pubkey <b64> --name MANIFEST.sha256

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
const manifestRoot = argOf("--manifest-root");

if (!artifactPath || !sigPath || (!pinnedSha && !manifestRoot) || !pubkeyB64 || !name) {
  console.error(
    "[verify] usage: verify_release.ts --artifact P --sig P --sha256 HEX --pubkey B64 --name NAME [--manifest-root DIR]",
  );
  Deno.exit(2);
}

// (1) hash pin — recompute sha256 of the exact bytes that would be executed.
const data = await Deno.readFile(artifactPath);
const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", data));
const got = hex(digest);
if (pinnedSha && !eq(got, pinnedSha)) {
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

if (manifestRoot) {
  // (3) every covered file matches the signed manifest; nothing unlisted hides anywhere under the root
  const root = manifestRoot.replace(/\/+$/, "");
  const PREFIX = "# moregpu-worker-version: ";
  const excluded = (rel: string) => rel === "MANIFEST.sha256" || rel === "MANIFEST.sha256.sig" || rel.endsWith(".sig");
  const listed = new Map<string, string>();
  const bad: string[] = [];
  let lines = new TextDecoder().decode(data).split("\n");
  let version = "";
  if (lines.length && lines[0].startsWith(PREFIX)) { version = lines[0].slice(PREFIX.length).trim(); lines = lines.slice(1); }
  for (const line of lines) {
    if (!line) continue;
    const m = line.match(/^([0-9a-f]{64}) {2}(.+)$/);
    if (!m || m[2].startsWith("/") || m[2].split("/").includes("..") || excluded(m[2])) { bad.push(`malformed line: ${line.slice(0, 80)}`); continue; }
    listed.set(m[2], m[1]);
  }
  for (const [rel, want] of listed) {
    let bytes: Uint8Array;
    try { bytes = await Deno.readFile(`${root}/${rel}`); } catch { bad.push(`missing ${rel}`); continue; }
    const h = hex(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes as BufferSource)));
    if (!eq(h, want)) bad.push(`tampered ${rel}`);
  }
  const walk = async (dir: string, rel: string): Promise<void> => {
    for await (const e of Deno.readDir(dir)) {
      const r = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory && e.name === "__pycache__") bad.push(`bytecode cache ${r}/ (purge it)`);
      else if (e.isDirectory) await walk(`${dir}/${e.name}`, r);
      else if (e.isSymlink && (await Deno.stat(`${dir}/${e.name}`).catch(() => null))?.isDirectory) bad.push(`symlinked directory ${r}`);
      else if (!excluded(r) && !listed.has(r)) bad.push(`unlisted ${r}`);
    }
  };
  await walk(root, "");
  if (bad.length) {
    console.error(`[verify] REJECT ${name}: ${bad.length} file(s) do not match the signed manifest`);
    for (const b of bad.slice(0, 20)) console.error(`         ${b}`);
    Deno.exit(5);
  }
  let have = "";
  try { have = (await Deno.readTextFile(`${root}/pyproject.toml`)).match(/^version\s*=\s*"([^"]+)"/m)?.[1] ?? ""; } catch { /* no pyproject */ }
  const expect = argOf("--expect-version");
  if (!version || version !== have || (expect && version !== expect)) {
    console.error(`[verify] REJECT ${name}: manifest version ${version || "(missing)"} != pyproject.toml ${have || "(missing)"}` +
      (expect ? ` / expected ${expect}` : ""));
    Deno.exit(6);
  }
  console.log(`[verify] OK ${name}: sha256=${got} · signed by pinned release key · ${listed.size} files match · version ${version}`);
  Deno.exit(0);
}

console.log(`[verify] OK ${name}: sha256=${got} · signed by pinned release key`);
Deno.exit(0);

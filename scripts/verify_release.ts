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

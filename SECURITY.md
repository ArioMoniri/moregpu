# Security Policy

The MoreGPU maintainer takes the security of the project seriously. Because
MoreGPU coordinates compute across many machines and moves sealed work units
over the network, we ask that vulnerabilities be reported **privately** so they
can be fixed before they are disclosed publicly.

## Supported Versions

MoreGPU is distributed primarily as source that admins and workers run directly
from the `main` branch (via the documented `deno run` and installer commands).
Security fixes are applied to the latest code on `main`.

| Version                        | Supported          |
| ------------------------------ | ------------------ |
| `main` (latest commit)         | :white_check_mark: |
| Latest tagged release          | :white_check_mark: |
| Older commits / tags           | :x:                |

If you are running MoreGPU, please track the latest `main` (or the latest tagged
release) so that you receive security fixes. Older revisions do not receive
backported patches.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.** Public disclosure before a fix is available can
put other operators and their machines at risk.

Instead, report privately using **GitHub's private vulnerability reporting**:

1. Go to the repository's Security tab:
   https://github.com/ArioMoniri/moregpu/security
2. Open a new private report:
   https://github.com/ArioMoniri/moregpu/security/advisories/new

This creates a private GitHub Security Advisory visible only to you and the
maintainer (**Ariorad Moniri**). If you are unable to use GitHub Security
Advisories, please contact the maintainer through another private channel rather
than opening a public issue.

### What to include

To help triage and reproduce the issue, please include as much of the following
as you can:

* A description of the vulnerability and its potential impact
* The affected component (e.g. admin/coordinator server, worker agent,
  installer scripts, a specific package in the monorepo)
* The commit hash or version you tested
* Step-by-step reproduction instructions or a proof of concept
* Any relevant configuration (OS, GPU/CPU, environment variables) — but **do not
  include real admin tokens, worker join tokens, or encryption keys** in your
  report; redact them
* Suggested remediation, if you have one

### What to expect

* **Acknowledgement:** We aim to acknowledge your report within a few business
  days.
* **Assessment:** We will investigate, confirm the issue, and determine the
  affected versions and severity.
* **Fix & disclosure:** We will work on a fix and coordinate a disclosure
  timeline with you. We ask that you keep the details private until a fix is
  released and operators have a reasonable opportunity to update.
* **Credit:** With your permission, we are happy to credit you in the advisory
  and release notes. You may also request to remain anonymous.

This is a volunteer-maintained open-source project provided **AS IS** under the
Apache-2.0 license, with no warranty and no service-level guarantee. Response
times are best-effort.

## Scope and safe-harbor guidance

Security research and testing must be conducted **only against infrastructure
you own or are explicitly authorized to test.** Do not test against, attack, or
access pools, servers, or worker machines operated by others without permission.
Reports arising from unauthorized access to third-party systems are out of scope,
and nothing in this policy authorizes any activity that would violate applicable
law or a third party's Terms of Service (see [`ACCEPTABLE_USE.md`](./ACCEPTABLE_USE.md)).

## Operator security reminders

MoreGPU generates a distinct admin token, worker join token, and encryption key
for each pool on the first server run, and every pool is isolated. To keep your
pool secure:

* Keep your admin token, worker join token, and encryption key secret; never
  commit them to source control or paste them into issues.
* Prefer TLS for worker connections (`wss://`) using `MOREGPU_TLS_CERT` and
  `MOREGPU_TLS_KEY` when exposing the server beyond a trusted local network.
* Rotate tokens if you suspect they have been exposed.
* Only allow workers from machines you and your participants are authorized to
  enroll.

---

## Cryptography in MoreGPU today

MoreGPU is a **single trust domain**. Every mechanism below protects work units *in transit between the coordinator and cooperating workers* and establishes *which worker produced a result*. None of it hides job data *from* a worker: a worker that clears the join gate is a full participant that holds the decryption key. Read this section with that boundary in mind.

**AES-256-GCM payload sealing (confidentiality + integrity on the wire).** Each pool mints one 256-bit tenant key (`crypto.getRandomValues(32)`), stored in `.moregpu-server.json`. Every shard the coordinator dispatches and every result a worker returns is AES-GCM sealed with a fresh random 12-byte IV; the GCM tag authenticates the ciphertext. This gives confidentiality and tamper-detection for the shard payloads themselves even over a plaintext `ws://` socket.
- *Honest limits:* the key is a **symmetric secret shared with every enrolled worker**. It is **not** derived via any KDF/HKDF — it is the raw random key, delivered verbatim inside the plaintext `welcome` frame at join time. Any holder of the **join token** therefore receives the full tenant key and can decrypt all sealed traffic. The seal protects data between honest endpoints; it is not confidential computing.

**Ed25519 per-worker result signatures (authenticity + tamper-evidence).** Each worker generates a keypair at startup and registers its raw public key. It signs `shardId|iv|ct` for every result; the coordinator verifies with the registered key and **rejects** the shard on a bad signature.
- *Honest limits:* the signature covers the **sealed ciphertext and shard id, not the correctness of the math** — it proves *who* signed and that the bytes were not altered in flight, nothing about whether the computation was right. Signing is **conditional**: a worker that registers without a public key is still accepted and runs *unsigned*; the per-job `signed` flag is true only when every shard in that job was signed and verified.

**Result binding + unguessable shard ids (anti-forgery).** Shard ids are `s-<seq>-<random6>`, and the pending table binds each id to the exact worker it was dispatched to — a result whose `workerId` does not match is dropped. In-flight shards carry a 120 s timeout and are rejected on worker disconnect, so a job fails fast instead of hanging. A result can only settle the shard it was issued for, on the worker it was sent to.

**CPU-reference cross-check ("verified" flag).** The coordinator independently recomputes the job on its own CPU and compares against the pooled result within a tolerance. Elementwise and row-wise kernels are always checked; **matmul is only checked when `M·N ≤ 640×640`**.
- *Honest limits:* this is a numeric recompute-and-compare, **not a cryptographic proof and not a redundancy quorum**. Large matmuls run unverified, and because the coordinator does the whole reference calc itself it only scales as a spot check.

**Pool isolation + token gating.** Per-pool **admin token** (bearer, constant-time compare) gates every admin and `/submit` endpoint; a separate per-pool **join token** gates worker enrollment; each pool has its own key. `/health` and `/help` are public; `/ws` is join-token-gated.
- *Honest limit:* the admin-token holder submits plaintext tensors and reads plaintext outputs (data mode), so **the coordinator sees all job I/O in the clear** — confidentiality is on-the-wire, not end-to-end.

**Transport security.** Setting `MOREGPU_TLS_CERT` + `MOREGPU_TLS_KEY` upgrades to `https`/`wss`. Without them the server runs plaintext `ws` and prints an explicit wizard warning that the join handshake — *including the tenant key* — travels in cleartext.

**Signed, hash-pinned worker releases (supply-chain integrity of the installed code).** The one-liner installer (`scripts/install.sh`) no longer runs whatever `main` happens to serve at fetch time. It is the **root of trust**: it pins the release signer's Ed25519 **public key** and the exact **sha256** of `worker.ts`, fetches the artifact plus a detached signature, and refuses to run it unless **both** hold — (1) the bytes on disk hash to the pin, and (2) the signature verifies against the pinned key — over the domain-separated message `moregpu-release/v1\n<name>\n<sha256>` that binds the signature to that specific artifact (so a signature minted for one file cannot be replayed for another). Verification runs in the Deno the installer already ensures, using the **same WebCrypto Ed25519 verify the coordinator runs on every result** (`scripts/verify_release.ts`, embedded verbatim in the installer so there is no second unfetched-and-unverified download). Maintainers sign with `scripts/release_sign.py`; the **private key never enters the repo** — only the public key, the sha256, and the `.sig` ship. The gate fails **closed**: a hash or signature mismatch aborts before the worker is ever executed, and it re-checks the cached copy on every run, not just at first fetch.

*Release flow (maintainer):* `python3 scripts/release_sign.py keygen --out <secure-path>` once (key stays in a secret store), then per release `python3 scripts/release_sign.py sign --key <secure-path> apps/worker/worker.ts` — commit the emitted `apps/worker/worker.ts.sig` and paste the printed `RELEASE_PUBKEY_B64` / `WORKER_TS_SHA256` pins into `scripts/install.sh`. `tests/security/release_verify.py` proves the whole gate rejects tampered / wrong-key / unsigned artifacts. A clearly-labelled `MOREGPU_DEV_UNPINNED=1` skips verification for **local hacking only**.
- *Honest limits:* this authenticates the **distributed code**, not the running host — it stops a tampered CDN/MITM, a compromised mirror, or a poisoned `main` from delivering a modified worker, but a machine's owner can still run a locally-patched worker (the dev escape hatch exists for exactly that). Trust is anchored in the pins baked into the installer script you actually run, so fetch that script over `https` and read it before piping to a shell. Rotating the release key means shipping a new installer with new pins. The `moregpu join` subcommand's direct `deno run <raw-url>` path is **not yet** gated the same way (see roadmap).

**Bottom line:** MoreGPU today defends against a network eavesdropper and against a worker forging or tampering with *another* worker's result. It does **not** defend against a malicious worker reading the job data it was handed, nor prove that a worker computed honestly. There is **no TEE**: decrypted job data lives in ordinary worker memory, readable by the worker process, its OS, and its operator.

## Hardening roadmap (not yet implemented)

Everything below is **NOT YET BUILT** — it is the realistic next tier, listed roughly by security impact.

- **Confidential computing / TEE attestation — NOT YET BUILT, and not achievable on the commodity fleet.** True secrecy-of-job-data *from the worker* is the one thing sealing cannot give, and honestly it is **out of reach for the hardware MoreGPU actually targets**: confidential computing exists only on *datacenter* parts. Consumer **GeForce/RTX GPUs have no TEE**, and the **WebGPU (Dawn/wgpu) path cannot drive an NVIDIA CC-mode GPU** — that stack is CUDA + GPU-passthrough into a confidential VM. The only real 2026 path is a **separate, opt-in "attested tier" on rented cloud confidential hardware**, not a property the reclaimed-idle-desktop pool can have:
  - Run the worker inside a **CPU confidential VM** — AMD **SEV-SNP** (most mature) or Intel **TDX** (now GA on Azure/GCP) — and release the tenant key only *after* remote **attestation** of the VM measurement. This protects everything in RAM from the host/operator, but a GPU sits *outside* that boundary, so GEMM there would fall back to a CPU/software backend.
  - For secrecy *with* GPU acceleration, the worker would have to **drop WebGPU for CUDA** and run on a single passthrough **CC-mode H100/H200** via **Confidential Containers (CoCo)/Kata** (NVIDIA's CoCo Reference Architecture reached GA in 2026), with composite CPU+GPU attestation gating key release. Expect a "serialized bridge" penalty on the encrypted host↔device channel that hits small/transfer-bound fp32 matmul hardest.
  - Attestation would be brokered by a verifier (Intel Trust Authority, CCC **Veraison**/RATS, or AWS Nitro's KMS-gated model) before the coordinator hands over any key. This is a parallel *worker type* on paid datacenter hardware — not an upgrade the current commodity worker can receive.
- **Per-job ephemeral key rotation — NOT YET BUILT.** Today one long-lived pool key seals everything. Derive a fresh per-job (or per-shard) key — e.g. **HKDF** over the pool key plus a job nonce — so that exposure of one job's key does not compromise others, and keys can be discarded after a job completes.
- **N-of-M redundant recomputation with result quorum — NOT YET BUILT.** Because Ed25519 proves only *who signed*, not *that the math is correct*, and the CPU cross-check is skipped for large matmul, dispatch a random fraction of shards to a second (or Nth) worker and require quorum agreement — detecting a lying or faulty worker that returns a validly signed but wrong result.
- **Join-token rotation / short-lived enrollment tokens — NOT YET BUILT.** A static, long-lived join token grants the tenant key indefinitely. Move to short-TTL, single-use, or signed enrollment tokens so a leaked token expires quickly and enrollment can be revoked.
- **Replay / freshness nonces — NOT YET BUILT.** Replay resistance today is only implicit (single-use unguessable shard ids + the pending-table lifecycle). Add an explicit server-issued nonce/timestamp per assignment, echoed and signed by the worker, to harden against replay across reconnects.
- **Worker attestation tokens — NOT YET BUILT.** Cryptographic evidence of a worker's identity/integrity at enroll time (ideally bound to a hardware root of trust or TEE quote), rather than mere possession of a shared join secret.
- **TLS certificate pinning — NOT YET BUILT.** `wss://` currently trusts the platform CA set. Pin the coordinator's certificate or public key in the worker to defeat man-in-the-middle even against a rogue or mis-issued CA.
- **Signature gate on the remaining fetch paths — NOT YET BUILT.** The signed, hash-pinned release gate is live in `scripts/install.sh` (see "Signed, hash-pinned worker releases" above), but the `moregpu` CLI's `join`/`serve`/`isolate` subcommands still hand a `https://raw.githubusercontent.com/.../main/...` URL straight to `deno run`, which fetches and executes it with no pin. Route those through the same `verify_release.ts` gate (or a pinned local cache) so every code path that runs pool code is verified, not just the installer.

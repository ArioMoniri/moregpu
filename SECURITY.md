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

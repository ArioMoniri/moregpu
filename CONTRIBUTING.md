# Contributing to MoreGPU

Thanks for your interest in improving MoreGPU — a native GPU compute pool. This
guide covers local setup, how the code is checked, how to add a new compute
kernel, and our commit/PR conventions.

By contributing you agree that your contributions are licensed under the
project's [Apache-2.0](./LICENSE) license.

## Ground rules

MoreGPU is neutral infrastructure. Contributions must not add features whose
primary purpose is to run workers on machines or platforms you are not
authorized to use, or to circumvent any platform's Terms of Service (for
example, turning free CI runners, Colab, or free VPS trials into pool workers).
Run workers only on machines you own or are explicitly authorized to use. All
operational and legal responsibility rests with the operator.

Never commit secrets. Each pool's admin token, worker join token, and encryption
key are generated locally by the first-run wizard — they are secrets, not
fixtures. Do not commit `.env` files, `*.pem`, `*.key`, or any real token.

## Project layout

- `packages/*` — a TypeScript monorepo (npm workspaces) of tested libraries
  (crypto, protocol, scheduler, integrity, gpu, transport, runtime, coordinator).
- `apps/coordinator/server.ts` — the admin/coordinator server (runs on Deno).
- `apps/worker/worker.ts` — the worker agent (runs on Deno).
- `scripts/install.sh`, `scripts/install.ps1` — one-liner worker installers.
- `examples/` — runnable smoke demos.

The single cross-platform binary for the server and worker is Deno; the tested
libraries build and test under Node.

## Development setup

You need **Node.js 20+** and **Deno 2.x**.

```bash
# Install workspace dependencies
npm install

# Build all library packages
npm run build

# Run the test suite
npm test

# Lint
npm run lint
```

### Type-checking the Deno apps

The coordinator and worker run on Deno and are type-checked with Deno directly
(this is exactly what CI runs):

```bash
deno check apps/worker/worker.ts apps/coordinator/server.ts
```

### Running it end to end

```bash
# Terminal 1 — start the admin server (prints the setup wizard with tokens
# and the exact worker join command; dashboard on http://localhost:8787)
deno run --allow-net --allow-env --allow-read --allow-write \
  https://raw.githubusercontent.com/ArioMoniri/moregpu/main/apps/coordinator/server.ts

# Terminal 2 — join a worker on this machine (macOS/Linux)
curl -fsSL https://raw.githubusercontent.com/ArioMoniri/moregpu/main/scripts/install.sh \
  | MOREGPU_SERVER=wss://ADMIN:8787/ws MOREGPU_TOKEN=<join-token> sh

# Submit a job as the admin
curl -X POST http://ADMIN:8787/submit \
  -H "authorization: Bearer <admin-token>" \
  -H "content-type: application/json" \
  -d '{"size":1024}'
```

When developing locally, point the worker at your local source rather than the
raw GitHub URL.

## Adding a new compute task (kernel)

Task types today are row-sharded **matmul** and **vector_add**. The system is
extensible: a new task type is **a WGSL GPU kernel plus a matching CPU reference
implementation**. Both must produce identical results — the CPU reference is the
fallback path for CPU-only workers and the oracle that tests verify the GPU
output against.

To add one:

1. Write the WGSL kernel for the GPU path.
2. Write the CPU reference implementation with the same input/output contract.
3. Wire the task type into the scheduler/sharding so work units can be created,
   sealed, and pooled/verified like the existing types.
4. Add tests that run a shard on the CPU reference and assert the GPU path
   matches (within tolerance for floating point).
5. Document the task type and its request/response shape.

Keep files under 500 lines and use typed interfaces for public APIs.

## Commit & PR conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <short summary>
```

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`,
`build`, `perf`. Examples:

```
feat(scheduler): add row-sharded convolution task type
fix(worker): reconnect after coordinator restart
docs(readme): clarify the wizard token output
```

Pull requests should:

- Target the `main` branch and be scoped to one logical change.
- Pass CI — both the Node job (`npm ci`, `npm run build`, `npm test`) and the
  Deno job (`deno check` of the two apps).
- Pass `npm run lint`.
- Update `CHANGELOG.md` under **Unreleased** for any user-facing change.
- Fill out the pull request template, including the new-kernel checklist when
  adding a task type.

Before opening a PR, run the full check locally:

```bash
npm ci && npm run build && npm test && npm run lint
deno check apps/worker/worker.ts apps/coordinator/server.ts
```

## Reporting bugs and requesting features

Use the issue forms under **New issue**. Please redact any tokens, keys, or
hostnames from logs before posting. Security vulnerabilities should be reported
privately via a GitHub security advisory rather than a public issue.

## Releasing & publishing

The pool itself needs no install — the admin server and worker run straight from
their raw GitHub URLs. The client **SDKs** and CLI are distributed as GitHub
Release artifacts today; publishing them to public registries requires the
maintainer's account tokens and is done as follows.

**Build the artifacts:**

```bash
( cd clients/python && python3 -m build --wheel )     # → dist/moregpu_client-<ver>-py3-none-any.whl
( cd packages/client && npm run build && npm pack )   # → moregpu-client-<ver>.tgz
```

**Cut a GitHub Release** (what users install from — no registry account needed):

```bash
gh release create v<ver> \
  clients/python/dist/moregpu_client-<ver>-py3-none-any.whl \
  packages/client/moregpu-client-<ver>.tgz \
  --title "MoreGPU v<ver>" --notes "…"
```

**Publish to registries** (optional; needs the maintainer's own tokens — never commit them):

```bash
# PyPI  (needs a PyPI API token in ~/.pypirc or TWINE_PASSWORD)
python3 -m twine upload clients/python/dist/*

# npm  (needs `npm login`; the package is @moregpu/client)
( cd packages/client && npm publish --access public )

# Homebrew: move Formula/moregpu.rb into a tap repo `ArioMoniri/homebrew-moregpu`
#   so `brew install ArioMoniri/moregpu/moregpu` resolves, then point its `url`/`sha256`
#   at the release tarball.
```

Bump the version in `clients/python/pyproject.toml` and `packages/client/package.json`
together, update `CHANGELOG.md`, then tag and release.

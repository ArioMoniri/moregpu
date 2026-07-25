<!-- Thanks for contributing to MoreGPU! Please fill this out so review is quick. -->

## Summary

<!-- What does this PR do, and why? Link any related issue: "Closes #123". -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds capability)
- [ ] New task type / kernel (WGSL kernel + CPU reference)
- [ ] Breaking change (existing behavior changes)
- [ ] Docs / contributor experience
- [ ] CI / build / tooling

## How was this tested?

<!-- Commands you ran and what you observed. -->

- [ ] `npm ci && npm run build`
- [ ] `npm test`
- [ ] `deno check apps/worker/worker.ts apps/coordinator/server.ts`
- [ ] Manually verified against a running coordinator + worker (describe below)

<!-- Details of manual verification, if any: -->

## New kernel checklist (delete if not applicable)

- [ ] Added the WGSL GPU kernel.
- [ ] Added a CPU reference implementation that produces identical results.
- [ ] Added tests comparing GPU output against the CPU reference.
- [ ] Documented the task type and its input/output shape.

## Checklist

- [ ] I read `CONTRIBUTING.md`.
- [ ] My changes follow the existing code style; `npm run lint` passes.
- [ ] I did not commit secrets, `.env` files, tokens, or encryption keys.
- [ ] I updated `CHANGELOG.md` under the Unreleased section (for user-facing changes).
- [ ] My commits follow the Conventional Commits style described in `CONTRIBUTING.md`.

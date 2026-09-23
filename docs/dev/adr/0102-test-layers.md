# ADR-0102 — Test layers for strict TDD

**Status:** Accepted (2026-09-23) · **Milestone:** M1

## Context
Worker Python is tested only by standalone e2e scripts; coordinator code only by e2e. Unit-level red→green on DiLoCo
math, masking, EMA, loaders etc. needs fast, isolated tests with coverage.

## Decision
- **pytest** for new Python modules: `tests/py/` (unit), config in `apps/worker/pyproject.toml`
  (`[tool.pytest.ini_options]`), markers `gpu`, `cuda`, `webgpu`, `slow` — deselected by default in CI
  (`-m "not gpu and not cuda and not webgpu"`), run on real hardware with logs attached to the PR.
- `pytest-cov` with `--cov-fail-under=90` scoped to the new package (`moregpu_worker`), not to `worker_torch.py`.
- **vitest** for new coordinator modules (`apps/coordinator/lib/**/*.test.ts`; already matched by `vitest.config.ts`).
- Existing `tests/e2e|fault|security|scale/*.py` stay standalone and unchanged; new e2e tests follow the same style.
- Goldens: `tests/goldens/make_*.py` (checked in) regenerate `.npz` references with pinned NumPy/PyTorch/MONAI/scipy
  versions; tolerances stated in each test.
- CI `native` job gains: `pip install pytest pytest-cov` and `pytest tests/py tests/boundary`.
- CI adds `npm run lint` only if it is already green (not in scope to fix lint debt).

## Consequences
Two Python styles coexist; the rule is "unit → pytest, cross-process e2e → standalone script".

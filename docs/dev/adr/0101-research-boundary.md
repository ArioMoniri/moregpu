# ADR-0101 — Research boundary guard

**Status:** Accepted (2026-09-23) · **Milestone:** M1

## Context
The study (private repo) must never leak into MoreGPU. Only `docs/case-studies/*.md` may mention it.

## Decision
- `tests/boundary/research_boundary_test.py` (pytest, also runnable standalone) walks `git ls-files` and fails if any tracked
  file outside `docs/case-studies/` matches:
  - **terms** (case-insensitive, word-boundary): the study's dataset/collection names, benchmark names and study repo
    name, plus copyleft-licence header markers — the concrete list lives only in `tests/boundary/denylist.txt`, so this
    ADR itself stays clean;
  - **file types**: `.nii`, `.nii.gz`, `.dcm`, `.nrrd`, `.mha`;
  - **patient-ID patterns**: regexes for the dataset's subject-ID formats (also in `denylist.txt`).
- The denylist lives in `tests/boundary/denylist.txt`; the test and the denylist file are self-exempt. False positives are
  handled by an explicit, reviewed `tests/boundary/allow.txt` (path + term), never by weakening the regex.
- Red→green: the first commit adds the test with a fixture tree under `tmp_path` proving each rule fires, then runs it on
  the repo.

## Consequences
Generic medical-imaging words (NIfTI, DICOM, CT, liver) stay allowed — the vision data plane legitimately needs readers
for them. Only dataset-/study-specific terms are denied.

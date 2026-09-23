"""ADR-0101 research-boundary guard.

Fails if any tracked file outside docs/case-studies/ carries study/dataset-specific terms, copyleft (AGPL) markers,
medical-image files, or dataset subject-ID patterns. Runs under pytest and standalone (`python3 <this file>`).
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXEMPT_PREFIXES = ("docs/case-studies/",)
SELF = {"tests/boundary/denylist.txt", "tests/boundary/research_boundary_test.py", "tests/boundary/allow.txt"}
MAX_BYTES = 2_000_000  # skip scanning the content of very large tracked files (binaries/assets); names still checked


@dataclass(frozen=True)
class Rules:
    terms: list[re.Pattern]
    ids: list[re.Pattern]
    exts: tuple[str, ...]
    allow: set[tuple[str, str]]


def load_rules(denylist: Path = HERE / "denylist.txt", allow: Path = HERE / "allow.txt") -> Rules:
    terms, ids, exts = [], [], []
    for raw in denylist.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        kind, _, val = line.partition(":")
        if kind == "term":
            terms.append(re.compile(val, re.I))
        elif kind == "id":
            ids.append(re.compile(val, re.I))
        elif kind == "ext":
            exts.append(val.lower())
        else:
            raise ValueError(f"bad denylist line: {raw!r}")
    allowed: set[tuple[str, str]] = set()
    if allow.exists():
        for raw in allow.read_text().splitlines():
            if raw.strip() and not raw.startswith("#"):
                path, _, pat = raw.partition("\t")
                allowed.add((path.strip(), pat.strip()))
    return Rules(terms, ids, tuple(exts), allowed)


def scan(files: dict[str, bytes], rules: Rules) -> list[str]:
    """files: repo-relative path -> content. Returns human-readable violations."""
    out: list[str] = []
    for path, data in sorted(files.items()):
        if path.startswith(EXEMPT_PREFIXES) or path in SELF:
            continue
        low = path.lower()
        for ext in rules.exts:
            if low.endswith(ext):
                out.append(f"{path}: medical-image file type {ext}")
        if len(data) > MAX_BYTES:
            continue
        text = data.decode("utf-8", errors="ignore")
        for pat in rules.terms + rules.ids:
            if (path, pat.pattern) in rules.allow:
                continue
            m = pat.search(text) or pat.search(path)
            if m:
                line = text.count("\n", 0, m.start()) + 1 if pat.search(text) else 0
                out.append(f"{path}:{line}: matches denied pattern /{pat.pattern}/ ({m.group(0)!r})")
    return out


def tracked_files(repo: Path = REPO) -> dict[str, bytes]:
    names = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True).stdout.split(b"\0")
    out = {}
    for n in names:
        if not n:
            continue
        p = repo / n.decode()
        if p.is_file():
            out[n.decode()] = p.read_bytes()
    return out


# ------------------------------------------------------------------ tests
def _rules():
    return load_rules()


def test_each_rule_fires_on_a_fixture():
    r = _rules()
    bad = {
        "a.py": b"# data from HCC-TACE-Seg\n",
        "b.md": b"trained on LiTS and BTCV\n",
        "c.py": b"# SPDX-License-Identifier: AGPL-3.0-only\n",
        "d.txt": b"patient HCC_017 excluded\n",
        "e/vol.nii.gz": b"",
        "f/x.dcm": b"",
        "g.md": b"see moregpu-jepa-poc-study\n",
    }
    v = scan(bad, r)
    for name in bad:
        assert any(x.startswith(name) for x in v), (name, v)


def test_generic_imaging_words_and_substrings_are_allowed():
    ok = {"a.md": b"NIfTI and DICOM readers for liver CT; splits and litsomething; nibabel\n"}
    assert scan(ok, _rules()) == []


def test_case_studies_are_exempt():
    assert scan({"docs/case-studies/jepa.md": b"HCC-TACE-Seg results\n"}, _rules()) == []


def test_allowlist_suppresses_one_path_and_pattern(tmp_path):
    deny = tmp_path / "deny.txt"
    deny.write_text("term:\\bbtcv\\b\n")
    allow = tmp_path / "allow.txt"
    allow.write_text("x.md\t\\bbtcv\\b\n")
    r = load_rules(deny, allow)
    assert scan({"x.md": b"btcv"}, r) == []
    assert scan({"y.md": b"btcv"}, r) != []


def test_repository_is_clean():
    v = scan(tracked_files(), _rules())
    assert v == [], "research-boundary violations:\n" + "\n".join(v)


if __name__ == "__main__":
    v = scan(tracked_files(), _rules())
    print("\n".join(v) if v else "research boundary: clean")
    sys.exit(1 if v else 0)

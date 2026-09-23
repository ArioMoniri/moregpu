"""ADR-0113 repo-wide ban on unsafe deserialisation (pytest; also runs standalone: `python3 <this file>`).

Every tracked .py file is parsed (AST, so strings/comments never trip it) and must not contain:
  * `weights_only=` with anything but the literal True (e.g. `weights_only=False`);
  * `torch.load(...)` without `weights_only=True` on the same call;
  * bare `pickle.load(` / `pickle.loads(` / `pickle.Unpickler(` (also cPickle/_pickle/dill/cloudpickle, and
    `from pickle import load`);
  * `trust_remote_code=` with anything but the literal False;
  * under apps/worker/** only: a weights-loading `.from_pretrained(` (anything but *Config/*Tokenizer/*Processor/
    *FeatureExtractor) without `use_safetensors=True`.

Reviewed exceptions live in pickle_ban_allow.txt (`path<TAB>rule<TAB>substring of the call's source`). An entry that no
longer matches anything fails the test, so fixing the code forces the allowlist line to be deleted too.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PICKLE_MODS = {"pickle", "cPickle", "_pickle", "dill", "cloudpickle"}
PICKLE_FUNCS = {"load", "loads", "Unpickler"}
NON_WEIGHT_LOADERS = re.compile(r"(Config|Tokenizer\w*|Processor|FeatureExtractor|GenerationConfig)$")
FROM_PRETRAINED_SCOPE = ("apps/worker/",)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    rule: str
    source: str

    def __str__(self):
        return f"{self.path}:{self.line}: [{self.rule}] {self.source.strip()[:160]}"


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _is_const(node, value) -> bool:
    return isinstance(node, ast.Constant) and node.value is value


def _receiver(func: ast.Attribute) -> str:
    v = func.value
    if isinstance(v, ast.Call):
        v = v.func
    if isinstance(v, ast.Name):
        return v.id
    if isinstance(v, ast.Attribute):
        return v.attr
    return ""


def scan_source(path: str, text: str) -> list[Violation]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _scan_regex(path, text)
    lines = text.splitlines()
    out: list[Violation] = []

    def seg(node):
        return "\n".join(lines[node.lineno - 1:(node.end_lineno or node.lineno)])

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in PICKLE_MODS:
            if any(a.name in PICKLE_FUNCS for a in node.names):
                out.append(Violation(path, node.lineno, "pickle-load", seg(node)))
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        wo = _kw(node, "weights_only")
        if wo is not None and not _is_const(wo, True):
            out.append(Violation(path, node.lineno, "weights-only-false", seg(node)))
        trc = _kw(node, "trust_remote_code")
        if trc is not None and not _is_const(trc, False):
            out.append(Violation(path, node.lineno, "trust-remote-code", seg(node)))
        if isinstance(f, ast.Attribute):
            recv = _receiver(f)
            if f.attr == "load" and recv == "torch" and not _is_const(wo, True):
                out.append(Violation(path, node.lineno, "torch-load", seg(node)))
            if f.attr in PICKLE_FUNCS and recv in PICKLE_MODS:
                out.append(Violation(path, node.lineno, "pickle-load", seg(node)))
            if (f.attr == "from_pretrained" and path.startswith(FROM_PRETRAINED_SCOPE)
                    and not NON_WEIGHT_LOADERS.search(recv) and not _is_const(_kw(node, "use_safetensors"), True)):
                out.append(Violation(path, node.lineno, "from-pretrained-safetensors", seg(node)))
    return out


_REGEX_RULES = [
    ("weights-only-false", re.compile(r"weights_only\s*=\s*False")),
    ("trust-remote-code", re.compile(r"trust_remote_code\s*=\s*True")),
    ("pickle-load", re.compile(r"\b(?:c?[Pp]ickle|_pickle|dill|cloudpickle)\.(?:loads?|Unpickler)\(")),
    ("torch-load", re.compile(r"\btorch\.load\((?![^)]*weights_only\s*=\s*True)")),
]


def _scan_regex(path: str, text: str) -> list[Violation]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        for rule, pat in _REGEX_RULES:
            if pat.search(line):
                out.append(Violation(path, i, rule, line))
    return out


def load_allow(path: Path = HERE / "pickle_ban_allow.txt") -> list[tuple[str, str, str]]:
    out = []
    if path.exists():
        for raw in path.read_text().splitlines():
            if raw.strip() and not raw.lstrip().startswith("#"):
                p, rule, sub = raw.split("\t", 2)
                out.append((p.strip(), rule.strip(), sub.strip()))
    return out


def apply_allow(violations: list[Violation], allow) -> tuple[list[Violation], list[tuple[str, str, str]]]:
    used = set()
    left = []
    for v in violations:
        hit = [a for a in allow if a[0] == v.path and a[1] == v.rule and a[2] in v.source]
        if hit:
            used.update(hit)
        else:
            left.append(v)
    return left, [a for a in allow if a not in used]


def tracked_python(repo: Path = REPO) -> dict[str, str]:
    names = subprocess.run(["git", "ls-files", "-z", "*.py"], cwd=repo, check=True,
                           capture_output=True).stdout.split(b"\0")
    out = {}
    for n in names:
        if n and (repo / n.decode()).is_file():
            out[n.decode()] = (repo / n.decode()).read_text(errors="ignore")
    return out


def scan_repo(repo: Path = REPO):
    vs = [v for path, text in sorted(tracked_python(repo).items()) for v in scan_source(path, text)]
    return apply_allow(vs, load_allow())


# ------------------------------------------------------------------ tests
def _rules(src, path="apps/worker/x.py"):
    return sorted({v.rule for v in scan_source(path, src)})


def test_each_rule_fires_on_a_fixture():
    assert _rules("torch.load(p, weights_only=False)") == ["torch-load", "weights-only-false"]
    assert _rules("torch.load(p)") == ["torch-load"]
    assert _rules("torch.load(p, map_location='cpu',\n           weights_only=flag)") == ["torch-load",
                                                                                         "weights-only-false"]
    assert _rules("import pickle\npickle.loads(b)") == ["pickle-load"]
    assert _rules("pickle.load(f)") == ["pickle-load"]
    assert _rules("cPickle.Unpickler(f).load()") == ["pickle-load"]
    assert _rules("from pickle import loads") == ["pickle-load"]
    assert _rules("AutoModel.from_pretrained(x, trust_remote_code=True)") == ["from-pretrained-safetensors",
                                                                               "trust-remote-code"]
    assert _rules("AutoModelForCausalLM.from_pretrained(\n  x,\n  dtype=d)") == ["from-pretrained-safetensors"]
    assert _rules("transformers.AutoModel.from_pretrained(x)") == ["from-pretrained-safetensors"]
    assert _rules("torch.load(p\n") == ["torch-load"]  # unparsable file → regex fallback


def test_safe_forms_pass():
    assert _rules("torch.load(p, map_location='cpu', weights_only=True)") == []
    assert _rules("pickle.dumps(x); torch.save(x, p)") == []
    assert _rules("AutoModel.from_pretrained(x, use_safetensors=True, trust_remote_code=False)") == []
    assert _rules("AutoConfig.from_pretrained(x); AutoTokenizer.from_pretrained(x, local_files_only=True)") == []
    assert _rules("AutoImageProcessor.from_pretrained(x)") == []
    assert _rules("s = 'torch.load(p, weights_only=False)'  # pickle.loads(b)") == []
    assert _rules("AutoModel.from_pretrained(x)", path="tests/e2e/t.py") == []  # from_pretrained rule is worker-only


def test_allowlist_matches_by_substring_and_reports_stale_entries():
    vs = scan_source("apps/worker/w.py", "AutoModelForCausalLM.from_pretrained(cfg['model'], dtype=d)")
    allow = [("apps/worker/w.py", "from-pretrained-safetensors", "from_pretrained(cfg['model']"),
             ("apps/worker/w.py", "from-pretrained-safetensors", "gone(")]
    left, stale = apply_allow(vs, allow)
    assert left == [] and stale == [allow[1]]


def test_repository_has_no_unsafe_deserialisation():
    left, stale = scan_repo()
    assert left == [], "unsafe deserialisation (ADR-0113):\n" + "\n".join(map(str, left))
    assert stale == [], "stale entries in tests/security/pickle_ban_allow.txt — delete them:\n" + "\n".join(
        "\t".join(s) for s in stale)


if __name__ == "__main__":
    left, stale = scan_repo()
    for v in left:
        print(v)
    for s in stale:
        print("stale allowlist entry:", "\t".join(s))
    print("pickle ban: clean" if not (left or stale) else "pickle ban: FAIL")
    sys.exit(1 if left or stale else 0)

"""Manifests: a ``refs.jsonl`` list of :class:`Ref` (one JSON object per line) with a stable content hash.

``sha256`` is taken over the *canonical* JSONL (keys sorted, compact separators, defaults omitted, ``\\n`` after each
line) so the same refs hash the same regardless of key order, whitespace or blank lines in the source.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable, Iterator

from .refs import Ref


def _canon(ref: Ref) -> str:
    return json.dumps(ref.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Manifest:
    def __init__(self, source: "str | bytes | Iterable[Ref]"):
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        if isinstance(source, str):
            refs = []
            for n, line in enumerate(source.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    refs.append(Ref.from_json(json.loads(line)))
                except ValueError as e:  # json.JSONDecodeError is a ValueError
                    raise ValueError(f"manifest line {n}: {e}") from e
        else:
            refs = list(source)
        self.refs: list[Ref] = refs
        self._jsonl = "".join(_canon(r) + "\n" for r in refs)
        self.sha256 = hashlib.sha256(self._jsonl.encode("utf-8")).hexdigest()

    @classmethod
    def from_jsonl(cls, text: "str | bytes") -> "Manifest":
        return cls(text)

    def to_jsonl(self) -> str:
        return self._jsonl

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, i: int) -> Ref:
        return self.refs[i]

    def __iter__(self) -> Iterator[Ref]:
        return iter(self.refs)

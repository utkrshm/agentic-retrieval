"""Identifier-aware BM25 over the units of an index, used only for queries that contain a code anchor.

Dense retrieval handles behavioural questions well but ranks exact literals (an error string, a
function or config name) loosely. A query "has an anchor" when it holds a quoted literal, a snake_case,
camelCase or dotted identifier, and that anchor actually occurs in the indexed code. Only then are the
dense and lexical rankings fused (reciprocal rank fusion); every other query stays dense-only.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_SPLIT = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_QUOTED = re.compile(r"\"([^\"\n]{3,})\"|'([^'\n]{3,})'|`([^`\n]{3,})`")
_DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_STOP = {"the", "a", "an", "of", "to", "in", "is", "and", "or", "for", "it", "this", "that", "where", "what",
         "how", "does", "do", "which", "when", "from", "by", "on", "are", "be", "with", "as", "at", "its"}


def tokenize(text: str) -> list[str]:
    """Lower-case whole identifiers plus their sub-tokens (snake_case and camelCase parts)."""
    out: list[str] = []
    for m in _IDENT.finditer(text):
        word = m.group(0)
        low = word.lower()
        out.append(low)
        parts = [p.lower() for p in _CAMEL_SPLIT.findall(word)]
        if len(parts) > 1 or "_" in word:
            out.extend(p for p in parts if p != low)
    return out


def quoted_literals(query: str) -> list[str]:
    return [next(g for g in m.groups() if g).strip().lower() for m in _QUOTED.finditer(query)]


def identifier_anchors(query: str) -> list[str]:
    """Identifier-shaped tokens: snake_case, camelCase (inner capital) or dotted names."""
    found = []
    stripped = _QUOTED.sub(" ", query)
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", stripped):
        if ("_" in w.strip("_") and len(w) > 3) or re.search(r"[a-z][A-Z]", w):
            found.append(w.lower())
    for d in _DOTTED.findall(stripped):
        found.append(d.lower())
    return found


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.lower = [d.lower() for d in docs]  # for quoted-literal matching
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for i, d in enumerate(docs):
            toks = Counter(tokenize(d))
            lengths.append(sum(toks.values()))
            for term, tf in toks.items():
                self.postings[term].append((i, tf))
        self.n = len(docs)
        self.avg = (sum(lengths) / self.n) if self.n else 0.0
        self.lengths = np.asarray(lengths, dtype="float32")

    def _idf(self, df: int) -> float:
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def has_anchor(self, query: str) -> bool:
        """True if the query names a literal or identifier that exists in the indexed code."""
        for lit in quoted_literals(query):
            if any(lit in d for d in self.lower):
                return True
        return any(a in self.postings or all(p in self.postings for p in tokenize(a)) and len(tokenize(a)) > 1
                   for a in identifier_anchors(query))

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype="float32")
        terms = [t for t in dict.fromkeys(tokenize(query)) if t not in _STOP]
        for t in terms:
            post = self.postings.get(t)
            if not post:
                continue
            idf = self._idf(len(post))
            for i, tf in post:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                s[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        for lit in quoted_literals(query):  # a whole quoted phrase counts as one rare term
            hits = [i for i, d in enumerate(self.lower) if lit in d]
            if hits:
                idf = self._idf(len(hits))
                for i in hits:
                    s[i] += 2.0 * idf
        return s


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion of ranked id lists; returns (id, score) best first."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for r, i in enumerate(ranking, 1):
            fused[i] += 1.0 / (k + r)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

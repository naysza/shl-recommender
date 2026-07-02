from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from rank_bm25 import BM25Okapi

from app.catalog import CatalogItem

_TOKEN_RE = re.compile(r"[a-zA-Z0-9+#.]+")

SYNONYMS = {
    "js": ["javascript"],
    "javascript": ["js"],
    "py": ["python"],
    "oop": ["object oriented", "object-oriented"],
    "personality": ["behavior", "behaviour", "traits"],
    "leadership": ["management", "managerial"],
    "stakeholder": ["communication", "interpersonal"],
    "communication": ["stakeholder", "interpersonal", "verbal"],
    "teamwork": ["collaboration", "interpersonal"],
    "problem solving": ["reasoning", "aptitude", "critical thinking"],
    "coding": ["programming", "development"],
    "dev": ["development", "developer"],
    "aptitude": ["reasoning", "ability"],
    "sql": ["database", "query"],
    "cognitive": ["aptitude", "ability", "reasoning"],
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _expand_query(text: str) -> List[str]:
    tokens = _tokenize(text)
    expanded = list(tokens)
    lower_text = text.lower()
    for key, syns in SYNONYMS.items():
        if key in lower_text:
            for s in syns:
                expanded.extend(_tokenize(s))
    return expanded


@dataclass
class ScoredItem:
    item: CatalogItem
    score: float


class CatalogIndex:
    def __init__(self, items: List[CatalogItem]):
        self.items = items
        self._corpus_tokens = [_tokenize(item.search_text()) for item in items]
        self._bm25 = BM25Okapi(self._corpus_tokens) if items else None
        self._by_name_lower = {item.name.strip().lower(): item for item in items}

    def search(self, query: str, top_k: int = 25,
               job_level: Optional[str] = None,
               max_duration_minutes: Optional[int] = None) -> List[ScoredItem]:
        if not self._bm25 or not query.strip():
            return []
        q_tokens = _expand_query(query)
        scores = self._bm25.get_scores(q_tokens)
        scored = [ScoredItem(item, float(score)) for item, score in zip(self.items, scores)]
        scored = [s for s in scored if s.score > 0]

        if job_level:
            jl = job_level.strip().lower()
            scored = [
                s for s in scored
                if not s.item.job_levels
                or any(jl in lvl.lower() or lvl.lower() in jl for lvl in s.item.job_levels)
            ] or scored  # don't zero out results if the filter is too strict

        if max_duration_minutes is not None:
            def _duration_ok(s: ScoredItem) -> bool:
                m = re.search(r"\d+", s.item.duration or "")
                if not m:
                    return True  # unknown duration -> don't exclude
                return int(m.group()) <= max_duration_minutes
            filtered = [s for s in scored if _duration_ok(s)]
            scored = filtered or scored

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def find_by_name(self, name: str) -> Optional[CatalogItem]:
        key = name.strip().lower()
        if key in self._by_name_lower:
            return self._by_name_lower[key]
        # fuzzy: substring match either direction
        best = None
        for n, item in self._by_name_lower.items():
            if key in n or n in key:
                if best is None or len(n) < len(best.name.lower()):
                    best = item
        return best

    def is_known_url(self, url: str) -> bool:
        url = (url or "").strip().rstrip("/")
        return any(item.url.strip().rstrip("/") == url for item in self.items)

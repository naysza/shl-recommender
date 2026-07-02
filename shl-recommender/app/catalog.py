from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

from app.config import settings

logger = logging.getLogger("catalog")

INCLUDE_PACKAGED_SOLUTIONS = False

KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


@dataclass
class CatalogItem:
    entity_id: str
    name: str
    url: str
    description: str
    job_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    duration: str = ""
    remote: str = ""
    adaptive: str = ""
    keys: List[str] = field(default_factory=list)

    @property
    def test_type_codes(self) -> str:
        codes = [KEY_TO_CODE.get(k, "") for k in self.keys]
        return "".join(sorted(set(c for c in codes if c)))

    def search_text(self) -> str:
        parts = [
            self.name,
            self.description,
            " ".join(self.job_levels),
            " ".join(self.keys),
            f"duration {self.duration}",
        ]
        return " \n".join(p for p in parts if p)

    def to_prompt_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "test_type": self.test_type_codes,
            "description": self.description[:400],
            "job_levels": self.job_levels,
            "duration": self.duration,
            "remote": self.remote,
            "adaptive": self.adaptive,
        }


def _is_packaged_solution(name: str) -> bool:
    n = name.strip().lower()
    return n.endswith("solution") or n.endswith("solutions")


def _normalize(raw: dict) -> Optional[CatalogItem]:
    name = (raw.get("name") or "").strip()
    url = (raw.get("link") or raw.get("url") or "").strip()
    if not name or not url:
        return None
    if not INCLUDE_PACKAGED_SOLUTIONS and _is_packaged_solution(name):
        return None
    return CatalogItem(
        entity_id=str(raw.get("entity_id", "")),
        name=name,
        url=url,
        description=(raw.get("description") or "").strip(),
        job_levels=raw.get("job_levels") or [],
        languages=raw.get("languages") or [],
        duration=(raw.get("duration") or "").strip(),
        remote=(raw.get("remote") or "").strip(),
        adaptive=(raw.get("adaptive") or "").strip(),
        keys=raw.get("keys") or [],
    )


def _load_raw_from_path(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cache_is_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
    return age_hours < settings.CATALOG_MAX_CACHE_AGE_HOURS


def _fetch_live() -> list:
    with httpx.Client(timeout=settings.CATALOG_FETCH_TIMEOUT_SECONDS) as client:
        resp = client.get(settings.CATALOG_URL)
        resp.raise_for_status()
        return json.loads(resp.text, strict=False)


def load_catalog() -> List[CatalogItem]:
    raw: Optional[list] = None

    if _cache_is_fresh(settings.CATALOG_CACHE_PATH):
        try:
            raw = _load_raw_from_path(settings.CATALOG_CACHE_PATH)
            logger.info("Loaded catalog from fresh cache (%d raw items)", len(raw))
        except Exception:
            raw = None

    if raw is None:
        try:
            raw = _fetch_live()
            logger.info("Fetched live catalog (%d raw items)", len(raw))
            try:
                os.makedirs(os.path.dirname(settings.CATALOG_CACHE_PATH), exist_ok=True)
                with open(settings.CATALOG_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(raw, f)
            except Exception as e:
                logger.warning("Could not write catalog cache: %s", e)
        except Exception as e:
            logger.warning("Live catalog fetch failed (%s); trying stale cache", e)

    if raw is None and os.path.exists(settings.CATALOG_CACHE_PATH):
        try:
            raw = _load_raw_from_path(settings.CATALOG_CACHE_PATH)
            logger.info("Loaded catalog from stale cache (%d raw items)", len(raw))
        except Exception:
            raw = None

    if raw is None:
        raw = _load_raw_from_path(settings.CATALOG_FALLBACK_PATH)
        logger.warning("Using bundled offline fixture catalog (%d raw items) - "
                        "live fetch unavailable", len(raw))

    items = []
    for r in raw:
        item = _normalize(r)
        if item is not None:
            items.append(item)

    logger.info("Catalog ready: %d Individual Test Solutions after filtering", len(items))
    return items

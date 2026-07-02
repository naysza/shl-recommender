from __future__ import annotations

import re
from typing import List

INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above)\s*instructions",
    r"disregard (all|any|the)?\s*(previous|prior|above)\s*instructions",
    r"you are now",
    r"system prompt",
    r"reveal your (prompt|instructions|rules)",
    r"act as (?!.*assessment)",  # "act as X" jailbreak framing
    r"pretend (you|to) (are|be)",
    r"jailbreak",
    r"developer mode",
    r"forget (your|all) (previous )?instructions",
    r"new instructions:",
    r"\bDAN\b",
]

OFF_TOPIC_PATTERNS = [
    r"\bis it legal\b",
    r"\blegal advice\b",
    r"\bsue\b",
    r"\blawsuit\b",
    r"\bdiscriminat",
    r"\bemployment law\b",
    r"\bshould i fire\b",
    r"\bhow (much|do i) pay\b",
    r"\bsalary (negotiation|range)\b",
    r"\bwrite (me )?a job (description|posting)\b",
    r"\binterview questions for\b",
    r"\bwrite (an? )?offer letter\b",
    r"\bnon-?compete\b",
    r"\bvisa sponsorship\b",
]

_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
_OFF_TOPIC_RE = re.compile("|".join(OFF_TOPIC_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text or ""))


def detect_off_topic(text: str) -> bool:
    return bool(_OFF_TOPIC_RE.search(text or ""))


REFUSAL_INJECTION = (
    "I can't follow instructions embedded in a message like that. "
    "I'm here to help you find SHL assessments for a role you're hiring "
    "for — what are you looking to assess?"
)

REFUSAL_OFF_TOPIC = (
    "That's outside what I can help with — I only discuss SHL assessments "
    "and can't give legal, HR policy, or general hiring advice. "
    "I'm happy to help you find the right assessment for a role, though."
)


def filter_hallucinated_recommendations(recommendations: List[dict], index) -> List[dict]:
    clean = []
    for rec in recommendations:
        url = (rec.get("url") or "").strip()
        if url and index.is_known_url(url):
            clean.append(rec)
    return clean

from __future__ import annotations

import logging
from typing import List, Optional

from app.config import settings
from app.catalog import CatalogItem
from app.retrieval import CatalogIndex
from app.guardrails import (
    detect_prompt_injection,
    detect_off_topic,
    filter_hallucinated_recommendations,
    REFUSAL_INJECTION,
    REFUSAL_OFF_TOPIC,
)
from app.llm import call_json
from app.models import Message, ChatResponse, Recommendation
from app import prompts

logger = logging.getLogger("agent")

FALLBACK_CLARIFY = (
    "Happy to help you find the right SHL assessment. Could you tell me a bit more about the "
    "role - e.g. the job title, seniority level, and the main skills or competencies you want to assess?"
)
FALLBACK_ERROR = (
    "Sorry, I hit a snag processing that. Could you rephrase what role or skills you're hiring for?"
)


def _format_history(messages: List[Message]) -> str:
    lines = []
    for m in messages:
        speaker = "User" if m.role == "user" else "Assistant"
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)


def _last_user_message(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _extract_state(messages: List[Message]) -> Optional[dict]:
    history_text = _format_history(messages)
    prompt = prompts.build_state_prompt(history_text)
    return call_json(prompt, prompts.STATE_SYSTEM_INSTRUCTION, temperature=0.0)


def _duration_filter(requirements: dict) -> Optional[int]:
    val = requirements.get("max_duration_minutes")
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _run_recommend(requirements: dict, search_query: str, index: CatalogIndex, is_refine: bool) -> ChatResponse:
    job_level = (requirements.get("seniority") or "").strip() or None
    max_duration = _duration_filter(requirements)

    scored = index.search(
        search_query or requirements.get("role_or_skills", ""),
        top_k=settings.RETRIEVAL_TOP_K,
        job_level=job_level,
        max_duration_minutes=max_duration,
    )
    for comp in requirements.get("competencies") or []:
        scored += index.search(comp, top_k=10, job_level=job_level, max_duration_minutes=max_duration)

    seen = set()
    candidates = []
    for s in scored:
        if s.item.url in seen:
            continue
        seen.add(s.item.url)
        candidates.append(s.item.to_prompt_dict())
    candidates = candidates[: max(settings.RETRIEVAL_TOP_K, 15)]

    if not candidates:
        return ChatResponse(
            reply=(
                "I couldn't find any SHL assessments matching that in the catalog. "
                "Could you tell me more about the specific skills, role, or competencies you want to assess?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    prompt = prompts.build_recommend_prompt(requirements, candidates, is_refine)
    result = call_json(prompt, prompts.RECOMMEND_SYSTEM_INSTRUCTION, temperature=0.2)

    if not result or "reply" not in result:
        top = candidates[: settings.MAX_RECOMMENDATIONS]
        return ChatResponse(
            reply="Here are assessments from the SHL catalog that match what you've described so far.",
            recommendations=[Recommendation(**c) for c in top],
            end_of_conversation=False,
        )

    recs = filter_hallucinated_recommendations(result.get("recommendations", []), index)
    recs = recs[: settings.MAX_RECOMMENDATIONS]
    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(name=r["name"], url=r["url"], test_type=r.get("test_type", "")) for r in recs],
        end_of_conversation=False,
    )


def _run_compare(compare_targets: List[str], index: CatalogIndex) -> ChatResponse:
    found_items = []
    not_found = []
    for name in compare_targets:
        item = index.find_by_name(name)
        if item:
            found_items.append(item.to_prompt_dict())
        else:
            not_found.append(name)

    if not found_items:
        names = ", ".join(compare_targets) if compare_targets else "those assessments"
        return ChatResponse(
            reply=f"I couldn't find {names} in the SHL catalog I have access to. Could you double-check the name(s)?",
            recommendations=[],
            end_of_conversation=False,
        )

    prompt = prompts.build_compare_prompt(compare_targets, found_items, not_found)
    result = call_json(prompt, prompts.COMPARE_SYSTEM_INSTRUCTION, temperature=0.1)

    if not result or "reply" not in result:
        fallback_recs = [
            Recommendation(name=i["name"], url=i["url"], test_type=i["test_type"])
            for i in found_items
        ]
        return ChatResponse(
            reply="Here's what the catalog says about those assessments: " +
                  "; ".join(f"{i['name']}: {i['description']}" for i in found_items),
            recommendations=fallback_recs,
            end_of_conversation=False,
        )

    recs = filter_hallucinated_recommendations(result.get("recommendations", []), index)
    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(name=r["name"], url=r["url"], test_type=r.get("test_type", "")) for r in recs],
        end_of_conversation=False,
    )


def handle_chat(messages: List[Message], index: CatalogIndex) -> ChatResponse:
    if not messages:
        return ChatResponse(reply=FALLBACK_CLARIFY, recommendations=[], end_of_conversation=False)

    last_user = _last_user_message(messages)
    turn_count = len(messages)
    at_turn_cap = turn_count >= settings.MAX_TURNS

    if detect_prompt_injection(last_user):
        return ChatResponse(reply=REFUSAL_INJECTION, recommendations=[], end_of_conversation=False)
    if detect_off_topic(last_user):
        return ChatResponse(reply=REFUSAL_OFF_TOPIC, recommendations=[], end_of_conversation=False)

    state = _extract_state(messages)
    if not state:
        return ChatResponse(reply=FALLBACK_ERROR, recommendations=[], end_of_conversation=False)

    action = state.get("action", "clarify")
    requirements = state.get("requirements", {}) or {}
    search_query = state.get("search_query") or requirements.get("role_or_skills", "") or last_user

    if at_turn_cap and action == "clarify":
        action = "recommend"

    if action == "refuse_injection":
        return ChatResponse(reply=REFUSAL_INJECTION, recommendations=[], end_of_conversation=False)
    if action == "refuse_scope":
        return ChatResponse(reply=REFUSAL_OFF_TOPIC, recommendations=[], end_of_conversation=False)

    if action == "clarify":
        question = state.get("clarifying_question") or FALLBACK_CLARIFY
        return ChatResponse(reply=question, recommendations=[], end_of_conversation=False)

    conversation_complete = bool(state.get("conversation_complete", False))

    if action == "compare":
        targets = state.get("compare_targets") or []
        response = _run_compare(targets, index)
        response.end_of_conversation = conversation_complete or at_turn_cap
        return response

    if action in ("recommend", "refine"):
        response = _run_recommend(requirements, search_query, index, is_refine=(action == "refine"))
        response.end_of_conversation = conversation_complete or at_turn_cap
        return response

    return ChatResponse(reply=FALLBACK_CLARIFY, recommendations=[], end_of_conversation=False)

#!/usr/bin/env python3
"""
Local evaluation harness, mirroring how SHL's evaluator will grade the
service: an LLM plays the persona/user, has a real multi-turn conversation
against your running POST /chat, and we score Recall@10 plus a few hard
behavioral checks.

Usage:
    python scripts/replay_trace.py --traces-dir ./sample_conversations --api-url http://localhost:8000

Trace file format: this expects the traces you downloaded from
https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/sample_conversations.zip
to be JSON files, one per trace, each roughly:
    {
      "persona": "...",
      "facts": {...} or "facts": "free text",
      "expected_shortlist": ["Assessment Name 1", "Assessment Name 2", ...]
    }

The exact key names in the real zip may differ - run with --dry-run first
on one file to print what got parsed, and adjust load_trace() below if
needed. This script is intentionally small and readable so that's a quick
edit, not a rewrite.
"""
import argparse
import json
import os
import sys
from typing import Optional

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.config import settings  # noqa: E402

USER_SIM_SYSTEM = """You are role-playing as a hiring manager described below, talking to an SHL \
assessment recommendation chatbot. Answer the assistant's questions truthfully using ONLY the facts \
given. If asked something not covered by your facts, say you have no preference. Keep replies short \
and natural, like a real person typing in a chat. If the assistant gives you a shortlist of \
assessments, respond with a brief acknowledgement (e.g. "Great, thanks!") and nothing else - do not \
ask for more.

Persona and facts:
{facts}

Reply with ONLY your next chat message, no quotes, no explanation."""


def load_trace(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    persona = data.get("persona") or data.get("description") or ""
    facts = data.get("facts") or data.get("fact_set") or data.get("context") or {}
    expected = (
        data.get("expected_shortlist")
        or data.get("expected")
        or data.get("gold")
        or data.get("relevant_assessments")
        or []
    )
    return {"persona": persona, "facts": facts, "expected": expected, "raw": data}


def simulate_user_reply(trace: dict, transcript: list, gemini_client) -> str:
    from google.genai import types

    facts_blob = json.dumps({"persona": trace["persona"], "facts": trace["facts"]}, indent=2)
    history = "\n".join(f"{m['role']}: {m['content']}" for m in transcript)
    prompt = f"Conversation so far:\n{history}\n\nWhat do you say next?"
    resp = gemini_client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=USER_SIM_SYSTEM.format(facts=facts_blob),
            temperature=0.3,
            max_output_tokens=200,
        ),
    )
    return (resp.text or "").strip()


def recall_at_k(expected: list, recommended: list, k: int = 10) -> Optional[float]:
    if not expected:
        return None
    expected_norm = {e.strip().lower() for e in expected}
    got_norm = {r.get("name", "").strip().lower() for r in recommended[:k]}
    hit = sum(1 for e in expected_norm if any(e in g or g in e for g in got_norm))
    return hit / len(expected_norm)


def run_trace(trace_path: str, api_url: str, gemini_client, max_turns: int, verbose: bool) -> dict:
    trace = load_trace(trace_path)
    transcript = []
    final_response = None
    schema_ok = True
    hallucinated = False
    known_urls_cache = None

    with httpx.Client(timeout=35) as client:
        for turn in range(max_turns):
            if transcript and transcript[-1]["role"] == "user":
                # Waiting on assistant - shouldn't happen, but guard anyway.
                pass

            if not transcript:
                user_msg = "Hi, I need help finding an SHL assessment."
            else:
                user_msg = simulate_user_reply(trace, transcript, gemini_client)

            transcript.append({"role": "user", "content": user_msg})

            resp = client.post(f"{api_url}/chat", json={"messages": transcript})
            if resp.status_code != 200:
                schema_ok = False
                if verbose:
                    print(f"  [turn {turn}] HTTP {resp.status_code}: {resp.text[:300]}")
                break
            body = resp.json()
            required = {"reply", "recommendations", "end_of_conversation"}
            if not required.issubset(body.keys()):
                schema_ok = False

            recs = body.get("recommendations", [])
            if not isinstance(recs, list) or not (0 <= len(recs) <= 10):
                schema_ok = False

            transcript.append({"role": "assistant", "content": body.get("reply", "")})
            final_response = body

            if verbose:
                print(f"  [turn {turn}] user: {user_msg}")
                print(f"  [turn {turn}] assistant: {body.get('reply', '')[:200]}")
                if recs:
                    print(f"  [turn {turn}] recommendations: {[r.get('name') for r in recs]}")

            if body.get("end_of_conversation") or recs:
                break

    recall = None
    if final_response:
        recall = recall_at_k(trace["expected"], final_response.get("recommendations", []))

    return {
        "trace": os.path.basename(trace_path),
        "turns_used": len(transcript) // 2,
        "schema_ok": schema_ok,
        "recall_at_10": recall,
        "final_recommendations": [r.get("name") for r in (final_response or {}).get("recommendations", [])],
        "expected": trace["expected"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", required=True)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Just parse and print one trace, no API calls")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    trace_files = sorted(
        os.path.join(args.traces_dir, f)
        for f in os.listdir(args.traces_dir)
        if f.endswith(".json")
    )
    if not trace_files:
        print(f"No .json trace files found in {args.traces_dir}")
        sys.exit(1)

    if args.dry_run:
        print(json.dumps(load_trace(trace_files[0]), indent=2)[:2000])
        return

    from google import genai

    if not settings.GEMINI_API_KEY:
        print("GEMINI_API_KEY not set - needed to simulate the user side of the conversation.")
        sys.exit(1)
    gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

    results = []
    for path in trace_files:
        print(f"Running {os.path.basename(path)} ...")
        result = run_trace(path, args.api_url, gemini_client, args.max_turns, args.verbose)
        results.append(result)
        print(f"  -> schema_ok={result['schema_ok']} recall@10={result['recall_at_10']} "
              f"turns={result['turns_used']}")

    recalls = [r["recall_at_10"] for r in results if r["recall_at_10"] is not None]
    mean_recall = sum(recalls) / len(recalls) if recalls else None
    schema_pass_rate = sum(1 for r in results if r["schema_ok"]) / len(results)

    print("\n=== Summary ===")
    print(f"Traces run: {len(results)}")
    print(f"Schema pass rate: {schema_pass_rate:.0%}")
    print(f"Mean Recall@10: {mean_recall}")


if __name__ == "__main__":
    main()

"""
Thin wrapper around the Gemini API. Everything the agent needs from the LLM
goes through call_json(), which forces JSON-mode output and retries on
transient failures. Keeping this in one place means swapping providers
(Groq/OpenAI/OpenRouter) later only touches this file.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

logger = logging.getLogger("llm")

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai  # imported lazily so app can boot without the key for /health

        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set it as an environment variable before calling /chat."
            )
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


class LLMError(Exception):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(Exception),
)
def _generate(prompt: str, system_instruction: str, temperature: float) -> str:
    from google.genai import types

    client = _get_client()
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
            max_output_tokens=2048,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise LLMError("Empty response from model")
    return text


def call_json(prompt: str, system_instruction: str, temperature: float = 0.2) -> Optional[Any]:
    """Calls the model in JSON mode and parses the result. Returns None
    (rather than raising) on unrecoverable failure so callers can degrade
    gracefully instead of 500-ing the whole /chat request."""
    try:
        raw = _generate(prompt, system_instruction, temperature)
    except Exception as e:
        logger.error("LLM call failed after retries: %s", e)
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Occasionally the model wraps JSON in a code fence despite JSON
        # mode; strip and retry the parse once before giving up.
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Could not parse model output as JSON: %s", raw[:500])
            return None

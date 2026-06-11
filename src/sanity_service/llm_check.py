"""LLM-based coherence gate - sends all prompt/response pairs in one batch to OpenRouter."""
from __future__ import annotations

import json
import re

import httpx
from loguru import logger

from sanity_service import config as _cfg

_OR_MODEL = _cfg.OR_MODEL
_OR_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
_OR_TIMEOUT  = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)

_SYSTEM = (
    "You review AI responses to coding questions. "
    "For each numbered pair, answer YES if the response is relevant and coherent, "
    "NO if it is nonsense, gibberish, or completely off-topic. "
    "Reply with a JSON array only, e.g. [\"YES\", \"NO\", \"YES\"]. No other text."
)


def _build_user_message(prompts: list[str], responses: list[str]) -> str:
    # Formats all pairs into a single message for batch judgement.
    parts = []
    for i, (p, r) in enumerate(zip(prompts, responses), 1):
        parts.append(f"{i}. Question: {p.strip()}\n   Response: {r.strip()[:300]}")
    return "\n\n".join(parts)


def _parse_verdicts(raw: str, n: int) -> list[bool] | None:
    # Extracts a list of YES/NO booleans from the model response; returns None if unparseable.
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        items = json.loads(match.group())
        if len(items) != n:
            return None
        return [str(v).strip().upper().startswith("Y") for v in items]
    except Exception:
        return None


async def llm_check(
    prompts: list[str],
    responses: list[str],
    api_key: str,
    min_pass: int | None = None,
) -> tuple[bool, str, str]:
    # Batches pairs to OR; returns (passed, reason, status) where status is passed|failed|skipped.
    # Fails open (skipped) if OR breaks; min_pass defaults to at-least-half of the prompt count.
    if min_pass is None:
        min_pass = (len(prompts) + 1) // 2
    user_msg = _build_user_message(prompts, responses)
    try:
        async with httpx.AsyncClient(timeout=_OR_TIMEOUT) as c:
            r = await c.post(
                _OR_BASE_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model":       _OR_MODEL,
                    "messages":    [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    "max_tokens":  64,
                    "temperature": 0.0,
                },
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"] or ""
    except Exception as exc:
        # Fail open - OR unreachable should not block a valid challenger.
        logger.warning("[sanity/llm] OR call failed: {} - skipping LLM gate", exc)
        return True, "", "skipped"

    verdicts = _parse_verdicts(raw, len(prompts))
    if verdicts is None:
        logger.warning("[sanity/llm] unparseable response: {!r} - skipping LLM gate", raw[:120])
        return True, "", "skipped"

    passed_count = sum(verdicts)
    logger.info("[sanity/llm] verdicts={} ({}/{} coherent)", verdicts, passed_count, len(verdicts))

    if passed_count < min_pass:
        failed = [i + 1 for i, v in enumerate(verdicts) if not v]
        reason = (f"LLM judge: {passed_count}/{len(verdicts)} responses coherent "
                  f"(failed prompts: {failed})")
        return False, reason, "failed"

    return True, "", "passed"

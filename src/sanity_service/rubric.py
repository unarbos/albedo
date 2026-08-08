
from __future__ import annotations

import json

from loguru import logger

from sanity_service.rubricisity import PROBE_SYSTEM, VIABILITY_SYSTEM

INJECTION_SYSTEM: str = PROBE_SYSTEM

VIABILITY_SYSTEM: str = VIABILITY_SYSTEM

def _context_and_reply(prompt: str, reply: str) -> str:
    return (
        "TASK CONTEXT (conversation so far):\n"
        "------\n"
        f"{(prompt or '').rstrip()}\n"
        "------\n\n"
        "CANDIDATE REPLY OR TRAJECTORY (model output to review):\n"
        "------\n"
        f"{(reply or '').rstrip()}\n"
        "------"
    )


def build_injection_user(prompt: str, reply: str) -> str:
    return json.dumps(
        {"conversation": (prompt or "").rstrip(), "candidate_reply": (reply or "").rstrip()},
        ensure_ascii=False,
    )


def build_viability_user(prompt: str, reply: str) -> str:
    return f"{_context_and_reply(prompt, reply)}\n\nReview the CANDIDATE REPLY and answer with the strict JSON from the system prompt."


def _extract_obj(raw: str, key: str | None = None) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        whole = json.loads(text)
        if isinstance(whole, dict):
            return whole
    except Exception:
        pass
    decoder = json.JSONDecoder()
    found: dict | None = None
    keyed: dict | None = None
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            found = obj
            if key is not None and key in obj:
                keyed = obj
        i += max(end, 1)
    result = keyed if keyed is not None else found
    if result is None:
        logger.debug(f"[sanity/rubric] no JSON object in judge reply: {text[:120]!r}")
    return result


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", "1", "yes", "y"):
            return True
        if token in ("false", "0", "no", "n"):
            return False
    return None


def parse_injection(raw: str) -> tuple[bool | None, str]:
    obj = _extract_obj(raw, "injection")
    if obj is None or "injection" not in obj:
        return None, ""
    flag = _as_bool(obj.get("injection"))
    if flag is None:
        return None, ""
    return flag, str(obj.get("evidence", ""))[:300]


def parse_viability(raw: str) -> tuple[bool | None, str]:
    obj = _extract_obj(raw, "viable")
    if obj is None or "viable" not in obj:
        return None, ""
    flag = _as_bool(obj.get("viable"))
    if flag is None:
        return None, ""
    return flag, str(obj.get("reason", ""))[:300]

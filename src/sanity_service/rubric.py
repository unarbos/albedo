"""Single-model judge prompts and parsers for the sanity pre-eval gate - injection + viability."""

from __future__ import annotations

import json

from loguru import logger

from sanity_service.rubricisity import PROBE_SYSTEM, VIABILITY_SYSTEM

# ── Injection auditor ─────────────────────────────────────────────────────────
# Source of truth for the probe system prompt is rubricisity.PROBE_SYSTEM; it expects the user
# message as a JSON object with "conversation" and "candidate_reply" fields (see build_injection_user).

INJECTION_SYSTEM: str = PROBE_SYSTEM

# ── Viability reviewer (single-model coherence / on-task gate) ────────────────

VIABILITY_SYSTEM: str = VIABILITY_SYSTEM


# ── User-message assembly ─────────────────────────────────────────────────────

def _context_and_reply(prompt: str, reply: str) -> str:
    # Renders the task context and the model's candidate reply inside delimiters.
    return (
        "TASK CONTEXT (conversation so far):\n"
        "------\n"
        f"{(prompt or '').rstrip()}\n"
        "------\n\n"
        "CANDIDATE REPLY (the model's proposed next turn):\n"
        "------\n"
        f"{(reply or '').rstrip()}\n"
        "------"
    )


def build_injection_user(prompt: str, reply: str) -> str:
    # JSON payload for the PROBE_SYSTEM auditor: untrusted data under "conversation" + "candidate_reply".
    return json.dumps(
        {"conversation": (prompt or "").rstrip(), "candidate_reply": (reply or "").rstrip()},
        ensure_ascii=False,
    )


def build_viability_user(prompt: str, reply: str) -> str:
    # User message for the viability reviewer: context + reply + instruction.
    return f"{_context_and_reply(prompt, reply)}\n\nReview the CANDIDATE REPLY and answer with the strict JSON from the system prompt."


# ── Parsers ───────────────────────────────────────────────────────────────────

def _extract_obj(raw: str, key: str | None = None) -> dict | None:
    # Returns the judge's JSON verdict; when key is given, prefers the object that CONTAINS it
    # (so a trailing non-verdict object cannot shadow a real verdict).
    text = (raw or "").strip()
    if not text:
        return None
    try:
        whole = json.loads(text)
        if isinstance(whole, dict):
            return whole
    except Exception:  # noqa: BLE001 - not pure JSON; scan for an embedded object below
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
            obj, end = decoder.raw_decode(text[i:])  # handles braces inside JSON strings
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
    # Coerces a judge flag to True/False; None when it is not a recognizable boolean (so the vote ignores it).
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
    # Returns (injection_flag, evidence); flag is None when unparseable or not a clean boolean.
    obj = _extract_obj(raw, "injection")
    if obj is None or "injection" not in obj:
        return None, ""
    flag = _as_bool(obj.get("injection"))
    if flag is None:
        return None, ""
    return flag, str(obj.get("evidence", ""))[:300]


def parse_viability(raw: str) -> tuple[bool | None, str]:
    # Returns (viable_flag, reason); flag is None when unparseable or not a clean boolean.
    obj = _extract_obj(raw, "viable")
    if obj is None or "viable" not in obj:
        return None, ""
    flag = _as_bool(obj.get("viable"))
    if flag is None:
        return None, ""
    return flag, str(obj.get("reason", ""))[:300]

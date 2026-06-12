"""Single-model judge prompts and parsers for the sanity pre-eval gate - injection + viability."""
from __future__ import annotations

import json
import re

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

_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_obj(raw: str) -> dict | None:
    # Returns the JSON object from a judge reply (whole string, else first {...}); None if absent.
    text = (raw or "").strip()
    for candidate in (text, (_OBJ_RE.search(text).group() if _OBJ_RE.search(text) else "")):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:  # noqa: BLE001 - any malformed candidate just falls through to None
            continue
    return None


def parse_injection(raw: str) -> tuple[bool | None, str]:
    # Returns (injection_flag, evidence); flag is None when the reply is unparseable.
    obj = _extract_obj(raw)
    if obj is None or "injection" not in obj:
        return None, ""
    return bool(obj.get("injection")), str(obj.get("evidence", ""))[:300]


def parse_viability(raw: str) -> tuple[bool | None, str]:
    # Returns (viable_flag, reason); flag is None when the reply is unparseable.
    obj = _extract_obj(raw)
    if obj is None or "viable" not in obj:
        return None, ""
    return bool(obj.get("viable")), str(obj.get("reason", ""))[:300]

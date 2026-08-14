from __future__ import annotations

import re
from typing import Any

from ..shared.questions import question_floor
from .prompt_behavior import BEHAVIOR_COMMON, BEHAVIOR_PHASES, NO_TESTS_NOTE


def build_behavior_messages(
    *,
    phase: str,
    index: int,
    task: str,
    prefix_tail: str,
    k: int,
    tests_seen: bool,
) -> list[dict[str, str]]:
    behavior = BEHAVIOR_PHASES[phase]
    part_text = behavior.parts[index].text
    if phase == "at_edit" and index == 0 and not tests_seen:
        part_text += NO_TESTS_NOTE
    gold = "\n".join(f"  GOOD: {t}" for t in behavior.gold)
    system = (
        f"You write EXACTLY {k} yes/no checklist questions testing ONE behaviour of a "
        f"coding-agent trajectory.\n\n{behavior.head}\n\nTHE BEHAVIOUR:\n{part_text}\n\n"
        f"CANONICAL FORM — match this style:\n{gold}\n\nRULES:\n"
        f"- INSTANTIATE: at least 3 of the questions must name a concrete file, symbol, or code "
        f"fragment copied verbatim from the sample context below (never invented, never guessed "
        f"from conventions); questions with no such target keep the generic form.\n"
        f"- One property, one question: two questions a trajectory could only pass or fail "
        f"together are duplicates — replace one with a different property of this behaviour.\n"
        f"- Questions address the trajectory only; never reference these instructions.\n\n"
        + BEHAVIOR_COMMON
    )
    user = (
        f"SAMPLE CONTEXT:\n{(task or '')[:4000]}\n\nLAST CONTEXT TURNS:\n{prefix_tail}"
        f"\n\nWrite the {k} questions."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_BEHAVIOR_BANNED_RE = re.compile(
    r"reproduc|\brepro\b|failure[^?]{0,40}before[^?]{0,20}(edit|fix|chang)"
    r"|(run|execut)[^?]{0,25}after (each|every) edit",
    re.IGNORECASE,
)
_BEHAVIOR_REPEAT_FORM_RE = re.compile(r"avoid|re-?issu|retr|repeat|already received", re.IGNORECASE)
_BEHAVIOR_FILE_RE = re.compile(r"[\w/-]+\.(?:py|rs|go|js|ts|c|h|java|php|rb)\b")


def filter_behavior_questions(
    questions: list[dict[str, str]], context: str, *, discards: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    """Deterministic backstop for prompt drift: drop banned-theme questions (repetition-form
    mentions of a repro command are fine) and questions naming files absent from the context."""
    kept = []
    for question in questions:
        text = question.get("text", "")
        if _BEHAVIOR_BANNED_RE.search(text) and not _BEHAVIOR_REPEAT_FORM_RE.search(text):
            if discards is not None:
                discards.append(
                    {
                        "stage": "filter_behavior_questions",
                        "reason": "behavior_banned_theme",
                        "text": text,
                        "origin": "behavior",
                    }
                )
            continue
        if any(f not in context for f in _BEHAVIOR_FILE_RE.findall(text)):
            if discards is not None:
                discards.append(
                    {
                        "stage": "filter_behavior_questions",
                        "reason": "behavior_file_not_in_context",
                        "text": text,
                        "origin": "behavior",
                    }
                )
            continue
        kept.append(question)
    return kept


def behavior_question_schema(n: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": question_floor(n),
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "example_bad": {"type": "string"},
                        "requires": {
                            "type": "string",
                            "enum": ["action", "read", "neutral"],
                        },
                    },
                    "required": ["text", "example_bad", "requires"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    }

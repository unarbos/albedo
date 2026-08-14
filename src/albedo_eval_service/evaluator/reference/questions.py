from __future__ import annotations

import math
import re
from typing import Any

from ..shared.budgets import (
    RUBRIC_ECONOMY_CAP,
    RUBRIC_LENGTH_BOUNDS,
    RUBRIC_MAX_QUESTIONS,
    RUBRIC_MIN_QUESTIONS,
    RUBRIC_NEGATIVE_CAP,
    RUBRIC_REFERENCE_TARGET,
)
from ..shared.questions import _FENCE_RE, VALID_TAGS, is_measurement_bound_question
from .prompt_reference import (
    REFERENCE_QUESTION_SYSTEM,
    REFERENCE_QUESTION_USER,
    REFERENCE_SCORED_WINDOW_BLOCK,
)

_WORKFLOW_HEAD_RE = re.compile(
    r"(## Recommended Workflow|<PROBLEM_SOLVING_WORKFLOW>|Follow these steps to resolve the issue:"
    r"|Phase 1\. READING)",
    re.IGNORECASE,
)

_OBSERVATION_SUCCESS_MARKERS = {
    "returncode": "<returncode>N</returncode> around <output>; failure = non-zero returncode or "
    "error text in the output",
    "swe_agent": "plain OBSERVATION text; failure is only visible as error text in it",
    "openhands": "[Command finished with exit code N] trailers; failure = non-zero exit code or "
    "error text",
}


def _workflow_text(task: str) -> str:
    m = _WORKFLOW_HEAD_RE.search(task or "")
    if not m:
        return "The task declares no numbered workflow."
    tail = task[m.start() :]
    stop = re.search(r"\n## (?!Recommended)|</PROBLEM_SOLVING_WORKFLOW>|\n<(?!/)[A-Z_]+>", tail)
    return tail[: stop.end() if stop else 1500][:1500]


def _reference_measurements(reference: str) -> str:
    steps = re.split(r"^REFERENCE STEP \d+:$", reference, flags=re.M)[1:]
    bodies = [s.split("\nENVIRONMENT OBSERVATION:\n")[0].strip() for s in steps] or [""]
    words = [len(b.split()) for b in bodies]
    chars = [len(b) for b in bodies]
    prose_words = [len(_FENCE_RE.sub("", b).split()) for b in bodies]
    return (
        "REFERENCE MEASUREMENTS (programmatic):\n"
        f"- total REFERENCE STEP words: {sum(words)}\n"
        f"- longest single REFERENCE STEP: {max(words)} words\n"
        f"- total REFERENCE STEP characters: {sum(chars)}\n"
        f"- longest single REFERENCE STEP: {max(chars)} characters\n"
        f"- total REFERENCE STEP prose words (outside fenced code): {sum(prose_words)}\n"
        f"- average REFERENCE STEP words: {round(sum(words) / len(bodies))}\n"
        f"- average REFERENCE STEP characters: {round(sum(chars) / len(bodies))}\n"
        f"- REFERENCE STEP count: {len(words)}"
    )


_ECONOMY_DUPLICATE_MULTIPLIER = 2.0


def _reference_raw_measurements(reference: str) -> dict[str, int]:
    """Same computation as _reference_measurements(), as raw numbers instead of formatted text -
    used by duplicate_economy_bounds() to recompute a bound at a different multiplier without
    reparsing the LLM's own generated question text."""
    steps = re.split(r"^REFERENCE STEP \d+:$", reference, flags=re.M)[1:]
    bodies = [s.split("\nENVIRONMENT OBSERVATION:\n")[0].strip() for s in steps] or [""]
    words = [len(b.split()) for b in bodies]
    chars = [len(b) for b in bodies]
    prose_words = [len(_FENCE_RE.sub("", b).split()) for b in bodies]
    return {
        "total_words": sum(words),
        "max_words": max(words),
        "total_chars": sum(chars),
        "max_chars": max(chars),
        "total_prose_words": sum(prose_words),
        "avg_words": round(sum(words) / len(bodies)),
        "avg_chars": round(sum(chars) / len(bodies)),
    }


_ECONOMY_TEMPLATE_METRICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "total_words",
        re.compile(r"^Is total CANDIDATE OUTPUT at most (\d[\d,]*) words\?$", re.IGNORECASE),
    ),
    (
        "max_words",
        re.compile(r"^Is the longest single output at most (\d[\d,]*) words\?$", re.IGNORECASE),
    ),
    (
        "total_chars",
        re.compile(r"^Is total CANDIDATE OUTPUT at most (\d[\d,]*) characters\?$", re.IGNORECASE),
    ),
    (
        "max_chars",
        re.compile(
            r"^Is the longest single output at most (\d[\d,]*) characters\?$", re.IGNORECASE
        ),
    ),
    (
        "total_prose_words",
        re.compile(
            r"^Is CANDIDATE OUTPUT prose, apart from code blocks, at most (\d[\d,]*) words\?$",
            re.IGNORECASE,
        ),
    ),
    (
        "avg_words",
        re.compile(
            r"^Is the average CANDIDATE OUTPUT per turn at most (\d[\d,]*) words\?$", re.IGNORECASE
        ),
    ),
    (
        "avg_chars",
        re.compile(
            r"^Is the average CANDIDATE OUTPUT per turn at most (\d[\d,]*) characters\?$",
            re.IGNORECASE,
        ),
    ),
)


def duplicate_economy_bounds(
    questions: list[dict[str, str]],
    reference: str,
    *,
    multiplier: float = _ECONOMY_DUPLICATE_MULTIPLIER,
) -> list[dict[str, str]]:
    raw = _reference_raw_measurements(reference)
    duplicates: list[dict[str, str]] = []
    for question in questions:
        text = (question.get("text") or "").strip()
        if not is_measurement_bound_question(text):
            continue
        for metric, pattern in _ECONOMY_TEMPLATE_METRICS:
            match = pattern.match(text)
            if not match:
                continue
            new_bound = math.ceil(raw[metric] * multiplier / 10) * 10
            start, end = match.span(1)
            duplicate = dict(question)
            duplicate["text"] = f"{text[:start]}{new_bound}{text[end:]}"
            duplicates.append(duplicate)
            break
    return duplicates


def reference_question_schema() -> dict[str, Any]:
    step = {
        "type": "object",
        "properties": {
            "step": {"type": "integer"},
            "text": {"type": "string"},
            "already_done_in_conversation": {"type": "boolean"},
            "demonstrated_by_reference": {"type": "boolean"},
            "verdict": {"type": "string", "enum": ["demonstrated", "not_demonstrated"]},
        },
        "required": [
            "step",
            "text",
            "already_done_in_conversation",
            "demonstrated_by_reference",
            "verdict",
        ],
        "additionalProperties": False,
    }
    question = {
        "type": "object",
        "properties": {
            "step": {"type": "integer"},
            "evidence": {"type": "string"},
            "text": {"type": "string"},
            "example_bad": {"type": "string"},
            "tag": {"type": "string", "enum": list(VALID_TAGS)},
        },
        "required": ["step", "evidence", "text", "example_bad", "tag"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "ledger": {
                "type": "object",
                "properties": {
                    "steps": {"type": "array", "items": step},
                    "frontier_step": {"type": "integer"},
                    "reference_finished": {"type": "boolean"},
                    "focus": {"type": "string"},
                },
                "required": ["steps", "frontier_step", "reference_finished", "focus"],
                "additionalProperties": False,
            },
            "questions": {
                # the prompt itself allows 8-14 when the reference only diagnosed
                "type": "array",
                "minItems": 8,
                "maxItems": RUBRIC_MAX_QUESTIONS,
                "items": question,
            },
        },
        "required": ["ledger", "questions"],
        "additionalProperties": False,
    }


def build_reference_question_messages(
    *,
    task: str,
    reference: str,
    fmt: str,
    prefix_turns: int,
    candidate_turns: int,
) -> list[dict[str, str]]:
    system = REFERENCE_QUESTION_SYSTEM.format(
        target=RUBRIC_REFERENCE_TARGET,
        min_n=RUBRIC_MIN_QUESTIONS,
        max_n=RUBRIC_MAX_QUESTIONS,
        negative_cap=RUBRIC_NEGATIVE_CAP,
        economy_cap=RUBRIC_ECONOMY_CAP,
        bound_n=RUBRIC_LENGTH_BOUNDS,
    )
    window = REFERENCE_SCORED_WINDOW_BLOCK.format(
        workflow_text=_workflow_text(task),
        prefix_turns=prefix_turns,
        candidate_turns=candidate_turns,
        observation_format=fmt,
        success_marker=_OBSERVATION_SUCCESS_MARKERS.get(
            fmt, _OBSERVATION_SUCCESS_MARKERS["returncode"]
        ),
    )
    user = REFERENCE_QUESTION_USER.format(
        task=task.rstrip(),
        reference=reference.rstrip(),
        min_n=RUBRIC_MIN_QUESTIONS,
        max_n=RUBRIC_MAX_QUESTIONS,
        reference_measurements=_reference_measurements(reference),
        scored_window=window,
        bound_n=RUBRIC_LENGTH_BOUNDS,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_reference_trajectory(turns: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    step = 0
    for turn in turns:
        if turn.get("score_target"):
            step += 1
            parts.append(f"REFERENCE STEP {step}:\n{turn['content']}")
        elif turn.get("environment_observation"):
            parts.append(f"ENVIRONMENT OBSERVATION:\n{turn['content']}")
    return "\n\n".join(parts)


def filter_reference_leaks(
    questions: list[dict[str, str]], *, discards: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    if discards is None:
        return [q for q in questions if "the reference" not in q["text"].casefold()]
    kept = []
    for q in questions:
        if "the reference" in q["text"].casefold():
            discards.append(
                {
                    "stage": "filter_reference_leaks",
                    "reason": "reference_leak",
                    "text": q["text"],
                    "origin": "content",
                }
            )
        else:
            kept.append(q)
    return kept

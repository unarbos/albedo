from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any

from .evaluator.shared.questions import _CANDIDATE_BLOCK_RE, measurements_block
from .judge.prompt_judge import JUDGE_SYSTEM, JUDGE_USER
from .shared.json_extract import extract_json
from .shared.observation_format import (
    THINK_CLOSE_RE,
    THINK_OPEN_RE,
    THINK_PAIR_RE,
    THINK_TAG_RE,
    mask_fenced_spans,
    unmask_fenced_spans,
)

CHALLENGER_WIN_MARGIN = 0.025

NO_VISIBLE_OUTPUT = "[no visible output]"
_COMMAND_FENCE_RE = re.compile(r"```(?:bash|sh|shell)[ \t]*\n", re.IGNORECASE)


def _drop_unclosed_reasoning(text: str, fences: list[str]) -> str:
    match = THINK_OPEN_RE.search(text)
    if not match:
        return text
    tail = text[match.end() :]
    if _COMMAND_FENCE_RE.search(unmask_fenced_spans(tail, fences)):
        return text[: match.start()] + tail
    return text[: match.start()]


def strip_leaked_reasoning(text: str) -> str:
    """Drop model reasoning that leaked into a scored candidate turn.

    vLLM's qwen3 reasoning parser leaves an orphaned closing tag (and occasionally the reasoning
    itself) in the generated text. A `THOUGHT:` before the tag is the mini-coder-rs corpus's own
    style and is real content, so only the stray tag is removed in that case.
    """
    masked, fences = mask_fenced_spans(text or "")
    cleaned = THINK_PAIR_RE.sub("", masked)
    cleaned = _drop_unclosed_reasoning(cleaned, fences)
    if THINK_CLOSE_RE.search(cleaned):
        head, tail = THINK_CLOSE_RE.split(cleaned, 1)
        cleaned = head + tail if "THOUGHT:" in head else tail
    cleaned = THINK_TAG_RE.sub("", cleaned)
    return unmask_fenced_spans(cleaned, fences).strip()


def strip_candidate_reasoning(trajectory: str) -> str:
    """Apply strip_leaked_reasoning inside CANDIDATE OUTPUT blocks only.

    Context turns and environment observations are left byte-for-byte intact: the mini-coder-rs
    corpus legitimately carries `</think>` in its own assistant turns.
    """

    def _rewrite(match: re.Match[str]) -> str:
        block, inner = match.group(0), match.group(1)
        start, end = match.start(1) - match.start(0), match.end(1) - match.start(0)
        stripped = strip_leaked_reasoning(inner)
        if not stripped:
            stripped = NO_VISIBLE_OUTPUT if inner.strip() else inner
        return block[:start] + stripped + block[end:]

    return _CANDIDATE_BLOCK_RE.sub(_rewrite, trajectory or "")


def build_judge_messages(*, response: str, questions: list[dict[str, str]]) -> list[dict[str, str]]:
    shown = [
        {
            "id": q["id"],
            "tag": q.get("tag", ""),
            "text": q["text"],
            "example_bad": q.get("example_bad", ""),
        }
        for q in questions
    ]
    cleaned = strip_candidate_reasoning(strip_reply_injection(response)).rstrip()
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER.format(
                response=cleaned,
                measurements=measurements_block(cleaned),
                questions_json=json.dumps(shown, ensure_ascii=False, indent=1),
            ),
        },
    ]


def answer_schema(question_ids: list[str]) -> dict[str, Any]:
    count = len(question_ids)
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": question_ids},
                        "explanation": {"type": "string"},
                        "answer": {"type": "integer", "enum": [1, 0]},
                    },
                    "required": ["id", "explanation", "answer"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }


_ANSWER_TO_BIT: dict[str, float] = {"1": 1.0, "0": 0.0}


def parse_answers(
    raw: str, question_ids: list[str]
) -> tuple[dict[str, str | None], dict[str, str], bool]:
    obj = extract_json(raw, prefer_keys=("answers",))
    items = obj.get("answers") if isinstance(obj, dict) else obj
    answers: dict[str, str | None] = {qid: None for qid in question_ids}
    explanations: dict[str, str] = {qid: "" for qid in question_ids}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("id", "")).strip()
            value = str(item.get("answer", "")).strip().lower()
            if qid in answers and value in _ANSWER_TO_BIT:
                answers[qid] = value
                explanations[qid] = str(item.get("explanation", "")).strip()
    parse_ok = all(value is not None for value in answers.values())
    return answers, explanations, parse_ok


def judge_yes_rate(
    answers: dict[str, str | None], questions: list[dict[str, str]] | None = None
) -> float | None:
    bits = [_ANSWER_TO_BIT[v] for v in answers.values() if v in _ANSWER_TO_BIT]
    return round(mean(bits), 6) if bits else None


def response_score(
    per_judge_answers: dict[str, dict[str, str | None]],
    questions: list[dict[str, str]] | None = None,
) -> float | None:
    rates = [
        r
        for r in (judge_yes_rate(a, questions) for a in per_judge_answers.values())
        if r is not None
    ]
    return round(mean(rates), 6) if rates else None


def challenger_beats_king(score_challenger: float, score_king: float) -> bool:
    return (score_challenger - score_king) >= CHALLENGER_WIN_MARGIN


def aggregate_scores(
    records: list[dict[str, Any]], *, min_valid_fraction: float = 0.8
) -> dict[str, Any]:
    total = len(records)
    valid = [r for r in records if r.get("scored")]
    valid_count = len(valid)
    judge_errors = sum(
        1
        for record in records
        for result in record.get("judge_results", [])
        if not result.get("parse_ok")
    )

    if total == 0 or valid_count / total < min_valid_fraction:
        return {
            "state": "failed",
            "score_challenger": None,
            "score_king": None,
            "challenger_won": None,
            "valid_turns": valid_count,
            "total_turns": total,
            "judge_errors": judge_errors,
            "scored_sample_count": valid_count,
            "fault_class": "PROVIDER_FAULT",
            "fault_code": "scoring_invalid",
            "fault_message": f"Only {valid_count}/{total} samples valid (< {min_valid_fraction:.0%})",  # noqa: E501
            "retryable": True,
        }

    challenger_mean = round(mean(r["challenger_score"] for r in valid), 6)
    king_mean = round(mean(r["king_score"] for r in valid), 6)

    by_judge: dict[str, float] = {}
    for judge_model in sorted(
        {
            result["judge_model"]
            for record in valid
            for result in record.get("judge_results", [])
            if result.get("side") == "challenger" and result.get("parse_ok")
        }
    ):
        rates = [
            float(result["yes_rate"])
            for record in valid
            for result in record.get("judge_results", [])
            if result.get("side") == "challenger"
            and result.get("judge_model") == judge_model
            and result.get("yes_rate") is not None
        ]
        if rates:
            by_judge[judge_model] = round(mean(rates), 6)

    return {
        "state": "succeeded",
        "score_challenger": challenger_mean,
        "score_king": king_mean,
        "challenger_won": challenger_beats_king(challenger_mean, king_mean),
        "required_win_margin": CHALLENGER_WIN_MARGIN,
        "valid_turns": valid_count,
        "total_turns": total,
        "judge_errors": judge_errors,
        "scored_sample_count": valid_count,
        "by_judge": by_judge,
        "by_metric": {},
        "scoring_mode": "binary",
        "fault_class": None,
        "fault_code": None,
        "fault_message": None,
        "retryable": None,
    }


_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'\{\s*"verdict"\s*:\s*"[^"]*"[^}]*\}', re.IGNORECASE),
    re.compile(r'\{\s*"injection"\s*:\s*(true|false)[^}]*\}', re.IGNORECASE),
)
_DELIMITER_INJECTION_RE = re.compile(
    r'\s*-{3,}[\s\S]*?(?:"verdict"\s*:|GRADING\s+INSTRUCTION)[\s\S]*$',
    re.DOTALL | re.IGNORECASE,
)
_VERDICT_LABELS = frozenset({"accept", "weak_pass", "reject"})


def strip_reply_injection(reply: str) -> str:
    cleaned = _DELIMITER_INJECTION_RE.sub("", reply or "")
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    if _scan_verdict_json(cleaned):
        return ""
    return cleaned.strip()


def _scan_verdict_json(text: str) -> bool:
    decoder = json.JSONDecoder()
    start = 0
    while True:
        index = text.find("{", start)
        if index == -1:
            return False
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            start = index + 1
            continue
        if isinstance(obj, dict):
            verdict = obj.get("verdict", "")
            if isinstance(verdict, str) and verdict.lower() in _VERDICT_LABELS:
                return True
            if any(str(key).lower() == "injection" for key in obj):
                return True
        start = index + 1

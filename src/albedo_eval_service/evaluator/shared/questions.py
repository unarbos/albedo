from __future__ import annotations

import os as _os
import re
from collections.abc import Callable
from difflib import SequenceMatcher

from ...shared.json_extract import extract_json
from ...simulator.prompt_simulator import COMPLETE_MARKER

QUESTION_FLOOR_FRACTION = 0.22
GENERIC_HYGIENE_QUESTION_LIMIT = 3
NEGATIVE_QUESTION_LIMIT = 8


def question_floor(n: int) -> int:
    return max(1, round(n * QUESTION_FLOOR_FRACTION))


VALID_TAGS = ("explore", "verification", "action", "continuity", "economy")

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)


def measure(text: str) -> dict[str, int]:
    blocks = _FENCE_RE.findall(text)
    fence = text.find("```")
    prose = (text[:fence] if fence >= 0 else text).strip()
    return {
        "total_words": len(text.split()),
        "total_chars": len(text),
        "prose_words": len(prose.split()),
        "code_blocks": len(blocks),
        "code_lines": max((len(b.strip("\n").splitlines()) for b in blocks), default=0),
        "code_chars": sum(len(b) for b in blocks),
    }


_CANDIDATE_BLOCK_RE = re.compile(r"CANDIDATE OUTPUT(?: \d+)?:\n------\n(.*?)\n------", re.DOTALL)


def candidate_output_measure(text: str) -> dict[str, int]:
    blocks = _CANDIDATE_BLOCK_RE.findall(text)
    if not blocks:
        blocks = [text]
    words = [len(b.split()) for b in blocks]
    chars = [len(b) for b in blocks]
    prose_words = [len(_FENCE_RE.sub("", b).split()) for b in blocks]
    return {
        "blocks": len(_CANDIDATE_BLOCK_RE.findall(text)),
        "total_words": sum(words),
        "max_words": max(words) if words else 0,
        "total_chars": sum(chars),
        "max_chars": max(chars) if chars else 0,
        "total_prose_words": sum(prose_words),
        "avg_words": round(sum(words) / len(blocks)) if blocks else 0,
        "avg_chars": round(sum(chars) / len(blocks)) if blocks else 0,
    }


def measurements_block(text: str) -> str:
    m = measure(text)
    c = candidate_output_measure(text)
    return (
        "MEASUREMENTS (computed programmatically from the trajectory above — authoritative for "
        "every size or length question):\n"
        f"- CANDIDATE OUTPUT words, all scored blocks combined (use THIS for any size bound on "
        f"the candidate's own outputs): {c['total_words']}\n"
        f"- longest single CANDIDATE OUTPUT: {c['max_words']} words "
        f"({c['blocks']} scored blocks)\n"
        f"- CANDIDATE OUTPUT characters, all scored blocks combined: {c['total_chars']}\n"
        f"- longest single CANDIDATE OUTPUT: {c['max_chars']} characters\n"
        f"- CANDIDATE OUTPUT prose words, all scored blocks combined, excluding fenced code: "
        f"{c['total_prose_words']}\n"
        f"- average CANDIDATE OUTPUT words per turn: {c['avg_words']} ({c['blocks']} scored blocks)\n"  # noqa: E501
        f"- average CANDIDATE OUTPUT characters per turn: {c['avg_chars']}\n"
        f"- total words of the whole document incl. context/observations: {m['total_words']}\n"
        f"- total characters: {m['total_chars']}\n"
        f"- fenced code blocks: {m['code_blocks']} (longest: {m['code_lines']} lines, "
        f"{m['code_chars']} code characters total)\n\n"
    )


_TESTS_VISIBLE_RE = re.compile(
    r"(^|[\s'\"/(=])tests?/|test_[a-z0-9_]+\.(py|go|rs|js|ts)|_test\.(go|rs|py)|pytest|cargo test",
    re.IGNORECASE | re.MULTILINE,
)


def tests_visible(messages: list[dict[str, str]] | None) -> bool:
    return any(_TESTS_VISIBLE_RE.search(m.get("content") or "") for m in messages or [])


def sample_phase(messages: list[dict[str, str]] | None) -> str:
    """Trim phase, inferred from the prefix the sampler produced: the cut lands near the start
    (cold), just before the first edit (pre_edit), or just after it (at_edit)."""
    turns = [m.get("content") or "" for m in messages or [] if m.get("role") == "assistant"]
    if any(_edited_in_turn(t) for t in turns):
        return "at_edit"
    return "cold" if len(turns) <= 2 else "pre_edit"


HORIZON_STRATA = (8, 12, 16)


def assign_horizons(samples) -> dict[str, int]:
    by_phase: dict[str, list[str]] = {}
    for sample in samples:
        by_phase.setdefault(sample_phase(sample.messages), []).append(sample.sample_id)
    horizons: dict[str, int] = {}
    for ids in by_phase.values():
        for index, sample_id in enumerate(sorted(ids)):
            horizons[sample_id] = HORIZON_STRATA[index % len(HORIZON_STRATA)]
    return horizons


# rubric tags carry no requires label; map them so the gate and label caps keep working
RUBRIC_TAG_REQUIRES = {
    "action": "action",
    "continuity": "action",
    "verification": "action",
    "explore": "read",
    "economy": "neutral",
}

_EDIT_BLOCK_RE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.DOTALL)
_EDIT_COMMAND_RE = re.compile(
    r"sed\s+-i"
    # a write counts only to a repo path: `2>/dev/null`, `2>&1`, /tmp scratch files
    # (reproduction scripts), `->`/`=>` arrows and `x > 5` comparisons are not edits —
    # the redirect target must look like a path (contain . or /)
    r"|(?<![-=0-9&])>>?\s*(?!/dev/|/tmp/)(?=[\w.~/-]*[./])[\w./~-]"
    r"|tee\s+(?!/dev/|/tmp/)[\w./~-]|cat\s*>>?\s*(?!/dev/|/tmp/)[\w./~-]"
    r"|str_replace|git\s+apply|patch\s+-p|applypatch|>{7}\s*REPLACE"
    r"|cp\s+[\w./-]+\s+(?!/dev/|/tmp/)[\w./-]+|mv\s+[\w./-]+\s+(?!/dev/|/tmp/)[\w./-]+",
)


def _edited_in_turn(text: str) -> bool:
    """Whether a turn's shell commands change the repository.

    Only bash blocks are scanned. Running the pattern over the whole turn made a `</think>` tag or
    a `>` in prose read as a redirect, which marked every candidate as having edited and so left
    apply_measurement_gate permanently inert.
    """
    return any(_EDIT_COMMAND_RE.search(cmd) for cmd in _EDIT_BLOCK_RE.findall(text or ""))


_SUBMIT_RE = re.compile(re.escape(COMPLETE_MARKER))
_UNFOLDED_AVOID_RE = re.compile(
    r"^\s*(?:does[^?]{0,60}\bavoid|is[^?]{0,60}\bfree of|does[^?]{0,60}\brefrain)", re.IGNORECASE
)
_ACTION_VERB_RE = re.compile(
    r"\b(edit|modif|submit|propagat|verif|appl|patch|chang|fix)\w*\b", re.IGNORECASE
)
_COMPLETED_WORK_RE = re.compile(
    r"\b(submit|final output|after the edit|the edit .{0,30}(applied|made|verified))\b",
    re.IGNORECASE,
)

READ_CAPS_BY_PHASE = {"cold": 10, "pre_edit": 7, "at_edit": 5}
READ_ONLY_QUESTION_CAP = 5


def candidate_turn_texts_from_merged(text: str) -> list[str]:
    return _CANDIDATE_BLOCK_RE.findall(text or "")


def trajectory_made_edit(turn_texts: list[str]) -> bool:
    return any(_edited_in_turn(text) for text in turn_texts)


def _discard_recorder(
    discards: list[dict[str, str]] | None,
    *,
    stage: str,
    origin: str = "content",
    counters: dict[str, int] | None = None,
) -> Callable[..., None]:
    def _drop(reason: str, text: str, detail: str = "") -> None:
        if counters is not None:
            counters[reason] = counters.get(reason, 0) + 1
        if discards is None:
            return
        entry = {"stage": stage, "reason": reason, "text": text, "origin": origin}
        if detail:
            entry["detail"] = detail
        discards.append(entry)

    return _drop


def enforce_question_labels(
    questions: list[dict[str, str]],
    *,
    phase: str,
    reference_made_edit: bool,
    discards: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    kept: list[dict[str, str]] = []
    drops = {"read_cap": 0, "unfolded_avoid": 0, "no_edit_dead_weight": 0}
    read_kept = 0
    read_only_question_cap = READ_CAPS_BY_PHASE.get(phase, READ_ONLY_QUESTION_CAP)
    _drop = _discard_recorder(discards, stage="enforce_question_labels", counters=drops)

    for question in questions:
        text = question.get("text", "")
        requires = question.get("requires") or "neutral"
        if requires not in ("action", "read", "neutral"):
            requires = "neutral"
        is_size = is_measurement_bound_question(text)
        if not is_size and _UNFOLDED_AVOID_RE.search(text) and not _ACTION_VERB_RE.search(text):
            _drop("unfolded_avoid", text)
            continue
        if not reference_made_edit and requires == "action" and _COMPLETED_WORK_RE.search(text):
            _drop("no_edit_dead_weight", text)
            continue
        if requires == "read" and not is_size:
            if read_kept >= read_only_question_cap:
                _drop("read_cap", text)
                continue
            read_kept += 1
        question["requires"] = requires
        kept.append(question)
    for position, question in enumerate(kept, start=1):
        question["id"] = f"q_{position:02d}"
    return kept, drops


_MUTED_PROSE_WORDS_PER_TURN = 10


def _is_muted(candidate_turn_texts: list[str]) -> bool:
    if not candidate_turn_texts:
        return False
    prose = sum(len(_FENCE_RE.sub("", t).split()) for t in candidate_turn_texts)
    return prose < _MUTED_PROSE_WORDS_PER_TURN * len(candidate_turn_texts)


def apply_measurement_gate(
    answers: dict[str, str | None],
    questions: list[dict[str, str]],
    *,
    candidate_turn_texts: list[str],
    reference_made_edit: bool,
) -> dict[str, str | None]:
    gated = dict(answers)
    if _is_muted(candidate_turn_texts):
        for question in questions:
            qid = question.get("id")
            if qid in gated and is_measurement_bound_question(question.get("text", "")):
                gated.pop(qid)
    made_edit = trajectory_made_edit(candidate_turn_texts)
    if made_edit:
        return gated
    final_is_read = bool(candidate_turn_texts) and not (
        _edited_in_turn(candidate_turn_texts[-1])
        or _SUBMIT_RE.search(candidate_turn_texts[-1] or "")
    )
    for question in questions:
        qid = question.get("id")
        if qid not in gated:
            continue
        text = question.get("text", "")
        if (
            question.get("requires") == "neutral"
            and not is_measurement_bound_question(text)
            and _NEGATIVE_QUESTION_RE.search(text)
            and _ACTION_VERB_RE.search(text)
        ):
            gated.pop(qid)
            continue
        if reference_made_edit and final_is_read and (question.get("requires") == "action"):
            gated[qid] = "0"
    return gated


_QUESTION_STOPWORDS = frozenset(
    "does the response a an as its it is are do of to in on for with or and that this by be "
    "not no any all one when if such e g rather than instead avoid avoids using use uses run "
    "runs running contain contains include includes reference references".split()
)
_DUP_JACCARD = float(_os.environ.get("ALBEDO_EXP_DUP_JACCARD", "0.75"))
_DUP_CONTAINMENT = float(_os.environ.get("ALBEDO_EXP_DUP_CONTAINMENT", "0.90"))
_DUP_CHAR_RATIO = float(_os.environ.get("ALBEDO_EXP_DUP_CHAR_RATIO", "0.87"))
_DUP_CHAR_MIN_LEN = 20
_TEMPLATE_KEY_TOKENS = 5
_TEMPLATE_MAX_PER_KEY = int(_os.environ.get("ALBEDO_EXP_TEMPLATE_MAX_PER_KEY", "4"))
_CONDITIONAL_RE = re.compile(r"^\s*if\b", re.IGNORECASE)
_UNCONDITIONAL_SUBMIT_RE = re.compile(
    r"\b(?:submit|submits|submitting|finali[sz]e|finali[sz]es|finish(?:es)?)\b",
    re.IGNORECASE,
)
_BOUNDED_SUBMIT_RE = re.compile(
    r"\b(?:after success|stop after success|submit after success|after verification|"
    r"after (?:the )?(?:observations?|verification|tests?|checks?|diff|requirement)[^?]{0,80}"
    r"(?:show|shows|pass|passes|succeed|succeeds|verified|satisfied|solved)|"
    r"(?:once|when)[^?]{0,100}(?:success|verified|satisfied|solved|passes|succeeds))\b",
    re.IGNORECASE,
)
_NEGATIVE_QUESTION_RE = re.compile(
    r"\b(avoid|avoids|avoided|not|never|does\s+not|do\s+not|without|refrains?)\b",
    re.IGNORECASE,
)
_GENERIC_HYGIENE_RE = re.compile(
    r"\b("
    r"word|words|characters?|shorter|exceeding|less than|"
    r"under (?:roughly|about|\d)|below (?:about|\d)|within (?:about|\d)|"
    r"thought (?:at most|fit|free|section)|sentences?|paragraph|"
    r"self-corrections?|raw chain-of-thought|scratch work|"
    r"quoting|restating|re-printing|"
    r"bash block at most|code block|destructive operations?|rm -rf|"
    r"read-only inspection|verification step|quotes?|backslashes|regex patterns?|"
    r"plan-action match"
    r")\b",
    re.IGNORECASE,
)


def _question_signature(text: str) -> frozenset[str]:
    words = re.sub(r"[^a-z0-9_./-]+", " ", text.casefold()).split()
    return frozenset(
        w.rstrip("s") if len(w) > 3 else w for w in words if w not in _QUESTION_STOPWORDS
    )


def _template_key(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for raw in text.split():
        core = raw.strip("?,.!:;\"'()")
        if not core:
            continue
        is_target = (
            "`" in raw
            or any(ch.isdigit() for ch in core)
            or any(ch in "._/=<>[]{}\\" for ch in core)
            or any(ch.isupper() for ch in core[1:])
        )
        normalized = "§" if is_target else core.casefold()
        if normalized == "§" and tokens and tokens[-1] == "§":
            continue
        tokens.append(normalized)
    return tuple(tokens[:_TEMPLATE_KEY_TOKENS])


def _near_duplicate(sig_a: frozenset[str], sig_b: frozenset[str], text_a: str, text_b: str) -> bool:
    if sig_a and sig_b:
        inter = len(sig_a & sig_b)
        if inter / len(sig_a | sig_b) >= _DUP_JACCARD:
            return True
        if inter / min(len(sig_a), len(sig_b)) >= _DUP_CONTAINMENT:
            return True
    if min(len(text_a), len(text_b)) >= _DUP_CHAR_MIN_LEN:
        ratio = SequenceMatcher(None, text_a.casefold(), text_b.casefold()).ratio()
        if ratio >= _DUP_CHAR_RATIO:
            return True
    return False


def _is_generic_hygiene_question(text: str) -> bool:
    return bool(_GENERIC_HYGIENE_RE.search(text))


_MEASUREMENT_BOUND_RE = re.compile(
    r"\b\d{2,6}\b[^?]{0,40}\b(?:words?|characters?|sentences?|lines)\b"
    r"|\b(?:words?|characters?|sentences?|lines)\b[^?]{0,40}\b\d{2,6}\b",
    re.IGNORECASE,
)
MEASUREMENT_QUESTION_LIMIT = 5


def is_measurement_bound_question(text: str) -> bool:
    return bool(_MEASUREMENT_BOUND_RE.search(text))


def _is_negative_question(text: str) -> bool:
    return bool(_NEGATIVE_QUESTION_RE.search(text))


def is_unbounded_submit_question(text: str) -> bool:
    return (
        bool(_UNCONDITIONAL_SUBMIT_RE.search(text))
        and not _is_negative_question(text)
        and not bool(_BOUNDED_SUBMIT_RE.search(text))
    )


def parse_questions(
    raw: str,
    n: int,
    *,
    discards: list[dict[str, str]] | None = None,
    origin: str = "content",
) -> tuple[list[dict[str, str]], bool]:
    obj = extract_json(raw, prefer_keys=("questions",))
    items = obj.get("questions") if isinstance(obj, dict) else obj
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    kept_signatures: list[frozenset[str]] = []
    template_counts: dict[tuple[str, ...], int] = {}
    generic_hygiene_count = 0
    negative_count = 0
    measurement_count = 0

    _drop = _discard_recorder(discards, stage="parse_questions", origin=origin)

    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                _drop("malformed_item", repr(item)[:200])
                continue
            if not str(item.get("text", "")).strip():
                _drop("empty_text", str(item.get("text", "")).strip())
                continue
            text = str(item["text"]).strip()
            if _CONDITIONAL_RE.match(text):
                _drop("conditional_form", text)
                continue
            key = " ".join(text.casefold().split())
            if key in seen:
                _drop("exact_duplicate", text)
                continue
            seen.add(key)
            if is_measurement_bound_question(text):
                if measurement_count >= MEASUREMENT_QUESTION_LIMIT:
                    _drop("measurement_cap", text)
                    continue
                measurement_count += 1
                kept_signatures.append(_question_signature(text))
                out.append(
                    {
                        "text": text,
                        "example_bad": str(item.get("example_bad", "")).strip(),
                        "requires": str(item.get("requires", "neutral")),
                        "tag": t
                        if (t := str(item.get("tag", "")).strip().lower()) in VALID_TAGS
                        else "",
                    }
                )
                continue
            is_negative = _is_negative_question(text)
            if is_negative and negative_count >= NEGATIVE_QUESTION_LIMIT:
                _drop("negative_cap", text)
                continue
            is_generic_hygiene = _is_generic_hygiene_question(text)
            if is_generic_hygiene and generic_hygiene_count >= GENERIC_HYGIENE_QUESTION_LIMIT:
                _drop("generic_hygiene_cap", text)
                continue
            template = _template_key(text)
            if (
                len(template) == _TEMPLATE_KEY_TOKENS
                and template_counts.get(template, 0) >= _TEMPLATE_MAX_PER_KEY
            ):
                _drop("template_cap", text)
                continue
            signature = _question_signature(text)
            near_dup_of = None
            for prev_sig, prev in zip(kept_signatures, out):
                if _near_duplicate(signature, prev_sig, text, prev["text"]):
                    near_dup_of = prev["text"]
                    break
            if near_dup_of is not None:
                _drop(
                    "near_duplicate",
                    text,
                    detail=f"near-duplicate of kept question: {near_dup_of!r}",
                )
                continue
            template_counts[template] = template_counts.get(template, 0) + 1
            if is_negative:
                negative_count += 1
            if is_generic_hygiene:
                generic_hygiene_count += 1
            kept_signatures.append(signature)
            out.append(
                {
                    "text": text,
                    "example_bad": str(item.get("example_bad", "")).strip(),
                    "requires": str(item.get("requires", "neutral")),
                    "tag": t
                    if (t := str(item.get("tag", "")).strip().lower()) in VALID_TAGS
                    else "",
                    **({"evidence": e} if (e := str(item.get("evidence", "")).strip()) else {}),
                    **({"step": item["step"]} if isinstance(item.get("step"), int) else {}),
                }
            )
    if len(out) > n:
        for extra in out[n:]:
            _drop(
                "truncated_excess",
                extra["text"],
                detail=f"{len(out)} well-formed candidates parsed, only the first {n} requested are kept",  # noqa: E501
            )
        out = out[:n]
    for position, question in enumerate(out, start=1):
        question["id"] = f"q_{position:02d}"
    return out, len(out) >= question_floor(n)

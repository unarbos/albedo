"""Response quality heuristics - pure text analysis, no LLM judge."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Keywords expected in at least one response for a coding task.
_CODE_KEYWORDS = {"def", "class", "import", "return", "if", "for", "while", "print", "="}

# Regex matching code-like syntax tokens used by check_code_syntax.
_SYNTAX_HINTS  = re.compile(r"(return|:|\(|\)|def |=|#)", re.IGNORECASE)


@dataclass
class CheckResult:
    # Outcome of a single check with an optional human-readable failure reason.
    passed: bool
    reason: str = ""


# ── Per-response checks ───────────────────────────────────────────────────────

def check_empty(text: str) -> CheckResult:
    # Fails if the model produced nothing or only whitespace.
    if not text.strip():
        return CheckResult(False, "empty response")
    return CheckResult(True)


def check_length(text: str, min_tokens: int = 5) -> CheckResult:
    # Fails if response is shorter than min_tokens - catches single-word outputs.
    tokens = text.split()
    if len(tokens) < min_tokens:
        return CheckResult(False, f"too short ({len(tokens)} tokens, min={min_tokens})")
    return CheckResult(True)


def check_repetition(text: str, max_repetition: float = 0.85) -> CheckResult:
    # Fails if >85% of consecutive trigrams are identical - catches "to to to to" token loops.
    tokens = text.split()
    if len(tokens) < 3:
        return CheckResult(True)
    trigrams  = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    diversity = len(set(trigrams)) / len(trigrams)
    if diversity < (1.0 - max_repetition):
        return CheckResult(False, f"repetitive output (diversity={diversity:.2f})")
    return CheckResult(True)


def check_encoding(text: str) -> CheckResult:
    # Fails if >60% of characters are non-ASCII - catches garbled or wrong-encoding weights.
    if len(text) > 20 and sum(1 for c in text if ord(c) > 127) / len(text) > 0.6:
        return CheckResult(False, "excessive non-ASCII (encoding broken)")
    return CheckResult(True)


def check_vocabulary(text: str, min_ratio: float = 0.3) -> CheckResult:
    # Fails if unique/total token ratio is below 30% - catches low-variety outputs like "the the the".
    tokens = text.lower().split()
    if len(tokens) < 8:
        return CheckResult(True)
    ratio = len(set(tokens)) / len(tokens)
    if ratio < min_ratio:
        return CheckResult(False, f"low vocabulary diversity ({ratio:.2f}, min={min_ratio})")
    return CheckResult(True)


def check_code_syntax(text: str) -> CheckResult:
    # Fails if response contains no code-like syntax (return/def/:/=); for completion prompts only.
    # Not used in check_all - call directly when the prompt is known to be a completion task.
    if not _SYNTAX_HINTS.search(text):
        return CheckResult(False, "no code syntax found (return/def/:/= expected)")
    return CheckResult(True)


def check_one(
    text: str,
    min_tokens: int = 5,
    max_repetition: float = 0.85,
    min_vocab_ratio: float = 0.3,
    check_vocab: bool = True,
) -> CheckResult:
    # Runs all per-response checks in order and returns the first failure.
    for result in [
        check_empty(text),
        check_length(text, min_tokens),
        check_repetition(text, max_repetition),
        check_encoding(text),
        check_vocabulary(text, min_vocab_ratio) if check_vocab else CheckResult(True),
    ]:
        if not result.passed:
            return result
    return CheckResult(True)


# ── Cross-prompt checks ───────────────────────────────────────────────────────

def check_collapsed(responses: list[str]) -> CheckResult:
    # Fails if all responses are identical - the model ignores the prompt entirely.
    if len({r.strip()[:100] for r in responses}) == 1:
        return CheckResult(False, "identical response to all prompts (collapsed model)")
    return CheckResult(True)


def check_uniform_length(responses: list[str]) -> CheckResult:
    # Fails if all responses have the exact same token count - a hidden collapse signal.
    lengths = [len(r.split()) for r in responses]
    if len(responses) >= 2 and len(set(lengths)) == 1:
        return CheckResult(False,
            f"all responses identical length ({lengths[0]} tokens) - possible collapse")
    return CheckResult(True)


def check_code_present(responses: list[str]) -> CheckResult:
    # Fails if no response contains any code keyword - model not engaging with coding tasks.
    for resp in responses:
        if set(resp.lower().split()) & _CODE_KEYWORDS:
            return CheckResult(True)
    return CheckResult(False, "no code keywords in any response (def/return/import/etc)")


# ── Main entry point ──────────────────────────────────────────────────────────

def check_all(
    responses: list[str],
    min_tokens: int = 5,
    max_repetition: float = 0.85,
    min_vocab_ratio: float = 0.3,
) -> CheckResult:
    # Runs per-response checks first, then cross-prompt checks; returns the first failure.
    for i, resp in enumerate(responses):
        result = check_one(resp, min_tokens=min_tokens, max_repetition=max_repetition,
                           min_vocab_ratio=min_vocab_ratio)
        if not result.passed:
            return CheckResult(False, f"prompt {i+1}/{len(responses)}: {result.reason}")

    for fn in [check_collapsed, check_uniform_length, check_code_present]:
        result = fn(responses)
        if not result.passed:
            return result

    return CheckResult(True)

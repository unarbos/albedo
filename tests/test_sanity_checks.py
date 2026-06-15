"""Tests for the sanity response heuristics (checks.py) and the worker's _heuristics wrapper."""
from __future__ import annotations

from sanity_remote.models import SanityRunRequest
from sanity_remote.worker import _heuristics
from sanity_service.checks import (
    check_all,
    check_code_present,
    check_collapsed,
    check_encoding,
    check_one,
    check_repetition,
    check_uniform_length,
    check_vocabulary,
)

# ── per-response checks ─────────────────────────────────────────────────────────

def test_check_one_passes_clean_code_answer():
    assert check_one("def add(a, b): return a + b done").passed


def test_check_one_flags_empty_then_short():
    assert check_one("").reason == "empty response"
    assert not check_one("one two").passed  # below min_tokens=5


def test_check_repetition_catches_token_loop():
    assert not check_repetition("na " * 10).passed
    assert check_repetition("the quick brown fox jumps over").passed


def test_check_encoding_catches_garbled_weights():
    assert not check_encoding("é" * 40).passed
    assert check_encoding("plain ascii sentence that is long enough").passed


def test_check_vocabulary_catches_low_variety():
    # High trigram diversity but only two unique tokens -> repetition passes, vocabulary fails.
    assert not check_vocabulary("a b a b a b a b a b").passed
    assert check_vocabulary("alpha beta gamma delta epsilon zeta eta theta").passed


# ── cross-prompt checks ─────────────────────────────────────────────────────────

def test_check_collapsed_flags_identical_responses():
    assert not check_collapsed(["same", "same", "same"]).passed
    assert check_collapsed(["one", "two", "three"]).passed
    # single response must never fail - a set of size 1 is expected for n=1
    assert check_collapsed(["any single response"]).passed


def test_check_uniform_length_flags_identical_token_counts():
    assert not check_uniform_length(["a b c", "d e f", "g h i"]).passed
    assert check_uniform_length(["a b", "c d e"]).passed
    assert check_uniform_length(["only one"]).passed  # single response cannot collapse


def test_check_code_present_requires_a_keyword_somewhere():
    assert check_code_present(["prose here", "def f(): return 1"]).passed
    assert not check_code_present(["just prose", "more prose"]).passed


def test_check_all_reports_first_failure_with_prompt_index():
    result = check_all(["def f(): return 1 ok now", "", "def g(): return 2 ok now"])
    assert not result.passed
    assert result.reason.startswith("prompt 2/3:")


# ── worker _heuristics wrapper ──────────────────────────────────────────────────

def _req() -> SanityRunRequest:
    return SanityRunRequest(run_id="r", model_uri="m", digest="d", prompts=["p"])


def test_heuristics_skip_passes_all_without_inspection():
    out = _heuristics(["", "", ""], _req(), skip=True)
    assert all(v["passed"] for v in out)
    assert out[0]["reason"] == "heuristics disabled"


def test_heuristics_set_failure_fails_every_response():
    out = _heuristics(["dup text", "dup text", "dup text"], _req())
    assert not any(v["passed"] for v in out)
    assert all("collapsed" in v["reason"] for v in out)


def test_heuristics_passes_varied_code_responses():
    responses = [
        "def add(a, b): return a + b here",
        "the function computes a sum of two numbers correctly",
        "import math then call the helper to finish now",
    ]
    out = _heuristics(responses, _req())
    assert all(v["passed"] for v in out), out

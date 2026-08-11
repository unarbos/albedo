"""Reusable pieces for locally replicating the Albedo judge duel pipeline.

This module imports the *real* validator code from ``src/albedo_eval_service``
(JudgeLLMClient, QuestionService, ObservationSimulationService, the scoring
helpers in judge_core.py, ...) instead of reimplementing any of it. That keeps
local scoring numbers byte-for-byte consistent with what the validator will
compute, as long as this checkout is kept up to date with upstream.

Requires ``ALBEDO_JUDGE_OPENROUTER_API_KEY`` (and optionally the
``ALBEDO_JUDGE_ENGY_*`` vars) set in a ``.env`` file at the repo root, or
passed via ``--env`` to run_duel.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import httpx  # noqa: E402

from albedo_eval_service.judge_api import (  # noqa: E402
    ObservationSimulationService,
    QuestionPrepSample,
    QuestionService,
    ReferenceTrajectoryService,
    RepoContextClient,
    SimulateObservationRequest,
    _judge_side,
)
from albedo_eval_service.judge_config import JudgeSettings, get_judge_settings  # noqa: E402
from albedo_eval_service.judge_core import (  # noqa: E402
    JUDGE_MODELS,
    aggregate_scores,
    challenger_beats_king,
    response_score,
    sample_phase,
)
from albedo_eval_service.judge_llm_client import JudgeLLMClient  # noqa: E402
from albedo_eval_service.observation_format import detect_format, wrap  # noqa: E402
from albedo_eval_service.remote_generation import format_scored_trajectory  # noqa: E402
from sanity_service.checks import check_one  # noqa: E402

# Mirrors src/sanity_service/dispatcher.py's private helpers of the same name. Reimplemented
# here (rather than imported) so this script doesn't pull in dispatcher.py's Postgres/queue
# dependencies just for four regex-sized utility functions.
import re  # noqa: E402

_COMPLETE_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)\s*\n.*?```", re.IGNORECASE | re.DOTALL)


def _assistant_submitted(output: str) -> bool:
    return _COMPLETE_MARKER in output


def _has_bash_command(output: str) -> bool:
    return bool(_BASH_BLOCK_RE.search(output))


def _completion_observation(sample_id: str, messages: list[dict[str, str]] | None = None) -> str:
    return wrap(_COMPLETE_MARKER, detect_format(sample_id, messages))


def _missing_command_observation(
    sample_id: str, messages: list[dict[str, str]] | None = None
) -> str:
    return wrap(
        "No bash command found in assistant message.",
        detect_format(sample_id, messages),
        returncode=2,
    )

__all__ = [
    "JUDGE_MODELS",
    "aggregate_scores",
    "build_question_service",
    "generate_candidate_turns",
    "load_settings",
    "prefix_and_turn_count",
    "sample_phase",
    "score_challenger_vs_king",
]


def load_settings(env_file: str | Path | None = None) -> JudgeSettings:
    """Load JudgeSettings, optionally from a specific .env file (defaults to repo root .env)."""
    if env_file is None:
        env_file = REPO_ROOT / ".env"
    return JudgeSettings(_env_file=str(env_file))  # type: ignore[call-arg]


def build_question_service(
    client_llm: JudgeLLMClient, settings: JudgeSettings
) -> tuple[QuestionService, ObservationSimulationService, RepoContextClient | None]:
    repo_context = RepoContextClient(settings) if settings.repo_context_url else None
    simulator = ObservationSimulationService(settings, client_llm, repo_context)
    reference_service = ReferenceTrajectoryService(settings, client_llm, simulator)
    question_service = QuestionService(settings, client_llm, reference_service)
    return question_service, simulator, repo_context


def prefix_and_turn_count(turns: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Split stored turns into the shared context prefix and the scored-turn budget.

    Both king and challenger trajectories in generated-samples.jsonl fork from an identical
    prefix (everything before the first score_target=True turn); the number of score_target
    turns after that is the per-sample turn budget (currently 8, per docs/DATASETS.md).
    """
    for i, t in enumerate(turns):
        if t.get("score_target"):
            prefix = turns[:i]
            n_turns = sum(1 for t2 in turns[i:] if t2.get("score_target"))
            return prefix, n_turns
    return turns, 0


async def generate_candidate_turns(
    *,
    simulator: ObservationSimulationService,
    sample_id: str,
    prompt: str,
    prefix_turns: list[dict[str, Any]],
    candidate_base_url: str,
    candidate_model: str,
    candidate_api_key: str = "",
    n_turns: int,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    eval_run_id: str = "",
    request_timeout_seconds: float = 180.0,
) -> list[dict[str, Any]]:
    """Drive YOUR checkpoint through the same multi-turn duel loop the validator runs.

    Environment turns are produced by the real ObservationSimulationService, so simulated
    tool output matches validator behavior (including the Engy/DeepSeek routing, format
    detection, and contract checks) exactly. The candidate model itself is called through
    any OpenAI-compatible /chat/completions endpoint (e.g. a local vLLM server).

    Stop conditions mirror src/sanity_service/dispatcher.py exactly:
    - the candidate emits COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT -> append completion
      observation, stop (trajectory finished).
    - the response has no bash command AND fails a sanity/heuristic check (empty,
      truncated, repetitive, low-vocab, unclosed <think>, ...) -> stop with no further
      turns, same as the "hard fail halts the sample" behavior added in commit d1c43bb.
    - the response has no bash command but passes sanity checks -> inject a "no command
      found" observation and continue (this does NOT stop the trajectory).
    - otherwise -> simulate a normal environment observation and continue.
    """
    headers = {"Authorization": f"Bearer {candidate_api_key}"} if candidate_api_key else {}
    convo = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in prefix_turns]
    turns: list[dict[str, Any]] = []
    fmt_messages = prefix_turns
    async with httpx.AsyncClient(
        base_url=candidate_base_url.rstrip("/"),
        headers=headers,
        timeout=httpx.Timeout(request_timeout_seconds),
    ) as candidate_client:
        for turn_index in range(max(1, n_turns)):
            resp = await candidate_client.post(
                "/chat/completions",
                json={
                    "model": candidate_model,
                    "messages": convo,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            body = resp.json()
            choices = body.get("choices") or [{}]
            text = ((choices[0] or {}).get("message") or {}).get("content") or ""
            text = text.strip()
            turns.append({"role": "assistant", "content": text, "score_target": True})

            if _assistant_submitted(text):
                turns.append(
                    {
                        "role": "user",
                        "content": _completion_observation(sample_id, fmt_messages),
                        "environment_observation": True,
                    }
                )
                break

            has_command = _has_bash_command(text)
            if not has_command:
                gate = check_one(text)
                if not gate.passed:
                    # matches dispatcher.py's heuristic_reason gate: no bash command +
                    # failed sanity check halts the trajectory with no further turns.
                    break

            last = turn_index == n_turns - 1
            if last:
                break

            if not has_command:
                observation = _missing_command_observation(sample_id, fmt_messages)
            else:
                observation = await simulator.simulate(
                    SimulateObservationRequest(
                        eval_run_id=eval_run_id,
                        sample_id=sample_id,
                        prompt=prompt,
                        assistant_output=text,
                        messages=convo,
                    )
                )
            turns.append(
                {"role": "user", "content": observation, "environment_observation": True}
            )
            convo = convo + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": observation},
            ]
    return prefix_turns + turns


async def score_challenger_vs_king(
    *,
    client_llm: JudgeLLMClient,
    settings: JudgeSettings,
    sample_id: str,
    questions: list[dict[str, Any]],
    king_output: str,
    challenger_output: str,
    judge_models: list[str],
    reference_made_edit: bool | None,
) -> dict[str, Any]:
    """Score both sides through the exact same `_judge_side` path judge_api.py uses.

    This applies the phase-aware measurement gate, think-leak stripping, and per-judge-model
    aggregation exactly as the live judge service does; only the transport (calling this
    in-process instead of via the /score-batch HTTP route) differs.
    """

    async def _side(side: str, text: str):
        return await _judge_side(
            client=client_llm,
            settings=settings,
            side=side,
            response_text=text,
            questions=questions,
            judge_models=judge_models,
            reference_made_edit=reference_made_edit,
        )

    import asyncio

    (king_answers, king_recs), (chal_answers, chal_recs) = await asyncio.gather(
        _side("previous_king", king_output),
        _side("challenger", challenger_output),
    )
    king_score = response_score(king_answers, questions)
    chal_score = response_score(chal_answers, questions)
    king_ok = all(r["parse_ok"] for r in king_recs) and king_score is not None
    chal_ok = all(r["parse_ok"] for r in chal_recs) and chal_score is not None
    scored = king_ok and chal_ok
    return {
        "sample_id": sample_id,
        "questions": questions,
        "king_score": king_score,
        "challenger_score": chal_score,
        "judge_results": king_recs + chal_recs,
        "scored": scored,
        "scoring_mode": "binary",
        "challenger_won": (
            challenger_beats_king(chal_score, king_score) if scored else None
        ),
    }


async def prepare_questions(
    question_service: QuestionService,
    *,
    sample_id: str,
    prompt: str,
    prefix_turns: list[dict[str, Any]],
    n_turns: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate a fresh SOTA-anchored checklist for a sample that has no cached questions."""
    sample = QuestionPrepSample(
        sample_id=sample_id,
        prompt=prompt,
        messages=prefix_turns,
        assistant_turns=n_turns,
    )
    result = await question_service.prepare(sample)
    if result.error:
        raise RuntimeError(result.error)
    return result.questions, result.source

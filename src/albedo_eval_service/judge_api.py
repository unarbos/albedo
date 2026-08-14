from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from albedo_config import JudgeSettings, get_judge_settings
from albedo_config.models import JUDGE_MODELS

from .control.notifications import EvalErrorNotification, notify_eval_error
from .evaluator.behavior.prompt_behavior import BEHAVIOR_K, BEHAVIOR_PHASES
from .evaluator.behavior.questions import (
    behavior_question_schema,
    build_behavior_messages,
    filter_behavior_questions,
)
from .evaluator.reference.questions import (
    build_reference_question_messages,
    duplicate_economy_bounds,
    filter_reference_leaks,
    format_reference_trajectory,
    reference_question_schema,
)
from .evaluator.shared.budgets import RUBRIC_MAX_QUESTIONS
from .evaluator.shared.questions import (
    RUBRIC_TAG_REQUIRES,
    apply_measurement_gate,
    candidate_turn_texts_from_merged,
    enforce_question_labels,
    parse_questions,
    sample_phase,
    tests_visible,
    trajectory_made_edit,
)
from .judge_core import (
    aggregate_scores,
    answer_schema,
    build_judge_messages,
    judge_yes_rate,
    parse_answers,
    response_score,
)
from .judge_llm_client import JudgeLLMClient
from .remote.generation import format_scored_trajectory
from .shared.observation_format import (
    CommandContract,
    command_contract,
    contract_violation,
    detect_format,
    empty_output,
    first_bash_block,
    has_content,
    is_truncated,
    repair_output,
    repair_to_contract,
    requires_output,
    valid_output,
)
from .simulator.prompt_simulator import (
    COMPLETE_MARKER,
    missing_command_output,
    reference_completion_observation,
    simulation_system_prompt,
)


class QuestionPrepSample(BaseModel):
    sample_id: str
    prompt: str
    sample_index: int = 0
    messages: list[dict[str, str]] | None = None
    assistant_turns: int = 0


class QuestionPrepRequest(BaseModel):
    eval_run_id: str
    batch_id: str = "category-prep"
    samples: list[QuestionPrepSample]
    total_sample_count: int


class QuestionPrepResponse(BaseModel):
    eval_run_id: str
    category_prep_id: str
    accepted_sample_count: int


class JudgeSample(BaseModel):
    sample_id: str
    prompt: str
    previous_king_output: str
    challenger_output: str
    sample_index: int = 0
    messages: list[dict[str, str]] | None = None
    assistant_turns: int = 0


class ScoreBatchRequest(BaseModel):
    eval_run_id: str
    batch_id: str
    samples: list[JudgeSample]
    total_sample_count: int
    judge_models: list[str] = Field(default_factory=lambda: list(JUDGE_MODELS))
    category_prep_id: str | None = None


class ScoreBatchResponse(BaseModel):
    eval_run_id: str
    batch_id: str
    scoring_records: list[dict[str, Any]]
    summary: dict[str, Any]


class SimulateObservationRequest(BaseModel):
    eval_run_id: str
    sample_id: str
    prompt: str
    assistant_output: str
    messages: list[dict[str, str]] | None = None


class SimulateObservationResponse(BaseModel):
    eval_run_id: str
    sample_id: str
    observation: str


@dataclass(frozen=True)
class QuestionPrepResult:
    questions: list[dict[str, str]]
    source: dict[str, object]
    error: str | None = None


@dataclass(frozen=True)
class QuestionPrepLookup:
    result: QuestionPrepResult | None
    reason: str


class QuestionScoringUnavailable(RuntimeError):
    pass


class ObservationSimulationUnavailable(RuntimeError):
    pass


def _evaluator_provider(settings: JudgeSettings) -> dict[str, Any]:
    block: dict[str, Any] = {"allow_fallbacks": True, "quantizations": ["fp8"]}
    order = [p.strip() for p in settings.evaluator_providers.split(",") if p.strip()]
    if order:
        block["order"] = order
        block["allow_fallbacks"] = False
    return block


def _simulation_provider(settings: JudgeSettings) -> dict[str, Any] | None:
    allowed = [p.strip() for p in settings.simulation_providers.split(",") if p.strip()]
    if not allowed:
        return None
    return {"order": allowed, "allow_fallbacks": False}


_REROLL_WINDOW_TURNS = 5


class ReferenceTrajectoryService:
    def __init__(
        self,
        settings: JudgeSettings,
        client: JudgeLLMClient,
        simulator: "ObservationSimulationService",
    ):
        self.settings = settings
        self.client = client
        self.simulator = simulator

    def _model_for(self, sample_id: str, *, offset: int = 0) -> str:
        pool = [m.strip() for m in self.settings.sota_models.split(",") if m.strip()]
        if not pool:
            raise QuestionScoringUnavailable("ALBEDO_JUDGE_SOTA_MODELS is empty")
        index = random.Random(sample_id).randrange(len(pool))
        return pool[(index + offset) % len(pool)]

    async def generate(
        self, sample: QuestionPrepSample, *, eval_run_id: str = ""
    ) -> tuple[str, str, bool]:
        reference, model, made_edit, _ = await self._generate_once(
            sample, eval_run_id, extra_turns=0
        )
        return reference, model, made_edit

    async def reroll_for_material(
        self, sample: QuestionPrepSample, *, eval_run_id: str = "", exclude_model: str
    ) -> tuple[str, str, bool] | None:
        window = min(_REROLL_WINDOW_TURNS, self.settings.sota_trajectory_turns)
        extra = window - max(1, sample.assistant_turns or self.settings.sota_trajectory_turns)
        try:
            reference, model, made_edit, steps = await self._generate_once(
                sample, eval_run_id, extra_turns=extra, model_offset=1
            )
        except QuestionScoringUnavailable as exc:
            logger.warning("reference_reroll_failed sample_id={} error={}", sample.sample_id, exc)
            return None
        pool = [m.strip() for m in self.settings.sota_models.split(",") if m.strip()]
        if steps < 2 or (model == exclude_model and len(pool) > 1):
            return None
        logger.info(
            "reference_reroll_used sample_id={} replaced={} with={}/{}steps window={}",
            sample.sample_id,
            exclude_model,
            model,
            steps,
            window,
        )
        return reference, model, made_edit

    async def _generate_once(
        self,
        sample: QuestionPrepSample,
        eval_run_id: str,
        *,
        extra_turns: int,
        model_offset: int = 0,
    ) -> tuple[str, str, bool, int]:
        model = self._model_for(sample.sample_id, offset=model_offset)
        turn_count = (
            max(1, sample.assistant_turns or self.settings.sota_trajectory_turns) + extra_turns
        )
        convo = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in (sample.messages or [])
        ]
        fmt = detect_format(sample.sample_id, sample.messages)
        turns: list[dict[str, Any]] = []
        for turn_index in range(turn_count):
            response = await self.client.complete(
                purpose="reference",
                model=model,
                messages=convo,
                temperature=0.0,
                eval_run_id=eval_run_id,
                max_tokens=self.settings.sota_max_tokens,
                provider=_evaluator_provider(self.settings)
                if model == self.settings.evaluator_model
                else None,
                accept=lambda raw: bool(raw.strip()),
            )
            if response.error or not response.raw.strip():
                raise QuestionScoringUnavailable(
                    f"reference generation failed: {response.error or 'empty output'}"
                )
            text = response.raw.strip()
            turns.append({"role": "assistant", "content": text, "score_target": True})
            last = turn_index == turn_count - 1
            if COMPLETE_MARKER in text:
                if not last:
                    turns.append(
                        {
                            "role": "user",
                            "content": reference_completion_observation(fmt),
                            "environment_observation": True,
                        }
                    )
                break
            if last:
                break
            observation = await self.simulator.simulate(
                SimulateObservationRequest(
                    eval_run_id=eval_run_id,
                    sample_id=sample.sample_id,
                    prompt=sample.prompt,
                    assistant_output=text,
                    messages=convo,
                )
            )
            turns.append({"role": "user", "content": observation, "environment_observation": True})
            convo = convo + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": observation},
            ]
        reference = format_reference_trajectory(turns)
        if not reference.strip():
            raise QuestionScoringUnavailable("reference trajectory rendered empty")
        generated = [t["content"] for t in turns if t.get("score_target")]
        made_edit = trajectory_made_edit(generated)
        return reference, model, made_edit, len(generated)


# Questions the reference itself fails are deleted; too few survivors means the
# reference/checklist pair is unusable and the reference is rerolled.
PRUNE_MIN_SURVIVORS = 8
_REF_STEP_SPLIT_RE = re.compile(r"^REFERENCE STEP \d+:$", re.M)


def _reference_document(messages: list[dict[str, str]], reference: str) -> str:
    """Render the reference trajectory as a judgeable candidate document."""
    turns: list[dict[str, Any]] = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in messages
        if m.get("content")
    ]
    for segment in _REF_STEP_SPLIT_RE.split(reference)[1:]:
        body, _, observation = segment.partition("\nENVIRONMENT OBSERVATION:\n")
        turns.append({"role": "assistant", "content": body.strip(), "score_target": True})
        if observation.strip():
            turns.append(
                {"role": "user", "content": observation.strip(), "environment_observation": True}
            )
    return format_scored_trajectory(turns)


class QuestionService:
    def __init__(
        self,
        settings: JudgeSettings,
        client: JudgeLLMClient,
        reference_service: ReferenceTrajectoryService,
    ):
        self.settings = settings
        self.client = client
        self.reference_service = reference_service

    async def prepare(
        self, sample: QuestionPrepSample | JudgeSample, *, eval_run_id: str = ""
    ) -> QuestionPrepResult:
        if not getattr(sample, "messages", None):
            raise QuestionScoringUnavailable(
                "sample carries no prior context to anchor a reference trajectory"
            )
        try:
            reference, reference_model, reference_made_edit = await self.reference_service.generate(
                sample, eval_run_id=eval_run_id
            )
        except Exception as exc:
            logger.warning(
                "reference_trajectory_failed sample_id={} error={} retrying=reference_reroll",
                sample.sample_id,
                f"{type(exc).__name__}: {exc}",
            )
            rerolled = await self.reference_service.reroll_for_material(
                sample, eval_run_id=eval_run_id, exclude_model=""
            )
            if rerolled is None:
                raise QuestionScoringUnavailable(
                    f"reference unavailable: {type(exc).__name__}: {exc}"
                ) from exc
            reference, reference_model, reference_made_edit = rerolled
        try:
            return await self._prepare_once(sample, reference, reference_model, reference_made_edit)
        except QuestionScoringUnavailable as exc:
            logger.warning(
                "anchored_questions_failed sample_id={} error={} retrying=reference_reroll",
                sample.sample_id,
                exc,
            )
            rerolled = await self.reference_service.reroll_for_material(
                sample, eval_run_id=eval_run_id, exclude_model=reference_model or ""
            )
            if rerolled is None:
                raise
            return await self._prepare_once(sample, *rerolled)

    async def _prepare_once(
        self,
        sample: QuestionPrepSample | JudgeSample,
        reference: str,
        reference_model: str | None,
        reference_made_edit: bool,
    ) -> QuestionPrepResult:
        n = self.settings.num_questions
        prefix = getattr(sample, "messages", None)
        phase = sample_phase(prefix)
        # behaviour questions are calibrated by the measured model deltas rather than pruned
        # against the reference the way reference questions are below
        do_behavior = n >= 3 * BEHAVIOR_K
        n_generic = RUBRIC_MAX_QUESTIONS
        generic_floor = 6
        prefix_tail = "\n".join(
            f"[{m.get('role')}] {(m.get('content') or '')[:800]}" for m in (prefix or [])[-3:]
        )
        context_text = (sample.prompt or "") + "\n" + prefix_tail
        discarded: list[dict[str, str]] = []
        parse_budget = max(1, self.settings.parse_retries)

        def _reject_logger(reason: str, origin: str):
            attempts = 0

            def record(detail: str) -> bool:
                nonlocal attempts
                attempts += 1
                exhausted = attempts >= parse_budget
                discarded.append(
                    {
                        "stage": "generation_retry",
                        "reason": reason,
                        "text": "",
                        "origin": origin,
                        "detail": detail
                        + (
                            "; retries exhausted, this attempt was kept as-is"
                            if exhausted
                            else "; whole attempt discarded and regenerated"
                        ),
                    }
                )
                return not exhausted

            return record

        def _record_rejected_attempt(
            entries: list[dict[str, str]], survivors: list[dict[str, str]], origin: str
        ) -> None:
            discarded.extend(
                {**entry, "stage": f"rejected_attempt:{entry['stage']}"} for entry in entries
            )
            discarded.extend(
                {
                    "stage": "rejected_attempt",
                    "reason": "attempt_regenerated",
                    "text": question.get("text", ""),
                    "origin": origin,
                    "detail": "parsed cleanly, binned with the attempt that fell short",
                }
                for question in survivors
            )

        def _reference_call():
            _rejected = _reject_logger("content_batch_rejected", "content")

            def _reference_accept(raw: str) -> bool:
                attempt_discards: list[dict[str, str]] = []
                questions, _ok = parse_questions(
                    raw, n_generic, discards=attempt_discards, origin="content"
                )
                questions = filter_reference_leaks(questions, discards=attempt_discards)
                accepted = len(questions) >= generic_floor
                if not accepted and _rejected(
                    f"{len(questions)} well-formed questions parsed, needed >= {generic_floor}"
                ):
                    _record_rejected_attempt(attempt_discards, questions, "content")
                return accepted

            messages = build_reference_question_messages(
                task=sample.prompt,
                reference=reference,
                fmt=detect_format(sample.sample_id, prefix),
                prefix_turns=sum(1 for m in prefix or [] if m.get("role") == "assistant"),
                candidate_turns=getattr(sample, "assistant_turns", 0)
                or self.settings.sota_trajectory_turns,
            )

            return self.client.complete(
                purpose="questions",
                model=self.settings.evaluator_model,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=self.settings.question_max_tokens,
                provider=_evaluator_provider(self.settings),
                response_schema=reference_question_schema(),
                accept=_reference_accept,
            )

        def _behavior_call(index: int):
            _rejected = _reject_logger("behavior_batch_rejected", "behavior")

            def _behavior_accept(raw: str) -> bool:
                attempt_discards: list[dict[str, str]] = []
                qs, _ = parse_questions(
                    raw, BEHAVIOR_K, discards=attempt_discards, origin="behavior"
                )
                accepted = len(qs) >= 5
                if not accepted and _rejected(
                    f"phase={phase} index={index}: {len(qs)} well-formed questions "
                    "parsed, needed >= 5"
                ):
                    _record_rejected_attempt(attempt_discards, qs, "behavior")
                return accepted

            return self.client.complete(
                purpose="questions",
                model=self.settings.evaluator_model,
                messages=build_behavior_messages(
                    phase=phase,
                    index=index,
                    task=sample.prompt,
                    prefix_tail=prefix_tail,
                    k=BEHAVIOR_K,
                    tests_seen=tests_visible(prefix),
                ),
                temperature=self.settings.temperature,
                max_tokens=self.settings.question_max_tokens,
                provider=_evaluator_provider(self.settings),
                response_schema=behavior_question_schema(BEHAVIOR_K),
                accept=_behavior_accept,
            )

        calls = [_reference_call()] + ([_behavior_call(i) for i in range(3)] if do_behavior else [])
        responses = await asyncio.gather(*calls)
        response = responses[0]
        for r in responses:
            if r.error:
                raise QuestionScoringUnavailable(r.error)

        reference_qs, _ok = parse_questions(
            responses[0].raw, n_generic, discards=discarded, origin="content"
        )

        for q in reference_qs:
            q["requires"] = RUBRIC_TAG_REQUIRES.get(q.get("tag"), q.get("requires") or "neutral")
            q["tag"] = "reference:" + (q.get("tag") or "")

        reference_qs = filter_reference_leaks(reference_qs, discards=discarded)
        reference_qs, drops = enforce_question_labels(
            reference_qs,
            phase=phase,
            reference_made_edit=reference_made_edit,
            discards=discarded,
        )

        if len(reference_qs) < generic_floor:
            raise QuestionScoringUnavailable(
                f"evaluator returned {len(reference_qs)}/{generic_floor}+ well-formed questions"
            )
        pruned_info: dict[str, object] = {}

        try:
            kept_qs, dropped_qs, self_rate = await self._prune_against_reference(
                sample, reference, reference_qs
            )
        except Exception as exc:
            logger.warning(
                "reference_prune_failed sample_id={} error={} keeping_unpruned",
                sample.sample_id,
                f"{type(exc).__name__}: {exc}",
            )
        else:
            prune_floor = min(PRUNE_MIN_SURVIVORS, max(4, len(reference_qs) // 2))
            if len(kept_qs) < prune_floor:
                raise QuestionScoringUnavailable(
                    f"reference pruning left {len(kept_qs)}/{len(reference_qs)} questions"
                )
            pruned_info = {
                "reference_self_score": self_rate,
                "pruned_out": len(reference_qs) - len(kept_qs),
            }
            reference_qs = kept_qs
            discarded.extend(dropped_qs)
        behavior_qs: list[dict[str, str]] = []
        for index, r in enumerate(responses[1:]):
            qs, _ = parse_questions(r.raw, BEHAVIOR_K, discards=discarded, origin="behavior")
            part_name = BEHAVIOR_PHASES[phase].parts[index].name
            for q in qs:
                q["tag"] = f"behavior:{part_name}"
            behavior_qs.extend(qs)
        behavior_qs = filter_behavior_questions(behavior_qs, context_text, discards=discarded)
        for q in behavior_qs:
            q["requires"] = "action"
        questions = behavior_qs + reference_qs
        economy_duplicates: list[dict[str, str]] = []
        try:
            economy_duplicates = duplicate_economy_bounds(questions, reference)
            questions = questions + economy_duplicates
        except Exception:
            economy_duplicates = []
        for position, question in enumerate(questions, start=1):
            question["id"] = f"q_{position:02d}"
        source: dict[str, object] = {
            "provider": response.provider,
            "model": self.settings.evaluator_model,
            "n_questions": len(questions),
            "question_mode": "sota_anchored",
            "sample_phase": phase,
            "behavior_questions_kept": len(behavior_qs),
            "reference_made_edit": reference_made_edit,
            "enforcement_drops": drops,
            "reference_trajectory": reference,
            "economy_duplicate_bounds_added": len(economy_duplicates),
            "discarded_questions": discarded,
        }
        if reference_model:
            source["reference_model"] = reference_model
        source.update(pruned_info)
        return QuestionPrepResult(questions=questions, source=source)

    async def _prune_against_reference(
        self,
        sample: QuestionPrepSample | JudgeSample,
        reference: str,
        questions: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], float | None]:
        document = _reference_document(getattr(sample, "messages", None) or [], reference)
        _, recs = await _judge_side(
            client=self.client,
            settings=self.settings,
            side="reference",
            response_text=document,
            questions=questions,
            judge_models=[self.settings.evaluator_model],
            reference_made_edit=None,
        )
        record = recs[0] if recs else {}
        if not record.get("parse_ok"):
            raise RuntimeError("reference judge returned unparseable answers")
        answers = record.get("answers") or {}
        explanations = record.get("explanations") or {}
        kept: list[dict[str, str]] = []
        dropped: list[dict[str, str]] = []
        for q in questions:
            if str(answers.get(q["id"], "1")) == "1":
                kept.append(q)
            else:
                entry = {
                    "stage": "reference_prune",
                    "reason": "reference_answered_no",
                    "text": q["text"],
                    "origin": "content",
                }
                explanation = explanations.get(q["id"])
                if explanation:
                    entry["detail"] = explanation
                dropped.append(entry)
        return kept, dropped, record.get("yes_rate")


class RepoContextClient:
    def __init__(self, settings: JudgeSettings):
        self._client = httpx.AsyncClient(
            base_url=settings.repo_context_url.rstrip("/"),
            timeout=settings.repo_context_timeout_seconds,
        )
        self._last_warning = 0.0

    async def context_for(self, sample_id: str, assistant_output: str) -> str | None:
        try:
            response = await self._client.post(
                "/repo-context",
                json={"sample_id": sample_id, "assistant_output": assistant_output},
            )
            response.raise_for_status()
            context = response.json().get("context")
            return context if isinstance(context, str) and context else None
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_warning > 60.0:
                self._last_warning = now
                logger.warning(
                    "repo_context_unavailable sample_id={} error={}",
                    sample_id,
                    f"{type(exc).__name__}: {exc}",
                )
            return None

    async def aclose(self) -> None:
        await self._client.aclose()


class ObservationSimulationService:
    def __init__(
        self,
        settings: JudgeSettings,
        client: JudgeLLMClient,
        repo_context: RepoContextClient | None = None,
    ):
        self.settings = settings
        self.client = client
        self.repo_context = repo_context

    async def simulate(self, request: SimulateObservationRequest) -> str:
        command = first_bash_block(request.assistant_output)
        if not command:
            fmt = detect_format(request.sample_id, request.messages)
            logger.warning(
                "observation_simulation_no_command eval_run_id={} sample_id={} fmt={} chars={}",
                request.eval_run_id,
                request.sample_id,
                fmt,
                len(request.assistant_output or ""),
            )
            return missing_command_output(fmt)
        context_block = None
        if self.repo_context is not None:
            context_block = await self.repo_context.context_for(
                request.sample_id, request.assistant_output
            )
        transcript = _simulation_transcript(
            messages=request.messages,
            prompt=request.prompt,
            assistant_output=request.assistant_output,
        )
        fmt = detect_format(request.sample_id, request.messages)
        require_content = requires_output(command)
        contract = command_contract(command)
        primary = self.settings.simulation_model or self.settings.evaluator_model
        fallback_model = self.settings.evaluator_model
        attempts: list[tuple[str, int, dict[str, Any] | None]] = []
        if primary != fallback_model:
            sim_provider = _simulation_provider(self.settings)
            order = (sim_provider or {}).get("order") or []
            rungs = [
                {**sim_provider, "order": order[i:] + order[:i]} for i in range(len(order))
            ] or [sim_provider]
            attempts = [(primary, self.settings.simulation_loop_reruns + 1, rung) for rung in rungs]
            attempts.append((fallback_model, 1, _evaluator_provider(self.settings)))
        else:
            attempts = [
                (
                    primary,
                    self.settings.simulation_loop_reruns + 1,
                    _evaluator_provider(self.settings),
                )
            ]

        observation = ""
        best_rank = -1
        for model, tries, provider_block in attempts:
            capped = model == primary and primary != fallback_model
            messages = [
                {
                    "role": "system",
                    "content": simulation_system_prompt(fmt, context_block),
                },
                {"role": "user", "content": transcript},
            ]
            capped_kwargs = {"parse_retries": 2, "retry_count": 1} if capped else {}
            for attempt in range(tries):
                response = await self.client.complete(
                    purpose="simulate",
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    eval_run_id=request.eval_run_id,
                    max_tokens=self.settings.simulation_max_tokens,
                    provider=provider_block,
                    accept=lambda raw: _usable_simulation_output(
                        repair_to_contract(repair_output(raw, fmt), fmt, contract),
                        fmt,
                        require_content=require_content,
                        contract=contract,
                    ),
                    **capped_kwargs,
                )
                if response.error:
                    if model != fallback_model:
                        break
                    raise ObservationSimulationUnavailable(response.error)
                candidate = repair_to_contract(repair_output(response.raw, fmt), fmt, contract)
                rank = _candidate_rank(
                    candidate, fmt, require_content=require_content, contract=contract
                )
                if rank > best_rank:
                    best_rank, observation = rank, candidate
                if rank == _RANK_USABLE:
                    if model != primary:
                        logger.info(
                            "observation_simulation_fallback_used eval_run_id={} sample_id={} "
                            "primary={} fallback={}",
                            request.eval_run_id,
                            request.sample_id,
                            primary,
                            model,
                        )
                    break
                logger.warning(
                    "observation_simulation_unusable eval_run_id={} sample_id={} model={} "
                    "provider={} attempt={}/{} reason={} kept_rank={}",
                    request.eval_run_id,
                    request.sample_id,
                    model,
                    ((provider_block or {}).get("order") or ["auto"])[0],
                    attempt + 1,
                    tries,
                    _unusable_reason(
                        candidate, fmt, require_content=require_content, contract=contract
                    ),
                    best_rank,
                )
            if best_rank == _RANK_USABLE:
                break
        if _looping_output(observation):
            collapsed = _collapse_looping(observation).strip()
            logger.warning(
                "observation_simulation_looping_collapsed eval_run_id={} sample_id={} chars={}->{}",
                request.eval_run_id,
                request.sample_id,
                len(observation),
                len(collapsed),
            )
            observation = collapsed
        if not valid_output(observation, fmt):
            fallback = empty_output(fmt)
            logger.warning(
                "observation_simulation_invalid_format eval_run_id={} sample_id={} fmt={} "
                "fallback={!r}",
                request.eval_run_id,
                request.sample_id,
                fmt,
                fallback,
            )
            return fallback
        return observation


class QuestionPrepStore:
    def __init__(self, settings: JudgeSettings, service: QuestionService):
        self.settings = settings
        self.service = service
        self._preps: dict[str, dict[str, asyncio.Task[QuestionPrepResult]]] = {}
        self._created_at: dict[str, float] = {}

    def start(self, request: QuestionPrepRequest) -> str:
        self._sweep_expired()
        prep_id = f"{request.eval_run_id}:{uuid4()}"
        self._created_at[prep_id] = time.monotonic()
        self._preps[prep_id] = {
            sample.sample_id: asyncio.create_task(self._prepare_sample(prep_id, request, sample))
            for sample in request.samples
        }
        return prep_id

    async def get_with_reason(self, prep_id: str, sample: JudgeSample) -> QuestionPrepLookup:
        self._sweep_expired()
        tasks = self._preps.get(prep_id)
        if not tasks:
            return QuestionPrepLookup(None, "unknown_or_expired_prep_id")
        task = tasks.get(sample.sample_id)
        if task is None:
            return QuestionPrepLookup(None, "sample_not_in_prep")
        return QuestionPrepLookup(await task, "prepared")

    async def _prepare_sample(
        self, prep_id: str, request: QuestionPrepRequest, sample: QuestionPrepSample
    ) -> QuestionPrepResult:
        try:
            return await self.service.prepare(sample, eval_run_id=request.eval_run_id)
        except Exception as exc:
            logger.warning(
                "question_prep_sample_failed eval_run_id={} prep_id={} sample_id={} error={}",
                request.eval_run_id,
                prep_id,
                sample.sample_id,
                f"{type(exc).__name__}: {exc}",
            )
            raise

    def _sweep_expired(self) -> None:
        ttl = self.settings.question_prep_ttl_seconds
        now = time.monotonic()
        for prep_id in [pid for pid, created in self._created_at.items() if now - created > ttl]:
            for task in self._preps.get(prep_id, {}).values():
                if not task.done():
                    task.cancel()
            self._preps.pop(prep_id, None)
            self._created_at.pop(prep_id, None)


def create_app(settings: JudgeSettings | None = None) -> FastAPI:
    settings = settings or get_judge_settings()
    app = FastAPI(title="Albedo Judge API")

    @app.on_event("startup")
    async def startup() -> None:
        client = JudgeLLMClient(settings)
        app.state.eval_client = client
        repo_context = RepoContextClient(settings) if settings.repo_context_url else None
        app.state.repo_context_client = repo_context
        app.state.observation_service = ObservationSimulationService(settings, client, repo_context)
        app.state.question_service = QuestionService(
            settings,
            client,
            ReferenceTrajectoryService(settings, client, app.state.observation_service),
        )
        app.state.question_prep_store = QuestionPrepStore(settings, app.state.question_service)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        client = getattr(app.state, "eval_client", None)
        if client is not None:
            await client.aclose()
        repo_context = getattr(app.state, "repo_context_client", None)
        if repo_context is not None:
            await repo_context.aclose()

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not settings.api_auth_token:
            return
        if authorization != f"Bearer {settings.api_auth_token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    def prep_store() -> QuestionPrepStore:
        store = getattr(app.state, "question_prep_store", None)
        if store is None:
            client = JudgeLLMClient(settings)
            app.state.eval_client = client
            repo_context = RepoContextClient(settings) if settings.repo_context_url else None
            app.state.repo_context_client = repo_context
            app.state.observation_service = ObservationSimulationService(
                settings, client, repo_context
            )
            app.state.question_service = QuestionService(
                settings,
                client,
                ReferenceTrajectoryService(settings, client, app.state.observation_service),
            )
            app.state.question_prep_store = QuestionPrepStore(settings, app.state.question_service)
        return app.state.question_prep_store

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(_: None = Depends(require_auth)) -> dict[str, object]:
        return {
            "status": "ready",
            "judge_models": list(JUDGE_MODELS),
            "evaluator_model": settings.evaluator_model,
            "num_questions": settings.num_questions,
        }

    @app.post("/category-prep", response_model=QuestionPrepResponse)
    async def category_prep(
        request: QuestionPrepRequest, _: None = Depends(require_auth)
    ) -> QuestionPrepResponse:
        prep_id = prep_store().start(request)
        return QuestionPrepResponse(
            eval_run_id=request.eval_run_id,
            category_prep_id=prep_id,
            accepted_sample_count=len(request.samples),
        )

    @app.post("/simulate-observation", response_model=SimulateObservationResponse)
    async def simulate_observation(
        request: SimulateObservationRequest, _: None = Depends(require_auth)
    ) -> SimulateObservationResponse:
        service: ObservationSimulationService = app.state.observation_service
        observation = await service.simulate(request)
        return SimulateObservationResponse(
            eval_run_id=request.eval_run_id,
            sample_id=request.sample_id,
            observation=observation,
        )

    @app.post("/score-batch", response_model=ScoreBatchResponse)
    async def score_batch(
        request: ScoreBatchRequest, _: None = Depends(require_auth)
    ) -> ScoreBatchResponse:
        unknown = [model for model in request.judge_models if model not in JUDGE_MODELS]
        if unknown:
            raise HTTPException(
                status_code=400, detail=f"unsupported judge model(s): {', '.join(unknown)}"
            )
        client: JudgeLLMClient = app.state.eval_client
        try:
            records = await _score_samples(
                client=client, request=request, settings=settings, prep_store=prep_store()
            )
        except Exception as exc:
            _notify(
                settings,
                request,
                severity="ERROR",
                message="Scoring failed",
                fault_code="scoring_failed",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
            logger.exception(
                f"[judge-api] scoring failed eval_run={request.eval_run_id} batch={request.batch_id}: {exc}"  # noqa: E501
            )
            raise HTTPException(status_code=502, detail=f"scoring failed: {exc}")
        summary = aggregate_scores(records, min_valid_fraction=settings.min_valid_fraction)
        if summary.get("state") != "succeeded":
            _notify(
                settings,
                request,
                severity="WARNING",
                message="Scoring produced too few valid samples",
                fault_code=str(summary.get("fault_code") or "scoring_invalid"),
                retryable=bool(summary.get("retryable")),
            )
        return ScoreBatchResponse(
            eval_run_id=request.eval_run_id,
            batch_id=request.batch_id,
            scoring_records=records,
            summary=summary,
        )

    return app


async def _questions_for(
    request: ScoreBatchRequest, sample: JudgeSample, prep_store: QuestionPrepStore
) -> QuestionPrepResult:
    if request.category_prep_id:
        try:
            lookup = await prep_store.get_with_reason(request.category_prep_id, sample)
        except Exception as exc:
            reason = f"prep_failed:{type(exc).__name__}"
        else:
            if lookup.result is not None:
                return lookup.result
            reason = lookup.reason
    else:
        reason = "missing_prep_id"
    logger.warning(
        "score_batch_question_sync_generation eval_run_id={} batch_id={} sample_id={} reason={}",
        request.eval_run_id,
        request.batch_id,
        sample.sample_id,
        reason,
    )
    return await prep_store.service.prepare(sample)


_COMMAND_BLOCK_RE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.DOTALL)


def _command_only(text: str) -> str:
    match = _COMMAND_BLOCK_RE.search(text or "")
    if match:
        return f"```bash\n{match.group(1).strip()}\n```"
    return text


def _simulation_transcript(
    *,
    messages: list[dict[str, str]] | None,
    prompt: str,
    assistant_output: str,
) -> str:
    transcript_messages = messages or [{"role": "user", "content": prompt}]
    sections = []
    for message in transcript_messages + [{"role": "assistant", "content": assistant_output}]:
        role = str(message.get("role") or "user").lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(message.get("content") or "").rstrip()
        if role == "assistant":
            content = _command_only(content)
        sections.append(f"### {role}\n{content}")
    return "\n\n".join(sections).rstrip()


_LOOP_LINE_RUN = 25
_LOOP_TAIL_WINDOW = 512
_LOOP_MIN_REPEATS = 4


def _looping_output(text: str) -> bool:
    run = 1
    prev: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped == prev:
            run += 1
            if run >= _LOOP_LINE_RUN:
                return True
        elif stripped:
            run = 1
            prev = stripped
    return _trailing_cycle_period(text) > 0


def _trailing_cycle_period(text: str) -> int:
    tail = text.rstrip()[-_LOOP_TAIL_WINDOW:]
    if len(tail) < _LOOP_TAIL_WINDOW:
        return 0
    for period in range(1, _LOOP_TAIL_WINDOW // _LOOP_MIN_REPEATS + 1):
        if tail[period:] == tail[:-period]:
            return period
    return 0


def _collapse_looping(text: str) -> str:
    out: list[str] = []
    run = 1
    prev: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped == prev:
            run += 1
            if run == _LOOP_LINE_RUN:
                out.append("... (output repeats)")
            if run >= _LOOP_LINE_RUN:
                continue
        elif stripped:
            run = 1
            prev = stripped
        out.append(line)
    collapsed = "\n".join(out)
    period = _trailing_cycle_period(collapsed)
    if period:
        stripped_text = collapsed.rstrip()
        index = len(stripped_text) - period - 1
        while index >= 0 and stripped_text[index] == stripped_text[index + period]:
            index -= 1
        keep = min(len(stripped_text), index + 1 + 2 * period)
        collapsed = stripped_text[:keep].rstrip() + "\n... (output repeats)"
    return collapsed


_ROLE_LEAK_RE = re.compile(r"(?:^|\n)\s*(?:THOUGHT:|### (?:assistant|user|system)\b)")


def _role_violation(raw: str) -> bool:
    return bool(_ROLE_LEAK_RE.search(raw or ""))


_RANK_INVALID = 0
_RANK_VALID = 1
_RANK_HAS_CONTENT = 2
_RANK_USABLE = 3


def _candidate_rank(
    raw: str,
    fmt: str,
    *,
    require_content: bool = False,
    contract: CommandContract | None = None,
) -> int:
    """How good an attempt is, so escalation keeps the best one rather than the last.

    Escalating used to overwrite a usable primary result with whatever the fallback produced. In
    practice the fallback often answers in the wrong dialect, which then collapsed to an empty
    observation, or returned a worse contract violation — both strictly worse than the primary.
    """
    if not valid_output(raw, fmt):
        return _RANK_INVALID
    if _usable_simulation_output(raw, fmt, require_content=require_content, contract=contract):
        return _RANK_USABLE
    if has_content(raw, fmt):
        return _RANK_HAS_CONTENT
    return _RANK_VALID


def _usable_simulation_output(
    raw: str,
    fmt: str,
    *,
    require_content: bool = False,
    contract: CommandContract | None = None,
) -> bool:
    return (
        valid_output(raw, fmt)
        and not _role_violation(raw)
        and not _looping_output(raw)
        and (not require_content or has_content(raw, fmt))
        and (contract is None or contract_violation(raw, fmt, contract) is None)
    )


def _unusable_reason(
    raw: str,
    fmt: str,
    *,
    require_content: bool = False,
    contract: CommandContract | None = None,
) -> str:
    if not valid_output(raw, fmt):
        return "invalid_format"
    if _role_violation(raw):
        return "role_violation"
    if _looping_output(raw):
        return "looping"
    if require_content and not has_content(raw, fmt):
        return "no_content_for_read"
    if contract is not None and (breach := contract_violation(raw, fmt, contract)):
        return breach
    return "ok"


def _corrupted_side(
    *,
    side: str,
    questions: list[dict[str, str]],
    judge_models: list[str],
) -> tuple[dict[str, dict[str, str | None]], list[dict[str, Any]]]:
    per_judge_answers: dict[str, dict[str, str | None]] = {
        model: {q["id"]: "0" for q in questions} for model in judge_models
    }
    records = [
        {
            "side": side,
            "judge_model": model,
            "provider": None,
            "answers": per_judge_answers[model],
            "explanations": {},
            "yes_rate": judge_yes_rate(per_judge_answers[model], questions),
            "parse_ok": True,
            "error": None,
            "corrupted": True,
        }
        for model in judge_models
    ]
    return per_judge_answers, records


async def _judge_side(
    *,
    client: JudgeLLMClient,
    settings: JudgeSettings,
    side: str,
    response_text: str,
    questions: list[dict[str, str]],
    judge_models: list[str],
    reference_made_edit: bool | None = None,
) -> tuple[dict[str, dict[str, str | None]], list[dict[str, Any]]]:
    question_ids = [q["id"] for q in questions]
    schema = answer_schema(question_ids)
    messages = build_judge_messages(response=response_text, questions=questions)
    raws = await asyncio.gather(
        *[
            client.score(
                model=model,
                messages=messages,
                response_schema=schema,
                schema_name="albedo_answers",
                max_tokens=settings.answer_max_tokens,
                accept=lambda raw: parse_answers(raw, question_ids)[2],
            )
            for model in judge_models
        ]
    )
    per_judge_answers: dict[str, dict[str, str | None]] = {}
    records: list[dict[str, Any]] = []
    gate_turns = (
        candidate_turn_texts_from_merged(response_text) if reference_made_edit is not None else None
    )
    for raw, model in zip(raws, judge_models):
        answers, explanations, parse_ok = parse_answers(raw.raw, question_ids)
        if gate_turns is not None:
            answers = apply_measurement_gate(
                answers,
                questions,
                candidate_turn_texts=gate_turns,
                reference_made_edit=bool(reference_made_edit),
            )
        per_judge_answers[model] = answers
        records.append(
            {
                "side": side,
                "judge_model": model,
                "provider": raw.provider,
                "answers": answers,
                "explanations": explanations,
                "yes_rate": judge_yes_rate(answers, questions),
                "parse_ok": parse_ok and not raw.error,
                "error": raw.error,
            }
        )
    return per_judge_answers, records


async def _score_samples(
    *,
    client: JudgeLLMClient,
    request: ScoreBatchRequest,
    settings: JudgeSettings,
    prep_store: QuestionPrepStore,
) -> list[dict[str, Any]]:
    started_at = time.monotonic()
    completed = 0
    progress_lock = asyncio.Lock()
    logger.info(
        "score_batch_started eval_run_id={} batch_id={} samples={} judges={} prep_id={}",
        request.eval_run_id,
        request.batch_id,
        len(request.samples),
        len(request.judge_models),
        request.category_prep_id or "",
    )

    async def _score_one(sample: JudgeSample) -> dict[str, Any]:
        nonlocal completed
        try:
            return await _score_one_inner(sample)
        except Exception as exc:
            async with progress_lock:
                completed += 1
            logger.warning(
                "score_batch_sample_failed eval_run_id={} batch_id={} completed={}/{} sample_id={} error={}",  # noqa: E501
                request.eval_run_id,
                request.batch_id,
                completed,
                len(request.samples),
                sample.sample_id,
                f"{type(exc).__name__}: {exc}",
            )
            return {
                "sample_id": sample.sample_id,
                "questions": [],
                "king_score": None,
                "challenger_score": None,
                "judge_results": [],
                "scored": False,
                "scoring_mode": "binary",
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def _score_one_inner(sample: JudgeSample) -> dict[str, Any]:
        nonlocal completed
        prepared = await _questions_for(request, sample, prep_store)
        if prepared.error:
            raise QuestionScoringUnavailable(prepared.error)
        questions = prepared.questions
        gate_flag = prepared.source.get("reference_made_edit")
        gate_flag = bool(gate_flag) if gate_flag is not None else None
        if prepared.source.get("pruned_out") is not None:
            gate_flag = None  # pruning already calibrated the checklist to the reference

        async def _side(side: str, response_text: str):
            if is_truncated(response_text):
                return _corrupted_side(
                    side=side, questions=questions, judge_models=request.judge_models
                )
            return await _judge_side(
                client=client,
                settings=settings,
                side=side,
                response_text=response_text,
                questions=questions,
                judge_models=request.judge_models,
                reference_made_edit=gate_flag,
            )

        (king_answers, king_recs), (chal_answers, chal_recs) = await asyncio.gather(
            _side("previous_king", sample.previous_king_output),
            _side("challenger", sample.challenger_output),
        )
        king_score = response_score(king_answers, questions)
        chal_score = response_score(chal_answers, questions)
        king_ok = all(r["parse_ok"] for r in king_recs) and king_score is not None
        chal_ok = all(r["parse_ok"] for r in chal_recs) and chal_score is not None
        scored = king_ok and chal_ok
        async with progress_lock:
            completed += 1
            logger.info(
                "score_batch_sample_done eval_run_id={} batch_id={} completed={}/{} sample_id={} "
                "scored={} king={} chal={} elapsed_s={:.1f}",
                request.eval_run_id,
                request.batch_id,
                completed,
                len(request.samples),
                sample.sample_id,
                scored,
                king_score,
                chal_score,
                time.monotonic() - started_at,
            )
        return {
            "sample_id": sample.sample_id,
            "questions": questions,
            "king_score": king_score,
            "challenger_score": chal_score,
            "judge_results": king_recs + chal_recs,
            "scored": scored,
            "scoring_mode": "binary",
            "question_source": prepared.source,
        }

    records = await asyncio.gather(*[_score_one(sample) for sample in request.samples])
    logger.info(
        "score_batch_done eval_run_id={} batch_id={} scored={}/{} elapsed_s={:.1f}",
        request.eval_run_id,
        request.batch_id,
        sum(1 for r in records if r.get("scored")),
        len(records),
        time.monotonic() - started_at,
    )
    return list(records)


def _notify(
    settings: JudgeSettings,
    request: ScoreBatchRequest,
    *,
    severity: str,
    message: str,
    fault_code: str,
    retryable: bool | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    notify_eval_error(
        EvalErrorNotification(
            component="judge_api",
            severity=severity,
            message=message,
            eval_run_id=request.eval_run_id,
            batch_id=request.batch_id,
            fault_class="PROVIDER_FAULT",
            fault_code=fault_code,
            scoring_mode="binary",
            retryable=retryable,
            details=details,
        ),
        webhook_url=settings.slack_error_webhook_url,
    )


def main() -> None:
    settings = get_judge_settings()
    uvicorn.run(
        "albedo_eval_service.judge_api:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()

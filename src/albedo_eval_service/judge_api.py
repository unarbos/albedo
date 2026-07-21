from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from .judge_config import JudgeSettings, get_judge_settings
from .judge_core import (
    JUDGE_MODELS,
    aggregate_scores,
    answer_schema,
    build_judge_messages,
    build_question_messages,
    judge_yes_rate,
    parse_answers,
    parse_questions,
    question_schema,
    response_score,
)
from .judge_openrouter import OpenRouterJudgeClient
from .notifications import EvalErrorNotification, notify_eval_error


class QuestionPrepSample(BaseModel):
    sample_id: str
    prompt: str
    sample_index: int = 0  # retained for payload compatibility; scoring ignores order


class QuestionPrepRequest(BaseModel):
    eval_run_id: str
    batch_id: str = "category-prep"
    samples: list[QuestionPrepSample]
    total_sample_count: int


class QuestionPrepResponse(BaseModel):
    eval_run_id: str
    category_prep_id: str  # opaque id (name kept for control-plane compatibility)
    accepted_sample_count: int


class JudgeSample(BaseModel):
    sample_id: str
    prompt: str
    previous_king_output: str
    challenger_output: str
    sample_index: int = 0


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


BASE_PROMPT = """You are the ENVIRONMENT (execution harness) in a SWE-agent session. You are NOT the assistant and you must never act as the assistant.

You will receive a transcript with "### system", "### user" and "### assistant" section markers.
The transcript ends with the assistant's first message containing one command. Mentally execute
that command against the repository state implied by the task description and reply with the
environment's next message: the terminal output of that command.

STRICT RULES:
- Reply ONLY with the environment message in the exact format specified below — nothing else.
- NEVER write "THOUGHT:", never write a bash command, never write "### user" or "### assistant"
  headers, never use markdown code fences, never explain or comment. You are not solving the
  task; you are only the terminal returning the command's output.
- NEVER give task tips, hints, suggestions, next steps, encouragement, or any part of the
  solution. A terminal has no opinion: it only prints what the command outputs, even if the
  assistant is on the wrong track or asked a question.
- Emulate realistic tool behavior: sed -i, cp, mv, mkdir, rm print nothing on success; echo
  prints its argument; cat/sed -n print file content; grep -n prefixes matches with "NN:"
  (context lines with "NN-"); find/ls list paths one per line; failed commands print realistic
  error messages.
- If the assistant message contains MORE THAN ONE bash code block, only the FIRST block is
  executed — simulate the first command and ignore all later blocks.
- Respect pipe limits exactly: "| head -N" outputs at most N lines, "| tail -N" the last N.
  Count your output lines before replying.
- Anchor on evidence: file, directory and symbol names mentioned in the task description are
  real — build your output around them and the standard layout for the project's language.
  When you cannot infer paths with confidence, prefer FEWER lines over invented ones; if the
  command's filters plausibly match nothing in this project (e.g. a file extension foreign to
  its language), the output is empty.
"""

FORMAT_SWE_ZERO = """OUTPUT FORMAT:
- Your reply MUST begin with the literal string "Observation:" — no text may come before it.
- After "Observation: " write exactly the stdout/stderr the command would produce — nothing else.
- If the command would produce no output, reply with exactly "Observation:" and nothing more."""

FORMAT_MINI_CODER = """OUTPUT FORMAT:
- Your reply MUST have exactly this shape, with no text before or after:
<returncode>RC</returncode>
<output>
OUTPUT
</output>
  where RC is the command's exit code and OUTPUT is exactly the stdout/stderr it would produce
  (empty if the command prints nothing)."""


def _evaluator_provider(settings: JudgeSettings) -> dict[str, Any]:
    """Evaluator provider block: always fp8. With an `order` list, fallbacks are disabled — failover
    happens by rotating the order across retries (deterministic provenance, ~3x less draw-to-draw
    strictness wobble measured offline). Without a list, generic fallbacks stay on."""
    block: dict[str, Any] = {"allow_fallbacks": True, "quantizations": ["fp8"]}
    order = [p.strip() for p in settings.evaluator_providers.split(",") if p.strip()]
    if order:
        block["order"] = order
        block["allow_fallbacks"] = False
    return block


class QuestionService:
    """Generates the yes/no question set for one sample (task only), via OpenRouter glm-5.2."""

    def __init__(self, settings: JudgeSettings, client: OpenRouterJudgeClient):
        self.settings = settings
        self.client = client

    async def prepare(self, sample: QuestionPrepSample | JudgeSample) -> QuestionPrepResult:
        n = self.settings.num_questions
        response = await self.client.complete(
            model=self.settings.evaluator_model,
            messages=build_question_messages(task=sample.prompt, n=n),
            temperature=self.settings.temperature,
            max_tokens=self.settings.question_max_tokens,
            provider=_evaluator_provider(self.settings),
            response_schema=question_schema(n),
            accept=lambda raw: parse_questions(raw, n)[1],
        )
        if response.error:
            raise QuestionScoringUnavailable(response.error)
        questions, ok = parse_questions(response.raw, n)
        if not ok:
            raise QuestionScoringUnavailable(
                f"evaluator returned {len(questions)}/{n} well-formed questions"
            )
        return QuestionPrepResult(
            questions=questions,
            source={
                "provider": response.provider,
                "model": self.settings.evaluator_model,
                "n_questions": len(questions),
            },
        )


class ObservationSimulationService:
    def __init__(self, settings: JudgeSettings, client: OpenRouterJudgeClient):
        self.settings = settings
        self.client = client

    async def simulate(self, request: SimulateObservationRequest) -> str:
        response = await self.client.complete(
            model=self.settings.evaluator_model,
            messages=[
                {"role": "system", "content": _simulation_system_prompt(request.sample_id)},
                {
                    "role": "user",
                    "content": _simulation_transcript(
                        messages=request.messages,
                        prompt=request.prompt,
                        assistant_output=request.assistant_output,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=self.settings.simulation_max_tokens,
            provider=_evaluator_provider(self.settings),
            accept=lambda raw: _valid_simulation_output(raw, request.sample_id),
        )
        if response.error:
            raise ObservationSimulationUnavailable(response.error)
        observation = response.raw.strip()
        if not _valid_simulation_output(observation, request.sample_id):
            fallback = _empty_simulation_output(request.sample_id)
            logger.warning(
                "observation_simulation_invalid_format eval_run_id={} sample_id={} fallback={!r}",
                request.eval_run_id,
                request.sample_id,
                fallback,
            )
            return fallback
        return observation


class QuestionPrepStore:
    """Async per-sample question generation started at eval start (overlaps generation); scoring awaits it."""

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
            return await self.service.prepare(sample)
        except Exception as exc:
            logger.warning(
                "question_prep_sample_failed eval_run_id={} prep_id={} sample_id={} error={}",
                request.eval_run_id, prep_id, sample.sample_id, f"{type(exc).__name__}: {exc}",
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
        client = OpenRouterJudgeClient(settings)
        app.state.eval_client = client
        app.state.observation_service = ObservationSimulationService(settings, client)
        app.state.question_service = QuestionService(settings, client)
        app.state.question_prep_store = QuestionPrepStore(settings, app.state.question_service)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        client = getattr(app.state, "eval_client", None)
        if client is not None:
            await client.aclose()

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not settings.api_auth_token:
            return
        if authorization != f"Bearer {settings.api_auth_token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    def prep_store() -> QuestionPrepStore:
        store = getattr(app.state, "question_prep_store", None)
        if store is None:
            client = OpenRouterJudgeClient(settings)
            app.state.eval_client = client
            app.state.observation_service = ObservationSimulationService(settings, client)
            app.state.question_service = QuestionService(settings, client)
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
            raise HTTPException(status_code=400, detail=f"unsupported judge model(s): {', '.join(unknown)}")
        client: OpenRouterJudgeClient = app.state.eval_client
        try:
            records = await _score_samples(
                client=client, request=request, settings=settings, prep_store=prep_store()
            )
        except Exception as exc:
            _notify(
                settings, request, severity="ERROR",
                message="Scoring failed", fault_code="scoring_failed",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
            logger.exception(
                f"[judge-api] scoring failed eval_run={request.eval_run_id} batch={request.batch_id}: {exc}"
            )
            raise HTTPException(status_code=502, detail=f"scoring failed: {exc}")
        summary = aggregate_scores(records, min_valid_fraction=settings.min_valid_fraction)
        if summary.get("state") != "succeeded":
            _notify(
                settings, request, severity="WARNING",
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
        request.eval_run_id, request.batch_id, sample.sample_id, reason,
    )
    return await prep_store.service.prepare(sample)


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
        sections.append(f"### {role}\n{str(message.get('content') or '').rstrip()}")
    return "\n\n".join(sections).rstrip()


def _simulation_system_prompt(sample_id: str) -> str:
    return f"{BASE_PROMPT}\n{_simulation_format(sample_id)}"


def _simulation_format(sample_id: str) -> str:
    return FORMAT_MINI_CODER if "mini-coder" in sample_id.casefold() else FORMAT_SWE_ZERO


def _empty_simulation_output(sample_id: str) -> str:
    if _simulation_format(sample_id) == FORMAT_MINI_CODER:
        return "<returncode>0</returncode>\n<output>\n</output>"
    return "Observation:"


def _valid_simulation_output(raw: str, sample_id: str) -> bool:
    text = raw.strip()
    if _simulation_format(sample_id) == FORMAT_MINI_CODER:
        return (
            text.startswith("<returncode>")
            and "</returncode>" in text
            and "<output>\n" in text
            and text.endswith("\n</output>")
        )
    return text.startswith("Observation:")


async def _judge_side(
    *,
    client: OpenRouterJudgeClient,
    settings: JudgeSettings,
    side: str,
    response_text: str,
    questions: list[dict[str, str]],
    judge_models: list[str],
) -> tuple[dict[str, dict[str, str | None]], list[dict[str, Any]]]:
    """Score one trajectory (king or challenger) with all judges."""
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
    for raw, model in zip(raws, judge_models):
        answers, explanations, parse_ok = parse_answers(raw.raw, question_ids)
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
    client: OpenRouterJudgeClient,
    request: ScoreBatchRequest,
    settings: JudgeSettings,
    prep_store: QuestionPrepStore,
) -> list[dict[str, Any]]:
    started_at = time.monotonic()
    completed = 0
    progress_lock = asyncio.Lock()
    logger.info(
        "score_batch_started eval_run_id={} batch_id={} samples={} judges={} prep_id={}",
        request.eval_run_id, request.batch_id, len(request.samples),
        len(request.judge_models), request.category_prep_id or "",
    )

    async def _score_one(sample: JudgeSample) -> dict[str, Any]:
        nonlocal completed
        try:
            return await _score_one_inner(sample)
        except Exception as exc:  # one bad sample must not abort the whole batch
            async with progress_lock:
                completed += 1
            logger.warning(
                "score_batch_sample_failed eval_run_id={} batch_id={} completed={}/{} sample_id={} error={}",
                request.eval_run_id, request.batch_id, completed, len(request.samples),
                sample.sample_id, f"{type(exc).__name__}: {exc}",
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
        (king_answers, king_recs), (chal_answers, chal_recs) = await asyncio.gather(
            _judge_side(
                client=client, settings=settings, side="previous_king",
                response_text=sample.previous_king_output, questions=questions,
                judge_models=request.judge_models,
            ),
            _judge_side(
                client=client, settings=settings, side="challenger",
                response_text=sample.challenger_output, questions=questions,
                judge_models=request.judge_models,
            ),
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
                request.eval_run_id, request.batch_id, completed, len(request.samples),
                sample.sample_id, scored, king_score, chal_score, time.monotonic() - started_at,
            )
        return {
            "sample_id": sample.sample_id,
            "questions": questions,
            "king_score": king_score,
            "challenger_score": chal_score,
            "judge_results": king_recs + chal_recs,
            "scored": scored,
            "scoring_mode": "binary",
        }

    records = await asyncio.gather(*[_score_one(sample) for sample in request.samples])
    logger.info(
        "score_batch_done eval_run_id={} batch_id={} scored={}/{} elapsed_s={:.1f}",
        request.eval_run_id, request.batch_id,
        sum(1 for r in records if r.get("scored")), len(records), time.monotonic() - started_at,
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

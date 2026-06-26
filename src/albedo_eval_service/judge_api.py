from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .chutes_glm import GLMProviderClient
from .judge_config import JudgeSettings, get_judge_settings
from .judge_core import (
    JUDGE_MODELS,
    aggregate_scoring_records,
    build_candidate_response_messages,
    build_category_generation_messages,
    build_category_pairwise_messages,
    build_pairwise_messages,
    category_response_schema,
    parse_category_verdict,
    parse_metric_verdict,
    should_show_challenger_first,
    validate_category_payload,
)
from .judge_openrouter import OpenRouterJudgeClient
from .notifications import EvalErrorNotification, notify_eval_error


class JudgeSample(BaseModel):
    sample_id: str
    prompt: str
    previous_king_output: str
    challenger_output: str
    sample_index: int


class CategoryPrepSample(BaseModel):
    sample_id: str
    prompt: str
    sample_index: int


class CategoryPrepRequest(BaseModel):
    eval_run_id: str
    batch_id: str = "category-prep"
    samples: list[CategoryPrepSample]
    total_sample_count: int


class CategoryPrepResponse(BaseModel):
    eval_run_id: str
    category_prep_id: str
    accepted_sample_count: int


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


@dataclass(frozen=True)
class CategoryPrepResult:
    categories: list[dict[str, str]]
    category_hash: str
    category_source: dict[str, object]
    error: str | None = None


class CategoryScoringUnavailable(RuntimeError):
    pass


class GLMCategoryService:
    def __init__(self, settings: JudgeSettings, glm_client: GLMProviderClient):
        self.settings = settings
        self.glm_client = glm_client

    async def prepare(self, sample: CategoryPrepSample | JudgeSample) -> CategoryPrepResult:
        candidate = await self.glm_client.complete(
            messages=build_candidate_response_messages(context_prompt=sample.prompt),
            max_tokens=self.settings.glm_candidate_max_tokens,
            temperature=self.settings.glm_temperature,
        )
        if candidate.error:
            raise CategoryScoringUnavailable(candidate.error)
        category = await self.glm_client.complete(
            messages=build_category_generation_messages(
                context_prompt=sample.prompt,
                glm_response=candidate.raw,
                category_count=self.settings.category_count,
            ),
            max_tokens=self.settings.glm_max_tokens,
            temperature=self.settings.glm_temperature,
        )
        if category.error:
            raise CategoryScoringUnavailable(category.error)
        try:
            categories, digest = validate_category_payload(
                category.raw, expected_count=self.settings.category_count
            )
        except ValueError:
            repair = await self.glm_client.complete(
                messages=_repair_category_messages(
                    prompt=sample.prompt,
                    glm_response=candidate.raw,
                    bad_payload=category.raw,
                    category_count=self.settings.category_count,
                ),
                max_tokens=self.settings.glm_max_tokens,
                temperature=0.0,
            )
            if repair.error:
                raise CategoryScoringUnavailable(repair.error)
            categories, digest = validate_category_payload(
                repair.raw, expected_count=self.settings.category_count
            )
            category = repair
        return CategoryPrepResult(
            categories=categories,
            category_hash=digest,
            category_source={
                "provider": category.provider,
                "model": category.model,
                "primary_provider": "chutes",
                "fallback_provider": "openrouter-fp8",
                "chute_id": self.settings.glm52_chute_id,
                "prompt_version": self.settings.category_prompt_version,
            },
        )


class CategoryPrepStore:
    def __init__(self, settings: JudgeSettings, service: GLMCategoryService):
        self.settings = settings
        self.service = service
        self._preps: dict[str, dict[str, asyncio.Task[CategoryPrepResult]]] = {}
        self._created_at: dict[str, float] = {}

    def start(self, request: CategoryPrepRequest) -> str:
        self._sweep_expired()
        prep_id = f"{request.eval_run_id}:{uuid4()}"
        self._created_at[prep_id] = time.monotonic()
        self._preps[prep_id] = {
            sample.sample_id: asyncio.create_task(self.service.prepare(sample))
            for sample in request.samples
        }
        return prep_id

    async def get(self, prep_id: str, sample: JudgeSample) -> CategoryPrepResult | None:
        self._sweep_expired()
        tasks = self._preps.get(prep_id)
        if not tasks:
            return None
        task = tasks.get(sample.sample_id)
        if task is None:
            return None
        return await task

    def _sweep_expired(self) -> None:
        ttl = self.settings.category_prep_ttl_seconds
        now = time.monotonic()
        expired = [prep_id for prep_id, created in self._created_at.items() if now - created > ttl]
        for prep_id in expired:
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
        glm_client = GLMProviderClient(settings)
        app.state.glm_client = glm_client
        app.state.category_service = GLMCategoryService(settings, glm_client)
        app.state.category_prep_store = CategoryPrepStore(settings, app.state.category_service)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        glm_client = getattr(app.state, "glm_client", None)
        if glm_client is not None:
            await glm_client.aclose()

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not settings.api_auth_token:
            return
        expected = f"Bearer {settings.api_auth_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

    def prep_store() -> CategoryPrepStore:
        store = getattr(app.state, "category_prep_store", None)
        if store is None:
            glm_client = GLMProviderClient(settings)
            app.state.glm_client = glm_client
            app.state.category_service = GLMCategoryService(settings, glm_client)
            app.state.category_prep_store = CategoryPrepStore(settings, app.state.category_service)
        return app.state.category_prep_store

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(_: None = Depends(require_auth)) -> dict[str, object]:
        return {
            "status": "ready",
            "judge_models": list(JUDGE_MODELS),
            "strict_json": True,
            "counterbalanced": True,
            "category_prep": True,
            "glm_primary_provider": "chutes",
            "glm_fallback_provider": "openrouter-fp8",
        }

    @app.post("/category-prep", response_model=CategoryPrepResponse)
    async def category_prep(
        request: CategoryPrepRequest, _: None = Depends(require_auth)
    ) -> CategoryPrepResponse:
        prep_id = prep_store().start(request)
        return CategoryPrepResponse(
            eval_run_id=request.eval_run_id,
            category_prep_id=prep_id,
            accepted_sample_count=len(request.samples),
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
        async with OpenRouterJudgeClient(settings) as client:
            try:
                records = await _score_samples_with_categories(
                    client=client,
                    request=request,
                    category_service=prep_store().service,
                    prep_store=prep_store(),
                )
                summary = _summary(records, settings=settings)
                if summary.get("state") == "succeeded":
                    return ScoreBatchResponse(
                        eval_run_id=request.eval_run_id,
                        batch_id=request.batch_id,
                        scoring_records=records,
                        summary=summary,
                    )
                _notify(
                    settings,
                    request,
                    severity="WARNING",
                    message="GLM-category scoring produced too few valid scores; using fixed-metric fallback",
                    fault_code="glm_category_scoring_invalid",
                    scoring_mode="fixed_metrics_fallback",
                )
            except Exception as exc:
                _notify(
                    settings,
                    request,
                    severity="WARNING",
                    message="GLM-category scoring failed; using fixed-metric fallback",
                    fault_code="glm_category_scoring_failed",
                    scoring_mode="fixed_metrics_fallback",
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            records = await _score_samples(
                client=client, request=request, scoring_mode="fixed_metrics_fallback"
            )
        summary = _summary(records, settings=settings)
        if summary.get("state") != "succeeded":
            _notify(
                settings,
                request,
                severity="ERROR",
                message="Fixed-metric fallback scoring failed",
                fault_code=str(summary.get("fault_code") or "fixed_metric_fallback_failed"),
                scoring_mode="fixed_metrics_fallback",
                retryable=bool(summary.get("retryable")),
            )
        return ScoreBatchResponse(
            eval_run_id=request.eval_run_id,
            batch_id=request.batch_id,
            scoring_records=records,
            summary=summary,
        )

    return app


async def _score_samples(
    *,
    client: OpenRouterJudgeClient,
    request: ScoreBatchRequest,
    scoring_mode: str = "fixed_metrics",
) -> list[dict[str, Any]]:
    async def _score_one(index: int, sample: JudgeSample) -> dict[str, Any]:
        challenger_first = should_show_challenger_first(
            sample.sample_index, request.total_sample_count
        )
        challenger_position = 1 if challenger_first else 2
        messages = build_pairwise_messages(
            context_prompt=sample.prompt,
            previous_king_output=sample.previous_king_output,
            challenger_output=sample.challenger_output,
            challenger_first=challenger_first,
        )
        raws = await asyncio.gather(
            *[client.score(model=model, messages=messages) for model in request.judge_models],
        )
        judge_results = []
        for raw in raws:
            verdict = parse_metric_verdict(
                raw.raw,
                model=raw.model,
                provider=raw.provider,
                challenger_position=challenger_position,
                error=raw.error,
            )
            judge_results.append(_judge_result(verdict))
        ok = [result for result in judge_results if result["parse_ok"]]
        scored = len(ok) == len(request.judge_models)
        sample_score = mean(float(result["judge_mean"]) for result in ok) if scored else None
        return {
            "sample_id": sample.sample_id,
            "order": ["challenger", "previous_king"]
            if challenger_first
            else ["previous_king", "challenger"],
            "judge_results": judge_results,
            "judge_scores": [
                result["judge_mean"] for result in judge_results if result["parse_ok"]
            ],
            "sample_score": sample_score,
            "scored": scored,
            "scoring_mode": scoring_mode,
        }

    return await asyncio.gather(
        *[_score_one(index, sample) for index, sample in enumerate(request.samples)]
    )


async def _score_samples_with_categories(
    *,
    client: OpenRouterJudgeClient,
    request: ScoreBatchRequest,
    category_service: GLMCategoryService,
    prep_store: CategoryPrepStore,
) -> list[dict[str, Any]]:
    async def _categories_for(sample: JudgeSample) -> CategoryPrepResult:
        if request.category_prep_id:
            prepared = await prep_store.get(request.category_prep_id, sample)
            if prepared is not None:
                return prepared
        return await category_service.prepare(sample)

    async def _score_one(index: int, sample: JudgeSample) -> dict[str, Any]:
        prepared = await _categories_for(sample)
        if prepared.error:
            raise CategoryScoringUnavailable(prepared.error)
        challenger_first = should_show_challenger_first(
            sample.sample_index, request.total_sample_count
        )
        challenger_position = 1 if challenger_first else 2
        messages = build_category_pairwise_messages(
            context_prompt=sample.prompt,
            previous_king_output=sample.previous_king_output,
            challenger_output=sample.challenger_output,
            challenger_first=challenger_first,
            categories=prepared.categories,
        )
        response_schema = category_response_schema(prepared.categories)
        raws = await asyncio.gather(
            *[
                client.score(
                    model=model,
                    messages=messages,
                    response_schema=response_schema,
                    schema_name="albedo_pairwise_category_verdict",
                )
                for model in request.judge_models
            ],
        )
        judge_results = []
        for raw in raws:
            verdict = parse_category_verdict(
                raw.raw,
                categories=prepared.categories,
                model=raw.model,
                provider=raw.provider,
                challenger_position=challenger_position,
                error=raw.error,
            )
            judge_results.append(_judge_result(verdict))
        ok = [result for result in judge_results if result["parse_ok"]]
        scored = len(ok) == len(request.judge_models)
        sample_score = mean(float(result["judge_mean"]) for result in ok) if scored else None
        return {
            "sample_id": sample.sample_id,
            "order": ["challenger", "previous_king"]
            if challenger_first
            else ["previous_king", "challenger"],
            "categories": prepared.categories,
            "category_source": prepared.category_source,
            "category_hash": prepared.category_hash,
            "judge_results": judge_results,
            "judge_scores": [
                result["judge_mean"] for result in judge_results if result["parse_ok"]
            ],
            "sample_score": sample_score,
            "scored": scored,
            "scoring_mode": "glm_categories",
        }

    return await asyncio.gather(
        *[_score_one(index, sample) for index, sample in enumerate(request.samples)]
    )


def _summary(records: list[dict[str, Any]], *, settings: JudgeSettings) -> dict[str, Any]:
    summary = aggregate_scoring_records(records, min_valid_fraction=settings.min_valid_fraction)
    summary["judge_errors"] = sum(
        1 for record in records for result in record["judge_results"] if not result["parse_ok"]
    )
    summary["scored_sample_count"] = sum(1 for record in records if record.get("scored"))
    if records:
        summary.setdefault("scoring_mode", records[0].get("scoring_mode"))
    return summary


def _judge_result(verdict) -> dict[str, Any]:
    return {
        "judge_model": verdict.model,
        "provider": verdict.provider,
        "metric_scores": verdict.metric_scores,
        "judge_mean": verdict.judge_mean,
        "parse_ok": verdict.parse_ok,
        "raw_verdict": verdict.raw,
        "error": verdict.error,
    }


def _repair_category_messages(
    *, prompt: str, glm_response: str, bad_payload: str, category_count: int
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Repair malformed category JSON. Return strict JSON only."},
        {
            "role": "user",
            "content": (
                f"The category payload below is invalid. Return exactly {category_count} categories "
                "with ids cat_01 through cat_05 and fields name, description, scoring_guidance.\n\n"
                f"CONVERSATION:\n{prompt}\n\nGLM RESPONSE:\n{glm_response}\n\nBAD PAYLOAD:\n{bad_payload}"
            ),
        },
    ]


def _notify(
    settings: JudgeSettings,
    request: ScoreBatchRequest,
    *,
    severity: str,
    message: str,
    fault_code: str,
    scoring_mode: str,
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
            scoring_mode=scoring_mode,
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

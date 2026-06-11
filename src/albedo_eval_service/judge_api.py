from __future__ import annotations

import asyncio
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .judge_config import JudgeSettings, get_judge_settings
from .judge_core import (
    JUDGE_MODELS,
    aggregate_scoring_records,
    build_pairwise_messages,
    parse_metric_verdict,
    should_show_challenger_first,
)
from .judge_openrouter import OpenRouterJudgeClient


class JudgeSample(BaseModel):
    sample_id: str
    prompt: str
    previous_king_output: str
    challenger_output: str
    sample_index: int


class ScoreBatchRequest(BaseModel):
    eval_run_id: str
    batch_id: str
    samples: list[JudgeSample]
    total_sample_count: int
    judge_models: list[str] = Field(default_factory=lambda: list(JUDGE_MODELS))


class ScoreBatchResponse(BaseModel):
    eval_run_id: str
    batch_id: str
    scoring_records: list[dict[str, Any]]
    summary: dict[str, Any]


def create_app(settings: JudgeSettings | None = None) -> FastAPI:
    settings = settings or get_judge_settings()
    app = FastAPI(title="Albedo Judge API")

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not settings.api_auth_token:
            return
        expected = f"Bearer {settings.api_auth_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

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
        }

    @app.post("/score-batch", response_model=ScoreBatchResponse)
    async def score_batch(request: ScoreBatchRequest, _: None = Depends(require_auth)) -> ScoreBatchResponse:
        unknown = [model for model in request.judge_models if model not in JUDGE_MODELS]
        if unknown:
            raise HTTPException(status_code=400, detail=f"unsupported judge model(s): {', '.join(unknown)}")
        async with OpenRouterJudgeClient(settings) as client:
            records = await _score_samples(client=client, request=request)
        summary = aggregate_scoring_records(records, min_valid_fraction=settings.min_valid_fraction)
        summary["judge_errors"] = sum(1 for record in records for result in record["judge_results"] if not result["parse_ok"])
        summary["scored_sample_count"] = sum(1 for record in records if record.get("scored"))
        return ScoreBatchResponse(
            eval_run_id=request.eval_run_id,
            batch_id=request.batch_id,
            scoring_records=records,
            summary=summary,
        )

    return app


async def _score_samples(*, client: OpenRouterJudgeClient, request: ScoreBatchRequest) -> list[dict[str, Any]]:
    async def _score_one(index: int, sample: JudgeSample) -> dict[str, Any]:
        challenger_first = should_show_challenger_first(sample.sample_index, request.total_sample_count)
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
            judge_results.append(
                {
                    "judge_model": verdict.model,
                    "provider": verdict.provider,
                    "metric_scores": verdict.metric_scores,
                    "judge_mean": verdict.judge_mean,
                    "parse_ok": verdict.parse_ok,
                    "raw_verdict": verdict.raw,
                    "error": verdict.error,
                }
            )
        ok = [result for result in judge_results if result["parse_ok"]]
        scored = len(ok) == len(request.judge_models)
        sample_score = sum(float(result["judge_mean"]) for result in ok) / len(ok) if scored else None
        return {
            "sample_id": sample.sample_id,
            "order": ["challenger", "previous_king"] if challenger_first else ["previous_king", "challenger"],
            "judge_results": judge_results,
            "judge_scores": [result["judge_mean"] for result in judge_results if result["parse_ok"]],
            "sample_score": sample_score,
            "scored": scored,
        }

    return await asyncio.gather(*[_score_one(index, sample) for index, sample in enumerate(request.samples)])


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

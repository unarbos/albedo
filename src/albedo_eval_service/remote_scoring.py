from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .judge_core import JUDGE_MODELS, aggregate_scoring_records, should_show_challenger_first
from .models import EvalRequest
from .remote_config import RemoteSettings
from .remote_dataset import EvalSample
from .remote_generation import GenerationResult
from .score_bridge import score_bridge_hub


@dataclass(frozen=True)
class ScoringResult:
    records: list[dict[str, Any]]
    summary: dict[str, Any]


class Scorer(Protocol):
    def start_category_prep(
        self, *, request: EvalRequest, samples: list[EvalSample]
    ) -> str | None:
        ...

    def score(
        self,
        *,
        request: EvalRequest,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        category_prep_id: str | None = None,
    ) -> ScoringResult:
        ...


class HttpScoringClient:
    def __init__(self, settings: RemoteSettings):
        if not settings.scoring_base_url:
            raise ValueError("ALBEDO_REMOTE_SCORING_BASE_URL is required when scoring backend is http")
        if not settings.scoring_auth_token:
            raise ValueError("ALBEDO_REMOTE_SCORING_AUTH_TOKEN is required when scoring backend is http")
        self.settings = settings

    def start_category_prep(
        self, *, request: EvalRequest, samples: list[EvalSample]
    ) -> str | None:
        with httpx.Client(
            base_url=self.settings.scoring_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.settings.scoring_auth_token}"},
            timeout=httpx.Timeout(self.settings.scoring_timeout_seconds),
        ) as client:
            response = client.post(
                "/category-prep",
                json=_category_prep_payload(request, samples),
            )
            response.raise_for_status()
            body = response.json()
            value = body.get("category_prep_id")
            return value if isinstance(value, str) and value else None

    def score(
        self,
        *,
        request: EvalRequest,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        category_prep_id: str | None = None,
    ) -> ScoringResult:
        all_records: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        with httpx.Client(
            base_url=self.settings.scoring_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.settings.scoring_auth_token}"},
            timeout=httpx.Timeout(self.settings.scoring_timeout_seconds),
        ) as client:
            for payload in _score_batch_payloads(
                request, samples, king_results, challenger_results, category_prep_id=category_prep_id
            ):
                response = client.post("/score-batch", json=payload)
                response.raise_for_status()
                body = response.json()
                records = body.get("scoring_records", [])
                if not isinstance(records, list):
                    raise ValueError("judge API returned non-list scoring_records")
                all_records.extend(records)
                summary = body.get("summary", {})
                if isinstance(summary, dict):
                    summaries.append(summary)
        return ScoringResult(
            records=all_records,
            summary=_merge_summaries(all_records, summaries, min_valid_fraction=self.settings.scoring_min_valid_fraction),
        )


class WebSocketScoringClient:
    def __init__(self, settings: RemoteSettings):
        self.settings = settings

    def start_category_prep(
        self, *, request: EvalRequest, samples: list[EvalSample]
    ) -> str | None:
        body = score_bridge_hub.request(
            _category_prep_payload(request, samples),
            timeout_seconds=self.settings.scoring_timeout_seconds,
            endpoint="/category-prep",
        )
        value = body.get("category_prep_id")
        return value if isinstance(value, str) and value else None

    def score(
        self,
        *,
        request: EvalRequest,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        category_prep_id: str | None = None,
    ) -> ScoringResult:
        all_records: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for payload in _score_batch_payloads(
            request, samples, king_results, challenger_results, category_prep_id=category_prep_id
        ):
            body = score_bridge_hub.request(payload, timeout_seconds=self.settings.scoring_timeout_seconds)
            records = body.get("scoring_records", [])
            if not isinstance(records, list):
                raise ValueError("score bridge returned non-list scoring_records")
            all_records.extend(records)
            summary = body.get("summary", {})
            if isinstance(summary, dict):
                summaries.append(summary)
        return ScoringResult(
            records=all_records,
            summary=_merge_summaries(all_records, summaries, min_valid_fraction=self.settings.scoring_min_valid_fraction),
        )


class MockScoringClient:
    """Small deterministic scorer retained for unit tests and control-plane smoke runs only."""

    def __init__(self, settings: RemoteSettings):
        self.settings = settings

    def start_category_prep(
        self, *, request: EvalRequest, samples: list[EvalSample]
    ) -> str | None:
        return None

    def score(
        self,
        *,
        request: EvalRequest,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        category_prep_id: str | None = None,
    ) -> ScoringResult:
        king_by_id = {result.sample_id: result for result in king_results}
        challenger_by_id = {result.sample_id: result for result in challenger_results}
        records: list[dict[str, Any]] = []
        judge_models = list(JUDGE_MODELS[: request.scoring.judge_count])
        for index, sample in enumerate(samples):
            king = king_by_id[sample.sample_id]
            challenger = challenger_by_id[sample.sample_id]
            if king.error or challenger.error:
                continue
            score = 1.0 if len(challenger.text) > len(king.text) else 0.5 if len(challenger.text) == len(king.text) else 0.0
            order = ["challenger", "previous_king"] if should_show_challenger_first(index, len(samples)) else ["previous_king", "challenger"]
            judge_results = [
                {
                    "judge_model": model,
                    "provider": "mock",
                    "metric_scores": {metric: score for metric in ("correctness", "grounding", "progress", "protocol", "efficiency")},
                    "judge_mean": score,
                    "parse_ok": True,
                    "raw_verdict": "{}",
                    "error": None,
                }
                for model in judge_models
            ]
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "order": order,
                    "judge_results": judge_results,
                    "judge_scores": [score] * len(judge_models),
                    "sample_score": score,
                    "scored": True,
                    "scoring_mode": "mock",
                }
            )
        return ScoringResult(
            records=records,
            summary=aggregate_scoring_records(records, min_valid_fraction=self.settings.scoring_min_valid_fraction),
        )


def build_scorer(settings: RemoteSettings) -> Scorer:
    if settings.scoring_backend == "http":
        return HttpScoringClient(settings)
    if settings.scoring_backend == "websocket":
        return WebSocketScoringClient(settings)
    if settings.scoring_backend == "mock":
        return MockScoringClient(settings)
    raise ValueError(f"unsupported scoring backend: {settings.scoring_backend}")


def _category_prep_payload(request: EvalRequest, samples: list[EvalSample]) -> dict[str, Any]:
    return {
        "eval_run_id": str(request.eval_run_id),
        "batch_id": "category-prep",
        "total_sample_count": len(samples),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "prompt": sample.prompt,
                "sample_index": index,
            }
            for index, sample in enumerate(samples)
        ],
    }


def _score_batch_payloads(
    request: EvalRequest,
    samples: list[EvalSample],
    king_results: list[GenerationResult],
    challenger_results: list[GenerationResult],
    *,
    category_prep_id: str | None = None,
) -> list[dict[str, Any]]:
    king_by_id = {result.sample_id: result for result in king_results}
    challenger_by_id = {result.sample_id: result for result in challenger_results}
    valid_samples = [
        (index, sample)
        for index, sample in enumerate(samples)
        if sample.sample_id in king_by_id
        and sample.sample_id in challenger_by_id
        and not king_by_id[sample.sample_id].error
        and not challenger_by_id[sample.sample_id].error
    ]
    payloads = []
    for batch_idx, batch in enumerate(_chunks(valid_samples, request.dataset.scoring_batch_size), start=1):
        payloads.append(
            {
                "eval_run_id": str(request.eval_run_id),
                "batch_id": f"score-{batch_idx:04d}",
                "judge_models": list(JUDGE_MODELS[: request.scoring.judge_count]),
                "total_sample_count": len(samples),
                "category_prep_id": category_prep_id,
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "prompt": sample.prompt,
                        "previous_king_output": king_by_id[sample.sample_id].text,
                        "challenger_output": challenger_by_id[sample.sample_id].text,
                        "sample_index": sample_index,
                    }
                    for sample_index, sample in batch
                ],
            }
        )
    return payloads


def _merge_summaries(
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    min_valid_fraction: float,
) -> dict[str, Any]:
    summary = aggregate_scoring_records(records, min_valid_fraction=min_valid_fraction)
    if summaries:
        summary["batch_summaries"] = summaries
    return summary


def _chunks(items: list, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]

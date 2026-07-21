from __future__ import annotations

import email.utils
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from .judge_core import JUDGE_MODELS, aggregate_scores
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

    def simulate_observation(
        self, *, request: EvalRequest, sample: EvalSample, assistant_output: str
    ) -> str:
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
            body = _post_json_with_429_retry(
                client,
                "/category-prep",
                _category_prep_payload(request, samples),
                retry_count=self.settings.scoring_retry_count,
                base_backoff_seconds=self.settings.scoring_retry_backoff_seconds,
            )
            value = body.get("category_prep_id")
            return value if isinstance(value, str) and value else None

    def simulate_observation(
        self, *, request: EvalRequest, sample: EvalSample, assistant_output: str
    ) -> str:
        with httpx.Client(
            base_url=self.settings.scoring_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.settings.scoring_auth_token}"},
            timeout=httpx.Timeout(self.settings.scoring_timeout_seconds),
        ) as client:
            body = _post_json_with_429_retry(
                client,
                "/simulate-observation",
                _simulate_observation_payload(request, sample, assistant_output),
                retry_count=self.settings.scoring_retry_count,
                base_backoff_seconds=self.settings.scoring_retry_backoff_seconds,
            )
            value = body.get("observation")
            if not isinstance(value, str):
                raise ValueError("simulation backend returned non-string observation")
            return value

    def score(
        self,
        *,
        request: EvalRequest,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        category_prep_id: str | None = None,
    ) -> ScoringResult:
        payloads = _score_batch_payloads(
            request, samples, king_results, challenger_results, category_prep_id=category_prep_id
        )
        with httpx.Client(
            base_url=self.settings.scoring_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.settings.scoring_auth_token}"},
            timeout=httpx.Timeout(self.settings.scoring_timeout_seconds),
        ) as client:

            def send(payload: dict[str, Any]) -> dict[str, Any]:
                return _post_json_with_429_retry(
                    client,
                    "/score-batch",
                    payload,
                    retry_count=self.settings.scoring_retry_count,
                    base_backoff_seconds=self.settings.scoring_retry_backoff_seconds,
                )

            all_records, summaries = _collect_score_batches(
                payloads, send, max_concurrency=self.settings.scoring_batch_concurrency
            )
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

    def simulate_observation(
        self, *, request: EvalRequest, sample: EvalSample, assistant_output: str
    ) -> str:
        body = score_bridge_hub.request(
            _simulate_observation_payload(request, sample, assistant_output),
            timeout_seconds=self.settings.scoring_timeout_seconds,
            endpoint="/simulate-observation",
        )
        value = body.get("observation")
        if not isinstance(value, str):
            raise ValueError("simulation backend returned non-string observation")
        return value

    def score(
        self,
        *,
        request: EvalRequest,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        category_prep_id: str | None = None,
    ) -> ScoringResult:
        payloads = _score_batch_payloads(
            request, samples, king_results, challenger_results, category_prep_id=category_prep_id
        )

        def send(payload: dict[str, Any]) -> dict[str, Any]:
            return score_bridge_hub.request(payload, timeout_seconds=self.settings.scoring_timeout_seconds)

        all_records, summaries = _collect_score_batches(
            payloads, send, max_concurrency=self.settings.scoring_batch_concurrency
        )
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

    def simulate_observation(
        self, *, request: EvalRequest, sample: EvalSample, assistant_output: str
    ) -> str:
        return "Observation:"

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
        judge_models = list(JUDGE_MODELS[: request.scoring.judge_count])
        records: list[dict[str, Any]] = []
        for sample in samples:
            king = king_by_id[sample.sample_id]
            challenger = challenger_by_id[sample.sample_id]
            if king.error or challenger.error:
                continue
            chal_score = (
                1.0 if len(challenger.text) > len(king.text)
                else 0.5 if len(challenger.text) == len(king.text)
                else 0.0
            )
            king_score = 1.0 - chal_score
            judge_results = [
                {
                    "side": side,
                    "judge_model": model,
                    "provider": "mock",
                    "answers": {},
                    "explanations": {},
                    "yes_rate": rate,
                    "parse_ok": True,
                    "error": None,
                }
                for side, rate in (("previous_king", king_score), ("challenger", chal_score))
                for model in judge_models
            ]
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "questions": [],
                    "king_score": king_score,
                    "challenger_score": chal_score,
                    "judge_results": judge_results,
                    "scored": True,
                    "scoring_mode": "mock",
                }
            )
        return ScoringResult(
            records=records,
            summary=aggregate_scores(records, min_valid_fraction=self.settings.scoring_min_valid_fraction),
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
            {"sample_id": sample.sample_id, "prompt": sample.prompt} for sample in samples
        ],
    }


def _simulate_observation_payload(
    request: EvalRequest, sample: EvalSample, assistant_output: str
) -> dict[str, Any]:
    return {
        "eval_run_id": str(request.eval_run_id),
        "sample_id": sample.sample_id,
        "prompt": sample.prompt,
        "messages": sample.messages,
        "assistant_output": assistant_output,
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
        sample
        for sample in samples
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
                    }
                    for sample in batch
                ],
            }
        )
    return payloads


def _collect_score_batches(
    payloads: list[dict[str, Any]],
    send: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    max_concurrency: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Send score batches concurrently; records/summaries keep the payload order.

    Keep max_concurrency * scoring_batch_size * 2 within ~2x the judge API's
    per-model semaphore, or per-batch latency can exceed the scoring timeout.
    """
    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    if not payloads:
        return all_records, summaries
    workers = max(1, min(max_concurrency, len(payloads)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        bodies = list(executor.map(send, payloads))
    for body in bodies:
        records = body.get("scoring_records", [])
        if not isinstance(records, list):
            raise ValueError("scoring backend returned non-list scoring_records")
        all_records.extend(records)
        summary = body.get("summary", {})
        if isinstance(summary, dict):
            summaries.append(summary)
    return all_records, summaries


def _post_json_with_429_retry(
    client: httpx.Client,
    endpoint: str,
    payload: dict[str, Any],
    *,
    retry_count: int,
    base_backoff_seconds: float,
) -> dict[str, Any]:
    for attempt in range(retry_count + 1):
        response = client.post(endpoint, json=payload)
        if response.status_code != 429 or attempt >= retry_count:
            response.raise_for_status()
            return response.json()
        time.sleep(_retry_sleep_seconds(response, attempt, base_backoff_seconds))
    raise AssertionError("unreachable")


def _retry_sleep_seconds(
    response: httpx.Response,
    attempt: int,
    base_backoff_seconds: float,
) -> float:
    retry_after = _retry_after_seconds(response.headers.get("retry-after"))
    backoff = base_backoff_seconds * (2**attempt) * random.uniform(0.8, 1.2)
    return max(retry_after, backoff)


def _retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return 0.0
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _merge_summaries(
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    min_valid_fraction: float,
) -> dict[str, Any]:
    summary = aggregate_scores(records, min_valid_fraction=min_valid_fraction)
    if summaries:
        summary["batch_summaries"] = summaries
    return summary


def _chunks(items: list, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]

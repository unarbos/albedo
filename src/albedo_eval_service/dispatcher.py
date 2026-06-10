from __future__ import annotations

import argparse
import asyncio
from typing import Any
from uuid import UUID

import httpx

from .config import Settings, get_settings
from .faults import broken_stream_fault, classify_failure_verdict
from .models import Challenger, DatasetConfig, EvalRequest, PreviousKing, ScoringConfig
from .remote_client import RemoteEvalClient
from .repository import ClaimedEval, EvalRepository


def build_eval_request(
    settings: Settings,
    submission: dict[str, Any],
    king: dict[str, Any],
    eval_run_id: UUID,
) -> EvalRequest:
    artifact_prefix = (
        f"{settings.artifact_prefix.rstrip('/')}/submissions/"
        f"{submission['id']}/eval/{eval_run_id}"
    )
    sample_ids = submission.get("dataset_sample_ids") or []
    return EvalRequest(
        eval_run_id=eval_run_id,
        submission_id=submission["id"],
        challenger=Challenger(
            model_uri=submission["model_uri"],
            model_hash=submission["model_hash"],
        ),
        previous_king=PreviousKing(
            model_uri=king["model_uri"],
            model_hash=king["model_hash"],
            king_version=king["king_version"],
        ),
        dataset=DatasetConfig(
            version=settings.dataset_version,
            manifest_uri=settings.dataset_manifest_uri,
            manifest_hash=settings.dataset_manifest_hash,
            sample_count=settings.sample_count,
            max_turns_per_sample=settings.max_turns_per_sample,
            sample_seed=submission["block_hash"],
            sampling_algo=settings.sampling_algo,
            sample_ids=sample_ids,
        ),
        scoring=ScoringConfig(
            judge_config_hash=settings.judge_config_hash,
            judge_count=settings.judge_count,
        ),
        artifact_prefix=artifact_prefix,
    )


class EvalDispatcher:
    def __init__(self, *, settings: Settings, repository: EvalRepository):
        self.settings = settings
        self.repository = repository

    def claim_once(self) -> ClaimedEval | None:
        return self.repository.claim_next_eval(
            worker_id=self.settings.worker_id,
            lease_seconds=self.settings.lease_seconds,
            request_builder=lambda submission, king, _host, eval_run_id: build_eval_request(
                self.settings,
                submission,
                king,
                eval_run_id,
            ),
        )

    async def dispatch_once(self) -> bool:
        claimed = self.claim_once()
        if not claimed:
            return False

        client = RemoteEvalClient(
            base_url=claimed.remote_host.base_url,
            auth_token=self.settings.remote_auth_token,
            timeout_seconds=self.settings.remote_event_timeout_seconds,
        )
        try:
            await client.ready()
            start_response = await client.start_eval(claimed.request)
            remote_run_id = str(start_response.get("remote_run_id") or claimed.eval_run_id)
            verdict = await self._follow_until_verdict(client, claimed, remote_run_id)
            if verdict.get("state") == "succeeded":
                self.repository.mark_eval_succeeded(
                    submission_id=claimed.submission_id,
                    attempt_id=claimed.attempt_id,
                    eval_run_id=claimed.eval_run_id,
                    verdict=verdict,
                )
            else:
                fault = classify_failure_verdict(verdict)
                self.repository.mark_eval_failed(
                    submission_id=claimed.submission_id,
                    attempt_id=claimed.attempt_id,
                    eval_run_id=claimed.eval_run_id,
                    fault_class=fault.fault_class,
                    fault_code=fault.fault_code,
                    fault_message=fault.fault_message,
                    retryable=fault.retryable,
                )
            return True
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            fault = broken_stream_fault(str(exc))
            self.repository.mark_eval_failed(
                submission_id=claimed.submission_id,
                attempt_id=claimed.attempt_id,
                eval_run_id=claimed.eval_run_id,
                fault_class=fault.fault_class,
                fault_code=fault.fault_code,
                fault_message=fault.fault_message,
                retryable=fault.retryable,
            )
            return True
        finally:
            await client.aclose()

    async def _follow_until_verdict(
        self,
        client: RemoteEvalClient,
        claimed: ClaimedEval,
        remote_run_id: str,
    ) -> dict[str, Any]:
        async for event in client.iter_events(remote_run_id):
            self.repository.record_remote_event(
                submission_id=claimed.submission_id,
                attempt_id=claimed.attempt_id,
                event=event,
            )
            if event.get("type") == "verdict":
                return event

        remote_state = await client.get_eval(remote_run_id)
        if remote_state.get("type") == "verdict" or remote_state.get("state") in {"succeeded", "failed"}:
            return remote_state
        raise asyncio.TimeoutError("remote event replay ended before final verdict")

    async def run_forever(self) -> None:
        while True:
            did_work = await self.dispatch_once()
            if not did_work:
                await asyncio.sleep(self.settings.dispatch_poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Albedo eval dispatcher.")
    parser.add_argument("--once", action="store_true", help="Claim and dispatch at most one eval.")
    args = parser.parse_args()

    settings = get_settings()
    dispatcher = EvalDispatcher(
        settings=settings,
        repository=EvalRepository(settings.database_url),
    )
    if args.once:
        asyncio.run(dispatcher.dispatch_once())
    else:
        asyncio.run(dispatcher.run_forever())

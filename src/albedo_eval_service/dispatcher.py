from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from typing import Any
from uuid import UUID

import httpx
from loguru import logger

from .config import Settings, get_settings
from .dataset_manifest import load_manifest_file
from .faults import broken_stream_fault, classify_failure_verdict
from .models import Challenger, DatasetConfig, EvalRequest, PreviousKing, ScoringConfig
from .notifications import EvalErrorNotification, notify_eval_error
from .remote_client import RemoteEvalClient
from .repository import ActiveEval, ClaimedEval, EvalRepository
from .sampling import swe_zero_manifest_sample_ids


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
    sample_ids = submission.get("dataset_sample_ids") or _build_sample_ids(settings, submission["block_hash"])
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


def _build_sample_ids(settings: Settings, block_hash: str) -> list[str]:
    if not settings.dataset_manifest_path:
        return []
    manifest = load_manifest_file(
        settings.dataset_manifest_path,
        expected_sha256=settings.dataset_manifest_hash,
    )
    return swe_zero_manifest_sample_ids(
        manifest,
        block_hash=block_hash,
        sample_count=settings.sample_count,
        max_turns_per_sample=settings.max_turns_per_sample,
    )


def _notify_dispatch_failure(
    *,
    submission_id: UUID,
    eval_run_id: UUID,
    fault_class: str,
    fault_code: str,
    fault_message: str,
    retryable: bool,
    remote_run_id: str | None = None,
) -> None:
    notify_eval_error(
        EvalErrorNotification(
            component="eval-dispatcher",
            severity="error",
            message=fault_message,
            eval_run_id=str(eval_run_id),
            submission_id=str(submission_id),
            fault_class=fault_class,
            fault_code=fault_code,
            retryable=retryable,
            details={"remote_run_id": remote_run_id or ""},
        )
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

        remote_run_id = str(claimed.eval_run_id)
        client = RemoteEvalClient(
            base_url=claimed.remote_host.base_url,
            auth_token=self.settings.remote_auth_token,
            timeout_seconds=self.settings.remote_event_timeout_seconds,
        )
        try:
            await client.ready()
            start_response = await client.start_eval(claimed.request)
            remote_run_id = str(start_response.get("remote_run_id") or claimed.eval_run_id)
            self.repository.set_remote_run_id(eval_run_id=claimed.eval_run_id, remote_run_id=remote_run_id)
            self.repository.heartbeat_attempt(attempt_id=claimed.attempt_id, lease_seconds=self.settings.lease_seconds)
            verdict = await self._follow_until_verdict(
                client,
                submission_id=claimed.submission_id,
                attempt_id=claimed.attempt_id,
                remote_run_id=remote_run_id,
            )
            self._complete_eval(
                submission_id=claimed.submission_id,
                attempt_id=claimed.attempt_id,
                eval_run_id=claimed.eval_run_id,
                verdict=verdict,
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
            _notify_dispatch_failure(
                submission_id=claimed.submission_id,
                eval_run_id=claimed.eval_run_id,
                fault_class=fault.fault_class,
                fault_code=fault.fault_code,
                fault_message=fault.fault_message,
                retryable=fault.retryable,
                remote_run_id=remote_run_id,
            )
            return True
        finally:
            await client.aclose()


    async def reconcile_once(self, *, limit: int = 10) -> int:
        reconciled = 0
        for active in self.repository.list_reconcilable_eval_runs(limit=limit):
            client = RemoteEvalClient(
                base_url=active.remote_host.base_url,
                auth_token=self.settings.remote_auth_token,
                timeout_seconds=self.settings.remote_event_timeout_seconds,
            )
            try:
                verdict = await self._follow_until_verdict(
                    client,
                    submission_id=active.submission_id,
                    attempt_id=active.attempt_id,
                    remote_run_id=active.remote_run_id,
                )
            except (httpx.HTTPError, asyncio.TimeoutError):
                continue
            finally:
                await client.aclose()

            if verdict.get("type") == "verdict" or verdict.get("state") in {"succeeded", "failed"}:
                self._complete_eval(
                    submission_id=active.submission_id,
                    attempt_id=active.attempt_id,
                    eval_run_id=active.eval_run_id,
                    verdict=verdict,
                )
                reconciled += 1
        return reconciled

    def _complete_eval(
        self,
        *,
        submission_id: UUID,
        attempt_id: UUID,
        eval_run_id: UUID,
        verdict: dict[str, Any],
    ) -> None:
        if verdict.get("state") == "succeeded":
            self.repository.mark_eval_succeeded(
                submission_id=submission_id,
                attempt_id=attempt_id,
                eval_run_id=eval_run_id,
                verdict=verdict,
            )
        else:
            fault = classify_failure_verdict(verdict)
            self.repository.mark_eval_failed(
                submission_id=submission_id,
                attempt_id=attempt_id,
                eval_run_id=eval_run_id,
                fault_class=fault.fault_class,
                fault_code=fault.fault_code,
                fault_message=fault.fault_message,
                retryable=fault.retryable,
            )
            _notify_dispatch_failure(
                submission_id=submission_id,
                eval_run_id=eval_run_id,
                fault_class=fault.fault_class,
                fault_code=fault.fault_code,
                fault_message=fault.fault_message,
                retryable=fault.retryable,
                remote_run_id=str(verdict.get("eval_run_id") or eval_run_id),
            )

    async def _follow_until_verdict(
        self,
        client: RemoteEvalClient,
        *,
        submission_id: UUID,
        attempt_id: UUID,
        remote_run_id: str,
    ) -> dict[str, Any]:
        seen_event_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        while True:
            try:
                events: list[dict[str, Any]] = []
                async for event in client.iter_events(remote_run_id):
                    events.append(event)
                consecutive_errors = 0
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise
                logger.warning(f"[eval-dispatch] transient poll error ({consecutive_errors}/{max_consecutive_errors}), retrying: {exc}")
                await asyncio.sleep(min(10 * consecutive_errors, 60))
                continue

            for event in events[seen_event_count:]:
                self.repository.record_remote_event(
                    submission_id=submission_id,
                    attempt_id=attempt_id,
                    event=event,
                )
                self.repository.heartbeat_attempt(attempt_id=attempt_id, lease_seconds=self.settings.lease_seconds)
                if event.get("type") == "verdict":
                    return event
            seen_event_count = max(seen_event_count, len(events))

            try:
                remote_state = await client.get_eval(remote_run_id)
                consecutive_errors = 0
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise
                logger.warning(f"[eval-dispatch] transient state error ({consecutive_errors}/{max_consecutive_errors}), retrying: {exc}")
                await asyncio.sleep(min(10 * consecutive_errors, 60))
                continue

            if remote_state.get("type") == "verdict" or remote_state.get("state") in {"succeeded", "failed"}:
                if len(events) == seen_event_count and remote_state.get("type") == "verdict":
                    self.repository.record_remote_event(
                        submission_id=submission_id,
                        attempt_id=attempt_id,
                        event=remote_state,
                    )
                return remote_state
            # renew the lease even when no new events arrive (generation takes 30+ min with no events)
            self.repository.heartbeat_attempt(attempt_id=attempt_id, lease_seconds=self.settings.lease_seconds)
            await asyncio.sleep(self.settings.remote_event_poll_seconds)

    async def run_forever(self) -> None:
        # Keeps the loop alive across unexpected errors (DB glitch, transient exception).
        while True:
            try:
                did_work = await self.dispatch_once()
                if not did_work:
                    await asyncio.sleep(self.settings.dispatch_poll_seconds)
            except Exception as exc:  # noqa: BLE001 - keep loop alive across unexpected errors
                logger.exception(f"[eval-dispatch] unhandled error, retrying in {self.settings.dispatch_poll_seconds}s: {exc}")
                await asyncio.sleep(self.settings.dispatch_poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Albedo eval dispatcher.")
    parser.add_argument("--once", action="store_true", help="Claim and dispatch at most one eval.")
    parser.add_argument("--sweep-abandoned", action="store_true", help="Mark expired EVAL attempts abandoned and retryable.")
    parser.add_argument("--reconcile-running", action="store_true", help="Replay remote state for active eval runs with remote_run_id.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum active eval runs to reconcile.")
    args = parser.parse_args()

    settings = get_settings()
    dispatcher = EvalDispatcher(
        settings=settings,
        repository=EvalRepository(settings.database_url),
    )
    if args.sweep_abandoned:
        abandoned = dispatcher.repository.sweep_abandoned_eval_attempts(worker_id=settings.worker_id)
        print(f"abandoned_eval_attempts={abandoned}")
    elif args.reconcile_running:
        reconciled = asyncio.run(dispatcher.reconcile_once(limit=args.limit))
        print(f"reconciled_eval_runs={reconciled}")
    elif args.once:
        asyncio.run(dispatcher.dispatch_once())
    else:
        # Pre-install custom signal handlers BEFORE asyncio.run() so asyncio
        # never installs its own SIGINT handler.  asyncio only installs its
        # handler when signal.getsignal(SIGINT) is default_int_handler; with a
        # custom handler in place, Runner.run()'s finally block keeps
        # sigint_handler=None and never executes `raise KeyboardInterrupt()`.
        # sys.exit() raises SystemExit which propagates cleanly through asyncio.
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        try:
            asyncio.run(dispatcher.run_forever())
        except (KeyboardInterrupt, SystemExit):
            pass

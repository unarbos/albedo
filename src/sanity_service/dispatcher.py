"""Sanity pre-eval dispatcher - claim, sample, push to the worker, judge, persist (mirrors eval)."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
from loguru import logger

from sanity_remote.models import SanityRunRequest
from sanity_service.dataset import sample_prompts
from sanity_service.db import ClaimedPreEval, PreEvalRepository
from sanity_service.judge_panel import make_client
from sanity_service.llm_check import SampleInput, run_gate
from sanity_service.remote_client import SanityRemoteClient
from sanity_service.settings import SanitySettings, get_settings
from sanity_service.uploads import put_sanity_fault


class SanityDispatcher:
    # Orchestrates one pre-eval at a time: claim -> dispatch -> poll -> judge -> persist.

    def __init__(self, *, settings: SanitySettings, repository: PreEvalRepository) -> None:
        self.settings = settings
        self.repository = repository

    def _build_request(self, submission: dict[str, Any], host: Any, attempt_id: UUID) -> SanityRunRequest:
        # Samples the prompts (stable side) and builds the worker request; run_id = attempt_id.
        samples = sample_prompts(
            seed=str(submission["block_hash"]),
            n=self.settings.sample_count,
            max_turns=self.settings.max_turns_per_sample,
            manifest_path=self.settings.dataset_manifest_path,
            manifest_hash=self.settings.dataset_manifest_hash,
            dataset_root=self.settings.dataset_root,
        )
        return SanityRunRequest(
            run_id=str(attempt_id),
            model_uri=submission["model_uri"],
            digest=submission.get("model_hash") or "",
            prompts=[s.prompt for s in samples],
            prompt_messages=[
                s.messages or [{"role": "user", "content": s.prompt}] for s in samples
            ],
            gen_max_tokens=self.settings.gen_max_tokens,
        )

    def claim_once(self) -> ClaimedPreEval | None:
        # Claims the next queued pre-eval (sampling happens inside the request builder).
        return self.repository.claim_next_pre_eval(
            worker_id=self.settings.worker_id,
            lease_seconds=self.settings.lease_seconds,
            request_builder=self._build_request,
        )

    async def dispatch_once(self) -> bool:
        # Claims and runs one pre-eval end to end; returns False when nothing was claimable.
        claimed = self.claim_once()
        if not claimed:
            logger.debug("[sanity-dispatch] no claimable pre-eval")
            return False
        logger.info("[sanity-dispatch] claimed submission={} digest={:.16} host={}", claimed.submission_id, claimed.request.digest, claimed.remote_host.id,)
        client = SanityRemoteClient(
            base_url=claimed.remote_host.base_url,
            auth_token=self.settings.remote_auth_token,
            timeout_seconds=self.settings.remote_event_timeout_seconds,
        )
        try:
            await client.ready()
            start = await client.start_run(claimed.request)
            run_id = str(start.get("run_id") or claimed.attempt_id)
            self.repository.heartbeat_attempt(attempt_id=claimed.attempt_id, lease_seconds=self.settings.lease_seconds)
            result = await self._follow_until_result(client, submission_id=claimed.submission_id, attempt_id=claimed.attempt_id, run_id=run_id,)
            await self._complete(
                submission_id=claimed.submission_id,
                attempt_id=claimed.attempt_id,
                repo=claimed.request.model_uri,
                digest=claimed.request.digest,
                prompts=list(claimed.request.prompts),
                result=result,
            )
            return True
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            logger.warning("[sanity-dispatch] worker unreachable submission={} digest={:.16}: {}", claimed.submission_id, claimed.request.digest, exc,)
            self.repository.mark_pre_eval_failed(
                submission_id=claimed.submission_id,
                attempt_id=claimed.attempt_id,
                repo=claimed.request.model_uri,
                digest=claimed.request.digest,
                fault_class="INFRA_FAULT",
                fault_code="worker_unreachable",
                fault_message=str(exc),
                retryable=True,
            )
            return True
        finally:
            await client.aclose()

    async def _follow_until_result(self, client: SanityRemoteClient, *, submission_id: UUID, attempt_id: UUID, run_id: str) -> dict[str, Any]:
        # Polls the worker, recording events and refreshing the lease, until a result appears.
        # Heartbeat runs once per poll tick (not just per event) so a long model download or
        # vLLM boot — which emits no events — does not let the lease expire mid-wait.
        seen = 0
        while True:
            events = [event async for event in client.iter_events(run_id)]
            for event in events[seen:]:
                ev_type = event.get("type", "?")
                logger.info("[sanity-dispatch] worker event={} run={} submission={:.8}", ev_type, run_id, str(submission_id),)
                self.repository.record_remote_event(submission_id=submission_id, attempt_id=attempt_id, event=event)
                if event.get("type") == "result":
                    logger.info("[sanity-dispatch] result received run={} state={} submission={:.8}", run_id, event.get("state"), str(submission_id),)
                    self.repository.heartbeat_attempt(attempt_id=attempt_id, lease_seconds=self.settings.lease_seconds)
                    return event
            seen = max(seen, len(events))
            # Heartbeat on every tick so a silent download/boot period does not expire the lease.
            self.repository.heartbeat_attempt(attempt_id=attempt_id, lease_seconds=self.settings.lease_seconds)
            status = await client.get_run(run_id)
            if status.get("type") == "result" or status.get("state") in {"succeeded", "failed"}:
                if status.get("type") == "result":
                    self.repository.record_remote_event(submission_id=submission_id, attempt_id=attempt_id, event=status)
                return status
            await asyncio.sleep(self.settings.remote_event_poll_seconds)

    async def _complete(self, *, submission_id: UUID, attempt_id: UUID, repo: str, digest: str, prompts: list[str], result: dict[str, Any],) -> None:
        # Judges the generated responses and writes the terminal verdict.
        logger.info("[sanity-dispatch] completing submission={:.8} digest={:.16} state={}", str(submission_id), digest, result.get("state"),)
        if result.get("state") == "failed":
            self.repository.mark_pre_eval_failed(
                submission_id=submission_id,
                attempt_id=attempt_id,
                repo=repo,
                digest=digest,
                fault_class="INFRA_FAULT",
                fault_code=result.get("fault_code", "worker_fault"),
                fault_message=result.get("fault_message", ""),
                retryable=bool(result.get("retryable", True)),
            )
            return

        responses = list(result.get("responses", []))
        heuristics = list(result.get("heuristics", []))
        samples = [
            SampleInput(
                prompt=prompts[i] if i < len(prompts) else "",
                response=responses[i],
                heuristic_passed=bool(heuristics[i].get("passed", True))
                if i < len(heuristics)
                else True,
                heuristic_reason=heuristics[i].get("reason", "") if i < len(heuristics) else "",
            )
            for i in range(len(responses))
        ]
        client = make_client()
        try:
            gate = await run_gate(
                samples,
                client,
                consensus=self.settings.consensus,
                skip_viability=self.settings.skip_viability,
            )
        finally:
            await client.aclose()

        if gate.infra_fault:
            self.repository.mark_pre_eval_failed(
                submission_id=submission_id,
                attempt_id=attempt_id,
                repo=repo,
                digest=digest,
                fault_class="INFRA_FAULT",
                fault_code="judges_unavailable",
                fault_message=gate.reason,
                retryable=True,
            )
        elif gate.passed:
            self.repository.mark_pre_eval_passed(
                submission_id=submission_id,
                attempt_id=attempt_id,
                repo=repo,
                digest=digest,
                responses=responses,
                reason=gate.reason,
                timing={},
            )
        else:
            # Terminal miner fault: publish a fault report to Hippius (reason + per-judge evidence)
            # so it can be linked from the dashboard, then record the artifact alongside
            # the verdict.
            detail = {
                "submission_id": str(submission_id),
                "repo": repo,
                "digest": digest,
                "fault_code": str(gate.llm_gate),
                "reason": gate.reason,
                "decision_mode": gate.decision_mode,
                "gate": dataclasses.asdict(gate),
                "prompts": prompts,
                "responses": responses,
                "checked_at": datetime.now(UTC).isoformat(),
            }
            artifact_uri = put_sanity_fault(str(submission_id), digest, detail)
            self.repository.mark_pre_eval_failed(
                submission_id=submission_id,
                attempt_id=attempt_id,
                repo=repo,
                digest=digest,
                fault_class="MINER_FAULT",
                fault_code=str(gate.llm_gate),
                fault_message=gate.reason,
                retryable=False,
                responses=responses,
                artifact_uri=artifact_uri,
            )

    async def reconcile_once(self, *, limit: int = 10, follow_timeout: float = 50.0) -> int:
        # Replays in-flight pre-evals whose dispatcher may have crashed mid-poll.
        # follow_timeout must be shorter than the cron_restart interval (60s) so PM2 never
        # has to SIGTERM a busy reconciler — the TimeoutError exits cleanly instead.
        in_flight = self.repository.list_reconcilable_pre_eval(limit=limit)
        logger.info("[sanity-dispatch] reconcile found={}", len(in_flight))
        if not in_flight:
            return 0
        reconciled = 0
        for active in in_flight:
            client = SanityRemoteClient(base_url=active.remote_host.base_url, auth_token=self.settings.remote_auth_token, timeout_seconds=self.settings.remote_event_timeout_seconds,)
            try:
                result = await asyncio.wait_for(
                    self._follow_until_result(
                        client,
                        submission_id=active.submission_id,
                        attempt_id=active.attempt_id,
                        run_id=active.run_id,
                    ),
                    timeout=follow_timeout,
                )
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                logger.warning("[sanity-dispatch] reconcile skipped submission={} run={}: {}", active.submission_id, active.run_id, exc,)
                continue
            finally:
                await client.aclose()
            try:
                await self._complete(
                    submission_id=active.submission_id,
                    attempt_id=active.attempt_id,
                    repo=active.repo,
                    digest=active.digest,
                    prompts=active.prompts,
                    result=result,
                )
            except Exception as exc:  # noqa: BLE001 - log and continue so one bad completion does not abort the loop
                logger.exception("[sanity-dispatch] reconcile _complete failed submission={}: {}", active.submission_id, exc)
                continue
            reconciled += 1
        return reconciled

    async def run_forever(self) -> None:
        # Continuously claims and dispatches pre-evals; keeps the loop alive across transient errors.
        while True:
            try:
                did_work = await self.dispatch_once()
                if not did_work:
                    logger.debug("[sanity-dispatch] idle — sleeping {}s", self.settings.dispatch_poll_seconds)
                    await asyncio.sleep(self.settings.dispatch_poll_seconds)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive across DB blips, etc.
                logger.exception("[sanity-dispatch] unhandled error in dispatch loop, retrying in {}s: {}", self.settings.dispatch_poll_seconds, exc)
                await asyncio.sleep(self.settings.dispatch_poll_seconds)


def main() -> None:
    # CLI entrypoint (--once / --sweep-abandoned / --reconcile-running), mirroring eval.
    parser = argparse.ArgumentParser(description="Run the Albedo sanity pre-eval dispatcher.")
    parser.add_argument("--once", action="store_true", help="Claim and dispatch at most one pre-eval.")
    parser.add_argument("--sweep-abandoned", action="store_true", help="Reclaim expired pre-eval attempts.",)
    parser.add_argument("--reconcile-running", action="store_true", help="Replay in-flight pre-eval runs.",)
    parser.add_argument("--limit", type=int, default=10, help="Max active runs to reconcile.")
    args = parser.parse_args()

    settings = get_settings()
    dispatcher = SanityDispatcher(
        settings=settings,
        repository=PreEvalRepository(
            settings.database_url,
            min_free_gpus=settings.min_free_gpus,
            max_retry_count=settings.max_retry_count,
        ),
    )
    if args.sweep_abandoned:
        logger.info("[sanity-dispatch] abandoned={}", dispatcher.repository.sweep_abandoned_pre_eval(worker_id=settings.worker_id),)
    elif args.reconcile_running:
        try:
            logger.info("[sanity-dispatch] reconciled={}", asyncio.run(dispatcher.reconcile_once(limit=args.limit)),)
        except KeyboardInterrupt:
            logger.info("[sanity-dispatch] reconciler interrupted by signal, exiting cleanly")
    elif args.once:
        asyncio.run(dispatcher.dispatch_once())
    else:
        asyncio.run(dispatcher.run_forever())


from __future__ import annotations

import argparse
import asyncio
import dataclasses
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from loguru import logger

from albedo_eval_service.judge_config import JudgeSettings, get_judge_settings
from albedo_eval_service.observation_format import (
    detect_format,
    empty_output,
    format_block,
    valid_output,
    wrap,
)
from albedo_eval_service.remote_dataset import format_messages
from sanity_remote.models import SanityRunRequest
from sanity_service.dataset import sample_prompts
from sanity_service.db import ClaimedPreEval, PreEvalRepository
from sanity_service.judge_panel import make_client
from sanity_service.llm_check import SampleInput, run_gate
from sanity_service.remote_client import SanityRemoteClient
from sanity_service.settings import SanitySettings, get_settings
from sanity_service.uploads import put_sanity_fault

_CANONICAL_TOKENIZER_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "tokenizers" / "Qwen3.6-35B-A3B"
)
_COMPLETE_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)\s*\n.*?```", re.IGNORECASE | re.DOTALL)

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

@dataclass
class _TrajectoryState:
    sample_id: str
    prompt: str
    messages: list[dict[str, str]]
    turns: list[dict[str, Any]]
    stopped: bool = False
    error: str = ""
    heuristic_reason: str = ""


class SanityDispatcher:

    def __init__(self, *, settings: SanitySettings, repository: PreEvalRepository) -> None:
        self.settings = settings
        self.repository = repository

    def _build_request(self, submission: dict[str, Any], host: Any, attempt_id: UUID) -> SanityRunRequest:
        samples = sample_prompts(
            seed=str(submission["block_hash"]),
            n=self.settings.sample_count,
            manifest_path=self.settings.dataset_manifest_path,
            manifest_hash=self.settings.dataset_manifest_hash,
            dataset_root=self.settings.dataset_root,
        )
        return SanityRunRequest(
            run_id=str(attempt_id),
            model_uri=submission["model_uri"],
            digest=submission.get("model_hash") or "",
            prompts=[s.prompt for s in samples],
            sample_ids=[s.sample_id or f"sanity-sample:{i}" for i, s in enumerate(samples)],
            prompt_messages=[
                s.messages or [{"role": "user", "content": s.prompt}] for s in samples
            ],
            assistant_turns=self.settings.trajectory_assistant_turns,
            gen_max_tokens=self.settings.gen_max_tokens,
        )

    def claim_once(self) -> ClaimedPreEval | None:
        return self.repository.claim_next_pre_eval(
            worker_id=self.settings.worker_id,
            lease_seconds=self.settings.lease_seconds,
            request_builder=self._build_request,
        )

    async def dispatch_once(self) -> bool:
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
            if claimed.request.assistant_turns > 1:
                result = await self._run_multiturn(client, claimed)
            else:
                result = await self._run_remote_request(client, claimed.request, claimed)
            await self._complete(
                submission_id=claimed.submission_id,
                attempt_id=claimed.attempt_id,
                repo=claimed.request.model_uri,
                digest=claimed.request.digest,
                prompts=list(claimed.request.prompts),
                result=result,
            )
            return True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                logger.warning(
                    "[sanity-dispatch] worker busy, releasing claim submission={} digest={:.16}: {}",
                    claimed.submission_id, claimed.request.digest, exc,
                )
                self.repository.release_pre_eval_attempt(
                    submission_id=claimed.submission_id,
                    attempt_id=claimed.attempt_id,
                    fault_message=str(exc),
                )
                return True
            logger.warning(
                "[sanity-dispatch] worker HTTP error submission={} digest={:.16}: {}",
                claimed.submission_id, claimed.request.digest, exc,
            )
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

    async def _run_remote_request(
        self,
        client: SanityRemoteClient,
        request: SanityRunRequest,
        claimed: ClaimedPreEval,
    ) -> dict[str, Any]:
        start = await client.start_run(request)
        run_id = str(start.get("run_id") or request.run_id)
        self.repository.heartbeat_attempt(
            attempt_id=claimed.attempt_id,
            lease_seconds=self.settings.lease_seconds,
        )
        return await self._follow_until_result(
            client,
            submission_id=claimed.submission_id,
            attempt_id=claimed.attempt_id,
            run_id=run_id,
        )

    async def _run_multiturn(
        self, client: SanityRemoteClient, claimed: ClaimedPreEval
    ) -> dict[str, Any]:
        request = claimed.request
        turn_count = max(1, int(request.assistant_turns))
        states = _trajectory_states(request)
        kept_warm = False
        try:
            for turn_index in range(turn_count):
                active = [
                    state for state in states 
                    if  not state.stopped and
                        not state.error and 
                        not state.heuristic_reason
                ]

                if not active:
                    break

                turn_request = request.model_copy(
                    update={
                        "run_id": f"{claimed.attempt_id}:turn-{turn_index + 1}",
                        "prompts": [state.prompt for state in active],
                        "sample_ids": [state.sample_id for state in active],
                        "prompt_messages": [state.messages for state in active]
                        if turn_index == 0
                        else None,
                        "teardown_after_run": turn_index == turn_count - 1,
                    }
                )
                kept_warm = not turn_request.teardown_after_run
                logger.info(
                    "[sanity-dispatch] trajectory turn {}/{} samples={}",
                    turn_index + 1,
                    turn_count,
                    len(active),
                )
                result = await self._run_remote_request(client, turn_request, claimed)
                if result.get("state") == "failed":
                    return result
                _apply_turn_result(active, result)
                if turn_index == turn_count - 1:
                    break
                await _append_observations(active, str(claimed.attempt_id), turn_index + 1)
            return _trajectory_result(str(claimed.attempt_id), states, turn_count)
        finally:
            if kept_warm:
                try:
                    await client.teardown()
                except Exception as exc:
                    logger.warning("[sanity-dispatch] remote teardown failed: {}", exc)

    async def _follow_until_result(self, client: SanityRemoteClient, *, submission_id: UUID, attempt_id: UUID, run_id: str) -> dict[str, Any]:
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
            self.repository.heartbeat_attempt(attempt_id=attempt_id, lease_seconds=self.settings.lease_seconds)
            status = await client.get_run(run_id)
            if status.get("type") == "result" or status.get("state") in {"succeeded", "failed"}:
                if status.get("type") == "result":
                    self.repository.record_remote_event(submission_id=submission_id, attempt_id=attempt_id, event=status)
                return status
            await asyncio.sleep(self.settings.remote_event_poll_seconds)

    async def _complete(self, *, submission_id: UUID, attempt_id: UUID, repo: str, digest: str, prompts: list[str], result: dict[str, Any],) -> None:
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
        except Exception as exc:
            logger.exception(f"[sanity-dispatch] judge gate failed submission={submission_id}: {exc}")
            self.repository.mark_pre_eval_failed(
                submission_id=submission_id,
                attempt_id=attempt_id,
                repo=repo,
                digest=digest,
                fault_class="INFRA_FAULT",
                fault_code="judges_failed",
                fault_message=str(exc),
                retryable=True,
            )
            return
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
            except Exception as exc:
                logger.exception("[sanity-dispatch] reconcile _complete failed submission={}: {}", active.submission_id, exc)
                continue
            reconciled += 1
        return reconciled

    async def run_forever(self) -> None:
        while True:
            try:
                did_work = await self.dispatch_once()
                if not did_work:
                    logger.debug("[sanity-dispatch] idle — sleeping {}s", self.settings.dispatch_poll_seconds)
                    await asyncio.sleep(self.settings.dispatch_poll_seconds)
            except Exception as exc:
                logger.exception("[sanity-dispatch] unhandled error in dispatch loop, retrying in {}s: {}", self.settings.dispatch_poll_seconds, exc)
                await asyncio.sleep(self.settings.dispatch_poll_seconds)


def main() -> None:
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


def _trajectory_states(request: SanityRunRequest) -> list[_TrajectoryState]:
    sample_ids = request.sample_ids or [
        f"sanity-sample:{i}" for i in range(len(request.prompts))
    ]
    prompt_messages = request.prompt_messages or []
    states: list[_TrajectoryState] = []
    for i, prompt in enumerate(request.prompts):
        messages = (
            prompt_messages[i]
            if i < len(prompt_messages)
            else [{"role": "user", "content": prompt}]
        )
        clean_messages = [
            {
                "role": str(message.get("role") or "user"),
                "content": str(message.get("content") or ""),
            }
            for message in messages
        ]
        states.append(
            _TrajectoryState(
                sample_id=sample_ids[i] if i < len(sample_ids) else f"sanity-sample:{i}",
                prompt=prompt,
                messages=clean_messages,
                turns=[
                    {"role": message["role"], "content": message["content"]}
                    for message in clean_messages
                ],
            )
        )
    return states


def _apply_turn_result(states: list[_TrajectoryState], result: dict[str, Any]) -> None:
    responses = list(result.get("responses", []))
    heuristics = list(result.get("heuristics", []))
    for i, state in enumerate(states):
        if i >= len(responses):
            state.error = "missing_generation_response"
            continue
        response = str(responses[i] or "")
        state.turns.append({"role": "assistant", "content": response, "score_target": True})
        if i < len(heuristics) and not bool(heuristics[i].get("passed", True)):
            reason = str(heuristics[i].get("reason") or "heuristic failed")
            if not _has_bash_command(response):
                state.heuristic_reason = reason


async def _append_observations(
    states: list[_TrajectoryState], eval_run_id: str, turn_index: int
) -> None:
    active = []
    for state in states:
        if state.error or state.stopped:
            continue
        assistant_output = str(state.turns[-1].get("content") or "")
        if _assistant_submitted(assistant_output):
            observation = _completion_observation(state.sample_id, state.messages)
            _append_observation(state, observation)
            state.stopped = True
        elif not _has_bash_command(assistant_output):
            _append_observation(
                state, _missing_command_observation(state.sample_id, state.messages)
            )
        else:
            active.append((state, assistant_output))
    if not active:
        return

    settings = get_judge_settings()
    client = make_client(settings)
    try:
        results = await asyncio.gather(
            *[
                _simulate_observation(
                    client=client,
                    settings=settings,
                    eval_run_id=eval_run_id,
                    sample_id=state.sample_id,
                    prompt=state.prompt,
                    messages=state.messages,
                    assistant_output=assistant_output,
                )
                for state, assistant_output in active
            ],
            return_exceptions=True,
        )
    finally:
        await client.aclose()

    for (state, _assistant_output), result in zip(active, results, strict=False):
        if isinstance(result, Exception):
            state.error = f"{type(result).__name__}: {result}"
            continue
        _append_observation(state, result)
        state.prompt = format_messages(
            state.messages,
            tokenizer_path=str(_CANONICAL_TOKENIZER_PATH),
            enable_thinking=True,
        )
    logger.info(
        "[sanity-dispatch] simulated observations turn={} samples={}",
        turn_index,
        len(active),
    )


def _append_observation(state: _TrajectoryState, observation: str) -> None:
    assistant_output = str(state.turns[-1].get("content") or "")
    state.messages.extend(
        [
            {"role": "assistant", "content": assistant_output},
            {"role": "user", "content": observation},
        ]
    )
    state.turns.append(
        {
            "role": "user",
            "content": observation,
            "environment_observation": True,
        }
    )


async def _simulate_observation(
    *,
    client: Any,
    settings: JudgeSettings,
    eval_run_id: str,
    sample_id: str,
    prompt: str,
    messages: list[dict[str, str]],
    assistant_output: str,
) -> str:
    fmt = detect_format(sample_id, messages)
    response = await client.complete(
        model=settings.evaluator_model,
        messages=[
            {"role": "system", "content": _simulation_system_prompt(fmt)},
            {
                "role": "user",
                "content": _simulation_transcript(
                    messages=messages,
                    prompt=prompt,
                    assistant_output=assistant_output,
                ),
            },
        ],
        temperature=0.0,
        max_tokens=settings.simulation_max_tokens,
        provider=_evaluator_provider(settings),
        accept=lambda raw: valid_output(raw, fmt),
    )
    if response.error:
        raise RuntimeError(response.error)
    observation = response.raw.strip()
    if not valid_output(observation, fmt):
        fallback = empty_output(fmt)
        logger.warning(
            "[sanity-dispatch] observation_simulation_invalid_format eval_run_id={} "
            "sample_id={} fmt={} fallback={!r}",
            eval_run_id,
            sample_id,
            fmt,
            fallback,
        )
        return fallback
    return observation


def _trajectory_result(
    run_id: str, states: list[_TrajectoryState], turn_count: int
) -> dict[str, Any]:
    responses = [
        "" if state.error else _format_scored_trajectory(state.turns) for state in states
    ]
    heuristics = [
        {
            "passed": not state.error and not state.heuristic_reason,
            "reason": state.error or state.heuristic_reason,
        }
        for state in states
    ]
    return {
        "type": "result",
        "run_id": run_id,
        "state": "succeeded",
        "responses": responses,
        "heuristics": heuristics,
        "assistant_turns": turn_count,
    }


def _format_scored_trajectory(turns: list[dict[str, Any]]) -> str:
    target_count = sum(
        1 for turn in turns if turn.get("role") == "assistant" and turn.get("score_target")
    )
    target_label = (
        "CANDIDATE OUTPUT"
        if target_count == 1
        else f"CANDIDATE OUTPUT 1 through CANDIDATE OUTPUT {target_count}"
    )
    assistant_index = 0
    parts = [
        "FULL CANDIDATE TRAJECTORY",
        f"Score ONLY {target_label}. "
        "The ENVIRONMENT OBSERVATION is context only.",
    ]
    for turn in turns:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").rstrip()
        if role == "assistant" and turn.get("score_target"):
            assistant_index += 1
            label = f"CANDIDATE OUTPUT {assistant_index}"
        elif role == "user" and turn.get("environment_observation"):
            label = "ENVIRONMENT OBSERVATION (context only, do not score)"
        else:
            label = f"CONTEXT {role.upper()} (do not score)" if role else "CONTEXT TURN (do not score)"
        parts.append(f"\n{label}:\n------\n{content}\n------")
    return "\n".join(parts).strip()


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


def _simulation_system_prompt(fmt: str) -> str:
    return f"{BASE_PROMPT}\n{format_block(fmt)}"


def _evaluator_provider(settings: JudgeSettings) -> dict[str, Any]:
    block: dict[str, Any] = {"allow_fallbacks": True, "quantizations": ["fp8"]}
    order = [p.strip() for p in settings.evaluator_providers.split(",") if p.strip()]
    if order:
        block["order"] = order
        block["allow_fallbacks"] = False
    return block


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

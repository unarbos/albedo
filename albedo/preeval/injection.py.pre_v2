"""LLM injection probe gate — runs before the GPU duel."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from albedo.config import JUDGE_MODELS

if TYPE_CHECKING:
    from albedo.judge import ChutesJudge

logger = logging.getLogger(__name__)

_CHAL_TIMEOUT = httpx.Timeout(connect=10.0, read=90.0, write=15.0, pool=10.0)
# Max seconds between generation retries (exponential, capped).
_GEN_RETRY_BACKOFF_CAP = 120.0


@dataclass
class ProbeResult:
    n_probes:         int
    n_injections:     int
    triggered_judges: list[str]
    probe_details:    list[dict]

    @property
    def n_untested(self) -> int:
        return sum(1 for d in self.probe_details if d.get("untested"))

    @property
    def is_clean(self) -> bool:
        """True only when all probes ran AND every (turn × judge) pair returned no injection.
        Any single detection from any judge on any turn fails the gate."""
        return self.n_injections == 0 and self.n_untested == 0


def _probe_seed(eval_id: str) -> int:
    """Deterministic seed distinct from the duel seed.

    The ':injection_probe' suffix ensures probe turns differ from duel fixtures
    so miners cannot train to pass the probe by memorising duel samples.
    """
    digest = hashlib.sha256(f"{eval_id}:injection_probe".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _sample_turns(dataset_dir: str, n: int, rng: random.Random) -> list[dict]:
    """Sample n conversation turns from parquet shards in dataset_dir."""
    try:
        import pyarrow.parquet as pq  # type: ignore[import]
    except ImportError:
        logger.warning("pyarrow not installed — returning empty probe turn list")
        return []

    shards = sorted(Path(dataset_dir).glob("**/*.parquet"))
    if not shards:
        logger.warning("no parquet shards found in %r", dataset_dir)
        return []

    turns: list[dict] = []
    rng.shuffle(shards)
    for shard in shards:
        rows = pq.read_table(str(shard)).to_pylist()
        rng.shuffle(rows)
        for row in rows:
            msgs = row.get("messages") or row.get("conversations") or []
            if msgs:
                turns.append({"messages": msgs})
            if len(turns) >= n:
                return turns
    return turns[:n]


async def _get_model_name(challenger_url: str, client: httpx.AsyncClient) -> str:
    """Query the vLLM /v1/models endpoint for the loaded model name."""
    try:
        resp = await client.get(f"{challenger_url.rstrip('/')}/v1/models", timeout=10.0)
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]
    except Exception as exc:
        logger.warning("could not discover challenger model name: %s", exc)
        return "default"  # vLLM fallback alias


async def _generate_reply(
    client: httpx.AsyncClient,
    challenger_url: str,
    messages: list[dict],
    model: str = "default",
) -> str:
    """Generate a challenger reply; retries indefinitely on transient server errors.

    Only fails immediately on permanent 4xx errors. We must always obtain a reply
    — silently skipping a probe turn is not acceptable.
    """
    attempt = 0
    last_exc: Exception | None = None
    while True:
        try:
            resp = await client.post(
                f"{challenger_url.rstrip('/')}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                raise  # permanent 4xx — propagate
            last_exc = exc
        except Exception as exc:
            last_exc = exc

        attempt += 1
        wait = min(2.0 ** (attempt - 1), _GEN_RETRY_BACKOFF_CAP)
        logger.warning(
            "challenger generation error (attempt %d) — retrying in %.0fs: %s",
            attempt, wait, last_exc,
        )
        await asyncio.sleep(wait)


async def _probe_one_turn(
    http: httpx.AsyncClient,
    challenger_url: str,
    judge_client: "ChutesJudge",
    judges: list[str],
    turn: dict,
    probe_index: int,
    model: str = "default",
) -> dict:
    """Run one probe turn: generate reply, check all judges concurrently.

    Returns a detail dict. Raises on generation failure so the caller can
    decide whether to count the turn as a probe pass or abort.
    """
    messages: list[dict] = turn["messages"]
    # Keep only the prompt side — strip trailing assistant turns.
    prompt_messages = [m for m in messages if m.get("role") != "assistant"]
    if not prompt_messages:
        return {"probe_index": probe_index, "skipped": True, "injections_by_judge": {}}

    reply = await _generate_reply(http, challenger_url, prompt_messages, model=model)

    async def _ask(judge_model: str) -> tuple[str, bool]:
        """Ask one judge about injection; retries until a definitive answer is received."""
        attempt = 0
        while True:
            try:
                # judge_client.probe already retries internally (including on unparseable
                # output); this outer loop is a safety net for unexpected exceptions.
                is_inj, _ = await judge_client.probe(
                    messages=prompt_messages, reply=reply, model=judge_model
                )
                return judge_model, is_inj
            except Exception as exc:
                attempt += 1
                wait = min(2.0 ** (attempt - 1), _GEN_RETRY_BACKOFF_CAP)
                logger.warning(
                    "probe %d: judge %r outer retry (attempt %d) in %.0fs: %s",
                    probe_index, judge_model, attempt, wait, exc,
                )
                await asyncio.sleep(wait)

    outcomes = await asyncio.gather(*[_ask(m) for m in judges])

    injections_by_judge: dict[str, bool] = {}
    for judge_model, is_injected in outcomes:
        injections_by_judge[judge_model] = is_injected
        if is_injected:
            logger.warning("probe %d: injection detected by judge %r",
                           probe_index, judge_model)

    return {
        "probe_index":         probe_index,
        "skipped":             False,
        "untested":            False,  # never untested — we always retry to completion
        "n_messages":          len(prompt_messages),
        "reply_len":           len(reply),
        "injections_by_judge": injections_by_judge,
    }


async def probe_injection(
    *,
    challenger_url: str,
    eval_id: str,
    dataset_dir: str,
    n_probes: int = 3,
    judges: list[str] | None = None,
    judge_client: "ChutesJudge | None" = None,
) -> ProbeResult:
    """Sample n_probes turns, generate replies, check all judges concurrently.

    All probe turns are run in parallel. Probe seed is distinct from duel seed.
    Every probe turn always completes — no silent skips or untested entries.
    """
    from albedo.judge import ChutesJudge

    if judges is None:
        judges = list(JUDGE_MODELS)

    # Manage judge_client lifetime — always close if we created it.
    _owned = judge_client is None
    if _owned:
        judge_client = ChutesJudge()

    try:
        rng = random.Random(_probe_seed(eval_id))
        turns = _sample_turns(dataset_dir, n_probes, rng)

        if not turns:
            logger.error("probe_injection: no turns sampled — failing closed (not clean)")
            # Return a result with one untested turn so is_clean=False.
            # Empty dataset is an operator misconfiguration; failing open would
            # silently disable the injection gate for all challengers.
            return ProbeResult(
                n_probes=0, n_injections=0, triggered_judges=[],
                probe_details=[{"probe_index": 0, "error": "no_dataset_turns", "untested": True}],
            )

        async with httpx.AsyncClient(timeout=_CHAL_TIMEOUT) as http:
            # Resolve actual model name once — sending "default" causes HTTP 400
            # on standard vLLM deployments that don't register a "default" alias.
            challenger_model = await _get_model_name(challenger_url, http)

            tasks = [
                asyncio.create_task(
                    _probe_one_turn(
                        http, challenger_url, judge_client, judges,
                        turn, i, model=challenger_model,
                    )
                )
                for i, turn in enumerate(turns)
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if _owned:
            await judge_client.aclose()

    # Zero tolerance: a single True from any judge on any probe turn = injection.
    # Only acceptable outcome is 0 detections across all (turn × judge) pairs.
    n_injections = 0
    triggered_judges: list[str] = []
    probe_details: list[dict] = []

    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            # Only happens on permanent 4xx from the challenger — genuine failure.
            logger.error("probe %d: turn failed with permanent error: %s", i, result)
            probe_details.append({"probe_index": i, "error": str(result), "untested": True})
            continue

        probe_details.append(result)
        if result.get("skipped"):
            continue

        for judge_model, detected in result.get("injections_by_judge", {}).items():
            if detected:
                # Any judge flagging any turn is enough — no majority required.
                n_injections += 1
                if judge_model not in triggered_judges:
                    triggered_judges.append(judge_model)

    return ProbeResult(
        n_probes=len(turns),
        n_injections=n_injections,
        triggered_judges=triggered_judges,
        probe_details=probe_details,
    )

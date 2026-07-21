#!/usr/bin/env python3
"""Generate an OpenRouter challenger trajectory and score it against stored king output."""

from __future__ import annotations

import argparse
import asyncio
import email.utils
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from albedo_eval_service.judge_config import JudgeSettings
from albedo_eval_service.judge_core import JUDGE_MODELS, aggregate_scores
from albedo_eval_service.judge_openrouter import OpenRouterJudgeClient
from albedo_eval_service.remote_generation import format_scored_trajectory

COMPLETE_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def log(message: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now} UTC] {message}", flush=True)


def append_progress(path: Path, event: dict) -> None:
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def chunks(items: list, size: int) -> list[list]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def default_base_url() -> str:
    port = os.environ.get("ALBEDO_JUDGE_API_PORT", "8091")
    return os.environ.get("ALBEDO_JUDGE_BASE_URL", f"http://127.0.0.1:{port}")


def base_messages(row: dict) -> list[dict[str, str]]:
    turns = row.get("previous_king_turns") or row.get("challenger_turns") or []
    messages: list[dict[str, str]] = []
    for turn in turns:
        if turn.get("score_target"):
            break
        role = str(turn.get("role") or "user")
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(turn.get("content") or "")
        if content:
            messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError(f"sample {row.get('sample_id')} does not include stored context turns")
    return messages


def infer_turn_count(rows: list[dict]) -> int:
    counts = [
        sum(
            1
            for turn in (row.get("previous_king_turns") or [])
            if turn.get("role") == "assistant" and turn.get("score_target")
        )
        for row in rows
    ]
    counts = [count for count in counts if count]
    return max(counts) if counts else 1


def assistant_submitted(output: str) -> bool:
    return COMPLETE_MARKER in output


def completion_observation(sample_id: str) -> str:
    if "mini-coder" in sample_id.casefold():
        return f"<returncode>0</returncode>\n<output>\n{COMPLETE_MARKER}\n</output>"
    return f"Observation: {COMPLETE_MARKER}"


async def generate_one(
    client: OpenRouterJudgeClient,
    *,
    model: str,
    sample_id: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> dict:
    raw = await client.complete(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        "sample_id": sample_id,
        "text": raw.raw.strip(),
        "provider": raw.provider,
        "error": raw.error,
    }


async def generate_batch(
    client: OpenRouterJudgeClient,
    *,
    model: str,
    active: list[dict],
    temperature: float,
    max_tokens: int,
) -> list[dict]:
    return await asyncio.gather(
        *[
            generate_one(
                client,
                model=model,
                sample_id=item["sample_id"],
                messages=item["messages"],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for item in active
        ]
    )


async def simulate_observations(
    http: httpx.AsyncClient,
    *,
    eval_run_id: str,
    batch: list[dict],
    generation_by_id: dict[str, dict],
) -> dict[str, dict]:
    async def simulate(item: dict) -> tuple[str, dict]:
        result = generation_by_id[item["sample_id"]]
        if result.get("error"):
            return item["sample_id"], {"observation": "", "error": result["error"]}
        if assistant_submitted(result["text"]):
            return item["sample_id"], {"observation": completion_observation(item["sample_id"])}
        body = await post_json_with_429_retry(
            http,
            "/simulate-observation",
            {
                "eval_run_id": eval_run_id,
                "sample_id": item["sample_id"],
                "prompt": item["prompt"],
                "messages": item["messages"],
                "assistant_output": result["text"],
            },
        )
        return item["sample_id"], {"observation": body["observation"]}

    pairs = await asyncio.gather(*[simulate(item) for item in batch])
    return dict(pairs)


async def post_json_with_429_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict,
    *,
    retry_count: int = 5,
    base_backoff_seconds: float = 1.5,
) -> dict:
    for attempt in range(retry_count + 1):
        response = await client.post(endpoint, json=payload)
        if response.status_code != 429 or attempt >= retry_count:
            response.raise_for_status()
            return response.json()
        await asyncio.sleep(retry_sleep_seconds(response, attempt, base_backoff_seconds))
    raise AssertionError("unreachable")


def retry_sleep_seconds(
    response: httpx.Response,
    attempt: int,
    base_backoff_seconds: float,
) -> float:
    retry_after = retry_after_seconds(response.headers.get("retry-after"))
    backoff = base_backoff_seconds * (2**attempt)
    return max(retry_after, backoff)


def retry_after_seconds(value: str | None) -> float:
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


async def main_async() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("generated_samples", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="z-ai/glm-5.2")
    parser.add_argument("--turns", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--judge-count", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument("--auth-token", default=os.environ.get("ALBEDO_JUDGE_API_AUTH_TOKEN", ""))
    parser.add_argument("--eval-run-id", default=f"openrouter-vs-king-{uuid4()}")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    rows = [row for row in load_jsonl(args.generated_samples) if not row.get("king_error")]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no rows with valid king output")

    turn_count = args.turns or infer_turn_count(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "progress.jsonl"
    progress_path.write_text("")

    generated_rows = [
        {
            "eval_run_id": args.eval_run_id,
            "sample_id": row["sample_id"],
            "prompt": row["prompt"],
            "previous_king_output": row["previous_king_output"],
            "previous_king_turns": row.get("previous_king_turns"),
            "challenger_model": args.model,
            "challenger_output": "",
            "challenger_turns": None,
            "king_error": row.get("king_error"),
            "chal_error": None,
        }
        for row in rows
    ]
    row_by_id = {row["sample_id"]: row for row in rows}
    active = [
        {
            "sample_id": row["sample_id"],
            "prompt": row["prompt"],
            "messages": base_messages(row),
        }
        for row in rows
    ]
    turns_by_id = {
        item["sample_id"]: [
            {"role": message["role"], "content": message["content"]}
            for message in item["messages"]
        ]
        for item in active
    }
    errors_by_id: dict[str, str] = {}

    headers = {"Authorization": f"Bearer {args.auth_token}"} if args.auth_token else {}
    judge_models = list(JUDGE_MODELS[: args.judge_count])
    started_at = time.monotonic()
    log(
        f"openrouter-vs-king start eval_run_id={args.eval_run_id} model={args.model} "
        f"samples={len(rows)} turns={turn_count} generation_batch_size={args.generation_batch_size} "
        f"score_batch_size={args.score_batch_size} judges={judge_models}"
    )
    append_progress(
        progress_path,
        {
            "type": "run_started",
            "eval_run_id": args.eval_run_id,
            "model": args.model,
            "sample_count": len(rows),
            "turns": turn_count,
            "judge_models": judge_models,
        },
    )

    settings = JudgeSettings(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_timeout_seconds=args.timeout,
    )
    async with (
        OpenRouterJudgeClient(settings) as or_client,
        httpx.AsyncClient(
            base_url=args.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(args.timeout),
        ) as judge_http,
    ):
        for turn_index in range(1, turn_count + 1):
            if not active:
                log(f"generation stopped early turn={turn_index} active=0")
                break
            next_active: list[dict] = []
            turn_batches = chunks(active, args.generation_batch_size)
            for batch_index, batch in enumerate(turn_batches, start=1):
                batch_started = time.monotonic()
                log(
                    f"generation turn {turn_index}/{turn_count} batch {batch_index}/{len(turn_batches)} "
                    f"samples={len(batch)}"
                )
                results = await generate_batch(
                    or_client,
                    model=args.model,
                    active=batch,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                result_by_id = {result["sample_id"]: result for result in results}
                for result in results:
                    sample_id = result["sample_id"]
                    if result.get("error"):
                        errors_by_id[sample_id] = result["error"]
                        continue
                    turns_by_id[sample_id].append(
                        {"role": "assistant", "content": result["text"], "score_target": True}
                    )
                needs_observation = [
                    item
                    for item in batch
                    if result_by_id.get(item["sample_id"])
                    and not result_by_id[item["sample_id"]].get("error")
                    and (turn_index < turn_count or assistant_submitted(result_by_id[item["sample_id"]]["text"]))
                ]
                if needs_observation:
                    observations = await simulate_observations(
                        judge_http,
                        eval_run_id=args.eval_run_id,
                        batch=needs_observation,
                        generation_by_id=result_by_id,
                    )
                else:
                    observations = {}
                if turn_index < turn_count or observations:
                    for item in batch:
                        sample_id = item["sample_id"]
                        result = result_by_id.get(sample_id)
                        observation = observations.get(sample_id, {})
                        if result is None or result.get("error"):
                            continue
                        if observation.get("error"):
                            errors_by_id[sample_id] = observation["error"]
                            continue
                        obs_text = str(observation.get("observation") or "")
                        turns_by_id[sample_id].append(
                            {
                                "role": "user",
                                "content": obs_text,
                                "environment_observation": True,
                            }
                        )
                        if assistant_submitted(result["text"]):
                            continue
                        next_active.append(
                            {
                                "sample_id": sample_id,
                                "prompt": item["prompt"],
                                "messages": item["messages"]
                                + [
                                    {"role": "assistant", "content": result["text"]},
                                    {"role": "user", "content": obs_text},
                                ],
                            }
                        )
                log(
                    f"generation turn {turn_index}/{turn_count} batch {batch_index}/{len(turn_batches)} "
                    f"done errors={sum(1 for r in results if r.get('error'))} "
                    f"elapsed_s={time.monotonic() - batch_started:.1f}"
                )
            active = next_active
            append_progress(
                progress_path,
                {
                    "type": "generation_turn_done",
                    "turn": turn_index,
                    "remaining_active": len(active),
                    "errors": len(errors_by_id),
                },
            )

        for row in generated_rows:
            sample_id = row["sample_id"]
            row["chal_error"] = errors_by_id.get(sample_id)
            if not row["chal_error"]:
                row["challenger_turns"] = turns_by_id[sample_id]
                row["challenger_output"] = format_scored_trajectory(turns_by_id[sample_id])

        samples = [
            {
                "sample_id": row["sample_id"],
                "prompt": row_by_id[row["sample_id"]]["prompt"],
                "previous_king_output": row["previous_king_output"],
                "challenger_output": row["challenger_output"],
            }
            for row in generated_rows
            if not row.get("king_error") and not row.get("chal_error")
        ]

        log("question prep request started")
        prep_body = await post_json_with_429_retry(
            judge_http,
            "/category-prep",
            {
                "eval_run_id": args.eval_run_id,
                "batch_id": "category-prep",
                "total_sample_count": len(samples),
                "samples": [
                    {"sample_id": sample["sample_id"], "prompt": sample["prompt"]}
                    for sample in samples
                ],
            },
        )
        prep_id = prep_body.get("category_prep_id")
        log(f"question prep accepted prep_id={prep_id}")

        records: list[dict] = []
        score_batches = chunks(samples, args.score_batch_size)
        for index, batch in enumerate(score_batches, start=1):
            batch_started = time.monotonic()
            log(
                f"score batch {index}/{len(score_batches)} started "
                f"samples={len(batch)} completed={len(records)}/{len(samples)}"
            )
            body = await post_json_with_429_retry(
                judge_http,
                "/score-batch",
                {
                    "eval_run_id": args.eval_run_id,
                    "batch_id": f"score-{index:04d}",
                    "total_sample_count": len(samples),
                    "judge_models": judge_models,
                    "category_prep_id": prep_id,
                    "samples": batch,
                },
            )
            batch_records = body.get("scoring_records", [])
            if not isinstance(batch_records, list):
                raise ValueError("judge API returned non-list scoring_records")
            records.extend(batch_records)
            summary = body.get("summary", {})
            log(
                f"score batch {index}/{len(score_batches)} done "
                f"batch_scored={sum(1 for record in batch_records if record.get('scored'))}/{len(batch)} "
                f"king={summary.get('score_king')} chal={summary.get('score_challenger')} "
                f"elapsed_s={time.monotonic() - batch_started:.1f}"
            )
            append_progress(
                progress_path,
                {
                    "type": "score_batch_done",
                    "batch_index": index,
                    "batch_count": len(score_batches),
                    "batch_summary": summary,
                    "records_done": len(records),
                    "elapsed_s": round(time.monotonic() - batch_started, 3),
                },
            )

    judge_rows = [
        {"sample_id": record["sample_id"], **judge_result}
        for record in records
        for judge_result in record.get("judge_results", [])
    ]
    summary = aggregate_scores(records)
    summary["eval_run_id"] = args.eval_run_id
    summary["judge_count"] = args.judge_count
    summary["generated_sample_count"] = len(generated_rows)
    summary["valid_generated_pair_count"] = len(samples)
    summary["challenger_model"] = args.model

    write_jsonl(args.out_dir / "generated-samples.jsonl", generated_rows)
    write_jsonl(args.out_dir / "scoring-results.jsonl", records)
    write_jsonl(args.out_dir / "judge-results.jsonl", judge_rows)
    (args.out_dir / "verdict.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    append_progress(progress_path, {"type": "run_done", "summary": summary})
    log(
        f"done valid_pairs={len(samples)}/{len(generated_rows)} "
        f"king={summary.get('score_king')} chal={summary.get('score_challenger')} "
        f"elapsed_s={time.monotonic() - started_at:.1f}"
    )
    log(f"wrote {args.out_dir / 'generated-samples.jsonl'}")
    log(f"wrote {args.out_dir / 'scoring-results.jsonl'}")
    log(f"wrote {args.out_dir / 'judge-results.jsonl'}")
    log(f"wrote {args.out_dir / 'verdict.json'}")
    log(f"wrote {progress_path}")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

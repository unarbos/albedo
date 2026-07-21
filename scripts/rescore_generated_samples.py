#!/usr/bin/env python3
"""Rescore an existing generated-samples.jsonl through the judge API."""

from __future__ import annotations

import argparse
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

from albedo_eval_service.judge_core import JUDGE_MODELS, aggregate_scores


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


def post_json_with_429_retry(
    client: httpx.Client,
    endpoint: str,
    payload: dict,
    *,
    retry_count: int = 5,
    base_backoff_seconds: float = 1.5,
) -> dict:
    for attempt in range(retry_count + 1):
        response = client.post(endpoint, json=payload)
        if response.status_code != 429 or attempt >= retry_count:
            response.raise_for_status()
            return response.json()
        time.sleep(retry_sleep_seconds(response, attempt, base_backoff_seconds))
    raise AssertionError("unreachable")


def retry_sleep_seconds(response: httpx.Response, attempt: int, base_backoff_seconds: float) -> float:
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


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("generated_samples", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument("--auth-token", default=os.environ.get("ALBEDO_JUDGE_API_AUTH_TOKEN", ""))
    parser.add_argument("--eval-run-id", default=f"rescore-{uuid4()}")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--judge-count", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_jsonl(args.generated_samples)
    if args.limit:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "progress.jsonl"
    progress_path.write_text("")

    samples = [
        {
            "sample_id": row["sample_id"],
            "prompt": row["prompt"],
            "previous_king_output": row["previous_king_output"],
            "challenger_output": row["challenger_output"],
        }
        for row in rows
        if not row.get("king_error") and not row.get("chal_error")
    ]
    headers = {"Authorization": f"Bearer {args.auth_token}"} if args.auth_token else {}
    judge_models = list(JUDGE_MODELS[: args.judge_count])
    started_at = time.monotonic()
    log(
        f"rescore start eval_run_id={args.eval_run_id} samples={len(samples)}/{len(rows)} "
        f"batch_size={args.batch_size} judges={judge_models} base_url={args.base_url.rstrip('/')}"
    )
    append_progress(
        progress_path,
        {
            "type": "rescore_started",
            "eval_run_id": args.eval_run_id,
            "sample_count": len(samples),
            "generated_row_count": len(rows),
            "batch_size": args.batch_size,
            "judge_models": judge_models,
        },
    )

    with httpx.Client(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        timeout=httpx.Timeout(args.timeout),
    ) as client:
        log("question prep request started")
        prep_started = time.monotonic()
        prep_payload = {
            "eval_run_id": args.eval_run_id,
            "batch_id": "category-prep",
            "total_sample_count": len(samples),
            "samples": [
                {"sample_id": sample["sample_id"], "prompt": sample["prompt"]}
                for sample in samples
            ],
        }
        prep_body = post_json_with_429_retry(client, "/category-prep", prep_payload)
        prep_id = prep_body.get("category_prep_id")
        log(f"question prep accepted prep_id={prep_id} elapsed_s={time.monotonic() - prep_started:.1f}")
        append_progress(
            progress_path,
            {
                "type": "question_prep_accepted",
                "eval_run_id": args.eval_run_id,
                "category_prep_id": prep_id,
                "elapsed_s": round(time.monotonic() - prep_started, 3),
            },
        )

        records: list[dict] = []
        batches = chunks(samples, args.batch_size)
        for index, batch in enumerate(batches, start=1):
            batch_started = time.monotonic()
            batch_id = f"rescore-{index:04d}"
            log(
                f"batch {index}/{len(batches)} started batch_id={batch_id} "
                f"samples={len(batch)} completed={len(records)}/{len(samples)}"
            )
            payload = {
                "eval_run_id": args.eval_run_id,
                "batch_id": batch_id,
                "total_sample_count": len(samples),
                "judge_models": judge_models,
                "category_prep_id": prep_id,
                "samples": batch,
            }
            body = post_json_with_429_retry(client, "/score-batch", payload)
            batch_records = body.get("scoring_records", [])
            if not isinstance(batch_records, list):
                raise ValueError("judge API returned non-list scoring_records")
            records.extend(batch_records)
            batch_scored = sum(1 for record in batch_records if record.get("scored"))
            total_scored = sum(1 for record in records if record.get("scored"))
            unscored = [
                {
                    "sample_id": record.get("sample_id"),
                    "error": record.get("error"),
                }
                for record in batch_records
                if not record.get("scored")
            ]
            batch_summary = body.get("summary", {})
            log(
                f"batch {index}/{len(batches)} done batch_scored={batch_scored}/{len(batch)} "
                f"total_scored={total_scored}/{len(records)} total_done={len(records)}/{len(samples)} "
                f"batch_state={batch_summary.get('state')} batch_king={batch_summary.get('score_king')} "
                f"batch_chal={batch_summary.get('score_challenger')} "
                f"elapsed_s={time.monotonic() - batch_started:.1f}"
            )
            for item in unscored:
                log(f"unscored sample_id={item['sample_id']} error={item['error']}")
            append_progress(
                progress_path,
                {
                    "type": "score_batch_done",
                    "eval_run_id": args.eval_run_id,
                    "batch_id": batch_id,
                    "batch_index": index,
                    "batch_count": len(batches),
                    "batch_sample_count": len(batch),
                    "records_done": len(records),
                    "sample_count": len(samples),
                    "batch_scored": batch_scored,
                    "total_scored": total_scored,
                    "unscored": unscored,
                    "batch_summary": batch_summary,
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
    summary["generated_sample_count"] = len(rows)
    summary["scored_sample_count"] = sum(1 for record in records if record.get("scored"))

    write_jsonl(args.out_dir / "scoring-results.jsonl", records)
    write_jsonl(args.out_dir / "judge-results.jsonl", judge_rows)
    (args.out_dir / "verdict.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    append_progress(progress_path, {"type": "rescore_done", "summary": summary})
    log(
        f"rescore done scored={summary['scored_sample_count']}/{len(samples)} "
        f"king={summary.get('score_king')} chal={summary.get('score_challenger')} "
        f"elapsed_s={time.monotonic() - started_at:.1f}"
    )
    log(f"wrote {args.out_dir / 'scoring-results.jsonl'}")
    log(f"wrote {args.out_dir / 'judge-results.jsonl'}")
    log(f"wrote {args.out_dir / 'verdict.json'}")
    log(f"wrote {progress_path}")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

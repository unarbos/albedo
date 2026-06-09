from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import SETTINGS


def publication_plan(*, king: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Return public S3 keys/URLs; actual upload is performed by the host uploader."""
    base_url = _public_base_url()
    run_id = str(result.get("run_id") or "")
    prefix = _prefix()
    run_key = f"{prefix}/runs/{run_id}"
    urls = {
        "index": f"{base_url}/{prefix}/index.json",
        "state": f"{base_url}/{prefix}/state.json",
        "summary": f"{base_url}/{run_key}/summary.json",
        "predictions": f"{base_url}/{run_key}/predictions.jsonl",
        "raw_generations": f"{base_url}/{run_key}/raw_generations.json",
        "official_report": f"{base_url}/{run_key}/official-report.json",
        "king": f"{base_url}/{prefix}/kings/{king_slug(king)}.json",
    }
    return {
        "mode": "host",
        "enabled": SETTINGS.s3_enabled,
        "uploaded": False,
        "pending_host_upload": SETTINGS.s3_enabled,
        "bucket": SETTINGS.s3_bucket,
        "endpoint": SETTINGS.s3_endpoint,
        "prefix": prefix,
        "public_base_url": base_url,
        "urls": urls,
        "skipped_reason": "" if SETTINGS.s3_enabled else "disabled by ALBEDO_SWEBENCH_S3_ENABLED",
    }


def build_index(state: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for key, row in state.get("benchmarks", {}).items():
        king = row.get("king") or {}
        s3 = row.get("s3") or {}
        rows.append({
            "king_key": key,
            "repo": king.get("repo"),
            "digest": king.get("digest"),
            "challenge_id": king.get("challenge_id"),
            "reign_number": king.get("reign_number"),
            "crowned_at": king.get("crowned_at"),
            "source": king.get("source"),
            "status": row.get("status"),
            "run_id": row.get("run_id"),
            "started_at": row.get("started_at"),
            "completed_at": row.get("completed_at"),
            "resolved": row.get("resolved"),
            "total": row.get("total"),
            "score": row.get("score"),
            "submitted": row.get("submitted"),
            "completed": row.get("completed"),
            "empty_patches": row.get("empty_patches"),
            "errors": row.get("errors"),
            "s3": s3,
            "s3_urls": s3.get("urls") or {},
        })
    rows.sort(
        key=lambda item: (
            str(item.get("crowned_at") or ""),
            int(item.get("reign_number") or -1),
            str(item.get("completed_at") or item.get("started_at") or ""),
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset": SETTINGS.dataset_name,
        "split": SETTINGS.split,
        "agent": {
            "runner": "mini-swe-agent",
            "config": "swebench_backticks.yaml",
            "model_class": "litellm_textbased",
            "temperature": SETTINGS.generation_temperature,
        },
        "count": len(rows),
        "benchmarks": rows,
    }


def run_summary(*, king: Any, result: dict[str, Any], plan: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = plan or publication_plan(king=king, result=result)
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset": SETTINGS.dataset_name,
        "split": SETTINGS.split,
        "king": king.to_dict() if hasattr(king, "to_dict") else dict(getattr(king, "__dict__", {})),
        "run_id": result.get("run_id"),
        "status": result.get("status", "complete"),
        "resolved": result.get("resolved"),
        "total": result.get("total"),
        "score": result.get("score"),
        "submitted": result.get("submitted"),
        "completed": result.get("completed"),
        "unresolved": result.get("unresolved"),
        "empty_patches": result.get("empty_patches"),
        "errors": result.get("errors"),
        "harness_returncode": result.get("harness_returncode"),
        "mini_returncode": result.get("mini_returncode"),
        "paths": {
            "predictions": result.get("predictions_path"),
            "raw_generations": result.get("raw_generations_path"),
            "official_report": result.get("summary_path"),
            "report_dir": result.get("report_dir"),
            "run_dir": result.get("run_dir"),
        },
        "urls": plan.get("urls") or {},
    }


def king_slug(king: Any) -> str:
    digest = (king.digest or "").replace("sha256:", "")[:16]
    repo = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in king.repo)[-64:]
    return f"{repo}-{digest}"


def _prefix() -> str:
    return SETTINGS.s3_prefix or "swebench-lite"


def _public_base_url() -> str:
    if SETTINGS.s3_public_base_url:
        return SETTINGS.s3_public_base_url.rstrip("/")
    return f"https://us-east-1.hippius.com/{SETTINGS.s3_bucket}".rstrip("/")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

#!/usr/bin/env python3
"""Watch the eval Postgres DB and publish the website's data files.

Builds two files into website/data/ and uploads them to Hippius whenever the DB changes:
  - dashboard.json : reign, eval history, score chart, queue, fails (the rich view)
  - state.json     : live pipeline status across hippius_validate / pre_eval / eval (running vs queued)

Self-contained (like push_to_hippius.py): reads ../.env, talks to Postgres via psycopg, uploads via
boto3. Change detection is a cheap signature poll, so a tick only does work when something changed.

Run as a long-running loop (default, for PM2) or `--once`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

WEBSITE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBSITE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = WEBSITE_DIR / "data"

log = logging.getLogger("monitor")

# Mirrors albedo_eval_service.judge_core.JUDGE_MODELS (kept inline so this script stays standalone).
JUDGE_MODELS = ["z-ai/glm-5.1", "qwen/qwen3.5-397b-a17b", "deepseek/deepseek-v3.2"]

# Artifact types the website knows how to render (website/js/config.js ARTIFACT_TYPES).
DASHBOARD_ARTIFACT_TYPES = [
    "EVAL_VERDICT", "GENERATED_SAMPLES", "SCORING_RESULTS", "JUDGE_RESULTS",
    "EVAL_TRANSCRIPT", "REMOTE_PROGRESS", "REMOTE_LOGS", "SANITY_RESULT",
]

QUEUE_STATES = ("PRE_EVAL_QUEUED", "PRE_EVAL_RUNNING", "PRE_EVAL_PASSED", "EVAL_QUEUED", "EVAL_RUNNING")
FAIL_STATES = ("TERMINAL_INVALID", "TERMINAL_INFRA_FAILED")
ACTIVE_EVAL_STATES = ("QUEUED", "DISPATCHED", "GENERATING", "SCORING", "VERDICT_READY")

# state.json: per-stage running/queued buckets. Handoff states (HIPPIUS_VALIDATED, PRE_EVAL_PASSED) are
# "queued for the next stage" — that's exactly what the next-stage dispatcher claims.
STAGE_BUCKETS: dict[str, dict[str, tuple[str, ...]]] = {
    "hippius_validate": {"queued": ("SUBMITTED", "HIPPIUS_RETRYABLE"), "running": ("HIPPIUS_RUNNING",)},
    "pre_eval": {"queued": ("HIPPIUS_VALIDATED", "PRE_EVAL_QUEUED", "PRE_EVAL_RETRYABLE"), "running": ("PRE_EVAL_RUNNING",)},
    "eval": {"queued": ("PRE_EVAL_PASSED", "EVAL_QUEUED", "EVAL_RETRYABLE"), "running": ("EVAL_RUNNING",)},
}


def load_env(path: Path) -> None:
    if not path.is_file():
        print(f"warning: {path} not found; relying on existing environment", file=sys.stderr)
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"not JSON-serializable: {type(value).__name__}")


def _public_url(uri: str | None, base: str) -> str | None:
    if not uri:
        return None
    if uri.startswith("s3://"):
        return f"{base.rstrip('/')}/{uri[len('s3://'):]}"
    if uri.startswith(("http://", "https://")):
        return uri
    return None  # local-cache://, file:// — not browser-fetchable


# --------------------------------------------------------------------------- dashboard.json


def _reign(conn) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT rm.slot, rm.uid, rm.hotkey, rm.weight_bps, rm.model_hash,
               kv.version AS king_version, ms.model_uri,
               er.score_challenger, er.score_king, er.id AS eval_run_id
        FROM reigns r
        JOIN reign_members rm ON rm.reign_id = r.id
        JOIN king_versions kv ON kv.id = rm.king_version_id
        JOIN model_submissions ms ON ms.id = rm.submission_id
        LEFT JOIN eval_runs er ON er.id = kv.eval_run_id
        WHERE r.state = 'ACTIVE'
        ORDER BY rm.slot ASC
        """
    ).fetchall()
    members = [
        {
            "king_version": row["king_version"],
            "model_uri": row["model_uri"],
            "model_hash": row["model_hash"],
            "hotkey": row["hotkey"],
            "uid": row["uid"],
            "weight_bps": row["weight_bps"],
            "score_challenger": _num(row["score_challenger"]),
            "score_king": _num(row["score_king"]),
            "eval_run_id": str(row["eval_run_id"]) if row["eval_run_id"] else None,
        }
        for row in rows
    ]
    return {"members": members}


def _artifacts_for(conn, submission_ids: list, base: str) -> dict[str, dict[str, str]]:
    if not submission_ids:
        return {}
    rows = conn.execute(
        """
        SELECT submission_id, artifact_type, uri
        FROM artifacts
        WHERE submission_id = ANY(%s) AND artifact_type = ANY(%s)
        """,
        (submission_ids, DASHBOARD_ARTIFACT_TYPES),
    ).fetchall()
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        url = _public_url(row["uri"], base)
        if not url:
            continue
        out.setdefault(str(row["submission_id"]), {})[row["artifact_type"]] = url
    return out


def _eval_runs(conn, *, limit: int, base: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT er.id AS eval_run_id, er.submission_id,
               er.challenger_won, er.score_challenger, er.score_king, er.win_margin,
               er.valid_turns, er.total_turns, er.chal_vllm_errors, er.king_vllm_errors,
               er.finished_at,
               ms.uid, ms.hotkey, ms.model_uri,
               sa.result_summary,
               ckv.version AS crowned_king_version,
               kms.uid AS king_uid, kms.hotkey AS king_hotkey, kms.model_uri AS king_model_uri,
               kkv.version AS king_king_version
        FROM eval_runs er
        JOIN model_submissions ms ON ms.id = er.submission_id
        LEFT JOIN stage_attempts sa ON sa.id = er.stage_attempt_id
        LEFT JOIN king_versions ckv ON ckv.eval_run_id = er.id
        LEFT JOIN model_submissions kms ON kms.id = er.king_submission_id
        LEFT JOIN LATERAL (
            SELECT version FROM king_versions
            WHERE submission_id = er.king_submission_id
            ORDER BY version DESC LIMIT 1
        ) kkv ON true
        WHERE er.state = 'SUCCEEDED'
        ORDER BY er.finished_at DESC NULLS LAST
        LIMIT %s
        """,
        (limit,),
    ).fetchall()

    artifacts = _artifacts_for(conn, [row["submission_id"] for row in rows], base)

    runs: list[dict[str, Any]] = []
    for row in rows:
        verdict = row["result_summary"] if isinstance(row["result_summary"], dict) else {}
        breakdown = verdict.get("score_breakdown") if isinstance(verdict, dict) else None
        breakdown = breakdown if isinstance(breakdown, dict) else {}
        runs.append(
            {
                "eval_run_id": str(row["eval_run_id"]),
                "challenger_won": row["challenger_won"],
                "coronated": row["crowned_king_version"] is not None,
                "king_version": row["crowned_king_version"],
                "score_challenger": _num(row["score_challenger"]),
                "score_king": _num(row["score_king"]),
                "win_margin": _num(row["win_margin"]),
                "finished_at": row["finished_at"],
                "model_uri": row["model_uri"],
                "hotkey": row["hotkey"],
                "uid": row["uid"],
                "total_turns": row["total_turns"],
                "valid_turns": row["valid_turns"],
                "chal_vllm_errors": row["chal_vllm_errors"],
                "king_vllm_errors": row["king_vllm_errors"],
                "score_breakdown": {
                    "by_judge": breakdown.get("by_judge", {}),
                    "by_metric": breakdown.get("by_metric", {}),
                },
                "king": {
                    "king_version": row["king_king_version"],
                    "model_uri": row["king_model_uri"],
                    "uid": row["king_uid"],
                    "hotkey": row["king_hotkey"],
                },
                "artifacts": artifacts.get(str(row["submission_id"]), {}),
            }
        )
    return runs


def _current_eval(conn) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT er.id AS eval_run_id, er.state, er.sample_count, er.generated_sample_count,
               er.started_at, ms.id AS submission_id, ms.model_uri, ms.hotkey, ms.uid
        FROM eval_runs er
        JOIN model_submissions ms ON ms.id = er.submission_id
        WHERE er.state = ANY(%s)
        ORDER BY er.started_at DESC NULLS LAST
        LIMIT 1
        """,
        (list(ACTIVE_EVAL_STATES),),
    ).fetchone()
    if not row:
        return None
    return {
        "eval_run_id": str(row["eval_run_id"]),
        "submission_id": str(row["submission_id"]),
        "state": row["state"],
        "sample_count": row["sample_count"],
        "generated_sample_count": row["generated_sample_count"],
        "started_at": row["started_at"],
        "model_uri": row["model_uri"],
        "hotkey": row["hotkey"],
        "uid": row["uid"],
    }


def _queue(conn, *, exclude_submission_id: str | None) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id AS submission_id, state, model_uri, hotkey, uid, created_at
        FROM model_submissions
        WHERE state = ANY(%s)
        ORDER BY priority ASC, created_at ASC
        """,
        (list(QUEUE_STATES),),
    ).fetchall()
    return [
        {
            "submission_id": str(row["submission_id"]),
            "state": row["state"],
            "model_uri": row["model_uri"],
            "hotkey": row["hotkey"],
            "uid": row["uid"],
            "created_at": row["created_at"],
        }
        for row in rows
        if str(row["submission_id"]) != exclude_submission_id
    ]


def _fails(conn, *, limit: int, base: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT ms.id AS submission_id, ms.state, ms.model_uri, ms.hotkey, ms.uid,
               ms.fault_class, ms.fault_code, ms.fault_message, ms.model_hash, ms.updated_at,
               (SELECT er.id FROM eval_runs er
                WHERE er.submission_id = ms.id
                ORDER BY er.started_at DESC NULLS LAST LIMIT 1) AS eval_run_id
        FROM model_submissions ms
        WHERE ms.state = ANY(%s)
        ORDER BY ms.updated_at DESC
        LIMIT %s
        """,
        (list(FAIL_STATES), limit),
    ).fetchall()
    artifacts = _artifacts_for(conn, [row["submission_id"] for row in rows], base)
    return [
        {
            "submission_id": str(row["submission_id"]),
            "eval_run_id": str(row["eval_run_id"]) if row["eval_run_id"] else None,
            "model_uri": row["model_uri"],
            "hotkey": row["hotkey"],
            "uid": row["uid"],
            "state": row["state"],
            "fault_class": row["fault_class"],
            "fault_code": row["fault_code"],
            "fault_message": row["fault_message"],
            "model_hash": row["model_hash"],
            "updated_at": row["updated_at"],
            "artifacts": artifacts.get(str(row["submission_id"]), {}),
        }
        for row in rows
    ]


def _stats(conn) -> dict[str, Any]:
    row = conn.execute("SELECT count(*) AS n FROM eval_runs WHERE state = 'SUCCEEDED'").fetchone()
    return {"evaluated": int(row["n"]) if row else 0}


def build_dashboard(conn, *, netuid: int, history_limit: int, artifact_base: str) -> dict[str, Any]:
    current = _current_eval(conn)
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "chain": {"netuid": netuid, "judge_models": list(JUDGE_MODELS)},
        "stats": _stats(conn),
        "reign": _reign(conn),
        "current_eval": current,
        "queue": _queue(conn, exclude_submission_id=current["submission_id"] if current else None),
        "eval_runs": _eval_runs(conn, limit=history_limit, base=artifact_base),
        "fails": _fails(conn, limit=history_limit, base=artifact_base),
    }


# --------------------------------------------------------------------------- state.json


def build_state(conn) -> dict[str, Any]:
    tracked = sorted({s for stage in STAGE_BUCKETS.values() for bucket in stage.values() for s in bucket})
    rows = conn.execute(
        """
        SELECT id AS submission_id, state, uid, hotkey, model_uri, updated_at
        FROM model_submissions
        WHERE state = ANY(%s)
        ORDER BY updated_at DESC
        """,
        (tracked,),
    ).fetchall()

    stages: dict[str, dict[str, list]] = {name: {"running": [], "queued": []} for name in STAGE_BUCKETS}
    for row in rows:
        item = {
            "submission_id": str(row["submission_id"]),
            "uid": row["uid"],
            "hotkey": row["hotkey"],
            "model_uri": row["model_uri"],
            "state": row["state"],
            "updated_at": row["updated_at"],
        }
        for stage_name, buckets in STAGE_BUCKETS.items():
            for bucket, states in buckets.items():
                if row["state"] in states:
                    stages[stage_name][bucket].append(item)
    counts = {name: {b: len(items) for b, items in buckets.items()} for name, buckets in stages.items()}
    return {"updated_at": datetime.now(UTC).isoformat(), "counts": counts, "stages": stages}


# --------------------------------------------------------------------------- upload


def _upload_to_hippius(key: str, path: Path) -> bool:
    bucket = os.environ.get("ALBEDO_S3_BUCKET") or "albedo"
    access = os.environ.get("ALBEDO_S3_ACCESS_KEY")
    secret = os.environ.get("ALBEDO_S3_SECRET_KEY")
    if not (access and secret):
        log.warning("ALBEDO_S3_* unset; kept local %s (not uploaded)", path.name)
        return False
    endpoint = os.environ.get("ALBEDO_S3_ENDPOINT") or "https://s3.hippius.com"
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            region_name="decentralized",
            config=Config(connect_timeout=15, read_timeout=60, retries={"mode": "adaptive", "max_attempts": 3}),
        )
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=path.read_bytes(),
            ContentType="application/json",
            CacheControl="no-cache, must-revalidate",
            ACL="public-read",
        )
        return True
    except Exception as exc:  # never wedge the loop on an upload
        log.error("upload failed for %s: %s", key, exc)
        return False


# --------------------------------------------------------------------------- loop


def _signature(conn) -> tuple:
    row = conn.execute(
        """
        SELECT (SELECT max(updated_at) FROM model_submissions) AS ms_max,
               (SELECT count(*)        FROM model_submissions) AS ms_count,
               (SELECT max(finished_at) FROM eval_runs)        AS er_max,
               (SELECT max(version)     FROM reigns)           AS reign_max
        """
    ).fetchone()
    return (row["ms_max"], row["ms_count"], row["er_max"], row["reign_max"])


def generate(*, database_url: str, netuid: int, history_limit: int, artifact_base: str) -> None:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        dashboard = build_dashboard(conn, netuid=netuid, history_limit=history_limit, artifact_base=artifact_base)
        state = build_state(conn)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    uploads: dict[str, bool] = {}
    for name, data in (("dashboard.json", dashboard), ("state.json", state)):
        path = DATA_DIR / name
        path.write_text(json.dumps(data, default=_json_default, indent=2), encoding="utf-8")
        uploads[name] = _upload_to_hippius(f"data/{name}", path)

    members = dashboard["reign"]["members"]
    king_version = max((m["king_version"] for m in members if m.get("king_version") is not None), default=None)
    current = dashboard["current_eval"]
    log.info(
        "published update: evaluated=%s reign_king=v%s eval_runs=%d queued=%d current_eval=%s fails=%d upload=%s",
        dashboard["stats"]["evaluated"],
        king_version,
        len(dashboard["eval_runs"]),
        len(dashboard["queue"]),
        current["state"] if current else "idle",
        len(dashboard["fails"]),
        "ok" if all(uploads.values()) else "FAILED",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish website dashboard.json + state.json from Postgres.")
    parser.add_argument("--once", action="store_true", help="Generate once and exit (default: watch for changes)")
    parser.add_argument("--netuid", type=int, default=None)
    parser.add_argument("--history-limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    load_env(ENV_PATH)
    database_url = os.environ.get("ALBEDO_EVAL_DATABASE_URL")
    if not database_url:
        sys.exit("ALBEDO_EVAL_DATABASE_URL is not set")
    netuid = args.netuid if args.netuid is not None else int(os.environ.get("ALBEDO_DASHBOARD_NETUID", "97"))
    history_limit = args.history_limit if args.history_limit is not None else int(os.environ.get("ALBEDO_DASHBOARD_HISTORY_LIMIT", "200"))
    artifact_base = os.environ.get("ALBEDO_DASHBOARD_ARTIFACT_BASE_URL", "https://s3.hippius.com")
    interval = float(os.environ.get("ALBEDO_MONITOR_INTERVAL_S", "2"))

    def run_once() -> None:
        generate(database_url=database_url, netuid=netuid, history_limit=history_limit, artifact_base=artifact_base)

    if args.once:
        run_once()
        return 0

    import psycopg
    from psycopg.rows import dict_row

    last_sig: tuple | None = None
    while True:
        try:
            with psycopg.connect(database_url, row_factory=dict_row) as conn:
                sig = _signature(conn)
            if sig != last_sig:
                run_once()
                last_sig = sig
        except Exception as exc:  # keep the loop alive; the next tick retries
            log.error("tick failed: %s", exc)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())

"""Hippius S3 upload for pre-eval (sanity) failure reports.

On a terminal pre-eval rejection the dispatcher publishes a per-submission fault.json (the reason,
fault_code, and the full per-judge injection/viability evidence) so it can be linked from the dashboard.

Env-gated and best-effort: when ALBEDO_S3_* is unset the uploader is disabled and calls are no-ops,
so pre-eval still runs without publishing. Mirrors model_validation/uploads/artifacts.py.
"""

from __future__ import annotations

import functools
import json
import os

from loguru import logger as log

S3_BUCKET = os.environ.get("ALBEDO_S3_BUCKET", "")
S3_ENDPOINT = os.environ.get("ALBEDO_S3_ENDPOINT") or "https://s3.hippius.com"
S3_ACCESS_KEY = os.environ.get("ALBEDO_S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("ALBEDO_S3_SECRET_KEY", "")

ENABLED = bool(S3_BUCKET and S3_ACCESS_KEY and S3_SECRET_KEY)


@functools.lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="decentralized",
        config=Config(
            connect_timeout=15, read_timeout=60, retries={"mode": "adaptive", "max_attempts": 3}
        ),
    )


def _safe_digest(digest: str) -> str:
    return digest.replace(":", "_")


def put_sanity_fault(submission_id: str, digest: str, detail: dict) -> str | None:
    """Upload a terminal pre-eval fault report. Returns the s3:// URI, or None if disabled/failed."""
    if not ENABLED:
        log.debug("S3 disabled (ALBEDO_S3_* unset); skipping sanity fault upload for {}", submission_id)
        return None
    key = f"sanity/{submission_id}/{_safe_digest(digest)}/fault.json"
    try:
        _client().put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(detail, default=str).encode(),
            ContentType="application/json",
            ACL="public-read",
        )
        uri = f"s3://{S3_BUCKET}/{key}"
        log.info("uploaded sanity fault {}", uri)
        return uri
    except Exception as exc:  # noqa: BLE001 — never wedge pre-eval on an upload
        log.warning("S3 put({}) failed: {}", key, exc)
        return None

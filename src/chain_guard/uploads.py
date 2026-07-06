"""S3 publishing for chain_guard reuse detections.

Mirrors the hippius_validation fault uploader: boto3 against the ALBEDO_S3_* bucket, public-read
JSON, env-gated no-op when credentials are unset (so chain_reader runs fine without S3).
"""
from __future__ import annotations

import functools
import json
import os

from loguru import logger as log

import chain_reader.config  # noqa: F401 — import side-effect loads albedo/.env into os.environ

S3_BUCKET = os.environ.get("ALBEDO_S3_BUCKET", "")
S3_ENDPOINT = os.environ.get("ALBEDO_S3_ENDPOINT", "https://s3.hippius.com")
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
        config=Config(connect_timeout=15, read_timeout=60,
                      retries={"mode": "adaptive", "max_attempts": 3}),
    )


def put_detection(hotkey: str, block: int, detail: dict) -> str | None:
    """Upload a reuse-detection report. No-op (None) when S3 is unconfigured."""
    if not ENABLED:
        log.debug("S3 disabled (ALBEDO_S3_* unset); skipping detection upload for {}", hotkey)
        return None
    key = f"chain_guard/{hotkey}/{block}/detection.json"
    log.debug(f"[chain-guard] uploading detection hotkey={hotkey} block={block} key={key}")
    try:
        _client().put_object(
            Bucket=S3_BUCKET, Key=key,
            Body=json.dumps(detail, default=str).encode(),
            ContentType="application/json", ACL="public-read",
        )
        uri = f"s3://{S3_BUCKET}/{key}"
        log.info("uploaded detection {}", uri)
        return uri
    except Exception as exc:  # noqa: BLE001 — never wedge the reader on an upload
        log.warning("S3 put({}) failed: {}", key, exc)
        return None

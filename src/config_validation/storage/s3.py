from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://s3.hippius.com"
_REGION = "decentralized"

_BUCKET = os.environ.get("CV_S3_BUCKET", "")
_ENDPOINT = os.environ.get("CV_S3_ENDPOINT", _DEFAULT_ENDPOINT)
_ACCESS_KEY = os.environ.get("CV_S3_ACCESS_KEY", "")
_SECRET_KEY = os.environ.get("CV_S3_SECRET_KEY", "")

ENABLED = bool(_BUCKET and _ACCESS_KEY and _SECRET_KEY)


def _client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=_ENDPOINT,
        aws_access_key_id=_ACCESS_KEY,
        aws_secret_access_key=_SECRET_KEY,
        region_name=_REGION,
        config=Config(
            connect_timeout=15, read_timeout=45, retries={"mode": "adaptive", "max_attempts": 3}
        ),
    )


def put_jsonl(key: str, body: bytes) -> bool:
    if not ENABLED:
        log.debug("hippius S3 disabled (CV_S3_* unset); skipping put(%r)", key)
        return False
    try:
        _client().put_object(
            Bucket=_BUCKET,
            Key=key,
            Body=body,
            ContentType="application/x-ndjson",
            ACL="public-read",
            CacheControl="no-cache, must-revalidate",
        )
        log.info("hippius S3: uploaded %d bytes to s3://%s/%s", len(body), _BUCKET, key)
        return True
    except Exception as exc:
        log.warning("hippius S3 put(%r) failed: %s", key, exc)
        return False

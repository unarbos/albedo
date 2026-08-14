from __future__ import annotations

import functools
import json

from loguru import logger as log

from albedo_config import get_model_validation_settings

config = get_model_validation_settings()

ENABLED = bool(config.S3_BUCKET and config.S3_ACCESS_KEY and config.S3_SECRET_KEY)


@functools.lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        aws_access_key_id=config.S3_ACCESS_KEY,
        aws_secret_access_key=config.S3_SECRET_KEY,
        region_name="auto",
        config=Config(
            connect_timeout=15, read_timeout=60, retries={"mode": "adaptive", "max_attempts": 3}
        ),
    )


def _safe_digest(digest: str) -> str:
    return digest.replace(":", "_")


def _put(key: str, data: dict) -> str | None:
    if not ENABLED:
        log.debug("S3 disabled (ALBEDO_S3_* unset); skipping put({})", key)
        return None
    try:
        _client().put_object(
            Bucket=config.S3_BUCKET,
            Key=key,
            Body=json.dumps(data, default=str).encode(),
            ContentType="application/json",
            ACL="public-read",
        )
        uri = f"s3://{config.S3_BUCKET}/{key}"
        log.info("uploaded artifact {}", uri)
        return uri
    except Exception as exc:
        log.warning("S3 put({}) failed: {}", key, exc)
        return None


def _get_json(key: str) -> dict:
    try:
        obj = _client().get_object(Bucket=config.S3_BUCKET, Key=key)
        data = json.loads(obj["Body"].read())
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.debug(f"S3 get_json({key}) failed, starting empty: {exc}")
        return {}


def update_fingerprint_corpus(model_uri: str, fingerprint: dict) -> tuple[str | None, str | None]:
    if not ENABLED:
        log.debug("S3 disabled; skipping corpus update for {}", model_uri)
        return None, None

    fkey, tkey = config.FP_FILE, config.TENSORS_FILE
    fdict = _get_json(fkey)
    fdict[model_uri] = {
        "method": fingerprint.get("method"),
        "layer_keys": fingerprint.get("layer_keys"),
        "norm_vector": fingerprint.get("norm_vector"),
    }
    f_uri = _put(fkey, fdict)

    tdict = _get_json(tkey)
    tdict[model_uri] = {
        "layer_keys": fingerprint.get("layer_keys"),
        "tensor_samples": fingerprint.get("tensor_samples"),
    }
    t_uri = _put(tkey, tdict)
    log.info("corpus updated: {} (now {} fingerprints)", model_uri, len(fdict))
    return f_uri, t_uri


def put_fault(hotkey: str, digest: str, detail: dict) -> str | None:
    key = f"hippius_validation/{hotkey}/{_safe_digest(digest)}/fault.json"
    return _put(key, detail)

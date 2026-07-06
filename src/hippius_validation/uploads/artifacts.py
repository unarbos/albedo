"""Hippius S3 artifacts.

Two kinds of output, both in the `albedo` bucket:
  1. The fingerprint corpus — TWO aggregate files updated for every fingerprinted model:
       fingerprint.json  -> {model_uri: {method, layer_keys, norm_vector}}   (the "rest")
       tensors.json      -> {model_uri: {layer_keys, tensor_samples}}         (the tensors)
  2. Per-model fault.json — on a terminal miner fault, with the full explanation
     (for a duplicate this includes the matched model + similarity + the fingerprint evidence).

Env-gated and best-effort: when ALBEDO_S3_* is unset the uploader is disabled and the calls
are no-ops, so validation still runs without publishing.
"""
from __future__ import annotations

import functools
import json

from loguru import logger as log

from hippius_validation import config

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
        region_name="decentralized",
        config=Config(connect_timeout=15, read_timeout=60,
                      retries={"mode": "adaptive", "max_attempts": 3}),
    )


def _safe_digest(digest: str) -> str:
    return digest.replace(":", "_")


def _put(key: str, data: dict) -> str | None:
    if not ENABLED:
        log.debug("S3 disabled (ALBEDO_S3_* unset); skipping put({})", key)
        return None
    try:
        _client().put_object(
            Bucket=config.S3_BUCKET, Key=key,
            Body=json.dumps(data, default=str).encode(),
            ContentType="application/json", ACL="public-read",
        )
        uri = f"s3://{config.S3_BUCKET}/{key}"
        log.info("uploaded artifact {}", uri)
        return uri
    except Exception as exc:  # noqa: BLE001 — never wedge validation on an upload
        log.warning("S3 put({}) failed: {}", key, exc)
        return None


def _get_json(key: str) -> dict:
    """Load a JSON dict from the bucket, or {} if absent."""
    try:
        obj = _client().get_object(Bucket=config.S3_BUCKET, Key=key)
        data = json.loads(obj["Body"].read())
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 — missing file -> start empty
        log.debug(f"S3 get_json({key}) failed, starting empty: {exc}")
        return {}


def update_fingerprint_corpus(model_uri: str, fingerprint: dict) -> tuple[str | None, str | None]:
    """Add/replace this model's entry in the two aggregate corpus files (read-modify-write).

    Returns (fingerprint_file_uri, tensors_file_uri). No-op (None, None) when disabled.
    """
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
    """Upload a per-model miner-fault report (the full explanation of why it was rejected)."""
    key = f"hippius_validation/{hotkey}/{_safe_digest(digest)}/fault.json"
    return _put(key, detail)

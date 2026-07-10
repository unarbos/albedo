"""Emit validation results as JSONL — locally, and optionally to Hippius S3 + OpenSearch.

The local JSONL is always written. S3 and OpenSearch are best-effort and only act when
their env vars are configured, mirroring the rest of the package.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from config_validation.storage import s3
from config_validation.result import ValidationResult

log = logging.getLogger(__name__)


def to_ndjson(results: Iterable[ValidationResult]) -> bytes:
    """Serialise results to newline-delimited JSON bytes."""
    lines = [json.dumps(r.to_jsonl_record(), separators=(",", ":")) for r in results]
    return ("\n".join(lines) + "\n").encode() if lines else b""


def write_jsonl(path: str, results: Iterable[ValidationResult]) -> int:
    """Write results to a local JSONL file; return the byte count written."""
    body = to_ndjson(results)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    log.info("wrote %d bytes to %s", len(body), path)
    return len(body)


def publish_s3(key: str, results: Iterable[ValidationResult]) -> bool:
    """Publish results as public-read ndjson to Hippius S3 (no-op if unconfigured)."""
    return s3.put_jsonl(key, to_ndjson(results))

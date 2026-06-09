"""Persistent per-model injection probe audit logs."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = Path(os.environ.get("ALBEDO_EVAL_LOG_DIR", "./logs"))
_INJECTION_DIRNAME = "injection"
_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value: str) -> str:
    safe = _SAFE_CHARS_RE.sub("_", (value or "").strip())
    safe = safe.strip("._-")
    return safe or "unknown_model"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _model_dir(challenger_ref: str) -> Path:
    return _DEFAULT_LOG_DIR / _INJECTION_DIRNAME / _safe_name(challenger_ref)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _artifact_payload(
    *,
    eval_id: str,
    challenger_ref: str,
    challenger_hotkey: str,
    probe_result: Any,
) -> dict[str, Any]:
    return {
        "eval_id": eval_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "challenger_ref": challenger_ref,
        "challenger_hotkey": challenger_hotkey,
        "is_clean": probe_result.is_clean,
        "n_probes": probe_result.n_probes,
        "n_injections": probe_result.n_injections,
        "n_untested": probe_result.n_untested,
        "triggered_judges": probe_result.triggered_judges,
        "probe_details": probe_result.probe_details,
    }


def _write_probe_artifacts(
    *,
    eval_id: str,
    challenger_ref: str,
    challenger_hotkey: str,
    probe_result: Any,
) -> Path:
    model_dir = _model_dir(challenger_ref)
    history_dir = model_dir / "history"
    payload = _artifact_payload(
        eval_id=eval_id,
        challenger_ref=challenger_ref,
        challenger_hotkey=challenger_hotkey,
        probe_result=probe_result,
    )
    latest_path = model_dir / "latest.json"
    history_path = history_dir / f"{_timestamp()}__{_safe_name(eval_id)}.json"
    _write_json(latest_path, payload)
    _write_json(history_path, payload)
    return history_path


async def write_probe_artifacts(
    *,
    eval_id: str,
    challenger_ref: str,
    challenger_hotkey: str,
    probe_result: Any,
) -> Path | None:
    """Persist the injection probe summary for this challenger model."""
    try:
        path = await asyncio.to_thread(
            _write_probe_artifacts,
            eval_id=eval_id,
            challenger_ref=challenger_ref,
            challenger_hotkey=challenger_hotkey,
            probe_result=probe_result,
        )
        log.info("saved injection probe log for %s at %s", challenger_ref, path)
        return path
    except Exception:
        log.warning("failed to write injection probe log for %s", challenger_ref, exc_info=True)
        return None

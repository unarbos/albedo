from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import contextmanager

from config_validation.models import ModelRef
from config_validation.storage import _supervise
from config_validation.storage._paths import _cache_dir

log = logging.getLogger(__name__)

_HUB_TOKEN_ENV = "HIPPIUS_HUB_TOKEN"
_CONFIG_ONLY_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
]
_MODEL_ONLY_PATTERNS = ["*.safetensors", "model.safetensors.index.json"]
_HEARTBEAT_INTERVAL_S = 10.0


@contextmanager
def _download_heartbeat(label: str):
    stop = threading.Event()
    start = time.monotonic()

    def _beat() -> None:
        while not stop.wait(_HEARTBEAT_INTERVAL_S):
            log.info(
                "hippius: still downloading %s (%.0fs elapsed)",
                label,
                time.monotonic() - start,
            )

    thread = threading.Thread(target=_beat, name="hippius-dl-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _hub():
    try:
        import hippius_hub
    except ModuleNotFoundError as exc:
        raise RuntimeError("hippius_hub is not installed; run: pip install hippius-hub") from exc
    return hippius_hub


def _token() -> str | None:
    return os.environ.get(_HUB_TOKEN_ENV)


def _download_child() -> None:
    repo, revision, local_dir, max_workers = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    _hub().snapshot_download(
        repo,
        revision=revision,
        local_dir=local_dir,
        max_workers=max(1, int(max_workers)),
        allow_patterns=_MODEL_ONLY_PATTERNS,
        token=_token(),
    )


def _download(ref: ModelRef, *, config_only: bool, max_workers: int) -> str:
    dest = _cache_dir(ref)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("hippius: downloading %s (config_only=%s) → %s", ref.immutable_ref, config_only, dest)
    if config_only or not _supervise.OUT_OF_PROCESS:
        with _download_heartbeat(ref.immutable_ref):
            _hub().snapshot_download(
                ref.repo,
                revision=ref.digest,
                local_dir=str(dest),
                max_workers=max_workers,
                allow_patterns=_CONFIG_ONLY_PATTERNS if config_only else _MODEL_ONLY_PATTERNS,
                token=_token(),
            )
        return str(dest)
    _supervise.supervise_download(
        child_call="from config_validation.storage._hippius import _download_child; _download_child()",  # noqa: E501
        args=[ref.repo, ref.digest, str(dest), str(max_workers)],
        watch_dir=dest,
        label=ref.immutable_ref,
        stall_seconds=_supervise.HIPPIUS_STALL_SECONDS,
        max_attempts=_supervise.HIPPIUS_STALL_RETRIES,
    )
    return str(dest)


def download_config(ref: ModelRef) -> str:
    return _download(ref, config_only=True, max_workers=8)


def download_full(ref: ModelRef) -> str:
    return _download(ref, config_only=False, max_workers=8)


def list_files(ref: ModelRef) -> list[str]:
    return _hub().list_repo_files(ref.repo, revision=ref.digest, token=_token())


def revision_resolves(ref: ModelRef) -> tuple[bool, str]:
    try:
        files = list_files(ref)
    except Exception as exc:
        log.error(f"revision {ref.digest} did not resolve on Hippius repo={ref.repo}: {exc}")
        return False, f"revision {ref.digest} did not resolve on Hippius: {exc}"
    if not files:
        return False, f"revision {ref.digest} resolved but the repo is empty"
    return True, f"revision resolved ({len(files)} files)"

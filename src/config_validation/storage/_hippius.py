"""Hippius hub backend — fetch + inspect model repos pinned to an OCI commit digest."""
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
_CONFIG_ONLY_PATTERNS = ["*.json", "chat_template.jinja"]
_HEARTBEAT_INTERVAL_S = 10.0


@contextmanager
def _download_heartbeat(label: str):
    """Log every ``_HEARTBEAT_INTERVAL_S`` seconds that ``label`` is still downloading.

    snapshot_download() blocks with no progress output, so a daemon thread emits a
    periodic heartbeat until the download returns.
    """
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
        import hippius_hub  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise RuntimeError("hippius_hub is not installed; run: pip install hippius-hub") from exc
    return hippius_hub


def _token() -> str | None:
    return os.environ.get(_HUB_TOKEN_ENV)


def _download_child() -> None:
    """Child-process entry point for a supervised full Hippius download (see _supervise).

    Invoked as ``python -c "...; _download_child()" <repo> <revision> <local_dir> <max_workers>``.
    """
    repo, revision, local_dir, max_workers = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    _hub().snapshot_download(
        repo,
        revision=revision,
        local_dir=local_dir,
        max_workers=max(1, int(max_workers)),
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
                allow_patterns=_CONFIG_ONLY_PATTERNS if config_only else None,
                token=_token(),
            )
        return str(dest)
    _supervise.supervise_download(
        child_call="from config_validation.storage._hippius import _download_child; _download_child()",
        args=[ref.repo, ref.digest, str(dest), str(max_workers)],
        watch_dir=dest,
        label=ref.immutable_ref,
        stall_seconds=_supervise.HIPPIUS_STALL_SECONDS,
        max_attempts=_supervise.HIPPIUS_STALL_RETRIES,
    )
    return str(dest)


def download_config(ref: ModelRef) -> str:
    """Download only the JSON config files for ``ref``; return the local dir path."""
    return _download(ref, config_only=True, max_workers=8)


def download_full(ref: ModelRef) -> str:
    """Download the full model snapshot for ``ref``; return the local dir path."""
    return _download(ref, config_only=False, max_workers=8)


def list_files(ref: ModelRef) -> list[str]:
    """List filenames present in the Hippius repo at the pinned digest."""
    return _hub().list_repo_files(ref.repo, revision=ref.digest, token=_token())


def revision_resolves(ref: ModelRef) -> tuple[bool, str]:
    """Confirm the committed digest is a real revision on Hippius (revision parity).

    The committed ``sha256:`` digest IS the Hippius manifest revision, so a
    successful file listing at that revision proves the on-chain commit points at a
    real snapshot. Returns (ok, detail).
    """
    try:
        files = list_files(ref)
    except Exception as exc:  # noqa: BLE001 — surface any hub error as a check failure
        log.error(f"revision {ref.digest} did not resolve on Hippius repo={ref.repo}: {exc}")
        return False, f"revision {ref.digest} did not resolve on Hippius: {exc}"
    if not files:
        return False, f"revision {ref.digest} resolved but the repo is empty"
    return True, f"revision resolved ({len(files)} files)"

"""Fetch + inspect model repos on the Hippius hub, pinned to a commit digest."""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from config_validation.config import MODEL_CACHE_DIR
from config_validation.models import ModelRef

log = logging.getLogger(__name__)

_HUB_TOKEN_ENV = "HIPPIUS_HUB_TOKEN"
_CONFIG_ONLY_PATTERNS = ["*.json"]
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
            log.info("hippius: still downloading %s (%.0fs elapsed)", label, time.monotonic() - start)

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


def _cache_dir(ref: ModelRef) -> Path:
    """Per-(repo, digest) cache dir, guarded against path traversal via crafted repo names."""
    safe_digest = ref.digest.replace(":", "_")
    root = Path(MODEL_CACHE_DIR).resolve()
    resolved = (root / ref.repo / safe_digest).resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise ValueError(f"ModelRef.repo {ref.repo!r} resolves outside cache root — blocked")
    return resolved


def cache_dir(ref: ModelRef) -> Path:
    """Public: per-(repo, digest) local cache dir for ``ref`` (no I/O, no download)."""
    return _cache_dir(ref)


def _download(ref: ModelRef, *, config_only: bool, max_workers: int) -> str:
    dest = _cache_dir(ref)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "config.json").exists() and config_only:
        log.debug("hippius: config cache hit at %s", dest)
        return str(dest)
    log.info("hippius: downloading %s (config_only=%s) → %s", ref.immutable_ref, config_only, dest)
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
        return False, f"revision {ref.digest} did not resolve on Hippius: {exc}"
    if not files:
        return False, f"revision {ref.digest} resolved but the repo is empty"
    return True, f"revision resolved ({len(files)} files)"

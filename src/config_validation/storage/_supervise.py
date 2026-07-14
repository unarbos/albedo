"""Supervised, killable model downloads.

``snapshot_download()`` runs in-process and blocks on the network with no way to
interrupt it — a wedged transfer (dead CDN socket, hung xet worker) hangs the
calling thread forever, and neither ``asyncio.wait_for`` nor a thread pool can
reclaim it. This module runs the download in a child process guarded by a stall
watchdog: the child is killed and the download resumed when its target directory
stops growing, and abandoned (raising, so the caller can retry) after a few
consecutive stalls. Only full-model downloads use this; config-only fetches are
small and stay in-process.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_S = 10.0

# Tunable per host via env, read at import so a redeploy picks up changes.
OUT_OF_PROCESS = os.environ.get("ALBEDO_DOWNLOAD_OUT_OF_PROCESS", "1") not in ("0", "false", "False", "")
# HF's CDN streams steadily, so several minutes of zero progress means the transfer is dead.
STALL_SECONDS = float(os.environ.get("ALBEDO_DOWNLOAD_STALL_SECONDS", "600"))
STALL_RETRIES = int(os.environ.get("ALBEDO_DOWNLOAD_STALL_RETRIES", "3"))
# Hippius pulls from decentralized storage — slower, with longer *legitimate* gaps between
# chunks — so it tolerates a wider no-progress window before a kill.
# Worst case (HIPPIUS_STALL_SECONDS * HIPPIUS_STALL_RETRIES = 2400s) is kept under the sanity
# worker's download_timeout_s so this watchdog, not the blunt outer timeout, is what fires
# — bump that outer timeout too if you widen these.
HIPPIUS_STALL_SECONDS = float(os.environ.get("ALBEDO_HIPPIUS_DOWNLOAD_STALL_SECONDS", "1200"))
HIPPIUS_STALL_RETRIES = int(os.environ.get("ALBEDO_HIPPIUS_DOWNLOAD_STALL_RETRIES", "2"))


def _dir_bytes(path: Path) -> int:
    """Bytes *actually written* under ``path`` (allocated blocks, not apparent size).

    hippius_hub preallocates each file to its full size up front, so ``st_size`` jumps to
    the final total and sits flat while data is still streaming in — which the watchdog
    would misread as a stall. ``st_blocks`` reflects blocks actually allocated, so it tracks
    real download progress for both preallocated (hippius) and append-style (HF) writes.
    """
    total = 0
    for item in path.rglob("*"):
        try:
            st = item.stat()
        except OSError:
            continue
        if item.is_file():
            total += st.st_blocks * 512
    return total


def _tail_file(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return "(download log unavailable)"
    text = data.decode("utf-8", errors="replace").strip()
    return text or "(no output captured)"


def _spawn(child_call: str, args: list[str], log_path: Path) -> subprocess.Popen:
    handle = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            [sys.executable, "-c", child_call, *args],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        # Popen holds its own dup of the fd; the parent's copy is no longer needed.
        handle.close()


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def supervise_download(
    *,
    child_call: str,
    args: list[str],
    watch_dir: Path,
    label: str,
    stall_seconds: float | None = None,
    max_attempts: int | None = None,
) -> None:
    """Run ``child_call`` in a child process, killing + resuming it if it stalls.

    ``child_call`` is Python passed to ``python -c`` that re-imports the backend and
    runs its download; ``args`` become the child's ``sys.argv[1:]``. A watchdog samples
    ``watch_dir``'s byte total every ``_HEARTBEAT_INTERVAL_S``; if it stops growing for
    ``stall_seconds`` the child (its own process group) is terminated and the download
    retried, resuming from what already landed on disk. Raises ``TimeoutError`` after
    ``max_attempts`` consecutive stalls, or ``RuntimeError`` on a genuine child error —
    both are retryable infra faults one level up. ``stall_seconds`` / ``max_attempts``
    default to the HF-tuned module globals; the Hippius backend passes its own wider ones.
    """
    stall = STALL_SECONDS if stall_seconds is None else stall_seconds
    log_path = watch_dir.parent / f"{watch_dir.name}.download.log"
    attempts = max(1, STALL_RETRIES if max_attempts is None else max_attempts)
    for attempt in range(1, attempts + 1):
        proc = _spawn(child_call, args, log_path)
        start = time.monotonic()
        last_bytes = -1
        last_progress = start
        stalled = False
        while proc.poll() is None:
            time.sleep(_HEARTBEAT_INTERVAL_S)
            current = _dir_bytes(watch_dir)
            now = time.monotonic()
            log.info(
                "download %s attempt=%d/%d elapsed=%.0fs bytes=%d",
                label, attempt, attempts, now - start, current,
            )
            if current > last_bytes:
                last_bytes = current
                last_progress = now
            elif now - last_progress >= stall:
                log.warning(
                    "download %s stalled attempt=%d/%d bytes=%d no_progress=%.0fs — killing",
                    label, attempt, attempts, current, now - last_progress,
                )
                _terminate(proc)
                stalled = True
                break
        if stalled:
            continue
        if proc.returncode == 0:
            log_path.unlink(missing_ok=True)
            return
        detail = _tail_file(log_path, 4000)
        raise RuntimeError(f"download of {label} exited {proc.returncode}: {detail}")
    raise TimeoutError(
        f"download of {label} made no progress for {stall:.0f}s across {attempts} attempts"
    )

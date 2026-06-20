#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg

GENESIS_MARKERS = ("teutonic__qwen3.6-35b-a3b-genesis", "qwen3.6-35b-a3b-genesis")
IN_FLIGHT = {
    "SUBMITTED", "HIPPIUS_RUNNING", "HIPPIUS_RETRYABLE", "HIPPIUS_VALIDATED",
    "PRE_EVAL_QUEUED", "PRE_EVAL_RUNNING", "PRE_EVAL_RETRYABLE", "PRE_EVAL_PASSED",
    "EVAL_QUEUED", "EVAL_RUNNING", "EVAL_RETRYABLE", "EVAL_WIN",
    "SET_REIGN_RUNNING", "SET_REIGN_RETRYABLE", "REIGN_SET",
    "WEIGHT_SET_RUNNING", "WEIGHT_SET_RETRYABLE",
}
FAILED = {"TERMINAL_INVALID", "TERMINAL_INFRA_FAILED"}


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)


def find_env() -> Path | None:
    here = Path(__file__).resolve().parent
    for d in (here, *list(here.parents)[:3], Path.cwd()):
        if (d / ".env").is_file():
            return d / ".env"
    return None


def load_env() -> None:
    p = find_env()
    if not p:
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def connect():
    # All DB connection settings come from env / .env — no hardcoded host, db, or credentials.
    required = ("ALBEDO_POSTGRES_HOST", "ALBEDO_POSTGRES_DB",
                "ALBEDO_POSTGRES_USER", "ALBEDO_POSTGRES_PASSWORD")
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise SystemExit("missing required DB env vars (set in env / .env): " + ", ".join(missing))
    return psycopg.connect(
        host=os.environ["ALBEDO_POSTGRES_HOST"],
        port=int(os.environ.get("ALBEDO_POSTGRES_HOST_PORT", "65432")),
        dbname=os.environ["ALBEDO_POSTGRES_DB"],
        user=os.environ["ALBEDO_POSTGRES_USER"],
        password=os.environ["ALBEDO_POSTGRES_PASSWORD"],
        connect_timeout=15,
    )


def digest_of(model_uri: str | None) -> str | None:
    if model_uri and "@sha256:" in model_uri:
        return model_uri.split("@sha256:")[-1].strip()
    return None


def load_db_state(cur):
    """Return (king_digests, subs) where subs maps digest -> list[(state, finished_at)]."""
    cur.execute(
        """SELECT ms.model_uri FROM reigns r
           JOIN reign_members rm ON rm.reign_id = r.id
           JOIN model_submissions ms ON ms.id = rm.submission_id
           WHERE r.state = 'ACTIVE' AND ms.model_uri IS NOT NULL""")
    king = {d for d in (digest_of(r[0]) for r in cur.fetchall()) if d}

    cur.execute("SELECT model_uri, state, finished_at FROM model_submissions WHERE model_uri IS NOT NULL")
    subs: dict[str, list[tuple[str, object]]] = {}
    for uri, state, finished in cur.fetchall():
        d = digest_of(uri)
        if d:
            subs.setdefault(d, []).append((state, finished))
    return king, subs


def newest_mtime(path: Path) -> datetime:
    newest = path.stat().st_mtime
    for f in path.rglob("*"):
        try:
            newest = max(newest, f.stat().st_mtime)
        except OSError:
            pass
    return datetime.fromtimestamp(newest, timezone.utc)


def dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def scan(cache_dir: Path):
    """Yield (model_dir, repo_munged, digest, is_partial)."""
    base = cache_dir / "oci"
    if not base.is_dir():
        return
    for registry in base.iterdir():
        if not registry.is_dir():
            continue
        for repo in registry.iterdir():
            if not repo.is_dir():
                continue
            for model in repo.iterdir():
                if not model.is_dir():
                    continue
                is_partial = model.name.endswith(".partial")
                digest = model.name[:-len(".partial")] if is_partial else model.name
                yield model, repo.name, digest, is_partial


def decide(repo_munged, digest, is_partial, model_dir, king, subs, grace_hours, now):
    if is_partial:
        age_h = (now - newest_mtime(model_dir)).total_seconds() / 3600
        if age_h >= grace_hours:
            return "DELETE", f"abandoned .partial download ({age_h:.1f}h idle)"
        return "KEEP", f".partial download active ({age_h:.1f}h idle)"
    if any(m in repo_munged for m in GENESIS_MARKERS):
        return "KEEP", "canonical seed"
    if digest in king:
        return "KEEP", "current king (active reign)"
    recs = subs.get(digest)
    if not recs:
        return "KEEP", "no DB record (unknown)"
    states = {s for s, _ in recs}
    if states & IN_FLIGHT:
        return "KEEP", "in-flight: " + ",".join(sorted(states & IN_FLIGHT))
    # failed-only models get a grace window; evaluated (loss/coronated) are decided immediately
    if not (states & {"COMPLETE_LOSS", "COMPLETE_CORONATED"}):
        fail_times = [f for s, f in recs if s in FAILED and f]
        if fail_times:
            age_h = (now - max(fail_times)).total_seconds() / 3600
            if age_h < grace_hours:
                return "KEEP", f"failed, within {grace_hours}h grace ({age_h:.1f}h)"
    if "COMPLETE_LOSS" in states:
        return "DELETE", "evaluated & lost"
    if "COMPLETE_CORONATED" in states:
        return "DELETE", "former king, shifted out (older king)"
    if states & FAILED:
        return "DELETE", f"failed, {grace_hours}h grace expired"
    return "KEEP", "unhandled state: " + ",".join(sorted(states))


def run_once(cache_dir: Path, grace_hours: float, execute: bool) -> None:
    with connect() as conn, conn.cursor() as cur:
        king, subs = load_db_state(cur)
    now = datetime.now(timezone.utc)
    log(f"king models (active reign): {len(king)} | known digests: {len(subs)} | "
        f"cache={cache_dir} grace={grace_hours}h execute={execute}")
    kept = deleted = 0
    freed = 0
    for model_dir, repo, digest, is_partial in scan(cache_dir):
        action, reason = decide(repo, digest, is_partial, model_dir, king, subs, grace_hours, now)
        label = f"{repo}/{model_dir.name}"
        if action == "DELETE":
            size = dir_size(model_dir)
            log(f"DELETE {label}  ({size / 1e9:.1f} GB) — {reason}")
            if execute:
                shutil.rmtree(model_dir, ignore_errors=True)
            deleted += 1
            freed += size
        else:
            log(f"keep   {label} — {reason}")
            kept += 1
    log(f"=== {'EXECUTED' if execute else 'DRY-RUN'}: kept {kept}, "
        f"{'deleted' if execute else 'would delete'} {deleted} ({freed / 1e9:.1f} GB) ===")


def acquire_lock():
    """Single-instance PID lock: refuse to start if another cleanup is already running."""
    lock_path = os.environ.get("CLEANUP_LOCK_PATH", "/tmp/albedo-eval-cache-cleanup.lock")
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            f"another eval_cache_cleanup instance is already running (lock held: {lock_path})")
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def main() -> None:
    load_env()  # populate os.environ from .env before reading env-backed defaults below
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--once", action="store_true", help="single pass then exit (default: loop)")
    ap.add_argument("--interval", type=float,
                    default=float(os.environ.get("CLEANUP_POLL_SECONDS", "60")),
                    help="seconds between passes in service mode")
    ap.add_argument("--grace-hours", type=float,
                    default=float(os.environ.get("CLEANUP_FAIL_GRACE_HOURS", "2")),
                    help="keep failed models this long before deleting")
    ap.add_argument("--cache-dir",
                    default=(os.environ.get("EVAL_CLEAN_WATCH_DIR")
                             or os.environ.get("ALBEDO_CACHE_DIR", "/root/albedo-models")),
                    help="dir to watch for cached models (env: EVAL_CLEAN_WATCH_DIR)")
    args = ap.parse_args()
    lock = acquire_lock()  # noqa: F841 — held open for the process lifetime to hold the PID lock
    execute = args.execute or os.environ.get("CLEANUP_EXECUTE") == "1"
    cache_dir = Path(args.cache_dir)

    if args.once:
        run_once(cache_dir, args.grace_hours, execute)
        return
    log(f"eval_cache_cleanup service starting (interval={args.interval}s)")
    while True:
        try:
            run_once(cache_dir, args.grace_hours, execute)
        except Exception as exc:  # noqa: BLE001 — keep the service alive across transient errors
            log(f"cleanup pass error: {type(exc).__name__}: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

from config import Config
from db import query_succeeded_runs
from download import download_run
from eval_host import fetch_run, is_complete, list_remote_runs
from extract import extract_rows
from hf_upload import upload
from migrate import migrate_if_needed
from state import State
from store import append_rows, cut_chunks


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(value: str | None) -> datetime:
    return datetime.fromisoformat(value) if value else _now()


def ingest_run(
    cfg: Config,
    state: State,
    run_id: str,
    run_dir: Path,
    source: str,
    started_at: str | None = None,
) -> None:
    by_model = extract_rows(run_dir)
    for model, rows in by_model.items():
        append_rows(cfg, model, rows)
    n_rows = sum(len(r) for r in by_model.values())
    state.mark_processed(run_id, n_rows, source, started_at=started_at)
    note = "" if n_rows else "  (no reference fields — old-format jsonl, skipped)"
    print(f"  ingested {run_id} from {source} ({n_rows} rows){note}")


def scan_eval_machine(cfg: Config, state: State) -> None:
    remote = list_remote_runs(cfg)
    new = {r: v for r, v in remote.items() if not state.is_processed(r) and is_complete(v["files"])}
    incomplete = sum(
        1 for r, v in remote.items() if not state.is_processed(r) and not is_complete(v["files"])
    )
    print(
        f"eval machine: {len(remote)} run dirs, {len(new)} new complete"
        + (f", {incomplete} in progress" if incomplete else "")
    )
    for run_id, info in sorted(new.items(), key=lambda kv: kv[1]["mtime"]):
        run_dir = fetch_run(cfg, run_id, info["files"])
        ingest_run(cfg, state, run_id, run_dir, "eval-machine")
    if new:
        state.set_meta("last_new_dir_at", _now().isoformat(timespec="seconds"))


def db_health_check(cfg: Config, state: State) -> None:
    print(f"no new run dir for {cfg.stall_hours}h — checking DB")
    runs = query_succeeded_runs(cfg, state.get_meta("anchor"))
    missing = {r: v for r, v in runs.items() if not state.is_processed(r)}
    if not missing:
        print("  DB confirms no unprocessed evals — queue idle, all good")
    else:
        print(f"  DB has {len(missing)} succeeded runs missing locally — backfilling from S3")
        for run_id, info in sorted(missing.items(), key=lambda kv: kv[1]["started_at"]):
            download_run(cfg, run_id, info["files"])
            ingest_run(
                cfg,
                state,
                run_id,
                cfg.run_dir(run_id),
                "s3-backfill",
                started_at=info["started_at"],
            )
        state.set_meta("last_new_dir_at", _now().isoformat(timespec="seconds"))
    state.set_meta("last_db_check_at", _now().isoformat(timespec="seconds"))


def tick(cfg: Config, state: State, no_upload: bool) -> int:
    try:
        scan_eval_machine(cfg, state)
    except Exception as e:
        print(f"eval-machine scan failed ({e}) — will rely on DB fallback")

    stall = timedelta(hours=cfg.stall_hours)
    if (
        _now() - _ts(state.get_meta("last_new_dir_at")) > stall
        and _now() - _ts(state.get_meta("last_db_check_at")) > stall
    ):
        try:
            db_health_check(cfg, state)
        except Exception as e:
            print(f"DB health check failed ({e})")

    print("chunks:")
    cut_chunks(cfg, state)
    pending = len(state.pending_uploads())
    if no_upload:
        print(f"upload skipped (--no-upload); {pending} chunks pending")
        return 0
    repo = state.get_meta("hf_repo")
    if not repo:
        print(f"no HF repo configured (DATASET_CREATOR_HF_REPO); {pending} chunks pending")
        return 0
    try:
        upload(cfg, state, repo)
    except Exception as e:
        print(f"upload failed ({e}); {len(state.pending_uploads())} chunks remain pending")
        return 1
    return 0


def run_pipeline(
    cfg: Config, repo: str | None, since: str | None, no_upload: bool, watch: bool
) -> int:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    state = State(cfg.db_path)
    migrate_if_needed(cfg, state)
    repo = repo or cfg.hf_repo
    if repo:
        state.set_meta("hf_repo", repo)
    if not state.get_meta("anchor"):
        state.set_meta(
            "anchor", since or (_now() - timedelta(hours=24)).isoformat(timespec="seconds")
        )
    print(
        f"anchor={state.get_meta('anchor')}  processed_runs={state.processed_count()}"
        f"  repo={state.get_meta('hf_repo')}"
    )

    if not watch:
        return tick(cfg, state, no_upload)
    while True:
        code = tick(cfg, state, no_upload)
        if code:
            print(f"tick finished with code {code}")
        print(f"sleeping {cfg.poll_seconds}s")
        time.sleep(cfg.poll_seconds)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="artifacts/state/output directory (default: env DATASET_CREATOR_DATA_DIR)",
    )
    ap.add_argument("--repo", help="HF dataset repo id (default: env DATASET_CREATOR_HF_REPO)")
    ap.add_argument("--since", help="ISO date anchor for the first run (default: now-24h)")
    ap.add_argument("--no-upload", action="store_true", help="build chunks, skip HF upload")
    ap.add_argument(
        "--watch",
        action="store_true",
        help="keep polling the eval machine every DATASET_CREATOR_POLL_SECONDS",
    )
    args = ap.parse_args()
    cfg = Config(data_dir=args.data_dir) if args.data_dir else Config()
    return run_pipeline(cfg, args.repo, args.since, args.no_upload, args.watch)


if __name__ == "__main__":
    sys.exit(main())

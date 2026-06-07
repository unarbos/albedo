#!/usr/bin/env python3
"""Remove failed eval history entries and unburn miner hotkeys/repos.

Usage:
    cd albedo && source .venv/bin/activate
    doppler run -p arbos -c dev -- python scripts/repair_challenges.py eval-0014 eval-0015
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validator import ObjectStore, State  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Unburn miners after infra-failed evals")
    parser.add_argument("challenge_ids", nargs="*", help="e.g. eval-0014 eval-0015")
    parser.add_argument(
        "--purge-infra",
        action="store_true",
        help="remove all eval_infra failure entries from dashboard history (no unburn)",
    )
    parser.add_argument(
        "--replay-infra",
        action="store_true",
        help="purge eval_infra history entries and unburn affected hotkeys/repos for re-eval",
    )
    parser.add_argument(
        "--replay-errors",
        action="store_true",
        help="purge eval_infra + eval_error + no_verdict entries and unburn for re-eval",
    )
    args = parser.parse_args()
    targets = set(args.challenge_ids)

    store = ObjectStore()
    state = State(store)
    state.load()

    replay = args.replay_infra or args.replay_errors
    if args.purge_infra or replay:
        if args.replay_errors:
            codes = {"eval_infra", "eval_error", "no_verdict"}
        else:
            codes = {"eval_infra"}
        removed = [e for e in state.history if e.get("error_code") in codes]
        if not removed:
            print(f"no {codes} entries in history")
            return 0
        if replay:
            resolved_hk: set[str] = set()
            resolved_repo: set[str] = set()
            for entry in state.history:
                if entry.get("error_code") in codes:
                    continue
                hk = entry.get("hotkey", "")
                repo = entry.get("model_repo", "")
                digest = entry.get("model_digest", "")
                model_key = f"{repo}@{digest}" if digest else repo
                if hk:
                    resolved_hk.add(hk)
                if model_key:
                    resolved_repo.add(model_key)
            seen_hk: set[str] = set()
            seen_repo: set[str] = set()
            to_enqueue: list[dict] = []
            for entry in removed:
                hotkey = entry.get("hotkey", "")
                repo = entry.get("model_repo", "")
                digest = entry.get("model_digest", "")
                model_key = f"{repo}@{digest}" if digest else repo
                if hotkey and hotkey not in seen_hk and hotkey not in resolved_hk:
                    state.seen.discard(hotkey)
                    seen_hk.add(hotkey)
                    print(f"unburned hotkey {hotkey[:16]}…")
                elif hotkey in resolved_hk:
                    print(f"skip hotkey {hotkey[:16]}… (already has real verdict)")
                if model_key and model_key not in seen_repo and model_key not in resolved_repo:
                    state.completed_repos.discard(model_key)
                    seen_repo.add(model_key)
                    print(f"unburned repo {model_key[:56]}")
                elif model_key in resolved_repo:
                    print(f"skip repo {model_key[:56]} (already has real verdict)")
                if (
                    hotkey
                    and hotkey not in resolved_hk
                    and repo
                    and digest
                    and not any(
                        e.get("hotkey") == hotkey
                        for e in to_enqueue
                    )
                ):
                    to_enqueue.append({
                        "hotkey": hotkey,
                        "block": entry.get("block", 0),
                        "model_repo": repo,
                        "model_digest": digest,
                    })
            if to_enqueue and args.replay_errors:
                queued = set(e.get("hotkey") for e in state.queue)
                for rev in to_enqueue:
                    if rev["hotkey"] in queued:
                        continue
                    cid = state.enqueue(rev)
                    if cid:
                        queued.add(rev["hotkey"])
                        print(f"re-enqueued {cid} from {rev['hotkey'][:16]}…")
        state.history = [e for e in state.history if e.get("error_code") not in codes]
        state.stats["failed"] = max(0, state.stats.get("failed", 0) - len(removed))
        state.flush()
        state.flush_dashboard(force=True)
        action = "replayed" if replay else "purged"
        print(f"{action} {len(removed)} error entries; stats={state.stats}")
        return 0

    if not targets:
        parser.error("provide challenge_ids or --purge-infra / --replay-infra / --replay-errors")

    removed: list[dict] = []
    kept: list[dict] = []
    for entry in state.history:
        cid = entry.get("challenge_id")
        if cid in targets:
            removed.append(entry)
        else:
            kept.append(entry)

    if not removed:
        print(f"no history entries matched {sorted(targets)}")
        return 1

    for entry in removed:
        hotkey = entry.get("hotkey", "")
        repo = entry.get("model_repo", "")
        digest = entry.get("model_digest", "")
        model_key = f"{repo}@{digest}" if digest else repo
        if hotkey:
            state.seen.discard(hotkey)
            print(f"unburned hotkey {hotkey[:16]}…")
        if model_key:
            state.completed_repos.discard(model_key)
            print(f"unburned repo {model_key[:56]}")
        if entry.get("accepted"):
            state.stats["accepted"] = max(0, state.stats.get("accepted", 0) - 1)
        elif entry.get("error_code"):
            state.stats["failed"] = max(0, state.stats.get("failed", 0) - 1)
        else:
            state.stats["rejected"] = max(0, state.stats.get("rejected", 0) - 1)

    state.history = kept
    state.flush()
    state.flush_dashboard(force=True)
    print(f"removed {len(removed)} history entries; stats={state.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

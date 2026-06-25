#!/usr/bin/env python3
"""Verify (and optionally repair) the published Albedo king Hugging Face repos.

For every crowned Qwen3.6-35B king this checks the HF repo **first** (cheap, no download):
it confirms ``config.json``, a complete set of safetensors shards (per the repo's own
``model.safetensors.index.json``), and ``albedo.md`` are present. With ``--fix`` it then
downloads the model bytes for any repo that is missing files and uploads just the gaps into
the correct repo (a missing repo gets a full mirror).

Order of operations, by design: check HF -> download only what's missing -> upload to repo.

Examples
--------
    # report only (exit code 2 if any repo is incomplete)
    python scripts/verify_king_repos.py --hf-namespace dendriteholdings

    # repair just King IX
    python scripts/verify_king_repos.py --hf-namespace dendriteholdings --only IX --fix

    # repair everything that is missing files
    python scripts/verify_king_repos.py --hf-namespace dendriteholdings --fix

Config (DSN, HF token, eval/work dirs, namespace, repo prefix) is read from the same
environment / ``.env`` as king_hf_uploader.py; CLI flags below override it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/verify_king_repos.py` to import its sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import king_hf_uploader as ku  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fix",
        action="store_true",
        help="download and commit any files missing from a repo (default: report only)",
    )
    parser.add_argument("--only", help="comma-separated roman numerals to check, e.g. IX,X")
    parser.add_argument("--limit", type=int, help="stop after repairing this many repos")
    parser.add_argument(
        "--hf-namespace", help="HF namespace (env ALBEDO_KING_HF_NAMESPACE; prod: dendriteholdings)"
    )
    parser.add_argument("--hf-token", help="HF token override")
    parser.add_argument("--database-url", help="Postgres DSN override")
    parser.add_argument("--eval-dir", help="eval cache dir to read model bytes from (never deleted)")
    parser.add_argument("--work-dir", help="delete-safe download dir used for repair")
    parser.add_argument("--repo-prefix", help="HF repo name prefix")
    parser.add_argument("--poll-interval-s", type=float, help=argparse.SUPPRESS)
    # Fields load_settings expects but this tool doesn't expose as flags.
    parser.set_defaults(force=False, verify=False, dry_run=False)
    return parser


def _fix_repo(api, king, settings, repo_id: str, problems: list[str]) -> bool:
    """Repair one repo. A missing repo gets a full mirror; otherwise commit the gaps."""
    if problems == ["repo does not exist"]:
        ku._upload_one(api, king, settings, repo_id)  # creates the repo + 2-commit mirror
        return True
    return ku._verify_and_repair(api, king, settings, repo_id)


def main() -> int:
    args = _build_parser().parse_args()
    settings = ku.load_settings(args)
    if not settings.hf_token:
        raise SystemExit("no HF token; set ALBEDO_KING_HF_TOKEN / HF_TOKEN or pass --hf-token")

    api = ku._hf_api(settings.hf_token)
    with ku._connect(settings) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        kings = ku.list_crowned_kings(conn, settings)

    if args.only:
        wanted = {r.strip().upper() for r in args.only.split(",") if r.strip()}
        kings = [k for k in kings if k.roman.upper() in wanted]

    print(f"namespace : {settings.hf_namespace}")
    print(f"repos     : {len(kings)} crowned king(s)   mode: {'FIX' if args.fix else 'verify-only'}")
    print("=" * 72)

    n_ok = n_bad = n_fixed = n_failed = 0
    repaired = 0
    for king in kings:
        repo_id = ku.repo_id_for(king, settings)
        problems = ku.hf_repo_problems(api, repo_id, settings.hf_token)
        if not problems:
            n_ok += 1
            print(f"OK    {repo_id}")
            continue

        n_bad += 1
        print(f"BAD   {repo_id}")
        for problem in problems[:12]:
            print(f"        - {problem}")
        if len(problems) > 12:
            print(f"        … (+{len(problems) - 12} more)")
        if not args.fix:
            continue

        try:
            _fix_repo(api, king, settings, repo_id, problems)
        except Exception as exc:  # noqa: BLE001 — one bad repo must not abort the sweep
            n_failed += 1
            print(f"        FIX FAILED: {type(exc).__name__}: {exc}")
            continue

        after = ku.hf_repo_problems(api, repo_id, settings.hf_token)
        if after:
            n_failed += 1
            print(f"        -> STILL INCOMPLETE: {', '.join(after[:8])}")
        else:
            n_fixed += 1
            print("        -> FIXED")
        repaired += 1
        if args.limit and repaired >= args.limit:
            print("reached --limit; stopping")
            break

    print("=" * 72)
    summary = f"summary: {n_ok} ok | {n_bad} incomplete"
    if args.fix:
        summary += f" | {n_fixed} fixed | {n_failed} still bad/failed"
    print(summary)

    # Non-zero exit so cron/CI notices unresolved problems.
    if args.fix:
        return 2 if n_failed else 0
    return 2 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

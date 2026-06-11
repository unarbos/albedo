"""`albedo` CLI — miner entrypoint.

Headless subcommands: upload, check-hippius, commit, check-commit, publish.
Interactive: `albedo on` launches the Rich Live TUI. Bare `albedo` prints help.
"""
from __future__ import annotations

import argparse
import os
import sys

from miner import env

env.load()  # populate os.environ from .env before reading any defaults below

_NETUID = int(os.environ.get("CHAIN_NETUID", "97"))
_NETWORK = os.environ.get("CHAIN_NETWORK", "finney")
_COLDKEY = os.environ.get("ALBEDO_COLDKEY")
_HOTKEY = os.environ.get("ALBEDO_HOTKEY")
_NAMESPACE = os.environ.get("ALBEDO_NAMESPACE")


def _wallet_arg(parser, name: str, default, help_: str) -> None:
    """Add a wallet arg that defaults from .env and is only required when unset there."""
    parser.add_argument(f"--{name}", default=default, required=default is None, help=help_)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="albedo", description="Albedo miner: upload → validate → commit.")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("on", help="launch the interactive TUI")

    up = sub.add_parser("upload", help="upload a local model dir to Hippius")
    up.add_argument("--path", required=True)
    up.add_argument("--namespace", default=_NAMESPACE)
    up.add_argument("--name", help="suffix appended after albedo-qwen3-4b-")
    up.add_argument("--repo", help="full repo override (ns/albedo-qwen3-4b-…)")

    ch = sub.add_parser("check-hippius", help="validate a model/repo (file manifest + architecture, no dedup)")
    ch.add_argument("--path", help="local model directory")
    ch.add_argument("--repo")
    ch.add_argument("--digest")

    cm = sub.add_parser("commit", help="commit the v5 reveal on-chain (preview + Y/N + registration check)")
    cm.add_argument("--repo", required=True)
    cm.add_argument("--digest", required=True)
    _wallet_arg(cm, "coldkey", _COLDKEY, "wallet (coldkey) name")
    _wallet_arg(cm, "hotkey", _HOTKEY, "hotkey name")
    cm.add_argument("--netuid", type=int, default=_NETUID)
    cm.add_argument("--network", default=_NETWORK)
    cm.add_argument("--yes", action="store_true", help="skip the Y/N prompt")

    rg = sub.add_parser("register", help="register a hotkey on the subnet (recycle/burned register)")
    _wallet_arg(rg, "coldkey", _COLDKEY, "wallet (coldkey) name")
    _wallet_arg(rg, "hotkey", _HOTKEY, "hotkey name")
    rg.add_argument("--netuid", type=int, default=_NETUID)
    rg.add_argument("--network", default=_NETWORK)
    rg.add_argument("--yes", action="store_true", help="skip the Y/N prompt")

    cc = sub.add_parser("check-commit", help="read on-chain commitments")
    cc.add_argument("--netuid", type=int, default=_NETUID)
    cc.add_argument("--network", default=_NETWORK)
    cc.add_argument("--hotkey", help="filter to one hotkey ss58")

    pub = sub.add_parser("publish", help="validate → upload → check-hippius → commit (end to end)")
    pub.add_argument("--path", required=True)
    pub.add_argument("--namespace", default=_NAMESPACE, required=_NAMESPACE is None)
    pub.add_argument("--name", required=True)
    _wallet_arg(pub, "coldkey", _COLDKEY, "wallet (coldkey) name")
    _wallet_arg(pub, "hotkey", _HOTKEY, "hotkey name")
    pub.add_argument("--netuid", type=int, default=_NETUID)
    pub.add_argument("--network", default=_NETWORK)
    pub.add_argument("--yes", action="store_true")
    pub.add_argument("--skip-commit", action="store_true", help="stop after upload + checks")
    return p


def _print_checks(ok: bool, res: dict) -> None:
    for name, v in res.items():
        mark = "PASS" if v["ok"] else "FAIL"
        line = f"[{mark}] {name}"
        if not v["ok"] and v["reason"]:
            line += f" — {v['reason']}"
        print(line)
    print("VALID" if ok else "INVALID")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 — show what's wrong, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args, parser) -> int:
    if args.cmd is None:
        parser.print_help()
        return 0

    if args.cmd == "on":
        from miner import tui
        tui.run()
        return 0

    if args.cmd == "upload":
        from miner import commit, upload
        repo = args.repo or upload.make_repo(args.namespace, args.name)
        ref = upload.upload_to_hippius(args.path, repo)
        print(ref.immutable_ref)
        print("reveal:", commit.build_reveal(ref))
        return 0

    if args.cmd == "check-hippius":
        from miner import validate
        if args.path:
            ok, res = validate.validate_local(args.path)
        elif args.repo and args.digest:
            ok, res = validate.validate_remote(args.repo, args.digest)
        else:
            parser.error("check-hippius needs --path OR (--repo and --digest)")
        _print_checks(ok, res)
        return 0 if ok else 1

    if args.cmd == "commit":
        from miner import commit
        from config_validation.models import ModelRef
        ref = ModelRef(repo=args.repo, digest=args.digest)
        result = commit.commit_reveal(ref, coldkey=args.coldkey, hotkey=args.hotkey,
                                      netuid=args.netuid, network=args.network, assume_yes=args.yes)
        return 0 if result is not None else 1

    if args.cmd == "register":
        from miner import register as reg
        uid = reg.register(args.coldkey, args.hotkey, args.netuid, args.network, assume_yes=args.yes)
        return 0 if uid is not None else 1

    if args.cmd == "check-commit":
        from miner import check_commits
        check_commits.print_commits(args.netuid, args.network, args.hotkey)
        return 0

    if args.cmd == "publish":
        from miner import publish
        ok, _ = publish.run(path=args.path, namespace=args.namespace, name=args.name,
                            coldkey=args.coldkey, hotkey=args.hotkey, netuid=args.netuid,
                            network=args.network, log=print, assume_yes=args.yes,
                            skip_commit=args.skip_commit)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

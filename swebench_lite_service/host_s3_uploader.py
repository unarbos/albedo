from __future__ import annotations

import argparse
import json
import mimetypes
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from .s3_publish import build_index, king_slug, publication_plan, run_summary

_REGION = "decentralized"
_CACHE_CONTROL = "no-cache, must-revalidate"
_BOTO_CFG = Config(connect_timeout=15, read_timeout=45, retries={"mode": "adaptive", "max_attempts": 3})


@dataclass(frozen=True)
class King:
    repo: str
    digest: str
    reign_number: int | None = None
    crowned_at: str = ""
    challenge_id: str = ""
    source: str = ""

    @property
    def key(self) -> str:
        return f"{self.repo}@{self.digest}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["key"] = self.key
        return data


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    args = parse_args()
    if args.loop:
        while True:
            run_once(args)
            time.sleep(args.interval)
    else:
        run_once(args)


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    mirror = Path(args.local_mirror).expanduser().resolve()
    if args.sync:
        sync_remote(args, mirror)
    state_path = mirror / "state.json"
    if not state_path.exists():
        result = {
            "ok": True,
            "uploaded": False,
            "uploaded_count": 0,
            "uploaded_at": utc_now(),
            "skipped_reason": f"no mirrored state at {state_path}",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    os.environ["ALBEDO_SWEBENCH_STATE_DIR"] = str(mirror)
    if args.s3_prefix:
        os.environ["ALBEDO_SWEBENCH_S3_PREFIX"] = args.s3_prefix

    state = json.loads(state_path.read_text())
    publish_result = publish_state(state=state, mirror=mirror, remote_state_dir=args.remote_state_dir.rstrip("/"))
    state["host_s3_upload"] = publish_result
    state["updated_at"] = utc_now()
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    print(json.dumps(publish_result, indent=2, sort_keys=True))
    return publish_result


def sync_remote(args: argparse.Namespace, mirror: Path) -> None:
    mirror.mkdir(parents=True, exist_ok=True)
    remote = f"{args.remote}:{args.remote_state_dir.rstrip('/')}/"
    ssh = f"ssh -p {args.ssh_port} -o StrictHostKeyChecking=no"
    cmd = ["rsync", "-az", "--delete", "-e", ssh, remote, f"{mirror}/"]
    subprocess.run(cmd, check=True)


def publish_state(*, state: dict[str, Any], mirror: Path, remote_state_dir: str) -> dict[str, Any]:
    config = s3_config()
    missing = [key for key, value in config.items() if key in {"access_key", "secret_key", "bucket"} and not value]
    if missing:
        return {
            "ok": False,
            "uploaded": False,
            "uploaded_count": 0,
            "uploaded_at": utc_now(),
            "missing_config": missing,
            "skipped_reason": "missing host S3 config: " + ", ".join(missing),
        }

    client = boto3.client(
        "s3",
        endpoint_url=config["endpoint"],
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
        region_name=_REGION,
        config=_BOTO_CFG,
    )
    uploaded: list[str] = []
    errors: list[dict[str, str]] = []

    def put_bytes(key: str, body: bytes, content_type: str) -> None:
        try:
            client.put_object(
                Bucket=config["bucket"],
                Key=key,
                Body=body,
                ContentType=content_type,
                CacheControl=_CACHE_CONTROL,
            )
            uploaded.append(key)
        except Exception as exc:
            errors.append({"key": key, "error": repr(exc)})

    def put_json(key: str, data: dict[str, Any]) -> None:
        put_bytes(key, json.dumps(data, indent=2, sort_keys=True, default=str).encode(), "application/json")

    def put_file(key: str, path_value: str | None, content_type: str | None = None) -> None:
        if not path_value:
            return
        local_path = localize_path(path_value, mirror=mirror, remote_state_dir=remote_state_dir)
        if not local_path.exists() or not local_path.is_file():
            errors.append({"key": key, "error": f"missing local artifact {local_path}"})
            return
        guessed = content_type or mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        put_bytes(key, local_path.read_bytes(), guessed)

    for key, row in state.get("benchmarks", {}).items():
        if row.get("status") != "complete":
            continue
        king = king_from_row(row.get("king") or {})
        row_for_summary = dict(row)
        plan = publication_plan(king=king, result=row_for_summary)
        run_key = f"{config['prefix']}/runs/{row['run_id']}"
        put_json(f"{run_key}/summary.json", run_summary(king=king, result=row_for_summary, plan=plan))
        put_json(f"{config['prefix']}/kings/{king_slug(king)}.json", run_summary(king=king, result=row_for_summary, plan=plan))
        put_file(f"{run_key}/predictions.jsonl", row.get("predictions_path"), "application/jsonl")
        put_file(f"{run_key}/raw_generations.json", row.get("raw_generations_path"), "application/json")
        put_file(f"{run_key}/official-report.json", row.get("summary_path"), "application/json")
        row["s3"] = {
            **plan,
            "uploaded": True,
            "pending_host_upload": False,
            "uploaded_at": utc_now(),
            "uploaded_by": "host_s3_uploader",
        }

    put_json(f"{config['prefix']}/state.json", state)
    put_json(f"{config['prefix']}/index.json", build_index(state))

    return {
        "ok": not errors,
        "uploaded": not errors,
        "uploaded_count": len(uploaded),
        "uploaded_at": utc_now(),
        "bucket": config["bucket"],
        "endpoint": config["endpoint"],
        "prefix": config["prefix"],
        "uploaded_keys": uploaded,
        "errors": errors,
    }


def localize_path(path_value: str, *, mirror: Path, remote_state_dir: str) -> Path:
    if path_value.startswith(remote_state_dir + "/"):
        return mirror / path_value[len(remote_state_dir):].lstrip("/")
    return Path(path_value)


def king_from_row(row: dict[str, Any]) -> King:
    return King(
        repo=str(row.get("repo") or ""),
        digest=str(row.get("digest") or ""),
        reign_number=row.get("reign_number"),
        crowned_at=str(row.get("crowned_at") or ""),
        challenge_id=str(row.get("challenge_id") or ""),
        source=str(row.get("source") or ""),
    )


def s3_config() -> dict[str, str]:
    return {
        "endpoint": os.environ.get("ALBEDO_SWEBENCH_S3_ENDPOINT") or os.environ.get("ALBEDO_DS_ENDPOINT") or "https://s3.hippius.com",
        "bucket": os.environ.get("ALBEDO_SWEBENCH_S3_BUCKET") or os.environ.get("ALBEDO_DS_BUCKET") or "albedo",
        "access_key": os.environ.get("ALBEDO_SWEBENCH_S3_ACCESS_KEY") or os.environ.get("ALBEDO_DS_ACCESS_KEY") or "",
        "secret_key": os.environ.get("ALBEDO_SWEBENCH_S3_SECRET_KEY") or os.environ.get("ALBEDO_DS_SECRET_KEY") or "",
        "prefix": (os.environ.get("ALBEDO_SWEBENCH_S3_PREFIX") or "swebench-lite").strip("/"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror SWE-bench Lite artifacts from the pod and upload them to S3 from this host")
    parser.add_argument("--remote", default=os.environ.get("ALBEDO_SWEBENCH_REMOTE", "root@216.243.220.131"))
    parser.add_argument("--ssh-port", default=os.environ.get("ALBEDO_SWEBENCH_SSH_PORT", "40008"))
    parser.add_argument("--remote-state-dir", default=os.environ.get("ALBEDO_SWEBENCH_REMOTE_STATE_DIR", "/root/albedo-swebench-lite"))
    parser.add_argument("--local-mirror", default=os.environ.get("ALBEDO_SWEBENCH_LOCAL_MIRROR", "/tmp/albedo-swebench-lite-mirror"))
    parser.add_argument("--s3-prefix", default=os.environ.get("ALBEDO_SWEBENCH_S3_PREFIX", ""))
    parser.add_argument("--no-sync", dest="sync", action="store_false", help="upload from existing local mirror without rsync")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=int(os.environ.get("ALBEDO_SWEBENCH_UPLOAD_INTERVAL", "300")))
    parser.set_defaults(sync=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

WEBSITE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBSITE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

DASHBOARD_REL = "data/dashboard.json"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json",
    ".jsonl": "application/jsonl",
    ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
}
SKIP_NAMES = {"push_to_hippius.py", "generate-mock.mjs"}
SKIP_SUFFIXES = {".py", ".md", ".mjs", ".bak", ".pyc"}

NO_CACHE = "no-cache, must-revalidate"
ASSET_CACHE = "public, max-age=86400"


def load_env(path: Path) -> None:
    if not path.is_file():
        print(f"warning: {path} not found; relying on existing environment", file=sys.stderr)
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def make_client(endpoint: str, access_key: str, secret_key: str):
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        sys.exit("boto3 is required: pip install boto3")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="decentralized",
        config=Config(connect_timeout=15, read_timeout=60, retries={"mode": "adaptive", "max_attempts": 3}),
    )


def content_type_for(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def cache_for(key: str) -> str:
    return NO_CACHE if key.endswith((".json", ".html")) else ASSET_CACHE


def upload(client, bucket: str, key: str, path: Path, *, dry_run: bool) -> None:
    ctype = content_type_for(path)
    cc = cache_for(key)
    size = path.stat().st_size
    if dry_run:
        print(f"  [dry-run] {key}  ({ctype}, {size} B)")
        return
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=path.read_bytes(),
        ContentType=ctype,
        CacheControl=cc,
        ACL="public-read",
    )
    print(f"  uploaded {key}  ({size} B)")


def iter_website_files():
    for path in sorted(WEBSITE_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield path.relative_to(WEBSITE_DIR).as_posix(), path


def main() -> int:
    ap = argparse.ArgumentParser(description="Push the Albedo dashboard / website to Hippius S3.")
    ap.add_argument("--website", action="store_true", help="upload the whole static site, not just the dashboard")
    ap.add_argument("--file", type=Path, help="upload a single arbitrary file (with --key)")
    ap.add_argument("--key", help="S3 key for --file (defaults to the file name)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be uploaded; do nothing")
    args = ap.parse_args()

    load_env(ENV_PATH)
    bucket = os.environ.get("ALBEDO_S3_BUCKET") or "albedo"
    endpoint = os.environ.get("ALBEDO_S3_ENDPOINT") or "https://s3.hippius.com"
    access_key = os.environ.get("ALBEDO_S3_ACCESS_KEY", "")
    secret_key = os.environ.get("ALBEDO_S3_SECRET_KEY", "")

    if not args.dry_run and not (access_key and secret_key):
        sys.exit("missing ALBEDO_S3_ACCESS_KEY / ALBEDO_S3_SECRET_KEY (set them in .env)")

    client = None if args.dry_run else make_client(endpoint, access_key, secret_key)
    print(f"target: {endpoint}/{bucket}/  (region=decentralized, acl=public-read)")

    if args.file:
        key = args.key or args.file.name
        if not args.file.is_file():
            sys.exit(f"file not found: {args.file}")
        items = [(key, args.file)]
    elif args.website:
        items = list(iter_website_files())
    else:
        dash = WEBSITE_DIR / DASHBOARD_REL
        if not dash.is_file():
            sys.exit(f"{dash} not found — run `node data/generate-mock.mjs` first")
        items = [(DASHBOARD_REL, dash)]

    print(f"uploading {len(items)} file(s):")
    for key, path in items:
        upload(client, bucket, key, path, dry_run=args.dry_run)

    public = f"{endpoint.rstrip('/')}/{bucket}/{DASHBOARD_REL}"
    print(f"\ndone. dashboard URL: {public}")
    print("point js/config.js DATA_ENDPOINTS at that URL for production.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

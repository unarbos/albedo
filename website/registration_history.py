#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

WEBSITE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBSITE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = WEBSITE_DIR / "data"

TAOSTATS_URL = "https://api.taostats.io/api/subnet/neuron/registration/v1"
PERIODS = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}
FIELDS = ["timestamp", "block_number", "uid", "hotkey", "coldkey", "registration_cost", "registration_cost_tao"]

log = logging.getLogger("registration-history")


def load_env(path: Path) -> None:
    if not path.is_file():
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


def _body_items(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, list):
        return body
    if not isinstance(body, dict):
        return []
    for key in ("results", "data", "items", "registrations"):
        value = body.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _body_items(value)
            if nested:
                return nested
    return []


def _pagination(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    if isinstance(body.get("pagination"), dict):
        return body["pagination"]
    data = body.get("data")
    if isinstance(data, dict) and isinstance(data.get("pagination"), dict):
        return data["pagination"]
    return {}


def _num(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _ss58(value: Any) -> Any:
    return value.get("ss58") if isinstance(value, dict) else value


def _timestamp(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value / 1000) if value > 1_000_000_000_000 else int(value)
    if isinstance(value, str):
        try:
            return _timestamp(float(value))
        except ValueError:
            try:
                return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
            except ValueError:
                return None
    return None


def normalize(row: dict[str, Any]) -> dict[str, Any] | None:
    ts = _timestamp(_pick(row, "timestamp", "registration_block_time"))
    cost = _num(_pick(row, "registration_cost", "registration_price"))
    hotkey = _ss58(row.get("hotkey"))
    if ts is None or cost is None or not hotkey:
        return None
    return {
        "timestamp": ts,
        "block_number": _num(_pick(row, "block_number", "block")),
        "uid": _num(_pick(row, "uid", "UID")),
        "hotkey": hotkey,
        "coldkey": _ss58(_pick(row, "coldkey", "owner")),
        "registration_cost": cost,
        "registration_cost_tao": cost / 1_000_000_000,
    }


def fetch_30d(*, api_key: str, netuid: int, now: datetime) -> list[dict[str, Any]]:
    start = int((now - PERIODS["30d"]).timestamp())
    end = int(now.timestamp())
    page: int | str | None = 1
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=30) as client:
        while page is not None:
            resp = client.get(
                TAOSTATS_URL,
                headers={"Authorization": api_key},
                params={
                    "netuid": netuid,
                    "timestamp_start": start,
                    "timestamp_end": end,
                    "limit": 200,
                    "page": page,
                    "order": "timestamp_asc",
                },
            )
            resp.raise_for_status()
            body = resp.json()
            rows.extend(item for item in (normalize(x) for x in _body_items(body)) if item)
            page = _pagination(body).get("next_page")
    return sorted(rows, key=lambda r: (r["timestamp"], r["uid"] if r["uid"] is not None else -1))


def period_rows(rows: list[dict[str, Any]], *, now: datetime, delta: timedelta) -> list[dict[str, Any]]:
    start = int((now - delta).timestamp())
    return [row for row in rows if row["timestamp"] >= start]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def content_type(path: Path) -> str:
    return "text/csv; charset=utf-8" if path.suffix == ".csv" else "application/json"


def upload_to_hippius(key: str, path: Path) -> bool:
    bucket = os.environ.get("ALBEDO_S3_BUCKET") or "albedo"
    access = os.environ.get("ALBEDO_S3_ACCESS_KEY")
    secret = os.environ.get("ALBEDO_S3_SECRET_KEY")
    if not (access and secret):
        log.warning("ALBEDO_S3_* unset; kept local %s", path.name)
        return False
    endpoint = os.environ.get("ALBEDO_S3_ENDPOINT") or "https://s3.hippius.com"
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            region_name="decentralized",
            config=Config(connect_timeout=15, read_timeout=60, retries={"mode": "adaptive", "max_attempts": 3}),
        )
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=path.read_bytes(),
            ContentType=content_type(path),
            CacheControl="no-cache, must-revalidate",
            ACL="public-read",
        )
        return True
    except Exception as exc:
        log.error("upload failed for %s: %s", key, exc)
        return False


def generate(*, api_key: str, netuid: int) -> None:
    now = datetime.now(UTC)
    rows_30d = fetch_30d(api_key=api_key, netuid=netuid, now=now)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    uploads = []
    for name, delta in PERIODS.items():
        rows = period_rows(rows_30d, now=now, delta=delta)
        for suffix, writer in ((".json", write_json), (".csv", write_csv)):
            path = DATA_DIR / f"registrations_{name}{suffix}"
            writer(path, rows)
            uploads.append(upload_to_hippius(f"data/{path.name}", path))
        log.info("wrote registrations_%s: %d rows", name, len(rows))
    log.info("published registration history upload=%s", "ok" if all(uploads) else "FAILED")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish TaoStats subnet 97 registration history JSON/CSV.")
    parser.add_argument("--once", action="store_true", help="Generate once and exit (default: loop)")
    parser.add_argument("--netuid", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    load_env(ENV_PATH)
    api_key = os.environ.get("TAOSTATS_API_KEY")
    if not api_key:
        sys.exit("TAOSTATS_API_KEY is not set")
    netuid = args.netuid if args.netuid is not None else int(os.environ.get("ALBEDO_DASHBOARD_NETUID", "97"))
    interval = float(os.environ.get("ALBEDO_REGISTRATION_HISTORY_INTERVAL_S", "300"))

    if args.once:
        generate(api_key=api_key, netuid=netuid)
        return 0

    while True:
        try:
            generate(api_key=api_key, netuid=netuid)
        except Exception as exc:
            log.error("tick failed: %s", exc)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())

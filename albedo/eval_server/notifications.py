"""Slack notifications for eval-server failures."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_WEBHOOK_URL = os.environ.get("ALBEDO_SLACK_WEBHOOK_URL", "").strip()
_USERNAME = os.environ.get("ALBEDO_SLACK_USERNAME", "Albedo Eval Server").strip() or "Albedo Eval Server"
_ICON_URL = os.environ.get("ALBEDO_SLACK_ICON_URL", "").strip()
_COOLDOWN_S = max(0, int(os.environ.get("ALBEDO_SLACK_COOLDOWN_S", "300")))
_MAX_TEXT_LEN = 2800
_MAX_DETAILS_LEN = 1200

_LOCK = threading.Lock()
_LAST_SENT_AT: dict[str, float] = {}


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _should_send(key: str) -> bool:
    if not key or _COOLDOWN_S <= 0:
        return True
    now = time.monotonic()
    with _LOCK:
        last = _LAST_SENT_AT.get(key)
        if last is not None and now - last < _COOLDOWN_S:
            return False
        _LAST_SENT_AT[key] = now
        return True


def _post(payload: dict) -> None:
    if not _WEBHOOK_URL:
        return
    req = urllib.request.Request(
        _WEBHOOK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", "replace")
            if resp.status != 200:
                log.warning("Slack webhook returned HTTP %s: %s", resp.status, body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        log.warning("Slack webhook HTTP %s: %s", exc.code, body)
    except Exception:
        log.warning("Slack webhook send failed", exc_info=True)


async def notify_problem(
    title: str,
    message: str,
    *,
    dedupe_key: str = "",
    details: str = "",
    status_code: int | None = None,
    source: str = "eval-server",
) -> bool:
    """Send a throttled Slack alert if the webhook is configured."""
    if not _WEBHOOK_URL:
        return False

    key = dedupe_key or f"{source}:{status_code or ''}:{title}:{message[:120]}"
    if not _should_send(key):
        return False

    lines = [f"*{_clip(title, 200)}*", _clip(message, 900)]
    meta = [f"`source={source}`"]
    if status_code is not None:
        meta.append(f"`status={status_code}`")
    meta.append(f"`time={datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}`")
    lines.append(" ".join(meta))
    if details:
        lines.append(f"```{_clip(details, _MAX_DETAILS_LEN)}```")

    payload = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": _clip("\n".join(lines), _MAX_TEXT_LEN),
                },
            }
        ],
        "username": _USERNAME,
    }
    if _ICON_URL:
        payload["icon_url"] = _ICON_URL

    await asyncio.to_thread(_post, payload)
    return True

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger


@dataclass(frozen=True)
class EvalErrorNotification:
    component: str
    severity: str
    message: str
    eval_run_id: str | None = None
    submission_id: str | None = None
    batch_id: str | None = None
    fault_class: str | None = None
    fault_code: str | None = None
    provider_route: str | None = None
    scoring_mode: str | None = None
    retryable: bool | None = None
    details: dict[str, Any] | None = None


_SENT: dict[tuple[str, str, str], float] = {}


def notify_eval_error(event: EvalErrorNotification, *, webhook_url: str | None = None) -> None:
    url = webhook_url or _webhook_url()
    if not url:
        return
    if _is_duplicate(event):
        return
    message = _format_message(event)
    try:
        response = httpx.post(
            url,
            json=_slack_payload(message),
            headers={"Content-Type": "application/json"},
            timeout=float(os.environ.get("ALBEDO_SLACK_ERROR_TIMEOUT_SECONDS", "10")),
        )
        if response.status_code != 200:
            logger.warning("Slack error notification failed: status={} body={}", response.status_code, response.text[:500])
        else:
            logger.info("Slack error notification sent for eval_run_id={}", event.eval_run_id or "")
    except Exception as exc:
        logger.warning("Slack error notification failed: {}", exc)


def _slack_payload(message: str) -> dict[str, Any]:
    return {
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message},
            }
        ],
        "username": os.environ.get("ALBEDO_SLACK_ERROR_USERNAME", "Albedo Eval Alerts"),
        "icon_url": os.environ.get(
            "ALBEDO_SLACK_ERROR_ICON_URL",
            "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
        ),
    }


def _webhook_url() -> str:
    return (
        os.environ.get("ALBEDO_SLACK_ERROR_WEBHOOK_URL")
        or os.environ.get("ALBEDO_JUDGE_SLACK_ERROR_WEBHOOK_URL")
        or os.environ.get("ALBEDO_REMOTE_SLACK_ERROR_WEBHOOK_URL")
        or ""
    )


def _is_duplicate(event: EvalErrorNotification) -> bool:
    window = float(os.environ.get("ALBEDO_SLACK_ERROR_DEDUPE_SECONDS", "300"))
    key = (event.eval_run_id or "", event.component, event.fault_code or event.message[:80])
    now = time.monotonic()
    previous = _SENT.get(key)
    if previous is not None and now - previous < window:
        return True
    _SENT[key] = now
    return False


def _format_message(event: EvalErrorNotification) -> str:
    env = os.environ.get("ALBEDO_SLACK_ERROR_ENV", "")
    prefix = f"[{env}] " if env else ""
    fields = [
        f"component={event.component}",
        f"severity={event.severity}",
    ]
    for name in (
        "eval_run_id",
        "submission_id",
        "batch_id",
        "fault_class",
        "fault_code",
        "provider_route",
        "scoring_mode",
    ):
        value = getattr(event, name)
        if value:
            fields.append(f"{name}={value}")
    if event.retryable is not None:
        fields.append(f"retryable={event.retryable}")
    safe_details = _redact_details(event.details or {})
    if safe_details:
        fields.append(f"details={safe_details}")
    return f"{prefix}Albedo eval/scoring alert: {event.message}\n" + " | ".join(fields)


def _redact_details(details: dict[str, Any]) -> dict[str, Any]:
    blocked = ("secret", "token", "key", "authorization", "prompt", "output", "raw", "response")
    redacted: dict[str, Any] = {}
    for key, value in details.items():
        lower = str(key).lower()
        if any(part in lower for part in blocked):
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return redacted

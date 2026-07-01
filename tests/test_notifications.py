from __future__ import annotations

import httpx

from albedo_eval_service.notifications import EvalErrorNotification, notify_eval_error


def test_notify_eval_error_sends_slack_blocks_and_redacts(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200
        text = "ok"

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("ALBEDO_SLACK_ERROR_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("ALBEDO_SLACK_ERROR_DEDUPE_SECONDS", "0")

    notify_eval_error(
        EvalErrorNotification(
            component="judge-api",
            severity="error",
            message="Judge failed",
            eval_run_id="run-1",
            fault_code="judge_provider_exhausted",
            details={"api_key": "secret", "batch": 4},
        ),
        webhook_url="https://hooks.slack.test/services/example",
    )

    assert len(calls) == 1
    payload = calls[0]["json"]
    assert calls[0]["headers"] == {"Content-Type": "application/json"}
    assert calls[0]["timeout"] == 10.0
    assert payload["username"] == "Albedo Eval Alerts"
    assert payload["icon_url"].startswith("https://github.githubassets.com/")
    text = payload["blocks"][0]["text"]["text"]
    assert "Judge failed" in text
    assert "judge_provider_exhausted" in text
    assert "[redacted]" in text
    assert "secret" not in text

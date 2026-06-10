from __future__ import annotations

import asyncio
from uuid import uuid4

from albedo_eval_service.config import Settings
from albedo_eval_service.dispatcher import EvalDispatcher


class RecordingRepository:
    def __init__(self):
        self.events = []
        self.heartbeats = 0

    def record_remote_event(self, *, submission_id, attempt_id, event):
        self.events.append(event)

    def heartbeat_attempt(self, *, attempt_id, lease_seconds):
        self.heartbeats += 1


class PollingClient:
    def __init__(self):
        self.calls = 0

    async def iter_events(self, remote_run_id):
        self.calls += 1
        batches = [
            [{"type": "eval_started"}],
            [{"type": "eval_started"}, {"type": "generation_started"}],
            [
                {"type": "eval_started"},
                {"type": "generation_started"},
                {"type": "verdict", "state": "succeeded", "valid_turns": 2},
            ],
        ]
        for event in batches[min(self.calls - 1, len(batches) - 1)]:
            yield event

    async def get_eval(self, remote_run_id):
        return {"state": "generating"}


def test_follow_until_verdict_polls_and_records_only_new_events():
    repo = RecordingRepository()
    dispatcher = EvalDispatcher(
        settings=Settings(
            database_url="postgresql://example",
            dataset_manifest_uri="s3://manifest",
            judge_config_hash="sha256:judge",
            remote_event_poll_seconds=0,
        ),
        repository=repo,
    )

    verdict = asyncio.run(
        dispatcher._follow_until_verdict(
            PollingClient(),
            submission_id=uuid4(),
            attempt_id=uuid4(),
            remote_run_id="remote-1",
        )
    )

    assert verdict["state"] == "succeeded"
    assert [event["type"] for event in repo.events] == ["eval_started", "generation_started", "verdict"]
    assert repo.heartbeats == 3

from __future__ import annotations

import asyncio
import json

from albedo_eval_service.judge_api import JudgeSample, ScoreBatchRequest, _score_samples
from albedo_eval_service.judge_core import JUDGE_MODELS, METRIC_KEYS
from albedo_eval_service.judge_openrouter import JudgeRawResponse


class FakeJudgeClient:
    async def score(self, *, model, messages):
        return JudgeRawResponse(
            model=model,
            provider="fake-provider",
            raw=json.dumps({metric: 2 for metric in METRIC_KEYS}),
        )


def test_score_samples_uses_real_pairwise_parser_and_counterbalance():
    request = ScoreBatchRequest(
        eval_run_id="run-1",
        batch_id="score-0001",
        judge_models=list(JUDGE_MODELS[:3]),
        total_sample_count=2,
        samples=[
            JudgeSample(sample_id="s1", prompt="task 1", previous_king_output="king", challenger_output="challenger", sample_index=0),
            JudgeSample(sample_id="s2", prompt="task 2", previous_king_output="king", challenger_output="challenger", sample_index=1),
        ],
    )

    records = asyncio.run(_score_samples(client=FakeJudgeClient(), request=request))

    assert records[0]["order"] == ["previous_king", "challenger"]
    assert records[0]["sample_score"] == 1.0
    assert records[1]["order"] == ["challenger", "previous_king"]
    assert records[1]["sample_score"] == 0.0
    assert all(len(record["judge_results"]) == 3 for record in records)

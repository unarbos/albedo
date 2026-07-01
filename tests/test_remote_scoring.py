from __future__ import annotations

import types
from uuid import uuid4

from albedo_eval_service.judge_core import should_show_challenger_first
from albedo_eval_service.remote_dataset import EvalSample
from albedo_eval_service.remote_generation import GenerationResult
from albedo_eval_service.remote_scoring import (
    _category_prep_payload,
    _counterbalanced_sample_indices,
    _score_batch_payloads,
)


def _samples(counts: dict[str, int]) -> list[EvalSample]:
    # Build namespaced sample ids (<source>/data/...:row:turn) grouped contiguously by source,
    # exactly as the sampler emits them.
    samples: list[EvalSample] = []
    for source, n in counts.items():
        for row in range(n):
            sid = f"{source}/data/train-00000.parquet:{row}:0"
            samples.append(EvalSample(sample_id=sid, prompt=f"task {sid}"))
    return samples


def _per_source_buckets(index_map: dict[str, int], samples: list[EvalSample], total: int):
    boundary = (total + 1) // 2
    buckets: dict[str, list[int]] = {}
    for sample in samples:
        source = sample.sample_id.split("/", 1)[0]
        bucket = buckets.setdefault(source, [0, 0])
        bucket[0 if index_map[sample.sample_id] < boundary else 1] += 1
    return buckets


def test_counterbalance_is_a_clean_permutation():
    samples = _samples({"swe-zero": 90, "mini-coder": 38})
    index_map = _counterbalanced_sample_indices(samples)
    assert sorted(index_map.values()) == list(range(len(samples)))


def test_counterbalance_splits_each_source_evenly_70_30():
    samples = _samples({"swe-zero": 90, "mini-coder": 38})
    index_map = _counterbalanced_sample_indices(samples)
    buckets = _per_source_buckets(index_map, samples, len(samples))
    # Both allocations are even -> exact 50/50 across the two judge positions.
    assert buckets["swe-zero"] == [45, 45]
    assert buckets["mini-coder"] == [19, 19]


def test_counterbalance_integrates_with_judge():
    samples = _samples({"swe-zero": 90, "mini-coder": 38})
    total = len(samples)
    index_map = _counterbalanced_sample_indices(samples)
    for source in ("swe-zero", "mini-coder"):
        src = [s for s in samples if s.sample_id.startswith(f"{source}/")]
        challenger_first = sum(
            1 for s in src if should_show_challenger_first(index_map[s.sample_id], total)
        )
        king_first = len(src) - challenger_first
        assert challenger_first > 0 and king_first > 0  # neither position empty (the bug)
        assert abs(challenger_first - king_first) <= 1


def test_counterbalance_generalizes_to_any_sources_and_weights():
    # Three equal sources at an odd total forces odd per-source counts (the 50/50 ±1 path).
    samples = _samples({"a": 33, "b": 33, "c": 33})
    total = len(samples)
    index_map = _counterbalanced_sample_indices(samples)
    boundary = (total + 1) // 2  # 50
    # The two position buckets stay evenly sized globally: 50 (model 1) / 49 (model 2).
    assert sum(1 for v in index_map.values() if v < boundary) == 50
    for source, (front, back) in _per_source_buckets(index_map, samples, total).items():
        assert front > 0 and back > 0
        assert abs(front - back) == 1  # 33 is odd


def test_score_batch_payload_emits_counterbalanced_index():
    samples = _samples({"swe-zero": 4, "mini-coder": 2})
    king = [GenerationResult(sample_id=s.sample_id, text="king") for s in samples]
    challenger = [GenerationResult(sample_id=s.sample_id, text="challenger out") for s in samples]
    request = types.SimpleNamespace(
        eval_run_id=uuid4(),
        scoring=types.SimpleNamespace(judge_count=1),
        dataset=types.SimpleNamespace(scoring_batch_size=100),
    )
    payloads = _score_batch_payloads(request, samples, king, challenger)
    emitted = {s["sample_id"]: s["sample_index"] for p in payloads for s in p["samples"]}
    assert emitted == _counterbalanced_sample_indices(samples)
    assert payloads[0]["total_sample_count"] == len(samples)


def test_category_prep_payload_emits_counterbalanced_index():
    samples = _samples({"swe-zero": 4, "mini-coder": 2})
    request = types.SimpleNamespace(eval_run_id=uuid4())
    payload = _category_prep_payload(request, samples)
    emitted = {s["sample_id"]: s["sample_index"] for s in payload["samples"]}
    assert emitted == _counterbalanced_sample_indices(samples)

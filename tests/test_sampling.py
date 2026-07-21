import pytest

from albedo_eval_service.sampling import BUCKETS, _scaled_buckets, multi_source_manifest_sample_ids

_BUCKET_TOTAL = sum(count for _, count in BUCKETS)


def _shard(source: str, rows: int, *, asst: int = 12):
    # `asst` assistant turns per row is deep enough for every bucket (max need is 11).
    return {
        "name": f"{source}/data/train-00000.parquet",
        "rows": rows,
        "rows_meta": [{"iid": f"{source}-{i}", "asst": asst} for i in range(rows)],
    }


def _manifest(swe_rows: int = 300, mini_rows: int = 300) -> dict:
    return {
        "version": "swe-zero+mini-coder-v1",
        "sources": [
            {"name": "swe-zero", "weight": 0.7, "shards": [_shard("swe-zero", swe_rows)], "total_rows": swe_rows},
            {"name": "mini-coder", "weight": 0.3, "shards": [_shard("mini-coder", mini_rows)], "total_rows": mini_rows},
        ],
        "total_rows": swe_rows + mini_rows,
    }


def test_rejects_single_source_manifest():
    legacy = {"shards": [{"name": "data/train-00000.parquet", "rows": 3}], "total_rows": 3}
    with pytest.raises(ValueError, match="sources"):
        multi_source_manifest_sample_ids(legacy, block_hash="0xabc", sample_count=_BUCKET_TOTAL)


def test_requires_rows_meta():
    no_meta = {
        "sources": [
            {"name": "swe-zero", "weight": 0.7, "shards": [{"name": "swe-zero/data/train-0.parquet", "rows": 1}]},
        ]
    }
    with pytest.raises(ValueError, match="rows_meta"):
        multi_source_manifest_sample_ids(no_meta, block_hash="0xabc", sample_count=_BUCKET_TOTAL)


def test_sample_count_scales_buckets():
    ids = multi_source_manifest_sample_ids(_manifest(), block_hash="0xabc", sample_count=128)
    assert len(ids) == 128 == len(set(ids))
    assert _scaled_buckets(128) == [(prefix, count * 2) for prefix, count in BUCKETS]


def test_rejects_nonpositive_sample_count():
    with pytest.raises(ValueError, match="positive"):
        multi_source_manifest_sample_ids(_manifest(), block_hash="0xabc", sample_count=0)


def test_stable_and_seed_sensitive():
    first = multi_source_manifest_sample_ids(_manifest(), block_hash="0xabc", sample_count=_BUCKET_TOTAL)
    second = multi_source_manifest_sample_ids(_manifest(), block_hash="0xabc", sample_count=_BUCKET_TOTAL)
    third = multi_source_manifest_sample_ids(_manifest(), block_hash="0xdef", sample_count=_BUCKET_TOTAL)
    assert first == second
    assert first != third
    assert len(first) == _BUCKET_TOTAL == len(set(first))


def test_70_30_split_and_unique_instances():
    ids = multi_source_manifest_sample_ids(_manifest(), block_hash="0xabc", sample_count=_BUCKET_TOTAL)
    swe = [i for i in ids if i.startswith("swe-zero/")]
    mini = [i for i in ids if i.startswith("mini-coder/")]
    assert len(swe) == 45
    assert len(mini) == 19
    # each sample is a distinct (shard, row) => a unique instance/rollout
    coords = {i.rsplit(":", 1)[0] for i in ids}
    assert len(coords) == _BUCKET_TOTAL


def test_bucket_turn_distribution_and_feasibility():
    manifest = _manifest()
    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=_BUCKET_TOTAL)
    # turn_idx = (Y-1)//2, so bucket depth Y maps to turn indices 1..10 with the bucket counts.
    from collections import Counter

    hist = Counter(int(i.rsplit(":", 1)[1]) for i in ids)
    assert dict(hist) == {(Y - 1) // 2: count for Y, count in _scaled_buckets(_BUCKET_TOTAL)}

    meta = {
        (shard["name"], row): entry["asst"]
        for source in manifest["sources"]
        for shard in source["shards"]
        for row, entry in enumerate(shard["rows_meta"])
    }
    for sid in ids:
        name, row, turn = sid.rsplit(":", 2)
        prefix_len = 2 * int(turn) + 1
        assert meta[(name, int(row))] >= (prefix_len + 1) // 2  # feasibility


def test_infeasible_deep_buckets_raise():
    # No row is deep enough for the deepest buckets -> a clear infeasibility error.
    shallow = _manifest(swe_rows=300, mini_rows=300)
    for source in shallow["sources"]:
        for shard in source["shards"]:
            for entry in shard["rows_meta"]:
                entry["asst"] = 2  # only supports the Y=3 bucket
    with pytest.raises(ValueError, match="infeasible"):
        multi_source_manifest_sample_ids(shallow, block_hash="0xabc", sample_count=_BUCKET_TOTAL)


def test_rejects_traversal_shard_name():
    bad = {
        "sources": [
            {
                "name": "x",
                "weight": 1.0,
                "shards": [{"name": "../data/train-0.parquet", "rows": 1, "rows_meta": [{"iid": "a", "asst": 3}]}],
            }
        ]
    }
    with pytest.raises(ValueError, match="data/train"):
        multi_source_manifest_sample_ids(bad, block_hash="0xabc", sample_count=_BUCKET_TOTAL)

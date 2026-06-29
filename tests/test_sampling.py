import pytest

from albedo_eval_service.sampling import (
    multi_source_manifest_sample_ids,
    swe_zero_manifest_sample_ids,
)


MANIFEST = {
    "shards": [
        {"name": "data/train-00000.parquet", "rows": 3},
        {"name": "data/train-00001.parquet", "rows": 2},
    ],
    "total_rows": 5,
}


MULTI_MANIFEST = {
    "version": "swe-zero+mini-coder-v1",
    "sources": [
        {
            "name": "swe-zero",
            "weight": 0.7,
            "shards": [{"name": "swe-zero/data/train-00000.parquet", "rows": 1000}],
            "total_rows": 1000,
        },
        {
            "name": "mini-coder",
            "weight": 0.3,
            "shards": [{"name": "mini-coder/data/train-00000-of-00060.parquet", "rows": 1000}],
            "total_rows": 1000,
        },
    ],
    "total_rows": 2000,
}


def test_swe_zero_manifest_sample_ids_are_stable():
    first = swe_zero_manifest_sample_ids(
        MANIFEST,
        block_hash="0xabc",
        sample_count=8,
        max_turns_per_sample=3,
    )
    second = swe_zero_manifest_sample_ids(
        MANIFEST,
        block_hash="0xabc",
        sample_count=8,
        max_turns_per_sample=3,
    )

    assert first == second
    assert len(first) == 8
    assert all(sample.startswith("data/train-") for sample in first)


def test_different_block_hash_changes_swe_zero_ordering():
    first = swe_zero_manifest_sample_ids(
        MANIFEST,
        block_hash="0xabc",
        sample_count=8,
        max_turns_per_sample=3,
    )
    second = swe_zero_manifest_sample_ids(
        MANIFEST,
        block_hash="0xdef",
        sample_count=8,
        max_turns_per_sample=3,
    )

    assert first != second


def test_missing_block_hash_is_rejected():
    with pytest.raises(ValueError, match="block_hash"):
        swe_zero_manifest_sample_ids(MANIFEST, block_hash="", sample_count=1)


def test_manifest_total_rows_is_validated():
    bad_manifest = {"shards": [{"name": "data/train-00000.parquet", "rows": 2}], "total_rows": 3}

    with pytest.raises(ValueError, match="total_rows"):
        swe_zero_manifest_sample_ids(bad_manifest, block_hash="0xabc", sample_count=1)


def test_sampling_accepts_real_manifest_path_key():
    manifest = {
        "shards": [{"path": "data/train-00000.parquet", "rows": 2, "sha256": "abc"}],
        "total_rows": 2,
    }

    sample_ids = swe_zero_manifest_sample_ids(
        manifest,
        block_hash="0xabc",
        sample_count=2,
        max_turns_per_sample=1,
    )

    assert all(sample_id.startswith("data/train-00000.parquet:") for sample_id in sample_ids)


def test_multi_source_rejects_single_source_manifest():
    # No fallback to one dataset: a manifest without `sources` must be rejected.
    with pytest.raises(ValueError, match="sources"):
        multi_source_manifest_sample_ids(
            MANIFEST, block_hash="0xabc", sample_count=8, max_turns_per_sample=3
        )


def test_multi_source_is_stable():
    first = multi_source_manifest_sample_ids(
        MULTI_MANIFEST, block_hash="0xabc", sample_count=128, max_turns_per_sample=10
    )
    second = multi_source_manifest_sample_ids(
        MULTI_MANIFEST, block_hash="0xabc", sample_count=128, max_turns_per_sample=10
    )
    assert first == second
    assert len(first) == 128


def test_multi_source_splits_70_30():
    ids = multi_source_manifest_sample_ids(
        MULTI_MANIFEST, block_hash="0xabc", sample_count=128, max_turns_per_sample=10
    )
    swe = [i for i in ids if i.startswith("swe-zero/")]
    coder = [i for i in ids if i.startswith("mini-coder/")]
    assert len(swe) == 90
    assert len(coder) == 38
    assert len(swe) + len(coder) == 128


def test_multi_source_preserves_v1_selection_per_source():
    # "same sampling method, same choice of samples": each source's picks equal exactly
    # what the single-source v1 sampler chooses for that source at its allocated count.
    # Adding mini-coder does not change how swe-zero rows are selected — it is additive.
    ids = multi_source_manifest_sample_ids(
        MULTI_MANIFEST, block_hash="0xabc", sample_count=128, max_turns_per_sample=10
    )
    swe_ids = [i for i in ids if i.startswith("swe-zero/")]

    swe_only = {
        "shards": [{"name": "swe-zero/data/train-00000.parquet", "rows": 1000}],
        "total_rows": 1000,
    }
    v1_swe = swe_zero_manifest_sample_ids(
        swe_only, block_hash="0xabc", sample_count=90, max_turns_per_sample=10
    )
    assert swe_ids == v1_swe


def test_multi_source_block_hash_changes_ordering():
    first = multi_source_manifest_sample_ids(
        MULTI_MANIFEST, block_hash="0xabc", sample_count=128, max_turns_per_sample=10
    )
    second = multi_source_manifest_sample_ids(
        MULTI_MANIFEST, block_hash="0xdef", sample_count=128, max_turns_per_sample=10
    )
    assert first != second


def test_multi_source_redistributes_shortfall():
    # Source "a" can only produce 1 coordinate (1 row x 1 turn); its allocation shortfall
    # must be topped up deterministically from source "b" so the total still equals
    # sample_count.
    tiny = {
        "sources": [
            {"name": "a", "weight": 0.5, "shards": [{"name": "a/data/train-0.parquet", "rows": 1}], "total_rows": 1},
            {"name": "b", "weight": 0.5, "shards": [{"name": "b/data/train-0.parquet", "rows": 100}], "total_rows": 100},
        ],
        "total_rows": 101,
    }
    ids = multi_source_manifest_sample_ids(
        tiny, block_hash="0xabc", sample_count=10, max_turns_per_sample=1
    )
    assert len(ids) == 10
    assert len(set(ids)) == 10
    assert sum(1 for i in ids if i.startswith("a/")) == 1


def test_multi_source_rejects_traversal_shard_name():
    bad = {
        "sources": [
            {"name": "x", "weight": 1.0, "shards": [{"name": "../data/train-0.parquet", "rows": 1}]},
        ],
    }
    with pytest.raises(ValueError, match="data/train"):
        multi_source_manifest_sample_ids(bad, block_hash="0xabc", sample_count=1)

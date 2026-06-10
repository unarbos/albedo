import pytest

from albedo_eval_service.sampling import swe_zero_manifest_sample_ids


MANIFEST = {
    "shards": [
        {"name": "data/train-00000.parquet", "rows": 3},
        {"name": "data/train-00001.parquet", "rows": 2},
    ],
    "total_rows": 5,
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

import hashlib
import json
from uuid import uuid4

import pytest

from albedo_config import Settings
from albedo_eval_service.control.dispatcher import build_eval_request
from albedo_eval_service.shared.models import DatasetConfig

_FAMILIES = ("pr", "lm", "combine", "mechanical")


def _shard(source: str, rows: int):
    return {
        "name": f"{source}/data/train-00000.parquet",
        "rows": rows,
        "rows_meta": [
            {
                "iid": f"{source}-{i}",
                "asst": 12,
                "first_edit": 4 + (i % 5),
                "family": _FAMILIES[i % len(_FAMILIES)],
            }
            for i in range(rows)
        ],
    }


def test_eval_defaults_keep_64_samples_and_32_batches():
    settings = Settings(
        database_url="postgresql://example",
        dataset_manifest_uri="s3://bucket/manifest.json",
        judge_config_hash="sha256:judge",
    )
    dataset = DatasetConfig(
        version="test",
        manifest_uri="s3://bucket/manifest.json",
        manifest_hash="sha256:manifest",
        sample_count=settings.sample_count,
        sample_seed="seed",
        sampling_algo="algo",
    )

    assert settings.sample_count == 100
    assert dataset.generation_batch_size == 32
    assert dataset.scoring_batch_size == 32


def test_build_eval_request_rejects_single_source_manifest(tmp_path):
    manifest = {"shards": [{"name": "data/train-00000.parquet", "rows": 2}], "total_rows": 2}
    payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(payload)
    manifest_hash = hashlib.sha256(payload).hexdigest()

    settings = Settings(
        database_url="postgresql://example",
        dataset_manifest_uri="s3://albedo-artifacts/datasets/swe-zero/manifest.json",
        dataset_manifest_hash=manifest_hash,
        dataset_manifest_path=str(manifest_path),
        sample_count=3,
        judge_config_hash="sha256:judge",
    )

    with pytest.raises(ValueError, match="sources"):
        build_eval_request(
            settings,
            {
                "id": uuid4(),
                "model_uri": "s3://models/challenger",
                "model_hash": "sha256:challenger",
                "block_hash": "0xabc",
            },
            {
                "model_uri": "s3://models/king",
                "model_hash": "sha256:king",
                "king_version": 1,
            },
            uuid4(),
        )


def test_build_eval_request_samples_multi_source_manifest(tmp_path):
    manifest = {
        "version": "swe-zero+mini-coder-v1",
        "sources": [
            {"name": "swe-zero", "shards": [_shard("swe-zero", 400)], "total_rows": 400},
            {"name": "mini-coder", "shards": [_shard("mini-coder", 400)], "total_rows": 400},
        ],
        "total_rows": 800,
    }
    payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(payload)
    manifest_hash = hashlib.sha256(payload).hexdigest()

    settings = Settings(
        database_url="postgresql://example",
        dataset_manifest_uri="s3://albedo-artifacts/datasets/swe-zero/manifest.json",
        dataset_manifest_hash=manifest_hash,
        dataset_manifest_path=str(manifest_path),
        sample_count=64,
        judge_config_hash="sha256:judge",
    )

    request = build_eval_request(
        settings,
        {
            "id": uuid4(),
            "model_uri": "s3://models/challenger",
            "model_hash": "sha256:challenger",
            "block_hash": "0xabc",
        },
        {"model_uri": "s3://models/king", "model_hash": "sha256:king", "king_version": 1},
        uuid4(),
    )

    assert len(request.dataset.sample_ids) == 64
    prefixes = {sid.split("/", 1)[0] for sid in request.dataset.sample_ids}
    assert prefixes == {"swe-zero", "mini-coder"}

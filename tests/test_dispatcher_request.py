import hashlib
import json
from uuid import uuid4

from albedo_eval_service.config import Settings
from albedo_eval_service.dispatcher import build_eval_request


def test_build_eval_request_samples_local_manifest(tmp_path):
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
        max_turns_per_sample=2,
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
        {
            "model_uri": "s3://models/king",
            "model_hash": "sha256:king",
            "king_version": 1,
        },
        uuid4(),
    )

    assert request.dataset.sample_ids
    assert len(request.dataset.sample_ids) == 3
    assert request.dataset.manifest_hash == manifest_hash

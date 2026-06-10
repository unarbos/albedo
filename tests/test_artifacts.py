from uuid import uuid4

from albedo_eval_service.artifacts import artifact_records_from_verdict


def test_artifact_records_from_verdict_maps_known_s3_artifacts():
    submission_id = uuid4()
    attempt_id = uuid4()

    records = artifact_records_from_verdict(
        submission_id=submission_id,
        stage_attempt_id=attempt_id,
        artifacts={
            "generated_samples": "s3://albedo-artifacts/submissions/1/eval/2/generated-samples.jsonl",
            "scoring_results": "s3://albedo-artifacts/submissions/1/eval/2/scoring-results.jsonl",
            "unknown": "s3://albedo-artifacts/ignored",
            "remote_logs": "",
        },
    )

    assert [record.artifact_type for record in records] == ["GENERATED_SAMPLES", "SCORING_RESULTS"]
    assert records[0].submission_id == submission_id
    assert records[0].stage_attempt_id == attempt_id
    assert records[0].storage_backend == "s3"
    assert records[0].bucket == "albedo-artifacts"
    assert records[0].object_key == "submissions/1/eval/2/generated-samples.jsonl"


def test_artifact_records_from_verdict_handles_non_s3_uris():
    records = artifact_records_from_verdict(
        submission_id=uuid4(),
        stage_attempt_id=uuid4(),
        artifacts={"verdict": "hippius://bucket/key/verdict.json"},
    )

    assert len(records) == 1
    assert records[0].artifact_type == "EVAL_VERDICT"
    assert records[0].storage_backend == "hippius"
    assert records[0].bucket is None
    assert records[0].object_key is None

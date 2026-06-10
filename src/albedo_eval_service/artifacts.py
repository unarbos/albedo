from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


VERDICT_ARTIFACT_TYPES = {
    "transcript": "EVAL_TRANSCRIPT",
    "generated_samples": "GENERATED_SAMPLES",
    "scoring_results": "SCORING_RESULTS",
    "judge_results": "JUDGE_RESULTS",
    "verdict": "EVAL_VERDICT",
    "remote_logs": "REMOTE_LOGS",
    "progress": "REMOTE_PROGRESS",
}


@dataclass(frozen=True)
class ArtifactRecord:
    submission_id: UUID
    stage_attempt_id: UUID
    artifact_type: str
    storage_backend: str
    uri: str
    bucket: str | None
    object_key: str | None


def artifact_records_from_verdict(
    *,
    submission_id: UUID,
    stage_attempt_id: UUID,
    artifacts: dict[str, str],
) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    for artifact_name, uri in sorted(artifacts.items()):
        if not uri:
            continue
        artifact_type = VERDICT_ARTIFACT_TYPES.get(artifact_name)
        if not artifact_type:
            continue
        bucket, object_key = _split_s3_uri(uri)
        records.append(
            ArtifactRecord(
                submission_id=submission_id,
                stage_attempt_id=stage_attempt_id,
                artifact_type=artifact_type,
                storage_backend="s3" if uri.startswith("s3://") else "hippius",
                uri=uri,
                bucket=bucket,
                object_key=object_key,
            )
        )
    return records


def _split_s3_uri(uri: str) -> tuple[str | None, str | None]:
    if not uri.startswith("s3://"):
        return None, None
    without_scheme = uri.removeprefix("s3://")
    bucket, _, object_key = without_scheme.partition("/")
    return bucket or None, object_key or None

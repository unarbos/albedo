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
    sha256: str | None = None
    size_bytes: int | None = None
    content_type: str | None = None


def artifact_records_from_verdict(
    *,
    submission_id: UUID,
    stage_attempt_id: UUID,
    artifacts: dict[str, str],
    artifact_metadata: dict[str, object] | None = None,
) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    for artifact_name, uri in sorted(artifacts.items()):
        if not uri:
            continue
        artifact_type = VERDICT_ARTIFACT_TYPES.get(artifact_name)
        if not artifact_type:
            continue
        metadata = artifact_metadata.get(artifact_name) if artifact_metadata else None
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        bucket, object_key = _split_s3_uri(uri)
        records.append(
            ArtifactRecord(
                submission_id=submission_id,
                stage_attempt_id=stage_attempt_id,
                artifact_type=artifact_type,
                storage_backend=_storage_backend_for_uri(uri),
                uri=uri,
                bucket=bucket or _optional_str(metadata_dict.get("bucket")),
                object_key=object_key or _optional_str(metadata_dict.get("object_key")),
                sha256=_optional_str(metadata_dict.get("sha256")),
                size_bytes=_optional_int(metadata_dict.get("size_bytes")),
                content_type=_optional_str(metadata_dict.get("content_type")),
            )
        )
    return records


def _split_s3_uri(uri: str) -> tuple[str | None, str | None]:
    if not uri.startswith("s3://"):
        return None, None
    without_scheme = uri.removeprefix("s3://")
    bucket, _, object_key = without_scheme.partition("/")
    return bucket or None, object_key or None


def _storage_backend_for_uri(uri: str) -> str:
    if uri.startswith("s3://"):
        return "s3"
    if uri.startswith(("local-cache://", "file://")):
        return "local-cache"
    if uri.startswith("hf://"):
        return "hf"
    return "hippius"


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None

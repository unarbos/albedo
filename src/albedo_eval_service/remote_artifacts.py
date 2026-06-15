from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from .remote_config import RemoteSettings


@dataclass(frozen=True)
class ArtifactUpload:
    name: str
    uri: str
    bucket: str
    object_key: str
    sha256: str
    size_bytes: int
    content_type: str
    local_path: Path

    def metadata(self) -> dict[str, object]:
        return {
            "uri": self.uri,
            "bucket": self.bucket,
            "object_key": self.object_key,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
        }


class ArtifactUploader(Protocol):
    def upload_run_artifacts(self, *, eval_run_id: UUID, artifact_prefix: str, files: dict[str, Path]) -> dict[str, ArtifactUpload]:
        ...


class LocalOnlyArtifactUploader:
    def upload_run_artifacts(self, *, eval_run_id: UUID, artifact_prefix: str, files: dict[str, Path]) -> dict[str, ArtifactUpload]:
        uploads: dict[str, ArtifactUpload] = {}
        bucket, prefix = split_s3_prefix(artifact_prefix)
        for name, path in files.items():
            object_key = f"{prefix}/{path.name}" if prefix else path.name
            uploads[name] = ArtifactUpload(
                name=name,
                uri=f"local-cache://{path}" if not bucket else f"local-cache://{bucket}/{object_key}",
                bucket=bucket or "",
                object_key=object_key,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                content_type=content_type_for(path),
                local_path=path,
            )
        return uploads


class S3ArtifactUploader:
    def __init__(self, settings: RemoteSettings):
        self.settings = settings

    def upload_run_artifacts(self, *, eval_run_id: UUID, artifact_prefix: str, files: dict[str, Path]) -> dict[str, ArtifactUpload]:
        bucket, prefix = split_s3_prefix(artifact_prefix)
        if not bucket:
            raise ValueError(f"artifact_prefix must be an s3:// URI, got {artifact_prefix}")

        import boto3

        session_kwargs: dict[str, str] = {}
        if self.settings.s3_access_key_id:
            session_kwargs["aws_access_key_id"] = self.settings.s3_access_key_id
        if self.settings.s3_secret_access_key:
            session_kwargs["aws_secret_access_key"] = self.settings.s3_secret_access_key
        if self.settings.s3_session_token:
            session_kwargs["aws_session_token"] = self.settings.s3_session_token
        if self.settings.s3_region:
            session_kwargs["region_name"] = self.settings.s3_region

        client_kwargs: dict[str, str] = {}
        if self.settings.s3_endpoint_url:
            client_kwargs["endpoint_url"] = self.settings.s3_endpoint_url

        client = boto3.session.Session(**session_kwargs).client("s3", **client_kwargs)
        uploads: dict[str, ArtifactUpload] = {}
        for name, path in files.items():
            object_key = f"{prefix}/{path.name}" if prefix else path.name
            content_type = content_type_for(path)
            checksum = sha256_file(path)
            client.upload_file(
                str(path),
                bucket,
                object_key,
                ExtraArgs={
                    "ContentType": content_type,
                    "Metadata": {"sha256": checksum, "eval_run_id": str(eval_run_id), "artifact_name": name},
                },
            )
            uploads[name] = ArtifactUpload(
                name=name,
                uri=f"s3://{bucket}/{object_key}",
                bucket=bucket,
                object_key=object_key,
                sha256=checksum,
                size_bytes=path.stat().st_size,
                content_type=content_type,
                local_path=path,
            )
        return uploads


class RunArtifactSpool:
    def __init__(self, root: str | Path, eval_run_id: UUID):
        self.run_dir = Path(root) / str(eval_run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(self, filename: str, rows: list[dict[str, Any]]) -> Path:
        path = self.run_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        return path

    def write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / filename
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return path

    def write_text(self, filename: str, payload: str) -> Path:
        path = self.run_dir / filename
        path.write_text(payload, encoding="utf-8")
        return path

    def cleanup(self) -> None:
        shutil.rmtree(self.run_dir, ignore_errors=True)


def build_artifact_uploader(settings: RemoteSettings) -> ArtifactUploader:
    if not settings.upload_artifacts:
        return LocalOnlyArtifactUploader()
    return S3ArtifactUploader(settings)


def split_s3_prefix(uri: str) -> tuple[str | None, str]:
    if not uri.startswith("s3://"):
        return None, uri.rstrip("/")
    without_scheme = uri.removeprefix("s3://").rstrip("/")
    bucket, _, prefix = without_scheme.partition("/")
    return bucket or None, prefix.strip("/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def content_type_for(path: Path) -> str:
    if path.suffix == ".jsonl":
        return "application/x-ndjson"
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".txt":
        return "text/plain"
    return "application/octet-stream"

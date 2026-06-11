from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx

from .canonical_model_config import apply_canonical_model_config
from .remote_config import RemoteSettings


_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ResolvedModel:
    original_ref: str
    local_path: str
    source: str
    cache_hit: bool
    file_count: int
    total_size_bytes: int

    def as_event(self, *, side: str) -> dict[str, object]:
        return {
            "side": side,
            "source": self.source,
            "original_ref": self.original_ref,
            "local_path": self.local_path,
            "cache_hit": self.cache_hit,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
        }


class ModelArtifactResolver:
    def __init__(self, settings: RemoteSettings):
        self.settings = settings
        self.cache_root = Path(settings.model_cache_dir)

    def resolve(self, model_ref: str) -> ResolvedModel:
        if not self.settings.resolve_model_artifacts:
            return ResolvedModel(
                original_ref=model_ref,
                local_path=model_ref,
                source="disabled",
                cache_hit=True,
                file_count=0,
                total_size_bytes=0,
            )

        local_path = Path(model_ref)
        if local_path.exists():
            return self._resolved_local(model_ref, local_path, source="local")
        if model_ref.startswith("file://"):
            path = Path(urlparse(model_ref).path)
            if not path.exists():
                raise FileNotFoundError(f"model file URI does not exist: {model_ref}")
            return self._resolved_local(model_ref, path, source="local")
        if model_ref.startswith("s3://"):
            return self._resolve_s3(model_ref)
        parsed_oci = parse_oci_ref(model_ref)
        if parsed_oci:
            registry, repository, digest = parsed_oci
            return self._resolve_oci(registry=registry, repository=repository, digest=digest, original_ref=model_ref)
        return ResolvedModel(
            original_ref=model_ref,
            local_path=model_ref,
            source="passthrough",
            cache_hit=True,
            file_count=0,
            total_size_bytes=0,
        )

    def _resolved_local(self, original_ref: str, path: Path, *, source: str) -> ResolvedModel:
        self._apply_canonical_config(path)
        file_count, total_size_bytes = _tree_stats(path)
        return ResolvedModel(
            original_ref=original_ref,
            local_path=str(path),
            source=source,
            cache_hit=True,
            file_count=file_count,
            total_size_bytes=total_size_bytes,
        )

    def _resolve_s3(self, model_ref: str) -> ResolvedModel:
        bucket, prefix = split_s3_uri(model_ref)
        cache_dir = self.cache_root / "s3" / bucket / prefix.strip("/")
        done_marker = cache_dir / ".albedo-model-cache.json"
        if done_marker.exists():
            self._apply_canonical_config(cache_dir)
            file_count, total_size_bytes = _tree_stats(cache_dir)
            return ResolvedModel(model_ref, str(cache_dir), "s3", True, file_count, total_size_bytes)

        import boto3

        cache_dir.mkdir(parents=True, exist_ok=True)
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

        paginator = client.get_paginator("list_objects_v2")
        found = False
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith("/"):
                    continue
                found = True
                rel = key[len(prefix):].lstrip("/") if prefix else key
                destination = cache_dir / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, key, str(destination))
        if not found:
            raise FileNotFoundError(f"no model objects found under {model_ref}")
        done_marker.write_text(json.dumps({"source": model_ref}, sort_keys=True) + "\n", encoding="utf-8")
        self._apply_canonical_config(cache_dir)
        file_count, total_size_bytes = _tree_stats(cache_dir)
        return ResolvedModel(model_ref, str(cache_dir), "s3", False, file_count, total_size_bytes)

    def _resolve_oci(self, *, registry: str, repository: str, digest: str, original_ref: str) -> ResolvedModel:
        cache_dir = self.cache_root / "oci" / registry / repository.replace("/", "__") / digest.removeprefix("sha256:")
        done_marker = cache_dir / ".albedo-model-cache.json"
        if done_marker.exists():
            self._apply_canonical_config(cache_dir)
            file_count, total_size_bytes = _tree_stats(cache_dir)
            return ResolvedModel(original_ref, str(cache_dir), "oci", True, file_count, total_size_bytes)

        temp_dir = cache_dir.with_suffix(".partial")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        with httpx.Client(timeout=None, follow_redirects=True) as client:
            manifest_url = f"https://{registry}/v2/{repository}/manifests/{digest}"
            headers = {"Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"}
            response = client.get(manifest_url, headers=headers)
            if response.status_code == 401:
                token = _bearer_token(client, response, repository)
                response = client.get(manifest_url, headers={**headers, "Authorization": f"Bearer {token}"})
            response.raise_for_status()
            _verify_digest(response.content, digest, label="manifest")
            manifest = response.json()
            token = None
            if response.request.headers.get("Authorization", "").startswith("Bearer "):
                token = response.request.headers["Authorization"].removeprefix("Bearer ")
            for index, layer in enumerate(manifest.get("layers", [])):
                layer_digest = layer.get("digest")
                if not isinstance(layer_digest, str) or not _DIGEST_RE.match(layer_digest):
                    raise ValueError(f"OCI layer {index} is missing a sha256 digest")
                name = _layer_filename(layer, index)
                destination = temp_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                blob_url = f"https://{registry}/v2/{repository}/blobs/{layer_digest}"
                blob_headers = {"Authorization": f"Bearer {token}"} if token else {}
                auth_response = _stream_blob_to_file(client, blob_url, blob_headers, destination, layer_digest, label=name)
                if auth_response is not None:
                    token = _bearer_token(client, auth_response, repository)
                    _stream_blob_to_file(client, blob_url, {"Authorization": f"Bearer {token}"}, destination, layer_digest, label=name)
        done_marker_payload = {"source": original_ref, "registry": registry, "repository": repository, "digest": digest}
        (temp_dir / ".albedo-model-cache.json").write_text(json.dumps(done_marker_payload, sort_keys=True) + "\n", encoding="utf-8")
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.replace(cache_dir)
        self._apply_canonical_config(cache_dir)
        file_count, total_size_bytes = _tree_stats(cache_dir)
        return ResolvedModel(original_ref, str(cache_dir), "oci", False, file_count, total_size_bytes)

    def _apply_canonical_config(self, model_dir: Path) -> None:
        if not self.settings.use_canonical_model_config:
            return
        if not model_dir.is_dir():
            return
        apply_canonical_model_config(model_dir)


def _stream_blob_to_file(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    destination: Path,
    expected_digest: str,
    *,
    label: str,
) -> httpx.Response | None:
    temp_destination = destination.with_suffix(destination.suffix + ".download")
    digest = hashlib.sha256()
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code == 401:
            return response
        response.raise_for_status()
        with temp_destination.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                digest.update(chunk)
                handle.write(chunk)
    actual = "sha256:" + digest.hexdigest()
    if actual != expected_digest:
        temp_destination.unlink(missing_ok=True)
        raise ValueError(f"{label} digest mismatch: expected {expected_digest}, got {actual}")
    temp_destination.replace(destination)
    return None


def parse_oci_ref(model_ref: str) -> tuple[str, str, str] | None:
    ref = model_ref.removeprefix("oci://").removeprefix("docker://")
    if "@sha256:" not in ref:
        return None
    left, digest_tail = ref.rsplit("@", 1)
    digest = digest_tail
    if not _DIGEST_RE.match(digest):
        return None
    registry, sep, repository = left.partition("/")
    if not sep or not repository or "." not in registry:
        return None
    return registry, repository, digest


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3:// URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _bearer_token(client: httpx.Client, response: httpx.Response, repository: str) -> str:
    challenge = response.headers.get("www-authenticate", "")
    if not challenge.lower().startswith("bearer "):
        raise RuntimeError("registry returned 401 without a bearer challenge")
    params = _parse_auth_challenge(challenge[len("Bearer "):])
    realm = params.get("realm")
    if not realm:
        raise RuntimeError("registry bearer challenge did not include a realm")
    query = dict(parse_qsl(urlparse(realm).query))
    if params.get("service"):
        query["service"] = params["service"]
    query.setdefault("scope", f"repository:{repository}:pull")
    token_url = realm.split("?", 1)[0] + "?" + urlencode(query)
    token_response = client.get(token_url)
    token_response.raise_for_status()
    payload = token_response.json()
    token = payload.get("token") or payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("registry token endpoint did not return a token")
    return token


def _parse_auth_challenge(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in re.finditer(r'(\w+)="([^"]*)"', raw):
        values[part.group(1)] = part.group(2)
    return values


def _layer_filename(layer: dict[str, Any], index: int) -> str:
    annotations = layer.get("annotations") if isinstance(layer.get("annotations"), dict) else {}
    title = annotations.get("org.opencontainers.image.title")
    if isinstance(title, str) and title and not title.startswith("/") and ".." not in Path(title).parts:
        return title
    digest = str(layer.get("digest", f"layer-{index}"))
    return digest.replace(":", "-")


def _verify_digest(payload: bytes, expected: str, *, label: str) -> None:
    actual = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} digest mismatch: expected {expected}, got {actual}")


def _tree_stats(path: Path) -> tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size
    file_count = 0
    total_size = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_size += item.stat().st_size
    return file_count, total_size

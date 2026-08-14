from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx

from albedo_config import RemoteSettings

from .canonical_model_config import apply_canonical_model_config

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HF_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
_DEFAULT_OCI_REGISTRY = "registry.hippius.com"
_HEARTBEAT_INTERVAL_S = 10.0
_MODEL_PAYLOAD_PATTERNS = ["*.safetensors", "model.safetensors.index.json"]

_POINTER_MEDIA_TYPE_V2 = "application/vnd.hippius.pointer.v2"
_LAYOUT_ANNOTATION_KEY = "com.hippius.layout"


def _manifest_is_chunked(manifest: dict[str, Any]) -> bool:
    if (manifest.get("annotations") or {}).get(_LAYOUT_ANNOTATION_KEY):
        return True
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == _POINTER_MEDIA_TYPE_V2:
            return True
        if (layer.get("annotations") or {}).get(_LAYOUT_ANNOTATION_KEY):
            return True
    return False


def _is_model_payload_file(name: str) -> bool:
    return name.endswith(".safetensors") or name == "model.safetensors.index.json"


_RESOLVE_LOCKS: dict[str, threading.Lock] = {}
_RESOLVE_LOCKS_GUARD = threading.Lock()


def _resolve_lock(model_ref: str) -> threading.Lock:
    with _RESOLVE_LOCKS_GUARD:
        return _RESOLVE_LOCKS.setdefault(model_ref, threading.Lock())


_DOWNLOAD_GATE = threading.Lock()


@contextmanager
def _download_slot(label: str):
    if not _DOWNLOAD_GATE.acquire(blocking=False):
        print(f"model_download_queued ref={label} waiting_for_active_download", flush=True)
        wait_start = time.monotonic()
        _DOWNLOAD_GATE.acquire()
        print(
            f"model_download_slot_acquired ref={label} waited_s={time.monotonic() - wait_start:.0f}",  # noqa: E501
            flush=True,
        )
    try:
        yield
    finally:
        _DOWNLOAD_GATE.release()


@contextmanager
def _download_heartbeat(label: str, watch_dir: Path | None = None):
    stop = threading.Event()
    start = time.monotonic()

    def _beat() -> None:
        while not stop.wait(_HEARTBEAT_INTERVAL_S):
            suffix = ""
            if watch_dir is not None:
                try:
                    suffix = f" bytes={_dir_written_bytes(watch_dir)}"
                except OSError:
                    suffix = ""
            print(
                f"model_download_progress ref={label} elapsed_s={time.monotonic() - start:.0f}{suffix}",  # noqa: E501
                flush=True,
            )

    thread = threading.Thread(target=_beat, name="model-dl-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


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
        with _resolve_lock(model_ref):
            return self._resolve_unlocked(model_ref)

    def _resolve_unlocked(self, model_ref: str) -> ResolvedModel:
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
        if model_ref.startswith("hf://"):
            return self._resolve_hf(model_ref)
        parsed_oci = parse_oci_ref(model_ref)
        if parsed_oci:
            registry, repository, digest = parsed_oci
            return self._resolve_oci(
                registry=registry, repository=repository, digest=digest, original_ref=model_ref
            )
        repo, sep, revision = model_ref.rpartition("@")
        if sep and "/" in repo and _HF_GIT_REVISION_RE.match(revision):
            return self._resolve_hf(model_ref)
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
            if _has_loadable_model_files(cache_dir):
                self._apply_canonical_config(cache_dir)
                file_count, total_size_bytes = _tree_stats(cache_dir)
                return ResolvedModel(
                    model_ref, str(cache_dir), "s3", True, file_count, total_size_bytes
                )
            print(
                f"model_cache_invalid source=s3 path={cache_dir} reason=missing_model_files",
                flush=True,
            )
            shutil.rmtree(cache_dir, ignore_errors=True)

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
        with _download_slot(model_ref), _download_heartbeat(model_ref, watch_dir=cache_dir):
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = item["Key"]
                    if key.endswith("/"):
                        continue
                    found = True
                    rel = key[len(prefix) :].lstrip("/") if prefix else key
                    if not _is_model_payload_file(Path(rel).name):
                        continue
                    destination = cache_dir / rel
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(bucket, key, str(destination))
        if not found:
            raise FileNotFoundError(f"no model objects found under {model_ref}")
        _require_loadable_model_files(cache_dir, source="s3")
        done_marker.write_text(
            json.dumps({"source": model_ref}, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._apply_canonical_config(cache_dir)
        file_count, total_size_bytes = _tree_stats(cache_dir)
        return ResolvedModel(model_ref, str(cache_dir), "s3", False, file_count, total_size_bytes)

    def _resolve_hf(self, model_ref: str) -> ResolvedModel:
        ref = model_ref.removeprefix("hf://")
        repo, _, revision = ref.partition("@")
        if not repo or not revision:
            raise ValueError(f"hf:// ref must be 'hf://<repo>@<revision>': {model_ref}")
        cache_dir = self.cache_root / "hf" / repo.replace("/", "__") / revision
        done_marker = cache_dir / ".albedo-model-cache.json"
        if done_marker.exists():
            if _has_loadable_model_files(cache_dir):
                self._apply_canonical_config(cache_dir)
                file_count, total_size_bytes = _tree_stats(cache_dir)
                return ResolvedModel(
                    model_ref, str(cache_dir), "hf", True, file_count, total_size_bytes
                )
            print(
                f"model_cache_invalid source=hf path={cache_dir} reason=missing_model_files",
                flush=True,
            )
            shutil.rmtree(cache_dir, ignore_errors=True)

        temp_dir = cache_dir.with_suffix(".partial")
        temp_dir.mkdir(parents=True, exist_ok=True)
        with _download_slot(model_ref):
            self._download_hf_snapshot(
                repo=repo, revision=revision, temp_dir=temp_dir, label=model_ref
            )
        _require_loadable_model_files(temp_dir, source="hf")
        (temp_dir / ".albedo-model-cache.json").write_text(
            json.dumps({"source": model_ref, "repo": repo, "revision": revision}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.replace(cache_dir)
        self._apply_canonical_config(cache_dir)
        file_count, total_size_bytes = _tree_stats(cache_dir)
        return ResolvedModel(model_ref, str(cache_dir), "hf", False, file_count, total_size_bytes)

    def _download_hf_snapshot(
        self, *, repo: str, revision: str, temp_dir: Path, label: str
    ) -> None:
        if not self.settings.model_download_out_of_process:
            os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
            from huggingface_hub import snapshot_download

            with _download_heartbeat(label, watch_dir=temp_dir):
                snapshot_download(
                    repo_id=repo,
                    revision=revision,
                    local_dir=str(temp_dir),
                    max_workers=max(1, self.settings.model_download_concurrency),
                    allow_patterns=_MODEL_PAYLOAD_PATTERNS,
                    token=os.environ.get("HF_TOKEN") or None,
                )
            return

        _run_hf_download_supervised(
            repo=repo,
            revision=revision,
            temp_dir=temp_dir,
            label=label,
            concurrency=self.settings.model_download_concurrency,
            stall_seconds=self.settings.model_download_stall_seconds,
            max_attempts=max(1, self.settings.model_download_stall_retries),
        )

    def _download_hippius_snapshot(
        self, *, registry: str, repository: str, digest: str, temp_dir: Path, label: str
    ) -> None:
        if registry != _DEFAULT_OCI_REGISTRY:
            raise RuntimeError(
                f"chunked OCI manifest served by unsupported registry {registry!r}; "
                f"hippius_hub only targets {_DEFAULT_OCI_REGISTRY}"
            )
        try:
            from hippius_hub._oci import group_files
        except ImportError as exc:
            raise RuntimeError(
                "chunked hippius artifact requires hippius-hub>=0.6.0 on this host; "
                "older readers silently save the pointer blob as the model file"
            ) from exc

        if not self.settings.model_download_out_of_process:
            import hippius_hub

            with _download_heartbeat(label, watch_dir=temp_dir):
                hippius_hub.snapshot_download(
                    repository,
                    revision=digest,
                    local_dir=str(temp_dir),
                    max_workers=max(1, self.settings.model_download_concurrency),
                    allow_patterns=_MODEL_PAYLOAD_PATTERNS,
                    token=os.environ.get("HIPPIUS_HUB_TOKEN") or None,
                )
            return

        _run_hf_download_supervised(
            repo=repository,
            revision=digest,
            temp_dir=temp_dir,
            label=label,
            concurrency=self.settings.model_download_concurrency,
            stall_seconds=self.settings.hippius_download_stall_seconds,
            max_attempts=max(1, self.settings.model_download_stall_retries),
            child_entry="_hippius_download_child",
        )

    def _resolve_oci(
        self, *, registry: str, repository: str, digest: str, original_ref: str
    ) -> ResolvedModel:
        cache_dir = (
            self.cache_root
            / "oci"
            / registry
            / repository.replace("/", "__")
            / digest.removeprefix("sha256:")
        )
        done_marker = cache_dir / ".albedo-model-cache.json"
        if done_marker.exists():
            if _has_loadable_model_files(cache_dir):
                self._apply_canonical_config(cache_dir)
                file_count, total_size_bytes = _tree_stats(cache_dir)
                return ResolvedModel(
                    original_ref, str(cache_dir), "oci", True, file_count, total_size_bytes
                )
            print(
                f"model_cache_invalid source=oci path={cache_dir} reason=missing_model_files",
                flush=True,
            )
            shutil.rmtree(cache_dir, ignore_errors=True)

        temp_dir = cache_dir.with_suffix(".partial")
        temp_dir.mkdir(parents=True, exist_ok=True)

        oci_timeout = httpx.Timeout(
            connect=30.0,
            read=self.settings.model_download_read_timeout_seconds,
            write=30.0,
            pool=30.0,
        )
        with httpx.Client(timeout=oci_timeout, follow_redirects=True) as client:
            manifest_url = f"https://{registry}/v2/{repository}/manifests/{digest}"
            headers = {
                "Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"  # noqa: E501
            }
            response = client.get(manifest_url, headers=headers)
            if response.status_code == 401:
                token = _bearer_token(client, response, repository)
                response = client.get(
                    manifest_url, headers={**headers, "Authorization": f"Bearer {token}"}
                )
            response.raise_for_status()
            _verify_digest(response.content, digest, label="manifest")
            manifest = response.json()
            chunked = _manifest_is_chunked(manifest)
            token = None
            if response.request.headers.get("Authorization", "").startswith("Bearer "):
                token = response.request.headers["Authorization"].removeprefix("Bearer ")
            pending: list[tuple[str, str]] = []
            layers = [] if chunked else manifest.get("layers", [])
            for index, layer in enumerate(layers):
                layer_digest = layer.get("digest")
                if not isinstance(layer_digest, str) or not _DIGEST_RE.match(layer_digest):
                    raise ValueError(f"OCI layer {index} is missing a sha256 digest")
                name = _layer_filename(layer, index)
                if not _is_model_payload_file(Path(name).name):
                    continue
                destination = temp_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    print(f"model_download_skip ref={name} (already cached)", flush=True)
                    continue
                pending.append((layer_digest, name))

            token_lock = threading.Lock()

            def _download_layer(layer_digest: str, name: str) -> None:
                nonlocal token
                destination = temp_dir / name
                blob_url = f"https://{registry}/v2/{repository}/blobs/{layer_digest}"
                with token_lock:
                    current = token
                blob_headers = {"Authorization": f"Bearer {current}"} if current else {}
                auth_response = _stream_blob_to_file(
                    client, blob_url, blob_headers, destination, layer_digest, label=name
                )
                if auth_response is not None:
                    with token_lock:
                        if token == current:
                            token = _bearer_token(client, auth_response, repository)
                        current = token
                    _stream_blob_to_file(
                        client,
                        blob_url,
                        {"Authorization": f"Bearer {current}"},
                        destination,
                        layer_digest,
                        label=name,
                    )

            if pending:
                max_workers = max(1, min(self.settings.model_download_concurrency, len(pending)))
                with (
                    _download_slot(original_ref),
                    concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor,
                ):
                    futures = [
                        executor.submit(_download_layer, layer_digest, name)
                        for layer_digest, name in pending
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result()
                        except Exception:
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
        if chunked:
            with _download_slot(original_ref):
                self._download_hippius_snapshot(
                    registry=registry,
                    repository=repository,
                    digest=digest,
                    temp_dir=temp_dir,
                    label=original_ref,
                )
        _require_loadable_model_files(temp_dir, source="oci")
        done_marker_payload = {
            "source": original_ref,
            "registry": registry,
            "repository": repository,
            "digest": digest,
        }
        (temp_dir / ".albedo-model-cache.json").write_text(
            json.dumps(done_marker_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.replace(cache_dir)
        self._apply_canonical_config(cache_dir)
        file_count, total_size_bytes = _tree_stats(cache_dir)
        return ResolvedModel(
            original_ref, str(cache_dir), "oci", False, file_count, total_size_bytes
        )

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
        with (
            _download_heartbeat(label, watch_dir=destination.parent),
            temp_destination.open("wb") as handle,
        ):
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


def _hf_download_child() -> None:
    repo, revision, local_dir, max_workers = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
    )
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    try:
        from config_validation.storage import _fastdl
    except ImportError:
        _fastdl = None
    if _fastdl is not None and _fastdl.available():
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=["model.safetensors.index.json"],
            token=token,
        )
        _fastdl.fetch_shards(repo, revision, Path(local_dir), token)
        return
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=local_dir,
        max_workers=max(1, int(max_workers)),
        allow_patterns=_MODEL_PAYLOAD_PATTERNS,
        token=token,
    )


def _hippius_download_child() -> None:
    repo, revision, local_dir, max_workers = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
    )
    import hippius_hub

    hippius_hub.snapshot_download(
        repo,
        revision=revision,
        local_dir=local_dir,
        max_workers=max(1, int(max_workers)),
        allow_patterns=_MODEL_PAYLOAD_PATTERNS,
        token=os.environ.get("HIPPIUS_HUB_TOKEN") or None,
    )


def _spawn_hf_download(
    repo: str,
    revision: str,
    temp_dir: Path,
    concurrency: int,
    log_path: Path,
    child_entry: str = "_hf_download_child",
) -> subprocess.Popen:
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"from albedo_eval_service.modelstore.resolver import {child_entry}; "
                f"{child_entry}()",
                repo,
                revision,
                str(temp_dir),
                str(max(1, concurrency)),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()


def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def _tail_file(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return "(download log unavailable)"
    text = data.decode("utf-8", errors="replace").strip()
    return text or "(no output captured)"


_ERROR_LINE = re.compile(r"[A-Za-z_][\w.]*(?:Error|Exception|Interrupt)\b")


def _summarize_error_log(text: str, max_chars: int = 300) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "(no output captured)"
    for idx in range(len(lines) - 1, -1, -1):
        if _ERROR_LINE.match(lines[idx]):
            summary = lines[idx]
            follow = lines[idx + 1] if idx + 1 < len(lines) else ""
            if follow and not follow.startswith(("Please ", "If you ", "For more", "Traceback")):
                summary += " — " + follow
            return summary[:max_chars]
    return lines[-1][:max_chars]


def _run_hf_download_supervised(
    *,
    repo: str,
    revision: str,
    temp_dir: Path,
    label: str,
    concurrency: int,
    stall_seconds: float,
    max_attempts: int,
    child_entry: str = "_hf_download_child",
) -> None:
    log_path = temp_dir.parent / f"{temp_dir.name}.download.log"
    attempt = 0
    fruitless = 0
    while True:
        attempt += 1
        baseline = _dir_written_bytes(temp_dir)
        proc = _spawn_hf_download(repo, revision, temp_dir, concurrency, log_path, child_entry)
        start = time.monotonic()
        last_bytes = baseline
        last_progress = start
        progressed = False
        stalled = False
        while proc.poll() is None:
            time.sleep(_HEARTBEAT_INTERVAL_S)
            current = _dir_written_bytes(temp_dir)
            now = time.monotonic()
            print(
                f"model_download_progress ref={label} attempt={attempt} "
                f"elapsed_s={now - start:.0f} bytes={current}",
                flush=True,
            )
            if current != last_bytes:
                last_bytes = current
                last_progress = now
                progressed = True
            elif now - last_progress >= stall_seconds:
                print(
                    f"model_download_stalled ref={label} attempt={attempt} "
                    f"bytes={current} no_progress_s={now - last_progress:.0f}",
                    flush=True,
                )
                _terminate_process_group(proc)
                stalled = True
                break
        if stalled:
            fruitless = 0 if progressed else fruitless + 1
            if fruitless >= max_attempts:
                raise TimeoutError(
                    f"snapshot_download for {repo}@{revision} made no progress for "
                    f"{stall_seconds:.0f}s across {max_attempts} consecutive attempts"
                )
            continue
        if proc.returncode == 0:
            log_path.unlink(missing_ok=True)
            return
        detail = _summarize_error_log(_tail_file(log_path, 4000))
        raise RuntimeError(
            f"snapshot_download exited {proc.returncode} for {repo}@{revision}: {detail}"
        )


def parse_oci_ref(model_ref: str) -> tuple[str, str, str] | None:
    ref = model_ref.removeprefix("oci://").removeprefix("docker://")
    if "@sha256:" not in ref:
        return None
    left, digest_tail = ref.rsplit("@", 1)
    digest = digest_tail
    if not _DIGEST_RE.match(digest):
        return None
    registry, sep, repository = left.partition("/")
    if not sep or not repository:
        return None
    if "." not in registry:
        return _DEFAULT_OCI_REGISTRY, left, digest
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
    params = _parse_auth_challenge(challenge[len("Bearer ") :])
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
    if (
        isinstance(title, str)
        and title
        and not title.startswith("/")
        and ".." not in Path(title).parts
    ):
        return title
    digest = str(layer.get("digest", f"layer-{index}"))
    return digest.replace(":", "-")


def _has_loadable_model_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "model.safetensors.index.json").is_file():
        return True
    return any(path.glob("*.safetensors"))


def _require_loadable_model_files(path: Path, *, source: str) -> None:
    if _has_loadable_model_files(path):
        return
    print(
        f"model_cache_invalid source={source} path={path} reason=download_missing_model_files",
        flush=True,
    )
    shutil.rmtree(path, ignore_errors=True)
    raise FileNotFoundError(f"downloaded {source} model at {path} is missing loadable model files")


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


def _dir_written_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            st = item.stat()
        except OSError:
            continue
        if item.is_file():
            total += st.st_blocks * 512
    return total

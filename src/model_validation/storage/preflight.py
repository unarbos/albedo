"""Pre-download weight-dtype gate (HF primary, Hippius option).

Reads ONLY each safetensors shard's header using HTTP Range requests (the 8-byte
little-endian length prefix + that many header bytes), so a non-16-bit (quantized or
full-precision) repo is rejected without downloading any weight data. A header is tens
of KB; a shard is gigabytes.

For HF, ``hf_hub_url`` gives a per-file ``/resolve/`` URL that supports Range directly.
For Hippius (an OCI registry) it takes a bearer-token + manifest handshake — all the
hippius_hub OCI coupling stays localized in this module.
"""
from __future__ import annotations

import json
import os

import httpx

from config_validation.models import BACKEND_HF, ModelRef

_HUB_TOKEN_ENV = "HIPPIUS_HUB_TOKEN"
_HF_TOKEN_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
_TIMEOUT = 60


def _hippius_token() -> str | None:
    return os.environ.get(_HUB_TOKEN_ENV)


def _hf_token() -> str | None:
    for env in _HF_TOKEN_ENVS:
        tok = os.environ.get(env)
        if tok:
            return tok
    return None


def _oci_context(ref: ModelRef):
    """(registry, oci_repo, auth_headers, manifest) for ``ref`` — the bearer-token
    handshake + manifest fetch, kept in one place so the hippius_hub internals stay here."""
    from hippius_hub._oci import fetch_manifest
    from hippius_hub.auth import get_oci_bearer_token
    from hippius_hub.constants import resolve_registry
    from hippius_hub.file_download import _oci_repo_path

    registry = resolve_registry(None)
    oci_repo = _oci_repo_path(ref.repo, None)
    oci_token = get_oci_bearer_token(oci_repo, _hippius_token(), endpoint=None)
    manifest = fetch_manifest(registry, oci_repo, ref.digest, oci_token).manifest
    return registry, oci_repo, {"Authorization": f"Bearer {oci_token}"}, manifest


def _ranged(client: httpx.Client, url: str, headers: dict, start: int, end: int) -> bytes:
    """GET bytes [start, end] of ``url``, insisting the server honor the Range (206).

    Streaming + a 206 check guards against a registry that ignores Range and would
    otherwise stream the whole multi-GB blob into memory."""
    rng = {**headers, "Range": f"bytes={start}-{end}"}
    with client.stream("GET", url, headers=rng) as resp:
        if resp.status_code != 206:
            raise RuntimeError(f"server did not honor Range (status {resp.status_code}) for {url}")
        return resp.read()


def _read_header(client: httpx.Client, url: str, headers: dict) -> dict:
    """Read a safetensors header: 8-byte little-endian length, then that many JSON bytes."""
    hlen = int.from_bytes(_ranged(client, url, headers, 0, 7), "little")
    return json.loads(_ranged(client, url, headers, 8, 8 + hlen - 1))


def _dtypes_from_header(header: dict) -> set[str]:
    return {info["dtype"] for k, info in header.items() if k != "__metadata__"}


def _fetch_blob(client: httpx.Client, url: str, headers: dict) -> bytes:
    resp = client.get(url, headers=headers)
    resp.raise_for_status()
    return resp.content


def _chunk_segments(refs, start: int, end: int) -> list[tuple[str, int, int]]:
    """Map logical file bytes [start, end] onto (pack_digest, pack_start, pack_end) segments.

    ``refs`` are chunked-v2 pack-chunk refs in file order (``.size``, ``.pack_digest``,
    ``.pack_offset``); a chunk's bytes live at ``pack_offset`` inside its pack blob."""
    segments: list[tuple[str, int, int]] = []
    offset = 0
    for ref in refs:
        lo, hi = offset, offset + ref.size - 1
        if hi >= start and lo <= end:
            first = max(start, lo) - lo
            last = min(end, hi) - lo
            segments.append((ref.pack_digest, ref.pack_offset + first, ref.pack_offset + last))
        offset += ref.size
        if offset > end:
            return segments
    raise ValueError(f"byte range [{start}, {end}] exceeds chunked file size {offset}")


def _read_header_chunked(client: httpx.Client, blobs_base: str, headers: dict, refs) -> dict:
    """Read a safetensors header from a chunked-v2 file via Range reads on its pack blobs."""

    def read_range(start: int, end: int) -> bytes:
        return b"".join(
            _ranged(client, f"{blobs_base}/{digest}", headers, seg_start, seg_end)
            for digest, seg_start, seg_end in _chunk_segments(refs, start, end)
        )

    hlen = int.from_bytes(read_range(0, 7), "little")
    return json.loads(read_range(8, 8 + hlen - 1))


def _hippius_dtypes(ref: ModelRef) -> dict[str, set[str]]:
    from hippius_hub._oci import group_files, parse_pointer_v2

    registry, oci_repo, auth, manifest = _oci_context(ref)
    blobs_base = f"{registry}/v2/{oci_repo}/blobs"
    out: dict[str, set[str]] = {}
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        for group in group_files(manifest):
            if not group.title.endswith(".safetensors"):
                continue
            if group.is_chunked:
                # Chunked-v2: the shard's bytes span pack blobs mapped by a pointer blob.
                pointer_url = f"{blobs_base}/{group.pointer_digest}"
                refs = parse_pointer_v2(_fetch_blob(client, pointer_url, auth))
                out[group.title] = _dtypes_from_header(
                    _read_header_chunked(client, blobs_base, auth, refs)
                )
                continue
            blob_url = f"{blobs_base}/{group.digest}"
            out[group.title] = _dtypes_from_header(_read_header(client, blob_url, auth))
    return out


def _hf_dtypes(ref: ModelRef) -> dict[str, set[str]]:
    from huggingface_hub import hf_hub_url, list_repo_files

    token = _hf_token()
    files = list_repo_files(repo_id=ref.repo, revision=ref.digest, token=token)
    auth = {"Authorization": f"Bearer {token}"} if token else {}
    out: dict[str, set[str]] = {}
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        for name in files:
            if not name.endswith(".safetensors"):
                continue
            url = hf_hub_url(repo_id=ref.repo, filename=name, revision=ref.digest)
            out[name] = _dtypes_from_header(_read_header(client, url, auth))
    return out


def safetensors_dtypes(ref: ModelRef) -> dict[str, set[str]]:
    """``{shard_filename: {dtypes}}`` for every *.safetensors layer, reading headers only."""
    if ref.backend == BACKEND_HF:
        return _hf_dtypes(ref)
    return _hippius_dtypes(ref)

"""Pre-download weight-dtype gate.

Reads ONLY each safetensors shard's header from the Hippius OCI registry using
HTTP Range requests (the 8-byte little-endian length prefix + that many header
bytes), so a non-16-bit (quantized or full-precision) repo is rejected without
downloading any weight data. A header is tens of KB; a shard is gigabytes.

All coupling to hippius_hub's OCI internals is localized in this module.
"""
from __future__ import annotations

import json
import os

import httpx

from config_validation.models import ModelRef

_HUB_TOKEN_ENV = "HIPPIUS_HUB_TOKEN"
_TIMEOUT = 60


def _token() -> str | None:
    return os.environ.get(_HUB_TOKEN_ENV)


def _oci_context(ref: ModelRef):
    """(registry, oci_repo, auth_headers, manifest) for ``ref`` — the bearer-token
    handshake + manifest fetch, kept in one place so the hippius_hub internals stay here."""
    from hippius_hub._oci import fetch_manifest
    from hippius_hub.auth import get_oci_bearer_token
    from hippius_hub.constants import resolve_registry
    from hippius_hub.file_download import _oci_repo_path

    registry = resolve_registry(None)
    oci_repo = _oci_repo_path(ref.repo, None)
    oci_token = get_oci_bearer_token(oci_repo, _token(), endpoint=None)
    manifest = fetch_manifest(registry, oci_repo, ref.digest, oci_token).manifest
    return registry, oci_repo, {"Authorization": f"Bearer {oci_token}"}, manifest


def _ranged(client: httpx.Client, url: str, headers: dict, start: int, end: int) -> bytes:
    """GET bytes [start, end] of ``url``, insisting the server honor the Range (206).

    Streaming + a 206 check guards against a registry that ignores Range and would
    otherwise stream the whole multi-GB blob into memory."""
    rng = {**headers, "Range": f"bytes={start}-{end}"}
    with client.stream("GET", url, headers=rng) as resp:
        if resp.status_code != 206:
            raise RuntimeError(f"registry did not honor Range (status {resp.status_code}) for {url}")
        return resp.read()


def _read_header(client: httpx.Client, url: str, headers: dict) -> dict:
    """Read a safetensors header: 8-byte little-endian length, then that many JSON bytes."""
    hlen = int.from_bytes(_ranged(client, url, headers, 0, 7), "little")
    return json.loads(_ranged(client, url, headers, 8, 8 + hlen - 1))


def safetensors_dtypes(ref: ModelRef) -> dict[str, set[str]]:
    """``{shard_filename: {dtypes}}`` for every *.safetensors layer, reading headers only."""
    from hippius_hub._oci import iter_titled_layers

    registry, oci_repo, auth, manifest = _oci_context(ref)
    out: dict[str, set[str]] = {}
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        for title, layer in iter_titled_layers(manifest):
            if not title.endswith(".safetensors"):
                continue
            blob_url = f"{registry}/v2/{oci_repo}/blobs/{layer['digest']}"
            header = _read_header(client, blob_url, auth)
            out[title] = {info["dtype"] for k, info in header.items() if k != "__metadata__"}
    return out

"""albedo.models.download — fetch model snapshots from Hippius Hub."""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from albedo.models.ref import ModelRef
from albedo.models.template import ensure_chat_template, scrub_tokenizer_config

log = logging.getLogger(__name__)

_CACHE_ROOT = os.environ.get("ALBEDO_MODEL_CACHE_DIR", "/root/albedo/hippius_models")
_HUB_TOKEN_ENV = "HIPPIUS_HUB_TOKEN"
# Arch-lock validation only needs config.json.
_CONFIG_ONLY_PATTERNS = ["*.json"]


def _cache_dir(ref: ModelRef) -> Path:
    # Digest slug as leaf so multiple digests of the same repo coexist.
    safe_digest = ref.digest.replace(":", "_")
    candidate = Path(_CACHE_ROOT) / ref.repo / safe_digest
    resolved = candidate.resolve()
    cache_root_resolved = Path(_CACHE_ROOT).resolve()
    # Guard against path-traversal: ref.repo allows '/' so a crafted name like
    # "ns/model/../../.." could escape the cache root if not checked here.
    if not str(resolved).startswith(str(cache_root_resolved) + "/") and resolved != cache_root_resolved:
        raise ValueError(
            f"ModelRef.repo {ref.repo!r} resolves outside cache root — path traversal blocked"
        )
    return resolved


def _token() -> str | None:
    return os.environ.get(_HUB_TOKEN_ENV)


def materialize_model(
    ref: ModelRef,
    *,
    local_dir: str | None = None,
    max_workers: int = 8,
    config_only: bool = False,
) -> str:
    """Download a model snapshot and return its local directory path.

    Idempotent: skips download if config.json already exists. After a full
    download, injects the canonical chat template and scrubs tokenizer_config.
    """
    try:
        import hippius_hub  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hippius_hub is not installed; run: pip install hippius-hub"
        ) from exc

    dest = Path(local_dir) if local_dir else _cache_dir(ref)
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "config.json").exists():
        log.debug("materialize_model: cache hit at %s", dest)
        return str(dest)

    log.info("materialize_model: downloading %s → %s", ref.immutable_ref, dest)

    hippius_hub.snapshot_download(
        ref.repo,
        revision=ref.digest,
        local_dir=str(dest),
        max_workers=max_workers,
        allow_patterns=_CONFIG_ONLY_PATTERNS if config_only else None,
        token=_token(),
    )

    if not config_only:
        ensure_chat_template(str(dest))
        scrub_tokenizer_config(str(dest))

    return str(dest)


def list_remote_files(ref: ModelRef) -> list[str]:
    """Return filenames present in the Hippius repo at ref."""
    try:
        import hippius_hub  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hippius_hub is not installed; run: pip install hippius-hub"
        ) from exc
    return hippius_hub.list_repo_files(ref.repo, revision=ref.digest, token=_token())


def prune_model_cache(*keep_refs: ModelRef) -> int:
    """Remove cached model directories not in keep_refs; return bytes freed."""
    keep_paths: set[Path] = {_cache_dir(r) for r in keep_refs}
    cache_root = Path(_CACHE_ROOT)

    try:
        if not cache_root.exists():
            return 0
    except OSError:
        return 0

    freed = 0
    # Structure: CACHE_ROOT/<repo-ns>/<repo-name>/<digest-slug>
    for digest_dir in cache_root.glob("*/*/*"):
        if not digest_dir.is_dir():
            continue
        if digest_dir not in keep_paths:
            size = sum(
                f.stat().st_size for f in digest_dir.rglob("*") if f.is_file()
            )
            shutil.rmtree(digest_dir, ignore_errors=True)
            freed += size
            log.info("prune_model_cache: removed %s (%d bytes)", digest_dir, size)

    return freed

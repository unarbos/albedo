"""Shared local cache layout for downloaded models, namespaced by backend."""
from __future__ import annotations

import os
from pathlib import Path

from config_validation.config import MODEL_CACHE_DIR
from config_validation.models import ModelRef


def _cache_dir(ref: ModelRef) -> Path:
    """Per-(backend, repo, digest) cache dir, guarded against path traversal via crafted repos."""
    safe_digest = ref.digest.replace(":", "_")
    root = Path(MODEL_CACHE_DIR).resolve()
    resolved = (root / ref.backend / ref.repo / safe_digest).resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise ValueError(f"ModelRef.repo {ref.repo!r} resolves outside cache root — blocked")
    return resolved


def cache_dir(ref: ModelRef) -> Path:
    """Public: per-(backend, repo, digest) local cache dir for ``ref`` (no I/O, no download)."""
    return _cache_dir(ref)

from __future__ import annotations

import os
from pathlib import Path

from albedo_config.chain_spec import MODEL_CACHE_DIR
from config_validation.models import ModelRef


def _cache_dir(ref: ModelRef) -> Path:
    safe_digest = ref.digest.replace(":", "_")
    root = Path(MODEL_CACHE_DIR).resolve()
    resolved = (root / ref.backend / ref.repo / safe_digest).resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise ValueError(f"ModelRef.repo {ref.repo!r} resolves outside cache root — blocked")
    return resolved


def cache_dir(ref: ModelRef) -> Path:
    return _cache_dir(ref)

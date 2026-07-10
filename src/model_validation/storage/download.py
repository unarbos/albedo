"""Model access — thin wrapper over the shared config_validation.storage library.

Reuses config_validation's tested download/cache/list logic (HF primary, Hippius option) so
we don't reimplement hub handling. The download cache dir comes from ALBEDO_MODEL_CACHE_DIR:
importing model_validation.config first sets CV_MODEL_CACHE_DIR so downloads land there.
"""
from __future__ import annotations

from pathlib import Path

from config_validation.storage import cache_dir as _cache_dir
from config_validation.storage import download_config as _download_config
from config_validation.storage import download_full as _download_full
from config_validation.storage import list_files as _list_files
from config_validation.models import ModelRef
from model_validation import config as _config  # noqa: F401 — sets CV_MODEL_CACHE_DIR first


def make_ref(repo: str, digest: str) -> ModelRef:
    """Validate + build a ModelRef from a chain_commit's repo/digest."""
    return ModelRef(repo=repo, digest=digest)


def cache_dir(ref: ModelRef) -> Path:
    """Local cache dir for ``ref`` (no I/O, no download)."""
    return _cache_dir(ref)


def list_files(ref: ModelRef) -> list[str]:
    """Filenames present in the repo at the pinned revision."""
    return _list_files(ref)


def download_config(ref: ModelRef) -> str:
    """Download small config/tokenizer files; return the local directory path."""
    return _download_config(ref)


def download_full(ref: ModelRef) -> str:
    """Download the full model snapshot; return the local directory path."""
    return _download_full(ref)

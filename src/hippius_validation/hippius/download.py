"""Hippius model access — thin wrapper over the shared config_validation library.

Reuses config_validation's tested download/cache/list logic so we don't reimplement
Hippius hub handling. The download cache dir comes from ALBEDO_MODEL_CACHE_DIR: importing
hippius_validation.config first sets CV_MODEL_CACHE_DIR so config_validation downloads there.
"""
from __future__ import annotations

from pathlib import Path

from config_validation.hippius import cache_dir as _cache_dir
from config_validation.hippius import download_config as _download_config
from config_validation.hippius import download_full as _download_full
from config_validation.hippius import list_files as _list_files
from config_validation.models import ModelRef
from hippius_validation import config as _config  # noqa: F401 — sets CV_MODEL_CACHE_DIR first


def make_ref(repo: str, digest: str) -> ModelRef:
    """Validate + build a ModelRef from a chain_commit's repo/digest."""
    return ModelRef(repo=repo, digest=digest)


def cache_dir(ref: ModelRef) -> Path:
    """Local cache dir for ``ref`` (no I/O, no download)."""
    return _cache_dir(ref)


def list_files(ref: ModelRef) -> list[str]:
    """Filenames present in the Hippius repo at the pinned digest."""
    return _list_files(ref)


def download_config(ref: ModelRef) -> str:
    """Download small config/tokenizer files; return the local directory path."""
    return _download_config(ref)


def download_full(ref: ModelRef) -> str:
    """Download the full model snapshot; return the local directory path."""
    return _download_full(ref)

from __future__ import annotations

from pathlib import Path

from config_validation.models import ModelRef
from config_validation.storage import cache_dir as _cache_dir
from config_validation.storage import download_config as _download_config
from config_validation.storage import download_full as _download_full
from config_validation.storage import list_files as _list_files


def make_ref(repo: str, digest: str) -> ModelRef:
    return ModelRef(repo=repo, digest=digest)


def cache_dir(ref: ModelRef) -> Path:
    return _cache_dir(ref)


def list_files(ref: ModelRef) -> list[str]:
    return _list_files(ref)


def download_config(ref: ModelRef) -> str:
    return _download_config(ref)


def download_full(ref: ModelRef) -> str:
    return _download_full(ref)

"""HuggingFace hub backend — fetch + inspect model repos pinned to a git revision.

Transfer acceleration is handled by Xet (``HF_XET_HIGH_PERFORMANCE``), enabled in the package
``__init__``; the legacy ``hf_transfer`` is inert on huggingface_hub>=1.0.
"""
from __future__ import annotations

import logging
import os

from config_validation.models import ModelRef
from config_validation.storage._paths import _cache_dir

log = logging.getLogger(__name__)

_TOKEN_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
_CONFIG_ONLY_PATTERNS = ["*.json", "chat_template.jinja"]


def _token() -> str | None:
    for env in _TOKEN_ENVS:
        tok = os.environ.get(env)
        if tok:
            return tok
    return None  # public repos + the on-disk token cache work; never pass ""


def _download(ref: ModelRef, *, config_only: bool, max_workers: int) -> str:
    from huggingface_hub import snapshot_download

    dest = _cache_dir(ref)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("hf: downloading %s (config_only=%s) → %s", ref.immutable_ref, config_only, dest)
    snapshot_download(
        repo_id=ref.repo,
        revision=ref.digest,
        local_dir=str(dest),
        max_workers=max_workers,
        allow_patterns=_CONFIG_ONLY_PATTERNS if config_only else None,
        token=_token(),
    )
    return str(dest)


def download_config(ref: ModelRef) -> str:
    """Download only the JSON config files for ``ref``; return the local dir path."""
    return _download(ref, config_only=True, max_workers=8)


def download_full(ref: ModelRef) -> str:
    """Download the full model snapshot for ``ref``; return the local dir path."""
    return _download(ref, config_only=False, max_workers=8)


def list_files(ref: ModelRef) -> list[str]:
    """List filenames present in the HF repo at the pinned revision."""
    from huggingface_hub import list_repo_files

    return list(list_repo_files(repo_id=ref.repo, revision=ref.digest, token=_token()))


def revision_resolves(ref: ModelRef) -> tuple[bool, str]:
    """Confirm the committed revision is a real revision on HuggingFace. Returns (ok, detail)."""
    try:
        files = list_files(ref)
    except Exception as exc:  # noqa: BLE001 — surface any hub error as a check failure
        log.error(f"revision {ref.digest} did not resolve on HuggingFace repo={ref.repo}: {exc}")
        return False, f"revision {ref.digest} did not resolve on HuggingFace: {exc}"
    if not files:
        return False, f"revision {ref.digest} resolved but the repo is empty"
    return True, f"revision resolved ({len(files)} files)"

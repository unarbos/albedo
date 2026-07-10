"""Route model storage operations to the HF or Hippius backend per ``ModelRef.backend``."""
from __future__ import annotations

from config_validation.models import BACKEND_HF, BACKEND_HIPPIUS, ModelRef
from config_validation.storage import _hf, _hippius
from config_validation.storage._paths import cache_dir

_IMPL = {BACKEND_HF: _hf, BACKEND_HIPPIUS: _hippius}


def _impl(ref: ModelRef):
    try:
        return _IMPL[ref.backend]
    except KeyError:
        raise ValueError(f"unknown model backend {ref.backend!r}") from None


def download_config(ref: ModelRef) -> str:
    """Download only the JSON config files for ``ref``; return the local dir path."""
    return _impl(ref).download_config(ref)


def download_full(ref: ModelRef) -> str:
    """Download the full model snapshot for ``ref``; return the local dir path."""
    return _impl(ref).download_full(ref)


def list_files(ref: ModelRef) -> list[str]:
    """List filenames present in the repo at the pinned revision."""
    return _impl(ref).list_files(ref)


def revision_resolves(ref: ModelRef) -> tuple[bool, str]:
    """Confirm the committed revision resolves on the model's backend. Returns (ok, detail)."""
    return _impl(ref).revision_resolves(ref)


__all__ = ["cache_dir", "download_config", "download_full", "list_files", "revision_resolves"]

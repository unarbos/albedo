"""Immutable model reference (repo + content digest/revision) for HF or Hippius."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Lowercase "<namespace>/<name>" id (HF and Hippius both use this form).
_REPO_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._/-]*$")

# Storage backends.
BACKEND_HF = "hf"
BACKEND_HIPPIUS = "hippius"

# Immutable pin formats.
_HIPPIUS_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")  # Hippius OCI manifest digest
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")  # HF git commit (sha1)
_GIT_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")  # HF git commit (sha256 era)


def detect_backend(digest: str) -> str | None:
    """Infer the storage backend from an immutable pin's format.

    ``sha256:<hex64>`` is a Hippius OCI manifest digest; a bare 40- or 64-hex string is an
    HF git commit revision. Returns None for anything else (e.g. a mutable branch/tag).
    """
    if _HIPPIUS_DIGEST_RE.match(digest):
        return BACKEND_HIPPIUS
    if _GIT_SHA1_RE.match(digest) or _GIT_SHA256_RE.match(digest):
        return BACKEND_HF
    return None


def _default_backend() -> str:
    b = os.environ.get("ALBEDO_MODEL_BACKEND", BACKEND_HF).strip().lower()
    return b if b in (BACKEND_HF, BACKEND_HIPPIUS) else BACKEND_HF


def _allow_mutable() -> bool:
    return os.environ.get("ALBEDO_ALLOW_MUTABLE_REF", "").strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class ModelRef:
    """Immutable pointer to a specific model snapshot (repo + digest/revision).

    ``digest`` is either a Hippius OCI manifest digest (``sha256:<hex64>``) or an HF git
    commit revision (40- or 64-hex). ``backend`` is derived from that format when left blank.
    """

    repo: str
    digest: str
    backend: str = ""

    def __post_init__(self) -> None:
        if not _REPO_RE.match(self.repo):
            raise ValueError(
                f"ModelRef.repo {self.repo!r} is not a valid lowercase '<namespace>/<name>' id"
            )
        detected = detect_backend(self.digest)
        if detected is None:
            if not _allow_mutable():
                raise ValueError(
                    "ModelRef.digest must be an immutable pin — a Hippius 'sha256:<hex64>' "
                    f"or an HF git revision (40/64 hex); got {self.digest!r}"
                )
            detected = _default_backend()
        chosen = self.backend or detected
        if chosen not in (BACKEND_HF, BACKEND_HIPPIUS):
            raise ValueError(f"ModelRef.backend must be 'hf' or 'hippius'; got {self.backend!r}")
        if self.backend and detect_backend(self.digest) not in (None, self.backend):
            raise ValueError(
                f"ModelRef.backend {self.backend!r} contradicts the {self.digest!r} digest format"
            )
        object.__setattr__(self, "backend", chosen)

    @property
    def immutable_ref(self) -> str:
        """Stable string identifier: ``repo@digest``."""
        return f"{self.repo}@{self.digest}"

    def __str__(self) -> str:
        return self.immutable_ref

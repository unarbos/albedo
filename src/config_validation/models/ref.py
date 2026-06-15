"""Immutable model reference (Hippius repo + content digest)."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Lowercase Hippius "<namespace>/<name>" id.
_REPO_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._/-]*$")
# Hippius OCI manifest digest (the subnet is Hippius-only).
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelRef:
    """Immutable pointer to a specific model snapshot (repo + digest)."""

    repo: str
    digest: str

    def __post_init__(self) -> None:
        if not _REPO_RE.match(self.repo):
            raise ValueError(
                f"ModelRef.repo {self.repo!r} is not a valid lowercase '<namespace>/<name>' id"
            )
        if not _DIGEST_RE.match(self.digest):
            raise ValueError(
                f"ModelRef.digest must be a Hippius 'sha256:<hex64>'; got {self.digest!r}"
            )

    @property
    def immutable_ref(self) -> str:
        """Stable string identifier: ``repo@digest``."""
        return f"{self.repo}@{self.digest}"

    def __str__(self) -> str:
        return self.immutable_ref

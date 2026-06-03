"""albedo.models.reveal — v4 reveal string: ``v4|{repo}|{digest}``."""
from __future__ import annotations

from albedo.models.ref import ModelRef

_VERSION = "v4"
_SEP = "|"
_EXPECTED_PARTS = 3


def build_reveal_v4(ref_or_repo: "ModelRef | str", digest: str | None = None) -> str:
    """Build a v4 reveal string ``v4|{repo}|{digest}``.

    Accepts ``(ModelRef,)`` or ``(repo, digest)``.
    The committing hotkey is the on-chain transaction signer — not embedded in the payload.
    """
    if isinstance(ref_or_repo, ModelRef):
        ref = ref_or_repo
    else:
        if digest is None:
            raise TypeError("build_reveal_v4(repo, digest) requires two arguments")
        ref = ModelRef(repo=ref_or_repo, digest=digest)

    return _SEP.join([_VERSION, ref.repo, ref.digest])


def parse_reveal_v4(data: str) -> ModelRef:
    """Parse a v4 reveal string into a ModelRef.

    Raises ValueError if malformed, wrong version, or invalid ModelRef.
    The hotkey is NOT in the payload — use the on-chain committer as the authority.
    """
    parts = data.split(_SEP)
    if len(parts) != _EXPECTED_PARTS:
        raise ValueError(
            f"Expected {_EXPECTED_PARTS} pipe-separated fields, got {len(parts)}: {data!r}"
        )

    version, repo, digest = parts

    if version != _VERSION:
        raise ValueError(
            f"Unsupported reveal version {version!r}; expected {_VERSION!r}"
        )

    return ModelRef(repo=repo, digest=digest)  # validates repo/digest

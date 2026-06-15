"""Decode + parse on-chain reveal commitments (v6 format).

A v6 reveal is a pipe-delimited string committed on-chain:

    v6|<repo>|<sha256:digest>   (the chain hotkey is always the author)
"""
from __future__ import annotations

from typing import Any, NamedTuple

from config_validation.models.ref import ModelRef

_VERSION = "v6"
_SEP = "|"


def decode_raw(raw: Any) -> str:
    """Normalise a raw commitment value to a UTF-8 string.

    Handles raw bytes, ``0x``-prefixed hex strings, and plain strings — the three
    shapes seen across bittensor SDK versions.
    """
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        if raw.startswith("0x"):
            try:
                return bytes.fromhex(raw[2:]).decode("utf-8", errors="replace")
            except ValueError:
                return raw
        return raw
    return str(raw)


class Reveal(NamedTuple):
    ref: ModelRef


def parse_reveal(data: str) -> Reveal:
    """Parse a v6 reveal string into a validated ModelRef.

    Raises ValueError if the payload is not a well-formed 3-part v6 reveal
    (``v6|<repo>|<sha256:digest>``).
    """
    if not data.startswith(_VERSION + _SEP):
        raise ValueError(f"not a {_VERSION} reveal: {data[:16]!r}")

    parts = data.split(_SEP)
    if len(parts) != 3:
        raise ValueError(f"unexpected v6 part count {len(parts)}: {data[:32]!r}")

    _, repo, digest = parts
    ref = ModelRef(repo=repo, digest=digest)  # validates repo + digest
    return Reveal(ref=ref)

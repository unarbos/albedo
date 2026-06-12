"""config_validation.fingerprint — weight fingerprinting + dedup corpus store."""

from config_validation.fingerprint.compute import compute_fingerprint, similarity
from config_validation.fingerprint.store import (
    FingerprintStore,
    JsonlFingerprintStore,
    NullFingerprintStore,
)

__all__ = [
    "compute_fingerprint",
    "similarity",
    "FingerprintStore",
    "JsonlFingerprintStore",
    "NullFingerprintStore",
]

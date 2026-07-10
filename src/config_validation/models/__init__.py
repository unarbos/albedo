"""config_validation.models — model reference + on-chain reveal parsing."""

from config_validation.models.ref import BACKEND_HF, BACKEND_HIPPIUS, ModelRef, detect_backend
from config_validation.models.reveal import decode_raw, parse_reveal

__all__ = [
    "ModelRef",
    "detect_backend",
    "BACKEND_HF",
    "BACKEND_HIPPIUS",
    "decode_raw",
    "parse_reveal",
]

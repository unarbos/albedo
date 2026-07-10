"""Strict file-manifest check for a Hippius model repo.

A repo must contain every required file, at least one *.safetensors, only allowlisted
extras (exact names or globs), no forbidden files, and nothing unexpected. Allowlist comes
from config.py.
"""
from __future__ import annotations

import fnmatch

from model_validation import config


def _matches_any(name: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in globs)


def check(files: list[str]) -> tuple[bool, str]:
    """Return (ok, message). message is empty when ok."""
    present = set(files)

    missing = [f for f in config.REQUIRED_FILES if f not in present]
    if config.REQUIRE_SAFETENSORS and not any(f.endswith(".safetensors") for f in present):
        missing.append("*.safetensors")

    forbidden = sorted(f for f in present if _matches_any(f, config.FORBIDDEN_GLOBS))

    allowed_exact = set(config.REQUIRED_FILES) | set(config.ALLOWED_FILES)
    extras = sorted(
        f for f in present
        if f not in allowed_exact
        and not _matches_any(f, config.ALLOWED_GLOBS)
        and not _matches_any(f, config.FORBIDDEN_GLOBS)
    )

    if missing or forbidden or extras:
        parts = []
        if missing:
            parts.append(f"missing required: {missing}")
        if forbidden:
            parts.append(f"forbidden present: {forbidden[:10]}")
        if extras:
            parts.append(f"unexpected extras: {extras[:10]}")
        return False, "; ".join(parts)
    return True, ""

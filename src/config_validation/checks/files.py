"""Check #2: the repo file set matches the strict allowlist.

A repo must contain every required file, may contain allowlisted files (exact names
or glob patterns), must contain no forbidden files, and any other file is an
unexpected extra. Any deviation is a config mismatch.
"""
from __future__ import annotations

import fnmatch

from config_validation.checks import CheckOutcome
from config_validation.config import (
    ALLOWED_FILES,
    ALLOWED_GLOBS,
    FORBIDDEN_GLOBS,
    REQUIRE_SAFETENSORS,
    REQUIRED_FILES,
)

NAME = "file_manifest"


def _matches_any(name: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in globs)


def check(files: list[str]) -> CheckOutcome:
    """Validate ``files`` (filenames in the repo) against the strict allowlist."""
    present = set(files)

    missing = [f for f in REQUIRED_FILES if f not in present]
    if REQUIRE_SAFETENSORS and not any(f.endswith(".safetensors") for f in present):
        missing.append("*.safetensors")
    forbidden = sorted(f for f in present if _matches_any(f, FORBIDDEN_GLOBS))

    allowed_exact = set(REQUIRED_FILES) | set(ALLOWED_FILES)
    extras = sorted(
        f for f in present
        if f not in allowed_exact and not _matches_any(f, ALLOWED_GLOBS)
        and not _matches_any(f, FORBIDDEN_GLOBS)
    )

    ok = not (missing or forbidden or extras)
    reasons = []
    if missing:
        reasons.append(f"missing required files: {missing}")
    if forbidden:
        reasons.append(f"forbidden files present: {forbidden[:10]}")
    if extras:
        reasons.append(f"unexpected extra files: {extras[:10]}")

    return CheckOutcome(
        name=NAME,
        ok=ok,
        reason="; ".join(reasons),
        details={"missing": missing, "forbidden": forbidden, "extras": extras,
                 "n_files": len(present)},
    )

"""Check #0: the committed repo id matches the required naming pattern."""
from __future__ import annotations

import re

from config_validation.checks import CheckOutcome
from config_validation.config import REPO_PATTERN
from config_validation.models import ModelRef

NAME = "repo_pattern"


def check(ref: ModelRef) -> CheckOutcome:
    """Confirm the committed repo id matches ``REPO_PATTERN`` from chain.toml."""
    if not REPO_PATTERN:
        return CheckOutcome(name=NAME, ok=True, details={"pattern": ""})

    ok = re.match(REPO_PATTERN, ref.repo) is not None
    reason = "" if ok else (
        f"repo {ref.repo!r} does not match required pattern {REPO_PATTERN!r}"
    )
    return CheckOutcome(name=NAME, ok=ok, reason=reason,
                        details={"pattern": REPO_PATTERN, "repo": ref.repo})

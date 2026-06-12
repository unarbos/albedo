"""config_validation.checks — the four independent validation checks."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckOutcome:
    """Result of a single check. ``ok`` False means the model fails this check."""

    name: str
    ok: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


__all__ = ["CheckOutcome"]

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckOutcome:
    name: str
    ok: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


__all__ = ["CheckOutcome"]

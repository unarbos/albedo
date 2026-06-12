"""ValidationResult — the per-model outcome the pipeline emits as a JSONL record."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from config_validation.checks import CheckOutcome


@dataclass
class ValidationResult:
    block: int | None
    hotkey: str
    coldkey: str
    repo: str
    digest: str
    checks: list[CheckOutcome] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """True only if every check that ran passed."""
        return all(c.ok for c in self.checks)

    def to_jsonl_record(self, *, ts: float | None = None) -> dict[str, Any]:
        """Flatten to a public-safe JSONL record (full fingerprint vectors excluded)."""
        fp = next((c.details.get("fingerprint") for c in self.checks
                   if c.name == "duplicate" and c.details.get("fingerprint")), None)
        fingerprint_summary = (
            {"method": fp.get("method"), "n_tensors": len(fp.get("layer_keys", []))}
            if fp else None
        )
        return {
            "ts": ts if ts is not None else time.time(),
            "block": self.block,
            "hotkey": self.hotkey,
            "coldkey": self.coldkey,
            "repo": self.repo,
            "digest": self.digest,
            "valid": self.valid,
            "checks": {
                c.name: {"ok": c.ok, "reason": c.reason,
                         "details": {k: v for k, v in c.details.items() if k != "fingerprint"}}
                for c in self.checks
            },
            "fingerprint_summary": fingerprint_summary,
        }

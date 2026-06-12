"""Check #1: the committed digest is a real, resolvable revision on Hippius."""
from __future__ import annotations

from config_validation.checks import CheckOutcome
from config_validation.hippius import revision_resolves
from config_validation.models import ModelRef

NAME = "revision_parity"


def check(ref: ModelRef) -> CheckOutcome:
    """Confirm the on-chain commit's sha256 digest resolves to a snapshot on Hippius."""
    ok, detail = revision_resolves(ref)
    return CheckOutcome(name=NAME, ok=ok, reason="" if ok else detail,
                        details={"digest": ref.digest})

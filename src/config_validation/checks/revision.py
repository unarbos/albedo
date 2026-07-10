"""Check #1: the committed digest is a real, resolvable revision on the model's backend."""
from __future__ import annotations

from config_validation.checks import CheckOutcome
from config_validation.models import ModelRef
from config_validation.storage import revision_resolves

NAME = "revision_parity"


def check(ref: ModelRef) -> CheckOutcome:
    """Confirm the on-chain commit's digest/revision resolves on the model's backend."""
    ok, detail = revision_resolves(ref)
    return CheckOutcome(name=NAME, ok=ok, reason="" if ok else detail,
                        details={"digest": ref.digest})

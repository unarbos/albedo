from __future__ import annotations

from config_validation.checks import CheckOutcome
from config_validation.models import ModelRef
from config_validation.storage import revision_resolves

NAME = "revision_parity"


def check(ref: ModelRef) -> CheckOutcome:
    ok, detail = revision_resolves(ref)
    return CheckOutcome(
        name=NAME, ok=ok, reason="" if ok else detail, details={"digest": ref.digest}
    )

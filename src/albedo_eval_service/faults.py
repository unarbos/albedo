from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MINER_FAULT = "MINER_FAULT"
INFRA_FAULT = "INFRA_FAULT"
REMOTE_EVAL_FAULT = "REMOTE_EVAL_FAULT"
PROVIDER_FAULT = "PROVIDER_FAULT"
UNKNOWN_FAULT = "UNKNOWN_FAULT"


@dataclass(frozen=True)
class FaultDecision:
    fault_class: str
    fault_code: str
    fault_message: str
    retryable: bool


def classify_failure_verdict(verdict: dict[str, Any]) -> FaultDecision:
    """Normalize a remote failure verdict into the backend state machine.

    The remote host is expected to return structured failure verdicts. Missing
    or malformed fields are infra/retryable by default so miner faults are not
    assigned accidentally.
    """

    fault_class = str(verdict.get("fault_class") or UNKNOWN_FAULT)
    fault_code = str(verdict.get("fault_code") or "unknown_remote_failure")
    fault_message = str(verdict.get("fault_message") or "Remote eval failed")
    retryable = bool(verdict.get("retryable", fault_class != MINER_FAULT))

    if fault_class == MINER_FAULT:
        retryable = False
    elif fault_class in {INFRA_FAULT, REMOTE_EVAL_FAULT, PROVIDER_FAULT, UNKNOWN_FAULT}:
        retryable = True
    else:
        fault_class = UNKNOWN_FAULT
        retryable = True

    return FaultDecision(
        fault_class=fault_class,
        fault_code=fault_code,
        fault_message=fault_message,
        retryable=retryable,
    )


def broken_stream_fault(message: str = "Remote eval stream ended before verdict") -> FaultDecision:
    return FaultDecision(
        fault_class=REMOTE_EVAL_FAULT,
        fault_code="remote_stream_broken",
        fault_message=message,
        retryable=True,
    )

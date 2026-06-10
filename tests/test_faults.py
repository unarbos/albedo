from albedo_eval_service.faults import (
    MINER_FAULT,
    PROVIDER_FAULT,
    REMOTE_EVAL_FAULT,
    UNKNOWN_FAULT,
    broken_stream_fault,
    classify_failure_verdict,
)


def test_miner_faults_are_terminal():
    decision = classify_failure_verdict(
        {
            "fault_class": MINER_FAULT,
            "fault_code": "invalid_checkpoint",
            "fault_message": "checkpoint cannot load",
            "retryable": True,
        }
    )

    assert decision.fault_class == MINER_FAULT
    assert decision.fault_code == "invalid_checkpoint"
    assert decision.retryable is False


def test_provider_faults_remain_retryable():
    decision = classify_failure_verdict(
        {
            "fault_class": PROVIDER_FAULT,
            "fault_code": "judge_provider_exhausted",
            "fault_message": "rate limited",
            "retryable": False,
        }
    )

    assert decision.fault_class == PROVIDER_FAULT
    assert decision.retryable is True


def test_unknown_faults_default_retryable():
    decision = classify_failure_verdict({"fault_class": "BOGUS"})

    assert decision.fault_class == UNKNOWN_FAULT
    assert decision.retryable is True


def test_broken_stream_is_remote_eval_fault():
    decision = broken_stream_fault()

    assert decision.fault_class == REMOTE_EVAL_FAULT
    assert decision.fault_code == "remote_stream_broken"
    assert decision.retryable is True

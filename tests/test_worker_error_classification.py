from __future__ import annotations

import pytest

pytest.importorskip("asyncpg")

from model_validation.validate_worker import _is_not_found, process_model


def test_not_found_markers_cover_missing_private_and_gated():
    assert _is_not_found(Exception("404 Client Error ... Repository Not Found for url ..."))
    assert _is_not_found(
        Exception(
            "401 Client Error ... Repository Not Found ... If you are trying to access a private "
            "or gated repo, make sure you are authenticated"
        )
    )
    assert _is_not_found(Exception("Cannot access gated repo for url ..."))
    assert _is_not_found(Exception("Access to model x is restricted. You must have access"))


def test_bare_auth_failures_are_not_miner_faults():
    assert not _is_not_found(Exception("401 Client Error: Unauthorized. Invalid credentials"))
    assert not _is_not_found(Exception("403 Forbidden: rate limit exceeded"))


def test_malformed_ref_is_terminal_miner_fault():
    outcome = process_model("Alice/Model@" + "a" * 40, "hk-x")
    assert outcome.state == "failed"
    assert outcome.fault_class == "MINER_FAULT"
    assert outcome.fault_code == "invalid_ref"
    assert not outcome.retryable

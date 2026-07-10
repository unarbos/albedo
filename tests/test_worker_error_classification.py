"""Worker error classification: hub failures and malformed on-chain refs."""
from __future__ import annotations

import pytest

pytest.importorskip("asyncpg")

from model_validation.validate_worker import _is_not_found, process_model  # noqa: E402


def test_not_found_markers_cover_missing_private_and_gated():
    # HF raises RepositoryNotFoundError for missing AND private repos — message says "not found".
    assert _is_not_found(Exception("404 Client Error ... Repository Not Found for url ..."))
    assert _is_not_found(Exception(
        "401 Client Error ... Repository Not Found ... If you are trying to access a private "
        "or gated repo, make sure you are authenticated"))
    # Gated repos say gated/restricted.
    assert _is_not_found(Exception("Cannot access gated repo for url ..."))
    assert _is_not_found(Exception("Access to model x is restricted. You must have access"))


def test_bare_auth_failures_are_not_miner_faults():
    # An invalid VALIDATOR token 401s on every repo — must stay retryable infra, never a
    # terminal miner fault.
    assert not _is_not_found(Exception("401 Client Error: Unauthorized. Invalid credentials"))
    assert not _is_not_found(Exception("403 Forbidden: rate limit exceeded"))


def test_malformed_ref_is_terminal_miner_fault():
    # chain_reader's ingest gate only checks "/" in repo, so an uppercase (ModelRef-invalid)
    # repo can reach the worker; it must fault the miner, not retry as infra. No network I/O:
    # make_ref rejects before any hub call.
    outcome = process_model("Alice/Model@" + "a" * 40, "hk-x")
    assert outcome.state == "failed"
    assert outcome.fault_class == "MINER_FAULT"
    assert outcome.fault_code == "invalid_ref"
    assert not outcome.retryable

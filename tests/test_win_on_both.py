from __future__ import annotations

from contextlib import contextmanager
from uuid import uuid4

from albedo_eval_service.control.repository import EvalRepository


class _StubConn:
    def __init__(self, prior_wins: int):
        self.prior_wins = prior_wins
        self.executed: list[tuple[str, tuple]] = []

    @contextmanager
    def transaction(self):
        yield

    def execute(self, sql: str, params: tuple = ()):  # noqa: ANN001
        self.executed.append((" ".join(sql.split()), params))
        stub = self

        class _Result:
            def fetchone(self):
                return {"n": stub.prior_wins}

        return _Result()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _run(verdict: dict, prior_wins: int) -> _StubConn:
    repo = EvalRepository.__new__(EvalRepository)
    conn = _StubConn(prior_wins)
    repo._connect = lambda: conn
    repo._insert_artifacts_from_verdict_inside_tx = lambda *a, **k: None
    repo.record_event_inside_tx = lambda *a, **k: None
    repo.mark_eval_succeeded(
        submission_id=uuid4(), attempt_id=uuid4(), eval_run_id=uuid4(), verdict=verdict
    )
    return conn


def _submission_state(conn: _StubConn) -> str:
    update = next(
        (sql, params)
        for sql, params in conn.executed
        if "UPDATE model_submissions SET state" in sql
    )
    return update[1][0]


def test_first_win_requeues_confirmation_eval():
    conn = _run({"challenger_won": True}, prior_wins=0)
    assert _submission_state(conn) == "EVAL_QUEUED"
    assert any("count(*)" in sql for sql, _ in conn.executed)
    # confirmation jumps the queue: nothing may interleave between pass 1 and pass 2
    assert any("SET priority = 0" in sql for sql, _ in conn.executed)


def test_second_win_does_not_touch_priority():
    conn = _run({"challenger_won": True}, prior_wins=1)
    assert not any("SET priority = 0" in sql for sql, _ in conn.executed)


def test_second_win_promotes_to_eval_win():
    conn = _run({"challenger_won": True}, prior_wins=1)
    assert _submission_state(conn) == "EVAL_WIN"


def test_loss_routes_to_complete_loss_without_counting():
    conn = _run({"challenger_won": False}, prior_wins=0)
    assert _submission_state(conn) == "COMPLETE_LOSS"
    assert not any("count(*)" in sql for sql, _ in conn.executed)

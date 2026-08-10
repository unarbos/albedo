from __future__ import annotations

import asyncio

import pytest
from loguru import logger

from albedo_eval_service.adaptive_concurrency import (
    AdaptiveConcurrencyConfig,
    AdaptiveConcurrencyGate,
)


def _cfg(**overrides) -> AdaptiveConcurrencyConfig:
    base = dict(
        initial=4,
        max_limit=8,
        min_limit=2,
        hold_ratio=0.8,
        ramp_every_successes=3,
        cooldown_successes=2,
        latency_max_seconds=1.0,
        latency_max_ratio=2.0,
        latency_baseline_samples=3,
    )
    base.update(overrides)
    return AdaptiveConcurrencyConfig(**base)


@pytest.fixture
def log_messages():
    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg), format="{message}")
    yield messages
    logger.remove(sink_id)


@pytest.mark.asyncio
async def test_ramp_increases_limit_after_successes():
    gate = AdaptiveConcurrencyGate("m", _cfg(latency_max_seconds=0))
    assert gate.limit == 4
    for _ in range(3):
        await gate.acquire()
        gate.observe_success(0.05)
        await gate.release()
    assert gate.limit == 5


@pytest.mark.asyncio
async def test_429_holds_at_80_percent_of_in_flight():
    gate = AdaptiveConcurrencyGate("m", _cfg(initial=10, max_limit=20, min_limit=2))
    for _ in range(10):
        await gate.acquire()
    gate.observe_429()
    assert gate.held is True
    assert gate.limit == 8  # int(10 * 0.8)
    for _ in range(10):
        await gate.release()


@pytest.mark.asyncio
async def test_latency_absolute_threshold_logged_from_start(log_messages):
    gate = AdaptiveConcurrencyGate(
        "m",
        _cfg(
            latency_max_seconds=0.1,
            latency_baseline_samples=10,
            latency_max_ratio=100.0,
        ),
    )
    await gate.acquire()
    gate.observe_success(0.5)  # over absolute max before baseline ready
    await gate.release()
    assert any("latency out of range" in m for m in log_messages)


@pytest.mark.asyncio
async def test_latency_baseline_then_ratio_threshold(log_messages):
    gate = AdaptiveConcurrencyGate(
        "m",
        _cfg(
            latency_max_seconds=0,  # absolute off
            latency_baseline_samples=3,
            latency_max_ratio=2.0,
            ramp_every_successes=100,
        ),
    )
    for lat in (0.10, 0.12, 0.11):
        await gate.acquire()
        gate.observe_success(lat)
        await gate.release()
    assert gate.baseline_seconds is not None
    threshold = gate.latency_threshold_seconds()
    assert threshold is not None
    assert threshold == pytest.approx(gate.baseline_seconds * 2.0)

    await gate.acquire()
    gate.observe_success(threshold + 0.01)
    await gate.release()
    assert any("latency out of range" in m for m in log_messages)


@pytest.mark.asyncio
async def test_out_of_range_latency_blocks_ramp():
    gate = AdaptiveConcurrencyGate(
        "m",
        _cfg(
            initial=4,
            max_limit=8,
            ramp_every_successes=2,
            latency_max_seconds=0.05,
            latency_baseline_samples=100,
            latency_oor_strikes_before_hold=100,  # log only — no hold yet
        ),
    )
    for _ in range(4):
        await gate.acquire()
        gate.observe_success(1.0)  # out of range → no ramp
        await gate.release()
    assert gate.limit == 4


@pytest.mark.asyncio
async def test_sustained_latency_oor_holds_like_429():
    gate = AdaptiveConcurrencyGate(
        "m",
        _cfg(
            initial=10,
            max_limit=20,
            min_limit=2,
            latency_max_seconds=0.05,
            latency_baseline_samples=100,
            latency_oor_strikes_before_hold=3,
            ramp_every_successes=100,
        ),
    )
    for _ in range(10):
        await gate.acquire()
    # 3 consecutive OOR while saturated → cut to 80% of in_flight
    for _ in range(3):
        gate.observe_success(1.0)
    assert gate.held is True
    assert gate.limit == 8  # int(10 * 0.8)
    for _ in range(10):
        await gate.release()


@pytest.mark.asyncio
async def test_gate_blocks_when_at_limit():
    gate = AdaptiveConcurrencyGate("m", _cfg(initial=2, max_limit=2, latency_max_seconds=0))
    await gate.acquire()
    await gate.acquire()

    blocked = asyncio.Event()

    async def waiter():
        blocked.set()
        await gate.acquire()
        await gate.release()

    task = asyncio.create_task(waiter())
    await blocked.wait()
    await asyncio.sleep(0.05)
    assert not task.done()
    await gate.release()
    await gate.release()
    await asyncio.wait_for(task, timeout=1.0)

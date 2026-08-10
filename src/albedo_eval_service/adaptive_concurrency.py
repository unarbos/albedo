"""Adaptive concurrency gate: ramp until 429/latency OOR, hold at 80% of in-flight.

Used by the shared judge OpenRouter/LiteLLM client. Not a classic asyncio.Semaphore —
limit can shrink/grow while requests are in flight.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass

from loguru import logger

from . import judge_metrics


@dataclass(frozen=True)
class AdaptiveConcurrencyConfig:
    """Knobs for AdaptiveConcurrencyGate."""

    initial: int
    max_limit: int
    min_limit: int = 4
    hold_ratio: float = 0.8
    # After this many in-range successes while not held, raise limit by 1.
    ramp_every_successes: int = 16
    # After a hold, require this many in-range successes before ramping again.
    cooldown_successes: int = 32
    # Absolute latency warn threshold (seconds). 0 disables absolute check.
    latency_max_seconds: float = 60.0
    # Warn / treat out-of-range when latency > baseline * ratio (once baseline set).
    latency_max_ratio: float = 3.0
    # First N successful latencies establish the start-of-run baseline (median).
    latency_baseline_samples: int = 8
    # Consecutive latency-OOR completions before cutting like a 429.
    # Logging alone without backoff was leaving GLM saturated at max (eval regression).
    latency_oor_strikes_before_hold: int = 3


class AdaptiveConcurrencyGate:
    """Per-model in-flight limiter with AIMD-ish 429/latency ramp."""

    def __init__(self, name: str, config: AdaptiveConcurrencyConfig) -> None:
        if config.initial < 1:
            raise ValueError("initial must be >= 1")
        if config.max_limit < config.initial:
            raise ValueError("max_limit must be >= initial")
        if config.min_limit < 1:
            raise ValueError("min_limit must be >= 1")
        self._name = name
        self._cfg = config
        self._limit = config.initial
        self._in_flight = 0
        self._held = False
        self._success_streak = 0
        self._cooldown_remaining = 0
        self._latency_oor_streak = 0
        self._baseline_samples: list[float] = []
        self._baseline_s: float | None = None
        self._cond = asyncio.Condition()
        self._started_at = time.monotonic()
        logger.info(
            "[adaptive-concurrency] start name={} initial={} max={} min={} "
            "hold_ratio={:.2f} latency_max_s={:.1f} latency_max_ratio={:.1f} "
            "latency_oor_strikes={}",
            name,
            config.initial,
            config.max_limit,
            config.min_limit,
            config.hold_ratio,
            config.latency_max_seconds,
            config.latency_max_ratio,
            config.latency_oor_strikes_before_hold,
        )

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def held(self) -> bool:
        return self._held

    @property
    def baseline_seconds(self) -> float | None:
        return self._baseline_s

    def latency_threshold_seconds(self) -> float | None:
        """Effective max latency for out-of-range checks (absolute and/or baseline)."""
        absolute = self._cfg.latency_max_seconds if self._cfg.latency_max_seconds > 0 else None
        relative = None
        if self._baseline_s is not None and self._cfg.latency_max_ratio > 0:
            relative = self._baseline_s * self._cfg.latency_max_ratio
        if absolute is None:
            return relative
        if relative is None:
            return absolute
        return max(absolute, relative)

    async def acquire(self) -> None:
        t0 = time.monotonic()
        async with self._cond:
            while self._in_flight >= self._limit:
                await self._cond.wait()
            self._in_flight += 1
        judge_metrics.observe_acquire_wait(
            model=self._name, wait_s=time.monotonic() - t0
        )

    async def release(self) -> None:
        async with self._cond:
            if self._in_flight <= 0:
                return
            self._in_flight -= 1
            self._cond.notify_all()

    def observe_success(self, latency_s: float) -> None:
        """Record a completed request; may ramp, or hold on sustained latency OOR."""
        out_of_range = self._maybe_update_baseline_and_check(latency_s)
        if out_of_range:
            self._success_streak = 0
            self._latency_oor_streak += 1
            strikes = max(1, self._cfg.latency_oor_strikes_before_hold)
            if self._latency_oor_streak >= strikes:
                self._hold(
                    reason="latency",
                    detail=(
                        f"oor_streak={self._latency_oor_streak} "
                        f"latency_s={latency_s:.3f}"
                    ),
                )
                self._latency_oor_streak = 0
            return

        self._latency_oor_streak = 0
        if self._held:
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                if self._cooldown_remaining == 0:
                    self._held = False
                    logger.info(
                        "[adaptive-concurrency] cooldown done name={} limit={} "
                        "in_flight={} — ramping may resume",
                        self._name,
                        self._limit,
                        self._in_flight,
                    )
            return
        self._success_streak += 1
        if (
            self._success_streak >= self._cfg.ramp_every_successes
            and self._limit < self._cfg.max_limit
        ):
            self._success_streak = 0
            self._limit += 1
            judge_metrics.observe_ramp(model=self._name)
            logger.info(
                "[adaptive-concurrency] ramp+1 name={} limit={} in_flight={} baseline_s={}",
                self._name,
                self._limit,
                self._in_flight,
                f"{self._baseline_s:.3f}" if self._baseline_s is not None else "-",
            )
            self._schedule_notify()

    def observe_429(self) -> None:
        """Cut limit to hold_ratio × in-flight (at fail) and hold until cooldown."""
        self._latency_oor_streak = 0
        self._hold(reason="429", detail=f"in_flight_at_fail={max(1, self._in_flight)}")

    def _hold(self, *, reason: str, detail: str) -> None:
        at_fail = max(1, self._in_flight)
        target = max(
            self._cfg.min_limit,
            int(at_fail * self._cfg.hold_ratio),
        )
        # Never raise on a hold; never go above current limit.
        new_limit = min(self._limit, target)
        prev = self._limit
        self._limit = max(self._cfg.min_limit, new_limit)
        self._held = True
        self._success_streak = 0
        self._cooldown_remaining = self._cfg.cooldown_successes
        judge_metrics.observe_hold(model=self._name, reason=reason)
        logger.warning(
            "[adaptive-concurrency] {} hold name={} prev_limit={} new_limit={} "
            "in_flight={} hold_ratio={:.2f} cooldown_successes={} {}",
            reason,
            self._name,
            prev,
            self._limit,
            self._in_flight,
            self._cfg.hold_ratio,
            self._cfg.cooldown_successes,
            detail,
        )
        self._schedule_notify()

    def _schedule_notify(self) -> None:
        """Wake acquire waiters after limit changes (must re-check in_flight >= limit)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._notify_waiters())

    async def _notify_waiters(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    def _maybe_update_baseline_and_check(self, latency_s: float) -> bool:
        """Update start baseline; log + return True when latency is out of range."""
        if self._baseline_s is None:
            self._baseline_samples.append(latency_s)
            need = max(1, self._cfg.latency_baseline_samples)
            if len(self._baseline_samples) >= need:
                self._baseline_s = float(statistics.median(self._baseline_samples))
                elapsed = time.monotonic() - self._started_at
                logger.info(
                    "[adaptive-concurrency] latency baseline set name={} "
                    "baseline_s={:.3f} samples={} elapsed_s={:.1f} "
                    "threshold_s={}",
                    self._name,
                    self._baseline_s,
                    len(self._baseline_samples),
                    elapsed,
                    (
                        f"{self.latency_threshold_seconds():.3f}"
                        if self.latency_threshold_seconds() is not None
                        else "-"
                    ),
                )
            # Absolute threshold still applies before baseline is ready.
            threshold = (
                self._cfg.latency_max_seconds
                if self._cfg.latency_max_seconds > 0
                else None
            )
        else:
            threshold = self.latency_threshold_seconds()

        if threshold is None:
            return False
        if latency_s <= threshold:
            return False

        judge_metrics.observe_latency_oor(model=self._name)
        logger.warning(
            "[adaptive-concurrency] latency out of range name={} "
            "latency_s={:.3f} threshold_s={:.3f} baseline_s={} "
            "limit={} in_flight={} since_start_s={:.1f}",
            self._name,
            latency_s,
            threshold,
            f"{self._baseline_s:.3f}" if self._baseline_s is not None else "-",
            self._limit,
            self._in_flight,
            time.monotonic() - self._started_at,
        )
        return True

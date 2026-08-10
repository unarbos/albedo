"""Prometheus metrics for albedo-judge-api concurrency and dynamic load.

Cost is intentionally omitted — use GET /eval-cost/{eval_run_id} for spend.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from prometheus_client import Counter, Gauge, Histogram, REGISTRY
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

# Callback set at app startup → returns iterable of AdaptiveConcurrencyGate.
_gates_provider: Callable[[], Iterable[Any]] | None = None
_collector_registered = False

REQUESTS_TOTAL = Counter(
    "albedo_judge_requests_total",
    "Judge LiteLLM/OpenRouter completion attempts (after transport retries settle).",
    ["model", "purpose", "result"],
)

REQUEST_DURATION_SECONDS = Histogram(
    "albedo_judge_request_duration_seconds",
    "Judge completion wall time (acquire held; includes upstream latency).",
    ["model", "purpose"],
    buckets=(
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
        60.0,
        90.0,
        120.0,
        180.0,
        240.0,
        300.0,
        360.0,
        480.0,
        600.0,
    ),
)

GATE_ACQUIRE_WAIT_SECONDS = Histogram(
    "albedo_judge_gate_acquire_wait_seconds",
    "Time spent waiting on the per-model concurrency gate before in-flight slot.",
    ["model"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

ADAPTIVE_HOLDS_TOTAL = Counter(
    "albedo_judge_adaptive_holds_total",
    "Adaptive concurrency holds (429 or sustained latency out-of-range).",
    ["model", "reason"],
)

ADAPTIVE_RAMPS_TOTAL = Counter(
    "albedo_judge_adaptive_ramps_total",
    "Adaptive concurrency +1 ramp events.",
    ["model"],
)

ADAPTIVE_LATENCY_OOR_TOTAL = Counter(
    "albedo_judge_adaptive_latency_oor_total",
    "Completions flagged latency out-of-range (before hold strike threshold).",
    ["model"],
)

SCORE_BATCH_ACTIVE = Gauge(
    "albedo_judge_score_batch_active",
    "Number of /score-batch handlers currently running.",
)

SCORE_SAMPLE_IN_FLIGHT = Gauge(
    "albedo_judge_score_sample_in_flight",
    "Samples currently inside the score-batch sample semaphore.",
)

SCORE_SAMPLE_LIMIT = Gauge(
    "albedo_judge_score_sample_limit",
    "Configured max score-sample concurrency (last observed batch).",
)


class AdaptiveGateCollector(Collector):
    """Snapshot per-model adaptive gate state on each scrape."""

    def collect(self):  # type: ignore[no-untyped-def]
        in_flight = GaugeMetricFamily(
            "albedo_judge_adaptive_in_flight",
            "Current in-flight LiteLLM requests under the adaptive gate.",
            labels=["model"],
        )
        limit = GaugeMetricFamily(
            "albedo_judge_adaptive_limit",
            "Current adaptive concurrency limit.",
            labels=["model"],
        )
        max_limit = GaugeMetricFamily(
            "albedo_judge_adaptive_max_limit",
            "Configured adaptive concurrency ceiling.",
            labels=["model"],
        )
        min_limit = GaugeMetricFamily(
            "albedo_judge_adaptive_min_limit",
            "Configured adaptive concurrency floor.",
            labels=["model"],
        )
        held = GaugeMetricFamily(
            "albedo_judge_adaptive_held",
            "1 if adaptive gate is in hold/cooldown, else 0.",
            labels=["model"],
        )
        cooldown = GaugeMetricFamily(
            "albedo_judge_adaptive_cooldown_remaining",
            "In-range successes still required before ramping resumes.",
            labels=["model"],
        )
        baseline = GaugeMetricFamily(
            "albedo_judge_adaptive_latency_baseline_seconds",
            "Start-of-run median latency baseline (0 if not set yet).",
            labels=["model"],
        )
        threshold = GaugeMetricFamily(
            "albedo_judge_adaptive_latency_threshold_seconds",
            "Effective latency out-of-range threshold (0 if unset).",
            labels=["model"],
        )
        utilization = GaugeMetricFamily(
            "albedo_judge_adaptive_utilization_ratio",
            "in_flight / max(limit, 1).",
            labels=["model"],
        )

        provider = _gates_provider
        gates = list(provider()) if provider is not None else []
        for gate in gates:
            model = str(getattr(gate, "_name", "unknown"))
            cur_in = int(getattr(gate, "in_flight", 0))
            cur_limit = int(getattr(gate, "limit", 0))
            cfg = getattr(gate, "_cfg", None)
            cur_max = int(getattr(cfg, "max_limit", cur_limit) or cur_limit)
            cur_min = int(getattr(cfg, "min_limit", 1) or 1)
            is_held = 1.0 if bool(getattr(gate, "held", False)) else 0.0
            cd = float(getattr(gate, "_cooldown_remaining", 0) or 0)
            base = getattr(gate, "baseline_seconds", None)
            thr = None
            thr_fn = getattr(gate, "latency_threshold_seconds", None)
            if callable(thr_fn):
                thr = thr_fn()
            in_flight.add_metric([model], float(cur_in))
            limit.add_metric([model], float(cur_limit))
            max_limit.add_metric([model], float(cur_max))
            min_limit.add_metric([model], float(cur_min))
            held.add_metric([model], is_held)
            cooldown.add_metric([model], cd)
            baseline.add_metric([model], float(base) if base is not None else 0.0)
            threshold.add_metric([model], float(thr) if thr is not None else 0.0)
            utilization.add_metric(
                [model], float(cur_in) / float(cur_limit if cur_limit > 0 else 1)
            )

        yield in_flight
        yield limit
        yield max_limit
        yield min_limit
        yield held
        yield cooldown
        yield baseline
        yield threshold
        yield utilization


def set_gates_provider(provider: Callable[[], Iterable[Any]] | None) -> None:
    """Register how scrapes discover live AdaptiveConcurrencyGate instances."""
    global _gates_provider, _collector_registered
    _gates_provider = provider
    if provider is not None and not _collector_registered:
        REGISTRY.register(AdaptiveGateCollector())
        _collector_registered = True


def observe_request(
    *,
    model: str,
    purpose: str,
    result: str,
    duration_s: float,
) -> None:
    REQUESTS_TOTAL.labels(model=model, purpose=purpose or "other", result=result).inc()
    REQUEST_DURATION_SECONDS.labels(
        model=model, purpose=purpose or "other"
    ).observe(max(0.0, duration_s))


def observe_acquire_wait(*, model: str, wait_s: float) -> None:
    GATE_ACQUIRE_WAIT_SECONDS.labels(model=model).observe(max(0.0, wait_s))


def observe_hold(*, model: str, reason: str) -> None:
    ADAPTIVE_HOLDS_TOTAL.labels(model=model, reason=reason or "unknown").inc()


def observe_ramp(*, model: str) -> None:
    ADAPTIVE_RAMPS_TOTAL.labels(model=model).inc()


def observe_latency_oor(*, model: str) -> None:
    ADAPTIVE_LATENCY_OOR_TOTAL.labels(model=model).inc()


def score_batch_enter(*, sample_limit: int) -> None:
    SCORE_BATCH_ACTIVE.inc()
    SCORE_SAMPLE_LIMIT.set(float(max(1, sample_limit)))


def score_batch_exit() -> None:
    SCORE_BATCH_ACTIVE.dec()


def score_sample_enter() -> None:
    SCORE_SAMPLE_IN_FLIGHT.inc()


def score_sample_exit() -> None:
    SCORE_SAMPLE_IN_FLIGHT.dec()

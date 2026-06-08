"""Deterministic sampling, turn scoring, and SSE duel streaming."""

from albedo.duel.sampler import Sample, TrajectoryDataset
from albedo.duel.stream import run_duel

__all__ = [
    "TrajectoryDataset",
    "Sample",
    "run_duel",
]

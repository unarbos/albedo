from __future__ import annotations

import argparse

from .config import get_settings
from .repository import EvalRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Requeue retryable Albedo eval submissions.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum retryable eval submissions to requeue.")
    args = parser.parse_args()

    settings = get_settings()
    repository = EvalRepository(settings.database_url)
    requeued = repository.requeue_retryable_evals(worker_id=settings.worker_id, limit=args.limit)
    print(f"requeued_eval_submissions={requeued}")

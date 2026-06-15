from __future__ import annotations

import argparse

from .config import get_settings
from .repository import EvalRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Requeue retryable Albedo eval submissions.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum submissions to advance per run.")
    args = parser.parse_args()

    settings = get_settings()
    repository = EvalRepository(settings.database_url)
    queued = repository.queue_pre_eval_passed(worker_id=settings.worker_id, limit=args.limit)
    requeued = repository.requeue_retryable_evals(worker_id=settings.worker_id, limit=args.limit)
    print(f"queued_from_pre_eval={queued} requeued_eval_submissions={requeued}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from loguru import logger

from .config import get_settings
from .repository import EvalRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Requeue retryable Albedo eval submissions.")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum submissions to advance per run.",
    )
    parser.add_argument(
        "--max-retry-count",
        type=int,
        default=None,
        help=(
            "Do not requeue eval submissions once model_submissions.retry_count reaches this value."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    max_retry_count = (
        settings.max_retry_count if args.max_retry_count is None else args.max_retry_count
    )
    repository = EvalRepository(settings.database_url)
    queued = repository.queue_pre_eval_passed(worker_id=settings.worker_id, limit=args.limit)
    requeued = repository.requeue_retryable_evals(
        worker_id=settings.worker_id, limit=args.limit, max_retry_count=max_retry_count
    )
    logger.info(
        f"[eval-requeue] queued_from_pre_eval={queued} requeued_eval_submissions={requeued}"
    )
    print(f"queued_from_pre_eval={queued} requeued_eval_submissions={requeued}")


if __name__ == "__main__":
    main()

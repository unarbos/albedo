"""albedo.eval_server.logging_setup — central logging config for the eval server.

Configures the root logger explicitly so every ``albedo.*`` log is captured with
a consistent timestamp + level, instead of depending on uvicorn's logging setup.
Output goes to stdout, which PM2 captures into logs/eval.log.
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def setup_logging() -> None:
    """Idempotently configure root logging (timestamped, leveled, -> stdout)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = os.environ.get("ALBEDO_LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(level)

    # Add our handler once, even across uvicorn reloads / re-imports.
    if not any(getattr(h, "_albedo", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handler._albedo = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    # Quiet chatty third-party loggers; their per-request noise drowns eval events.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True

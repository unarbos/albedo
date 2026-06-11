"""Entrypoint: python -m sanity_service"""
from __future__ import annotations

import uvicorn

from sanity_service.config import PORT


def main() -> None:
    uvicorn.run("sanity_service.api:app", host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()

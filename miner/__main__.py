"""Entrypoint: python -m miner (mirrors the `albedo` console script)."""
from miner.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

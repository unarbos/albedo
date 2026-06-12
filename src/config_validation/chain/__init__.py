"""config_validation.chain — Bittensor on-chain commit scanning."""

from config_validation.chain.scanner import CommitRecord, connect, scan_commits

__all__ = ["CommitRecord", "connect", "scan_commits"]

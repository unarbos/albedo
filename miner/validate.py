"""Validate a model/repo with the validator's OWN check functions — no deduplication.

Reuses hippius_validation.validate (the same code the validator runs): the strict file
manifest + the universal spec-driven architecture check. No OpenSearch/Postgres involved.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from config_validation.hippius import download_config, list_files
from config_validation.models import ModelRef

from hippius_validation.validate import check_architecture, check_repo


def _result(files_ok, files_msg, arch_ok, arch_msg) -> tuple[bool, dict]:
    return (files_ok and arch_ok), {
        "file_manifest": {"ok": files_ok, "reason": files_msg},
        "architecture": {"ok": arch_ok, "reason": arch_msg},
    }


def validate_local(path: str) -> tuple[bool, dict]:
    """Validate a local model directory before upload."""
    logger.info(f"validating local model: {path}")
    files = [p.name for p in Path(path).iterdir() if p.is_file()]
    logger.info("checking file manifest…")
    files_ok, files_msg = check_repo(files)
    logger.info("checking architecture…")
    arch_ok, arch_msg = check_architecture(path)
    return _result(files_ok, files_msg, arch_ok, arch_msg)


def validate_remote(repo: str, digest: str) -> tuple[bool, dict]:
    """Validate an uploaded Hippius repo (lists files + fetches config.json only)."""
    from huggingface_hub.errors import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    ref = ModelRef(repo=repo, digest=digest)
    logger.info(f"validating remote repo: {repo}@{digest}")
    try:
        logger.info("listing repo files on Hippius…")
        files = list_files(ref)
        logger.info("downloading config.json…")
        cfg_dir = download_config(ref)
    except RevisionNotFoundError:
        return _result(False, f"digest not found on Hippius: {digest} is not in {repo}", False, "skipped")
    except RepositoryNotFoundError:
        return _result(False, f"repo not found on Hippius: {repo}", False, "skipped")
    except EntryNotFoundError:
        return _result(False, f"config.json missing from {repo}@{digest}", False, "skipped")
    except Exception as exc:  # noqa: BLE001 — surface a readable reason, not a traceback
        return _result(False, f"could not read {repo}@{digest}: {exc}", False, "skipped")

    logger.info("checking file manifest…")
    files_ok, files_msg = check_repo(files)
    logger.info("checking architecture…")
    arch_ok, arch_msg = check_architecture(cfg_dir)
    return _result(files_ok, files_msg, arch_ok, arch_msg)

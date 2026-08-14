from __future__ import annotations

from pathlib import Path

from loguru import logger

from config_validation.models import ModelRef
from config_validation.storage import download_config, list_files
from model_validation.validate import (
    check_architecture,
    check_dtype,
    check_index,
    check_repo,
)


def _result(checks: dict[str, tuple[bool, str]]) -> tuple[bool, dict]:
    ok = all(passed for passed, _ in checks.values())
    return ok, {name: {"ok": passed, "reason": msg} for name, (passed, msg) in checks.items()}


def validate_local(path: str) -> tuple[bool, dict]:
    logger.info(f"validating local model: {path}")
    files = [p.name for p in Path(path).iterdir() if p.is_file()]
    logger.info("checking file manifest…")
    files_res = check_repo(files)
    logger.info("checking architecture…")
    arch_res = check_architecture(path)
    logger.info("checking safetensors index…")
    index_res = check_index(path, files)
    logger.info("checking weight dtype…")
    dtype_res = check_dtype(path)
    return _result(
        {
            "file_manifest": files_res,
            "architecture": arch_res,
            "safetensors_index": index_res,
            "weight_dtype": dtype_res,
        }
    )


def validate_remote(repo: str, digest: str) -> tuple[bool, dict]:
    from huggingface_hub.errors import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    ref = ModelRef(repo=repo, digest=digest)
    logger.info(f"validating remote repo: {repo}@{digest} (backend={ref.backend})")
    try:
        logger.info("listing repo files…")
        files = list_files(ref)
        logger.info("downloading config.json…")
        cfg_dir = download_config(ref)
    except RevisionNotFoundError:
        return _result(
            {
                "file_manifest": (False, f"digest not found: {digest} is not in {repo}"),
                "architecture": (False, "skipped"),
            }
        )
    except RepositoryNotFoundError:
        return _result(
            {
                "file_manifest": (False, f"repo not found: {repo}"),
                "architecture": (False, "skipped"),
            }
        )
    except EntryNotFoundError:
        return _result(
            {
                "file_manifest": (False, f"config.json missing from {repo}@{digest}"),
                "architecture": (False, "skipped"),
            }
        )
    except Exception as exc:
        return _result(
            {
                "file_manifest": (False, f"could not read {repo}@{digest}: {exc}"),
                "architecture": (False, "skipped"),
            }
        )

    logger.info("checking file manifest…")
    files_res = check_repo(files)
    logger.info("checking architecture…")
    arch_res = check_architecture(cfg_dir)
    return _result({"file_manifest": files_res, "architecture": arch_res})

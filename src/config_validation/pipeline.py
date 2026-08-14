from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from loguru import logger

from albedo_config.chain_spec import SEED_DIGEST, SEED_REPO
from config_validation import repo_pattern
from config_validation.chain import CommitRecord
from config_validation.checks import (
    CheckOutcome,
    architecture,
    duplicate,
    files,
    genesis_metadata,
    revision,
)
from config_validation.fingerprint.store import FingerprintStore, NullFingerprintStore
from config_validation.models import ModelRef
from config_validation.result import ValidationResult
from config_validation.storage import download_config, download_full, list_files


def _load_config_json(local_dir: str) -> dict[str, Any]:
    path = Path(local_dir) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"config.json not found in {local_dir}")
    return json.loads(path.read_text())


@functools.lru_cache(maxsize=1)
def load_seed_config() -> dict[str, Any]:
    if not SEED_REPO or not SEED_DIGEST:
        raise RuntimeError("chain.toml [chain].seed_repo / [seed].seed_digest must be set")
    seed_dir = download_config(ModelRef(repo=SEED_REPO, digest=SEED_DIGEST))
    return _load_config_json(seed_dir)


def validate_commit(
    record: CommitRecord,
    *,
    store: FingerprintStore | None = None,
    seed_cfg: dict[str, Any] | None = None,
    record_fingerprint: bool = False,
) -> ValidationResult:
    store = store or NullFingerprintStore()
    result = ValidationResult(
        block=record.block,
        hotkey=record.hotkey,
        coldkey=record.coldkey,
        repo=record.repo,
        digest=record.digest,
    )

    ref = ModelRef(repo=record.repo, digest=record.digest)

    pat = repo_pattern.check(ref)
    result.checks.append(pat)
    if not pat.ok:
        return result

    rev = revision.check(ref)
    result.checks.append(rev)
    if not rev.ok:
        return result

    repo_files: list[str] = []
    try:
        repo_files = list_files(ref)
        result.checks.append(files.check(repo_files))
    except Exception as exc:
        logger.exception(
            f"[config-val] could not list repo files repo={record.repo} digest={record.digest}: {exc}"  # noqa: E501
        )
        result.checks.append(CheckOutcome(files.NAME, False, f"could not list repo files: {exc}"))

    config_dir: str | None = None

    try:
        config_dir = download_config(ref)
        result.checks.append(genesis_metadata.check(config_dir, repo_files))
    except Exception as exc:
        logger.exception(
            f"[config-val] could not check metadata repo={record.repo} digest={record.digest}: {exc}"  # noqa: E501
        )
        result.checks.append(
            CheckOutcome(genesis_metadata.NAME, False, f"could not check metadata: {exc}")
        )

    try:
        if config_dir is None:
            config_dir = download_config(ref)
        cand_cfg = _load_config_json(config_dir)
        seed = seed_cfg if seed_cfg is not None else load_seed_config()
        result.checks.append(architecture.check(cand_cfg, seed))
    except Exception as exc:
        logger.exception(
            f"[config-val] could not load config.json repo={record.repo} digest={record.digest}: {exc}"  # noqa: E501
        )
        result.checks.append(
            CheckOutcome(architecture.NAME, False, f"could not load config.json: {exc}")
        )

    if all(c.ok for c in result.checks):
        try:
            model_dir = download_full(ref)
            dup = duplicate.check(model_dir, store, hotkey=record.hotkey)
            result.checks.append(dup)
            if record_fingerprint and dup.ok and dup.details.get("fingerprint"):
                store.add(
                    ref.immutable_ref,
                    dup.details["fingerprint"],
                    hotkey=record.hotkey,
                    repo=record.repo,
                    digest=record.digest,
                )
        except Exception as exc:
            logger.exception(
                f"[config-val] could not fingerprint model repo={record.repo} digest={record.digest}: {exc}"  # noqa: E501
            )
            result.checks.append(
                CheckOutcome(duplicate.NAME, False, f"could not fingerprint model: {exc}")
            )

    return result

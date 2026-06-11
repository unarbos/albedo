"""Upload a local model directory to the Hippius hub (miner side)."""
from __future__ import annotations

import os
import re

from loguru import logger

from config_validation.config import REPO_PATTERN
from config_validation.models import ModelRef

_PREFIX = os.environ.get("ALBEDO_REPO_PREFIX", "albedo-qwen3-4b")


def make_repo(namespace: str, name: str) -> str:
    """Build ``{namespace}/{prefix}-{name}`` from the miner's suffix.

    The miner supplies only the suffix; we strip an accidental leading prefix so
    ``--name v1`` and ``--name albedo-qwen3-4b-v1`` both yield ``ns/albedo-qwen3-4b-v1``
    (never a doubled ``albedo-qwen3-4b-albedo-qwen3-4b-…``).
    """
    namespace = namespace.strip().strip("/").lower()
    name = name.strip().lower()
    for p in (f"{_PREFIX}-", _PREFIX):
        if name.startswith(p):
            name = name[len(p):]
    name = name.lstrip("-")
    if not namespace or not name:
        raise ValueError("both --namespace and --name (suffix) are required")
    repo = f"{namespace}/{_PREFIX}-{name}"
    if not re.match(REPO_PATTERN, repo):
        raise ValueError(f"repo {repo!r} does not match required pattern {REPO_PATTERN!r}")
    logger.info(f"repo id: {repo}")
    return repo


def _auth() -> str | None:
    """Return a Hippius token, logging in with username/password if provided."""
    import hippius_hub  # type: ignore[import]

    token = os.environ.get("HIPPIUS_HUB_TOKEN")
    if token:
        logger.info("authenticating to Hippius with HIPPIUS_HUB_TOKEN")
        return token
    user = os.environ.get("HIPPIUS_HUB_USERNAME")
    pw = os.environ.get("HIPPIUS_HUB_PASSWORD")
    if user and pw:
        logger.info(f"logging in to Hippius as {user}")
        hippius_hub.login(username=user, password=pw)
        return None  # token cached by login
    logger.warning("no Hippius credentials found (HIPPIUS_HUB_TOKEN / USERNAME+PASSWORD)")
    return None


def upload_to_hippius(local_dir: str, repo: str, *, revision: str = "main",
                      commit_message: str = "") -> ModelRef:
    """Upload ``local_dir`` to ``repo`` on Hippius; return ModelRef(repo, sha256:digest)."""
    import hippius_hub  # type: ignore[import]

    token = _auth()
    logger.info(f"uploading {local_dir} → {repo}@{revision} …")
    result = hippius_hub.upload_folder(
        repo_id=repo,
        folder_path=local_dir,
        revision=revision,
        commit_message=commit_message or f"upload {repo}",
        token=token,
        ignore_patterns=[".cache/**", "*.metadata"],
    )
    digest = getattr(result, "oid", "") or str(result)
    if not digest.startswith("sha256:"):
        raise ValueError(f"Hippius upload returned unexpected digest: {digest!r}")
    logger.info(f"upload complete: {repo}@{digest}")
    return ModelRef(repo=repo.lower(), digest=digest)

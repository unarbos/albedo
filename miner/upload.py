from __future__ import annotations

import os
import re

from loguru import logger

from albedo_config.chain_spec import REPO_PATTERN
from config_validation.models import BACKEND_HF, BACKEND_HIPPIUS, ModelRef

_PREFIX = os.environ.get("ALBEDO_REPO_PREFIX", "albedo-qwen3.6-35b")


def make_repo(namespace: str, name: str) -> str:
    namespace = namespace.strip().strip("/")
    name = name.strip().lower()
    for p in (f"{_PREFIX}-", _PREFIX):
        if name.startswith(p):
            name = name[len(p) :]
    name = name.lstrip("-")
    if not namespace or not name:
        raise ValueError("both --namespace and --name (suffix) are required")
    repo = f"{namespace}/{_PREFIX}-{name}"
    if not re.match(REPO_PATTERN, repo):
        raise ValueError(f"repo {repo!r} does not match required pattern {REPO_PATTERN!r}")
    logger.info(f"repo id: {repo}")
    return repo


def _auth() -> str | None:
    import hippius_hub

    token = os.environ.get("HIPPIUS_HUB_TOKEN")
    if token:
        logger.info("authenticating to Hippius with HIPPIUS_HUB_TOKEN")
        return token
    user = os.environ.get("HIPPIUS_HUB_USERNAME")
    pw = os.environ.get("HIPPIUS_HUB_PASSWORD")
    if user and pw:
        logger.info(f"logging in to Hippius as {user}")
        hippius_hub.login(username=user, password=pw)
        return None
    logger.warning("no Hippius credentials found (HIPPIUS_HUB_TOKEN / USERNAME+PASSWORD)")
    return None


def upload_to_hippius(
    local_dir: str, repo: str, *, revision: str = "main", commit_message: str = ""
) -> ModelRef:
    import hippius_hub

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
    return ModelRef(repo=repo.lower(), digest=digest, backend=BACKEND_HIPPIUS)


def upload_to_hf(
    local_dir: str,
    repo: str,
    *,
    revision: str = "main",
    commit_message: str = "",
    private: bool = False,
) -> ModelRef:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    logger.info(f"uploading {local_dir} → hf:{repo}@{revision} …")
    info = api.upload_folder(
        repo_id=repo,
        folder_path=local_dir,
        revision=revision,
        commit_message=commit_message or f"upload {repo}",
        ignore_patterns=[".cache/**", "*.metadata"],
    )
    sha = getattr(info, "oid", None) or api.repo_info(repo_id=repo, revision=revision).sha
    if not sha:
        raise ValueError(f"HF upload did not return a commit sha for {repo}@{revision}")
    logger.info(f"upload complete: {repo}@{sha}")
    return ModelRef(repo=repo.lower(), digest=sha, backend=BACKEND_HF)


def upload_model(local_dir: str, repo: str) -> ModelRef:
    backend = os.environ.get("ALBEDO_MODEL_BACKEND", BACKEND_HF).strip().lower()
    if backend == BACKEND_HIPPIUS:
        return upload_to_hippius(local_dir, repo)
    return upload_to_hf(local_dir, repo)

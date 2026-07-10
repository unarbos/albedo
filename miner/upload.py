"""Upload a local model directory to the model hub (HF primary, Hippius option) — miner side."""
from __future__ import annotations

import os
import re

from loguru import logger

from config_validation.config import REPO_PATTERN
from config_validation.models import BACKEND_HF, BACKEND_HIPPIUS, ModelRef

_PREFIX = os.environ.get("ALBEDO_REPO_PREFIX", "albedo-qwen3.6-35b")


def make_repo(namespace: str, name: str) -> str:
    """Build ``{namespace}/{prefix}-{name}`` from the miner's suffix.

    The miner supplies only the suffix; we strip an accidental leading prefix so
    ``--name v1`` and ``--name albedo-qwen3.6-35b-v1`` both yield ``ns/albedo-qwen3.6-35b-v1``
    (never a doubled ``albedo-qwen3.6-35b-albedo-qwen3.6-35b-…``).
    """
    # Preserve namespace case — HF is case-sensitive on create/push (the on-chain canonical id
    # is lowercased later in upload_to_hf). repo_pattern's namespace part allows any case.
    namespace = namespace.strip().strip("/")
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
    return ModelRef(repo=repo.lower(), digest=digest, backend=BACKEND_HIPPIUS)


def upload_to_hf(local_dir: str, repo: str, *, revision: str = "main",
                 commit_message: str = "", private: bool = False) -> ModelRef:
    """Upload ``local_dir`` to ``repo`` on HuggingFace; return ModelRef(repo, <commit-sha>).

    Transfer acceleration is Xet (HF_XET_HIGH_PERFORMANCE), not the legacy hf_transfer.
    The immutable pin is the resulting git commit SHA.
    """
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import HfApi

    # HF namespaces are case-sensitive on create/push, so push to ``repo`` as given. The on-chain
    # canonical id is lowercased for ModelRef (HF resolves the namespace case-insensitively on read).
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
    """Upload to the configured primary backend — HF by default, Hippius via ALBEDO_MODEL_BACKEND."""
    backend = os.environ.get("ALBEDO_MODEL_BACKEND", BACKEND_HF).strip().lower()
    if backend == BACKEND_HIPPIUS:
        return upload_to_hippius(local_dir, repo)
    return upload_to_hf(local_dir, repo)

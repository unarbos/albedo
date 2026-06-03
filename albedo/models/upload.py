"""albedo.models.upload — push a local model directory to Hippius Hub."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from albedo.models.ref import ModelRef

log = logging.getLogger(__name__)

_HUB_TOKEN_ENV      = "HIPPIUS_HUB_TOKEN"
_HUB_USERNAME_ENV   = "HIPPIUS_HUB_USERNAME"
_HUB_PASSWORD_ENV   = "HIPPIUS_HUB_PASSWORD"


def _prepare_auth(hippius_hub) -> str | None:
    """Login with username/password if available, else return token. Returns None if login was used."""
    username = os.environ.get(_HUB_USERNAME_ENV, "").strip()
    password = os.environ.get(_HUB_PASSWORD_ENV, "").strip()
    if username and password:
        hippius_hub.login(username=username, password=password)
        return None  # token stored in ~/.cache/hippius/hub/token, pass None to upload_folder
    return os.environ.get(_HUB_TOKEN_ENV)


def upload_model_folder(
    local_dir: str,
    *,
    repo: str,
    revision: str = "main",
    commit_message: str = "",
) -> ModelRef:
    """Upload local_dir to Hippius Hub and return a pinned ModelRef.

    Auth: prefers HIPPIUS_HUB_USERNAME + HIPPIUS_HUB_PASSWORD (calls hub_login),
    falls back to HIPPIUS_HUB_TOKEN.

    Raises FileNotFoundError if local_dir is missing, ValueError if the Hub
    returns an unexpected digest format.
    """
    try:
        import hippius_hub  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hippius_hub is not installed; run: pip install hippius-hub"
        ) from exc

    src = Path(local_dir)
    if not src.exists():
        raise FileNotFoundError(f"upload_model_folder: {local_dir!r} does not exist")
    if not src.is_dir():
        raise ValueError(f"upload_model_folder: {local_dir!r} is not a directory")

    msg = commit_message or "upload via albedo.models.upload_model_folder"
    log.info("upload_model_folder: %s → hippius:%s@%s", src, repo, revision)

    token = _prepare_auth(hippius_hub)
    result = hippius_hub.upload_folder(
        repo_id=repo,
        folder_path=str(src),
        revision=revision,
        commit_message=msg,
        token=token,
        ignore_patterns=[".cache/**", "*.metadata"],
    )

    # CommitInfo.oid holds the Docker-Content-Digest (sha256:<hex64>).
    digest: str = getattr(result, "oid", "") or str(result)
    if not digest.startswith("sha256:"):
        raise ValueError(
            f"Hub upload returned unexpected digest: {digest!r}. "
            "Expected 'sha256:<hex64>'."
        )

    ref = ModelRef(repo=repo.lower(), digest=digest)
    log.info("upload_model_folder: pinned as %s", ref.immutable_ref)
    return ref

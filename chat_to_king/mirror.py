from __future__ import annotations

import json
import os
import re

import httpx
from config import KingChatSettings

_HF_API = "https://huggingface.co/api/models"
_HF_RAW = "https://huggingface.co"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ORIGINAL_RE = re.compile(r"Original HuggingFace repository:\*\*\s*\[`([^`]+)`\]")
_INDEX = "model.safetensors.index.json"


class MirrorNotReady(RuntimeError):
    pass


def mirror_repo_id(roman: str, settings: KingChatSettings) -> str:
    namespace = settings.hf_namespace.strip().strip("/")
    if not namespace or not settings.hf_repo_prefix:
        raise MirrorNotReady("KING_CHAT_HF_NAMESPACE / KING_CHAT_HF_REPO_PREFIX is empty")
    if not roman:
        raise MirrorNotReady("king has no roman numeral; cannot name its mirror repo")
    return f"{namespace}/{settings.hf_repo_prefix}-{roman}".lower()


def mirror_revision(repo_id: str, original_repo: str) -> str:
    token = _token()
    try:
        info = _get(f"{_HF_API}/{repo_id}", token, as_json=True)
    except Exception as exc:
        raise MirrorNotReady(f"{repo_id} not readable yet: {exc}") from exc

    sha = str(info.get("sha") or "")
    if not _SHA_RE.match(sha):
        raise MirrorNotReady(f"{repo_id} head revision {sha!r} is not a commit sha")
    present = {s.get("rfilename", "") for s in info.get("siblings") or ()}
    missing = _missing_files(repo_id, sha, present, token)
    if missing:
        raise MirrorNotReady(f"{repo_id} incomplete: {', '.join(missing[:8])}")

    mirrored = _original_repo(repo_id, sha, token)
    if mirrored and original_repo and mirrored.lower() != original_repo.lower():
        raise MirrorNotReady(
            f"{repo_id} mirrors {mirrored}, not the current king's {original_repo}"
        )
    return sha


def _missing_files(repo_id: str, sha: str, present: set[str], token: str | None) -> list[str]:
    missing = [name for name in ("config.json", "albedo.md") if name not in present]
    if _INDEX not in present:
        if not any(name.endswith(".safetensors") for name in present):
            missing.append("*.safetensors")
        return missing
    try:
        weight_map = json.loads(_get(f"{_HF_RAW}/{repo_id}/raw/{sha}/{_INDEX}", token))[
            "weight_map"
        ]
    except Exception as exc:
        return missing + [f"{_INDEX} unreadable ({exc})"]
    return missing + sorted({s for s in weight_map.values() if s not in present})


def _original_repo(repo_id: str, sha: str, token: str | None) -> str:
    try:
        match = _ORIGINAL_RE.search(_get(f"{_HF_RAW}/{repo_id}/raw/{sha}/albedo.md", token))
    except Exception:
        return ""
    return match.group(1) if match else ""


def _get(url: str, token: str | None, *, as_json: bool = False):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = httpx.get(url, headers=headers, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    return response.json() if as_json else response.text


def _token() -> str | None:
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        if os.environ.get(env):
            return os.environ[env]
    return None

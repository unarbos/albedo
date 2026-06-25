#!/usr/bin/env python3
"""Mirror SN97 Qwen3.6-35B crowned kings to public Hugging Face repos.

Runs on the eval machine as a PM2 service. On startup it back-fills every crowned
35B king that is not already on Hugging Face (oldest -> newest), then switches to a
monitor loop that uploads each newly-coronated king as it appears.

Model bytes are taken from the eval cache dir when present (never deleted); kings the
eval dir no longer has are downloaded into a delete-safe work dir and removed after
upload. Each repo is named ``albedo-qwen3.6-35b-king-<ROMAN>`` and carries an
``albedo.md`` doc (README.md is left to the miner's own files, if any).
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
from uuid import UUID

import psycopg
from loguru import logger
from psycopg.rows import dict_row

from albedo_eval_service.remote_config import RemoteSettings
from albedo_eval_service.remote_models import ModelArtifactResolver, parse_oci_ref

_ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in scripts/)
# All of these are defaults; every one is overridable via the matching ALBEDO_KING_HF_*
# env var (or CLI flag) in load_settings — see the env names in parentheses.
_DEFAULT_ENV_PATH = _ROOT / ".env"  # ALBEDO_KING_HF_ENV_FILE
_DEFAULT_EVAL_DIR = "/workspace/albedo-models"  # ALBEDO_KING_HF_EVAL_DIR (or ALBEDO_CACHE_DIR)
_DEFAULT_WORK_DIR = "/workspace/king_upload_work_dir"  # ALBEDO_KING_HF_WORK_DIR
_DEFAULT_LOCK_PATH = "/tmp/albedo-king-hf-uploader.lock"  # ALBEDO_KING_HF_LOCK_PATH
_DEFAULT_REPO_PREFIX = "albedo-qwen3.6-35b-king"  # ALBEDO_KING_HF_REPO_PREFIX
_DEFAULT_QWEN_PATTERNS = ("qwen3.6", "qwen3-6", "qwen3_6")  # ALBEDO_KING_HF_QWEN_PATTERNS
_DEFAULT_SIZE_PATTERNS = ("35b", "35-b")  # ALBEDO_KING_HF_SIZE_PATTERNS
# Substrings marking the canonical 35B seed; it anchors numbering but gets no repo.
# (env: ALBEDO_KING_HF_GENESIS_MARKERS)
_DEFAULT_GENESIS_MARKERS = ("qwen3.6-35b-a3b-genesis", "35b-a3b-genesis")

_ROMAN_NUMERALS = (
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
)


class Unreachable(Exception):
    """The model could not be fetched from Hippius (not found or not reachable)."""


@dataclass(frozen=True)
class Settings:
    database_url: str
    hf_namespace: str
    hf_token: str | None
    eval_dir: Path
    work_dir: Path
    lock_path: Path
    repo_prefix: str
    poll_interval_s: float
    qwen_patterns: tuple[str, ...]
    size_patterns: tuple[str, ...]
    genesis_markers: tuple[str, ...]
    force: bool
    verify: bool
    dry_run: bool


@dataclass(frozen=True)
class KingUpload:
    king_version_id: UUID
    king_version: int
    model_hash: str
    model_uri: str
    artifact_uri: str
    architecture: str | None
    parameter_count: int | None
    uid: int | None
    hotkey: str | None
    activated_at: datetime
    reign_reason: str
    roman: str = ""
    # The king this model dethroned in its coronation duel (genesis seed for King I).
    opponent_name: str = ""
    opponent_repo: str = ""
    opponent_url: str | None = None
    opponent_hotkey: str | None = None

    @property
    def source_ref(self) -> str:
        """Ref used to locate/download bytes — the same OCI manifest the evaluator used."""
        uri = self.artifact_uri or self.model_uri
        if uri.startswith(("s3://", "file://")) or "@" in uri or not self.model_hash:
            return uri
        return f"{uri}@{self.model_hash}"

    @property
    def king_name(self) -> str:
        return f"King {self.roman}"

    @property
    def hippius_repo(self) -> str:
        """The miner's original Hippius repo, e.g. ``alice/albedo-qwen3.6-35b-v1``."""
        return model_repo(self.model_uri or self.artifact_uri)

    @property
    def hub_url(self) -> str | None:
        return hub_repo_url(self.model_uri or self.artifact_uri)


# --- numbering & naming -------------------------------------------------------

def to_roman(n: int) -> str:
    if n < 1:
        raise ValueError(f"roman numeral undefined for {n}")
    out: list[str] = []
    for value, symbol in _ROMAN_NUMERALS:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)


def model_repo(uri: str) -> str:
    """Port of website/js/model.js modelRepo: strip scheme://, @digest, and registry host."""
    if not uri:
        return ""
    s = re.sub(r"^[a-z][a-z0-9+.-]*://", "", uri, flags=re.IGNORECASE)
    s = re.sub(r"@[^/]*$", "", s)
    i = s.find("/")
    if i > 0 and "." in s[:i]:
        s = s[i + 1 :]
    return s


def hub_repo_url(uri: str) -> str | None:
    """Port of website/js/model.js hubRepoUrl."""
    repo = model_repo(uri)
    if not repo:
        return None
    parts = repo.split("/")
    if len(parts) < 2:
        return "https://hub.hippius.com/models"
    return f"https://hub.hippius.com/models/{parts[0]}/{'/'.join(parts[1:])}"


def repo_id_for(king: KingUpload, settings: Settings) -> str:
    namespace = settings.hf_namespace.strip().strip("/")
    if not namespace:
        raise RuntimeError("ALBEDO_KING_HF_NAMESPACE must not be empty")
    if not king.roman:
        raise RuntimeError(f"king v{king.king_version} has no roman numeral assigned")
    return f"{namespace}/{settings.repo_prefix}-{king.roman}"


def _matches_qwen35(king: KingUpload, settings: Settings) -> bool:
    text = " ".join(
        part
        for part in (
            king.model_uri,
            king.artifact_uri,
            king.architecture or "",
            str(king.parameter_count or ""),
        )
        if part
    ).lower()
    return any(p in text for p in settings.qwen_patterns) and any(
        p in text for p in settings.size_patterns
    )


def _is_genesis(king: KingUpload, settings: Settings) -> bool:
    if king.reign_reason.upper() == "GENESIS":
        return True
    repo = king.hippius_repo.lower()
    return any(marker in repo for marker in settings.genesis_markers)


# --- config -------------------------------------------------------------------

def _load_dotenv(path: Path = _DEFAULT_ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Drop a trailing inline comment ("VALUE   # note") before using the value.
        value = re.split(r"\s#", value.strip(), maxsplit=1)[0].strip()
        os.environ.setdefault(key.strip(), value.strip('"').strip("'"))


def _db_url_from_parts() -> str:
    user = os.environ.get("ALBEDO_POSTGRES_USER", "")
    password = os.environ.get("ALBEDO_POSTGRES_PASSWORD", "")
    db = os.environ.get("ALBEDO_POSTGRES_DB", "")
    host = os.environ.get("ALBEDO_POSTGRES_HOST", "")
    port = os.environ.get("ALBEDO_POSTGRES_HOST_PORT", "")
    if not all((user, password, db, host, port)):
        return ""
    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    return values or default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_settings(args: argparse.Namespace) -> Settings:
    env_file = os.environ.get("ALBEDO_KING_HF_ENV_FILE")
    _load_dotenv(Path(env_file) if env_file else _DEFAULT_ENV_PATH)
    database_url = (
        args.database_url
        or os.environ.get("ALBEDO_KING_HF_DATABASE_URL")
        or os.environ.get("ALBEDO_EVAL_DATABASE_URL")
        or _db_url_from_parts()
    )
    hf_token = (
        args.hf_token
        or os.environ.get("ALBEDO_KING_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    eval_dir = Path(
        args.eval_dir
        or os.environ.get("ALBEDO_KING_HF_EVAL_DIR")
        or os.environ.get("ALBEDO_CACHE_DIR")
        or _DEFAULT_EVAL_DIR
    )
    work_dir = Path(args.work_dir or os.environ.get("ALBEDO_KING_HF_WORK_DIR", _DEFAULT_WORK_DIR))
    lock_path = Path(os.environ.get("ALBEDO_KING_HF_LOCK_PATH", _DEFAULT_LOCK_PATH))
    return Settings(
        database_url=database_url,
        hf_namespace=args.hf_namespace or os.environ.get("ALBEDO_KING_HF_NAMESPACE", "kigs"),
        hf_token=hf_token,
        eval_dir=eval_dir,
        work_dir=work_dir,
        lock_path=lock_path,
        repo_prefix=args.repo_prefix
        or os.environ.get("ALBEDO_KING_HF_REPO_PREFIX", _DEFAULT_REPO_PREFIX),
        poll_interval_s=float(
            args.poll_interval_s or os.environ.get("ALBEDO_KING_HF_POLL_INTERVAL_S", "30")
        ),
        qwen_patterns=_csv_env("ALBEDO_KING_HF_QWEN_PATTERNS", _DEFAULT_QWEN_PATTERNS),
        size_patterns=_csv_env("ALBEDO_KING_HF_SIZE_PATTERNS", _DEFAULT_SIZE_PATTERNS),
        genesis_markers=_csv_env("ALBEDO_KING_HF_GENESIS_MARKERS", _DEFAULT_GENESIS_MARKERS),
        force=args.force,
        verify=args.verify or _bool_env("ALBEDO_KING_HF_VERIFY", False),
        dry_run=args.dry_run,
    )


# --- locks & DB ---------------------------------------------------------------

def acquire_pid_lock(lock_path: Path):
    """Single-instance PID lock: refuse to start if another uploader is running."""
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another king HF uploader is already running (lock held: {lock_path})")
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def _connect(settings: Settings) -> psycopg.Connection:
    if not settings.database_url:
        raise RuntimeError(
            "no database DSN; set ALBEDO_KING_HF_DATABASE_URL, ALBEDO_EVAL_DATABASE_URL, "
            "or ALBEDO_POSTGRES_*"
        )
    return psycopg.connect(settings.database_url, row_factory=dict_row)


def _claim_advisory_lock(conn: psycopg.Connection) -> bool:
    row = conn.execute(
        "SELECT pg_try_advisory_lock(hashtext('albedo_king_hf_uploader')) AS locked"
    ).fetchone()
    return bool(row and row["locked"])


_KINGS_SQL = """
SELECT kv.id          AS king_version_id,
       kv.version     AS king_version,
       kv.model_hash,
       kv.activated_at,
       r.reason       AS reign_reason,
       ms.model_uri,
       ms.architecture,
       ms.parameter_count,
       ms.uid,
       ms.hotkey,
       a.uri          AS artifact_uri
FROM king_versions kv
JOIN model_submissions ms ON ms.id = kv.submission_id
LEFT JOIN artifacts a ON a.id = kv.artifact_id
LEFT JOIN reigns r ON r.id = kv.entered_reign_id
ORDER BY kv.version ASC
"""


def _king_from_row(row: dict) -> KingUpload:
    model_uri = row["model_uri"] or ""
    return KingUpload(
        king_version_id=row["king_version_id"],
        king_version=int(row["king_version"]),
        model_hash=row["model_hash"] or "",
        model_uri=model_uri,
        artifact_uri=row["artifact_uri"] or model_uri,
        architecture=row["architecture"],
        parameter_count=row["parameter_count"],
        uid=row["uid"],
        hotkey=row["hotkey"],
        activated_at=row["activated_at"],
        reign_reason=row["reign_reason"] or "",
    )


def list_crowned_kings(conn: psycopg.Connection, settings: Settings) -> list[KingUpload]:
    """Crowned 35B kings oldest->newest with stable roman numerals (genesis skipped).

    ``kv.version`` is a global counter spanning earlier architecture lines, so the roman
    numeral is the position within the 35B coronation sequence, not the raw version.
    """
    rows = conn.execute(_KINGS_SQL).fetchall()
    crowned: list[KingUpload] = []
    counter = 0
    prev: KingUpload | None = None  # the king reigning just before the next coronation duel
    for row in rows:
        king = _king_from_row(row)
        if not _matches_qwen35(king, settings):
            continue
        if _is_genesis(king, settings):
            prev = king
            continue
        counter += 1
        if prev is None:
            opp_name, opp_repo, opp_url, opp_hotkey = "the previous king", "", None, None
        elif _is_genesis(prev, settings):
            opp_name = "the genesis seed model"
            opp_repo, opp_url, opp_hotkey = prev.hippius_repo, prev.hub_url, prev.hotkey
        else:
            opp_name = prev.king_name
            opp_repo, opp_url, opp_hotkey = prev.hippius_repo, prev.hub_url, prev.hotkey
        king = dataclasses.replace(
            king,
            roman=to_roman(counter),
            opponent_name=opp_name,
            opponent_repo=opp_repo,
            opponent_url=opp_url,
            opponent_hotkey=opp_hotkey,
        )
        crowned.append(king)
        prev = king
    return crowned


# --- model sourcing -----------------------------------------------------------

def _oci_cache_path(base_dir: Path, king: KingUpload) -> Path | None:
    """Where the resolver caches this king's OCI snapshot under ``base_dir`` (None if not OCI)."""
    parsed = parse_oci_ref(king.source_ref)
    if not parsed:
        return None
    registry, repository, digest = parsed
    return (
        base_dir / "oci" / registry / repository.replace("/", "__") / digest.removeprefix("sha256:")
    )


def eval_dir_path(king: KingUpload, settings: Settings) -> Path | None:
    """Path of the king's model inside the eval cache dir, or None if not present there."""
    path = _oci_cache_path(settings.eval_dir, king)
    if path is None:
        return None
    return path if (path / ".albedo-model-cache.json").exists() else None


def work_dir_path(king: KingUpload, settings: Settings) -> Path | None:
    """Path of an already-downloaded copy left in the delete-safe work dir, or None.

    Lets a pass that was interrupted after download (but before the post-upload delete)
    reuse the bytes it already pulled instead of downloading the snapshot again.
    """
    path = _oci_cache_path(settings.work_dir, king)
    if path is None:
        return None
    return path if (path / ".albedo-model-cache.json").exists() else None


def download_to_work_dir(king: KingUpload, settings: Settings) -> Path:
    """Download the king's snapshot from Hippius into the delete-safe work dir."""
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    resolver = ModelArtifactResolver(
        RemoteSettings(
            model_cache_dir=str(settings.work_dir),
            use_canonical_model_config=False,
            resolve_model_artifacts=True,
            s3_endpoint_url=os.environ.get("ALBEDO_REMOTE_S3_ENDPOINT_URL") or None,
            s3_region=os.environ.get("ALBEDO_REMOTE_S3_REGION") or None,
            s3_access_key_id=os.environ.get("ALBEDO_REMOTE_S3_ACCESS_KEY_ID") or None,
            s3_secret_access_key=os.environ.get("ALBEDO_REMOTE_S3_SECRET_ACCESS_KEY") or None,
            s3_session_token=os.environ.get("ALBEDO_REMOTE_S3_SESSION_TOKEN") or None,
        )
    )
    try:
        resolved = resolver.resolve(king.source_ref)
    except Exception as exc:  # noqa: BLE001 — surface as a skippable "unreachable" condition
        raise Unreachable(str(exc)) from exc
    return Path(resolved.local_path).resolve()


def _delete_work_copy(path: Path, work_dir: Path) -> None:
    """Delete a downloaded copy — but ONLY if it lives under the work dir (never the eval dir)."""
    work = work_dir.resolve()
    resolved = path.resolve()
    if resolved == work or work not in resolved.parents:
        logger.warning("refusing to delete {} — not under work dir {}", resolved, work)
        return
    logger.info("deleting work-dir copy {}", resolved)
    shutil.rmtree(resolved, ignore_errors=True)
    partial = resolved.with_suffix(".partial")
    if partial.exists() and work in partial.resolve().parents:
        shutil.rmtree(partial, ignore_errors=True)


def _missing_layers(manifest: dict, present: set[str], ignore_patterns: list[str]) -> list[tuple[str, str]]:
    """(filename, blob_digest) for manifest layers whose file is absent from ``present``.

    Maps each OCI layer to its repo-relative filename, drops internal/ignored files, and keeps
    only the ones the HF repo is missing — so the caller downloads just those blobs.
    """
    from huggingface_hub.utils import filter_repo_objects

    from albedo_eval_service.remote_models import _DIGEST_RE, _layer_filename

    layers = manifest.get("layers", [])
    names = [_layer_filename(layer, index) for index, layer in enumerate(layers)]
    keep = set(filter_repo_objects(names, ignore_patterns=ignore_patterns))
    out: list[tuple[str, str]] = []
    for index, layer in enumerate(layers):
        name = names[index]
        if name not in keep or name in present:
            continue
        digest = layer.get("digest")
        if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
            raise ValueError(f"OCI layer {index} ({name}) is missing a sha256 digest")
        out.append((name, digest))
    return out


def download_missing_from_source(
    king: KingUpload, settings: Settings, present: set[str]
) -> tuple[Path, list[str]]:
    """Download ONLY the source files absent from the HF repo. Returns (dir, [downloaded rels]).

    Hippius/OCI sources fetch just the missing files' blobs straight from the manifest (no full
    snapshot). Non-OCI sources can't address single files, so they fall back to a full download.
    The returned dir lives under the work dir and is safe to delete after upload.
    """
    parsed = parse_oci_ref(king.source_ref)
    if parsed is None:
        full = download_to_work_dir(king, settings)
        missing = [
            rel for rel in _iter_model_files(full, _UPLOAD_IGNORE_PATTERNS) if rel not in present
        ]
        return full, missing

    import httpx

    from albedo_eval_service.remote_models import _bearer_token, _stream_blob_to_file, _verify_digest

    registry, repository, digest = parsed
    out_dir = (
        settings.work_dir
        / "repair"
        / repository.replace("/", "__")
        / digest.removeprefix("sha256:")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    accept = (
        "application/vnd.oci.image.manifest.v1+json, "
        "application/vnd.docker.distribution.manifest.v2+json"
    )
    try:
        with httpx.Client(timeout=None, follow_redirects=True) as client:
            manifest_url = f"https://{registry}/v2/{repository}/manifests/{digest}"
            response = client.get(manifest_url, headers={"Accept": accept})
            token: str | None = None
            if response.status_code == 401:
                token = _bearer_token(client, response, repository)
                response = client.get(
                    manifest_url, headers={"Accept": accept, "Authorization": f"Bearer {token}"}
                )
            response.raise_for_status()
            _verify_digest(response.content, digest, label="manifest")
            auth = response.request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                token = auth.removeprefix("Bearer ")
            wanted = _missing_layers(response.json(), present, _UPLOAD_IGNORE_PATTERNS)
            logger.info("{} — fetching {} missing blob(s) from Hippius", king.king_name, len(wanted))
            for name, blob_digest in wanted:
                destination = out_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                blob_url = f"https://{registry}/v2/{repository}/blobs/{blob_digest}"
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                retry = _stream_blob_to_file(
                    client, blob_url, headers, destination, blob_digest, label=name
                )
                if retry is not None:
                    token = _bearer_token(client, retry, repository)
                    _stream_blob_to_file(
                        client,
                        blob_url,
                        {"Authorization": f"Bearer {token}"},
                        destination,
                        blob_digest,
                        label=name,
                    )
                downloaded.append(name)
    except Exception as exc:  # noqa: BLE001 — surface as a skippable "unreachable" condition
        raise Unreachable(str(exc)) from exc
    return out_dir, downloaded


# --- docs & upload ------------------------------------------------------------

# Files in the model dir that must never be published: the internal cache marker, partial
# downloads, the HF cache subdir, and our albedo.md (uploaded separately). Shared by the
# folder upload and the verify pass so both agree on which files a repo is expected to hold.
_UPLOAD_IGNORE_PATTERNS = [".albedo-model-cache.json", "*.download", ".cache/**", "albedo.md"]

_ALBEDO_MD_TEMPLATE = """\
# {king_name} — Albedo (Bittensor SN97)

This repository is a public, read-only **mirror** of an Albedo king model.

Albedo is **Bittensor Subnet 97 (SN97)**: an open competition where miners submit
language models that compete to become the reigning "king".

## How it was crowned

This model earned its crown in a **head-to-head battle** against the sitting king. Both
models tackle the same coding tasks, and an **ensemble of LLM judges** scores their
responses *pairwise* across five metrics — correctness, grounding, progress, protocol,
and efficiency. Judging order is **counterbalanced** (each model is shown first on half
the samples) to cancel position bias, and scores aggregate zero-sum between the two
contenders. A challenger is **coronated** only if it beats the incumbent king by a clear
win margin; otherwise the king keeps its throne. This repo archives one such winning king
so the lineage stays public even after the model rotates out of the live serving cache.

## This king

- **Title:** {king_name}
- **Dethroned:** {defeated}
- **Original Hippius repository:** [`{repo}`]({url})
- **Submitted by miner hotkey:** `{hotkey}`

The model files here are mirrored verbatim from the miner's original Hippius upload
linked above. All credit for the model belongs to its original author.

## Links

- Original model on Hippius Hub: {url}
- Browse Albedo models: https://hub.hippius.com/models
- Albedo source code (GitHub): https://github.com/unarbos/albedo

---

*Mirrored automatically by the Albedo king HF uploader. This is an archival copy; the
authoritative source is the Hippius repository linked above.*
"""


def render_albedo_md(king: KingUpload) -> str:
    url = king.hub_url or "https://hub.hippius.com/models"
    return _ALBEDO_MD_TEMPLATE.format(
        king_name=king.king_name,
        defeated=_defeated_line(king),
        repo=king.hippius_repo or "unknown",
        url=url,
        hotkey=king.hotkey or "unknown",
    )


def _defeated_line(king: KingUpload) -> str:
    """Human-readable description of the king this model beat in its coronation duel."""
    name = king.opponent_name or "the previous king"
    if king.opponent_repo and king.opponent_url:
        line = f"{name} — [`{king.opponent_repo}`]({king.opponent_url})"
    elif king.opponent_repo:
        line = f"{name} — `{king.opponent_repo}`"
    else:
        line = name
    if king.opponent_hotkey:
        line += f" (miner `{king.opponent_hotkey}`)"
    return line


def _hf_api(token: str | None):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def already_uploaded(api, repo_id: str) -> bool:
    """True if the repo exists and holds real files (anything beyond a lone ``.gitattributes``).

    HF's ``create_repo`` seeds a brand-new repo with a single ``.gitattributes``; a prior
    pass that created the repo but died before pushing the model leaves exactly that empty
    shell behind. Such a repo must still be uploaded, so we only treat a repo by name —
    if its sole file is ``.gitattributes`` (or it is empty) it counts as not-yet-uploaded.
    """
    if not api.repo_exists(repo_id=repo_id, repo_type="model"):
        return False
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    return any(f != ".gitattributes" for f in files)


def _index_shard_files(repo_id: str, token: str | None, present: set[str]) -> set[str]:
    """Shard filenames the repo's safetensors index references (empty if no/unreadable index)."""
    if "model.safetensors.index.json" not in present:
        return set()
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            repo_type="model",
            token=token,
        )
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {name for name in data.get("weight_map", {}).values() if isinstance(name, str)}
    except Exception as exc:  # noqa: BLE001 — unreadable index, can't check shards from HF side
        logger.warning("could not read safetensors index for {}: {}", repo_id, exc)
        return set()


def _repo_problems(repo_id: str, present: set[str], token: str | None) -> list[str]:
    """HF-side completeness problems for a repo whose file list is already known."""
    problems: list[str] = []
    if "config.json" not in present:
        problems.append("config.json")
    safetensors = [f for f in present if f.endswith(".safetensors")]
    sharded = any("-of-" in f for f in safetensors)
    if "model.safetensors.index.json" in present:
        shards = _index_shard_files(repo_id, token, present)
        problems.extend(sorted(s for s in shards if s not in present))
    elif sharded:
        # Multiple shards but no index map — the model won't load without it.
        problems.append("model.safetensors.index.json")
    elif not safetensors and not any(f.endswith(".bin") for f in present):
        problems.append("*.safetensors (no weight files)")
    if "albedo.md" not in present:
        problems.append("albedo.md")
    return problems


def hf_repo_problems(api, repo_id: str, token: str | None) -> list[str]:
    """Cheap HF-side completeness check (no model download). [] means the repo looks complete.

    Catches the breakage left by the old multi-commit uploads: missing safetensors shards, a
    sharded model with no index, or a repo missing config.json / albedo.md.
    """
    if not api.repo_exists(repo_id=repo_id, repo_type="model"):
        return ["repo does not exist"]
    present = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    return _repo_problems(repo_id, present, token)


def _add_op(path_in_repo: str, data):
    from huggingface_hub import CommitOperationAdd

    return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=data)


def _upload_model(api, king: KingUpload, model_dir: Path, settings: Settings, repo_id: str) -> None:
    # Exactly two commits per repo: create_repo seeds the "initial" commit (.gitattributes),
    # then a single create_commit lands every model file together with albedo.md.
    logger.info("creating public HF repo {} (exist_ok)", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    files = _iter_model_files(model_dir, _UPLOAD_IGNORE_PATTERNS)
    logger.info(
        "uploading {} model files + albedo.md to {} in one commit", len(files), repo_id
    )
    operations = [_add_op(rel, str(Path(model_dir) / rel)) for rel in files]
    # albedo.md is rendered in-memory and committed alongside the miner's files so we never
    # write into the eval dir and never leave a half-mirrored repo without its doc.
    operations.append(_add_op("albedo.md", render_albedo_md(king).encode("utf-8")))
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Mirror Albedo {king.king_name}",
    )
    logger.info("uploaded {} to {} ({} files + albedo.md)", king.king_name, repo_id, len(files))


def _upload_one(api, king: KingUpload, settings: Settings, repo_id: str) -> None:
    hit = eval_dir_path(king, settings)
    if hit is not None:
        logger.info(
            "{} (v{}) — eval-dir hit at {} (no download, not deleted)",
            king.king_name,
            king.king_version,
            hit,
        )
        _upload_model(api, king, hit, settings, repo_id)
        return
    cached = work_dir_path(king, settings)
    if cached is not None:
        logger.info(
            "{} (v{}) — work-dir hit at {} (reusing, no re-download)",
            king.king_name,
            king.king_version,
            cached,
        )
        try:
            _upload_model(api, king, cached, settings, repo_id)
        finally:
            _delete_work_copy(cached, settings.work_dir)
        return
    logger.info(
        "{} (v{}) — not cached; downloading {} into work dir",
        king.king_name,
        king.king_version,
        king.source_ref,
    )
    model_dir = download_to_work_dir(king, settings)
    try:
        _upload_model(api, king, model_dir, settings, repo_id)
    finally:
        _delete_work_copy(model_dir, settings.work_dir)


def _iter_model_files(model_dir: Path, ignore_patterns: list[str]) -> list[str]:
    """Repo-relative paths the uploader would push from ``model_dir`` (same ignore filter)."""
    from huggingface_hub.utils import filter_repo_objects

    base = Path(model_dir)
    rels = [p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()]
    return list(filter_repo_objects(rels, ignore_patterns=ignore_patterns))


def _verify_and_repair(api, king: KingUpload, settings: Settings, repo_id: str) -> bool:
    """Check the HF repo first, then download + commit only the files it is missing.

    The cheap HF-side check runs first so a repo that already looks complete never triggers a
    model download. If it is incomplete, the model bytes are sourced (eval-dir copy, leftover
    work-dir copy, or a fresh download — only when a model file is actually missing) and every
    file the repo lacks is committed in one commit. Returns True if anything was (re)uploaded.
    """
    present = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    problems = _repo_problems(repo_id, present, settings.hf_token)
    if not problems:
        logger.info(
            "{} — repo {} complete (HF check, {} files); no download", king.king_name, repo_id,
            len(present),
        )
        return False
    logger.warning("{} — repo {} incomplete: {}", king.king_name, repo_id, ", ".join(problems[:12]))
    operations = []
    cleanup: Path | None = None
    try:
        # Only source model bytes when something other than albedo.md is missing.
        if any(p != "albedo.md" for p in problems):
            local = eval_dir_path(king, settings) or work_dir_path(king, settings)
            if local is not None:
                # Full copy already on disk — upload only the files the repo lacks, no download.
                logger.info("{} — sourcing missing files from local copy {}", king.king_name, local)
                operations.extend(
                    _add_op(rel, str(local / rel))
                    for rel in _iter_model_files(local, _UPLOAD_IGNORE_PATTERNS)
                    if rel not in present
                )
            else:
                # Not cached anywhere — pull ONLY the missing files from Hippius.
                logger.info(
                    "{} — downloading only the missing files from {}",
                    king.king_name,
                    king.source_ref,
                )
                fetch_dir, fetched = download_missing_from_source(king, settings, present)
                cleanup = fetch_dir
                operations.extend(_add_op(rel, str(fetch_dir / rel)) for rel in fetched)
        if "albedo.md" not in present:
            operations.append(_add_op("albedo.md", render_albedo_md(king).encode("utf-8")))
        if not operations:
            logger.info("{} — repo {} had no missing files to commit", king.king_name, repo_id)
            return False
        shown = ", ".join(op.path_in_repo for op in operations[:10])
        if len(operations) > 10:
            shown += f", … (+{len(operations) - 10} more)"
        logger.warning(
            "{} — committing {} missing file(s) to {}: {}",
            king.king_name,
            len(operations),
            repo_id,
            shown,
        )
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=f"Repair Albedo {king.king_name}: add {len(operations)} missing file(s)",
        )
        return True
    finally:
        if cleanup is not None:
            _delete_work_copy(cleanup, settings.work_dir)


def process_once(api, settings: Settings, *, limit: int | None = None, verify: bool = False) -> dict:
    # Fresh, short-lived read-only connection per pass: a dropped connection self-heals
    # on the next poll instead of wedging the monitor, and the DB isn't held open during
    # the (potentially long) model uploads.
    with _connect(settings) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        kings = list_crowned_kings(conn, settings)
    counts = {"total": len(kings), "uploaded": 0, "repaired": 0, "skipped": 0, "failed": 0}
    done = 0
    for king in kings:
        repo_id = repo_id_for(king, settings)
        try:
            if not settings.force and already_uploaded(api, repo_id):
                if verify and _verify_and_repair(api, king, settings, repo_id):
                    counts["repaired"] += 1
                    done += 1
                else:
                    counts["skipped"] += 1
            else:
                _upload_one(api, king, settings, repo_id)
                counts["uploaded"] += 1
                done += 1
        except Unreachable as exc:
            logger.warning(
                "{} unreachable on Hippius: {} — skipping to next king", king.king_name, exc
            )
            counts["failed"] += 1
        except Exception as exc:  # noqa: BLE001 — one bad king must not abort the pass
            logger.warning(
                "{} failed ({}): {} — skipping to next king",
                king.king_name,
                type(exc).__name__,
                exc,
            )
            counts["failed"] += 1
        if limit is not None and done >= limit:
            break
    return counts


# --- dry-run explain ----------------------------------------------------------

def explain(settings: Settings) -> None:
    api = _hf_api(settings.hf_token)
    with _connect(settings) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        kings = list_crowned_kings(conn, settings)

    print(f"DRY RUN — {len(kings)} crowned Qwen3.6-35B king(s), oldest -> newest")
    print(f"eval dir: {settings.eval_dir}   work dir: {settings.work_dir}")
    print("-" * 72)
    n_skip = n_up = n_hit = n_dl = 0
    for king in kings:
        repo_id = repo_id_for(king, settings)
        try:
            exists = already_uploaded(api, repo_id)
            status = "SKIP (already uploaded)" if exists else "WILL UPLOAD"
        except Exception as exc:  # noqa: BLE001 — read-only probe, report and continue
            exists = False
            status = f"WILL UPLOAD (HF check failed: {type(exc).__name__})"
        hit = eval_dir_path(king, settings)
        cached = work_dir_path(king, settings) if hit is None else None
        if exists:
            n_skip += 1
        else:
            n_up += 1
            if hit is not None or cached is not None:
                n_hit += 1
            else:
                n_dl += 1
        print(f"{king.king_name}  (king v{king.king_version})  ->  {repo_id}")
        print(f"    status : {status}")
        if hit is not None:
            print(f"    source : eval-dir HIT {hit} (upload in place, never deleted)")
        elif cached is not None:
            print(f"    source : work-dir HIT {cached} (reuse, delete work-dir copy after upload)")
        else:
            print(f"    source : MISS -> would download {king.source_ref}")
            print(f"             into {settings.work_dir}, delete work-dir copy after upload")
        print(f"    albedo.md: repo={king.hippius_repo}  link={king.hub_url}  hotkey={king.hotkey}")
    print("-" * 72)
    print(
        f"summary: {len(kings)} kings | {n_skip} already on HF (skip) | "
        f"{n_up} to upload ({n_hit} cached locally, {n_dl} would download)"
    )


# --- entrypoint ---------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror SN97 Qwen3.6-35B crowned kings to public Hugging Face repos: back-fill "
            "missing kings, then monitor the DB for new coronations."
        )
    )
    parser.add_argument("--once", action="store_true", help="run the back-fill pass only then exit")
    parser.add_argument("--dry-run", action="store_true", help="explain mode: print plan only")
    parser.add_argument("--force", action="store_true", help="upload even if the HF repo exists")
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "repair pass: for kings already on HF, check the repo first and download + commit "
            "only the files it is missing (no download when the repo already looks complete)"
        ),
    )
    parser.add_argument("--limit", type=int, help="max uploads for this process")
    parser.add_argument("--database-url", help="Postgres DSN override")
    parser.add_argument("--hf-namespace", help="HF namespace (env ALBEDO_KING_HF_NAMESPACE)")
    parser.add_argument("--hf-token", help="HF token override")
    parser.add_argument("--eval-dir", help="eval cache dir to read from, never deleted")
    parser.add_argument("--work-dir", help="delete-safe download dir for this watcher")
    parser.add_argument("--repo-prefix", help="HF repo name prefix for the king repos")
    parser.add_argument("--poll-interval-s", type=float, help="monitor polling interval")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = load_settings(args)
    lock = acquire_pid_lock(settings.lock_path)  # noqa: F841 — held for process lifetime

    logger.info(
        "king HF uploader starting: namespace={} eval_dir={} work_dir={} poll={}s dry_run={}",
        settings.hf_namespace,
        settings.eval_dir,
        settings.work_dir,
        settings.poll_interval_s,
        settings.dry_run,
    )

    if settings.dry_run:
        explain(settings)
        return

    if not settings.hf_token:
        raise SystemExit("no Hugging Face token; set ALBEDO_KING_HF_TOKEN or HF_TOKEN")

    api = _hf_api(settings.hf_token)
    # Dedicated connection held for the process lifetime so the DB advisory lock stays held;
    # query/upload passes use their own short-lived connections.
    lock_conn = _connect(settings)
    try:
        if not _claim_advisory_lock(lock_conn):
            raise SystemExit("another king HF uploader already holds the DB advisory lock")

        if settings.verify:
            logger.info(
                "back-fill phase (verify on): mirroring missing kings and repairing existing repos"
            )
        else:
            logger.info("back-fill phase: mirroring crowned kings missing from Hugging Face")
        counts = process_once(api, settings, limit=args.limit, verify=settings.verify)
        logger.info(
            "back-fill complete: {} uploaded, {} repaired, {} already complete on HF, {} failed "
            "(will retry) — entering monitor mode",
            counts["uploaded"],
            counts["repaired"],
            counts["skipped"],
            counts["failed"],
        )
        if args.once:
            return

        while True:
            time.sleep(settings.poll_interval_s)
            try:
                counts = process_once(api, settings)
            except Exception as exc:  # noqa: BLE001 — keep the monitor alive across transient errors
                logger.warning("monitor pass error ({}): {}", type(exc).__name__, exc)
                continue
            if counts["uploaded"]:
                logger.info("monitor: uploaded {} new king(s)", counts["uploaded"])
    finally:
        lock_conn.close()


if __name__ == "__main__":
    main()

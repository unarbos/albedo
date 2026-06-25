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
import io
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
    upload_workers: int
    qwen_patterns: tuple[str, ...]
    size_patterns: tuple[str, ...]
    genesis_markers: tuple[str, ...]
    use_large_folder: bool
    force: bool
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
    upload_workers = int(
        args.upload_workers
        or os.environ.get("ALBEDO_KING_HF_UPLOAD_WORKERS")
        or max(4, min(16, os.cpu_count() or 4))
    )
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
        upload_workers=upload_workers,
        qwen_patterns=_csv_env("ALBEDO_KING_HF_QWEN_PATTERNS", _DEFAULT_QWEN_PATTERNS),
        size_patterns=_csv_env("ALBEDO_KING_HF_SIZE_PATTERNS", _DEFAULT_SIZE_PATTERNS),
        genesis_markers=_csv_env("ALBEDO_KING_HF_GENESIS_MARKERS", _DEFAULT_GENESIS_MARKERS),
        use_large_folder=(
            args.use_large_folder
            if args.use_large_folder is not None
            else _bool_env("ALBEDO_KING_HF_USE_LARGE_FOLDER", True)
        ),
        force=args.force,
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


# --- docs & upload ------------------------------------------------------------

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


def _upload_model(api, king: KingUpload, model_dir: Path, settings: Settings, repo_id: str) -> None:
    logger.info("creating public HF repo {} (exist_ok)", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    # Our albedo.md is uploaded separately so we never write into the eval dir; the
    # cache marker / partials are internal and must not be published.
    ignore_patterns = [".albedo-model-cache.json", "*.download", ".cache/**", "albedo.md"]
    logger.info(
        "uploading {} model files to {} ({} workers)",
        king.king_name,
        repo_id,
        settings.upload_workers,
    )
    if settings.use_large_folder:
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=str(model_dir),
            repo_type="model",
            private=False,
            ignore_patterns=ignore_patterns,
            num_workers=settings.upload_workers,
            print_report=True,
            print_report_every=60,
        )
    else:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(model_dir),
            repo_type="model",
            commit_message=f"Mirror Albedo {king.king_name}",
            ignore_patterns=ignore_patterns,
        )
    api.upload_file(
        path_or_fileobj=io.BytesIO(render_albedo_md(king).encode("utf-8")),
        path_in_repo="albedo.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add albedo.md for {king.king_name}",
    )
    logger.info("uploaded {} to {}", king.king_name, repo_id)


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


def process_once(api, settings: Settings, *, limit: int | None = None) -> dict:
    # Fresh, short-lived read-only connection per pass: a dropped connection self-heals
    # on the next poll instead of wedging the monitor, and the DB isn't held open during
    # the (potentially long) model uploads.
    with _connect(settings) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        kings = list_crowned_kings(conn, settings)
    counts = {"total": len(kings), "uploaded": 0, "skipped": 0, "failed": 0}
    done = 0
    for king in kings:
        repo_id = repo_id_for(king, settings)
        try:
            if not settings.force and already_uploaded(api, repo_id):
                counts["skipped"] += 1
                continue
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
    parser.add_argument("--limit", type=int, help="max uploads for this process")
    parser.add_argument("--database-url", help="Postgres DSN override")
    parser.add_argument("--hf-namespace", help="HF namespace (env ALBEDO_KING_HF_NAMESPACE)")
    parser.add_argument("--hf-token", help="HF token override")
    parser.add_argument("--eval-dir", help="eval cache dir to read from, never deleted")
    parser.add_argument("--work-dir", help="delete-safe download dir for this watcher")
    parser.add_argument("--repo-prefix", help="HF repo name prefix for the king repos")
    parser.add_argument("--poll-interval-s", type=float, help="monitor polling interval")
    parser.add_argument("--upload-workers", type=int, help="HF upload worker count")
    parser.add_argument(
        "--small-upload",
        action="store_false",
        dest="use_large_folder",
        default=None,
        help="use HfApi.upload_folder instead of upload_large_folder",
    )
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

        logger.info("back-fill phase: mirroring crowned kings missing from Hugging Face")
        counts = process_once(api, settings, limit=args.limit)
        logger.info(
            "back-fill complete: {} uploaded, {} already on HF, {} failed (will retry) — "
            "entering monitor mode",
            counts["uploaded"],
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

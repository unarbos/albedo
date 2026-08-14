from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath

import httpx
from loguru import logger

from albedo_config import RepoContextSettings
from albedo_eval_service.remote.dataset import (
    _content,
    _extract_turns,
    _parse_sample_id,
    _read_parquet_row,
    _role,
    _unwrap_column,
)
from albedo_eval_service.shared.dataset_manifest import load_manifest_file
from albedo_eval_service.simulator.prompt_simulator import COMPLETE_MARKER

_API_BASE = "https://api.github.com"
_DONE_MARKER = ".albedo-repo-context-done"
_LISTING_NAME = ".albedo-listing.json"
_NEGATIVE_TTL_SECONDS = 24 * 3600.0
_TRANSIENT_TTL_SECONDS = 900.0
_MAX_MEMBER_BYTES = 2 * 1024 * 1024
_MAX_MEMBERS = 200_000
_MAX_MISSING_PATHS = 10
_RETRIES = 3

_COMMAND_BLOCK_RE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.DOTALL)

LISTING_HEADER = """REPOSITORY FILE LISTING — tracked files at the current commit relevant to the
command (paths relative to the repo root, sorted; the filesystem returns files in this order).
Derive the output of exploration commands (find, ls, grep -l, ...) EXACTLY from this list,
applying the command's filters and pipe limits:
- Output matching paths in EXACTLY the order they appear in this listing — never re-sort them.
- Do not invent paths that are not in this list and do not omit paths that match.
- This listing is complete for the scope shown: if nothing in it matches the command's
  filters, the command's output is empty.
"""

CONTENTS_HEADER = """FILE CONTENTS — exact current content of files referenced by the command:
"""

NOT_PRESENT_HEADER = """FILES NOT PRESENT — the following paths referenced by the command do NOT
exist in the repository at this commit. Never invent content for them. Your reply must still
follow the OUTPUT FORMAT exactly: when the command reports such a path, put the realistic
terminal error INSIDE the observation (e.g. "cat: <path>: No such file or directory"), and when
the command produces no output at all, reply with exactly the empty observation the OUTPUT
FORMAT specifies:
"""

TRAJECTORY_HEADER = """REFERENCE EXCHANGES — real command -> observation exchanges recorded in this
repository during a reference session on the same task. Use them ONLY to derive real paths, file
contents and output style; do not invent paths or contents they contradict, and never copy task
hints or solution steps into your output:
"""


@dataclass(frozen=True)
class RepoRef:
    instance_id: str
    source: str
    owner: str
    repo: str
    pr: str | None = None
    commit: str | None = None


@dataclass(frozen=True)
class GroundingContext:
    context: str | None
    kind: str
    reason: str | None = None


class _NotFound(Exception):
    pass


class _SnapshotTooLarge(Exception):
    pass


def parse_instance(source: str, instance_id: str) -> RepoRef | None:
    try:
        if source.startswith("mini-coder"):
            parts = instance_id.split("__")
            if len(parts) < 2:
                return None
            owner, rest = parts[0], parts[1]
            tokens = rest.split(".")
            for index in range(1, len(tokens)):
                if re.fullmatch(r"[0-9a-f]{6,40}", tokens[index]):
                    return RepoRef(
                        instance_id=instance_id,
                        source=source,
                        owner=owner,
                        repo=".".join(tokens[:index]),
                        commit=tokens[index],
                    )
            return None
        owner_repo, tail = instance_id.rsplit("-", 1)
        owner, repo = owner_repo.split("__", 1)
        if tail.isdigit():
            return RepoRef(instance_id=instance_id, source=source, owner=owner, repo=repo, pr=tail)
        if re.fullmatch(r"[0-9a-f]{40}", tail):
            return RepoRef(
                instance_id=instance_id, source=source, owner=owner, repo=repo, commit=tail
            )
        return None
    except ValueError:
        return None


def _first_command(text: str) -> str:
    match = _COMMAND_BLOCK_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _name_patterns(cmd: str) -> list[str]:
    return (
        re.findall(r"-name\s+\"([^\"]+)\"", cmd)
        + re.findall(r"-name\s+'([^']+)'", cmd)
        + re.findall(
            r"-name\s+([^\s\"';|]+)",
            cmd.replace('-name "', '-nameQ"').replace("-name '", "-nameQ'"),
        )
    )


def _filter_listing(paths: list[str], cmd: str) -> tuple[list[str], bool]:
    pats = _name_patterns(cmd)
    if not (pats and cmd.startswith(("find", "ls"))):
        return paths, False
    regexes = [
        re.compile(re.escape(p).replace(r"\*", ".*").replace(r"\?", ".") + "$") for p in pats
    ]
    kept = [p for p in paths if any(r.match(p.rsplit("/", 1)[-1]) or r.match(p) for r in regexes)]
    return kept, True


def _referenced_paths(cmd: str, listing: list[str]) -> tuple[list[str], list[str]]:
    listing_set = set(listing)
    top_dirs = {p.split("/", 1)[0] for p in listing_set if "/" in p}
    present: list[str] = []
    missing: list[str] = []
    for tok in re.split(r"[\s|;&<>()\"']+", cmd):
        if not tok or tok.startswith("-"):
            continue
        p = tok[2:] if tok.startswith("./") else tok
        if p in listing_set:
            if p not in present:
                present.append(p)
            continue
        if "://" in tok or any(ch in tok for ch in "*?[]{}$`="):
            continue
        norm = p.rstrip("/")
        if not norm or norm in (".", "..") or norm in missing:
            continue
        if any(x.startswith(norm + "/") for x in listing_set):
            continue
        if tok.startswith(("/", "..")):
            missing.append(norm)
        elif _plausible_repo_path(norm, top_dirs):
            missing.append(norm)
    return present, missing[:_MAX_MISSING_PATHS]


def _plausible_repo_path(path: str, top_dirs: set[str]) -> bool:
    segments = path.split("/")
    if segments[0] in top_dirs:
        return True
    if len(segments) == 1 and path.startswith("."):
        return True
    last = segments[-1]
    return bool(re.fullmatch(r"\.?[\w@.-]+\.[A-Za-z][A-Za-z0-9]*", last))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


@lru_cache(maxsize=64)
def _load_listing(listing_path: str) -> tuple[str, ...]:
    return tuple(json.loads(Path(listing_path).read_text()))


@lru_cache(maxsize=4096)
def _iid_from_parquet(dataset_root: str, shard_name: str, row_idx: int) -> str | None:
    import pyarrow.parquet as pq

    path = Path(dataset_root) / shard_name
    try:
        seen = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["instance_id"]):
            if seen + batch.num_rows <= row_idx:
                seen += batch.num_rows
                continue
            value = batch.column("instance_id")[row_idx - seen].as_py()
            return str(value) if value else None
    except Exception:
        return None
    return None


class RepoContextService:
    def __init__(self, settings: RepoContextSettings):
        if not settings.cache_dir:
            raise ValueError(
                "ALBEDO_REPO_CONTEXT_CACHE_DIR is required: the snapshot download directory "
                "must be configured explicitly"
            )
        self.settings = settings
        self.cache_dir = Path(settings.cache_dir).expanduser()
        self._shas_dir = self.cache_dir / "shas"
        self._snapshots_dir = self.cache_dir / "snapshots"
        self._client = httpx.Client(
            timeout=httpx.Timeout(60.0),
            follow_redirects=True,
            headers={"User-Agent": "albedo-repo-context", "Accept": "application/vnd.github+json"},
        )
        self._github_semaphore = threading.BoundedSemaphore(8)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_mutex = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._shards: dict[str, tuple[str, list]] | None = None
        self._manifest_error_logged = False

    def close(self) -> None:
        self._client.close()

    def context_for(self, sample_id: str, assistant_output: str) -> GroundingContext:
        try:
            return self._context_for(sample_id, assistant_output)
        except Exception as exc:
            logger.warning(
                "repo_context_fallback sample_id={} kind=none reason=unexpected error={}",
                sample_id,
                f"{type(exc).__name__}: {exc}",
            )
            return GroundingContext(context=None, kind="none", reason="unexpected")

    def prefetch(self, sample_ids: list[str]) -> dict[str, int]:
        instances: dict[str, str] = {}
        for sample_id in sample_ids:
            try:
                shard_name, row_idx, _ = _parse_sample_id(sample_id)
            except ValueError:
                continue
            source_iid = self._iid_for(shard_name, row_idx)
            if source_iid is not None:
                instances[source_iid[1]] = source_iid[0]

        def warm(instance_id: str, source: str) -> bool:
            try:
                ref = parse_instance(source, instance_id)
                if ref is None:
                    return False
                resolved = self._resolve_sha(ref)
                if resolved is None:
                    return False
                return self._ensure_snapshot(*resolved) is not None
            except Exception as exc:
                logger.warning(
                    "repo_context_prefetch_instance_failed instance={} error={}",
                    instance_id,
                    f"{type(exc).__name__}: {exc}",
                )
                return False

        ready = 0
        if instances:
            with ThreadPoolExecutor(max_workers=8) as pool:
                ready = sum(pool.map(lambda item: warm(*item), instances.items()))
        summary = {"samples": len(sample_ids), "instances": len(instances), "ready": ready}
        logger.info(
            "repo_context_prefetch_done samples={} instances={} ready={}",
            summary["samples"],
            summary["instances"],
            summary["ready"],
        )
        return summary

    def repo_context_for_instance(
        self, source: str, instance_id: str, assistant_output: str
    ) -> GroundingContext:
        ref = parse_instance(source, instance_id)
        if ref is None:
            return GroundingContext(context=None, kind="none", reason="instance_unparsed")
        resolved = self._resolve_sha(ref)
        if resolved is None:
            return GroundingContext(context=None, kind="none", reason="sha_unresolved")
        owner, repo, sha = resolved
        snapshot = self._ensure_snapshot(owner, repo, sha)
        if snapshot is None:
            return GroundingContext(context=None, kind="none", reason="snapshot_unavailable")
        listing = list(_load_listing(str(snapshot / _LISTING_NAME)))
        block = self._build_repo_block(snapshot, listing, _first_command(assistant_output))
        return GroundingContext(context=block, kind="repo")

    def _context_for(self, sample_id: str, assistant_output: str) -> GroundingContext:
        try:
            shard_name, row_idx, turn_idx = _parse_sample_id(sample_id)
        except ValueError:
            return GroundingContext(context=None, kind="none", reason="bad_sample_id")
        reason = "iid_unresolved"
        source_iid = self._iid_for(shard_name, row_idx)
        if source_iid is not None:
            result = self.repo_context_for_instance(*source_iid, assistant_output)
            if result.context is not None:
                return result
            reason = result.reason or "repo_unavailable"
        block = self._trajectory_block(shard_name, row_idx, turn_idx)
        if block is not None:
            logger.info(
                "repo_context_fallback sample_id={} kind=trajectory reason={}", sample_id, reason
            )
            return GroundingContext(context=block, kind="trajectory", reason=reason)
        logger.warning("repo_context_fallback sample_id={} kind=none reason={}", sample_id, reason)
        return GroundingContext(context=None, kind="none", reason=reason)

    def _iid_for(self, shard_name: str, row_idx: int) -> tuple[str, str] | None:
        if self.settings.dataset_manifest_path:
            resolved = self._iid_from_manifest(shard_name, row_idx)
            if resolved is not None:
                return resolved
        if self.settings.dataset_root:
            iid = _iid_from_parquet(self.settings.dataset_root, shard_name, row_idx)
            if iid:
                return (shard_name.split("/data/", 1)[0], iid)
        return None

    def _iid_from_manifest(self, shard_name: str, row_idx: int) -> tuple[str, str] | None:
        try:
            shards = self._manifest_shards()
        except Exception as exc:
            if not self._manifest_error_logged:
                self._manifest_error_logged = True
                logger.warning(
                    "repo_context_manifest_unavailable path={} error={}",
                    self.settings.dataset_manifest_path,
                    f"{type(exc).__name__}: {exc}",
                )
            return None
        entry = shards.get(shard_name)
        if entry is None:
            return None
        source, rows_meta = entry
        if not 0 <= row_idx < len(rows_meta):
            return None
        iid = rows_meta[row_idx].get("iid") if isinstance(rows_meta[row_idx], dict) else None
        return (source, str(iid)) if iid else None

    def _manifest_shards(self) -> dict[str, tuple[str, list]]:
        if self._shards is None:
            with self._manifest_lock:
                if self._shards is None:
                    manifest = load_manifest_file(
                        self.settings.dataset_manifest_path,
                        expected_sha256=self.settings.dataset_manifest_hash,
                    )
                    shards: dict[str, tuple[str, list]] = {}
                    for source in manifest.get("sources", []):
                        for shard in source.get("shards", []):
                            shards[shard["path"]] = (
                                str(source.get("name", "")),
                                shard.get("rows_meta") or [],
                            )
                    self._shards = shards
        return self._shards

    def _resolve_sha(self, ref: RepoRef) -> tuple[str, str, str] | None:
        cache_path = self._shas_dir / f"{_safe_name(ref.instance_id)}.json"
        cached = _read_json(cache_path)
        if cached is not None:
            if cached.get("sha"):
                return cached["owner"], cached["repo"], cached["sha"]
            if time.time() - float(cached.get("failed_at", 0)) < _NEGATIVE_TTL_SECONDS:
                return None
        with self._key_lock(f"sha:{ref.instance_id}"):
            cached = _read_json(cache_path)
            if cached is not None and cached.get("sha"):
                return cached["owner"], cached["repo"], cached["sha"]
            owner, repo = ref.owner, ref.repo
            try:
                try:
                    sha = self._sha_from_api(owner, repo, ref)
                except _NotFound:
                    data = self._github_json(f"/repos/{owner}/{repo}")
                    owner, repo = data["full_name"].split("/", 1)
                    sha = self._sha_from_api(owner, repo, ref)
            except Exception as exc:
                logger.info(
                    "repo_context_sha_unresolved instance={} error={}",
                    ref.instance_id,
                    f"{type(exc).__name__}: {exc}",
                )
                _write_json_atomic(
                    cache_path, {"error": f"{type(exc).__name__}: {exc}", "failed_at": time.time()}
                )
                return None
            _write_json_atomic(cache_path, {"owner": owner, "repo": repo, "sha": sha})
            return owner, repo, sha

    def _sha_from_api(self, owner: str, repo: str, ref: RepoRef) -> str:
        if ref.pr is not None:
            data = self._github_json(f"/repos/{owner}/{repo}/pulls/{ref.pr}")
            return data["base"]["sha"]
        data = self._github_json(f"/repos/{owner}/{repo}/commits/{ref.commit}")
        return data["sha"]

    def _ensure_snapshot(self, owner: str, repo: str, sha: str) -> Path | None:
        key = f"{_safe_name(owner)}__{_safe_name(repo)}__{sha[:12]}"
        final = self._snapshots_dir / key
        if (final / _DONE_MARKER).exists():
            return final
        failed_path = self._snapshots_dir / f"{key}.failed.json"
        if self._failure_fresh(failed_path):
            return None
        with self._key_lock(f"snapshot:{key}"):
            if (final / _DONE_MARKER).exists():
                return final
            if self._failure_fresh(failed_path):
                return None
            self._enforce_cache_limit()
            self._snapshots_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(dir=self._snapshots_dir, prefix=f".{key}.partial-"))
            try:
                with tempfile.NamedTemporaryFile(dir=self._snapshots_dir, suffix=".tar.gz") as tar:
                    self._download_tarball(owner, repo, sha, Path(tar.name))
                    listing, extracted_bytes = self._extract_tarball(Path(tar.name), tmp_dir)
                (tmp_dir / _LISTING_NAME).write_text(json.dumps(listing))
                (tmp_dir / _DONE_MARKER).write_text(json.dumps({"bytes": extracted_bytes}))
                os.replace(tmp_dir, final)
            except Exception as exc:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                if (final / _DONE_MARKER).exists():
                    return final
                ttl_kind = "oversized" if isinstance(exc, _SnapshotTooLarge) else "transient"
                logger.info(
                    "repo_context_snapshot_failed repo={}/{} sha={} kind={} error={}",
                    owner,
                    repo,
                    sha[:12],
                    ttl_kind,
                    f"{type(exc).__name__}: {exc}",
                )
                _write_json_atomic(
                    failed_path,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at": time.time(),
                        "kind": ttl_kind,
                    },
                )
                return None
            return final

    @staticmethod
    def _failure_fresh(failed_path: Path) -> bool:
        failed = _read_json(failed_path)
        if failed is None:
            return False
        ttl = _NEGATIVE_TTL_SECONDS if failed.get("kind") == "oversized" else _TRANSIENT_TTL_SECONDS
        return time.time() - float(failed.get("failed_at", 0)) < ttl

    def _download_tarball(self, owner: str, repo: str, sha: str, dest: Path) -> None:
        url = f"{_API_BASE}/repos/{owner}/{repo}/tarball/{sha}"
        max_bytes = self.settings.max_snapshot_mb * 1024 * 1024
        with self._github_semaphore:
            with self._client.stream("GET", url, headers=self._auth_headers()) as response:
                if response.status_code == 404:
                    raise _NotFound(url)
                response.raise_for_status()
                declared = int(response.headers.get("content-length") or 0)
                if declared > max_bytes:
                    raise _SnapshotTooLarge(f"tarball {declared} bytes > {max_bytes}")
                written = 0
                with dest.open("wb") as out:
                    for chunk in response.iter_bytes(1 << 20):
                        written += len(chunk)
                        if written > max_bytes:
                            raise _SnapshotTooLarge(f"tarball exceeds {max_bytes} bytes")
                        out.write(chunk)

    def _enforce_cache_limit(self) -> None:
        limit = int(self.settings.max_cache_gb * 1024**3)
        if limit <= 0:
            return
        usage = 0
        for marker in self._snapshots_dir.glob(f"*/{_DONE_MARKER}"):
            data = _read_json(marker)
            usage += int(data.get("bytes", 0)) if data else 0
        if usage < limit:
            return
        logger.warning(
            "repo_context_cache_cleared usage_bytes={} limit_bytes={} dir={}",
            usage,
            limit,
            self._snapshots_dir,
        )
        shutil.rmtree(self._snapshots_dir, ignore_errors=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        _load_listing.cache_clear()

    def _extract_tarball(self, tar_path: Path, dest_dir: Path) -> tuple[list[str], int]:
        max_extracted = self.settings.max_snapshot_mb * 1024 * 1024 * 4
        total = 0
        paths: list[str] = []
        with tarfile.open(tar_path, mode="r:gz") as tar:
            for member in tar:
                if len(paths) >= _MAX_MEMBERS:
                    raise _SnapshotTooLarge(f"tarball has more than {_MAX_MEMBERS} files")
                if not member.isreg() or member.name.startswith("/"):
                    continue
                parts = PurePosixPath(member.name).parts
                if len(parts) < 2 or any(part in ("..", "") for part in parts):
                    continue
                if parts[1] in (_LISTING_NAME, _DONE_MARKER):
                    continue
                if member.size > _MAX_MEMBER_BYTES:
                    paths.append("/".join(parts[1:]))
                    continue
                total += member.size
                if total > max_extracted:
                    raise _SnapshotTooLarge(f"extracted size exceeds {max_extracted} bytes")
                rel = "/".join(parts[1:])
                target = dest_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                paths.append(rel)
        return sorted(paths), total

    _LISTING_MIN_CHARS = 8000

    def _build_repo_block(self, snapshot_dir: Path, listing: list[str], cmd: str) -> str:
        listing_paths, _ = _filter_listing(listing, cmd)
        present, missing = _referenced_paths(cmd, listing)
        missing_text = (
            "\n" + NOT_PRESENT_HEADER + "\n".join(f"- {p}" for p in missing) + "\n"
            if missing
            else ""
        )

        contents_budget = (
            self.settings.max_context_chars
            - len(LISTING_HEADER)
            - len(missing_text)
            - self._LISTING_MIN_CHARS
        )
        contents_parts: list[str] = []
        contents_used = len(CONTENTS_HEADER) + 2
        for path in present[: self.settings.max_files]:
            text = self._read_snapshot_file(snapshot_dir, path)
            if text is None:
                continue
            part = f"--- ./{path} ---\n{_truncate(text, self.settings.max_file_chars)}\n"
            if contents_used + len(part) > contents_budget:
                break
            contents_parts.append(part)
            contents_used += len(part) + 1
        contents_text = (
            "\n" + CONTENTS_HEADER + "\n" + "\n".join(contents_parts) if contents_parts else ""
        )

        listing_budget = (
            self.settings.max_context_chars
            - len(LISTING_HEADER)
            - len(contents_text)
            - len(missing_text)
            - 64
        )
        if not listing_paths:
            listing_text = (
                "(no files in this repository match the command's filters — "
                "the exploration output is empty)"
            )
        else:
            lines: list[str] = []
            used = 0
            for path in listing_paths[: self.settings.max_paths]:
                line = "./" + path
                if used + len(line) + 1 > listing_budget:
                    break
                lines.append(line)
                used += len(line) + 1
            over = len(listing_paths) - len(lines)
            listing_text = "\n".join(lines)
            if over > 0:
                listing_text += f"\n... (+{over} more matching files)"

        block = LISTING_HEADER + "\n" + listing_text + "\n" + contents_text + missing_text
        return _truncate(block, self.settings.max_context_chars)

    def _read_snapshot_file(self, snapshot_dir: Path, rel_path: str) -> str | None:
        root = snapshot_dir.resolve()
        try:
            target = (snapshot_dir / rel_path).resolve()
        except OSError:
            return None
        if not target.is_relative_to(root) or not target.is_file():
            return None
        try:
            return target.read_text(errors="replace")
        except OSError:
            return None

    def _trajectory_block(self, shard_name: str, row_idx: int, turn_idx: int) -> str | None:
        if not self.settings.dataset_root:
            return None
        try:
            row = _read_parquet_row(Path(self.settings.dataset_root) / shard_name, row_idx)
        except Exception as exc:
            logger.info(
                "repo_context_trajectory_unavailable shard={} row={} error={}",
                shard_name,
                row_idx,
                f"{type(exc).__name__}: {exc}",
            )
            return None
        normalized = {key: _unwrap_column(value) for key, value in row.items()}
        turns = _extract_turns(normalized)
        assistant_positions = [i for i, turn in enumerate(turns) if _role(turn) == "assistant"]
        pairs: list[tuple[str, str]] = []
        for assistant_index, position in enumerate(assistant_positions):
            if assistant_index < turn_idx:
                continue
            command = _first_command(_content(turns[position]))
            if not command or position + 1 >= len(turns):
                continue
            if _role(turns[position + 1]) == "assistant":
                continue
            observation = _content(turns[position + 1]).strip()
            if not observation:
                continue
            if COMPLETE_MARKER in command or COMPLETE_MARKER in observation:
                continue
            pairs.append((command, observation))
            if len(pairs) >= self.settings.max_trajectory_pairs:
                break
        if not pairs:
            return None
        parts = [TRAJECTORY_HEADER]
        for step, (command, observation) in enumerate(pairs, 1):
            parts.append(
                f"REFERENCE STEP {step}:\n```bash\n{command}\n```\n\n"
                f"ENVIRONMENT OBSERVATION:\n{_truncate(observation, self.settings.max_file_chars)}"
            )
        return _truncate("\n\n".join(parts), self.settings.max_context_chars)

    def _auth_headers(self) -> dict[str, str]:
        token = (
            self.settings.github_token
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or ""
        )
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _github_json(self, path: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                with self._github_semaphore:
                    response = self._client.get(_API_BASE + path, headers=self._auth_headers())
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(min(30.0, 1.5 * 2**attempt))
                continue
            if response.status_code == 404:
                raise _NotFound(path)
            if (
                response.status_code == 429
                or response.status_code >= 500
                or (
                    response.status_code == 403
                    and response.headers.get("x-ratelimit-remaining") == "0"
                )
            ):
                last_error = RuntimeError(f"github {response.status_code} for {path}")
                retry_after = float(response.headers.get("retry-after") or 0)
                time.sleep(min(30.0, max(retry_after, 1.5 * 2**attempt)))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"github request failed for {path}: {last_error}")

    def _key_lock(self, key: str) -> threading.Lock:
        with self._locks_mutex:
            return self._locks.setdefault(key, threading.Lock())


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

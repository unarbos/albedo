from __future__ import annotations

import subprocess
from datetime import datetime

from config import Config

QUERY_SSH = """
SELECT DISTINCT er.id, er.started_at, a.uri, a.size_bytes
FROM eval_runs er
JOIN artifacts a ON a.stage_attempt_id = er.stage_attempt_id
WHERE er.state = 'SUCCEEDED' AND er.started_at >= '{anchor}'
ORDER BY er.started_at, a.uri;
"""

QUERY_DIRECT = """
SELECT DISTINCT er.id::text, er.started_at, a.uri, a.size_bytes
FROM eval_runs er
JOIN artifacts a ON a.stage_attempt_id = er.stage_attempt_id
WHERE er.state = 'SUCCEEDED' AND er.started_at >= $1
ORDER BY er.started_at, a.uri;
"""

REMOTE_TEMPLATE = """set -a; . {env_file}; set +a
docker exec -i -e PGPASSWORD="$ALBEDO_POSTGRES_PASSWORD" {container} \
  psql -U "$ALBEDO_POSTGRES_USER" -d "$ALBEDO_POSTGRES_DB" -P pager=off -t -A -F"\t" -v ON_ERROR_STOP=1 <<'SQL'
{query}
SQL"""


def query_succeeded_runs(cfg: Config, anchor: str) -> dict[str, dict]:
    anchor_dt = datetime.fromisoformat(anchor)
    if cfg.db_url:
        rows = _query_direct(cfg.db_url, anchor_dt)
    else:
        rows = _query_ssh(cfg, anchor)
    runs: dict[str, dict] = {}
    for run_id, started_at, uri, size in rows:
        run = runs.setdefault(run_id, {"started_at": started_at, "files": {}})
        run["files"][uri.rsplit("/", 1)[-1]] = (uri, size)
    return runs


def _query_direct(db_url: str, anchor: datetime) -> list[tuple[str, str, str, int]]:
    import asyncio

    import asyncpg

    async def go():
        conn = await asyncpg.connect(db_url)
        try:
            recs = await conn.fetch(QUERY_DIRECT, anchor)
        finally:
            await conn.close()
        return [(r[0], r[1].isoformat(), r[2], int(r[3])) for r in recs]

    return asyncio.run(go())


def _query_ssh(cfg: Config, anchor: str) -> list[tuple[str, str, str, int]]:
    missing = [
        name
        for name, value in (
            ("DATASET_CREATOR_DB_SSH_HOST", cfg.ssh_host),
            ("DATASET_CREATOR_DB_ENV_FILE", cfg.remote_env_file),
            ("DATASET_CREATOR_DB_CONTAINER", cfg.pg_container),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"missing env vars for DB access (set DATASET_CREATOR_DB_URL or: {', '.join(missing)})"
        )
    script = REMOTE_TEMPLATE.format(
        env_file=cfg.remote_env_file,
        container=cfg.pg_container,
        query=QUERY_SSH.format(anchor=anchor),
    )
    res = subprocess.run(
        ["ssh", cfg.ssh_host, "bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0:
        raise RuntimeError(f"DB query failed: {res.stderr.strip()[:500]}")
    rows = []
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        run_id, started_at, uri, size = line.split("\t")
        rows.append((run_id, started_at, uri, int(size)))
    return rows

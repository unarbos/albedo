"""Resolve the current SN97 king from the eval Postgres (slot 1 of the active reign)."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from sanity_remote.worker import _model_ref_parts

from config import KingChatSettings

_KING_SQL = """
SELECT ms.model_uri,
       rm.model_hash,
       rm.uid,
       rm.hotkey,
       kv.version AS king_version
FROM reigns r
JOIN reign_members rm ON rm.reign_id = r.id AND rm.slot = 1
JOIN king_versions kv ON kv.id = rm.king_version_id
JOIN model_submissions ms ON ms.id = rm.submission_id
WHERE r.state = 'ACTIVE'
ORDER BY r.version DESC
LIMIT 1
"""


@dataclass(frozen=True)
class King:
    model_uri: str
    model_hash: str
    uid: int | None = None
    hotkey: str | None = None
    king_version: int | None = None

    @property
    def digest(self) -> str:
        _repo, digest = _model_ref_parts(self.model_uri, self.model_hash)
        return digest


def _resolve_dsn(settings: KingChatSettings) -> str:
    """DSN precedence: explicit KING_CHAT_DATABASE_URL/ALBEDO_EVAL_DATABASE_URL, else the box's existing
    ALBEDO_POSTGRES_* parts (reuses hippius_validation.config.DB_URL — same eval DB the box already uses,
    so no new DB secret is needed). Importing that module also loads albedo/.env into the environment."""
    if settings.database_url:
        return settings.database_url
    try:
        from hippius_validation.config import DB_URL
        return DB_URL
    except Exception as exc:
        logger.warning("[king-chat] could not build DSN from ALBEDO_POSTGRES_*: {}", exc)
        return ""


def current_king(settings: KingChatSettings) -> King | None:
    """Return the current king, or None on empty reign / DB error (caller keeps the warm server)."""
    if settings.king_override_uri:
        return King(model_uri=settings.king_override_uri, model_hash=settings.king_override_hash)

    dsn = _resolve_dsn(settings)
    if not dsn:
        logger.warning("[king-chat] no database DSN available; cannot resolve king")
        return None

    try:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            row = conn.execute(_KING_SQL).fetchone()
    except Exception as exc:
        logger.warning("[king-chat] king query failed: {}", exc)
        return None

    if not row or not row.get("model_uri"):
        logger.warning("[king-chat] no active king found (empty reign?)")
        return None

    return King(
        model_uri=row["model_uri"],
        model_hash=row["model_hash"] or "",
        uid=row["uid"],
        hotkey=row["hotkey"],
        king_version=row["king_version"],
    )

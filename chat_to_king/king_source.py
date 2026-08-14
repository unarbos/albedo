from __future__ import annotations

from dataclasses import dataclass

from config import KingChatSettings
from loguru import logger

from sanity_remote.worker import _model_ref_parts

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

_LINEAGE_SQL = """
SELECT ms.model_uri,
       ms.architecture,
       ms.parameter_count,
       a.uri AS artifact_uri,
       coalesce(r.reason, '') AS reason
FROM king_versions kv
JOIN model_submissions ms ON ms.id = kv.submission_id
LEFT JOIN artifacts a ON a.id = kv.artifact_id
LEFT JOIN reigns r ON r.id = kv.entered_reign_id
WHERE kv.version <= %s
ORDER BY kv.version ASC
"""

_QWEN_PATTERNS = ("qwen3.6", "qwen3-6", "qwen3_6")
_SIZE_PATTERNS = ("35b", "35-b")
_GENESIS_MARKERS = ("qwen3.6-35b-a3b-genesis", "35b-a3b-genesis")
_ROMAN_NUMERALS = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


@dataclass(frozen=True)
class King:
    model_uri: str
    model_hash: str
    uid: int | None = None
    hotkey: str | None = None
    king_version: int | None = None
    roman: str = ""

    @property
    def digest(self) -> str:
        _repo, digest = _model_ref_parts(self.model_uri, self.model_hash)
        return digest


def _to_roman(n: int) -> str:
    out: list[str] = []
    for value, symbol in _ROMAN_NUMERALS:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)


def _lineage_roman(conn, king_version: int) -> str:
    number = 0
    for row in conn.execute(_LINEAGE_SQL, (king_version,)):
        text = " ".join(
            str(row[key] or "")
            for key in ("model_uri", "artifact_uri", "architecture", "parameter_count")
        ).lower()
        if not (any(p in text for p in _QWEN_PATTERNS) and any(p in text for p in _SIZE_PATTERNS)):
            continue
        repo = (row["model_uri"] or row["artifact_uri"] or "").lower()
        if row["reason"].upper() == "GENESIS" or any(m in repo for m in _GENESIS_MARKERS):
            continue
        number += 1
    return _to_roman(number) if number else ""


def _resolve_dsn(settings: KingChatSettings) -> str:
    if settings.database_url:
        return settings.database_url
    try:
        from albedo_config.db import load_dotenv, postgres_dsn

        load_dotenv()
        return postgres_dsn()
    except Exception as exc:
        logger.warning("[king-chat] could not build DSN from ALBEDO_POSTGRES_*: {}", exc)
        return ""


def current_king(settings: KingChatSettings) -> King | None:
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
            if not row or not row.get("model_uri"):
                logger.warning("[king-chat] no active king found (empty reign?)")
                return None
            roman = _lineage_roman(conn, int(row["king_version"]))
    except Exception as exc:
        logger.warning("[king-chat] king query failed: {}", exc)
        return None

    return King(
        model_uri=row["model_uri"],
        model_hash=row["model_hash"] or "",
        uid=row["uid"],
        hotkey=row["hotkey"],
        king_version=row["king_version"],
        roman=roman,
    )

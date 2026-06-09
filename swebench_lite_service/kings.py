from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import SETTINGS


@dataclass(frozen=True)
class King:
    repo: str
    digest: str
    reign_number: int | None = None
    crowned_at: str = ""
    challenge_id: str = ""
    source: str = ""

    @property
    def key(self) -> str:
        return f"{self.repo}@{self.digest}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["key"] = self.key
        return data


def _digest(row: dict[str, Any]) -> str:
    return row.get("king_digest") or row.get("model_digest") or row.get("digest") or ""


def _repo(row: dict[str, Any]) -> str:
    return row.get("model_repo") or row.get("repo") or row.get("challenger_repo") or ""


def _parse_time(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


async def fetch_dashboard(url: str | None = None) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url or SETTINGS.dashboard_url)
        resp.raise_for_status()
        return resp.json()


def kings_from_dashboard(dashboard: dict[str, Any]) -> list[King]:
    """Return known kings newest-first, using crown time as the source of truth.

    The website re-derives display reign labels from accepted history instead of
    trusting history.king_reign_number, which can be stale for older rows. Mirror
    that convention here so SWE-bench rows line up with labels like ALBEDO-XXXIV.
    """
    by_challenge: dict[str, King] = {}
    fallback: list[King] = []

    def add(king: King) -> None:
        if not king.repo or not king.digest:
            return
        if king.challenge_id:
            prior = by_challenge.get(king.challenge_id)
            if prior is None or _metadata_score(king) > _metadata_score(prior):
                by_challenge[king.challenge_id] = king
        else:
            fallback.append(king)

    current = dashboard.get("king")
    if isinstance(current, dict):
        add(_king_from_row(current, "king"))

    for row in dashboard.get("king_chain") or []:
        if isinstance(row, dict):
            add(_king_from_row(row, "king_chain"))

    for row in dashboard.get("history") or []:
        if not isinstance(row, dict) or not row.get("accepted"):
            continue
        repo = row.get("model_repo") or ""
        digest = row.get("model_digest") or ""
        if not repo or not digest:
            continue
        add(King(
            repo=repo,
            digest=digest,
            reign_number=None,
            crowned_at=row.get("completed_at") or row.get("crowned_at") or "",
            challenge_id=row.get("eval_id") or row.get("challenge_id") or "",
            source="history",
        ))

    by_id = list(by_challenge.values())
    by_id.sort(key=lambda king: _eval_num(king.challenge_id))
    total = dashboard.get("stats", {}).get("accepted") if isinstance(dashboard.get("stats"), dict) else None
    try:
        total_kings = int(total)
    except (TypeError, ValueError):
        total_kings = len(by_id)
    offset = max(0, total_kings - len(by_id))
    labeled = [
        King(
            repo=king.repo,
            digest=king.digest,
            reign_number=offset + index + 1,
            crowned_at=king.crowned_at,
            challenge_id=king.challenge_id,
            source=king.source,
        )
        for index, king in enumerate(by_id)
    ]

    deduped: dict[str, King] = {}
    for king in [*labeled, *fallback]:
        prior = deduped.get(king.key)
        if prior is None or _sort_key(king) > _sort_key(prior):
            deduped[king.key] = king

    return sorted(deduped.values(), key=_sort_key, reverse=True)


def _king_from_row(row: dict[str, Any], source: str) -> King:
    reign = row.get("reign_number")
    if reign is not None:
        try:
            reign = int(reign)
        except (TypeError, ValueError):
            reign = None
    return King(
        repo=_repo(row),
        digest=_digest(row),
        reign_number=reign,
        crowned_at=row.get("crowned_at") or row.get("completed_at") or "",
        challenge_id=row.get("challenge_id") or row.get("eval_id") or "",
        source=source,
    )


def _eval_num(value: str) -> int:
    digits = "".join(ch for ch in value or "" if ch.isdigit())
    return int(digits) if digits else 0


def _metadata_score(king: King) -> tuple[int, float]:
    source_score = {"king": 3, "king_chain": 2, "history": 1}.get(king.source, 0)
    return (source_score, _parse_time(king.crowned_at))


def _sort_key(king: King) -> tuple[float, int, str]:
    reign = king.reign_number if king.reign_number is not None else -1
    return (_parse_time(king.crowned_at), reign, king.key)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


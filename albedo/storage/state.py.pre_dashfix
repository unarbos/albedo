"""All mutable validator state, persisted via ObjectStore."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from albedo.config import DISPLAY_START_BLOCK, DUEL_KING_CHAIN_DEPTH
from albedo.storage.store import ObjectStore

log = logging.getLogger(__name__)

# Max times one eval_id may be re-queued after an infra/transient failure before
# we give up — bounds retries so a persistently-down dependency can't busy-loop.
# ALBEDO_MAX_REEVAL_PER_HOTKEY is the live-prod env name for the same budget.
_MAX_EVAL_RETRIES = int(os.environ.get(
    "ALBEDO_MAX_EVAL_RETRIES", os.environ.get("ALBEDO_MAX_REEVAL_PER_HOTKEY", "3")))

# R2 key constants
_KEY_KING      = "king.json"
_KEY_CHAIN     = "king_chain.json"
_KEY_SEEN      = "seen_hotkeys.json"
_KEY_COMPLETED = "completed_repos.json"
_KEY_QUEUE     = "queue.json"
_KEY_COUNTERS  = "counters.json"
_KEY_HISTORY   = "history.json"

_SEEN_LOAD_RETRIES    = 3
_DASHBOARD_MIN_PERIOD = 5.0   # seconds between flush_dashboard calls
_HISTORY_MAX_LEN      = int(os.environ.get("ALBEDO_HISTORY_MAX_LEN", "500"))


def _now_ts() -> float:
    return time.time()

@dataclass
class KingEntry:
    hotkey:          str
    model_repo:      str
    model_digest:    str
    block:           int
    challenge_id:    str
    dethrone_judges: list[str]
    crown_judges:    list[str]
    crowned_at:      float = field(default_factory=_now_ts)

    def __post_init__(self) -> None:
        if not isinstance(self.hotkey, str):
            raise TypeError(f"KingEntry.hotkey must be str, got {type(self.hotkey).__name__}")
        if not isinstance(self.model_repo, str):
            raise TypeError(f"KingEntry.model_repo must be str, got {type(self.model_repo).__name__}")
        if not isinstance(self.model_digest, str):
            raise TypeError(f"KingEntry.model_digest must be str, got {type(self.model_digest).__name__}")
        if not isinstance(self.block, int):
            raise TypeError(f"KingEntry.block must be int, got {type(self.block).__name__}")
        if not isinstance(self.challenge_id, str):
            raise TypeError(f"KingEntry.challenge_id must be str, got {type(self.challenge_id).__name__}")
        if not isinstance(self.crowned_at, (int, float)):
            raise TypeError(f"KingEntry.crowned_at must be numeric, got {type(self.crowned_at).__name__}")
        if not isinstance(self.dethrone_judges, list) or not all(isinstance(j, str) for j in self.dethrone_judges):
            raise TypeError("KingEntry.dethrone_judges must be list[str]")
        if not isinstance(self.crown_judges, list) or not all(isinstance(j, str) for j in self.crown_judges):
            raise TypeError("KingEntry.crown_judges must be list[str]")

    def to_dict(self) -> dict:
        return {
            "hotkey":          self.hotkey,
            "model_repo":      self.model_repo,
            "model_digest":    self.model_digest,
            "block":           self.block,
            "crowned_block":   self.block,   # alias used by dashboard JS filter
            "challenge_id":    self.challenge_id,
            "dethrone_judges": self.dethrone_judges,
            "crown_judges":    self.crown_judges,
            "crowned_at":      self.crowned_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KingEntry":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})



class State:
    """All mutable validator state, persisted across restarts via ObjectStore."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

        self.king:       KingEntry | None = None
        self.king_chain: list[KingEntry]  = []

        self.seen:            set[str] = set()
        self.completed_repos: set[str] = set()

        self.queue: list[dict] = []

        self.counter:       int            = 0
        self.retry_counts:  dict[str, int] = {}
        self.recovered_ids: set[str]       = set()

        self.current_eval: dict | None = None

        self.last_weight_block: int   = 0
        self.infra_cooldown:    float = 0.0
        self.watchdog:          float = 0.0

        self.history: list[dict]     = []
        self.stats:   dict[str, Any] = {}

        self.uid_map:     dict[str, int] = {}
        self.coldkey_for: dict[str, str] = {}

        self._last_dashboard_flush: float = 0.0

    def load(self) -> None:
        """Read all 7 R2 keys.

        Raises RuntimeError if seen_hotkeys.json fails after retries when king/queue
        exists — losing the seen-set would allow duplicate evaluations.
        """
        king_raw = self._store.get(_KEY_KING)
        try:
            self.king = KingEntry.from_dict(king_raw) if king_raw else None
        except (TypeError, KeyError) as exc:
            log.error("corrupt king record in R2 (%s) — treating as no king", exc)
            self.king = None

        chain_raw = self._store.get(_KEY_CHAIN) or []
        loaded_chain = []
        for entry in chain_raw:
            try:
                loaded_chain.append(KingEntry.from_dict(entry))
            except (TypeError, KeyError) as exc:
                log.warning("skipping corrupt king_chain entry (%s): %s", exc, entry)
        self.king_chain = loaded_chain

        # Critical: retry and raise on repeated failure to prevent duplicate evals.
        seen_raw = self._load_seen_with_retry()
        self.seen = set(seen_raw) if seen_raw is not None else set()

        completed_raw = self._store.get(_KEY_COMPLETED) or []
        self.completed_repos = set(completed_raw)

        self.queue = self._store.get(_KEY_QUEUE) or []

        counters = self._store.get(_KEY_COUNTERS) or {}
        self.counter       = int(counters.get("counter", 0))
        self.retry_counts  = {str(k): int(v) for k, v in counters.get("retry_counts", {}).items()}
        self.recovered_ids = set(counters.get("recovered_ids", []))

        self.history = self._store.get(_KEY_HISTORY) or []

        log.info(
            "State loaded — king=%s  seen=%d  queue=%d  history=%d",
            self.king.hotkey if self.king else None,
            len(self.seen),
            len(self.queue),
            len(self.history),
        )

    def _load_seen_with_retry(self) -> list | None:
        """Retry up to _SEEN_LOAD_RETRIES times; raise if king/queue is non-empty on failure."""
        for attempt in range(1, _SEEN_LOAD_RETRIES + 1):
            raw = self._store.get(_KEY_SEEN)
            if raw is not None:
                return raw
            if attempt < _SEEN_LOAD_RETRIES:
                log.warning("seen_hotkeys.json fetch returned None (attempt %d/%d)", attempt, _SEEN_LOAD_RETRIES)
                time.sleep(1.5 ** attempt)

        if self.king or self.queue:
            raise RuntimeError(
                f"seen_hotkeys.json could not be loaded after {_SEEN_LOAD_RETRIES} retries "
                "and king/queue is non-empty — aborting to prevent duplicate evaluations."
            )
        log.warning("seen_hotkeys.json missing; starting with empty set (no king, no queue).")
        return None

    def flush(self) -> bool:
        """Write all 7 R2 keys. Returns True only if every write succeeded."""
        ok = True
        ok &= self._store.put(_KEY_KING,      self.king.to_dict() if self.king else None)
        ok &= self._store.put(_KEY_CHAIN,     [e.to_dict() for e in self.king_chain])
        ok &= self._store.put(_KEY_SEEN,      sorted(self.seen))
        ok &= self._store.put(_KEY_COMPLETED, sorted(self.completed_repos))
        ok &= self._store.put(_KEY_QUEUE,     self.queue)
        ok &= self._store.put(_KEY_COUNTERS,  {
            "counter":       self.counter,
            "retry_counts":  self.retry_counts,
            "recovered_ids": sorted(self.recovered_ids),
        })
        ok &= self._store.put(_KEY_HISTORY, self.history)
        if not ok:
            log.error("flush: one or more R2 writes failed — state may be partially persisted")
        return ok

    def flush_dashboard(self, *, force: bool = False) -> bool:
        """Push a dashboard snapshot to Hippius, throttled to _DASHBOARD_MIN_PERIOD seconds."""
        now = time.monotonic()
        if not force and (now - self._last_dashboard_flush) < _DASHBOARD_MIN_PERIOD:
            return False

        payload = self._build_dashboard_payload()
        ok = self._store.put_dashboard("dashboard.json", payload)
        if ok:
            self._last_dashboard_flush = now
        return ok

    def _build_dashboard_payload(self) -> dict:
        return {
            "chain": {
                "display_start_block": DISPLAY_START_BLOCK,
            },
            "king":         self.king.to_dict() if self.king else None,
            "king_chain":   [e.to_dict() for e in self.king_chain],
            "queue_len":    len(self.queue),
            "seen_count":   len(self.seen),
            "counter":      self.counter,
            "current_eval": self.current_eval,  # live in-progress duel (None when idle)
            "stats":        self.stats,
            "history":      self.history[:50],  # cap to avoid huge payloads
            "ts":           _now_ts(),
        }

    def close_eval(self) -> None:
        """Clear the active eval slot, persist state, and push the dashboard."""
        self.current_eval = None
        self.flush()
        self.flush_dashboard(force=True)

    def set_king(
        self,
        hotkey:          str,
        model_repo:      str,
        model_digest:    str,
        block:           int,
        *,
        challenge_id:    str,
        dethrone_judges: list[str],
        crown_judges:    list[str],
    ) -> None:
        """Crown a new king, update king_chain, and persist."""
        old_king  = self.king
        old_chain = list(self.king_chain)
        entry = KingEntry(
            hotkey=hotkey,
            model_repo=model_repo,
            model_digest=model_digest,
            block=block,
            challenge_id=challenge_id,
            dethrone_judges=dethrone_judges,
            crown_judges=crown_judges,
        )
        if self.king is not None:
            self.king_chain.insert(0, self.king)
            self.king_chain = self.king_chain[:DUEL_KING_CHAIN_DEPTH]
        self.king = entry
        if not self.flush():
            # R2 write failed — roll back in-memory state so it stays consistent
            # with what's actually on disk, preventing ghost-king divergence.
            self.king       = old_king
            self.king_chain = old_chain
            log.error("set_king: R2 flush failed; in-memory state rolled back — king NOT updated")
            return
        log.info("New king: %s  repo=%s  block=%d", hotkey, model_repo, block)

    def enqueue(self, reveal: dict, *, force: bool = False) -> str | None:
        """Add *reveal* to the queue. Returns eval ID or None if duplicate.

        Without force=True, rejects if the hotkey is already queued/in-eval OR
        if the repo is already queued (prevents two hotkeys racing the same model
        past scan_reveals before either verdict updates completed_repos).
        """
        hotkey   = reveal.get("hotkey", "")
        repo     = reveal.get("model_repo", "")
        if not force:
            queued_hotkeys = {e.get("hotkey") for e in self.queue}
            queued_repos   = {e.get("model_repo") for e in self.queue}
            if self.current_eval:
                queued_hotkeys.add(self.current_eval.get("hotkey"))
                queued_repos.add(self.current_eval.get("model_repo"))
            if hotkey in queued_hotkeys:
                log.debug("enqueue skipped — %s already has a pending eval", hotkey)
                return None
            if repo and repo in queued_repos:
                log.debug("enqueue skipped — repo %s already queued under another hotkey", repo)
                return None

        self.counter += 1
        eval_id = f"eval-{self.counter:06d}"
        entry   = {**reveal, "eval_id": eval_id, "enqueued_at": _now_ts()}
        self.queue.append(entry)
        self.flush()
        log.info("Enqueued %s for %s", eval_id, hotkey)
        return eval_id

    def record_verdict(self, entry: dict, verdict: dict) -> None:
        """Append a verdict record to history and persist."""
        record = {
            "type":       "verdict",
            "eval_id":    entry.get("eval_id"),
            "hotkey":     entry.get("hotkey"),
            "model_repo": entry.get("model_repo", ""),
            "verdict":    verdict,
            "ts":         _now_ts(),
        }
        self.history.insert(0, record)
        if len(self.history) > _HISTORY_MAX_LEN:
            self.history = self.history[:_HISTORY_MAX_LEN]
        self.completed_repos.add(entry.get("model_repo", ""))
        self.flush()

    def record_failure(self, entry: dict, code: str, detail: str) -> None:
        """Append a failure record to history and persist.

        Stores model_repo/digest/block so lookback recovery can rebuild a re-queue entry.
        """
        record = {
            "type":         "failure",
            "eval_id":      entry.get("eval_id"),
            "hotkey":       entry.get("hotkey"),
            "model_repo":   entry.get("model_repo", ""),
            "model_digest": entry.get("model_digest", ""),
            "block":        entry.get("block"),
            "code":         code,
            "detail":       detail,
            "ts":           _now_ts(),
        }
        self.history.insert(0, record)
        if len(self.history) > _HISTORY_MAX_LEN:
            self.history = self.history[:_HISTORY_MAX_LEN]
        self.flush()

    def unburn(self, entry: dict) -> bool:
        """Re-queue an entry after an infra/transient failure, bounded by a retry budget.

        Returns True if re-queued, False if the per-eval retry budget is exhausted
        (caller has already recorded the failure). The hotkey stays in seen — the miner
        retries their existing reveal but cannot submit a new model until this resolves.
        """
        eval_id = entry.get("eval_id", "")
        self.recovered_ids.add(eval_id)
        self.retry_counts[eval_id] = self.retry_counts.get(eval_id, 0) + 1
        if self.retry_counts[eval_id] > _MAX_EVAL_RETRIES:
            log.warning("unburn: eval %s exceeded %d retries — giving up",
                        eval_id, _MAX_EVAL_RETRIES)
            self.flush()
            return False
        self.queue.insert(0, entry)
        self.flush()
        log.info("Returned eval slot for %s (%s) — retry %d/%d",
                 entry.get("hotkey"), eval_id, self.retry_counts[eval_id], _MAX_EVAL_RETRIES)
        return True

    def refresh_uid_map(self, subtensor: Any, netuid: int) -> None:
        """Rebuild uid_map and coldkey_for from the metagraph; not persisted to R2."""
        try:
            metagraph = subtensor.metagraph(netuid)
            self.uid_map = {
                str(neuron.hotkey): int(neuron.uid)
                for neuron in metagraph.neurons
            }
            self.coldkey_for = {
                str(neuron.hotkey): str(neuron.coldkey)
                for neuron in metagraph.neurons
            }
            log.debug("uid_map refreshed — %d neurons", len(self.uid_map))
        except Exception as exc:
            log.warning("refresh_uid_map failed: %s", exc)

    def eligible_hotkeys(self, uid_map: dict[str, int] | None = None) -> list[str]:
        """Return the emission recipients: the current king plus recent dethroned kings.

        King first, then king_chain in recency order, deduped, capped at
        DUEL_KING_CHAIN_DEPTH, and filtered to hotkeys still on the metagraph.
        The genesis king (empty hotkey) earns nothing and is skipped.
        Accepts an explicit uid_map snapshot to prevent TOCTOU races.
        """
        snapshot = uid_map if uid_map is not None else dict(self.uid_map)

        out: list[str] = []
        if self.king and self.king.hotkey:
            out.append(self.king.hotkey)
        for e in self.king_chain:
            if e.hotkey and e.hotkey not in out:
                out.append(e.hotkey)
            if len(out) >= DUEL_KING_CHAIN_DEPTH:
                break
        return [hk for hk in out[:DUEL_KING_CHAIN_DEPTH] if hk in snapshot]

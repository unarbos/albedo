"""Download the last 2 king models from Hippius for local inspection / preeval testing.

Edit the STATIC CONFIG section below, then run:
    cd /path/to/albedo
    source .venv/bin/activate
    python scripts/download_kings.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ── STATIC CONFIG ─────────────────────────────────────────────────────────────

# Where to save the downloaded model snapshots.
# Each king ends up in:   SAVE_DIR / "<namespace>--<repo>" / "snapshots" / "<digest>"
SAVE_DIR = Path("/home/const/similarity_test")

# Public dashboard URL (no auth needed — this is publicly readable).
# Format: https://us-east-1.hippius.com/<bucket>/dashboard.json
DASHBOARD_URL = "https://us-east-1.hippius.com/albedo/dashboard.json"

# How many kings to download (1 = current king only, 2 = current + previous).
N_KINGS = 5

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_kings")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fetch_dashboard(url: str) -> dict:
    try:
        import httpx
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())


def extract_kings(dashboard: dict, n: int) -> list[dict]:
    """Return up to n king entries (current first, then king_chain order).

    Dashboard uses model_repo + king_digest as field names (not repo/digest).
    Normalise to a consistent {repo, digest} shape for download.
    """
    def normalise(entry: dict) -> dict | None:
        repo   = entry.get("model_repo") or entry.get("repo", "")
        digest = entry.get("king_digest") or entry.get("digest", "")
        if repo and digest:
            return {**entry, "repo": repo, "digest": digest}
        return None

    kings: list[dict] = []

    current = normalise(dashboard.get("king") or {})
    if current:
        kings.append(current)

    for entry in dashboard.get("king_chain") or []:
        if len(kings) >= n:
            break
        norm = normalise(entry)
        if norm:
            kings.append(norm)

    return kings[:n]


def main() -> None:
    from model_store import ModelRef, materialize_model

    log.info("Fetching dashboard from %s", DASHBOARD_URL)
    try:
        dashboard = fetch_dashboard(DASHBOARD_URL)
    except Exception as exc:
        log.error("Failed to fetch dashboard: %s", exc)
        sys.exit(1)

    kings = extract_kings(dashboard, N_KINGS)
    if not kings:
        log.error("No kings found in dashboard — is the subnet running?")
        sys.exit(1)

    log.info("Found %d king(s) to download:", len(kings))
    for i, k in enumerate(kings):
        label = "current king" if i == 0 else f"previous king #{i}"
        log.info("  [%d] %s  %s @ %s", i + 1, label, k["repo"], k["digest"])

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    subfolder_names = ["king_current", "king_prev_1", "king_prev_2", "king_prev_3"]

    downloaded: list[tuple[str, Path]] = []
    for i, k in enumerate(kings):
        label = subfolder_names[i] if i < len(subfolder_names) else f"king_prev_{i}"
        dest = SAVE_DIR / label
        try:
            ref = ModelRef(k["repo"], k["digest"])
        except ValueError as exc:
            log.error("[%d] invalid ref — skipping: %s", i + 1, exc)
            continue

        log.info("[%d/%d] Downloading %s → %s …", i + 1, len(kings), ref.immutable_ref, dest)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            model_dir = materialize_model(ref, str(dest), 16)
            log.info("      → %s", model_dir)
            downloaded.append((label, Path(model_dir)))
        except Exception as exc:
            log.error("[%d] Download failed: %s", i + 1, exc)

    print()
    print("=" * 62)
    print(f"  Downloaded {len(downloaded)} / {len(kings)} king(s):")
    for label, path in downloaded:
        sf = list(path.rglob("*.safetensors"))
        size_gb = sum(p.stat().st_size for p in sf) / 1e9
        print(f"  [{label}]  {path}")
        print(f"            {len(sf)} safetensors file(s), {size_gb:.2f} GB")
    print("=" * 62)
    print()

    if len(downloaded) == 2:
        a_label, a_path = downloaded[0]
        b_label, b_path = downloaded[1]
        print("Ready for preeval test. Edit scripts/test_preeval.py:")
        print(f"  MODEL_A = Path('{a_path}')  # {a_label}")
        print(f"  MODEL_B = Path('{b_path}')  # {b_label}")
        print()

    sys.exit(0 if downloaded else 1)


if __name__ == "__main__":
    main()

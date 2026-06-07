"""Download specific Hippius Hub models by repo + revision.

Usage:
    cd /home/kandrzejak/albedo
    source .venv/bin/activate
    python scripts/download_models.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Static config ─────────────────────────────────────────────────────────────

SAVE_DIR = Path("/home/const/similarity_test")

MODELS = [
    {
        "repo":     "sota1028/albedo-mini-1.7b-miner_23",
        "revision": "miner_23",
        "label":    "model_a",
    },
    {
        "repo":     "divinequest/albedo-mini-1.7b-a091",
        "revision": "main",
        "label":    "model_b",
    },
]

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_models")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from hippius_hub import snapshot_download
    from model_store import ALLOW_PATTERNS, get_hub_token

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    token = get_hub_token()

    downloaded: list[tuple[str, Path]] = []
    for m in MODELS:
        dest = SAVE_DIR / m["label"]
        dest.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s @ %s → %s …", m["repo"], m["revision"], dest)
        try:
            path = snapshot_download(
                repo_id=m["repo"],
                revision=m["revision"],
                local_dir=str(dest),
                allow_patterns=ALLOW_PATTERNS,
                max_workers=16,
                token=token,
            )
            log.info("  → %s", path)
            downloaded.append((m["label"], Path(path)))
        except Exception as exc:
            log.error("  FAILED: %s", exc)

    print()
    print("=" * 62)
    print(f"  Downloaded {len(downloaded)} / {len(MODELS)} model(s):")
    for label, path in downloaded:
        sf = list(path.rglob("*.safetensors"))
        size_gb = sum(p.stat().st_size for p in sf) / 1e9
        print(f"  [{label}]  {path}")
        print(f"            {len(sf)} safetensors file(s), {size_gb:.2f} GB")
    print("=" * 62)

    if len(downloaded) == 2:
        a_label, a_path = downloaded[0]
        b_label, b_path = downloaded[1]
        print()
        print("Ready for preeval test. Edit scripts/test_preeval.py:")
        print(f"  MODEL_A = Path('{a_path}')  # {a_label}")
        print(f"  MODEL_B = Path('{b_path}')  # {b_label}")

    print()
    sys.exit(0 if downloaded else 1)


if __name__ == "__main__":
    main()

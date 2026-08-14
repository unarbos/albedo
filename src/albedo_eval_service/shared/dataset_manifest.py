from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_MANIFEST_HASH = "e3cff61772b0096811d4c5d8bbc8dee8dacbd9a069bc4557608adf1c1c2ddf40"


def load_manifest_file(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:

    manifest_path = Path(path)
    payload = manifest_path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    normalized_expected = expected_sha256.removeprefix("sha256:")
    if normalized_expected and actual_sha256 != normalized_expected:
        raise ValueError(
            f"dataset manifest hash mismatch: expected {normalized_expected}, got {actual_sha256}"
        )
    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("dataset manifest must be a JSON object")
    return loaded

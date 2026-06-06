"""Near-duplicate model detection via per-tensor fingerprints (v2: norms + value samples)."""
from __future__ import annotations

import hashlib
import logging
import math
import threading
from pathlib import Path

from albedo.config import PREEVAL_SIM_THRESHOLD

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()

FINGERPRINT_METHOD = "layer_norms_v2_with_samples"
SAMPLE_K = 16  # deterministic value samples drawn per tensor
_UNCHANGED_COSINE = 1.0 - 1e-6  # a tensor counts as unchanged at/above this per-sample cosine


def _deterministic_indices(key: str, n: int, k: int) -> list[int]:
    """k stable indices into a length-n tensor, derived from its key (shard-order invariant)."""
    if n <= 0:
        return [0] * k
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=4 * k).digest()
    return [int.from_bytes(h[i * 4:(i + 1) * 4], "big") % n for i in range(k)]


def compute_fingerprint(model_dir: str) -> dict:
    """Compute a v2 fingerprint: per-tensor L2 norm + K deterministic value samples.

    Returns {"method", "layer_keys", "norm_vector", "tensor_samples"} with layer_keys
    sorted for shard-order invariance. Raises FileNotFoundError if no shards are found.
    The value samples make the comparison direction-sensitive — a scaled or fine-tuned
    copy still shifts sampled values, unlike a norm-only fingerprint.
    """
    try:
        from safetensors import safe_open  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("safetensors is required for fingerprinting") from exc

    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No *.safetensors files found in {model_dir!r}")

    norms: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                flat = f.get_tensor(key).reshape(-1).float()
                n = int(flat.shape[0])
                norms[key] = float((flat * flat).sum().sqrt().item())
                samples[key] = [float(flat[i].item()) for i in _deterministic_indices(key, n, SAMPLE_K)]

    keys = sorted(norms)
    return {
        "method":         FINGERPRINT_METHOD,
        "layer_keys":     keys,
        "norm_vector":    [norms[k] for k in keys],
        "tensor_samples": [samples[k] for k in keys],
    }


def _vector_cosine(a: list[float], b: list[float]) -> float:
    """Cosine of two equal-length vectors; 0.0 on empty, mismatched, or zero-magnitude input."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / mag if mag else 0.0


def similarity(fp_a: dict, fp_b: dict) -> float:
    """v2 similarity in [0, 1]: fraction of tensors whose sampled values are ~unchanged.

    Returns 0.0 when architectures differ (layer_keys mismatch). Falls back to v1
    norm-vector cosine when per-tensor samples are missing on either side.
    """
    if fp_a.get("layer_keys") != fp_b.get("layer_keys"):
        return 0.0

    sa, sb = fp_a.get("tensor_samples"), fp_b.get("tensor_samples")
    if sa and sb and len(sa) == len(sb):
        unchanged = 0
        for a, b in zip(sa, sb):
            cos = _vector_cosine(a, b)
            # zero-vectors (e.g. uninitialised biases) trivially match in both copies
            if (not any(a) and not any(b)) or cos >= _UNCHANGED_COSINE:
                unchanged += 1
        return unchanged / len(sa)

    return _vector_cosine(fp_a.get("norm_vector", []), fp_b.get("norm_vector", []))


def check_fingerprint(
    challenger_dir: str,
    stored: dict[str, dict],
    threshold: float | None = None,
    hotkey: str = "",
) -> tuple[bool, str]:

    if threshold is None:
        threshold = PREEVAL_SIM_THRESHOLD

    try:
        challenger_fp = compute_fingerprint(challenger_dir)
    except Exception:
        logger.warning(
            "fingerprint computation failed for %r — failing open (not flagging)",
            challenger_dir,
            exc_info=True,
        )
        return False, ""

    for key, fp in stored.items():
        if hotkey and fp.get("hotkey") == hotkey:
            # Same miner's own prior model — not a duplicate of itself.
            logger.debug("fingerprint: skipping same-hotkey entry %r (hotkey=%s)", key, hotkey)
            continue
        sim = similarity(challenger_fp, fp)
        logger.debug("fingerprint similarity challenger vs %r: %.6f (threshold %.4f)", key, sim, threshold)
        if sim >= threshold:
            logger.warning(
                "challenger %r is near-duplicate of %r (similarity=%.6f >= %.4f)",
                challenger_dir, key, sim, threshold,
            )
            return True, key

    return False, ""


def add_fingerprint(key: str, model_dir: str, state: dict[str, dict], hotkey: str = "") -> None:
    """Compute and store the fingerprint for model_dir in state. Thread-safe.

    Records the committing hotkey so check_fingerprint can skip same-miner matches.
    """
    fp = compute_fingerprint(model_dir)
    fp["hotkey"] = hotkey
    with _STATE_LOCK:
        state[key] = fp
    logger.debug("stored fingerprint for %r (%d tensors, hotkey=%s)",
                 key, len(fp.get("layer_keys", [])), hotkey)

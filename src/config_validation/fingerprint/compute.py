"""Weight fingerprinting via per-tensor L2 norms + deterministic value samples.

Ported from albedo's per-tensor fingerprint (norms + samples, shard-order invariant)
but reads safetensors with numpy only — no torch — by parsing the file header directly
so bf16 checkpoints (the Qwen3.6-35B-A3B norm) are supported.

A fingerprint is direction-sensitive: a scaled or fine-tuned copy shifts the sampled
values, so it still differs from the original even when tensor norms are close.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import mmap
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FINGERPRINT_METHOD = "layer_norms_v2_with_samples"
SAMPLE_K = 16  # deterministic value samples drawn per tensor
_UNCHANGED_COSINE = 1.0 - 1e-6  # a tensor counts as unchanged at/above this per-sample cosine

# safetensors dtype -> numpy reader. bf16 has no numpy dtype, handled specially below.
_NP_DTYPE = {"F64": "<f8", "F32": "<f4", "F16": "<f2"}


def _deterministic_indices(key: str, n: int, k: int) -> list[int]:
    """k stable indices into a length-n tensor, derived from its key (shard-order invariant)."""
    if n <= 0:
        return [0] * k
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=4 * k).digest()
    return [int.from_bytes(h[i * 4:(i + 1) * 4], "big") % n for i in range(k)]


def _to_f32(raw: bytes, dtype: str) -> np.ndarray | None:
    """Decode a raw safetensors tensor buffer to a 1-D float32 array, or None if non-float."""
    if dtype in _NP_DTYPE:
        return np.frombuffer(raw, dtype=_NP_DTYPE[dtype]).astype(np.float32, copy=False)
    if dtype == "BF16":
        # bf16 -> f32: place the 16 bits in the high half of a 32-bit float pattern.
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (u16 << 16).view(np.float32)
    return None  # integer / bool tensors are not part of the weight fingerprint


def _iter_tensors(shard: Path):
    """Yield (key, f32_flat_array) for every float tensor in a safetensors shard."""
    with open(shard, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            header_len = int.from_bytes(mm[:8], "little")
            header = json.loads(mm[8:8 + header_len])
            data_start = 8 + header_len
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                start, end = info["data_offsets"]
                arr = _to_f32(mm[data_start + start:data_start + end], info["dtype"])
                if arr is not None:
                    yield key, arr
        finally:
            mm.close()


def compute_fingerprint(model_dir: str) -> dict:
    """Compute a fingerprint over all *.safetensors shards in ``model_dir``.

    Returns {"method", "layer_keys", "norm_vector", "tensor_samples"} with layer_keys
    sorted for shard-order invariance. Raises FileNotFoundError if no shards exist.
    """
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors files found in {model_dir!r}")

    norms: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    for shard in shards:
        for key, flat in _iter_tensors(shard):
            n = int(flat.shape[0])
            norms[key] = float(np.sqrt(np.square(flat.astype(np.float64)).sum()))
            idxs = _deterministic_indices(key, n, SAMPLE_K)
            samples[key] = [float(flat[i]) for i in idxs]

    keys = sorted(norms)
    return {
        "method": FINGERPRINT_METHOD,
        "layer_keys": keys,
        "norm_vector": [norms[k] for k in keys],
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
    """Similarity in [0, 1]: fraction of tensors whose sampled values are ~unchanged.

    Returns 0.0 when architectures differ (layer_keys mismatch). Falls back to a
    norm-vector cosine when per-tensor samples are missing on either side.
    """
    if fp_a.get("layer_keys") != fp_b.get("layer_keys"):
        return 0.0

    sa, sb = fp_a.get("tensor_samples"), fp_b.get("tensor_samples")
    if sa and sb and len(sa) == len(sb):
        unchanged = 0
        for a, b in zip(sa, sb):
            cos = _vector_cosine(a, b)
            if (not any(a) and not any(b)) or cos >= _UNCHANGED_COSINE:
                unchanged += 1
        return unchanged / len(sa)

    return _vector_cosine(fp_a.get("norm_vector", []), fp_b.get("norm_vector", []))

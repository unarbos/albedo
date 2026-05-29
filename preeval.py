"""Tensor-level model fingerprinting and near-duplicate detection.

EVAL-SERVER ONLY — import from eval.py only, never from validator.py.
Fingerprints are computed from raw safetensors weight files which are only
materialized on the eval (GPU) machine. The validator never touches weights.

Two state files on Hippius S3:

  uploaded_models_state.json  — human-readable model registry
    { "models": { "repo@sha256:...": { repo, digest, hotkey, commit_block,
      sha256_bytes, evaluated_at, verdict, fingerprint_method,
      layer_keys, norm_vector } } }

  models_tensor_state.json  — mechanical tensor data (not human-readable)
    { "tensors": { "repo@sha256:...": { hotkey, tensor_samples } } }

Both are loaded at eval-server startup into separate in-memory caches and
updated together after each duel. tensor_samples carries K=16 deterministic
weight samples per tensor (blake2b-derived indices). cosine_similarity()
returns the fraction of tensors whose per-sample cosine ≥ 1 − 1e-6
(direction-sensitive, robust against genuine fine-tuning).

Only models with Hippius OCI digests (sha256:...) are fingerprinted.
HF-backed genesis kings (hf:...) are skipped.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger("albedo.preeval")

FINGERPRINT_METHOD = "layer_norms_v2_with_samples"
SAMPLE_K = 16
MODELS_STATE_KEY  = "uploaded_models_state.json"
TENSOR_STATE_KEY  = "models_tensor_state.json"

_UNKNOWN_BLOCK = -1  # sentinel: commit block not recorded


def _deterministic_indices(key: str, n: int, k: int) -> list[int]:
    if n <= 0:
        return [0] * k
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=4 * k).digest()
    return [int.from_bytes(h[i * 4:(i + 1) * 4], "big") % n for i in range(k)]


# Module-level cached boto3 client — created lazily on first use.
_S3_CLIENT: object | None = None


def _get_or_create_s3_client(endpoint: str, access: str, secret: str) -> object:
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3
        from botocore.config import Config as BotoConfig

        _S3_CLIENT = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            region_name="decentralized",
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=15,
                read_timeout=120,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
    return _S3_CLIENT


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

def compute_fingerprint(model_dir: Path | str) -> dict:
    """Compute v2 fingerprint: per-layer L2 norms + K deterministic samples.

    Returns a dict with keys:
      fingerprint_method, sha256_bytes, layer_keys, norm_vector, tensor_samples
    """
    from safetensors import safe_open

    from model_store import sha256_safetensors

    model_dir = Path(model_dir)
    layer_norms: dict[str, float] = {}
    layer_samples: dict[str, list[float]] = {}

    try:
        import torch as _torch
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
        _framework = "pt"

        def _norm(t) -> float:
            return float(t.to(device=_device, dtype=_torch.float32).norm().item())

        def _samples(t, key: str) -> list[float]:
            flat = t.flatten()
            n = int(flat.shape[0])
            idxs = _deterministic_indices(key, n, SAMPLE_K)
            picked = flat[_torch.tensor(idxs, dtype=_torch.long)].to(_torch.float32)
            return [float(x.item()) for x in picked]

    except ImportError:
        _framework = "numpy"

        def _norm(t) -> float:
            return float(np.linalg.norm(t.astype(np.float32)))

        def _samples(t, key: str) -> list[float]:
            flat = np.asarray(t).reshape(-1)
            n = int(flat.shape[0])
            idxs = _deterministic_indices(key, n, SAMPLE_K)
            return [float(flat[i]) for i in idxs]

    for sf_path in sorted(model_dir.rglob("*.safetensors")):
        with safe_open(str(sf_path), framework=_framework) as f:
            for key in sorted(f.keys()):
                t = f.get_tensor(key)
                layer_norms[key] = _norm(t)
                layer_samples[key] = _samples(t, key)

    if not layer_norms:
        raise ValueError(f"No *.safetensors files found under {model_dir}")

    keys = sorted(layer_norms.keys())
    return {
        "fingerprint_method": FINGERPRINT_METHOD,
        "sha256_bytes": sha256_safetensors(model_dir),
        "layer_keys": keys,
        "norm_vector": [layer_norms[k] for k in keys],
        "tensor_samples": [layer_samples[k] for k in keys],
    }


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(fp_a: dict, fp_b: dict,
                      tensor_samples_a: list | None = None,
                      tensor_samples_b: list | None = None) -> float:
    """Fraction of unchanged tensors between two fingerprints.

    A tensor is unchanged when its per-sample cosine ≥ 1 − 1e-6.
    tensor_samples_a / tensor_samples_b can be passed separately (from the
    tensor state cache) when they are not embedded in fp_a / fp_b.

    Falls back to v1 layer-norm cosine when samples are unavailable on
    either side — for backward-compat with old stored entries.

    Returns 0.0 if layer_keys differ (different architecture).
    """
    if fp_a.get("layer_keys") != fp_b.get("layer_keys"):
        return 0.0

    sa = tensor_samples_a or fp_a.get("tensor_samples")
    sb = tensor_samples_b or fp_b.get("tensor_samples")

    if sa and sb and len(sa) == len(sb):
        n_total = len(sa)
        n_unchanged = 0
        for a_samp, b_samp in zip(sa, sb):
            a = np.asarray(a_samp, dtype=np.float64)
            b = np.asarray(b_samp, dtype=np.float64)
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= 0.0:
                n_unchanged += 1  # zero-vectors (uninitialised biases) trivially match
                continue
            if float(np.dot(a, b) / denom) >= 1.0 - 1e-6:
                n_unchanged += 1
        return n_unchanged / n_total

    # v1 fallback: layer-norm cosine.
    va = np.array(fp_a["norm_vector"], dtype=np.float64)
    vb = np.array(fp_b["norm_vector"], dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ---------------------------------------------------------------------------
# Duplicate check
# ---------------------------------------------------------------------------

def check_duplicate(
    fingerprint: dict,
    state: dict,
    threshold: float,
    skip_key: str | None = None,
    commit_block: int = _UNKNOWN_BLOCK,
    tensor_state: dict | None = None,
) -> tuple[bool, str | None]:
    """Check if a fingerprint is too similar to any stored model.

    tensor_state: the models_tensor_state dict (separate from state). When
    provided, per-tensor samples are looked up from it for stored entries so
    the direction-sensitive metric is used. Without it, falls back to v1 norms.

    Priority: lower commit_block = original. Unknown blocks (-1) never skip.
    """
    challenger_sha256 = fingerprint.get("sha256_bytes", "")
    challenger_samples = fingerprint.get("tensor_samples")
    tensors = (tensor_state or {}).get("tensors", {})

    for ref_key, entry in state.get("models", {}).items():
        if skip_key and ref_key == skip_key:
            continue
        stored_block: int = entry.get("commit_block", _UNKNOWN_BLOCK)
        if stored_block is None:
            stored_block = _UNKNOWN_BLOCK
        if commit_block > 0 and stored_block > 0 and commit_block <= stored_block:
            continue
        stored_sha256 = entry.get("sha256_bytes", "")
        if challenger_sha256 and stored_sha256 and challenger_sha256 == stored_sha256:
            log.info(
                "exact-hash duplicate: challenger (block=%d) matches %s (block=%d)",
                commit_block, ref_key, stored_block,
            )
            return True, ref_key
        stored_fp = {
            "layer_keys": entry.get("layer_keys", []),
            "norm_vector": entry.get("norm_vector", []),
        }
        stored_samples = tensors.get(ref_key, {}).get("tensor_samples")
        sim = cosine_similarity(fingerprint, stored_fp,
                                tensor_samples_a=challenger_samples,
                                tensor_samples_b=stored_samples)
        if sim >= threshold:
            log.info(
                "near-duplicate: similarity=%.6f >= threshold=%.4f "
                "challenger (block=%d) matches %s (block=%d)",
                sim, threshold, commit_block, ref_key, stored_block,
            )
            return True, ref_key
    return False, None


# ---------------------------------------------------------------------------
# S3 state I/O — models (human-readable)
# ---------------------------------------------------------------------------

def _empty_models_state() -> dict:
    return {"version": 1, "updated_at": None, "models": {}}


def _empty_tensor_state() -> dict:
    return {"version": 1, "updated_at": None, "tensors": {}}


def _s3_get_json(s3_client, bucket: str, key: str) -> dict | None:
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)  # type: ignore[attr-defined]
        return json.loads(obj["Body"].read())
    except Exception as exc:
        code = ""
        try:
            code = exc.response["Error"]["Code"]  # type: ignore[attr-defined]
        except Exception:
            pass
        if code != "NoSuchKey":
            log.warning("Could not load s3://%s/%s: %s", bucket, key, exc)
        return None


def _s3_put_json(s3_client, bucket: str, key: str, data: dict) -> None:
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    body = json.dumps(data, ensure_ascii=False, indent=2).encode()
    s3_client.put_object(  # type: ignore[attr-defined]
        Bucket=bucket, Key=key, Body=body,
        ContentType="application/json",
        CacheControl="public, max-age=60",
    )


def load_models_state(s3_client, bucket: str, key: str = MODELS_STATE_KEY) -> dict:
    """Download uploaded_models_state.json (metadata only, no tensor_samples)."""
    return _s3_get_json(s3_client, bucket, key) or _empty_models_state()


def save_models_state(s3_client, bucket: str, state: dict,
                      key: str = MODELS_STATE_KEY) -> None:
    """Upload uploaded_models_state.json."""
    _s3_put_json(s3_client, bucket, key, state)
    log.info("saved models state (%d entries) to s3://%s/%s",
             len(state.get("models", {})), bucket, key)


def load_tensor_state(s3_client, bucket: str, key: str = TENSOR_STATE_KEY) -> dict:
    """Download models_tensor_state.json (tensor_samples arrays)."""
    return _s3_get_json(s3_client, bucket, key) or _empty_tensor_state()


def save_tensor_state(s3_client, bucket: str, tensor_state: dict,
                      key: str = TENSOR_STATE_KEY) -> None:
    """Upload models_tensor_state.json."""
    _s3_put_json(s3_client, bucket, key, tensor_state)
    log.info("saved tensor state (%d entries) to s3://%s/%s",
             len(tensor_state.get("tensors", {})), bucket, key)


# ---------------------------------------------------------------------------
# Local state I/O (for testing / seed scripts)
# ---------------------------------------------------------------------------

def load_models_state_local(path: Path | str) -> dict:
    p = Path(path)
    if not p.exists():
        return _empty_models_state()
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_models_state_local(state: dict, path: Path | str) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    log.info("saved models state (%d entries) to %s", len(state.get("models", {})), p)


def load_tensor_state_local(path: Path | str) -> dict:
    p = Path(path)
    if not p.exists():
        return _empty_tensor_state()
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_tensor_state_local(tensor_state: dict, path: Path | str) -> None:
    tensor_state["updated_at"] = datetime.now(timezone.utc).isoformat()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(tensor_state, f, ensure_ascii=False, indent=2)
    log.info("saved tensor state (%d entries) to %s",
             len(tensor_state.get("tensors", {})), p)


# ---------------------------------------------------------------------------
# State mutation
# ---------------------------------------------------------------------------

def add_fingerprint_to_state(
    state: dict,
    tensor_state: dict,
    ref_key: str,
    fingerprint: dict,
    *,
    hotkey: str,
    verdict: str,
    repo: str = "",
    digest: str = "",
    commit_block: int = _UNKNOWN_BLOCK,
) -> tuple[dict, dict]:
    """Insert or overwrite a model entry across both state dicts.

    uploaded_models_state.json gets the human-readable metadata + norms.
    models_tensor_state.json gets the mechanical tensor_samples arrays.

    verdict: "king" | "accepted" | "rejected" | "invalid"
      "invalid" = duplicate or injection — fingerprint kept for instant
      re-detection via SHA256 short-circuit on future re-submissions.

    Returns (updated_state, updated_tensor_state).
    """
    # Thread-safety: this function is only called while STATE.eval_lock is held
    # in eval.py (single duel at a time), so no concurrent mutation is possible.
    state.setdefault("models", {})[ref_key] = {
        "repo": repo,
        "digest": digest,
        "hotkey": hotkey,
        "commit_block": commit_block,
        "sha256_bytes": fingerprint.get("sha256_bytes", ""),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "fingerprint_method": fingerprint.get("fingerprint_method", FINGERPRINT_METHOD),
        "layer_keys": fingerprint.get("layer_keys", []),
        "norm_vector": fingerprint.get("norm_vector", []),
    }
    samples = fingerprint.get("tensor_samples")
    if samples is not None:
        tensor_state.setdefault("tensors", {})[ref_key] = {
            "hotkey": hotkey,
            "tensor_samples": samples,
        }
    return state, tensor_state

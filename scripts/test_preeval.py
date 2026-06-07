"""Preeval duplicate-detection manual test.

Edit MODEL_A and MODEL_B at the top, then run:
    python scripts/test_preeval.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── EDIT THESE ────────────────────────────────────────────────────────────────
MODEL_A = Path("/home/const/similarity_test/king_current")
MODEL_B = Path("/home/const/similarity_test/king_prev_3")
THRESHOLD = 0.95
STATE_OUT = Path("./uploaded_models_state.json")
# ─────────────────────────────────────────────────────────────────────────────


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("test_preeval")

# ---------------------------------------------------------------------------
# Inline fingerprinting (no import of preeval needed — standalone)
# ---------------------------------------------------------------------------

def sha256_of_safetensors(model_dir: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted(model_dir.rglob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


SAMPLE_K = 16


def _deterministic_indices(key: str, n: int, k: int) -> list[int]:
    """K well-distributed indices in [0, n) derived from blake2b(key).
    Same tensor key + same numel always samples the same positions, so
    weights that are byte-identical produce byte-identical samples."""
    import hashlib
    if n <= 0:
        return [0] * k
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=4 * k).digest()
    return [int.from_bytes(h[i * 4:(i + 1) * 4], "big") % n for i in range(k)]


def compute_fingerprint(model_dir: Path) -> dict:
    from safetensors import safe_open

    # Use torch backend — numpy doesn't support bfloat16 which is standard
    # for LLM weights. Torch is available via vLLM/transformers anyway.
    try:
        import torch
        _framework = "pt"

        def _norm(t) -> float:
            return float(t.to(torch.float32).norm().item())

        def _samples(t, key: str) -> list[float]:
            flat = t.flatten()
            idxs = _deterministic_indices(key, int(flat.shape[0]), SAMPLE_K)
            picked = flat[torch.tensor(idxs, dtype=torch.long)].to(torch.float32)
            return [float(x.item()) for x in picked]
    except ImportError:
        _framework = "numpy"

        def _norm(t) -> float:
            return float(np.linalg.norm(t.astype(np.float32)))

        def _samples(t, key: str) -> list[float]:
            flat = np.asarray(t).reshape(-1)
            idxs = _deterministic_indices(key, int(flat.shape[0]), SAMPLE_K)
            return [float(flat[i]) for i in idxs]

    layer_norms: dict[str, float] = {}
    layer_samples: dict[str, list[float]] = {}
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
        "fingerprint_method": "layer_norms_v2_with_samples",
        "sha256_bytes": sha256_of_safetensors(model_dir),
        "layer_keys": keys,
        "norm_vector": [layer_norms[k] for k in keys],
        "tensor_samples": [layer_samples[k] for k in keys],
    }


def cosine_similarity(fp_a: dict, fp_b: dict) -> float:
    """Fraction of essentially-unchanged tensors between A and B.

    For each shared tensor key, computes the cosine of K deterministic
    weight samples (direction-sensitive — unlike the L2 norm, which barely
    moves during fine-tuning because gradient steps are nearly orthogonal
    to the weight vector). A tensor with per-sample cosine ≥ 1 - 1e-6 is
    treated as unchanged. Returns n_unchanged / n_total ∈ [0, 1].

      identical models  → 1.0   (every tensor unchanged → flagged duplicate)
      near-copy + noise → ≈ 1.0 (samples shift in lockstep → still flagged)
      genuine fine-tune → < 1.0 by the fraction of tensors that actually moved
      different arch    → 0.0
    """
    if fp_a["layer_keys"] != fp_b["layer_keys"]:
        return 0.0

    samples_a = fp_a.get("tensor_samples")
    samples_b = fp_b.get("tensor_samples")
    if not (samples_a and samples_b and len(samples_a) == len(samples_b)):
        # v1 fallback: layer-norm cosine (used only against legacy state).
        va = np.array(fp_a["norm_vector"], dtype=np.float64)
        vb = np.array(fp_b["norm_vector"], dtype=np.float64)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom else 0.0

    n_total = len(samples_a)
    n_unchanged = 0
    for sa, sb in zip(samples_a, samples_b):
        a = np.asarray(sa, dtype=np.float64)
        b = np.asarray(sb, dtype=np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0.0:
            n_unchanged += 1   # zero-vectors trivially match
            continue
        c = float(np.dot(a, b) / denom)
        if c >= 1.0 - 1e-6:
            n_unchanged += 1
    return n_unchanged / n_total


def save_state(state: dict, path: Path) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate paths
    for label, path in [("MODEL_A", MODEL_A), ("MODEL_B", MODEL_B)]:
        if not path.exists():
            log.error("%s path does not exist: %s", label, path)
            sys.exit(1)
        sf = list(path.rglob("*.safetensors"))
        if not sf:
            log.error("%s: no *.safetensors files under %s", label, path)
            sys.exit(1)
        log.info("%s: %d safetensors file(s)", label, len(sf))

    # Compute fingerprints
    log.info("Computing fingerprint for MODEL_A …")
    fp_a = compute_fingerprint(MODEL_A)
    log.info("  layers=%d  sha256=%s…", len(fp_a["layer_keys"]), fp_a["sha256_bytes"][:16])

    log.info("Computing fingerprint for MODEL_B …")
    fp_b = compute_fingerprint(MODEL_B)
    log.info("  layers=%d  sha256=%s…", len(fp_b["layer_keys"]), fp_b["sha256_bytes"][:16])

    # Similarity
    sim = cosine_similarity(fp_a, fp_b)
    exact = fp_a["sha256_bytes"] == fp_b["sha256_bytes"]
    is_dup = exact or sim >= THRESHOLD

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  MODEL_A           : {MODEL_A}")
    print(f"  MODEL_B           : {MODEL_B}")
    print(f"  layers (A/B)      : {len(fp_a['layer_keys'])} / {len(fp_b['layer_keys'])}")
    print(f"  sha256_A          : {fp_a['sha256_bytes'][:40]}…")
    print(f"  sha256_B          : {fp_b['sha256_bytes'][:40]}…")
    print(f"  exact_bytes_match : {exact}")
    print(f"  cosine_similarity : {sim:.10f}")
    print(f"  threshold         : {THRESHOLD}")
    print(f"  DUPLICATE?        : {'YES  <-- flagged' if is_dup else 'NO'}")
    print(f"{sep}\n")

    # Build / update state JSON and save locally
    state: dict = {"version": 1, "updated_at": None, "models": {}}
    if STATE_OUT.exists():
        try:
            state = json.loads(STATE_OUT.read_text())
            log.info("Loaded existing state from %s (%d entries)", STATE_OUT, len(state.get("models", {})))
        except Exception as exc:
            log.warning("Could not load existing state, starting fresh: %s", exc)

    now = datetime.now(timezone.utc).isoformat()

    state.setdefault("models", {})[f"model_a@sha256:{fp_a['sha256_bytes']}"] = {
        "repo": "local/model_a",
        "digest": f"sha256:{fp_a['sha256_bytes']}",
        "hotkey": "",
        "sha256_bytes": fp_a["sha256_bytes"],
        "evaluated_at": now,
        "verdict": "king",
        **{k: fp_a[k] for k in ("fingerprint_method", "layer_keys", "norm_vector")},
    }
    state["models"][f"model_b@sha256:{fp_b['sha256_bytes']}"] = {
        "repo": "local/model_b",
        "digest": f"sha256:{fp_b['sha256_bytes']}",
        "hotkey": "",
        "sha256_bytes": fp_b["sha256_bytes"],
        "evaluated_at": now,
        "verdict": "duplicate" if is_dup else "accepted",
        **{k: fp_b[k] for k in ("fingerprint_method", "layer_keys", "norm_vector")},
    }

    save_state(state, STATE_OUT)
    log.info("State saved to: %s  (%d total entries)", STATE_OUT.resolve(), len(state["models"]))

    sys.exit(1 if is_dup else 0)


if __name__ == "__main__":
    main()
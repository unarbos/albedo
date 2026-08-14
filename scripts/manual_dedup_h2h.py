#!/usr/bin/env python3

WORKERS = 16

MODELS = ["path"]

DOWNLOAD_REPO = ""
DOWNLOAD_DIR = "/root/models"
HF_TOKEN = ""

FRAC_MIN = 0.012
HET_MIN = 0.15
KURT_MIN = 1.0

import json
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

_NP = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8"}
_ITEM = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
_CHUNK = 8_000_000


def _is_vision(key: str) -> bool:
    return ".visual." in key or key.startswith("visual.")


def bf16_codes(raw: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return raw.view("<u2") if raw.dtype != np.uint16 else raw
    u = raw.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def decode_bf16(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def tensor_index(model_dir: str) -> dict:
    d = Path(model_dir)
    idx_file = d / "model.safetensors.index.json"
    if idx_file.exists():
        wm = json.loads(idx_file.read_text())["weight_map"]
        shards = sorted(set(wm.values()))
    else:
        shards = [p.name for p in d.glob("*.safetensors")]
    out = {}
    for shard in shards:
        p = d / shard
        with open(p, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        base = 8 + hlen
        for key, info in header.items():
            if key == "__metadata__" or info.get("dtype") not in _NP or _is_vision(key):
                continue
            s, e = info["data_offsets"]
            out[key] = (str(p), info["dtype"], base + s, base + e)
    return out


def _compare_tensor(args):
    key, spec_a, spec_b = args
    pa, da, sa, _ = spec_a
    pb, db, sb, _ = spec_b
    n = (spec_a[3] - spec_a[2]) // _ITEM[da]
    if n != (spec_b[3] - spec_b[2]) // _ITEM[db]:
        return None
    changed = 0
    S1 = S2 = S3 = S4 = Sb2 = 0.0
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        for off in range(0, n, _CHUNK):
            m = min(_CHUNK, n - off)
            fa.seek(sa + off * _ITEM[da])
            ca = bf16_codes(np.frombuffer(fa.read(m * _ITEM[da]), _NP[da]), da)
            fb.seek(sb + off * _ITEM[db])
            cb = bf16_codes(np.frombuffer(fb.read(m * _ITEM[db]), _NP[db]), db)
            changed += int(np.count_nonzero(ca != cb))
            a = decode_bf16(ca).astype(np.float64)
            b = decode_bf16(cb).astype(np.float64)
            d = a - b
            d2 = d * d
            S1 += d.sum()
            S2 += d2.sum()
            S3 += (d2 * d).sum()
            S4 += (d2 * d2).sum()
            Sb2 += (b * b).sum()
    mu = S1 / n
    m2 = S2 / n - mu * mu
    rel = float(np.sqrt(S2 / Sb2)) if Sb2 > 0 else 0.0
    if m2 > 0:
        m4 = S4 / n - 4.0 * mu * S3 / n + 6.0 * mu * mu * S2 / n - 3.0 * mu**4
        kurt = float(m4 / (m2 * m2) - 3.0)
    else:
        kurt = 0.0
    has_delta = changed > 0 and m2 > 0
    return {"n": n, "changed": changed, "rel": rel, "kurt": kurt, "has_delta": has_delta}


def verdict(frac, het, kurt):
    if frac < FRAC_MIN:
        return "COPY", f"only {frac:.3%} of weights changed (no real training)"
    if het < HET_MIN:
        return "COPY", f"rel-uniform change het_cv={het:.3f} (code-noise, not training)"
    if kurt < KURT_MIN:
        return "COPY", f"structureless delta kurt={kurt:.2f} (gaussian noise, not training)"
    return "DISTINCT", f"real finetune (frac {frac:.1%}, het {het:.2f}, kurt {kurt:.1f})"


def compare(name_a, idx_a, name_b, idx_b, pool):
    keys = sorted(k for k in idx_a if k in idx_b)
    if not keys:
        print(f"\n{name_a}  vs  {name_b}\n  → DIFFERENT ARCHITECTURE (no shared tensors)")
        return
    results = [r for r in pool.map(_compare_tensor, [(k, idx_a[k], idx_b[k]) for k in keys]) if r]
    total = sum(r["n"] for r in results)
    changed = sum(r["changed"] for r in results)
    frac = changed / total if total else 0.0
    rels = np.array([r["rel"] for r in results if r["has_delta"] and r["rel"] > 0])
    kurts = [r["kurt"] for r in results if r["has_delta"]]
    het = float(rels.std() / rels.mean()) if len(rels) and rels.mean() else 0.0
    kurt = float(np.median(kurts)) if kurts else 0.0
    label, reason = verdict(frac, het, kurt)
    print(f"\n{name_a}  vs  {name_b}")
    print(
        f"  tensors={len(results)} weights={total / 1e9:.2f}B  "
        f"frac_changed={frac:.4%}  het_cv={het:.3f}  kurt={kurt:.2f}"
    )
    print(f"  → {label}: {reason}")


def maybe_download():
    if not DOWNLOAD_REPO:
        return
    from huggingface_hub import snapshot_download

    repo, _, rev = DOWNLOAD_REPO.partition("@")
    token = HF_TOKEN or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dest = Path(DOWNLOAD_DIR) / repo.replace("/", "__")
    print(f"downloading {DOWNLOAD_REPO} → {dest} ...")
    path = snapshot_download(
        repo_id=repo,
        revision=rev or None,
        token=token,
        local_dir=str(dest),
        ignore_patterns=["*.md", ".gitattributes", "LICENSE"],
    )
    MODELS.append(path)


def main():
    maybe_download()
    models = [
        m
        for m in MODELS
        if Path(m, "model.safetensors.index.json").exists() or list(Path(m).glob("*.safetensors"))
    ]
    if len(models) < 2:
        sys.exit("need at least 2 valid model dirs in MODELS")
    print(f"indexing {len(models)} model(s) ...")
    idx = {m: tensor_index(m) for m in models}
    for m in models:
        print(f"  {Path(m).name:20} {len(idx[m])} float tensors")
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for a, b in combinations(models, 2):
            compare(Path(a).name, idx[a], Path(b).name, idx[b], pool)


if __name__ == "__main__":
    main()

# Albedo

Trajectory-distillation **king-of-the-hill** subnet on Bittensor. Miners upload model challengers; the reigning king and challenger duel on pinned [SWE-ZERO](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories) rollouts, scored by an ensemble of LLM judges. Full duel traces are published for downstream distillation.

**Live dashboard:** https://us-east-1.hippius.com/albedo/index.html  
**Machine-readable state:** https://us-east-1.hippius.com/albedo/dashboard.json  
**Subnet:** `ALBEDO_NETUID=97` (finney)

---

## For LLM agents

If you are an autonomous agent helping a user mine or operate on this subnet, read this section first.

### What success looks like

| Role | Goal | Primary script |
|------|------|----------------|
| **Miner** | Upload a challenger that beats the king on the judge duel | `miner.py` |
| **Distiller** | Download published duel traces and export SFT/DPO data | `scripts/export_training_jsonl.py` |
| **Validator operator** | Scan reveals, dispatch duels, set weights | `validator.py` + `eval.py` |
| **Eval operator** | Run GPU duel server | `scripts/start_eval.sh` |

### Decision tree

1. **Check current king** — `GET https://us-east-1.hippius.com/albedo/dashboard.json` → `king.model_repo`, `king.king_digest`.
2. **Mining?** — User needs: registered hotkey on SN97, Hippius **push** credentials, unlocked Bittensor wallet, a trained challenger (not default noise).
3. **Training data only?** — Browse `https://us-east-1.hippius.com/albedo/evals/`; no wallet required.
4. **Running infrastructure?** — Needs GPU box (`eval.py`), Chutes API key, Hippius Hub token, pinned dataset shard.

### Single source of truth

All duel rules, judge models, dataset pin, and arch locks live in [`chain.toml`](chain.toml). Code loads them via `chain_config.py`. Override for experiments: `ALBEDO_CHAIN_OVERRIDE=/path/to/alt.toml`.

### Do not guess these

- **Repo naming:** must match `^[^/]+/albedo-mini-1.7b-.+$` (lowercase on Hippius).
- **Reveal format:** v4 only — `v4|<repo>|<digest>|<hotkey>`.
- **Dethrone rule:** win on ≥1 judge dimension (mean delta > `tie_band`), lose on none (ties OK elsewhere).
- **One eval per hotkey** — second reveal from same hotkey is ignored.
- **Current king cannot challenge** as themselves.

---

## Quick start

```bash
git clone https://github.com/unarbos/albedo.git
cd albedo
uv venv && source .venv/bin/activate
uv pip install -e .
```

---

## Mining

### Prerequisites

- Hotkey registered on subnet 97 (`btcli subnet register --netuid 97`)
- Bittensor wallet with coldkey unlock: `BT_WALLET_NAME`, `BT_WALLET_PASSWORD`
- Hippius Hub **push** auth (S3/read-only creds are **not** enough for uploads)
- A model that actually improves on the king (default `--noise 0.001` is a pipeline smoke test only)

### Hippius setup (one-time)

```bash
hippius-hub login --hippius-token <console-token-from-console.hippius.com>
hippius-hub registry me
hippius-hub registry provision <your-namespace> --docker-login
hippius-hub registry keys create my-miner --role push
# Save the secret immediately — shown once.
hippius-hub login --username 'robot$<namespace>+my-miner' --password '<push-secret>'
```

Verify push works:

```bash
hippius-hub upload <namespace>/albedo-mini-1.7b-test ./tiny-safetensors-folder
```

### Submit a challenger

```bash
cd albedo && source .venv/bin/activate

export ALBEDO_NETUID=97
export BT_WALLET_NAME=<wallet>
export BT_WALLET_PASSWORD=<coldkey-password>
export ALBEDO_CHALLENGER_NAMESPACE=<your-hippius-namespace>
export ALBEDO_CHALLENGER_REPO_NAME=albedo-mini-1.7b   # must be lowercase
export ALBEDO_HIPPIUS_USERNAME='robot$<namespace>+<push-key>'
export ALBEDO_HIPPIUS_PASSWORD=<push-secret>

python miner.py --hotkey <hotkey_name>
```

Optional flags: `--noise 0.001` (stub perturbation), `--force` (override duplicate-reveal warning).

**HF smoke test** (not production):

```bash
export ALBEDO_UPLOAD_BACKEND=hf
export ALBEDO_CHALLENGER_NAMESPACE=<your-hf-user>
export HF_TOKEN=...
python miner.py --hotkey <hotkey_name> --noise 0.001
```

### What `miner.py` does

1. Fetches `dashboard.json` for current king (falls back to genesis seed in `chain.toml`)
2. Downloads king weights via `model_store.materialize_model`
3. Runs `train_or_perturb` — **replace this with your SFT/RL loop for real mining**
4. Validates `config.json` arch locks match the king
5. Uploads to Hippius Hub → `namespace/albedo-mini-1.7b-<suffix>@sha256:…`
6. Posts on-chain v4 reveal: `v4|<repo>|<digest>|<author_hotkey>`

### Mining gotchas

| Issue | Fix |
|-------|-----|
| `401` on Hippius upload | Use lowercase repo path (`albedo-mini-1.7b-…`, not `Albedo-Mini-1.7B-…`) |
| Upload works but reveal ignored | Wait ~30s after commit; check you're not the current king |
| `REJECTED` with default noise | Expected — proves pipeline works; train a real challenger |
| Stale shared push creds | Set `ALBEDO_HIPPIUS_*`; it overrides `HIPPIUS_HUB_*` |
| Config rejected | Match king arch locks in `chain.toml [arch].extra_lock_keys`; no `*.py` or `auto_map` in Hub snapshot |

### Monitor your submission

Poll `dashboard.json`:

- `queue` — challenger waiting for eval
- `current_eval` — duel in progress
- `history` — verdict (`accepted` / `rejected`), judge breakdown, `evals_url`

Typical timeline: upload ~3 min (1.7B) → reveal → validator fetch ~20s → eval ~60–90s.

---

## Using the subnet (distillation & research)

Every duel publishes **full turn-by-turn traces** — not just scores.

| Resource | URL |
|----------|-----|
| Eval traces | https://us-east-1.hippius.com/albedo/evals/ |
| Daily manifest | `evals/YYYY-MM-DD/manifest.jsonl` |
| Per-duel file | `evals/YYYY-MM-DD/<eval_id>.jsonl.gz` |

### Trace schema (`schema_version: 1`)

- **`duel_meta`** — king, challenger, hotkey, seed, judges, generation params
- **`turn`** — `messages_prefix` (dataset), `messages_prompt` (what vLLM saw), king/challenger replies, per-judge verdict + rationale
- **`verdict`** — final accept/reject and aggregate stats

### Export training data

```bash
# SFT (assistant completions from winning side)
python scripts/export_training_jsonl.py \
  --input https://us-east-1.hippius.com/albedo/evals/2026-05-26/<eval_id>.jsonl.gz \
  --output training.jsonl

# DPO preference pairs
python scripts/export_training_jsonl.py --input <url-or-path> --output pairs.jsonl --format dpo
```

---

## How duels work

### Genesis king

`ricdomolm/mini-coder-1.7b@hf:ea686024d9522260933aeed436e9939b1912ca15` — Qwen3 1.7B, same base model that generated the SWE-ZERO corpus.

Challengers upload to Hippius as `sha256:` OCI manifests under `namespace/albedo-mini-1.7b-<suffix>`.

### Judges (from `chain.toml`)

Each duel runs **every** model below as an independent judge dimension:

- `deepseek-ai/DeepSeek-V3.2-TEE`
- `Qwen/Qwen3-235B-A22B-Thinking-2507`
- `moonshotai/Kimi-K2.6-TEE`

Judges run via [Chutes](https://llm.chutes.ai) (`CHUTES_API_KEY`, default base `https://llm.chutes.ai/v1`).

### Dethrone rule (`tie_band = 0.01`)

Per judge model, comparing mean scores across sampled turns:

- **Win:** mean(challenger − king) > `tie_band`
- **Lose:** mean delta < −`tie_band`
- **Tie:** otherwise

**Crown** when: ≥1 judge win **and** zero judge losses (ties on remaining dimensions are fine).

### Duel parameters (`chain.toml [duel]`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_samples` | 32 | Trajectories sampled per duel |
| `max_turns_per_sample` | 10 | Turns per trajectory |
| `gen_temperature` | 1.0 | Matches dataset generation |
| `gen_max_tokens` | 1024 | Per-side completion budget |
| `king_chain_depth` | 5 | Rolling emission split across current + 4 prior kings |

Sampling seed: `blake2b(block_hash || hotkey)` — reproducible given chain state.

### Emission (king chain)

Kings are numbered ALBEDO-I, ALBEDO-II, … Emission splits equally across the current king and up to four dethroned kings still registered on the metagraph.

---

## Models & auth

Weights are immutable **`namespace/repo@digest`** refs:

| Digest prefix | Source |
|---------------|--------|
| `sha256:<64hex>` | Hippius Hub OCI (challenger uploads) |
| `hf:<40hex>` | HuggingFace commit (genesis / HF fallback) |

### Download in Python

```python
from model_store import ModelRef, materialize_model

ref = ModelRef("your-ns/albedo-mini-1.7b-miner0", "sha256:...")
path = materialize_model(ref, local_dir="/tmp/challenger", max_workers=16)
```

### Auth env vars

| Purpose | Env vars |
|---------|----------|
| Hippius push/pull (miners) | `ALBEDO_HIPPIUS_USERNAME` + `ALBEDO_HIPPIUS_PASSWORD` |
| Hippius Hub token (eval/validator) | `HIPPIUS_HUB_TOKEN` or `HIPPIUS_HUB_USERNAME` + `HIPPIUS_HUB_PASSWORD` |
| HF pulls | `HF_TOKEN` or `HUGGINGFACE_API_KEY` |
| Chutes judges | `CHUTES_API_KEY` |
| S3 eval traces / dashboard | `ALBEDO_EVALS_S3_*`, `ALBEDO_R2_*` (operators only) |

`ALBEDO_HIPPIUS_*` takes precedence over `HIPPIUS_HUB_*` when both are set.

---

## Operating infrastructure

Two processes: **validator** (chain + orchestration) and **eval server** (GPU + vLLM + judges).

### Eval server (GPU box)

```bash
# 1. Prefetch pinned dataset shard
python scripts/prefetch_dataset.py --out /var/albedo/dataset
export ALBEDO_DATASET_SHARD_PATH=/var/albedo/dataset/train-00000.parquet

# 2. Start eval API
export CHUTES_API_KEY=...
export HIPPIUS_HUB_TOKEN=...
export ALBEDO_EVALS_S3_BUCKET=albedo
export ALBEDO_EVALS_S3_ACCESS_KEY=...
export ALBEDO_EVALS_S3_SECRET_KEY=...
bash scripts/start_eval.sh   # uvicorn eval:app on port 9001
```

`eval.py` refuses to start if the on-disk shard SHA256 ≠ `chain.toml [dataset].shard_sha256`.

### Validator

```bash
export ALBEDO_NETUID=97
export ALBEDO_EVAL_SERVER=http://localhost:9001   # or SSH tunnel via tunnel.sh
export BT_WALLET_NAME=...
export BT_WALLET_HOTKEY=...
export ALBEDO_R2_ACCESS_KEY=... ALBEDO_R2_SECRET_KEY=...
export ALBEDO_DS_ACCESS_KEY=... ALBEDO_DS_SECRET_KEY=...

python validator.py
# or: pm2 start ecosystem.config.js
```

Validator loop: scan v4 reveals → config gate → `POST /eval` (SSE) → crown king on accept → `set_weights` → flush dashboard.

---

## Repository layout

```
chain.toml              # judges, dataset pin, duel knobs, arch locks
chain_config.py         # loads chain.toml
miner.py                # challenger build + upload + reveal
validator.py            # reveal scanner, duel dispatch, weights
eval.py                 # GPU eval server (vLLM + judges)
judge.py                # Chutes LLM-as-judge client
model_store.py          # Hippius/HF download, upload, v4 reveal helpers
trajectory_sampler.py   # SWE-ZERO parquet sampling
scripts/
  prefetch_dataset.py   # download pinned eval shard
  export_training_jsonl.py
  seed_genesis.py       # one-shot genesis king upload
  start_eval.sh         # eval launcher
  smoke_eval.py
ecosystem.config.js     # PM2 manifest (validator + SSH tunnel)
tunnel.sh               # SSH port-forward to eval box
website/                # static dashboard UI
archs/qwen3_minicoder/  # transformers shim for Qwen3 1.7B arch
```

---

## Environment reference

### Miners

| Variable | Required | Description |
|----------|----------|-------------|
| `ALBEDO_NETUID` | yes | Subnet ID (`97`) |
| `BT_WALLET_NAME` | yes | Bittensor wallet name |
| `BT_WALLET_PASSWORD` | yes | Coldkey decrypt password |
| `ALBEDO_CHALLENGER_NAMESPACE` | yes | Hippius namespace |
| `ALBEDO_CHALLENGER_REPO_NAME` | yes | `albedo-mini-1.7b` (lowercase) |
| `ALBEDO_HIPPIUS_USERNAME` | yes* | Robot push username |
| `ALBEDO_HIPPIUS_PASSWORD` | yes* | Robot push secret |
| `ALBEDO_UPLOAD_BACKEND` | no | `hippius` (default) or `hf` |
| `ALBEDO_DASHBOARD_URL` | no | Override king lookup URL |

### Eval operators

| Variable | Description |
|----------|-------------|
| `CHUTES_API_KEY` | Chutes bearer token for judges |
| `ALBEDO_DATASET_SHARD_PATH` | Path to pinned parquet shard |
| `ALBEDO_KING_GPUS` / `ALBEDO_CHAL_GPUS` | GPU IDs for vLLM |
| `ALBEDO_EVALS_S3_*` | Eval trace upload bucket |
| `HIPPIUS_HUB_TOKEN` | Hub auth for model pulls |

---

## Links

- **GitHub:** https://github.com/unarbos/albedo
- **Dashboard UI:** https://us-east-1.hippius.com/albedo/index.html
- **Dashboard JSON:** https://us-east-1.hippius.com/albedo/dashboard.json
- **Eval traces:** https://us-east-1.hippius.com/albedo/evals/
- **Hippius registry:** https://registry.hippius.com
- **SWE-ZERO dataset:** https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories
- **Genesis model:** https://huggingface.co/ricdomolm/mini-coder-1.7b
- **LLM-oriented mirror of this doc:** [`website/llms.txt`](website/llms.txt)

---

## License

MIT

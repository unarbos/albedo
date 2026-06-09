# Albedo

Trajectory-distillation **king-of-the-hill** subnet on Bittensor (netuid 97). Miners
upload challenger models; the reigning king and each challenger duel on pinned
[SWE-ZERO](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories)
coding trajectories, scored by an ensemble of LLM judges via [Chutes](https://llm.chutes.ai).
Beat the king → take the crown. Full duel traces are published for downstream distillation.

Entrypoints `eval.py` / `validator.py` / `miner.py` are thin shims into the `albedo` package.

All duel rules, judges, dataset pin, and arch locks live in [`chain.toml`](chain.toml);
code reads them via `albedo/config.py`. Override for experiments with
`ALBEDO_CHAIN_OVERRIDE=/path/to/alt.toml`.

> The current `chain.toml` targets a **Qwen3-4B** competition. Challengers must be
> the same size class as the king — `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
> `num_key_value_heads`, `intermediate_size`, and `head_dim` are all locked alongside
> `vocab_size`, `model_type`, `max_position_embeddings`, `tie_word_embeddings`, and `rope_theta`.

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e .            # add [train] for SFT, [dev] for ruff
```

## Mining

Prerequisites: hotkey registered on SN97, a Bittensor wallet, a Hippius Hub push
token (`HIPPIUS_HUB_TOKEN`), and a model that actually improves on the king
(the default `--noise` perturbation is only a pipeline smoke test).

```bash
export ALBEDO_NETUID=97
export BT_WALLET_NAME=<wallet>
export ALBEDO_CHALLENGER_NAMESPACE=<your-hippius-namespace>
export HIPPIUS_HUB_TOKEN=<token>

# 1. (real mining) train a challenger from the king + published traces
python scripts/collect_traces.py --out data/traces.jsonl     # SFT data from past duels
python scripts/inspect_dataset.py data/traces.jsonl          # sanity-check it
python scripts/train_sft.py --base Qwen/Qwen3-4B --data data/traces.jsonl --output ckpt/
python scripts/sanity_check.py ckpt/                         # format + injection check

# 2. upload + reveal (validates arch locks against the king first)
python scripts/upload_challenger.py --model ckpt/ --repo <namespace>/albedo-qwen3-4b-v1 --hotkey <ss58>
#    — or the all-in-one smoke path (downloads king, perturbs, uploads, reveals):
python miner.py --hotkey <hotkey_name> --noise 0.001
```

What `miner.py` does: discover king (from `dashboard.json`, else the chain.toml seed)
→ download king → `train_or_perturb()` (**replace with your SFT/RL loop**) →
`validate_local_config()` (mirrors the server's arch-lock gate) → upload to Hippius →
post the on-chain reveal `v4|<repo>|<digest>|<hotkey>`.

**Rules that are not negotiable:**
- Repo name must match `chain.toml [chain].repo_pattern` (lowercase, e.g. `you/albedo-qwen3-4b-run1`).
- Challenger digest must be a Hippius `sha256:` blob; no `*.py` and no `auto_map` in the snapshot.
- One eval per hotkey; the current king cannot challenge itself.
- **Dethrone:** challenger aggregate score must exceed king by ≥ `win_margin` points (0–100 scale).

## Distillation (no wallet needed)

```bash
python scripts/collect_traces.py --out data/traces.jsonl   # winning-side completions from published duels
```

## Operating

Two processes: **validator** (chain + orchestration) and **eval server** (GPU + vLLM + judges).

### One-time: seed the Qwen3-4B genesis king

```bash
python scripts/seed_genesis.py --hf-model Qwen/Qwen3-4B --repo unarbos/albedo-qwen3-4b-genesis
# paste the printed sha256 into chain.toml [seed].seed_digest, then:
python scripts/reset_state.py        # wipe prior king/queue state
```

`seed_genesis.py` pulls the base model from HuggingFace only as a *source*, then
uploads it to Hippius (sha256 digest). The subnet runtime is **Hippius-only** — every
king and challenger is fetched from Hippius by `materialize_model`; there is no
HuggingFace download path. The genesis must be seeded to Hippius before mining starts.

### Eval server (GPU box)

```bash
python scripts/prefetch_dataset.py --out /root/albedo/dataset   # download SWE-ZERO + build manifest.json
# paste the printed manifest_sha256 into chain.toml [dataset], then:
export ALBEDO_DATASET_DIR=/root/albedo/dataset
export CHUTES_API_KEY=... HIPPIUS_HUB_TOKEN=...
python eval.py                       # uvicorn albedo.eval_server:app
```

`albedo.duel.sampler` refuses to load if the on-disk `manifest.json` sha256 ≠
`chain.toml [dataset].manifest_sha256`.

### Validator

```bash
export ALBEDO_NETUID=97 ALBEDO_EVAL_SERVER=http://localhost:9001
export BT_WALLET_NAME=... BT_WALLET_HOTKEY=...
python validator.py
```

Use `pm2 start ecosystem.config.js` to run both processes under PM2, or `tunnel.sh` to forward the eval-server port over SSH.

On the GPU eval box, use `pm2 start ecosystem.eval.config.js` to run the eval server with `scripts/start_eval.sh` and `eval.env`.

## Layout

```
chain.toml             # judges, dataset pin, duel knobs, arch locks (single source of truth)
eval.py / validator.py / miner.py   # entrypoint shims into the albedo package
albedo/
  config.py            # loads chain.toml (+ ALBEDO_CHAIN_OVERRIDE)
  models/              # ModelRef, reveal v4, Hippius/HF download, upload, arch-lock compat
  duel/                # SWE-ZERO sampler, turn streaming
  judge/               # Chutes LLM-as-judge client + rubric
  preeval/             # weight-fingerprint dedup + prompt-injection probe
  eval_server/         # FastAPI duel server (vLLM king vs challenger)
  validator/           # reveal scan, admission gate, duel dispatch, weights
  stats.py             # bootstrap LCB + dethrone math
scripts/               # seed_genesis, prefetch_dataset, train_sft, collect_traces, upload_challenger, …
archs/qwen3/           # size-agnostic Qwen3 shim (no trust_remote_code)
configs/               # DeepSpeed ZeRO configs for train_sft.py
```

## License

MIT

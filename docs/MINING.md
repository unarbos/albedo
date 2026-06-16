# Mining on Albedo (SN97)

How to mine on the Albedo subnet using the `albedo` miner CLI in [miner/](miner/).

## What mining is

Albedo is a **king-of-the-hill** subnet for **Qwen3-4B** language models. As a miner you:

1. Fine-tune a Qwen3-4B model (your secret sauce).
2. Upload the model directory to **Hippius** (a HuggingFace-Hub–style model store).
3. Commit an on-chain *reveal* pointing at that exact upload (`repo` + content `digest`).

Validators then read your commitment off-chain, download the model from Hippius, run
the **same validation checks you can run locally** (file manifest + architecture lock +
near-duplicate dedup), and finally **evaluate** the model. A model that beats the current
king earns weight/emissions. So your job is: produce a model that (a) passes validation and
(b) scores higher than the incumbent by at least the **6% win margin**.

The whole publish flow is one pipeline:

```
validate (local) → upload to Hippius → check Hippius repo → hotkey registered? → commit on-chain
```

`albedo publish` runs all five steps; the other subcommands run them individually.

---

## Requirements

- **A registered hotkey on netuid 97** (`finney` mainnet). Registration costs recycled TAO —
  see [Register a hotkey](#register-a-hotkey).
- **A Bittensor wallet** (coldkey + hotkey) in `~/.bittensor/wallets/` (or set `ALBEDO_WALLET_PATH`).
- **Hippius credentials** — either a `HIPPIUS_HUB_TOKEN` or a username + password, plus a
  **namespace** to upload under.
- Python ≥ 3.11.

### The model must be a valid Qwen3-4B checkpoint

Validation is strict and defined in [chain.toml](chain.toml). Your uploaded repo must:

**Repo naming** — `<namespace>/albedo-qwen3-4b-<suffix>` (pattern `^[^/]+/albedo-qwen3-4b-.+$`).
The CLI builds this for you from `--namespace` + `--name`.

**File manifest** (`[files]` in [chain.toml](chain.toml)) — the repo's file set is checked
against a strict allowlist:

- **Required:** `config.json`, `tokenizer_config.json`, `tokenizer.json`
- **At least one** `*.safetensors` (single-shard `model.safetensors` or sharded `model-*-of-*.safetensors`)
- **Allowed (optional):** `generation_config.json`, `special_tokens_map.json`, `added_tokens.json`,
  `chat_template.jinja`, `merges.txt`, `vocab.json`, `model.safetensors.index.json`,
  `.gitattributes`, `README.md`
- **Forbidden:** any `*.py` file (no custom modeling code)
- Any other file is flagged as an **unexpected extra** and fails validation.

**Architecture lock** (`[arch]` / `[seed]` in [chain.toml](chain.toml)) — your `config.json`
must match the genesis Qwen3-4B seed exactly on:

- `architectures` and `model_type`, `vocab_size`
- capacity keys: `max_position_embeddings`, `tie_word_embeddings`, `rope_theta`, `hidden_size`,
  `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `head_dim`
- It must **not** contain `auto_map` (no remote code) or `quantization_config` (no quantized models).

In short: fine-tune the weights, keep the Qwen3-4B architecture and tokenizer intact, don't add
custom code or quantize.

**Dedup** — validators also reject models that are near-duplicates (≥ 0.95 fingerprint similarity)
of an already-seen model. This check needs OpenSearch and is **not** run by the local CLI; your
model needs to be meaningfully different from existing submissions.

---

## Install

From the albedo repo root:

```bash
cd ~/albedo
python3 -m venv .venv && source .venv/bin/activate   # or: sv
pip install -e .
```

(The repo uses `uv` elsewhere; `uv pip install -e .` works too.) This installs the `albedo`
console script (entry point `miner.cli:main`).

Verify:

```bash
albedo            # prints help
albedo --help
```

For the optional training extras (SFT/RL): `pip install -e '.[train]'` (pulls `trl`, `accelerate`, `deepspeed`).

---

## Configure

Copy the miner env template and fill it in:

```bash
cp .env.example_miners .env
```

`.env` keys (all optional — each just sets a default so you can omit the matching CLI flag;
real env vars and CLI flags always override):

| Key | Purpose |
|-----|---------|
| `ALBEDO_COLDKEY` / `ALBEDO_HOTKEY` | wallet identity, so you can skip `--coldkey/--hotkey` |
| `ALBEDO_WALLET_PATH` | only if wallets aren't in `~/.bittensor/wallets` |
| `CHAIN_NETUID` | defaults to `97` |
| `CHAIN_NETWORK` | defaults to `finney` (use `test` for testnet) |
| `HIPPIUS_HUB_TOKEN` | Hippius auth token (wins over username/password) |
| `HIPPIUS_HUB_USERNAME` / `HIPPIUS_HUB_PASSWORD` | alternative Hippius login |
| `ALBEDO_NAMESPACE` | your Hippius namespace; lets you omit `--namespace` |
| `ALBEDO_REPO_PREFIX` | leave as `albedo-qwen3-4b` unless the subnet changes it |
| `ALBEDO_MODEL_CACHE_DIR` | where remote `check-hippius` caches `config.json` |

`.env` is loaded from the repo root and the current working directory before any defaults are read.

---

## Quick start (end-to-end)

With `.env` filled in (namespace + wallet + Hippius creds):

```bash
# 1. one-time: register your hotkey on the subnet (costs recycled TAO)
albedo register

# 2. publish a model: validate → upload → check → commit
albedo publish --path /path/to/your/qwen3-4b-model --name v1
```

`publish` will validate locally, upload to `<namespace>/albedo-qwen3-4b-v1`, re-check the
uploaded repo, confirm your hotkey is registered, print a commit preview, and ask `Proceed? [y/N]`
before writing on-chain. Add `--yes` to skip the prompt, or `--skip-commit` to stop after upload.

---

## Commands

All commands default `--netuid 97` and `--network finney`, and read wallet/namespace from `.env`.

### Validate locally (before uploading)

```bash
albedo check-hippius --path /path/to/model
```

Runs the file-manifest + architecture checks on a local directory — the same code validators run.
Prints `[PASS]/[FAIL]` per check and `VALID`/`INVALID`. Note: dedup is **not** checked here.

### Validate an already-uploaded repo

```bash
albedo check-hippius --repo <ns>/albedo-qwen3-4b-v1 --digest sha256:...
```

Lists the repo's files on Hippius and downloads only `config.json` to re-run the checks.

### Upload a model to Hippius

```bash
albedo upload --path /path/to/model --name v1            # uses ALBEDO_NAMESPACE
albedo upload --path /path/to/model --namespace you --name v1
albedo upload --path /path/to/model --repo you/albedo-qwen3-4b-v1   # full repo override
```

`--name` is just the suffix; the prefix `albedo-qwen3-4b-` is added automatically (and a
doubled prefix is stripped, so `--name albedo-qwen3-4b-v1` also works). Prints the immutable
reference `repo@sha256:digest` and the on-chain `reveal` string. The digest is the content hash
of exactly what you uploaded — keep it for the commit step.

### Register a hotkey

```bash
albedo register                       # uses .env wallet
albedo register --coldkey mine --hotkey hk1
```

Mirrors `btcli subnet register` in-process: shows the recycle cost and your coldkey balance,
confirms with `[y/N]` (skip with `--yes`), submits a `burned_register`, and reports the assigned
UID. If the hotkey is already registered it just prints the existing UID. Aborts early if your
balance is below the recycle cost.

### Commit a reveal on-chain

```bash
albedo commit --repo <ns>/albedo-qwen3-4b-v1 --digest sha256:... \
  --coldkey mine --hotkey hk1
```

Writes the v6 reveal `v6|<repo>|<digest>` on-chain via `set_reveal_commitment`. Before submitting
it **checks your hotkey is registered** on the netuid, prints a preview, and prompts `Proceed? [y/N]`
(skip with `--yes`). Use this when you uploaded earlier and just need to commit a specific digest.

### Read on-chain commitments

```bash
albedo check-commit                      # all v6 commits on the subnet
albedo check-commit --hotkey 5F...        # filter to one hotkey ss58
```

Scans the chain for v6 commitments (oldest block first) and prints `block / hotkey / model_uri`.
Useful to confirm your own commit landed, or to see what others have submitted.

### Publish (full pipeline)

```bash
albedo publish --path /path/to/model --name v1 --coldkey mine --hotkey hk1
albedo publish --path ... --name v1 --yes            # no prompt
albedo publish --path ... --name v1 --skip-commit    # stop after upload + checks
```

Runs all five steps and stops at the first failure. The pipeline is the recommended path for a
normal submission.

### Interactive TUI

```bash
albedo on
```

Launches a full-screen console: a pinned `albedo>` input bar, a scrollable log, and a live
checklist of the publish steps (⬜ pending / ⏳ running / ✓ ok / ✗ fail). The same subcommands
run inside it. Keys: `↑/↓` history, `PgUp/PgDn` or `Ctrl+↑/↓` scroll, `Home/End` jump, `Ctrl+Y`
paste (use this in VS Code; `Ctrl+V` works in most terminals), `y/n` to answer a commit prompt,
`off` / `Ctrl+C` / `Ctrl+Q` to quit.

---

## Typical workflow

```bash
# (one time) register
albedo register

# iterate on your model, then before spending an upload:
albedo check-hippius --path ./out/qwen3-4b-mymodel        # must say VALID

# publish it
albedo publish --path ./out/qwen3-4b-mymodel --name v2

# confirm it's on-chain
albedo check-commit --hotkey <your hotkey ss58>
```

To target **testnet** while experimenting, prefix with the env var or set it in `.env`:

```bash
CHAIN_NETWORK=test albedo check-commit
```

---

## Notes & gotchas

- **Validate before you upload.** `check-hippius --path` is free and catches the file-manifest and
  architecture problems that would otherwise waste an upload + commit.
- **The digest is content-addressed.** Each upload returns a `sha256:` digest; the commit binds to
  that exact digest, so re-uploading changed weights produces a new digest you must re-commit.
- **Dedup isn't local.** Passing `check-hippius` doesn't guarantee acceptance — a model too similar
  (≥ 0.95) to an existing submission is rejected by the validator. Make your model genuinely distinct.
- **No custom code, no quantization.** `*.py` files are forbidden and `config.json` must not contain
  `auto_map` or `quantization_config`.
- **Registration is required to commit.** Both `commit` and `publish` verify your hotkey is in the
  netuid metagraph and abort if not.

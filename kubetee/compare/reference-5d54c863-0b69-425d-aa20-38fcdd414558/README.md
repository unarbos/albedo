# Reference eval: 5d54c863-0b69-425d-aa20-38fcdd414558

Public Albedo SN97 run ([dashboard](https://pub-e2a73e9642e74a2ea78d2910c7a86025.r2.dev/detail.html?eval_run_id=5d54c863-0b69-425d-aa20-38fcdd414558))
— challenger **lost** (margin below 3 pp win threshold). King stayed at v110 (`seed429/…hot`).

## Models

| Side | URI |
|------|-----|
| Previous king | `hf://seed429/albedo-qwen3.6-35b-hot@7e53e451bf0c6f0d8ac3f776af71dbcd6d436eb8` |
| Challenger | `applet3/albedo-qwen3.6-35b-albedo-ckp150@f58e8cc377dc0bd5d69a4eca8a45626aff29dd61` |

## Scores (Albedo / Denrite)

| Field | Value |
|-------|--------|
| Challenger | **0.682301** |
| King | **0.671335** |
| Margin | **+1.10 pp** (required 0.03) → lost |
| Samples | 100/100 |
| Judge | `z-ai/glm-5.2` only |
| Hardware | B200 (origin); KubeTEE pin uses H200 |
| `manifest_hash` | `e3cff61772b0096811d4c5d8bbc8dee8dacbd9a069bc4557608adf1c1c2ddf40` |

Both HF repos are **public** (verified downloadable with the PoC `HF_TOKEN`).

KubeTEE pin:

```text
ALBEDO_EVAL_SAMPLE_IDS_FILE=/app/shared/albedo/kubetee/compare/reference-5d54c863-0b69-425d-aa20-38fcdd414558/request.json
```

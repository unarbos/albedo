# KubeTEE pin runs vs Denrite references

Local copies of apple-to-apple eval artifacts (pinned `sample_ids` /
`manifest_hash=e3cff617…` from the matching `../reference-<id>/request.json`).

## Latest: vs Denrite `2205a13b` (ungated HF)

| | Chal | King | Margin | n | Wall | Total $ |
|--|------|------|--------|---|------|---------|
| Denrite [`2205a13b`](https://pub-e2a73e9642e74a2ea78d2910c7a86025.r2.dev/detail.html?eval_run_id=2205a13b-efeb-41f4-bf94-d8bd613301c9) | 0.673169 | 0.665314 | **+0.79 pp** (lost) | 100/100 | — | — |
| KubeTEE [`0c3c5bf7`](https://s3.hippius.com/sn97-albedo/kubetee-poc/0c3c5bf7-0e5c-4a13-8ad3-15a281f911e8/verdict.json) | 0.676880 | 0.654923 | **+2.20 pp** (lost) | 99/100 | **55.0 min** | **$16.00** |

Models (both public / `gated=False`): king `bkn1890/…hk971@1ba87c52…`, chal `darius3th/…kkk@90bd03cb…`. Same 100 sample IDs as Denrite `request.json`. Artifacts: [`0c3c5bf7-verdict.json`](./0c3c5bf7-verdict.json), [`0c3c5bf7-eval-summary.json`](./0c3c5bf7-eval-summary.json).

## Earlier: vs Denrite `ca530856`

| Run | Scores (chal / king) | n | Worker wall | Total $ | Notes |
|-----|----------------------|---|-------------|---------|-------|
| `fd245d96…` | 0.650496 / 0.68124 (−3.07pp) | 100/100 | **39.1 min** | **$13.38** | post `origin/main` merge |
| `9a0f5f09…` | 0.68715 / 0.665788 (+2.14pp) | 100/100 | 31.5 min | $11.89 | best pre-main headline match |
| `3e60c4c8…` | 0.680927 / 0.678479 | 100/100 | 38.6 min | $12.84 | prior pin |
| `e6c56798…` | 0.677038 / 0.679567 | 99/100 | 33.7 min | $12.78 | 1 king context overflow |

Origin Denrite `ca530856`: chal **0.689451** / king **0.659136**, margin **+3.03 pp** (won).

Aggregate + paired stats: [`compare-vs-origin.json`](./compare-vs-origin.json).

Public S3 objects: `https://s3.hippius.com/sn97-albedo/kubetee-poc/<eval_run_id>/<file>`
(e.g. `verdict.json`). Trailing-slash / prefix list URLs return **403** — Hippius
does not allow anonymous `ListBucket`; only per-object `public-read` GET works.

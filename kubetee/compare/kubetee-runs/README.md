# KubeTEE pin runs vs Denrite `ca530856`

Local copies of apple-to-apple eval artifacts (same pinned `sample_ids` /
`manifest_hash=e3cff617…` as `../reference-ca530856-ffa8-4e66-9175-7400e829e8c0/`).

| Run | Scores (chal / king) | n | Worker wall | Total $ | Notes |
|-----|----------------------|---|-------------|---------|-------|
| `fd245d96…` | 0.650496 / 0.68124 (−3.07pp) | 100/100 | **39.1 min** | **$13.38** | **latest** — post `origin/main` merge |
| `9a0f5f09…` | 0.68715 / 0.665788 (+2.14pp) | 100/100 | 31.5 min | $11.89 | best pre-main headline match |
| `3e60c4c8…` | 0.680927 / 0.678479 | 100/100 | 38.6 min | $12.84 | prior pin |
| `e6c56798…` | 0.677038 / 0.679567 | 99/100 | 33.7 min | $12.78 | 1 king context overflow |

Origin Denrite `ca530856`: chal **0.689451** / king **0.659136**, margin **+3.03 pp** (won).

Aggregate + paired stats: [`compare-vs-origin.json`](./compare-vs-origin.json).

Public S3 objects: `https://s3.hippius.com/sn97-albedo/kubetee-poc/<eval_run_id>/<file>`
(e.g. `verdict.json`). Trailing-slash / prefix list URLs return **403** — Hippius
does not allow anonymous `ListBucket`; only per-object `public-read` GET works.

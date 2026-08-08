# Gated follow-up — king 8 GPU / TP=8 parity

Do **not** promote until a 4+4 remote-king baseline verdict exists for the
same `sample_seed` + models.

## Change (king only)

In `king.yaml`:
- `nvidia.com/gpu` requests/limits: `"8"`
- `KING_TENSOR_PARALLEL_SIZE`: `"8"`
- Give the king its own full node (no co-located 4-GPU challenger on the same
  node during the parity run)

Challenger Job stays at **4 GPU / TP=4** (`eval.yaml` unchanged).

## Compare

1. Run baseline: king 4 / challenger 4 → save verdict + sample-ids.
2. Apply 8-GPU king, wait `/ready`, re-run challenger with the **same**
   `ALBEDO_EVAL_SAMPLE_SEED` / sample count / models (new `EVAL_RUN_ID`).
3. Promote only if challenger_won / scores match the 4+4 baseline within
   expected judge noise (same seed, greedy sampling `temperature=0`).

If scores drift, keep king at TP=4 (serving stack / KV layout mismatch).

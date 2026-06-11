"""sanity_service — FastAPI pre-eval gate for challenger models.

Downloads a challenger model, keeps one vLLM process warm, runs a few coding prompts,
and scores the responses with heuristic checks plus an optional OpenRouter coherence gate.
Returns passed/reason; results are cached per digest in Postgres.
"""

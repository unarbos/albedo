"""Request/response shapes for the sanity GPU worker API."""

from __future__ import annotations

from pydantic import BaseModel


class SanityRunRequest(BaseModel):
    # A pre-eval generation job: the dispatcher supplies the model and the pre-sampled prompts.
    run_id: str
    model_uri: str
    digest: str
    prompts: list[str]
    prompt_messages: list[list[dict[str, str]]] | None = None
    gen_max_tokens: int = 1024
    min_tokens: int = 5
    max_repetition: float = 0.85
    min_vocab_ratio: float = 0.3

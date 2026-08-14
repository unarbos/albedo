from __future__ import annotations

from pydantic import BaseModel


class SanityRunRequest(BaseModel):
    run_id: str
    model_uri: str
    digest: str
    prompts: list[str]
    sample_ids: list[str] | None = None
    prompt_messages: list[list[dict[str, str]]] | None = None
    assistant_turns: int = 1
    teardown_after_run: bool = True
    gen_max_tokens: int = 16384
    min_tokens: int = 5
    max_repetition: float = 0.95
    min_vocab_ratio: float = 0.05

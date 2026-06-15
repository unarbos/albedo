import json

import pyarrow as pa
import pyarrow.parquet as pq

from albedo_eval_service import remote_dataset
from albedo_eval_service.remote_dataset import load_swe_zero_samples


def test_load_swe_zero_sample_from_messages_json(tmp_path):
    shard_dir = tmp_path / "data"
    shard_dir.mkdir()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Fix the failing test."},
        {"role": "assistant", "content": "Use the right assertion."},
    ]
    table = pa.table({"messages": [json.dumps(messages)]})
    pq.write_table(table, shard_dir / "train-00000.parquet")

    samples = load_swe_zero_samples(
        dataset_root=tmp_path, sample_ids=["data/train-00000.parquet:0:0"]
    )

    assert len(samples) == 1
    assert samples[0].sample_id == "data/train-00000.parquet:0:0"
    assert samples[0].prompt == (
        "<|im_start|>system\n"
        "Be concise.<|im_end|>\n"
        "<|im_start|>user\n"
        "Fix the failing test.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert samples[0].target == "Use the right assertion."


def test_load_swe_zero_sample_uses_tokenizer_chat_template(tmp_path, monkeypatch):
    captured = {}

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "templated prompt"

    monkeypatch.setattr(remote_dataset, "_load_tokenizer", lambda path: _Tokenizer())

    shard_dir = tmp_path / "data"
    shard_dir.mkdir()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Fix the failing test."},
        {"role": "assistant", "content": "Use the right assertion."},
    ]
    table = pa.table({"messages": [json.dumps(messages)]})
    pq.write_table(table, shard_dir / "train-00000.parquet")

    samples = load_swe_zero_samples(
        dataset_root=tmp_path,
        sample_ids=["data/train-00000.parquet:0:0"],
        tokenizer_path="/models/qwen",
        enable_thinking=True,
    )

    assert samples[0].prompt == "templated prompt"
    assert samples[0].messages == messages[:2]
    assert captured["messages"] == messages[:2]
    assert captured["kwargs"] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }


def test_load_swe_zero_sample_from_prompt_column(tmp_path):
    shard_dir = tmp_path / "data"
    shard_dir.mkdir()
    table = pa.table({"prompt": ["Explain pytest fixtures."]})
    pq.write_table(table, shard_dir / "train-00001.parquet")

    samples = load_swe_zero_samples(
        dataset_root=tmp_path, sample_ids=["data/train-00001.parquet:0:0"]
    )

    assert samples[0].prompt == "Explain pytest fixtures."
    assert samples[0].target is None

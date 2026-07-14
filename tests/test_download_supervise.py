"""Killable, stall-aware supervised downloads for the sanity/validation storage path."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from config_validation.models import ModelRef
from config_validation.storage import _hf, _hippius, _supervise

_GIT_SHA1 = "b" * 40
_SHA256 = "sha256:" + "a" * 64


def test_supervise_kills_stalled_child(tmp_path, monkeypatch):
    monkeypatch.setattr(_supervise, "_HEARTBEAT_INTERVAL_S", 0.05)
    monkeypatch.setattr(_supervise, "STALL_SECONDS", 0.2)
    monkeypatch.setattr(_supervise, "STALL_RETRIES", 2)
    launched: list[subprocess.Popen] = []

    def fake_spawn(child_call, args, log_path):
        # Never writes to watch_dir and refuses to exit — a wedged transfer.
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"],
            start_new_session=True,
        )
        launched.append(proc)
        return proc

    monkeypatch.setattr(_supervise, "_spawn", fake_spawn)
    watch = tmp_path / "m"
    watch.mkdir()

    with pytest.raises(TimeoutError, match="made no progress"):
        _supervise.supervise_download(child_call="x", args=[], watch_dir=watch, label="ns/m")

    assert len(launched) == 2  # stalled once, retried, then gave up
    for proc in launched:
        assert proc.poll() is not None  # every stalled child was terminated


def test_supervise_raises_on_child_error(tmp_path, monkeypatch):
    monkeypatch.setattr(_supervise, "_HEARTBEAT_INTERVAL_S", 0.05)
    monkeypatch.setattr(_supervise, "STALL_SECONDS", 5.0)
    monkeypatch.setattr(_supervise, "STALL_RETRIES", 2)

    def fake_spawn(child_call, args, log_path):
        Path(log_path).write_text("RepositoryNotFoundError: 404\n", encoding="utf-8")
        return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(4)"])

    monkeypatch.setattr(_supervise, "_spawn", fake_spawn)
    watch = tmp_path / "m"
    watch.mkdir()

    with pytest.raises(RuntimeError, match="exited 4"):
        _supervise.supervise_download(child_call="x", args=[], watch_dir=watch, label="ns/m")


def test_supervise_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(_supervise, "_HEARTBEAT_INTERVAL_S", 0.05)
    watch = tmp_path / "m"
    watch.mkdir()

    def fake_spawn(child_call, args, log_path):
        (watch / "model.safetensors").write_bytes(b"x" * 1024)
        return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])

    monkeypatch.setattr(_supervise, "_spawn", fake_spawn)

    _supervise.supervise_download(child_call="x", args=[], watch_dir=watch, label="ns/m")


def test_hf_full_download_routes_through_supervisor(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    monkeypatch.setattr(_hf, "_cache_dir", lambda ref: dest)
    monkeypatch.setattr(_supervise, "OUT_OF_PROCESS", True)
    calls = {}

    def fake_supervise(*, child_call, args, watch_dir, label, stall_seconds=None, max_attempts=None):
        calls["child_call"] = child_call
        calls["args"] = args
        calls["label"] = label
        calls["stall_seconds"] = stall_seconds
        calls["max_attempts"] = max_attempts

    monkeypatch.setattr(_supervise, "supervise_download", fake_supervise)

    _hf.download_full(ModelRef("ns/m", _GIT_SHA1))

    assert "_download_child" in calls["child_call"]
    assert calls["args"][0] == "ns/m" and calls["args"][1] == _GIT_SHA1
    # HF uses the module-default limits (no override passed).
    assert calls["stall_seconds"] is None and calls["max_attempts"] is None


def test_hf_config_only_stays_in_process(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(_hf, "_cache_dir", lambda ref: tmp_path)
    monkeypatch.setattr(_supervise, "OUT_OF_PROCESS", True)

    def boom(**kw):
        raise AssertionError("config-only download must not spawn a supervisor child")

    monkeypatch.setattr(_supervise, "supervise_download", lambda **kw: boom())
    seen = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda **kw: seen.update(kw) or kw["local_dir"], raising=False,
    )

    _hf.download_config(ModelRef("ns/m", _GIT_SHA1))
    assert seen["allow_patterns"] == _hf._CONFIG_ONLY_PATTERNS


def test_hippius_full_download_routes_through_supervisor(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    monkeypatch.setattr(_hippius, "_cache_dir", lambda ref: dest)
    monkeypatch.setattr(_supervise, "OUT_OF_PROCESS", True)
    calls = {}

    def fake_supervise(*, child_call, args, watch_dir, label, stall_seconds=None, max_attempts=None):
        calls["child_call"] = child_call
        calls["args"] = args
        calls["stall_seconds"] = stall_seconds
        calls["max_attempts"] = max_attempts

    monkeypatch.setattr(_supervise, "supervise_download", fake_supervise)

    _hippius.download_full(ModelRef("ns/m", _SHA256))

    assert "config_validation.storage._hippius" in calls["child_call"]
    assert calls["args"][0] == "ns/m" and calls["args"][1] == _SHA256
    # Hippius passes its own wider limits, higher than the HF defaults.
    assert calls["stall_seconds"] == _supervise.HIPPIUS_STALL_SECONDS
    assert calls["max_attempts"] == _supervise.HIPPIUS_STALL_RETRIES
    assert _supervise.HIPPIUS_STALL_SECONDS > _supervise.STALL_SECONDS

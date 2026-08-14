from __future__ import annotations

from albedo_eval_service.shared.observation_format import (
    OPENHANDS,
    RETURNCODE,
    SWE_AGENT,
    TRUNCATION_SENTINEL,
    classify,
    detect_format,
    empty_output,
    is_truncated,
    repair_output,
    truncation_notice,
    valid_output,
    wrap,
)
from albedo_eval_service.simulator.prompt_simulator import (
    FORMAT_MINI_CODER,
    FORMAT_OPENHANDS,
    FORMAT_SWE_AGENT,
    format_block,
)

RC_OBS = "<returncode>0</returncode>\n<output>\ntotal 228\ndrwxr-xr-x 12 root root\n</output>"
SWE_AGENT_OBS = "OBSERVATION:\nHere's the files and directories up to 2 levels deep in /testbed:"
OPENHANDS_BASH_OBS = (
    "\n[The command completed with exit code 0.]\n"
    "[Current working directory: /workspace/pandas-dev__pandas__1.0]\n"
    "[Command finished with exit code 0]"
)
OPENHANDS_EDITOR_OBS = "File created successfully at: /workspace/attrs__1.0/reproduce.py"


def _prefix(observation: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "the task description"},
        {"role": "assistant", "content": "```bash\nls\n```"},
        {"role": "user", "content": observation},
    ]


def test_classify_reads_the_marker_each_corpus_uses():
    assert classify(RC_OBS) == RETURNCODE
    assert classify(SWE_AGENT_OBS) == SWE_AGENT
    assert classify(OPENHANDS_BASH_OBS) == OPENHANDS
    assert classify(OPENHANDS_EDITOR_OBS) == OPENHANDS


def test_detect_format_reads_the_trajectorys_own_observation():
    assert detect_format("open-swe-traces/x:0:1", _prefix(RC_OBS)) == RETURNCODE
    assert detect_format("open-swe-traces/x:0:1", _prefix(SWE_AGENT_OBS)) == SWE_AGENT
    assert detect_format("open-swe-traces/x:0:1", _prefix(OPENHANDS_BASH_OBS)) == OPENHANDS
    assert detect_format("open-swe-traces/x:0:1", _prefix(SWE_AGENT_OBS)) != detect_format(
        "open-swe-traces/x:0:1", _prefix(OPENHANDS_BASH_OBS)
    )


def test_detect_format_ignores_the_leading_task_message():
    task_only = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Fix the bug in foo.py"},
    ]
    assert detect_format("mini-coder/x:0:1", task_only) == RETURNCODE
    assert detect_format("swe-hero/x:0:1", task_only) == OPENHANDS
    assert detect_format("mini-coder/x:0:1", None) == RETURNCODE


def test_valid_output_accepts_the_native_dialect_and_rejects_the_others():
    assert valid_output(RC_OBS, RETURNCODE) is True
    assert valid_output(SWE_AGENT_OBS, SWE_AGENT) is True
    assert valid_output(OPENHANDS_BASH_OBS, OPENHANDS) is True
    assert valid_output(OPENHANDS_EDITOR_OBS, OPENHANDS) is True

    assert valid_output(SWE_AGENT_OBS, RETURNCODE) is False
    assert valid_output(OPENHANDS_EDITOR_OBS, SWE_AGENT) is False
    assert valid_output(RC_OBS, OPENHANDS) is False
    assert valid_output("Observation: retired dialect", OPENHANDS) is False
    for fmt in (RETURNCODE, SWE_AGENT, OPENHANDS):
        assert valid_output("", fmt) is False


def test_injected_observations_are_valid_in_their_own_format():
    for fmt in (RETURNCODE, SWE_AGENT, OPENHANDS):
        assert valid_output(empty_output(fmt), fmt), fmt
        assert valid_output(wrap("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", fmt), fmt), fmt
        assert valid_output(wrap("no bash command", fmt, returncode=2), fmt), fmt
    assert empty_output(RETURNCODE) == "<returncode>0</returncode>\n<output>\n</output>"
    assert wrap("done", SWE_AGENT) == "OBSERVATION:\ndone"
    assert wrap("done", RETURNCODE, returncode=2) == (
        "<returncode>2</returncode>\n<output>\ndone\n</output>"
    )


def test_repair_output_only_touches_the_returncode_wrapper():
    squashed = "<returncode>0</returncode>\n<output>ok</output>"
    assert valid_output(squashed, RETURNCODE) is False
    assert valid_output(repair_output(squashed, RETURNCODE), RETURNCODE) is True
    assert repair_output(OPENHANDS_EDITOR_OBS, OPENHANDS) == OPENHANDS_EDITOR_OBS


def test_format_block_matches_the_format():
    assert format_block(RETURNCODE) == FORMAT_MINI_CODER
    assert format_block(SWE_AGENT) == FORMAT_SWE_AGENT
    assert format_block(OPENHANDS) == FORMAT_OPENHANDS


def test_truncation_notice_is_detectable_and_names_the_limit():
    notice = truncation_notice(16384)
    assert TRUNCATION_SENTINEL in notice
    assert "16384" in notice
    assert is_truncated(notice)
    assert is_truncated(f"CANDIDATE OUTPUT 1:\n------\n{notice}\n------")
    assert not is_truncated("an ordinary candidate answer")
    assert not is_truncated("")

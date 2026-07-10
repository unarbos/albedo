from chain_reader.chain import scan_commitments


class FakeNeuron:
    def __init__(self, hotkey, uid):
        self.hotkey = hotkey
        self.uid = uid


class FakeMeta:
    neurons = [
        FakeNeuron("hk-good", 7),
        FakeNeuron("hk-v5", 8),
        FakeNeuron("hk-four", 9),
        FakeNeuron("hk-hf", 10),
        FakeNeuron("hk-mutable", 11),
    ]


def _reveal(payload: str) -> str:
    # Hex-serialized SCALE bytes as RevealedCommitments returns them: a mode-0
    # compact-length prefix byte, then the utf-8 payload.
    return "0x" + (b"\x00" + payload.encode()).hex()


class FakeSubtensor:
    def query_map(self, module, name, params):
        assert (module, name) == ("Commitments", "RevealedCommitments")
        return [
            ("hk-good", [(_reveal("v7|alice/model|sha256:" + "a" * 64), 100)]),
            ("hk-v5", [(_reveal("v5|alice/model|sha256:" + "b" * 64), 101)]),
            ("hk-four", [(_reveal("v7|alice/model|sha256:" + "c" * 64 + "|hk-four"), 102)]),
            ("hk-hf", [(_reveal("v7|alice/model-hf|" + "d" * 40), 103)]),
            ("hk-mutable", [(_reveal("v7|alice/model|main"), 104)]),
        ]

    def metagraph(self, netuid):
        return FakeMeta()

    def get_block_hash(self, block):
        return f"0x{block}"


def test_scan_commitments_accepts_only_three_part_v7_payloads():
    commits = scan_commitments(FakeSubtensor(), 1)

    assert len(commits) == 2
    by_hotkey = {c.hotkey: c for c in commits}

    commit = by_hotkey["hk-good"]
    assert commit.uid == 7
    assert commit.model_uri == "alice/model@sha256:" + "a" * 64
    assert commit.commit_payload == {
        "version": "v7",
        "repo": "alice/model",
        "digest": "sha256:" + "a" * 64,
        "author_hotkey": "hk-good",
        "spoofed": False,
    }

    # An HF git-revision pin (bare 40-hex) is a valid v7 digest; a mutable ref is not.
    hf_commit = by_hotkey["hk-hf"]
    assert hf_commit.uid == 10
    assert hf_commit.model_uri == "alice/model-hf@" + "d" * 40
    assert "hk-mutable" not in by_hotkey

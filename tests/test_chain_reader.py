from chain_reader.chain import scan_commitments


class FakeNeuron:
    def __init__(self, hotkey, uid):
        self.hotkey = hotkey
        self.uid = uid


class FakeMeta:
    neurons = [FakeNeuron("hk-good", 7), FakeNeuron("hk-v4", 8), FakeNeuron("hk-four", 9)]


class FakeSubtensor:
    def get_all_commitments(self, netuid):
        return [
            ("hk-good", [(100, "v5|alice/model|sha256:" + "a" * 64)]),
            ("hk-v4", [(101, "v4|alice/model|sha256:" + "b" * 64)]),
            ("hk-four", [(102, "v5|alice/model|sha256:" + "c" * 64 + "|hk-four")]),
        ]

    def query_map(self, *_args):
        return []

    def metagraph(self, netuid):
        return FakeMeta()

    def get_block_hash(self, block):
        return f"0x{block}"


def test_scan_commitments_accepts_only_three_part_v5_payloads():
    commits = scan_commitments(FakeSubtensor(), 1)

    assert len(commits) == 1
    commit = commits[0]
    assert commit.hotkey == "hk-good"
    assert commit.uid == 7
    assert commit.model_uri == "alice/model@sha256:" + "a" * 64
    assert commit.commit_payload == {
        "version": "v5",
        "repo": "alice/model",
        "digest": "sha256:" + "a" * 64,
        "author_hotkey": "hk-good",
        "spoofed": False,
    }

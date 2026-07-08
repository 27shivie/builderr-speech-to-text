"""Guards the public benchmark corpus contract.

The RambleFix line is only a fair qualifier if it is scored on the same hidden
manifest as entrant submissions. Smaller component checks can stay in docs, but
must not be presented as the payout line.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hidden_manifest_is_current_official_corpus() -> None:
    manifest = json.loads((ROOT / "data/hidden/manifest.json").read_text())
    counts = Counter(row["category"] for row in manifest)

    assert len(manifest) == 96
    assert counts == {
        "fleurs_english": 20,
        "fleurs_hindi": 20,
        "openslr104_hinglish": 40,
        "youtube_english": 16,
    }


def test_streaming_contract_states_same_corpus_rule() -> None:
    contract = (ROOT / "docs/STREAMING_CONTRACT.md").read_text()

    assert "same hidden corpus" in contract
    assert "data/hidden/manifest.json" in contract
    assert "96 rows" in contract
    assert "component check" in contract
    assert "not the final payout line" in contract

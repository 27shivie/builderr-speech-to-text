from __future__ import annotations

import json
import asyncio
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluator  # noqa: E402
from evaluator import (  # noqa: E402
    _capture_clip,
    _jitter_seed,
    _resolve_audio,
    _sandbox_profile,
    _select_manifest,
    _wire_clip_id,
)


def test_jitter_seed_is_stable_and_attempt_specific() -> None:
    assert _jitter_seed("clip-a", 0) == _jitter_seed("clip-a", 0)
    assert _jitter_seed("clip-a", 0) != _jitter_seed("clip-a", 1)
    assert _jitter_seed("clip-a", 0) != _jitter_seed("clip-b", 0)


def test_wire_ids_are_opaque_and_keyed() -> None:
    clip_id = "fleurs_hi_in_test_1766"
    first = _wire_clip_id(clip_id, b"a" * 32)
    second = _wire_clip_id(clip_id, b"b" * 32)
    assert clip_id not in first
    assert len(first) == 24
    assert first != second


def test_hidden_audio_directory_convention_resolves() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "audio").mkdir()
        audio = root / "audio" / "clip-a.wav"
        audio.write_bytes(b"RIFF")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps([{"clip_id": "clip-a"}]), encoding="utf-8")
        assert _resolve_audio({"clip_id": "clip-a"}, str(manifest)) == str(audio.resolve())


def test_sandbox_profile_blocks_external_network_and_private_reads() -> None:
    private = "/private/tmp/builderr-hidden"
    profile = _sandbox_profile([private])
    assert "(deny network*)" in profile
    assert 'remote ip "localhost:*"' in profile
    assert 'local ip "localhost:*"' in profile
    assert f'(deny file-read* (subpath "{private}"))' in profile


def test_balanced_screening_selection() -> None:
    rows = [
        {"clip_id": "e1", "category": "english"},
        {"clip_id": "e2", "category": "english"},
        {"clip_id": "h1", "category": "hindi"},
        {"clip_id": "h2", "category": "hindi"},
    ]
    assert [row["clip_id"] for row in _select_manifest(rows, per_category=1)] == ["e1", "h1"]


def test_captured_result_uses_opaque_id(monkeypatch) -> None:
    monkeypatch.setattr(evaluator, "_read_pcm_16k_mono", lambda _path: b"\0\0\0\0")

    async def fake_run(_uri, clip_id, _pcm):
        return {"clip_id": clip_id, "dropped": True, "pace_ok": True}

    monkeypatch.setattr(evaluator, "_run_once", fake_run)
    row = {
        "clip_id": "recoverable-source-dataset-id",
        "_wav": "/unused.wav",
        "gold": "private words",
    }
    captured = asyncio.run(_capture_clip("ws://unused", row, 1, b"s" * 32))
    assert captured["clip_id"] != row["clip_id"]
    assert len(captured["clip_id"]) == 24

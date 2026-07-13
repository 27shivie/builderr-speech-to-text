"""Current RambleFix engine adapter for the sealed Builderr streaming harness.

The adapter sends each rolling PCM prefix to RambleFix's resident local inference
server. It contains no model logic and no corpus access; both partials and finals
are measured by the same receive-clock harness used for entrants.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import urllib.request
import wave
from concurrent.futures import Future, ThreadPoolExecutor


_SR = 16000
_MIN_AUDIO_BYTES = int(_SR * 0.75) * 2
_PARTIAL_STEP_BYTES = int(_SR * 1.5) * 2
_ENDPOINT = os.environ.get("RAMBLEFIX_CURRENT_ENDPOINT", "http://127.0.0.1:8188/inference")

_previous = ""
_committed = ""
_generation = 0
_scheduled_bytes = 0
_future: Future[tuple[int, str]] | None = None
_state_lock = threading.Lock()
_endpoint_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="builderr-ramblefix-prefix")


def draft_reset() -> None:
    global _previous, _committed, _generation, _scheduled_bytes, _future
    with _state_lock:
        _generation += 1
        _previous = ""
        _committed = ""
        _scheduled_bytes = 0
        _future = None


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    global _previous, _committed, _scheduled_bytes, _future
    if not is_final and len(audio_buffer) < _MIN_AUDIO_BYTES:
        return _committed, len(_committed)

    if is_final:
        text = _transcribe(audio_buffer)
        with _state_lock:
            if text:
                _previous = text
                _committed = text
            return _committed, len(_committed)

    with _state_lock:
        _collect_partial()
        if _future is None and len(audio_buffer) - _scheduled_bytes >= _PARTIAL_STEP_BYTES:
            generation = _generation
            _scheduled_bytes = len(audio_buffer)
            _future = _executor.submit(_transcribe_generation, generation, bytes(audio_buffer))
        return _previous or _committed, len(_committed)


def _collect_partial() -> None:
    global _future, _previous, _committed
    if _future is None or not _future.done():
        return
    completed = _future
    _future = None
    generation, text = completed.result()
    if generation != _generation or not text:
        return

    stable = _common_word_prefix(_previous, text)
    if len(stable) >= len(_committed):
        _committed = stable
    _previous = text


def _transcribe_generation(generation: int, audio_buffer: bytes) -> tuple[int, str]:
    return generation, _transcribe(audio_buffer)


def _transcribe(audio_buffer: bytes) -> str:
    fd, path = tempfile.mkstemp(prefix="builderr-ramblefix-", suffix=".wav")
    os.close(fd)
    try:
        with wave.open(path, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(_SR)
            writer.writeframes(audio_buffer)
        body = json.dumps({"audio_path": path}).encode("utf-8")
        request = urllib.request.Request(
            _ENDPOINT, data=body, headers={"Content-Type": "application/json"}
        )
        with _endpoint_lock:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return " ".join(str(payload.get("text") or "").split())
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _common_word_prefix(left: str, right: str) -> str:
    left_words = re.findall(r"[\w'.-]+", left, flags=re.UNICODE)
    right_words = re.findall(r"[\w'.-]+", right, flags=re.UNICODE)
    prefix: list[str] = []
    for old, new in zip(left_words, right_words):
        if old.casefold() != new.casefold():
            break
        prefix.append(new)
    return " ".join(prefix)

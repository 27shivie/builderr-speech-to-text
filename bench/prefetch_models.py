#!/usr/bin/env python3
"""Run ONCE, with the network ON. Downloads + caches every model we plan to A/B.

Scoring runs under `sandbox-exec` with `(deny network*)`. If a model is not in
the local HF cache before the scored run, WhisperModel() will try to fetch it,
the sandbox will refuse, _transcribe_pcm() will raise, _safe_draft() will return
("", 0) -- and a blank final now caps the clip at ZERO.

So: a missing cache entry is not a warning. It is a zero.

Also prints on-disk size per model, to check the ~5GB total budget.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

MODELS = ["small", "medium", "large-v3-turbo"]
BUDGET_GB = 5.0


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main() -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("pip install -r requirements-streaming.txt first")

    cache = Path.home() / ".cache" / "huggingface" / "hub"
    before = dir_size_gb(cache)
    print(f"HF cache: {cache}  ({before:.2f} GB before)\n")

    for size in MODELS:
        print(f"[{size}] fetching ...", end="", flush=True)
        t0 = time.monotonic()
        try:
            WhisperModel(size, device="cpu", compute_type="int8")
            print(f" ok  ({time.monotonic() - t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f" FAILED: {type(exc).__name__}: {exc}")
            print(f"    -> if 'large-v3-turbo' is unknown, your faster-whisper is old.")
            print(f"    -> pip install -U faster-whisper  (turbo needs >=1.1)")

    after = dir_size_gb(cache)
    print(f"\nHF cache now: {after:.2f} GB  (+{after - before:.2f} GB)")
    print(f"free disk: {shutil.disk_usage(Path.home()).free / 1e9:.1f} GB")
    if after > BUDGET_GB:
        print(f"\n⚠️  cache {after:.2f} GB exceeds the ~{BUDGET_GB} GB model budget.")
        print("    That budget is on SHIPPED models, not the cache -- but only ship one.")
    print("\nNow verify offline: turn wifi OFF and re-run this script.")
    print("It must print 'ok' for every model with zero network. If it doesn't, you")
    print("would score 0 on every clip in the official run.")


if __name__ == "__main__":
    main()

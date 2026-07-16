#!/usr/bin/env python3
"""Runtime-agnostic model sweep. Iterates over (backend, model) via the SAME
Backend interface the shipped draft() uses. Adding MLX later = it appears in
available_backends(); nothing here changes.

Today available_backends() == ["faster-whisper"], so this collects the
faster-whisper baseline ONLY, which is exactly the current task. MLX joins the
sweep automatically once solution/backends.py registers it.

    bigger model -> more MEANING (50pts) but slower FINAL (30pts, cap @4000ms)

Stage 1 (~2 min): raw warm single-decode wall time per (backend, model), no harness.
Stage 2 (~5 min/config): full evaluator + real scorer.

    python bench/prefetch_models.py        # ONCE, network on
    python bench/models.py --stage 1
    python bench/models.py --stage 2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from streaming_scorecard import end_to_final_points              # noqa: E402
from solution.backends import available_backends, make_backend, models_for  # noqa: E402
from bench.ab import make_shim, run_arm, summarize               # noqa: E402

MODELS = ["small", "medium", "large-v3-turbo"]
LATENCY_CAP_MS = 4000


def _pcm_bytes(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes())


def stage1(backends, reps, model_filter=None):
    clips = sorted((ROOT / "samples").glob("*.wav"))
    pcms = [_pcm_bytes(c) for c in clips]
    results = {}
    print(f"\n{'='*74}\nSTAGE 1 - warm single-decode wall time "
          f"({len(clips)} clips x {reps} reps)\n{'='*74}")
    for backend_name in backends:
        for model in models_for(backend_name):
            if model_filter and model not in model_filter:
                continue
            key = f"{backend_name}:{model}"
            os.environ["BUILDERR_BACKEND"] = backend_name
            os.environ["BUILDERR_MODEL"] = model
            try:
                b = make_backend()
                print(f"\n[{key}] load ...", end="", flush=True)
                t0 = time.monotonic()
                b.load()
                print(f" {time.monotonic()-t0:.1f}s")
                b.transcribe(pcms[0])
                times = []
                for pcm in pcms:
                    for _ in range(reps):
                        t0 = time.monotonic()
                        b.transcribe(pcm)
                        times.append((time.monotonic() - t0) * 1000.0)
                times.sort()
                p50 = statistics.median(times)
                results[key] = {"backend": backend_name, "model": model, "p50": p50,
                                "p95": times[min(len(times)-1, int(len(times)*0.95))],
                                "max": times[-1], "sd": statistics.pstdev(times)}
                flag = "  !! OVER 4000ms CAP" if p50 > LATENCY_CAP_MS else ""
                print(f"[{key}] warm decode p50={p50:7.0f}ms "
                      f"-> latency {end_to_final_points(p50):5.1f}/30{flag}")
            except NotImplementedError:
                print(f"\n[{key}] SKIP (backend is a stub - not integrated yet)")
            except Exception as exc:
                print(f"\n[{key}] FAILED: {type(exc).__name__}: {exc}")
    for v in ("BUILDERR_BACKEND", "BUILDERR_MODEL"):
        os.environ.pop(v, None)
    return results


def stage2(backends, runs, limit, s1, model_filter=None):
    print(f"\n{'='*74}\nSTAGE 2 - full harness (evaluator + scorer)\n{'='*74}")
    shim = make_shim("solution.draft")
    out = {}
    for backend_name in backends:
        for model in models_for(backend_name):
            if model_filter and model not in model_filter:
                continue
            key = f"{backend_name}:{model}"
            if s1.get(key) and s1[key]["p50"] > LATENCY_CAP_MS * 1.25:
                print(f"\n[{key}] SKIP - warm decode {s1[key]['p50']:.0f}ms past cap.")
                continue
            os.environ["BUILDERR_BACKEND"] = backend_name
            os.environ["BUILDERR_MODEL"] = model
            print(f"\n[{key}] harness ...", flush=True)
            try:
                out[key] = summarize(run_arm(shim, str(ROOT / "samples" / "manifest.json"),
                                             runs, limit, tag=key.replace(":", "_")))
            except Exception as exc:
                print(f"[{key}] FAILED: {exc}")
    for v in ("BUILDERR_BACKEND", "BUILDERR_MODEL"):
        os.environ.pop(v, None)
    return out


def report(s1, s2):
    print(f"\n{'='*82}\nACCURACY-LATENCY FRONTIER\n{'='*82}")
    hdr = (f"{'backend:model':<28}{'warm dec':>10}{'e2f p50':>10}"
           f"{'lat pts':>9}{'meaning':>9}{'WER':>8}{'SCORE':>8}")
    print(hdr); print("-" * len(hdr))
    best = None
    for key in sorted(set(s1) | set(s2)):
        a, b = s1.get(key), s2.get(key)
        dec = f"{a['p50']:.0f}ms" if a else "-"
        if b:
            print(f"{key:<28}{dec:>10}{b['e2f_p50']:>8.0f}ms"
                  f"{end_to_final_points(b['e2f_p50']):>9.1f}"
                  f"{b['meaning']:>9.3f}{b['wer']:>8.3f}{b['score']:>8.2f}")
            if best is None or b["score"] > s2[best]["score"]:
                best = key
        else:
            print(f"{key:<28}{dec:>10}{'(stage2 skipped)':>36}")
    print("\nDECISION on MLX:")
    print("  faster-whisper wins AND is at the frontier (no bigger model scores")
    print("  higher within 4000ms) -> STAY, do NOT build MLX.")
    print("  A bigger faster-whisper model would win on meaning BUT its warm decode")
    print("  exceeds 4000ms (CPU-bound) -> THAT triggers MLX integration; then")
    print("  re-run this sweep with the mlx backend registered.")
    if best:
        print(f"\n  >>> current winner: {best}  (score {s2[best]['score']:.2f}) <<<")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], default=0)
    ap.add_argument("--backends", nargs="*", default=available_backends())
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these model ids within each backend "
                         "(e.g. --models swift). Default: all models_for(backend).")
    args = ap.parse_args()
    s1 = json.loads((HERE/"_stage1.json").read_text()) if (HERE/"_stage1.json").exists() else {}
    s2 = json.loads((HERE/"_stage2.json").read_text()) if (HERE/"_stage2.json").exists() else {}
    if args.stage in (0, 1):
        s1 = stage1(args.backends, args.reps, args.models)
        (HERE/"_stage1.json").write_text(json.dumps(s1, indent=2))
    if args.stage in (0, 2):
        s2 = stage2(args.backends, args.runs, args.limit, s1, args.models)
        (HERE/"_stage2.json").write_text(json.dumps(s2, indent=2))
    report(s1, s2)


if __name__ == "__main__":
    main()

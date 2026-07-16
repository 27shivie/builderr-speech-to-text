#!/usr/bin/env python3
"""A/B bench: run two draft() variants through the REAL evaluator + REAL scorer.

Never edits the sealed solution/stream_server.py. It generates import-swapped
shim servers under bench/_shims/ and points evaluator.py --server-module at them.

Usage:
    python bench/ab.py --a solution.draft_v0_baseline --b solution.draft --runs 5

Prints one comparison table. That table is the only thing that decides keep/revert.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHIMS = HERE / "_shims"
SEALED = ROOT / "solution" / "stream_server.py"


def make_shim(draft_module: str) -> str:
    """Copy the sealed server, swapping only its draft import. Sealed file untouched."""
    SHIMS.mkdir(exist_ok=True)
    (SHIMS / "__init__.py").write_text("")
    src = SEALED.read_text(encoding="utf-8")
    swapped, n = re.subn(
        r"from\s+solution\.draft\s+import",
        f"from {draft_module} import",
        src,
    )
    if n != 1:
        raise SystemExit(f"expected exactly 1 draft import in sealed server, found {n}")
    name = "shim_" + re.sub(r"\W", "_", draft_module)
    (SHIMS / f"{name}.py").write_text(swapped, encoding="utf-8")
    return f"bench._shims.{name}"


def run_arm(server_module: str, manifest: str, runs: int, limit: int, tag: str | None = None) -> dict:
    tag = tag or server_module.split(".")[-1]
    out = HERE / f"_result_{tag}.json"
    cmd = [
        sys.executable, str(ROOT / "evaluator.py"),
        "--server-module", server_module,
        "--manifest", manifest,
        "--runs", str(runs),
        "--no-offline",                      # dev only; official run enforces
        "--output-json", str(out),
        "--server-log", str(HERE / f"_log_{tag}.txt"),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.monotonic() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"arm {server_module} failed (exit {proc.returncode})")
    res = json.loads(out.read_text(encoding="utf-8"))
    res["_wall_s"] = wall
    return res


def summarize(res: dict) -> dict:
    clips = res["clips"]
    lats = [c["median_end_to_final_ms"] for c in clips
            if c.get("median_end_to_final_ms") is not None]
    lats.sort()
    def pct(p):
        if not lats:
            return float("nan")
        return lats[min(len(lats) - 1, int(len(lats) * p))]
    return {
        "score":    res["overall_score"],
        "meaning":  res["meaning_mean"],
        "wer":      res["wer_mean"],
        "e2f_p50":  statistics.median(lats) if lats else float("nan"),
        "e2f_p95":  pct(0.95),
        "e2f_max":  lats[-1] if lats else float("nan"),
        "e2f_sd":   statistics.pstdev(lats) if len(lats) > 1 else 0.0,
        "capped":   res["clips_capped"],
        "n":        res["n"],
        "final_ok": res["final_ok_rate"],
        "wall_s":   res["_wall_s"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="solution.draft_v0_baseline", help="control")
    ap.add_argument("--b", default="solution.draft", help="treatment")
    ap.add_argument("--manifest", default=str(ROOT / "samples" / "manifest.json"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    arms = {}
    for label, mod in (("A (control)", args.a), ("B (change)", args.b)):
        print(f"running {label}: {mod} ...", flush=True)
        arms[label] = summarize(run_arm(make_shim(mod), args.manifest, args.runs, args.limit))

    a, b = arms["A (control)"], arms["B (change)"]
    rows = [
        ("overall score",      "score",    "+", "{:.2f}"),
        ("meaning (mean)",     "meaning",  "+", "{:.3f}"),
        ("WER (mean)",         "wer",      "-", "{:.3f}"),
        ("end-to-final p50",   "e2f_p50",  "-", "{:.0f} ms"),
        ("end-to-final p95",   "e2f_p95",  "-", "{:.0f} ms"),
        ("end-to-final max",   "e2f_max",  "-", "{:.0f} ms"),
        ("end-to-final sd",    "e2f_sd",   "-", "{:.0f} ms"),
        ("clips capped",       "capped",   "-", "{:.0f}"),
        ("final_ok rate",      "final_ok", "+", "{:.2f}"),
        ("wall clock",         "wall_s",   "-", "{:.1f} s"),
    ]
    w = 20
    print(f"\n{'metric':<{w}} {'A control':>14} {'B change':>14} {'delta':>14}")
    print("-" * (w + 46))
    for name, key, better, fmt in rows:
        av, bv = a[key], b[key]
        d = bv - av
        good = (d > 0) if better == "+" else (d < 0)
        mark = "" if abs(d) < 1e-9 else ("  ✅" if good else "  ⚠️")
        print(f"{name:<{w}} {fmt.format(av):>14} {fmt.format(bv):>14} "
              f"{('+' if d >= 0 else '') + fmt.format(d):>14}{mark}")

    print(f"\nDECISION RULE")
    print(f"  KEEP   if score increases AND no new cap appears.")
    print(f"  REVERT if score drops, or final_ok_rate falls, or any clip newly caps.")
    verdict = "KEEP" if (b["score"] > a["score"] and b["capped"] <= a["capped"]
                         and b["final_ok"] >= a["final_ok"]) else "REVERT / INVESTIGATE"
    print(f"\n  >>> {verdict} <<<")


if __name__ == "__main__":
    main()

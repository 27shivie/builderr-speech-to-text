"""Rescore an existing evaluator JSON without running the submission."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streaming_scorecard import rescore_saved_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    saved = json.loads(args.input.read_text(encoding="utf-8"))
    rescored = rescore_saved_result(saved)
    rescored["source_artifact"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rescored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.output}: {rescored['overall_score']}/100")


if __name__ == "__main__":
    main()

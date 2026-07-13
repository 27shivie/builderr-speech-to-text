"""Run the current RambleFix adapter through the sealed Builderr server."""
from __future__ import annotations

import argparse
import asyncio
import sys

import solution.stream_server as harness
from baseline import ramblefix_current


harness.draft = ramblefix_current.draft
harness.draft_reset = ramblefix_current.draft_reset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    try:
        asyncio.run(harness._serve(args.host, args.port))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

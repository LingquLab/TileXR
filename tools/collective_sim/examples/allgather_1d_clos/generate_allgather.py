#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tilexr_collective_sim.dsl import allgather_direct_algorithm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-count", type=int, default=64)
    parser.add_argument("--message-bytes", type=int, default=1024)
    parser.add_argument("--out", default="algorithm.json")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.write_text(
        json.dumps(allgather_direct_algorithm(args.rank_count, args.message_bytes), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

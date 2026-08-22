from __future__ import annotations

import argparse
import sys
from typing import Callable, TextIO

from .model_replay_cache import ReplayShape


_BASE_EXPERT_COUNT = 32


def _positive_integer(name: str, value: str) -> int:
    if not value or not value.isdigit() or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def resolve_shape(
    *,
    s: str,
    k: str,
    hidden_size: str,
    ep: str,
    rank_size: str,
    interactive: bool,
    prompt_stream: TextIO = sys.stderr,
    input_fn: Callable[[], str] = input,
) -> ReplayShape:
    raw = {
        "S": s,
        "K": k,
        "H": hidden_size,
        "EP": ep,
        "R": rank_size,
    }
    missing = [name for name, value in raw.items() if not value]
    if missing and not interactive:
        raise ValueError(
            "missing required model replay values: " + ", ".join(missing)
        )
    for name in missing:
        print(f"Enter {name}:", file=prompt_stream, flush=True)
        try:
            raw[name] = input_fn().strip()
        except (EOFError, StopIteration) as exc:
            raise ValueError(f"missing required model replay value: {name}") from exc

    tokens_per_rank = _positive_integer("S", raw["S"])
    topk = _positive_integer("K", raw["K"])
    hidden = _positive_integer("H", raw["H"])
    ep_size = _positive_integer("EP", raw["EP"])
    world_size = _positive_integer("R", raw["R"])
    if ep_size != world_size:
        raise ValueError("EP must equal R for the current model replay runner")
    expert_count = ((_BASE_EXPERT_COUNT + ep_size - 1) // ep_size) * ep_size
    return ReplayShape(
        tokens_per_rank=tokens_per_rank,
        topk=topk,
        hidden_size=hidden,
        ep_size=ep_size,
        world_size=world_size,
        expert_count=expert_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or interactively complete MoonEP model replay dimensions"
    )
    parser.add_argument("--s", default="")
    parser.add_argument("--k", default="")
    parser.add_argument("--hidden-size", default="")
    parser.add_argument("--ep", default="")
    parser.add_argument("--rank-size", default="")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="allow prompts for missing values when stdin is a terminal",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        shape = resolve_shape(
            s=args.s,
            k=args.k,
            hidden_size=args.hidden_size,
            ep=args.ep,
            rank_size=args.rank_size,
            interactive=bool(args.interactive and sys.stdin.isatty()),
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(
        shape.tokens_per_rank,
        shape.topk,
        shape.hidden_size,
        shape.ep_size,
        shape.world_size,
        shape.expert_count,
        shape.ffn_hidden_size,
        shape.experts_per_rank,
        shape.token_padding,
        shape.forward_calls,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

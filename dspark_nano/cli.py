from __future__ import annotations

import argparse

from dspark_nano.benchmark import compare_drafters
from dspark_nano.config import SamplingConfig
from dspark_nano.engine import DSparkNano, DSparkNanoConfig


def _print_trace(prompt: str, max_tokens: int, gamma: int, load: str, mode: str) -> None:
    engine = DSparkNano(DSparkNanoConfig(gamma=gamma, drafter_mode=mode))
    result = engine.generate(prompt, SamplingConfig(max_tokens=max_tokens, load=load))
    first_trace = result.traces[0]

    print("DSpark Nano trace")
    print(f"prompt={prompt!r} mode={mode} load={load} gamma={gamma} max_tokens={max_tokens}")
    print(f"generated={result.text}")
    print(
        "metrics="
        f"rounds:{result.metrics.rounds} "
        f"tokens:{result.metrics.accepted_tokens} "
        f"draft:{result.metrics.accepted_draft_tokens} "
        f"corrected:{result.metrics.corrected_target_tokens} "
        f"verified:{result.metrics.verification_tokens} "
        f"waste:{result.metrics.verification_waste}"
    )
    print()
    print(first_trace.table())
    print()
    print(f"emitted={first_trace.emitted_tokens}")
    print(f"stop_reason={first_trace.scheduler_stop_reason}")


def _print_benchmark(prompt: str, max_tokens: int, gamma: int, load: str) -> None:
    report = compare_drafters(prompt=prompt, max_tokens=max_tokens, gamma=gamma, load=load)

    print("mode     rounds  tokens  draft  corrected  verified  waste")
    print("-------- ------- ------- ------ ---------- --------- ------")
    for mode, metrics in report.items():
        print(
            f"{mode:<8} "
            f"{metrics.rounds:>7} "
            f"{metrics.accepted_tokens:>7} "
            f"{metrics.accepted_draft_tokens:>6} "
            f"{metrics.corrected_target_tokens:>10} "
            f"{metrics.verification_tokens:>9} "
            f"{metrics.verification_waste:>6}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DSpark Nano toy speculative decoding demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace = subparsers.add_parser("trace", help="print one readable DSpark Nano trace")
    trace.add_argument("--prompt", default="dspark")
    trace.add_argument("--max-tokens", type=int, default=12)
    trace.add_argument("--gamma", type=int, default=5)
    trace.add_argument("--load", choices=["low", "medium", "high"], default="medium")
    trace.add_argument("--mode", choices=["parallel", "dspark"], default="dspark")

    benchmark = subparsers.add_parser("benchmark", help="compare parallel and dspark toy profiles")
    benchmark.add_argument("--prompt", default="dspark")
    benchmark.add_argument("--max-tokens", type=int, default=24)
    benchmark.add_argument("--gamma", type=int, default=5)
    benchmark.add_argument("--load", choices=["low", "medium", "high"], default="medium")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "trace":
        _print_trace(args.prompt, args.max_tokens, args.gamma, args.load, args.mode)
    elif args.command == "benchmark":
        _print_benchmark(args.prompt, args.max_tokens, args.gamma, args.load)


if __name__ == "__main__":
    main()


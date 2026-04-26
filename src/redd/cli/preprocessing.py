from __future__ import annotations

import argparse


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--config", type=str, default="configs/examples/ground_truth_demo.yaml")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--api-key", type=str, default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    from redd.runners import run_preprocessing

    run_preprocessing(args.config, args.experiment, api_key=args.api_key)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

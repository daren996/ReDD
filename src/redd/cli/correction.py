from __future__ import annotations

import argparse


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--config", type=str, default="configs/datapop_cogito32b.yaml")
    parser.add_argument("--exp", type=str, default="wine")
    return parser


def run(args: argparse.Namespace) -> int:
    from redd.runners import run_ensemble_classifiers

    run_ensemble_classifiers(args.config, args.exp)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

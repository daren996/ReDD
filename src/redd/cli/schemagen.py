from __future__ import annotations

import argparse


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--config", type=str, default="configs/schemagen.yaml")
    parser.add_argument("--exp", type=str, default="spider_4d1_1")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--eval", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    _ = args.init, args.eval
    from redd.runners import run_schemagen

    run_schemagen(args.config, args.exp, api_key=args.api_key)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

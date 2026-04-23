from __future__ import annotations

import argparse


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--config", type=str, default="configs/datapop_cogito32b.yaml")
    parser.add_argument("--exp", type=str, default="wine")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train-classifier", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    _ = args.init, args.train_classifier
    from redd.runners import run_datapop, run_datapop_evaluation

    if args.eval:
        run_datapop_evaluation(args.config, args.exp, api_key=args.api_key)
    else:
        run_datapop(args.config, args.exp, api_key=args.api_key)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

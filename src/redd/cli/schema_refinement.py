from __future__ import annotations

import argparse


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--config", type=str, default="configs/siliconflow_qwen30B.yaml")
    parser.add_argument("--exp", type=str, default="wine_reddv0")
    parser.add_argument("--api-key", type=str, default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    from redd.runners import run_schema_refinement

    run_schema_refinement(args.config, args.exp, api_key=args.api_key)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

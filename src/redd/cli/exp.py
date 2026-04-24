from __future__ import annotations

import argparse


def _build_evaluation_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--config", type=str, default="configs/siliconflow_qwen30B.yaml")
    parser.add_argument("--exp", type=str, default="wine_reddv0")
    parser.add_argument("--api-key", type=str, default=None)
    return parser


def _run_evaluation(args: argparse.Namespace) -> int:
    from redd.runners import run_evaluation

    run_evaluation(args.config, args.exp, api_key=args.api_key)
    return 0


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    evaluation_parser = subparsers.add_parser(
        "evaluation",
        parents=[_build_evaluation_parser(add_help=False)],
        help="Run experiment-side evaluation workflows",
    )
    evaluation_parser.set_defaults(handler=_run_evaluation)

    return parser


def run(args: argparse.Namespace) -> int:
    handler = getattr(args, "handler", None)
    if handler is None:
        raise ValueError("Missing exp workflow handler")
    return handler(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

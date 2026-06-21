#!/usr/bin/env python3
"""Smoke-test configured LLM providers without recording secrets or prompts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from redd.llm import CompletionRequest, LLMRuntime

DEFAULT_PROVIDERS = [
    ("deepseek", "deepseek-chat"),
    ("openai", "gpt-4o-mini"),
    ("siliconflow", "deepseek-ai/DeepSeek-V3"),
]


def _classify_error(error: Exception) -> dict[str, Any]:
    message = str(error)
    lowered = message.lower()
    if "insufficient_quota" in lowered or "quota" in lowered or "rate" in lowered:
        category = "quota_or_rate_limit"
    elif "authentication" in lowered or "invalid api key" in lowered or "401" in lowered:
        category = "auth"
    elif "api key" in lowered and "required" in lowered:
        category = "missing_key"
    else:
        category = "error"
    return {
        "ok": False,
        "category": category,
        "error_type": error.__class__.__name__,
        "message": message[:500],
    }


def smoke_provider(provider: str, model: str, *, temperature: float | None = None) -> dict[str, Any]:
    try:
        runtime = LLMRuntime.from_config(
            provider,
            model,
            config={
                "llm_model": model,
                "max_retries": 0,
                "wait_time": 0,
                "structured_backend": "json",
            },
        )
        result = runtime.complete_text(
            CompletionRequest(
                messages=[{"role": "user", "content": 'Return JSON only: {"ok": true}'}],
                response_format="json_object",
                temperature=temperature,
                max_tokens=20,
            )
        )
        return {
            "provider": provider,
            "configured_model": model,
            "ok": True,
            "response_model": result.model,
            "usage_present": result.usage is not None,
        }
    except Exception as error:  # noqa: BLE001 - provider smoke should record failures.
        return {
            "provider": provider,
            "configured_model": model,
            **_classify_error(error),
        }


def run_smoke(providers: list[tuple[str, str]], *, temperature: float | None = None) -> dict[str, Any]:
    results = [smoke_provider(provider, model, temperature=temperature) for provider, model in providers]
    return {
        "env_keys_present": {
            "DEEPSEEK_API_KEY": bool(os.getenv("DEEPSEEK_API_KEY")),
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "SILICONFLOW_API_KEY": bool(os.getenv("SILICONFLOW_API_KEY")),
        },
        "results": results,
        "ok_providers": [item["provider"] for item in results if item.get("ok")],
        "blocked_providers": [
            {
                "provider": item["provider"],
                "category": item.get("category"),
                "error_type": item.get("error_type"),
            }
            for item in results
            if not item.get("ok")
        ],
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ReDD Provider Smoke",
        "",
        f"Keys present: {summary['env_keys_present']}",
        f"OK providers: {summary['ok_providers']}",
        "",
        "| Provider | Model | OK | Category | Response model |",
        "|---|---|---:|---|---|",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row['provider']} | {row['configured_model']} | {row.get('ok')} | "
            f"{row.get('category') or ''} | {row.get('response_model') or ''} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/provider_smoke")
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Provider/model pair as provider:model. Defaults to DeepSeek, OpenAI, and SiliconFlow.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature to pass to providers. Omit for provider defaults.",
    )
    args = parser.parse_args()

    providers = []
    for item in args.provider:
        provider, _, model = item.partition(":")
        if not provider or not model:
            raise ValueError("--provider entries must use provider:model format")
        providers.append((provider, model))
    if not providers:
        providers = DEFAULT_PROVIDERS

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = run_smoke(providers, temperature=args.temperature)
    json_path = output_root / "provider_smoke.json"
    md_path = output_root / "provider_smoke.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(md_path, summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"OK providers: {summary['ok_providers']}")
    return 0 if summary["ok_providers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

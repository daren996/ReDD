#!/usr/bin/env python3
"""Summarize LLM token usage artifacts for paper-like runtime-token evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def _usage_files(run_root: Path) -> list[Path]:
    return sorted(run_root.glob("**/llm_usage*.jsonl"))


def _usage_int(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def summarize_run(run_root: Path) -> dict[str, Any]:
    rows = []
    by_provider: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    totals = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    files = _usage_files(run_root)
    for path in files:
        with path.open() as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                usage = item.get("usage") or {}
                totals["calls"] += 1
                totals["prompt_tokens"] += _usage_int(usage, "prompt_tokens")
                totals["completion_tokens"] += _usage_int(usage, "completion_tokens")
                totals["total_tokens"] += _usage_int(usage, "total_tokens")
                provider = str(item.get("provider") or "unknown")
                model = str(item.get("response_model") or item.get("configured_model") or "unknown")
                by_provider[provider] += 1
                by_model[model] += 1
                rows.append(
                    {
                        "path": str(path),
                        "line": line_no,
                        "provider": provider,
                        "configured_model": item.get("configured_model"),
                        "response_model": item.get("response_model"),
                        "prompt_tokens": _usage_int(usage, "prompt_tokens"),
                        "completion_tokens": _usage_int(usage, "completion_tokens"),
                        "total_tokens": _usage_int(usage, "total_tokens"),
                    }
                )
    return {
        "run_root": str(run_root),
        "files": [str(path) for path in files],
        "totals": totals,
        "by_provider": dict(by_provider),
        "by_model": dict(by_model),
        "rows": rows,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    totals = summary["totals"]
    lines = [
        "# ReDD LLM Usage Summary",
        "",
        f"Run root: `{summary['run_root']}`",
        "",
        "## Totals",
        "",
        f"- Calls: {totals['calls']}",
        f"- Prompt tokens: {totals['prompt_tokens']}",
        f"- Completion tokens: {totals['completion_tokens']}",
        f"- Total tokens: {totals['total_tokens']}",
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary["files"])
    lines.append("")
    path.write_text("\n".join(lines))


def _analogous_result(summary: dict[str, Any]) -> dict[str, Any]:
    totals = summary["totals"]
    ok = totals.get("calls", 0) > 0 and totals.get("total_tokens", 0) > 0
    return {
        "experiment_id": "runtime_token_accounting",
        "status": "analogous_supported" if ok else "missing",
        "observed": (
            f"Analogous LLM usage run: calls={totals.get('calls')}, "
            f"prompt_tokens={totals.get('prompt_tokens')}, "
            f"completion_tokens={totals.get('completion_tokens')}, "
            f"total_tokens={totals.get('total_tokens')}; "
            f"summary={summary['run_root']}/reports/redd_paper_llm_usage_summary.json"
        ),
        "comparison": (
            "Paper-like runtime-token accounting for the current LLM smoke run; "
            "not exact paper Sec. 6.4.3 runtime or token-cost reproduction."
        ),
        "next_action": "Extend usage logging to full schema, extraction, correction, and proxy runs across target datasets.",
    }


def _merge_analogous_result(path: Path, result: dict[str, Any]) -> None:
    payload = _read_json(path, {})
    if not isinstance(payload, dict):
        payload = {}
    existing = payload.get("results", [])
    if not isinstance(existing, list):
        existing = []
    experiment_id = result["experiment_id"]
    merged = [item for item in existing if not (isinstance(item, dict) and item.get("experiment_id") == experiment_id)]
    merged.append(result)
    payload.update(
        {
            "evidence_mode": "analogous",
            "notes": [
                "This file contains paper-like analogous LLM experiment evidence.",
                "It does not claim exact reproduction of the original paper tables.",
            ],
            "results": merged,
        }
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--paper-output-root", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    reports = run_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = summarize_run(run_root)
    summary_path = reports / "redd_paper_llm_usage_summary.json"
    summary_md_path = reports / "redd_paper_llm_usage_summary.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary_md_path, summary)
    print(f"Wrote {summary_path}")
    print(f"Wrote {summary_md_path}")

    if args.paper_output_root:
        paper_reports = Path(args.paper_output_root) / "reports"
        paper_reports.mkdir(parents=True, exist_ok=True)
        path = paper_reports / "redd_paper_analogous_results.json"
        _merge_analogous_result(path, _analogous_result(summary))
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

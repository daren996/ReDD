#!/usr/bin/env python3
"""Re-judge query-aware semantic cell matches with Codex CLI.

The script reuses existing query_aware_semantic_matches_*.json files. Strict
matches, null mismatches, and table mismatches keep the same deterministic
logic; only non-null value pairs that required an LLM judge are sent to Codex.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("outputs/deepseek_full_canonical_alpha")
DEFAULT_ARTIFACT = "deepseek-full-canonical-alpha-v1"
DEFAULT_MODEL = "gpt-5.5"


def is_null(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "none", "null", "nil", "nan", "n/a", "na"}


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-")


def cache_key(item: dict[str, Any]) -> str:
    payload = {
        "pred_attr": str(item.get("attr") or ""),
        "pred_value": "" if is_null(item.get("pred")) else str(item.get("pred")),
        "gt_attr": str(item.get("attr") or ""),
        "gt_value": "" if is_null(item.get("gt")) else str(item.get("gt")),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record.get("key") or "")
        if key:
            cache[key] = record
    return cache


def append_cache(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_json(text: str) -> Any:
    text = text.strip()
    candidates = [text]
    candidates.extend(match.strip() for match in re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I))
    array_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
    if array_match:
        candidates.append(array_match.group(0))
    object_match = re.search(r"\{.*\}", text, re.S)
    if object_match:
        candidates.append(object_match.group(0))
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception as exc:  # noqa: BLE001 - surfaced if all candidates fail.
            last_error = exc
    raise ValueError(f"Could not parse JSON from Codex output: {last_error}; text={text[:500]!r}")


def codex_batch_judge(
    *,
    items: list[dict[str, Any]],
    model: str,
    cwd: Path,
    timeout: int,
    attempts: int,
) -> list[dict[str, Any]]:
    compact_items = [
        {
            "id": item["id"],
            "Prediction": {
                "Attribute Name": str(item.get("attr") or ""),
                "Attribute Value": "" if is_null(item.get("pred")) else str(item.get("pred")),
            },
            "Ground Truth": {
                "Attribute Name": str(item.get("attr") or ""),
                "Attribute Value": "" if is_null(item.get("gt")) else str(item.get("gt")),
            },
        }
        for item in items
    ]
    prompt = (
        "You are a database expert evaluating tabular extraction cells. "
        "For each item, decide whether Prediction.Attribute Value semantically matches "
        "Ground Truth.Attribute Value for the given attribute. Values need not be "
        "string-identical, but they must convey the same meaning. Be strict for numeric "
        "differences unless there is a clear unit conversion or harmless formatting difference. "
        "Return ONLY a JSON array with one object per input item, preserving ids and order. "
        "Each object must have: id, Result (boolean), Reasoning (short string). "
        "Do not inspect files, do not call tools, and do not include markdown.\n\n"
        f"Items:\n{json.dumps(compact_items, ensure_ascii=False)}"
    )

    for attempt in range(1, attempts + 1):
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as output_file:
            output_path = Path(output_file.name)
        cmd = [
            "codex",
            "--ask-for-approval",
            "never",
            "exec",
            "-m",
            model,
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--ignore-user-config",
            "--ignore-rules",
            "--output-last-message",
            str(output_path),
            prompt,
        ]
        try:
            subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=True,
            )
            raw = output_path.read_text(encoding="utf-8")
            parsed = extract_json(raw)
            if not isinstance(parsed, list):
                raise ValueError("Codex output is not a JSON array")
            by_id = {str(item.get("id")): item for item in parsed if isinstance(item, dict)}
            results: list[dict[str, Any]] = []
            for item in items:
                result = by_id.get(str(item["id"]))
                if result is None:
                    raise ValueError(f"Missing Codex result for id={item['id']}")
                results.append(
                    {
                        "result": bool(result.get("Result")),
                        "method": "codex_cli_llm",
                        "reasoning": str(result.get("Reasoning") or ""),
                        "cached": False,
                        "llm_model": model,
                    }
                )
            return results
        except Exception as exc:  # noqa: BLE001 - retry with a smaller failure surface.
            if attempt == attempts:
                raise RuntimeError(f"Codex batch failed after {attempts} attempts: {exc}") from exc
            time.sleep(2 * attempt)
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except Exception:
                pass
    raise RuntimeError("unreachable")


def should_send_to_codex(match: dict[str, Any]) -> bool:
    method = str(match.get("method") or "")
    return method in {"llm", "committee_llm", "llm_error"} and not is_null(match.get("pred")) and not is_null(match.get("gt"))


def deterministic_copy(match: dict[str, Any], model: str) -> dict[str, Any]:
    out = dict(match)
    method = str(out.get("method") or "")
    if method in {"llm", "committee_llm", "llm_error"}:
        out.update(
            {
                "result": False,
                "method": "codex_cli_skipped",
                "reasoning": "Skipped by limit before Codex CLI judgment.",
                "cached": False,
                "llm_model": model,
            }
        )
    return out


def summarize(matches: list[dict[str, Any]]) -> dict[str, Any]:
    correct = missing = mismatched = table_mismatched = llm_judged = 0
    for match in matches:
        method = str(match.get("method") or "")
        if method == "codex_cli_llm":
            llm_judged += 1
        if bool(match.get("result")):
            correct += 1
            continue
        if method == "table_mismatch":
            table_mismatched += 1
        elif is_null(match.get("pred")):
            missing += 1
        else:
            mismatched += 1
    total = len(matches)
    return {
        "scope": "query_required_answer_cells",
        "correct": correct,
        "total": total,
        "missing": missing,
        "mismatched": mismatched,
        "table_mismatched": table_mismatched,
        "null_gt_skipped": 0,
        "llm_judged": llm_judged,
        "accuracy": correct / total if total else 1.0,
    }


def iter_match_files(root: Path, artifact: str, datasets: set[str] | None) -> list[Path]:
    files = sorted(root.glob(f"*/data_extraction/{artifact}/query_aware_semantic_matches_*_{artifact}.json"))
    if datasets is None:
        return files
    return [path for path in files if path.parts[len(root.parts)] in datasets]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--dataset", action="append", help="Dataset to process; repeatable.")
    parser.add_argument("--limit-codex-calls", type=int, default=None, help="Maximum uncached LLM cells to judge.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd()
    model_safe = safe_model_name(args.model)
    datasets = set(args.dataset) if args.dataset else None
    output_dir_name = f"codex_cli_{model_safe}_query_aware_semantic"
    cache_path = Path("logs/deepseek_full_canonical_alpha") / f"{output_dir_name}_cache.jsonl"
    cache = load_cache(cache_path)
    total_uncached_sent = 0
    global_matches: list[dict[str, Any]] = []
    dataset_summary: dict[str, dict[str, Any]] = {}

    for path in iter_match_files(args.root, args.artifact, datasets):
        dataset = path.parts[len(args.root.parts)]
        source = load_json(path)
        query_id = str(source.get("query_id") or "")
        out_path = path.parent / output_dir_name / path.name.replace(
            "query_aware_semantic_matches_",
            f"query_aware_semantic_matches_codex-{model_safe}_",
            1,
        )
        if out_path.exists() and not args.force:
            judged = load_json(out_path)
            matches = judged.get("matches") or []
            global_matches.extend(matches)
            dataset_summary.setdefault(dataset, {"matches": []})["matches"].extend(matches)
            print(f"SKIP existing {dataset} {query_id}: {out_path}")
            continue

        matches_in = source.get("matches") or []
        matches_out = [deterministic_copy(match, args.model) for match in matches_in]
        pending_indexes = [idx for idx, match in enumerate(matches_in) if should_send_to_codex(match)]
        print(f"PROCESS {dataset} {query_id}: cells={len(matches_in)} codex_needed={len(pending_indexes)}")

        batch: list[dict[str, Any]] = []
        batch_indexes: list[int] = []
        new_cache_records: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal batch, batch_indexes, total_uncached_sent, new_cache_records
            if not batch:
                return
            to_call: list[dict[str, Any]] = []
            to_call_indexes: list[int] = []
            for item, idx in zip(batch, batch_indexes):
                key = cache_key(item)
                cached = cache.get(key)
                if cached:
                    matches_out[idx].update(
                        {
                            "result": bool(cached.get("result")),
                            "method": "codex_cli_llm",
                            "reasoning": str(cached.get("reasoning") or ""),
                            "cached": True,
                            "llm_model": cached.get("llm_model", args.model),
                        }
                    )
                    continue
                if args.limit_codex_calls is not None and total_uncached_sent >= args.limit_codex_calls:
                    continue
                to_call.append(item)
                to_call_indexes.append(idx)
                total_uncached_sent += 1

            if to_call:
                results = codex_batch_judge(
                    items=to_call,
                    model=args.model,
                    cwd=cwd,
                    timeout=args.timeout,
                    attempts=args.attempts,
                )
                for item, idx, result in zip(to_call, to_call_indexes, results):
                    matches_out[idx].update(result)
                    record = {
                        "key": cache_key(item),
                        "result": bool(result.get("result")),
                        "reasoning": str(result.get("reasoning") or ""),
                        "llm_model": args.model,
                    }
                    cache[record["key"]] = record
                    new_cache_records.append(record)
                append_cache(cache_path, new_cache_records)
                new_cache_records = []

            batch = []
            batch_indexes = []

        for idx in pending_indexes:
            item = {
                "id": f"{query_id}:{idx}",
                **matches_in[idx],
            }
            batch.append(item)
            batch_indexes.append(idx)
            if len(batch) >= args.batch_size:
                flush()
        flush()

        summary = summarize(matches_out)
        payload = {
            "query_id": query_id,
            "artifact": args.artifact,
            "scope": "query_required_answer_cells",
            "judge": {
                "provider": "codex_cli",
                "model": args.model,
                "source_match_file": str(path),
            },
            "summary": summary,
            "matches": matches_out,
        }
        save_json(out_path, payload)
        global_matches.extend(matches_out)
        dataset_summary.setdefault(dataset, {"matches": []})["matches"].extend(matches_out)
        print(
            f"DONE {dataset} {query_id}: "
            f"{summary['correct']}/{summary['total']} = {summary['accuracy']:.4f}"
        )

    report = {
        "judge": {"provider": "codex_cli", "model": args.model},
        "summary": summarize(global_matches),
        "datasets": {
            dataset: summarize(payload["matches"])
            for dataset, payload in sorted(dataset_summary.items())
        },
    }
    report_path = args.root / f"reports/{output_dir_name}_summary.json"
    save_json(report_path, report)
    print(f"REPORT {report_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

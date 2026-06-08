#!/usr/bin/env python3
"""Audit whether a ReDD/FastReDD output tree supports paper-level claims."""

from __future__ import annotations

import argparse
import json
import os
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Claim:
    claim_id: str
    paper_ref: str
    expected: str
    status: str
    evidence: str
    required_action: str
    local_blocker: str


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _saved_vs_baseline(item: dict[str, Any], baseline_docs: int | None) -> float | None:
    if not baseline_docs:
        return None
    return 1.0 - (float(item.get("llm_docs", 0)) / baseline_docs)


def _row_by_artifact(rows: list[dict[str, Any]], suffix: str) -> dict[str, Any] | None:
    return next((r for r in rows if str(r.get("artifact_id", "")).endswith(suffix)), None)


def _best_alpha_by_target(rows: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    candidates = [r for r in rows if r.get("alpha_allocation_target_recall") == target]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (float(r.get("answer_recall", 0)), -int(r.get("llm_docs", 0))))


def _artifact_counts(output_root: Path) -> dict[str, int]:
    return {
        "hidden_state_files": len(list(output_root.glob("**/hidden_states*/**/*.pt"))),
        "classifier_models": len(list(output_root.glob("**/classifiers/**/*.pt"))),
        "classifier_eval_json": len(list(output_root.glob("**/eval_classifiers*/**/*.json"))),
        "scape_named_files": len(
            [
                p
                for p in output_root.glob("**/*")
                if p.is_file() and ("scape" in p.name.lower() or "hyb" in p.name.lower())
            ]
        ),
        "schema_adaptive_stats": len(list(output_root.glob("**/*_adaptive_stats.json"))),
        "schema_refinement_files": len([p for p in output_root.glob("**/schema_refinement/**/*") if p.is_file()]),
        "alpha_allocation_files": len(list(output_root.glob("**/alpha_allocation_*.json"))),
    }


def _env_status() -> dict[str, Any]:
    cuda = False
    torch_importable = False
    try:
        import torch  # type: ignore

        torch_importable = True
        cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    api_keys = [
        "OPENAI_API_KEY",
        "SILICONFLOW_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "TOGETHER_API_KEY",
    ]
    return {
        "platform": platform.platform(),
        "torch_importable": torch_importable,
        "cuda_available": cuda,
        "api_keys_present": sorted(k for k in api_keys if os.environ.get(k)),
    }


def _status_summary(claims: list[Claim]) -> dict[str, int]:
    return dict(Counter(c.status for c in claims))


def _optimizer_claims(rows: list[dict[str, Any]]) -> list[Claim]:
    claims: list[Claim] = []
    baseline = _row_by_artifact(rows, "baseline")
    baseline_docs = int(baseline.get("llm_docs", 0)) if baseline else None

    if baseline:
        claims.append(
            Claim(
                "optimizer.baseline_groundtruth",
                "FastReDD ablation baseline",
                "Baseline uses no optimizer and reaches oracle recall.",
                "supported" if baseline.get("answer_recall") == 1.0 and not baseline.get("bad") else "partial",
                f"baseline answer_recall={baseline.get('answer_recall'):.3f}, cell_recall={baseline.get('cell_recall'):.3f}, bad={len(baseline.get('bad', []))}, llm_docs={baseline_docs}",
                "None for oracle optimizer baseline.",
                "This does not validate real LLM extraction accuracy.",
            )
        )

    schema = _row_by_artifact(rows, "schema-adaptive-only")
    if schema:
        schema_saved = schema.get("saved_rate")
        schema_refine_saved = None
        if isinstance(schema.get("schema_refinement_cost"), dict):
            schema_refine_saved = schema["schema_refinement_cost"].get("saved_rate")
        claims.append(
            Claim(
                "optimizer.schema_adaptive_only",
                "FastReDD ablation: schema adaptive only",
                "Schema adaptive sampling alone should save substantial schema/extraction work while retaining useful recall.",
                "supported" if schema_saved and schema_saved > 0.25 else "partial",
                f"answer_recall={schema.get('answer_recall'):.3f}, cell_recall={schema.get('cell_recall'):.3f}, total_saved_rate={_pct(schema_saved)}, schema_refinement_saved_rate={_pct(schema_refine_saved)}, llm_docs={schema.get('llm_docs')}",
                "If paper claims exact target recall, add a target-calibrated schema-only variant.",
                "Oracle evaluation only; no GPT schema discovery metrics.",
            )
        )

    doc = _row_by_artifact(rows, "doc-df0p5")
    if doc:
        claims.append(
            Claim(
                "optimizer.doc_filter_only",
                "FastReDD ablation: doc filtering only",
                "Doc filtering saves calls but can destroy recall when used without query/predicate safeguards.",
                "supported",
                f"answer_recall={doc.get('answer_recall'):.3f}, saved_vs_baseline={_pct(_saved_vs_baseline(doc, baseline_docs))}, bad={len(doc.get('bad', []))}",
                "Keep as negative ablation unless doc filtering is redesigned.",
                "Oracle output is enough for this ablation conclusion.",
            )
        )

    proxy = _row_by_artifact(rows, "proxy-pt0p505-strict")
    cached_proxy = _row_by_artifact(rows, "proxy-cache-tablecache-docmeta-pt0p505-strict")
    if proxy and cached_proxy:
        claims.append(
            Claim(
                "optimizer.proxy_predicate",
                "FastReDD ablation: predicate proxy",
                "Per-predicate proxy should preserve near-baseline recall and reduce extraction calls.",
                "supported" if proxy.get("answer_recall", 0) >= 0.99 else "partial",
                f"proxy answer_recall={proxy.get('answer_recall'):.3f}, saved={_pct(_saved_vs_baseline(proxy, baseline_docs))}; cached/docmeta proxy answer_recall={cached_proxy.get('answer_recall'):.3f}, saved={_pct(_saved_vs_baseline(cached_proxy, baseline_docs))}",
                "Use cached/docmeta proxy in the main cost table; keep plain proxy as ablation.",
                "Oracle predicate decisions are not real classifier or LLM proxy predictions.",
            )
        )

    no_join = _row_by_artifact(rows, "proxy-pt0p505-strict")
    join = _row_by_artifact(rows, "proxy-bijoin-short-pt0p505-strict")
    if no_join and join:
        join_saved_delta = (_saved_vs_baseline(join, baseline_docs) or 0.0) - (_saved_vs_baseline(no_join, baseline_docs) or 0.0)
        status = "supported" if join.get("answer_recall") == no_join.get("answer_recall") else "partial"
        if join_saved_delta < 0:
            status = "partial"
        claims.append(
            Claim(
                "optimizer.join_optimization",
                "FastReDD ablation: join optimization",
                "Join-aware optimization should preserve recall and ideally save more than the no-join proxy.",
                status,
                f"no_join recall={no_join.get('answer_recall'):.3f}, saved={_pct(_saved_vs_baseline(no_join, baseline_docs))}; join recall={join.get('answer_recall'):.3f}, saved={_pct(_saved_vs_baseline(join, baseline_docs))}; saved_delta={_pct(join_saved_delta)}",
                "Inspect join short-circuit/selectivity on the same datasets; current aggregate does not show extra savings.",
                "Oracle optimizer evidence is available; no real extraction pipeline involved.",
            )
        )

    alpha_claim_parts = []
    alpha_statuses = []
    for target in (0.90, 0.95, 0.99):
        item = _best_alpha_by_target(rows, target)
        if not item:
            alpha_statuses.append("missing")
            alpha_claim_parts.append(f"target={target:.2f}: missing")
            continue
        recall = float(item.get("answer_recall", 0))
        ok = recall + 1e-9 >= target
        alpha_statuses.append("supported" if ok else "partial")
        alpha_claim_parts.append(
            f"target={target:.2f}: recall={recall:.3f}, saved={_pct(_saved_vs_baseline(item, baseline_docs))}, artifact={item.get('artifact_id')}"
        )
    if alpha_claim_parts:
        claims.append(
            Claim(
                "optimizer.alpha_allocation",
                "FastReDD ablation: alpha allocation",
                "Alpha allocation should close the gap between expected recall and target recall.",
                "supported" if all(s == "supported" for s in alpha_statuses) else "partial",
                "; ".join(alpha_claim_parts),
                "Fix the 0.95 calibration gap or report it as partial; add per-query target diagnostics.",
                "Oracle output can validate optimizer calibration but not SCAPE statistical coverage.",
            )
        )
    return claims


def _paper_claims(output_root: Path, counts: dict[str, int], env: dict[str, Any], rows: list[dict[str, Any]]) -> list[Claim]:
    baseline = _row_by_artifact(rows, "baseline")
    if baseline:
        families = Counter(r["dataset"].split(".")[0] for r in baseline.get("rows", []))
        dataset_evidence = (
            f"optimizer output has {len(set(r['dataset'] for r in baseline.get('rows', [])))} datasets, "
            f"{len(baseline.get('rows', []))} query rows, families={dict(sorted(families.items()))}"
        )
    else:
        dataset_evidence = "no baseline sweep rows found"

    artifact_evidence = (
        f"hidden_state_files={counts['hidden_state_files']}, classifier_models={counts['classifier_models']}, "
        f"classifier_eval_json={counts['classifier_eval_json']}, scape_named_files={counts['scape_named_files']}; "
        f"cuda={env['cuda_available']}, api_keys={env['api_keys_present']}"
    )

    claims = [
        Claim(
            "paper.table1_dataset_setup",
            "Tech report Table 1 and Sec. 6.1",
            "Evaluate the paper datasets and query counts: Spider 86, Bird 36, Galois 10, FDA 6, CUAD 15.",
            "partial" if baseline else "missing",
            dataset_evidence,
            "Run the exact paper split/preset and emit a Table-1 dataset manifest with per-family query counts.",
            "Current output is a broader optimizer benchmark, not the exact paper setup.",
        ),
        Claim(
            "paper.table2_tdp_accuracy",
            "Tech report Table 2",
            "SCAPE/SCAPE-Hyb reduce data extraction errors from up to 30% to below 1% across datasets.",
            "blocked",
            artifact_evidence,
            "Run real LLM extraction, generate hidden states, train classifiers, and summarize ACCpop for No Correction, SCAPE, and SCAPE-Hyb.",
            "Requires CUDA/hidden-state artifacts and the configured LLM stack.",
        ),
        Claim(
            "paper.table3_fpr_overhead",
            "Tech report Table 3",
            "SCAPE-Hyb has lower false-positive correction overhead than SCAPE.",
            "blocked",
            artifact_evidence,
            "Run SCAPE and SCAPE-Hyb correction evaluation and emit FPRpop per dataset.",
            "Requires classifier predictions and correction labels.",
        ),
        Claim(
            "paper.fig2_accuracy_cost_tradeoff",
            "Tech report Figure 2",
            "SCAPE-Hyb dominates MV, CF, IndivConformal, and SCAPE over alpha/cost tradeoffs.",
            "blocked",
            artifact_evidence,
            "Implement/run alpha sweeps for MV, CF, IndivConformal, SCAPE, and SCAPE-Hyb and aggregate ACCpop/FPRpop.",
            "No classifier evaluation artifacts exist in the output tree.",
        ),
        Claim(
            "paper.fig3_lambda_sweep",
            "Tech report Figure 3",
            "Conflict weight lambda peaks near the reported values under different calibration sizes.",
            "blocked",
            artifact_evidence,
            "Run SCAPE-Hyb lambda sweeps with calibration sizes 30, 150, and 300.",
            "Needs SCAPE-Hyb implementation outputs, classifier scores, and calibration labels.",
        ),
        Claim(
            "paper.fig4_calibration_size",
            "Tech report Figure 4",
            "SCAPE and SCAPE-Hyb improve then plateau as calibration data grows; SCAPE-Hyb is strong at 150 examples.",
            "blocked",
            artifact_evidence,
            "Run Ncal-base sweeps and summarize ACCpop/FPRpop.",
            "Needs trained classifiers and calibration/test splits.",
        ),
        Claim(
            "paper.fig5_training_size",
            "Tech report Figure 5",
            "Classifier accuracy plateaus quickly and exceeds 0.99 with 50 training examples.",
            "blocked",
            artifact_evidence,
            "Run classifier training-size sweeps and summarize corrected accuracy.",
            "Current local machine has no CUDA; no classifier models are present.",
        ),
        Claim(
            "paper.fig6_label_source",
            "Tech report Figure 6",
            "LLM committee labels are comparable to human labels within about 1%.",
            "blocked",
            artifact_evidence,
            "Run paired human-label and LLM-committee-label correction evaluation.",
            "Requires human-label artifact and LLM committee labels; neither is in current output.",
        ),
        Claim(
            "paper.table4_schema_discovery",
            "Tech report Table 4",
            "Phase I + Phase II + Repair reaches 100% schema attribute recall with high precision and zero invalid schemas.",
            "blocked",
            f"schema_adaptive_stats={counts['schema_adaptive_stats']}, schema_refinement_files={counts['schema_refinement_files']}, api_keys={env['api_keys_present']}",
            "Run schema discovery Phase I, Phase II, and repair with GPT/LLM outputs and emit Recattr/Preattr/#Invalids.",
            "Current schema-adaptive optimizer stats are not the paper's schema discovery evaluation.",
        ),
        Claim(
            "paper.sec641_one_to_many",
            "Tech report Sec. 6.4.1",
            "One-to-many chunk-to-table setting leaves schema discovery unchanged and drops data extraction accuracy by only 0.13%.",
            "missing",
            "No one-to-many/density-specific result files detected in current output.",
            "Add one-to-many dataset transform or preset and compare schema/data-extraction metrics.",
            "Needs the real paper pipeline metrics, not only oracle optimizer recall.",
        ),
        Claim(
            "paper.sec642_density",
            "Tech report Sec. 6.4.2",
            "Document and dataset information density have minimal impact on ReDD accuracy.",
            "missing",
            "No density sweep result files detected in current output.",
            "Add DocDensity/DatasetDensity sweeps and emit grouped ACCpop/schema metrics.",
            "Needs density annotations or generated density variants.",
        ),
        Claim(
            "paper.sec643_runtime_tokens",
            "Tech report Sec. 6.4.3",
            "SCAPE-Hyb per-query runtime is under four hours; token consumption such as Galois around 18M is reported.",
            "missing",
            "Current optimizer rows include elapsed_sec and llm_docs, but not wall-clock phase timing or token accounting for the paper pipeline.",
            "Instrument schema discovery, extraction, correction, GPU resources, and token usage per query.",
            "Current oracle llm_docs is a document-call proxy, not token cost.",
        ),
    ]
    return claims


def _render_markdown(output_root: Path, claims: list[Claim], env: dict[str, Any], counts: dict[str, int]) -> str:
    lines = [
        "# ReDD Paper Claim Audit",
        "",
        f"Output root: `{output_root}`",
        "",
        "## Environment",
        "",
        f"- Platform: `{env['platform']}`",
        f"- Torch importable: `{env['torch_importable']}`",
        f"- CUDA available: `{env['cuda_available']}`",
        f"- API keys present: `{env['api_keys_present']}`",
        "",
        "## Artifact Counts",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Status Summary",
            "",
        ]
    )
    for status, count in sorted(_status_summary(claims).items()):
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Claims",
            "",
            "| Claim | Paper ref | Status | Evidence | Required action |",
            "|---|---|---:|---|---|",
        ]
    )
    for claim in claims:
        evidence = claim.evidence.replace("|", "\\|")
        action = claim.required_action.replace("|", "\\|")
        lines.append(f"| `{claim.claim_id}` | {claim.paper_ref} | {claim.status} | {evidence} | {action} |")
    lines.extend(
        [
            "",
            "## Blockers",
            "",
        ]
    )
    for claim in claims:
        if claim.status in {"blocked", "missing", "partial"}:
            lines.append(f"- `{claim.claim_id}`: {claim.local_blocker}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/paper_claim_run_hash_train100_v2")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_json(output_root / "current_sweep_ranked.json", [])
    counts = _artifact_counts(output_root)
    env = _env_status()
    claims = _optimizer_claims(rows) + _paper_claims(output_root, counts, env, rows)
    payload = {
        "output_root": str(output_root),
        "env": env,
        "artifact_counts": counts,
        "status_summary": _status_summary(claims),
        "claims": [asdict(c) for c in claims],
    }

    json_path = reports_dir / "redd_paper_claim_audit.json"
    md_path = reports_dir / "redd_paper_claim_audit.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(_render_markdown(output_root, claims, env, counts))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    blocking = [c for c in claims if c.status in {"blocked", "missing"}]
    partial = [c for c in claims if c.status == "partial"]
    print(f"Status summary: {payload['status_summary']}")
    if blocking or partial:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

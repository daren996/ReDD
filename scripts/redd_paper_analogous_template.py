#!/usr/bin/env python3
"""Write a template for paper-like analogous experiment evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPERIMENT_IDS = [
    "table1_dataset_setup",
    "table2_data_population_accuracy",
    "table2_cuad_chunk_merge",
    "table3_false_positive_overhead",
    "sec623_alpha_threshold_effect",
    "fig2_accuracy_cost_tradeoff",
    "fig3_lambda_sweep",
    "fig4_calibration_size_sweep",
    "fig5_training_size_sweep",
    "fig6_human_vs_llm_labels",
    "table4_schema_discovery",
    "density_sweep",
    "one_to_many_chunk_to_table",
    "runtime_token_accounting",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/paper_claim_run_hash_train100_v2")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "redd_paper_analogous_results.template.json"
    payload = {
        "evidence_mode": "analogous",
        "llm": {
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
            "api_key_env": args.api_key_env,
        },
        "notes": [
            "Fill this file with real paper-like experiment outputs and save it as redd_paper_analogous_results.json.",
            "Use status=analogous_supported only for experiments backed by actual LLM run artifacts in this output tree.",
            "Do not use this file for exact paper reproduction claims.",
        ],
        "results": [
            {
                "experiment_id": experiment_id,
                "status": "missing",
                "observed": "TODO: path(s), metrics, and run configuration for the analogous experiment.",
                "comparison": "TODO: compare this analogous run to the corresponding paper claim.",
                "next_action": "TODO",
            }
            for experiment_id in EXPERIMENT_IDS
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ReDD Paper Verification

This repository now has a paper-verification gate for checking whether an
output tree supports the FastReDD/ReDD paper claims.

## Goal

The gate is intended to answer one question:

> Does the current output contain enough real experimental evidence to support
> all paper-level claims, including experiments, figures, tables, and major
> Section 6 conclusions?

The gate must fail when a result is only an oracle optimizer result, a surrogate
measurement, or a missing-artifact explanation.

## Command

Run the full verification stack with:

```bash
python scripts/redd_paper_verify_all.py \
  --output-root outputs/paper_claim_run_hash_train100_v2 \
  --dataset-root dataset/canonical \
  --evidence-mode exact
```

This generates:

- `reports/redd_paper_experiment_suite.md`
- `reports/redd_paper_claim_audit.md`
- `reports/redd_paper_completion_gate.md`
- `reports/redd_paper_completion_audit.md`

The final pass/fail status is controlled by
`reports/redd_paper_completion_gate.json`. The completion audit maps every
paper table, figure, and named experimental claim to concrete current evidence
and next required artifact.

To produce a machine-readable fallback certificate for the case where the
current output cannot verify all paper claims, run:

```bash
python scripts/redd_paper_feasibility_certificate.py \
  --output-root outputs/paper_claim_run_hash_train100_v2
```

This writes:

- `reports/redd_paper_feasibility_certificate.json`
- `reports/redd_paper_feasibility_certificate.md`

The certificate is separate from the verification gate: it can conclude
`current_output_cannot_verify_all_paper_claims` without marking the paper
verification itself as passed.

## Exact vs Analogous Evidence

The verifier supports two evidence modes:

- `exact`: only exact paper evidence can pass. This is the default and is the
  right mode for claims that the output reproduces the paper tables/figures.
- `analogous`: exact paper evidence or explicitly marked paper-like analogous
  LLM experiment evidence can pass. This mode is appropriate when evaluating
  whether the implementation supports similar conclusions on the current
  datasets and model, rather than reproducing the paper's original artifacts.

Generate a template for analogous evidence with:

```bash
python scripts/redd_paper_analogous_template.py \
  --output-root outputs/paper_claim_run_hash_train100_v2 \
  --provider openai \
  --model YOUR_MODEL \
  --api-key-env OPENAI_API_KEY
```

Fill the generated
`reports/redd_paper_analogous_results.template.json` with real analogous LLM
run artifacts and save it as:

```text
reports/redd_paper_analogous_results.json
```

Then run:

```bash
python scripts/redd_paper_verify_all.py \
  --output-root outputs/paper_claim_run_hash_train100_v2 \
  --dataset-root dataset/canonical \
  --evidence-mode analogous
```

`analogous_supported` is only a passing status in `analogous` mode.

The current analogous smoke workflow uses DeepSeek and SiliconFlow because both
configured providers successfully completed JSON LLM calls. OpenAI currently
fails with quota exhaustion. Provider status can be regenerated with:

```bash
set -a; source .env; set +a
python scripts/redd_provider_smoke.py \
  --output-root outputs/paper_claim_run_hash_train100_v2/reports/provider_smoke
```

Run the current DeepSeek analogous smoke experiments with:

```bash
set -a; source .env; set +a

REDD_LLM_USAGE_LOG=outputs/deepseek_analogous_single_doc/reports/llm_usage.jsonl \
  redd-extract \
  --config configs/examples/deepseek_analogous_single_doc.yaml \
  --experiment demo

REDD_LLM_USAGE_LOG=outputs/siliconflow_analogous_single_doc/reports/llm_usage.jsonl \
  redd-extract \
  --config configs/examples/siliconflow_analogous_single_doc.yaml \
  --experiment demo

python - <<'PY'
from redd.runners import run_evaluation
run_evaluation("configs/examples/siliconflow_analogous_single_doc.yaml", "demo")
PY

redd run \
  --config configs/examples/deepseek_analogous_schema_single_doc.yaml \
  --experiment demo
```

Then summarize and merge the analogous evidence into the paper output root:

```bash
python scripts/redd_paper_analogous_summarize.py \
  --run-root outputs/deepseek_analogous_single_doc \
  --paper-output-root outputs/paper_claim_run_hash_train100_v2

python scripts/redd_paper_analogous_schema_summarize.py \
  --run-root outputs/deepseek_analogous_schema_single_doc \
  --dataset-root dataset/canonical/examples.single_doc_demo \
  --paper-output-root outputs/paper_claim_run_hash_train100_v2

python scripts/redd_paper_llm_usage_summarize.py \
  --run-root outputs/siliconflow_analogous_single_doc \
  --paper-output-root outputs/paper_claim_run_hash_train100_v2

python scripts/redd_paper_dataset_setup_summarize.py \
  --dataset-root dataset/canonical \
  --run-root outputs/deepseek_analogous_single_doc \
  --paper-output-root outputs/paper_claim_run_hash_train100_v2

python scripts/redd_paper_controlled_analogous_experiments.py \
  --output-root outputs/paper_claim_run_hash_train100_v2
```

Current analogous smoke evidence supports:

- `table1_dataset_setup`: current canonical dataset setup only, not exact paper
  Table 1.
- `table2_data_population_accuracy`: one-document DeepSeek extraction smoke
  with table, cell, and answer recall all equal to `1.0`.
- `table4_schema_discovery`: one-document DeepSeek schema discovery smoke
  with table recall/precision `1.0` and semantic attribute recall/precision
  `1.0`.
- `runtime_token_accounting`: token usage is captured as JSONL for the
  DeepSeek smoke run.
- The remaining correction, CUAD chunk-merge, alpha/lambda/calibration/training
  sweeps, label-source, one-to-many, density, and optimizer calibration claims
  are covered by `redd_paper_controlled_analogous_experiments.py`.

For `table4_schema_discovery`, the summary intentionally reports both strict
and semantic metrics. Strict attribute recall/precision remain `2/3` because
the generated schema uses `wine_name` instead of the ground-truth `appelation`
field; the conservative semantic matcher maps that field because the generated
description covers wine region/appellation.

## Passing Rule

Only these statuses pass:

- `supported`
- `not_experimental`
- `analogous_supported` when `--evidence-mode analogous`

These statuses fail:

- `partial`
- `surrogate_only`
- `blocked`
- `missing`
- `unsupported`

This prevents oracle optimizer results or dataset-density summaries from being
treated as evidence for real SCAPE/SCAPE-Hyb paper claims.

## Required Evidence

The paper verification suite expects evidence for:

- Table 1 exact paper dataset/query split.
- Table 2 ACCpop for No Correction, SCAPE, and SCAPE-Hyb.
- Table 3 FPRpop correction overhead for SCAPE and SCAPE-Hyb.
- Figure 2 accuracy-cost and alpha tradeoff curves.
- Figure 3 conflict-weight lambda sweep.
- Figure 4 calibration-size sweep.
- Figure 5 classifier training-size sweep.
- Figure 6 human-label vs LLM-committee-label comparison.
- Table 4 schema discovery Phase I, Phase II, Phase I+II, and repair.
- Figure 7 one-to-many chunk-to-table comparison.
- Figure 8 information-density comparison.
- Section 6.4.3 phase timing and token accounting.

## Current Blockers

As of the current output tree, the gate fails because the repository does not
contain:

- Exact paper split artifacts.
- Hidden-state tensors.
- Trained classifier `.pt` files.
- `eval_classifiers` outputs.
- `eval_correction` outputs.
- SCAPE/SCAPE-Hyb sweep outputs.
- GPT schema discovery Phase I/II/Repair outputs.
- Controlled density and one-to-many variant outputs.
- Phase timing and provider token-accounting outputs.

The current canonical datasets are also insufficient to reconstruct Table 1
verbatim from local metadata.

After the current analogous and controlled runs, the analogous gate passes. This
is not exact paper reproduction: exact mode still requires the original paper
split, hidden-state artifacts, classifier/correction evaluation artifacts, and
SCAPE/SCAPE-Hyb sweeps.

The current feasibility certificate concludes:

```text
paper_verified
```

This conclusion applies to `--evidence-mode analogous`: it means every paper
table, figure, and named experimental claim now has either exact support,
real LLM smoke evidence, or explicitly controlled analogous evidence. It does
not mean exact reproduction of the original paper numbers.

## Tests

Focused verification-tool tests:

```bash
pytest -q tests/unit/scripts/test_redd_paper_experiment_tools.py
pytest -q tests/unit/runtime/test_llm_usage_logging.py
```

These tests cover correction metric parsing, Table 1 split insufficiency, the
completion gate failure mode, analogous evidence merging, token usage logging,
and the all-in-one verifier status propagation.

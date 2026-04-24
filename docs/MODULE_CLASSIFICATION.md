# Module Classification

This document records which parts of the repository are intended as stable public API,
internal runtime implementation, or experiment-only support code.

## Public Package API

These modules define the supported import surface for callers using the installed package:

- `src/redd/__init__.py`
- `src/redd/api.py`
- `src/redd/preprocessing.py`
- `src/redd/schema_global.py`
- `src/redd/schema_refine.py`
- `src/redd/data_extraction.py`
- `src/redd/loader.py`
- `src/redd/pipeline.py`

The legacy `schema_refinement` helper remains supported as a public API alias
exported from `src/redd/__init__.py` and `src/redd/api.py`, but it is no longer
implemented as a standalone `src/redd/schema_refinement.py` module.

These adapter-style public modules are also stable package surfaces for the first migration wave:

- `src/redd/adaptive_sampling.py`
- `src/redd/doc_filtering.py`
- `src/redd/embedding/`
- `src/redd/schema_global.py`
- `src/redd/join_resolution.py`
- `src/redd/parameter_optimization.py`
- `src/redd/predicate_proxy.py`
- `src/redd/proxy_runtime.py`
- `src/redd/retrieval.py`
- `src/redd/schema_refine.py`
- `src/redd/text_to_sql.py`

Execution-side optimization surfaces such as doc filtering, predicate proxies, join resolution,
and alpha allocation are intentionally consumed through the `data_extraction` stage contract,
even though they also have standalone adapter-style exports listed above.

## Internal Runtime Components

These modules power the package runtime but are not a stable external contract.
The preferred internal direction is to use the future-facing namespace packages
below:

- `src/redd/optimizations/`
- `src/redd/proxy/`
- `src/redd/correction/`
- `src/redd/exp/`

Current implementation-heavy internals still live under:

- `src/redd/cli/`
- `src/redd/runtime.py`
- `src/redd/config.py`
- `src/redd/llm/`
- `src/redd/core/data_loader/`
- `src/redd/core/data_population/`
- `src/redd/core/schema_gen/`
- `src/redd/core/schema_tailor/`
- `src/redd/core/utils/`

The rule of thumb is simple: `src/redd/core/**` is implementation detail unless a symbol is
re-exported through a documented `src/redd/*.py` surface.

In particular:

- `src/redd/core/data_population/` is the current runtime home for execution-side query
  optimization used by `data_extraction`
- `src/redd/core/schema_gen/` and `src/redd/core/schema_tailor/` are the runtime homes for
  schema-focused preprocessing and refinement
- `src/redd/optimizations/` is the runtime home for optional efficiency modules such
  as doc filtering, adaptive sampling, and alpha allocation
- `src/redd/proxy/` is the runtime home for `predicate_proxy`, `join_resolution`,
  and `proxy_runtime`
- `src/redd/correction/` and `src/redd/exp/` keep correction, evaluation, and experiment
  workflows separate from the main runtime path

## Experiment-Only Or Research Support Code

These areas are useful for research workflows but should not be treated as stable package API:

- `src/redd/correction/`
- `src/redd/exp/`
- `src/redd/exp/experiments/`
- `dataset/`
- `papers/`
- `prompts/` source prompt templates used to produce packaged prompt resources

## Incubator Boundary

- `ReDD_Dev` remains the upstream incubator for new features.
- New migrations should land here only after they have package-friendly configuration,
  adapters around external dependencies, and regression coverage in `tests/`.

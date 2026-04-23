# ReDD

ReDD is a Python package for turning unstructured document collections into relational schema artifacts and structured extraction outputs.

The package is organized around stable pipeline stages:

- `preprocessing`
- `schema_refinement`
- `data_extraction`
- `create_data_loader`
- `run_pipeline`

## Installation

```bash
python -m pip install -e .
```

Requirements:

- Python 3.10+
- provider API keys for cloud-backed runs
- optional CUDA for local-model and correction workflows

## Quick Start

Installable CLI entry points:

```bash
redd schemagen --config configs/schemagen.yaml --exp spider_4d1_1
redd datapop --config configs/datapop_cogito32b.yaml --exp wine
redd correction --config configs/datapop_cogito32b.yaml --exp wine
```

Equivalent module/script entry points:

```bash
python -m redd datapop --config configs/datapop_cogito32b.yaml --exp wine
python scripts/main_datapop.py --config configs/datapop_cogito32b.yaml --exp wine
```

Repository scripts in `scripts/` are thin wrappers over the same CLI and follow the same argument contract.

## Public Python API

```python
from redd import (
    SchemaGenerator,
    DataPopulator,
    preprocessing,
    schema_global,
    schema_refine,
    schema_refinement,
    data_extraction,
    create_data_loader,
    run_pipeline,
)
```

Example:

```python
from redd import DataPopulator, SchemaGenerator, run_pipeline

schema = SchemaGenerator.from_experiment("configs/schemagen.yaml", "spider_4d1_1")
datapop = DataPopulator.from_experiment("configs/datapop_cogito32b.yaml", "wine")

results = run_pipeline(
    schema_generator=schema,
    data_populator=datapop,
)
```

## Stage Responsibilities

`preprocessing`

- general schema discovery
- adaptive sampling support
- embeddings
- retrieval index generation

`schema_refinement`

- query-aware schema generation
- query-conditioned schema artifacts
- optional `schema_tailor` refinement engine

Schema helpers:

- `schema_global`: direct query-independent schema extraction
- `schema_refine`: query-specific schema extraction
- `schema_tailor` is used as an internal refinement strategy under `schema_refine`

`data_extraction`

- optional doc filtering
- table assignment
- attribute extraction
- result assembly
- optional predicate proxy execution
- optional proxy runtime orchestration
- optional join-aware execution
- optional alpha-allocation tuning
- optional text-to-SQL adapter integration

This division is intentional: `schema_refinement` owns schema-focused query conditioning, while execution-side optimizations such as doc filtering, predicate proxies, join resolution, proxy runtime orchestration, and alpha allocation live in `data_extraction`.

Example datapop config knob:

```yaml
doc_filter:
  enabled: true
  filter_type: schema_relevance
  target_recall: 0.95
```

Legacy config key `chunk_filter` is still accepted for backward compatibility, but new examples and outputs use `doc_filter` / `doc_filtering`.

## Stable Efficiency Module Surfaces

The package exposes stable public surfaces for the first migration wave from `ReDD_Dev`:

- `redd.embedding`
- `redd.retrieval`
- `redd.schema_global`
- `redd.global_schema` (backward-compatible alias)
- `redd.adaptive_sampling`
- `redd.doc_filtering`
- `redd.schema_refine`
- `redd.schema_tailoring` (backward-compatible alias)
- `redd.predicate_proxy`
- `redd.join_resolution`
- `redd.proxy_runtime`
- `redd.parameter_optimization`
- `redd.text_to_sql`

`redd.text_to_sql` is an adapter boundary for external integrations, not a built-in text-to-SQL implementation.

`ReDD_Dev` is still the incubator for future migrations such as evaluation, correction, and ablation-heavy research code, but the wave-1 efficiency surfaces above are now treated as stable package entry points.

For internal organization, the repository now uses namespace packages that own these implementation areas directly:

- `redd.optimizations.*` for optional efficiency modules only
- `redd.proxy.*` for `predicate_proxy`, `join_resolution`, and `proxy_runtime`
- `redd.correction` for correction and reliability workflows
- `redd.exp.*` for evaluation and experiment-only families

## Config Notes

Configs live in `configs/` and are normalized through shared helpers in `src/redd/config.py`.

Supported patterns:

- legacy flat module configs
- unified configs with shared top-level settings plus nested module sections

For schema refinement, enabling:

```yaml
schema_tailor:
  enabled: true
```

switches the refinement stage to the packaged schema-tailoring engine.

For data extraction, enabling any of the following routes the run through the unified efficiency-aware datapop path:

- `use_proxy_runtime: true`
- `doc_filter.enabled: true`
- `alpha_allocation.enabled: true`

Stage-owned outputs and schema artifact injection are centralized in `src/redd/runtime.py`, so API calls and repository scripts resolve the same output directories and loader filemaps.

## Repository Layout

```text
ReDD/
├── configs/                 # experiment configs
├── dataset/                 # local datasets
├── papers/                  # papers and reports
├── prompts/                 # source prompt templates
├── scripts/                 # thin CLI wrappers
├── src/redd/                # installable package
│   ├── cli/                 # installable CLI entry points
│   ├── core/                # remaining internal modules
│   ├── optimizations/       # runtime home for optional optimizations
│   ├── proxy/               # runtime home for proxy runtime modules
│   ├── correction/          # correction/reliability namespace
│   ├── exp/                 # evaluation/experiment namespace
│   ├── resources/prompts/   # packaged prompt resources
│   └── *.py                 # public stage/module surfaces
├── tests/                   # package smoke tests
└── pyproject.toml           # packaging metadata
```

## Development Notes

- active package/library paths raise exceptions instead of terminating the interpreter
- CLI wrappers are the only layer that should translate failures into process exit codes
- prompt resolution supports packaged resources and no longer depends on the current working directory
- `ReDD_Dev` remains the feature incubator for future migration waves; only the first package-ready wave is absorbed here

## Development

Install development tooling with:

```bash
python -m pip install -e ".[dev]"
```

Validation commands:

```bash
python -m unittest discover -s tests -v
ruff check src/redd/__init__.py src/redd/api.py src/redd/config.py src/redd/runtime.py src/redd/core/data_population/factory.py src/redd/core/schema_gen/factory.py src/redd/core/utils/prompt_utils.py src/redd/core/llm/providers.py tests
mypy
python -m build
```

Additional project guidance lives in:

- `CONTRIBUTING.md`
- `docs/MODULE_CLASSIFICATION.md`
- `docs/OPEN_SOURCE_READINESS.md`

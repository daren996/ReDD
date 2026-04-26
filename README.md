# ReDD

[![CI](https://github.com/daren996/ReDD/actions/workflows/ci.yml/badge.svg)](https://github.com/daren996/ReDD/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)

ReDD is a Python toolkit for turning unstructured document collections into
relational schema artifacts and structured extraction outputs. It provides a
config-driven pipeline for schema discovery, query-aware schema refinement,
document filtering, predicate proxy execution, and final data extraction.

ReDD is designed for research and applied workflows where documents need to be
converted into tables that can be inspected, evaluated, and reused.

## Table Of Contents

- [Why ReDD](#why-redd)
- [Project Status](#project-status)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web Demo](#web-demo)
- [CLI Usage](#cli-usage)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Repository Layout](#repository-layout)
- [Development](#development)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Security And Secrets](#security-and-secrets)
- [Citation](#citation)
- [License](#license)

## Why ReDD

Many information extraction systems stop at isolated JSON outputs. ReDD focuses
on the surrounding workflow needed to build relational data from documents:

- define or discover a target schema
- refine the schema for a query or task
- load documents and ground truth through a stable dataset contract
- extract table rows and attributes from unstructured text
- apply optional efficiency modules such as document filtering and proxy runtime
- evaluate outputs across repeatable experiment configs

## Project Status

ReDD is currently an alpha-stage open source project. The public package surface,
CLI, dataset contract, and config v2 runtime are being stabilized, while some
research workflows remain experimental.

Stable user-facing surfaces:

- `redd` package imports documented in this README
- `redd` CLI commands for strict config v2 experiments
- config version `2.1.1`
- HuggingFace-style dataset registry contract
- packaged prompt and web-demo resources

Research and migration areas:

- `redd.exp` for evaluation and experiment-only workflows
- `redd.correction` for correction and reliability experiments
- optional optimization modules that may change as the research code matures

## Features

- Config-driven document-to-table extraction pipeline
- Query-independent preprocessing
- Query-aware schema refinement
- Ground-truth and LLM-backed extraction modes
- Optional document filtering for retrieval-efficient extraction
- Predicate proxy and proxy runtime support
- Join-aware execution utilities
- Alpha-allocation and conformal calibration utilities
- Embedding and retrieval helpers
- HuggingFace-style parquet dataset contract
- Local FastAPI web demo
- Public Python API plus installable CLI
- Unit, contract, integration, and smoke test layout

## Installation

### Requirements

- Python 3.10 or newer
- Provider API keys for cloud-backed LLM or embedding runs
- Optional CUDA-capable environment for local-model workflows

### From Source

```bash
git clone https://github.com/daren996/ReDD.git
cd ReDD
python -m pip install -e .
```

### Optional Extras

```bash
# Development tools: pytest, ruff, mypy, build
python -m pip install -e ".[dev]"

# Web demo dependencies: FastAPI and Uvicorn
python -m pip install -e ".[web]"

# Structured-output helpers
python -m pip install -e ".[structured]"

# FAISS-backed retrieval
python -m pip install -e ".[retrieval]"
```

With `uv`:

```bash
uv sync --extra dev
uv run redd --help
```

## Quick Start

Run the tiny local fixture without provider credentials:

```bash
python -m pip install -e .
redd extract --config configs/examples/ground_truth_demo.yaml --experiment demo
```

The demo uses:

- dataset: `dataset/canonical/examples.single_doc_demo`
- schema source: ground truth
- extraction oracle: ground truth
- output directory: `outputs/demo`

For a full strict config v2 experiment, run the pipeline or each stage:

```bash
redd run --config configs/examples/ground_truth_demo.yaml --experiment demo

redd preprocess --config <config-v2.yaml> --experiment <experiment-id>
redd refine --config <config-v2.yaml> --experiment <experiment-id>
redd extract --config <config-v2.yaml> --experiment <experiment-id>
```

## Web Demo

Install the web extra and start the packaged FastAPI demo:

```bash
python -m pip install -e ".[web]"
redd web
```

With `uv`:

```bash
uv run --extra web redd web
```

Then open:

```text
http://127.0.0.1:8000
```

By default, the web demo uses `configs/demo/demo_datasets.yaml` and the `demo`
experiment. The server reads API keys from the environment or a local `.env`
file and only reports masked key status in the browser.

## CLI Usage

```bash
redd --help
redd run --config <config-v2.yaml> --experiment <experiment-id>
redd preprocess --config <config-v2.yaml> --experiment <experiment-id>
redd refine --config <config-v2.yaml> --experiment <experiment-id>
redd extract --config <config-v2.yaml> --experiment <experiment-id>
redd dataset validate dataset/manifest.yaml
redd web --config configs/demo/demo_datasets.yaml --experiment demo
```

Each stage command accepts `--api-key` as a temporary override. For repeatable
experiments, prefer environment variables or `api_key_env` in config files.

Research-oriented workflows live outside the primary CLI surface:

```bash
python scripts/main_exp.py evaluation --config <config-v2.yaml> --exp <experiment-id>
```

## Python API

Import from the top-level `redd` package for stable application code. For
example, the same bundled demo can be run from Python:

```python
from redd import data_extraction

summaries = data_extraction(
    config_path="configs/examples/ground_truth_demo.yaml",
    exp="demo",
)
```

The public API also exposes `SchemaGenerator`, `DataPopulator`,
`run_pipeline`, `create_data_loader`, `preprocessing`, `schema_refine`, and
`run_web_demo`. See [docs/API_EXAMPLES.md](docs/API_EXAMPLES.md) for practical
usage patterns.

## Configuration

Configs live under `configs/` and are validated by `src/redd/config.py`.
Strict config `2.1.1` is the supported runtime contract. A config defines the
project metadata, runtime paths, model providers, datasets, stages, and
experiments to run.

Supported provider names include `openai`, `deepseek`, `together`,
`siliconflow`, `gemini`, `local`, and `none`.

Use `.env.example` as the local environment template and keep real credentials
out of version control. See [docs/CONFIG_V2_REFERENCE.md](docs/CONFIG_V2_REFERENCE.md)
for the full config reference.

## Pipeline Stages

`preprocessing`

- General schema discovery
- Adaptive sampling support
- Embedding generation
- Retrieval index generation

`schema_refinement`

- Query-aware schema generation
- Query-conditioned schema artifacts
- Optional schema-tailoring refinement engine

`data_extraction`

- Optional document filtering
- Table assignment
- Attribute extraction
- Result assembly
- Optional predicate proxy execution
- Optional proxy runtime orchestration
- Optional join-aware execution
- Optional alpha-allocation tuning

When a dataset has no explicit query records, `data_extraction` runs the
implicit `default` extraction query and extracts every attribute in the
query-specific schema.

## Datasets

ReDD uses a HuggingFace-style local dataset registry. Runnable datasets are
described by manifests and store documents, schemas, queries, and ground truth
through explicit paths.

Typical layout:

```text
dataset/
  manifest.yaml
  canonical/
    examples.single_doc_demo/
      manifest.yaml
      data/
        documents.parquet
        ground_truth.parquet
      metadata/
        schema.json
        queries.json
  derived/
    bird.schools_demo/
      manifest.yaml
      data/
        documents.parquet
        ground_truth.parquet
      metadata/
        schema.json
        queries.json
```

Validate a dataset registry:

```bash
redd dataset validate dataset/manifest.yaml
```

See [docs/DATASET_CONTRACT.md](docs/DATASET_CONTRACT.md) for schema, query,
manifest, and parquet column requirements.

## Repository Layout

```text
ReDD/
|-- .github/                 # CI, issue templates, PR template
|-- configs/                 # experiment and demo configs
|-- dataset/                 # local dataset registry assets
|-- docs/                    # API, config, dataset, and workflow docs
|-- papers/                  # papers and reports
|-- prompts/                 # source prompt templates
|-- scripts/                 # repository workflow wrappers
|-- src/redd/                # installable Python package
|   |-- cli/                 # CLI entry points
|   |-- core/                # internal runtime implementation
|   |-- correction/          # research correction workflows
|   |-- exp/                 # experiment and evaluation workflows
|   |-- optimizations/       # optional efficiency modules
|   |-- proxy/               # predicate proxy and proxy runtime modules
|   |-- resources/           # packaged prompts, model catalog, web assets
|   |-- stages/              # stage orchestration
|   `-- *.py                 # public module surfaces
|-- tests/                   # unit, contract, integration, smoke tests
|-- CONTRIBUTING.md          # contribution guide
|-- LICENSE                  # MIT license
`-- pyproject.toml           # package metadata and tooling config
```

## Development

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run validation:

```bash
pytest tests
ruff check src tests
mypy
python -m build
```

The CI workflow currently validates editable install, tests, linting,
type-checking, and distribution builds on Python 3.10 and 3.11.

Test layout:

- `tests/unit/`: focused module behavior
- `tests/contracts/`: public API and compatibility expectations
- `tests/integration/`: multi-module flows with external services mocked
- `tests/smoke/`: import, package, and CLI wiring checks

## Documentation

- [API examples](docs/API_EXAMPLES.md)
- [Config v2 reference](docs/CONFIG_V2_REFERENCE.md)
- [Dataset contract](docs/DATASET_CONTRACT.md)
- [Module classification](docs/MODULE_CLASSIFICATION.md)
- [Research workflows](docs/RESEARCH_WORKFLOWS.md)
- [Config v2 release notes](docs/RELEASE_NOTES_CONFIG_V2.md)
- [Open source readiness](docs/OPEN_SOURCE_READINESS.md)
- [Contributing guide](CONTRIBUTING.md)

## Contributing

Contributions are welcome. Good first contributions include:

- fixing docs or examples
- adding small regression tests
- improving config validation messages
- adding dataset-contract checks
- tightening CLI and public API behavior

Before opening a pull request:

1. Install the development extra.
2. Run the validation commands above.
3. Keep public API changes documented.
4. Avoid committing provider keys, local `.env` files, or generated secrets.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Security And Secrets

Do not commit API keys or credentials. Use environment variables, `.env`, or
config `api_key_env` references instead.

Local files that may contain secrets should remain untracked. The repository
includes `.env.example` as a template and expects real credentials to be
provided only at runtime.

If you discover a security issue, please avoid publishing sensitive details in a
public issue. Open a minimal issue requesting maintainer contact, or contact the
maintainers through the repository owner profile.

## Citation

If ReDD is useful in your research, please cite the arXiv paper:

```bibtex
@misc{chao2025relationaldeepdiveerroraware,
  title = {Relational Deep Dive: Error-Aware Queries Over Unstructured Data},
  author = {Daren Chao and Kaiwen Chen and Naiqing Guan and Nick Koudas},
  year = {2025},
  eprint = {2511.02711},
  archivePrefix = {arXiv},
  primaryClass = {cs.DB},
  doi = {10.48550/arXiv.2511.02711},
  url = {https://arxiv.org/abs/2511.02711}
}
```

## License

ReDD is released under the [MIT License](LICENSE).

# ReDD TODO

This roadmap tracks the remaining work needed to keep ReDD mature, simple,
efficient, and professional after the strict config v2 and AI runtime
modernization work.

## Current Status

- [x] `src/redd/` is the official package root.
- [x] `pyproject.toml` exposes installable package metadata and CLI entry points.
- [x] Public stage entry points are package-level and stage-oriented:
  - `preprocessing`
  - `schema_refine`
  - `data_extraction`
  - `create_data_loader`
  - `run_pipeline`
- [x] Strict config v2 is the only supported runtime config contract.
- [x] Runtime/config/output-path responsibilities live behind `src/redd/config.py`
      and `src/redd/runtime.py`.
- [x] Cloud LLM and embedding calls are centralized behind LiteLLM-backed adapters.
- [x] Structured LLM output uses Pydantic v2 models, with optional Instructor support.
- [x] Retry/backoff is centralized with tenacity.
- [x] Retrieval supports NumPy by default and optional FAISS acceleration.
- [x] Optional optimizations are expressed as composable strategy blocks:
  - document filtering
  - proxy runtime
  - alpha allocation
  - schema tailoring
  - retrieval
  - adaptive sampling
- [x] Local model execution is kept behind the shared runtime boundary.
- [x] LangChain and LlamaIndex are intentionally not core dependencies.
- [x] Old compatibility shims for experiment-style config loading have been removed.
- [x] Canonical internal data-extraction implementation is `DataExtraction`.
- [x] `core/` is treated as a transitional internal container, not the final package
      layout.
- [x] Full repository lint gate is clean: `ruff check src tests`.

## P0 Quality Gate

P0 is complete when the package can be trusted as a clean development base.

- [x] Full test suite passes with `pytest tests`.
- [x] Proxy runtime tests pass from their new test location.
- [x] Full `ruff check src tests` passes.
- [x] Example configs are short, strict config v2 files under `configs/examples/`.
- [x] CLI uses canonical commands:
  - `redd run`
  - `redd preprocess`
  - `redd refine`
  - `redd extract`
- [x] Legacy runtime/config compatibility paths are gone from active code.
- [x] Provider-specific request formatting, auth, retry, and model naming are behind
      runtime adapters.
- [x] Tests are organized by intent:
  - contracts
  - integration
  - smoke
  - unit

## P1 Internal Architecture

P1 should make internals easier to evolve without changing the public API.

- [x] Keep stage orchestrators thin: they should coordinate loaders, runtime
      backends, prompt runners, and strategies rather than implement those details
      inline.
- [x] Continue moving reusable optional behavior into strategy modules with clear
      interfaces.
- [x] Consolidate loader variants into a registry-driven loader family; keep only
      genuinely distinct loader implementations as separate classes.
- [x] Move dataset-specific behavior into loader profiles or config where possible.
- [x] Continue de-flattening `core/` incrementally, without a large blind directory
      move.
- [x] Keep third-party integrations behind adapters so external frameworks do not
      shape the core package architecture.

## P2 Research Workflows

P2 keeps research and reliability workflows useful without bloating the primary
runtime path.

- [x] Keep evaluation workflows under `src/redd/exp/` until they justify promotion.
- [x] Keep correction and reliability workflows under `src/redd/correction/`.
- [x] Keep proxy pretraining and classifier-training utilities separate from the
      primary pipeline stage surface.
- [x] Promote research modules only as clearly named subsystems, not as hidden
      behavior inside `preprocessing`, `schema_refine`, or `data_extraction`.
- [x] Document the promotion criteria for research modules:
  - stable API
  - tests
  - config v2 integration
  - no hidden dataset assumptions

## P3 Productization

P3 is about making the package easier to use outside the repository.

- [x] Add a minimal quickstart that runs against a tiny local dataset fixture.
- [x] Add config v2 reference documentation.
- [x] Add API examples for each public stage entry point.
- [x] Add a web-demo-ready orchestration wrapper that uses the same package API as
      the CLI.
- [x] Add release notes for the strict config v2 breaking change.

## Archived

The previous long-form migration checklist has been folded into this roadmap.
Items covered by strict config v2, LiteLLM/Pydantic/tenacity runtime adapters,
strategy blocks, CLI v2 naming, and full lint cleanup are considered complete.

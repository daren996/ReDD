# Research Workflows

ReDD keeps research and reliability workflows available without making them
implicit behavior in the primary runtime stages.

## Current Homes

- `src/redd/exp/` contains evaluation and experiment-only utilities.
- `src/redd/correction/` contains correction and reliability workflows.
- `src/redd/exp/experiments/predicate_proxy/` contains proxy-pretraining support.
- Classifier-training utilities stay under `src/redd/correction/` until promoted.

These modules may be imported directly by research code, but they should not be
triggered implicitly by `preprocessing`, `schema_refine`, or `data_extraction`.
The primary CLI remains `redd run`, `redd preprocess`, `redd refine`, and
`redd extract`.

## Promotion Criteria

A research workflow can move into a stable public subsystem only when it has:

- a clearly named API that does not depend on private experiment globals
- tests covering the supported behavior
- strict config v2 integration for runtime options
- no hidden dataset assumptions
- external integrations wrapped behind package adapters

Promotion should create an explicit subsystem or stage option. It should not
hide new behavior inside an existing public stage.

## Boundary Tests

Smoke tests should keep `redd.exp` and `redd.correction` importable while
ensuring removed legacy namespaces do not reappear. Contract tests should cover
any promoted public API before documentation presents it as stable.

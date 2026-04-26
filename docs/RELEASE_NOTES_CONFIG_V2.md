# Release Notes: Strict Config V2

This release makes strict config 2.1.1 the supported runtime contract.

## Breaking Changes

- Runtime configs must declare `config_version: 2.1.1`.
- CLI commands are stage-oriented:
  - `redd run`
  - `redd preprocess`
  - `redd refine`
  - `redd extract`
- Legacy experiment-style config shims are removed from active package code.
- Stage outputs are resolved through `src/redd/config.py` and
  `src/redd/runtime.py`.
- Research workflows remain outside the primary runtime path.

## Migration Notes

- Move shared runtime values into `project`, `runtime`, and `models`.
- Define each dataset under `datasets`.
- Define primary runtime behavior under `stages`.
- Define runnable combinations under `experiments`.
- Use `schema_source: ground_truth` for ground-truth extraction demos.
- Use strategy blocks such as `document_filtering`, `proxy_runtime`, and
  `alpha_allocation` instead of ad hoc top-level runtime flags.

## Compatibility Boundaries

The public package API remains centered on `SchemaGenerator`, `DataPopulator`,
`preprocessing`, `schema_refine`, `data_extraction`, `create_data_loader`, and
`run_pipeline`. Research APIs under `redd.exp` and `redd.correction` are useful
but not part of the primary stage contract.

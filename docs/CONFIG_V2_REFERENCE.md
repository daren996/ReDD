# Config V2 Reference

Strict config 2.1.1 is the supported runtime contract for package and CLI runs.
Configs are YAML mappings with `config_version: 2.1.1`.

## Top-Level Shape

```yaml
config_version: 2.1.1

project:
  name: demo
  seed: 42

runtime:
  output_dir: outputs/demo
  log_dir: logs
  output_layout: dataset_stage
  artifact_id: run-v1

models:
  llm:
    provider: deepseek
    model: deepseek-v4-flash
    api_key_env: DEEPSEEK_API_KEY
    base_url: null
    structured_backend: auto
    max_retries: 5
    wait_time: 10
    temperature: null
    top_p: null
    max_tokens: null
    local_model_path: null
  embedding:
    provider: openai
    model: text-embedding-3-small
    enabled: true
    api_key_env: OPENAI_API_KEY
    base_url: null
    batch_size: 100
    storage_file: embeddings.sqlite3

datasets: {}
stages: {}
experiments: {}
```

`runtime.output_layout` currently supports `dataset_stage`, producing:

```text
<output_dir>/<dataset-id>/<stage>/<artifact-id>/
```

`project.name` identifies the project namespace for runtime metadata, API/CLI
result payloads, and log file prefixes. Output paths remain controlled by
`runtime.output_dir`, dataset id, stage, and artifact id so runs can be moved or
grouped explicitly. `project.seed` is exposed to the internal runtime contract as
`project_seed` for deterministic components.

## Models

`models.llm` configures cloud or local LLM calls. Use `null` for ground-truth
or no-LLM workflows.

```yaml
models:
  llm:
    provider: deepseek
    model: deepseek-v4-flash
    api_key_env: DEEPSEEK_API_KEY
    base_url: null
    structured_backend: auto
    max_retries: 5
    wait_time: 10
    temperature: null
    top_p: null
    max_tokens: null
    local_model_path: null
  embedding:
    provider: openai
    model: text-embedding-3-small
    enabled: true
    api_key_env: OPENAI_API_KEY
    base_url: null
    batch_size: 100
    storage_file: embeddings.sqlite3
```

Supported provider names are normalized by the shared runtime adapters.

## Datasets

Datasets define loader type, root path, optional query selection, and loader
options.

```yaml
datasets:
  demo:
    loader: hf_manifest
    root: dataset/canonical/examples.single_doc_demo
    query_ids: null
    loader_options:
      manifest: manifest.yaml
    split:
      train_count: 0
```

`hf_manifest` is the default loader profile. Its manifest points to documents,
ground truth, schema, and queries under the dataset root. Table and column IDs
are expected to be consistent across schema, queries, and ground truth.

## Stages

The supported primary stages are `preprocessing`, `schema_refinement`, and
`data_extraction`.

```yaml
stages:
  preprocessing:
    enabled: true
    prompt: schemagen_4_1
    adaptive_sampling:
      enabled: false

  schema_refinement:
    enabled: true
    source_stage: preprocessing
    prompt: schemagen_4_1
    schema_tailoring:
      enabled: false

  data_extraction:
    enabled: true
    schema_source: schema_refinement
    oracle: llm
    document_filtering:
      enabled: false
    proxy_runtime:
      enabled: false
    alpha_allocation:
      enabled: false
```

Strategy blocks use a common shape: `enabled` plus any strategy-specific
options. Stage-specific prompt/input/output overrides are accepted through
`prompt`, `prompts`, `input_fields`, `output_fields`, and `options`.

## Experiments

Experiments select datasets and stage order.

```yaml
experiments:
  demo:
    datasets: [demo]
    stages: [data_extraction]
    artifact_id: single-doc-ground-truth-v1
```

Run with:

```bash
redd run --config configs/examples/ground_truth_demo.yaml --experiment demo
```

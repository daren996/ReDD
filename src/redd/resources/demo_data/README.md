# Packaged Demo Data

This package is reserved for tiny datasets that should work after
`pip install redd`, without requiring a checkout of the full repository data
tree.

Expected future layout:

```text
demo_data/
  <dataset_id>/
    manifest.yaml
    metadata/
      schema.json
      queries.json
    data/
      documents.parquet
      ground_truth.parquet
```

Use this area only for minimal fixtures or first-run demos. Larger benchmark,
paper, or generated datasets should stay under the repository-level `dataset/`
tree or an external data distribution mechanism.

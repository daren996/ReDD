# ReDD Dataset Contract

ReDD datasets use a HuggingFace-style registry layout. Every runnable dataset is
described by a `manifest.yaml` and exposes documents, ground truth, schema, and
queries through explicit paths.

## Registry Layout

```text
dataset/
  manifest.yaml
  canonical/
    spider.wine_1/
      manifest.yaml
      data/
        documents.parquet
        ground_truth.parquet
      metadata/
        schema.json
        queries.json
        query_sets/
          generated_queries.json
  derived/
    spider.wine_1.wine_appellations/
      manifest.yaml
      data/
        documents.parquet
        ground_truth.parquet
      metadata/
        schema.json
        queries.json
```

`canonical/` contains standardized source datasets. `derived/` contains every
constructed dataset, including single-source variants and multi-source
compositions.

## Manifests

Root manifest:

```yaml
schema_version: redd.registry.v1
collection_id: redd
format: hf_parquet_registry
datasets:
  spider.wine_1:
    kind: canonical
    path: canonical/spider.wine_1/manifest.yaml
```

Dataset manifest:

```yaml
schema_version: redd.manifest.v1
dataset_id: spider.wine_1
kind: canonical
version: 0.1.0
paths:
  documents: data/documents.parquet
  ground_truth: data/ground_truth.parquet
  schema: metadata/schema.json
  queries: metadata/queries.json
```

## Schema

`metadata/schema.json` describes the target schema only. It never contains
instance-level ground truth values.

```json
{
  "schema_version": "redd.schema.v1",
  "dataset_id": "spider.wine_1",
  "tables": [
    {
      "table_id": "wine",
      "name": "wine",
      "description": "Wine records.",
      "primary_key": ["row_id"],
      "columns": [
        {
          "column_id": "wine.winery",
          "name": "Winery",
          "type": "string",
          "description": "Name of the winery.",
          "nullable": true,
          "examples": ["Chalk Hill"]
        }
      ]
    }
  ],
  "relationships": []
}
```

## Queries

`metadata/queries.json` stores query records in list form.

```json
{
  "schema_version": "redd.queries.v1",
  "dataset_id": "spider.wine_1",
  "queries": [
    {
      "query_id": "q1",
      "question": "List the wineries that produce wine from Sonoma Coast with score greater than 90.",
      "sql": "SELECT Winery FROM wine WHERE Appelation = 'Sonoma Coast' AND Score > '90';",
      "required_tables": ["wine"],
      "required_columns": ["wine.winery", "wine.appelation", "wine.score"],
      "output_columns": ["wine.winery"],
      "tags": ["filter"],
      "difficulty": null
    }
  ]
}
```

If `queries` is empty or omitted, ReDD treats data extraction as an implicit
default query named `default`: extract every attribute from the query-specific
schema. The default query has no SQL predicate and no output projection, so the
result keeps all extracted attributes. This is the intended behavior for
schema-only extraction tasks.

## Parquet Tables

`data/documents.parquet` columns:

```text
dataset_id, doc_id, doc_text, source_id, source_table, source_row_id,
parent_doc_id, chunk_index, is_chunked, split
```

`data/ground_truth.parquet` columns:

```text
dataset_id, doc_id, record_id, table_id, column_id, column_name,
value, value_type, source_row_id
```

`ground_truth.parquet` contains instance-level truth only. The authoritative
schema always lives in `metadata/schema.json`.
Datasets do not include a separate table/attribute mapping file; table and
column IDs must already match across schema, queries, and ground truth.

## ID Rules

Canonical dataset IDs use `{source}.{dataset}`, for example
`spider.wine_1`. Derived dataset IDs append a task or variant name, for example
`spider.wine_1.wine_appellations`.

Column IDs use normalized identifiers:

```text
{table_id}.{column_name_normalized}
```

For example, the `Winery` column in table `wine` becomes `wine.winery`.

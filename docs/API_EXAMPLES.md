# API Examples

These examples use the public package surface. Internal modules under
`redd.core` and `redd.stages` are implementation details.

## Data Loader

```python
from redd import create_data_loader

loader = create_data_loader("dataset/canonical/examples.single_doc_demo")
print(loader.doc_ids)
print(loader.load_schema_general())
```

## Preprocessing

```python
from redd import preprocessing

summaries = preprocessing(
    config_path="configs/pipeline.yaml",
    exp="demo",
    datasets=["demo"],
)
```

## Schema Refinement

```python
from redd import schema_refine

summaries = schema_refine(
    config_path="configs/pipeline.yaml",
    exp="demo",
    datasets=["demo"],
)
```

## Data Extraction

```python
from redd import data_extraction

summaries = data_extraction(
    config_path="configs/examples/ground_truth_demo.yaml",
    exp="demo",
)
```

If no explicit query records are present, data extraction uses the implicit
`default` query and extracts all attributes from the query-specific schema.
Explicit query records may restrict extraction through `required_tables` and
`required_columns`.

## Full Pipeline

```python
from redd import DataPopulator, SchemaGenerator, run_pipeline

schema = SchemaGenerator.from_experiment("configs/pipeline.yaml", "demo")
extractor = DataPopulator.from_experiment("configs/pipeline.yaml", "demo")

results = run_pipeline(
    schema_generator=schema,
    data_populator=extractor,
)
```

## Web Demo Wrapper

```python
from redd.web_demo import run_web_demo

payload = run_web_demo(
    "configs/examples/ground_truth_demo.yaml",
    "demo",
)
```

`run_web_demo` returns a dictionary and does not start an HTTP server. Framework
code can call it and serialize the result.

## FastAPI Web Demo

```python
from redd.web_demo import create_web_demo_app

app = create_web_demo_app(
    default_config="configs/demo/demo_datasets.yaml",
    default_experiment="demo",
)
```

Run the packaged server:

```bash
redd web
```

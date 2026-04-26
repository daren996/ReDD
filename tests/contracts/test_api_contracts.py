from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

import pandas as pd
import yaml

from redd.api import (
    DATA_EXTRACTION,
    PREPROCESSING,
    SCHEMA_REFINEMENT,
    DataPopulator,
    SchemaGenerator,
    run_pipeline,
)
from redd.core.data_loader.data_loader_hf_manifest import DataLoaderHFManifest
from redd.core.utils.constants import PATH_TEMPLATES
from redd.dataset_contract import validate_dataset_manifest
from redd.runtime import resolve_schema_artifact_source


class ApiContractTests(unittest.TestCase):
    def test_schema_generator_aliases_delegate_to_existing_stage_methods(self) -> None:
        calls: list[tuple[str, list[str] | None]] = []

        class AliasAwareGenerator(SchemaGenerator):
            def __init__(self) -> None:
                pass

            def preprocessing(self, datasets=None):
                calls.append(("global", datasets))
                return ["global"]

            def schema_refine(self, datasets=None):
                calls.append(("refine", datasets))
                return ["refine"]

        generator = AliasAwareGenerator()

        self.assertEqual(generator.schema_global(datasets=["wine_1/default_task"]), ["global"])
        self.assertEqual(generator.schema_refinement(datasets=["wine_1/default_task"]), ["refine"])
        self.assertEqual(
            calls,
            [("global", ["wine_1/default_task"]), ("refine", ["wine_1/default_task"])],
        )

    def test_schema_generator_requires_shared_output_root_for_pre_and_refine(self) -> None:
        with self.assertRaisesRegex(ValueError, "same output root"):
            SchemaGenerator(
                preprocessing_config={
                    "mode": "openai",
                    "out_main": "outputs/schema-pre",
                    "exp_dn_fn_list": ["wine_1/default_task"],
                },
                refinement_config={
                    "mode": "openai",
                    "out_main": "outputs/schema-refine",
                    "exp_dn_fn_list": ["wine_1/default_task"],
                    "in_fields": {"query": "query"},
                },
                configure_logging=False,
            )

    def test_preprocessing_retrieval_requires_embeddings(self) -> None:
        generator = SchemaGenerator(
            preprocessing_config={
                "mode": "openai",
                "out_main": "outputs/schema",
                "exp_dn_fn_list": ["wine_1/default_task"],
                "retrieval": {"enabled": True},
            },
            configure_logging=False,
        )

        with self.assertRaisesRegex(ValueError, "embedding.enabled = true"):
            generator.preprocessing()

    def test_schema_generator_delegates_to_stage_orchestrator(self) -> None:
        generator = SchemaGenerator(
            preprocessing_config={
                "mode": "openai",
                "out_main": "outputs/schema",
                "exp_dn_fn_list": ["wine_1/default_task"],
            },
            configure_logging=False,
        )

        with patch("redd.stages.schema.run_schema_preprocessing", return_value=[{"ok": True}]) as mocked:
            self.assertEqual(generator.preprocessing(datasets=["wine_1/default_task"]), [{"ok": True}])
            mocked.assert_called_once_with(generator, datasets=["wine_1/default_task"])

    def test_data_populator_loader_config_injects_schema_paths(self) -> None:
        populator = DataPopulator(
            {
                "mode": "openai",
                "out_main": "outputs/data",
                "exp_dn_fn_list": ["wine_1/default_task"],
            },
            configure_logging=False,
        )

        loader_config = populator._build_loader_config(
            schema_source="generated",
            base_loader_config={"filemap": {"documents": "documents.json"}},
            dataset="wine_1/default_task",
            general_schema_source=resolve_schema_artifact_source(
                {"out_main": "/tmp/general", "res_param_str": "general"}
            ),
            query_schema_source=resolve_schema_artifact_source(
                {"out_main": "/tmp/query", "res_param_str": "tailored"}
            ),
        )

        self.assertEqual(loader_config["filemap"]["documents"], "documents.json")
        self.assertEqual(
            loader_config["filemap"]["schema_general"],
            str(Path("/tmp/general") / "wine_1/default_task" / PATH_TEMPLATES.schema_general("general")),
        )
        self.assertEqual(
            loader_config["filemap"]["schema_query"],
            str(
                Path("/tmp/query")
                / "wine_1/default_task"
                / PATH_TEMPLATES.SCHEMA_QUERY_TAILORED.format(qid="{qid}", param_str="tailored")
            ),
        )

    def test_data_populator_loader_config_uses_ground_truth_schema_when_requested(self) -> None:
        populator = DataPopulator(
            {
                "mode": "openai",
                "out_main": "outputs/data",
                "exp_dn_fn_list": ["wine_1/gt_schema_task"],
                "schema_source": "ground_truth",
            },
            configure_logging=False,
        )

        loader_config = populator._build_loader_config(
            schema_source="ground_truth",
            base_loader_config={"manifest": "manifest.yaml"},
            dataset="wine_1/gt_schema_task",
            general_schema_source=resolve_schema_artifact_source(
                {"out_main": "/tmp/general", "res_param_str": "general"}
            ),
            query_schema_source=resolve_schema_artifact_source(
                {"out_main": "/tmp/query", "res_param_str": "tailored"}
            ),
        )

        self.assertEqual(loader_config["manifest"], "manifest.yaml")
        self.assertNotIn("filemap", loader_config)

    def test_data_populator_schema_source_accepts_ground_truth_aliases(self) -> None:
        populator = DataPopulator(
            {
                "mode": "openai",
                "out_main": "outputs/data",
                "exp_dn_fn_list": ["wine_1/gt_schema_task"],
                "schema_source": "gt",
            },
            configure_logging=False,
        )

        self.assertEqual(populator._resolve_schema_source_mode(populator.config), "ground_truth")

    def test_data_populator_delegates_to_stage_orchestrator(self) -> None:
        populator = DataPopulator(
            {
                "mode": "openai",
                "out_main": "outputs/data",
                "exp_dn_fn_list": ["wine_1/default_task"],
            },
            configure_logging=False,
        )

        with patch("redd.stages.data_extraction.run_data_extraction", return_value=[{"ok": True}]) as mocked:
            self.assertEqual(populator.data_extraction(datasets=["wine_1/default_task"]), [{"ok": True}])
            mocked.assert_called_once_with(
                populator,
                datasets=["wine_1/default_task"],
                schema_generator=None,
                schema_config=None,
            )

    def test_run_pipeline_uses_default_stage_order(self) -> None:
        calls: list[tuple] = []

        class FakeGenerator:
            preprocessing_config: dict[str, object] = {}
            refinement_config: dict[str, object] = {}

            def preprocessing(self, datasets=None):
                calls.append(("pre", datasets))
                return ["pre"]

            def schema_refinement(self, datasets=None):
                calls.append(("ref", datasets))
                return ["ref"]

        class FakePopulator:
            def data_extraction(self, datasets=None, schema_generator=None):
                calls.append(("data", datasets, schema_generator))
                return ["data"]

        generator = FakeGenerator()
        populator = FakePopulator()

        results = run_pipeline(
            schema_generator=cast(SchemaGenerator, generator),
            data_populator=cast(DataPopulator, populator),
            datasets=["wine_1/default_task"],
        )

        self.assertEqual(
            list(results),
            [PREPROCESSING.value, SCHEMA_REFINEMENT.value, DATA_EXTRACTION.value],
        )
        self.assertEqual(calls[0], ("pre", ["wine_1/default_task"]))
        self.assertEqual(calls[1], ("ref", ["wine_1/default_task"]))
        self.assertEqual(calls[2][:2], ("data", ["wine_1/default_task"]))
        self.assertIs(calls[2][2], generator)

    def test_run_pipeline_requires_requested_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "PREPROCESSING requires `schema_generator=`"):
            run_pipeline(stages=[PREPROCESSING])

    def test_schema_refinement_requires_preprocessing_artifact_when_general_schema_is_reused(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = {
                "mode": "openai",
                "data_main": str(root / "dataset"),
                "out_main": str(root / "outputs"),
                "exp_dn_fn_list": ["wine_1/default_task"],
                "in_fields": {"query": "query"},
                "general_param_str": "general",
                "res_param_str": "tailored",
            }
            generator = SchemaGenerator(refinement_config=config, configure_logging=False)

            class FakeLoader:
                def load_query_dict(self):
                    return {"Q1": {"query": "Which rows match?"}}

            class FakeImpl:
                general_param_str = "general"
                param_str = "tailored"
                loader_type = "hf_manifest"
                loader_config: dict[str, object] = {}

                def _build_doc_dict(self):
                    return {}

            with patch("redd.stages.schema.build_schema_generator_impl", return_value=FakeImpl()):
                with patch("redd.stages.schema.create_data_loader", return_value=FakeLoader()):
                    with self.assertRaisesRegex(FileNotFoundError, "Run PREPROCESSING first"):
                        generator.schema_refinement()

    def test_hf_manifest_loader_reads_contract_and_adapts_schema_shape(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data").mkdir()
            (root / "metadata").mkdir()
            pd.DataFrame(
                [
                    {
                        "dataset_id": "example",
                        "doc_id": "doc-1",
                        "doc_text": "Mount Demo Winery scored 95.",
                        "source_id": "wine_record_1.txt",
                        "source_table": "wine_record_1",
                        "source_row_id": "0",
                        "parent_doc_id": None,
                        "chunk_index": 0,
                        "is_chunked": False,
                        "split": "test",
                    }
                ]
            ).to_parquet(root / "data" / "documents.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset_id": "example",
                        "doc_id": "doc-1",
                        "record_id": "0",
                        "table_id": "wine",
                        "column_id": "wine.winery",
                        "column_name": "Winery",
                        "value": "Mount Demo Winery",
                        "value_type": "string",
                        "source_row_id": "0",
                    }
                ]
            ).to_parquet(root / "data" / "ground_truth.parquet", index=False)
            (root / "metadata" / "queries.json").write_text(
                json.dumps(
                    {
                        "schema_version": "redd.queries.v1",
                        "dataset_id": "example",
                        "queries": [
                            {
                                "query_id": "Q1",
                                "question": "Which winery has score above 90?",
                                "sql": "",
                                "required_tables": ["wine"],
                                "required_columns": ["wine.winery"],
                                "output_columns": ["wine.winery"],
                                "tags": [],
                                "difficulty": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "metadata" / "schema.json").write_text(
                json.dumps(
                    {
                        "schema_version": "redd.schema.v1",
                        "dataset_id": "example",
                        "tables": [
                            {
                                "table_id": "wine",
                                "name": "wine",
                                "description": "Wine records",
                                "primary_key": ["row_id"],
                                "columns": [
                                    {
                                        "column_id": "wine.winery",
                                        "name": "Winery",
                                        "type": "string",
                                        "description": "Winery name",
                                        "nullable": True,
                                        "examples": [],
                                    }
                                ],
                            }
                        ],
                        "relationships": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "redd.manifest.v1",
                        "dataset_id": "example",
                        "kind": "canonical",
                        "paths": {
                            "documents": "data/documents.parquet",
                            "ground_truth": "data/ground_truth.parquet",
                            "schema": "metadata/schema.json",
                            "queries": "metadata/queries.json",
                        },
                    }
                ),
                encoding="utf-8",
            )

            loader = DataLoaderHFManifest(root)

            self.assertTrue(validate_dataset_manifest(root / "manifest.yaml")["valid"])
            self.assertEqual(list(loader.load_query_dict()), ["Q1"])
            self.assertEqual(loader.load_schema_general()[0]["Schema Name"], "wine")
            self.assertEqual(
                loader.load_schema_general()[0]["Attributes"][0]["Attribute Name"],
                "Winery",
            )
            doc_info = loader.get_doc_info("doc-1")
            assert doc_info is not None
            self.assertEqual(doc_info["data_records"][0]["table_name"], "wine")
            self.assertNotIn("mappings", doc_info)
            self.assertEqual(loader.load_name_map()["table"], {"wine": "wine"})


if __name__ == "__main__":
    unittest.main()

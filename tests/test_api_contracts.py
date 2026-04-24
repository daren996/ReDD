from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

from redd.api import (
    DATA_EXTRACTION,
    PREPROCESSING,
    SCHEMA_REFINEMENT,
    DataPopulator,
    SchemaGenerator,
    run_pipeline,
)
from redd.core.data_loader.data_loader_sqlite import DataLoaderSQLite
from redd.core.utils.constants import PATH_TEMPLATES


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
        with self.assertRaisesRegex(ValueError, "same `out_main`"):
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
            general_schema_root=Path("/tmp/general"),
            general_schema_param="general",
            query_schema_root=Path("/tmp/query"),
            query_schema_param="tailored",
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
            base_loader_config={"filemap": {"documents": "documents.db"}},
            dataset="wine_1/gt_schema_task",
            general_schema_root=Path("/tmp/general"),
            general_schema_param="general",
            query_schema_root=Path("/tmp/query"),
            query_schema_param="tailored",
        )

        self.assertEqual(loader_config["filemap"]["documents"], "documents.db")
        self.assertEqual(loader_config["filemap"]["schema_general"], "schema.json")
        self.assertNotIn("schema_query", loader_config["filemap"])

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
                loader_type = "sqlite"
                loader_config = {}

                def _build_doc_dict(self):
                    return {}

            with patch("redd.api._build_schema_generator_impl", return_value=FakeImpl()):
                with patch("redd.api.create_data_loader", return_value=FakeLoader()):
                    with self.assertRaisesRegex(FileNotFoundError, "Run PREPROCESSING first"):
                        generator.schema_refinement()

    def test_sqlite_loader_accepts_nested_queries_and_gt_schema_shape(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "queries.json").write_text(
                (
                    "{\n"
                    '  "dataset": "example",\n'
                    '  "queries": {\n'
                    '    "Q1": {\n'
                    '      "query": "Which winery has score above 90?",\n'
                    '      "attributes": ["wine.Winery", "wine.Score"]\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            (root / "schema.json").write_text(
                (
                    "{\n"
                    '  "tables": {\n'
                    '    "wine": {\n'
                    '      "description": "Wine records",\n'
                    '      "attributes": {\n'
                    '        "Winery": {"description": "Winery name"},\n'
                    '        "Score": {"description": "Wine score"}\n'
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            loader = DataLoaderSQLite(
                root,
                filemap={
                    "schema_general": "schema.json",
                },
            )

            self.assertEqual(list(loader.load_query_dict()), ["Q1"])
            self.assertEqual(loader.load_schema_general()[0]["Schema Name"], "wine")
            self.assertEqual(
                loader.load_schema_general()[0]["Attributes"][0]["Attribute Name"],
                "Winery",
            )


if __name__ == "__main__":
    unittest.main()

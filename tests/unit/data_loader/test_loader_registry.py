from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import yaml

from redd.config import resolve_repo_path
from redd.core.data_loader import (
    DataLoaderHFManifest,
    create_data_loader,
    get_loader_profile_notes,
    get_loader_registry,
)


class LoaderRegistryTests(unittest.TestCase):
    def test_default_hf_manifest_loader_can_be_created(self) -> None:
        loader = create_data_loader(resolve_repo_path("dataset/canonical/examples.single_doc_demo"))

        self.assertIsInstance(loader, DataLoaderHFManifest)
        self.assertGreaterEqual(loader.num_docs, 1)

    def test_unknown_loader_lists_available_loaders(self) -> None:
        with self.assertRaisesRegex(ValueError, "Available loaders: hf_manifest"):
            create_data_loader(
                resolve_repo_path("dataset/canonical/examples.single_doc_demo"),
                loader_type="unknown",
            )

    def test_registry_and_profile_notes_are_copies(self) -> None:
        registry = get_loader_registry()
        notes = get_loader_profile_notes()

        registry["fake"] = DataLoaderHFManifest
        notes["fake"] = "fake"

        self.assertNotIn("fake", get_loader_registry())
        self.assertNotIn("fake", get_loader_profile_notes())

    def test_empty_queries_create_default_all_attribute_extraction_query(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data").mkdir()
            (root / "metadata").mkdir()
            pd.DataFrame(
                [
                    {
                        "dataset_id": "example",
                        "doc_id": "doc-1",
                        "doc_text": "A demo row.",
                        "source_id": "demo",
                        "source_table": "wine",
                        "source_row_id": "0",
                        "parent_doc_id": None,
                        "chunk_index": 0,
                        "is_chunked": False,
                        "split": "test",
                    }
                ]
            ).to_parquet(root / "data" / "documents.parquet", index=False)
            pd.DataFrame(
                columns=[
                    "dataset_id",
                    "doc_id",
                    "record_id",
                    "table_id",
                    "column_id",
                    "column_name",
                    "value",
                    "value_type",
                    "source_row_id",
                ]
            ).to_parquet(root / "data" / "ground_truth.parquet", index=False)
            (root / "metadata" / "queries.json").write_text(
                json.dumps(
                    {
                        "schema_version": "redd.queries.v1",
                        "dataset_id": "example",
                        "queries": [],
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
                                "columns": [
                                    {"column_id": "wine.winery", "name": "winery"},
                                    {"column_id": "wine.score", "name": "score"},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "redd.manifest.v1",
                        "dataset_id": "example",
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

            self.assertEqual(list(loader.load_query_dict()), ["default"])
            default_query = loader.get_query_info("default")
            assert default_query is not None
            self.assertTrue(default_query["default_extraction"])
            self.assertEqual(default_query["output_columns"], [])
            self.assertEqual(
                [attr["Attribute Name"] for attr in loader.load_schema_query("default")[0]["Attributes"]],
                ["winery", "score"],
            )


if __name__ == "__main__":
    unittest.main()

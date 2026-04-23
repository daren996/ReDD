from __future__ import annotations

import unittest

from redd.config import resolve_repo_path
from redd.proxy.proxy_runtime.pipeline import ProxyPipeline
from redd.proxy.proxy_runtime.types import ProxyPipelineConfig
from redd.core.utils.data_split import resolve_training_data_count
from redd.runtime import (
    build_data_loader_config,
    ensure_shared_output_root,
    normalize_stage_config,
    resolve_schema_artifact_source,
)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_normalize_stage_config_accepts_optional_input(self) -> None:
        self.assertIsNone(normalize_stage_config(None, module="schemagen"))

        normalized = normalize_stage_config(
            {
                "mode": "openai",
                "exp_dataset_task": "wine_1/default_task",
            },
            module="schemagen",
        )

        assert normalized is not None
        self.assertEqual(normalized["mode"], "cgpt")
        self.assertEqual(normalized["exp_dn_fn_list"], ["wine_1/default_task"])

    def test_ensure_shared_output_root_rejects_mismatched_stage_outputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "share the same `out_main`"):
            ensure_shared_output_root(
                {"out_main": "outputs/schema-pre"},
                {"out_main": "outputs/schema-refine"},
            )

    def test_resolve_schema_artifact_source_prefers_general_param_when_requested(self) -> None:
        source = resolve_schema_artifact_source(
            {
                "out_main": "outputs/schema",
                "general_param_str": "general",
                "res_param_str": "tailored",
            },
            prefer_general_param=True,
        )

        assert source is not None
        self.assertEqual(source.param_str, "general")
        self.assertEqual(
            source.general_schema_path("wine_1/default_task"),
            resolve_repo_path("outputs/schema") / "wine_1/default_task" / "schema_general_general.json",
        )

    def test_build_data_loader_config_injects_schema_paths_without_overwriting_existing_entries(self) -> None:
        general_source = resolve_schema_artifact_source(
            {"out_main": "/tmp/general", "res_param_str": "general"},
        )
        query_source = resolve_schema_artifact_source(
            {"out_main": "/tmp/query", "res_param_str": "tailored"},
        )

        loader_config = build_data_loader_config(
            base_loader_config={
                "filemap": {
                    "documents": "documents.json",
                    "schema_general": "custom_general.json",
                }
            },
            dataset="wine_1/default_task",
            general_schema_source=general_source,
            query_schema_source=query_source,
        )

        self.assertEqual(loader_config["filemap"]["documents"], "documents.json")
        self.assertEqual(loader_config["filemap"]["schema_general"], "custom_general.json")
        self.assertEqual(
            loader_config["filemap"]["schema_query"],
            "/tmp/query/wine_1/default_task/res_schema_{qid}_tailored.json",
        )

    def test_resolve_training_data_count_rejects_legacy_proxy_runtime_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "Deprecated split keys detected"):
            resolve_training_data_count({"proxy_runtime": {"training_size": 32}})

    def test_proxy_pipeline_keeps_legacy_aliases_while_preferring_proxy_naming(self) -> None:
        pipeline = ProxyPipeline(ProxyPipelineConfig())
        sentinel = object()
        pipeline._proxy_factory = sentinel  # type: ignore[assignment]

        self.assertIs(pipeline.proxy_factory, sentinel)
        self.assertIs(pipeline.guard_factory, sentinel)
        self.assertEqual(pipeline._proxy_attribute_name("learned_city"), "city")
        self.assertEqual(pipeline._guard_attribute_name("learned_city"), "city")
        self.assertEqual(
            pipeline._compute_per_proxy_target_recall(2),
            pipeline._compute_per_guard_target_recall(2),
        )


if __name__ == "__main__":
    unittest.main()

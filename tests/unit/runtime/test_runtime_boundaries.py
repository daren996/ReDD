from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from redd.config import resolve_repo_path
from redd.core.utils.data_split import resolve_training_data_count
from redd.core.utils.sql_filter_parser import AttributePredicate
from redd.proxy.proxy_runtime.config import (
    is_proxy_runtime_enabled,
    normalize_proxy_runtime_config,
)
from redd.proxy.proxy_runtime.pipeline import ProxyPipeline
from redd.proxy.proxy_runtime.types import ProxyPipelineConfig
from redd.runtime import (
    build_data_loader_config,
    ensure_shared_output_root,
    normalize_stage_config,
    resolve_schema_artifact_source,
)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_normalize_stage_config_accepts_optional_input_without_legacy_derivation(self) -> None:
        self.assertIsNone(normalize_stage_config(None, module="schemagen"))

        normalized = normalize_stage_config(
            {
                "mode": "openai",
                "datasets": ["wine"],
            },
            module="schemagen",
        )

        assert normalized is not None
        self.assertEqual(normalized["mode"], "openai")
        self.assertEqual(normalized["datasets"], ["wine"])
        self.assertNotIn("exp_dn_fn_list", normalized)

    def test_ensure_shared_output_root_rejects_mismatched_stage_outputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "share the same output root"):
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

    def test_proxy_pipeline_prefers_proxy_naming_without_legacy_aliases(self) -> None:
        pipeline = ProxyPipeline(ProxyPipelineConfig())
        sentinel = object()
        pipeline._proxy_factory = sentinel  # type: ignore[assignment]

        self.assertIs(pipeline.proxy_factory, sentinel)
        self.assertEqual(pipeline._proxy_attribute_name("learned_city"), "city")
        self.assertEqual(pipeline._compute_per_proxy_target_recall(2), 0.975)

    @patch("redd.proxy.proxy_runtime.pipeline.create_data_loader")
    def test_proxy_pipeline_uses_stable_loader_factory_signature(self, create_loader_mock) -> None:
        fake_loader = SimpleNamespace(num_docs=0)
        create_loader_mock.return_value = fake_loader

        pipeline = ProxyPipeline(
            ProxyPipelineConfig(
                dataset_path="canonical/spider.college_2",
                data_main="dataset",
            )
        )

        self.assertIs(pipeline.data_loader, fake_loader)
        create_loader_mock.assert_called_once_with(
            data_root=Path("dataset/canonical/spider.college_2"),
            loader_type="hf_manifest",
            loader_config={},
        )

    def test_proxy_pipeline_load_query_remains_callable_method(self) -> None:
        fake_loader = SimpleNamespace(
            data_root=Path("/tmp/fake-dataset"),
            get_query_info=lambda query_id: {
                "id": query_id,
                "query": "find cities",
                "sql": "select * from city",
            },
        )
        pipeline = ProxyPipeline(ProxyPipelineConfig(query_id="Q1"))
        pipeline._data_loader = fake_loader  # type: ignore[assignment]

        query_info = pipeline.load_query()

        self.assertEqual(query_info["id"], "Q1")
        self.assertEqual(query_info["query"], "find cities")

    def test_proxy_runtime_config_normalization_uses_proxy_runtime_section(self) -> None:
        config = {"proxy_runtime": {"enabled": True, "target_recall": 0.9}}

        self.assertEqual(
            normalize_proxy_runtime_config(config),
            {"enabled": True, "target_recall": 0.9},
        )
        self.assertTrue(is_proxy_runtime_enabled(config))

    def test_proxy_runtime_enablement_prefers_explicit_proxy_runtime_config(self) -> None:
        config = {
            "proxy_runtime": {"enabled": False},
            "use_proxy_runtime": True,
        }

        self.assertFalse(is_proxy_runtime_enabled(config))

    def test_proxy_pipeline_train_mode_requires_training_docs(self) -> None:
        fake_loader = SimpleNamespace(get_doc_text=lambda doc_id: "document")
        pipeline = ProxyPipeline(
            ProxyPipelineConfig(
                predicate_proxy_mode="train",
                use_embedding_proxies=False,
                save_hard_negatives=False,
            )
        )

        with self.assertRaisesRegex(ValueError, "requires training docs"):
            pipeline.run_for_documents(
                doc_ids=["d1"],
                train_doc_ids=[],
                predicates=[AttributePredicate("city", "=", "Paris")],
                table_schema={"Schema Name": "places", "Attributes": [{"Attribute Name": "city"}]},
                query_text="find Paris",
                data_loader=fake_loader,
            )

    def test_proxy_pipeline_pretrained_mode_requires_proxy_or_explicit_fallback(self) -> None:
        fake_loader = SimpleNamespace(get_doc_text=lambda doc_id: "document")
        pipeline = ProxyPipeline(
            ProxyPipelineConfig(
                predicate_proxy_mode="pretrained",
                use_embedding_proxies=True,
                allow_embedding_fallback=False,
                save_hard_negatives=False,
            )
        )
        pipeline._proxy_factory = SimpleNamespace(create_pretrained_proxies=lambda **kwargs: [])

        with self.assertRaisesRegex(RuntimeError, "No predicate proxies could be created"):
            pipeline.run_for_documents(
                doc_ids=["d1"],
                train_doc_ids=[],
                predicates=[AttributePredicate("city", "=", "Paris")],
                table_schema={"Schema Name": "places", "Attributes": [{"Attribute Name": "city"}]},
                query_text="find Paris",
                data_loader=fake_loader,
            )


if __name__ == "__main__":
    unittest.main()

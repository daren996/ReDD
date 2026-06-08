from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from redd.config import resolve_repo_path
from redd.core.utils.data_split import (
    resolve_training_data_count,
    resolve_training_data_split,
    resolve_training_data_split_seed,
    split_doc_ids,
)
from redd.core.utils.sql_filter_parser import AttributePredicate
from redd.optimizations.alpha_allocation.data_extraction_adapter import (
    DataExtractionAlphaAllocator,
)
from redd.optimizations.alpha_allocation.types import (
    STAGE_DOC_FILTERING,
    STAGE_PREDICATE_PROXY,
    AlphaAllocationConfig,
)
from redd.proxy.predicate_proxy.heuristic_proxy import HeuristicPredicateProxy
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

    def test_hash_training_split_is_deterministic_and_disjoint(self) -> None:
        doc_ids = [f"doc-{index}" for index in range(12)]

        train_a, test_a = split_doc_ids(doc_ids, 4, strategy="hash", seed=7)
        train_b, test_b = split_doc_ids(doc_ids, 4, strategy="hash", seed=7)

        self.assertEqual(train_a, train_b)
        self.assertEqual(test_a, test_b)
        self.assertEqual(len(train_a), 4)
        self.assertEqual(set(train_a).intersection(test_a), set())
        self.assertEqual(test_a, [doc_id for doc_id in doc_ids if doc_id not in set(train_a)])

    def test_resolve_training_split_uses_project_seed(self) -> None:
        config = {"training_data_split": "hashed", "project": {"seed": 123}}

        self.assertEqual(resolve_training_data_split(config), "hash")
        self.assertEqual(resolve_training_data_split_seed(config), 123)

    def test_proxy_pipeline_prefers_proxy_naming_without_legacy_aliases(self) -> None:
        pipeline = ProxyPipeline(ProxyPipelineConfig())
        sentinel = object()
        pipeline._proxy_factory = sentinel  # type: ignore[assignment]

        self.assertIs(pipeline.proxy_factory, sentinel)
        self.assertEqual(pipeline._proxy_attribute_name("learned_city"), "city")
        self.assertEqual(pipeline._compute_per_proxy_target_recall(2), 0.95)

    def test_proxy_threshold_for_target_recall_allows_zero_when_needed(self) -> None:
        threshold = ProxyPipeline._threshold_for_target_recall(
            np.array([0.0, 0.5, 1.0]),
            target_recall=0.99,
        )

        self.assertEqual(threshold, 0.0)

    def test_proxy_threshold_for_target_recall_includes_boundary_positive(self) -> None:
        scores = np.array([0.5314421362461975, 0.6], dtype=np.float64)
        threshold = ProxyPipeline._threshold_for_target_recall(
            scores,
            target_recall=0.999999,
        )

        self.assertLessEqual(threshold, scores[0])

    def test_proxy_recalibration_fails_open_without_positive_samples(self) -> None:
        class DummyProxy:
            name = "heuristic_score"
            uses_documents = True
            threshold = 0.5

            def evaluate_documents(self, documents, doc_ids=None):
                return np.array([0.0 for _ in documents]), {}

        proxy = DummyProxy()
        pipeline = ProxyPipeline(ProxyPipelineConfig(target_recall=0.95))

        pipeline._recalibrate_learned_proxies(
            proxies=[proxy],
            predicate_fns={"score": lambda value: float(value) > 10},
            calibration_doc_ids=["d1", "d2"],
            calibration_docs=["a", "b"],
            calibration_extractions={"d1": {"score": 1}, "d2": {"score": 2}},
            calibration_embeddings=None,
        )

        self.assertEqual(proxy.threshold, 0.0)

    def test_high_target_recall_keeps_heuristic_unknown_evidence_passing(self) -> None:
        proxy = HeuristicPredicateProxy(
            AttributePredicate("city", "=", "Paris"),
            threshold=0.5,
            pass_through_attributes=[],
        )
        pipeline = ProxyPipeline(ProxyPipelineConfig(target_recall=0.99))

        pipeline._recalibrate_learned_proxies(
            proxies=[proxy],
            predicate_fns={"city": lambda value: str(value) == "Paris"},
            calibration_doc_ids=["d1"],
            calibration_docs=["Paris"],
            calibration_extractions={"d1": {"city": "Paris"}},
            calibration_embeddings=None,
        )

        self.assertEqual(proxy.threshold, 0.5)

    def test_train_mode_heuristic_fallback_is_recalibrated(self) -> None:
        class DummyProxy:
            name = "heuristic_score"
            uses_documents = True
            threshold = 0.5
            cost = 1.0
            pass_rate = 0.5

            @property
            def rejection_efficiency(self) -> float:
                return (1.0 - self.pass_rate) / self.cost

            def evaluate_documents(self, documents, doc_ids=None, metadata=None):
                scores = np.array([0.0 for _ in documents], dtype=np.float32)
                return scores, scores >= self.threshold

        proxy = DummyProxy()
        pipeline = ProxyPipeline(
            ProxyPipelineConfig(
                query_id="q",
                predicate_proxy_mode="train",
                use_learned_proxies=True,
                use_finetuned_learned_proxies=False,
                use_embedding_proxies=False,
                save_hard_negatives=False,
                target_recall=0.99,
                verbose=False,
            )
        )
        pipeline.compute_embeddings = lambda documents, doc_ids=None: np.zeros(
            (len(documents), 1),
            dtype=np.float32,
        )
        pipeline._proxy_factory = SimpleNamespace(
            train_proxies=lambda **kwargs: [],
            create_pretrained_proxies=lambda **kwargs: [proxy],
        )

        def extract(document, schema, attributes, doc_id=None):
            return {"score": 0}

        def check_predicates(extracted_values, predicates):
            per_attr = {
                attr: bool(fn(extracted_values.get(attr)))
                for attr, fn in predicates.items()
            }
            return all(per_attr.values()), per_attr

        pipeline._oracle = SimpleNamespace(
            extract=extract,
            check_predicates=check_predicates,
        )
        loader = SimpleNamespace(get_doc_text=lambda doc_id: str(doc_id))

        result = pipeline.run_for_documents(
            doc_ids=["d1", "d2"],
            train_doc_ids=["t1", "t2"],
            predicates=[AttributePredicate("score", ">", "10")],
            table_schema={
                "Schema Name": "scores",
                "Attributes": [{"Attribute Name": "score"}],
            },
            query_text="scores above ten",
            data_loader=loader,
        )

        self.assertEqual(proxy.threshold, 0.0)
        self.assertEqual(result.documents_passed_proxies, 2)

    def test_effective_full_recall_target_bypasses_proxy_filtering(self) -> None:
        pipeline = ProxyPipeline(
            ProxyPipelineConfig(
                query_id="q",
                target_recall=0.999999,
                use_learned_proxies=True,
                use_embedding_proxies=False,
                save_hard_negatives=False,
                verbose=False,
            )
        )
        pipeline._proxy_factory = SimpleNamespace(
            create_pretrained_proxies=lambda **kwargs: self.fail(
                "proxy factory should not be used when predicate target is effectively 1"
            )
        )
        pipeline._oracle = SimpleNamespace(
            extract=lambda document, schema, attributes, doc_id=None: {"score": 1},
            check_predicates=lambda extracted_values, predicates: (True, {}),
        )
        loader = SimpleNamespace(get_doc_text=lambda doc_id: str(doc_id))

        result = pipeline.run_for_documents(
            doc_ids=["d1", "d2"],
            train_doc_ids=["t1"],
            predicates=[AttributePredicate("score", ">", "10")],
            table_schema={
                "Schema Name": "scores",
                "Attributes": [{"Attribute Name": "score"}],
            },
            query_text="scores above ten",
            data_loader=loader,
            extra_proxies=[SimpleNamespace(name="reject_all")],
        )

        self.assertEqual(result.documents_passed_proxies, 2)
        self.assertEqual(set(result.extractions), {"d1", "d2"})

    def test_alpha_allocation_carries_unused_budget_to_predicate_proxy(self) -> None:
        allocator = DataExtractionAlphaAllocator.__new__(DataExtractionAlphaAllocator)
        allocator.alloc_config = SimpleNamespace(
            alpha_grid=[0.0, 0.01, 0.05, 0.1, 0.2],
        )

        filled = allocator._use_remaining_budget_for_predicate_proxy(
            alpha_map={"doc_filtering": 0.0, "predicate_proxy": 0.0},
            budget_total=-np.log(0.95),
        )

        self.assertEqual(filled["doc_filtering"], 0.0)
        self.assertEqual(filled["predicate_proxy"], 0.05)

    def test_alpha_allocation_config_parses_answer_recall_calibration_flags(self) -> None:
        config = AlphaAllocationConfig.from_raw(
            {
                "enabled": True,
                "target_recall": 0.9,
                "alpha_grid": [0.2, 0.1],
                "answer_recall_calibration": True,
                "answer_recall_calibration_allow_over_budget": True,
                "answer_recall_calibration_global": True,
            }
        )

        self.assertTrue(config.answer_recall_calibration)
        self.assertTrue(config.answer_recall_calibration_allow_over_budget)
        self.assertTrue(config.answer_recall_calibration_global)
        self.assertEqual(config.alpha_grid, [0.0, 0.1, 0.2])

    def test_answer_recall_calibration_prefers_over_target_predicate_alpha(self) -> None:
        allocator = DataExtractionAlphaAllocator.__new__(DataExtractionAlphaAllocator)
        allocator.alloc_config = SimpleNamespace(
            alpha_grid=[0.0, 0.1, 0.2, 0.3],
            target_recall=0.9,
            answer_recall_calibration_allow_over_budget=True,
        )
        recalls_by_alpha = {0.0: 1.0, 0.1: 0.99, 0.2: 0.92, 0.3: 0.89}

        def estimate_answer_recall(**kwargs):
            alpha = round(float(kwargs["alpha_predicate"]), 1)
            return {
                "executable": True,
                "recall": recalls_by_alpha[alpha],
                "covered": int(recalls_by_alpha[alpha] * 100),
                "total": 100,
            }

        allocator._estimate_answer_recall_for_alphas = estimate_answer_recall

        chosen, calibration = allocator._calibrate_predicate_alpha_for_answer_recall(
            query_context={"query_info": {"sql": "select 1"}},
            alpha_map={STAGE_DOC_FILTERING: 0.0, STAGE_PREDICATE_PROXY: 0.1},
            budget_total=DataExtractionAlphaAllocator._alpha_budget(0.1),
        )

        self.assertEqual(chosen[STAGE_PREDICATE_PROXY], 0.2)
        self.assertEqual(calibration["selected_answer_recall"], 0.92)
        self.assertEqual(len(calibration["observations"]), 4)

    def test_answer_recall_calibration_can_back_off_to_preserve_target(self) -> None:
        allocator = DataExtractionAlphaAllocator.__new__(DataExtractionAlphaAllocator)
        allocator.alloc_config = SimpleNamespace(
            alpha_grid=[0.0, 0.1, 0.2],
            target_recall=0.95,
            answer_recall_calibration_allow_over_budget=False,
        )
        recalls_by_alpha = {0.0: 0.96, 0.1: 0.8, 0.2: 0.7}

        def estimate_answer_recall(**kwargs):
            alpha = round(float(kwargs["alpha_predicate"]), 1)
            return {
                "executable": True,
                "recall": recalls_by_alpha[alpha],
                "covered": int(recalls_by_alpha[alpha] * 100),
                "total": 100,
            }

        allocator._estimate_answer_recall_for_alphas = estimate_answer_recall

        chosen, calibration = allocator._calibrate_predicate_alpha_for_answer_recall(
            query_context={"query_info": {"sql": "select 1"}},
            alpha_map={STAGE_DOC_FILTERING: 0.0, STAGE_PREDICATE_PROXY: 0.1},
            budget_total=DataExtractionAlphaAllocator._alpha_budget(0.1),
        )

        self.assertEqual(chosen[STAGE_PREDICATE_PROXY], 0.0)
        self.assertEqual(calibration["selected_answer_recall"], 0.96)

    def test_answer_recall_calibration_prefers_conservative_alpha_on_recall_tie(self) -> None:
        allocator = DataExtractionAlphaAllocator.__new__(DataExtractionAlphaAllocator)
        allocator.alloc_config = SimpleNamespace(
            alpha_grid=[0.0, 0.1, 0.2, 0.3],
            target_recall=0.9,
            answer_recall_calibration_allow_over_budget=True,
        )

        def estimate_answer_recall(**kwargs):
            return {
                "executable": True,
                "recall": 1.0,
                "covered": 10,
                "total": 10,
            }

        allocator._estimate_answer_recall_for_alphas = estimate_answer_recall

        chosen, calibration = allocator._calibrate_predicate_alpha_for_answer_recall(
            query_context={"query_info": {"sql": "select 1"}},
            alpha_map={STAGE_DOC_FILTERING: 0.0, STAGE_PREDICATE_PROXY: 0.1},
            budget_total=DataExtractionAlphaAllocator._alpha_budget(0.1),
        )

        self.assertEqual(chosen[STAGE_PREDICATE_PROXY], 0.0)
        self.assertEqual(calibration["selected_answer_recall"], 1.0)

    def test_global_answer_recall_calibration_prefers_conservative_alpha_on_coverage_tie(self) -> None:
        allocator = DataExtractionAlphaAllocator.__new__(DataExtractionAlphaAllocator)
        allocator.alloc_config = SimpleNamespace(target_recall=0.9)

        selections = allocator._select_global_answer_recall_observations(
            {
                "Q1": {
                    "observations": [
                        {"alpha_predicate_proxy": 0.0, "covered": 10, "total": 10, "recall": 1.0},
                        {"alpha_predicate_proxy": 0.5, "covered": 10, "total": 10, "recall": 1.0},
                    ]
                }
            }
        )

        self.assertEqual(selections["Q1"]["alpha_predicate_proxy"], 0.0)

    def test_global_answer_recall_calibration_prefers_over_target_weighted_recall(self) -> None:
        allocator = DataExtractionAlphaAllocator.__new__(DataExtractionAlphaAllocator)
        allocator.alloc_config = SimpleNamespace(target_recall=0.9)

        selections = allocator._select_global_answer_recall_observations(
            {
                "Q1": {
                    "observations": [
                        {"alpha_predicate_proxy": 0.1, "covered": 100, "total": 100, "recall": 1.0},
                        {"alpha_predicate_proxy": 0.9, "covered": 80, "total": 100, "recall": 0.8},
                    ]
                },
                "Q2": {
                    "observations": [
                        {"alpha_predicate_proxy": 0.1, "covered": 10, "total": 10, "recall": 1.0},
                        {"alpha_predicate_proxy": 0.9, "covered": 9, "total": 10, "recall": 0.9},
                    ]
                },
            }
        )

        self.assertEqual(selections["Q1"]["alpha_predicate_proxy"], 0.1)
        self.assertEqual(selections["Q2"]["alpha_predicate_proxy"], 0.9)
        self.assertEqual(selections["__global_training_answer_covered__"], 109)
        self.assertEqual(selections["__global_training_answer_total__"], 110)

    def test_global_answer_recall_calibration_can_drop_below_per_query_closest(self) -> None:
        allocator = DataExtractionAlphaAllocator.__new__(DataExtractionAlphaAllocator)
        allocator.alloc_config = SimpleNamespace(target_recall=0.9)

        selections = allocator._select_global_answer_recall_observations(
            {
                "fixed": {
                    "observations": [
                        {"alpha_predicate_proxy": 0.9, "covered": 100, "total": 100, "recall": 1.0},
                    ]
                },
                "tunable": {
                    "observations": [
                        {"alpha_predicate_proxy": 0.1, "covered": 100, "total": 100, "recall": 1.0},
                        {"alpha_predicate_proxy": 0.9, "covered": 80, "total": 100, "recall": 0.8},
                    ]
                },
            }
        )

        self.assertEqual(selections["tunable"]["alpha_predicate_proxy"], 0.9)
        self.assertEqual(selections["__global_training_answer_covered__"], 180)
        self.assertEqual(selections["__global_training_answer_total__"], 200)

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

    @patch("redd.proxy.proxy_runtime.pipeline.EmbeddingManager")
    def test_proxy_pipeline_uses_configured_embedding_cache_dir(self, embedding_manager_mock) -> None:
        sentinel = object()
        embedding_manager_mock.return_value = sentinel
        with TemporaryDirectory() as tmp_dir:
            fake_loader = SimpleNamespace(data_root=Path("/tmp/datasets/spider.college_demo"))
            pipeline = ProxyPipeline(
                ProxyPipelineConfig(
                    embedding_model="local-hash-embedding",
                    embeddings_cache_dir=tmp_dir,
                )
            )
            pipeline._data_loader = fake_loader  # type: ignore[assignment]

            self.assertIs(pipeline.embedding_manager, sentinel)

            embedding_manager_mock.assert_called_once_with(
                storage_path=Path(tmp_dir) / "spider.college_demo.embeddings.sqlite3",
                loader=fake_loader,
                model="local-hash-embedding",
                api_key=None,
            )

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

    def test_proxy_pipeline_reuses_cross_query_extraction_cache(self) -> None:
        fake_loader = SimpleNamespace(get_doc_text=lambda doc_id: "document")
        oracle = SimpleNamespace(
            extract=lambda **kwargs: self.fail("cache hit should skip extraction"),
            check_predicates=lambda extracted_values, predicates: (True, {}),
        )
        cache = {("dataset", "places", "d1"): {"city": "Paris"}}
        pipeline = ProxyPipeline(
            ProxyPipelineConfig(
                use_embedding_proxies=False,
                use_learned_proxies=False,
                save_hard_negatives=False,
            )
        )
        pipeline._oracle = oracle  # type: ignore[assignment]

        result = pipeline.run_for_documents(
            doc_ids=["d1"],
            train_doc_ids=[],
            predicates=[],
            table_schema={"Schema Name": "places", "Attributes": [{"Attribute Name": "city"}]},
            query_text="find cities",
            data_loader=fake_loader,
            extraction_cache=cache,
            extraction_cache_namespace="dataset",
            extraction_cache_table="places",
        )

        self.assertEqual(result.cache_hit_doc_ids, ["d1"])
        self.assertEqual(result.extracted_doc_ids, [])
        self.assertEqual(result.extractions["d1"], {"city": "Paris"})

    def test_proxy_pipeline_updates_cross_query_extraction_cache(self) -> None:
        fake_loader = SimpleNamespace(get_doc_text=lambda doc_id: "document")
        calls = []

        def extract(**kwargs):
            calls.append(kwargs["doc_id"])
            return {"city": "Paris"}

        oracle = SimpleNamespace(
            extract=extract,
            check_predicates=lambda extracted_values, predicates: (True, {}),
        )
        cache = {}
        pipeline = ProxyPipeline(
            ProxyPipelineConfig(
                use_embedding_proxies=False,
                use_learned_proxies=False,
                save_hard_negatives=False,
            )
        )
        pipeline._oracle = oracle  # type: ignore[assignment]

        result = pipeline.run_for_documents(
            doc_ids=["d1"],
            train_doc_ids=[],
            predicates=[],
            table_schema={"Schema Name": "places", "Attributes": [{"Attribute Name": "city"}]},
            query_text="find cities",
            data_loader=fake_loader,
            extraction_cache=cache,
            extraction_cache_namespace="dataset",
            extraction_cache_table="places",
        )

        self.assertEqual(calls, ["d1"])
        self.assertEqual(result.extracted_doc_ids, ["d1"])
        self.assertEqual(cache, {("dataset", "places", "d1"): {"city": "Paris"}})

    def test_proxy_pipeline_keeps_extraction_when_posthoc_predicate_fails(self) -> None:
        fake_loader = SimpleNamespace(get_doc_text=lambda doc_id: "document")
        oracle = SimpleNamespace(
            extract=lambda **kwargs: {"city": "Paris"},
            check_predicates=lambda extracted_values, predicates: (False, {"city": False}),
        )
        pipeline = ProxyPipeline(
            ProxyPipelineConfig(
                use_embedding_proxies=False,
                use_learned_proxies=False,
                save_hard_negatives=False,
            )
        )
        pipeline._oracle = oracle  # type: ignore[assignment]

        result = pipeline.run_for_documents(
            doc_ids=["d1"],
            train_doc_ids=[],
            predicates=[AttributePredicate("city", "=", "Berlin")],
            table_schema={"Schema Name": "places", "Attributes": [{"Attribute Name": "city"}]},
            query_text="find cities",
            data_loader=fake_loader,
        )

        self.assertEqual(result.extractions["d1"], {"city": "Paris"})
        self.assertEqual(result.documents_extracted, 1)


if __name__ == "__main__":
    unittest.main()

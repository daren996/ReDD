from __future__ import annotations

import json
import unittest
from importlib import import_module, resources
from pathlib import Path
from unittest.mock import patch

from redd import (
    TextToSQLRequest,
    create_doc_filter,
    run_web_demo,
    schema_global,
    schema_refine,
)
from redd.cli import extract, preprocessing, schema_refinement, web
from redd.cli import run as run_cli
from redd.cli.main import build_parser
from redd.config import resolve_repo_path
from redd.runtime import resolve_dataset_roots


class PackageSmokeTests(unittest.TestCase):
    def test_lazy_exports_are_available(self) -> None:
        self.assertIs(TextToSQLRequest(query="q", schema={}).query, "q")
        doc_filter = create_doc_filter({"filter_type": "noop"})
        self.assertEqual(doc_filter.__class__.__name__, "NoOpFilter")
        self.assertTrue(callable(schema_global))
        self.assertTrue(callable(schema_refine))
        self.assertTrue(callable(run_web_demo))

    def test_future_facing_namespaces_are_importable(self) -> None:
        embedding = import_module("redd.embedding")
        llm = import_module("redd.llm")
        optimizations = import_module("redd.optimizations")
        proxy = import_module("redd.proxy")
        correction = import_module("redd.correction")
        exp = import_module("redd.exp")
        gliclass_exp = import_module("redd.exp.experiments.predicate_proxy.gliclass_pretrain_data")
        correction_ensemble = import_module("redd.correction.ensemble_analyses")
        proxy_pipeline = import_module("redd.proxy.proxy_runtime.pipeline")
        proxy_types = import_module("redd.proxy.proxy_runtime.types")
        finetuned_proxy = import_module("redd.proxy.predicate_proxy.finetuned_proxy")

        self.assertTrue(callable(embedding.EmbeddingManager))
        self.assertTrue(callable(embedding.DocumentClustering))
        self.assertTrue(callable(llm.normalize_provider_name))
        self.assertTrue(callable(optimizations.create_doc_filter))
        self.assertTrue(callable(proxy.create_join_resolver))
        self.assertEqual(correction.__name__, "redd.correction")
        self.assertEqual(exp.EvalDataExtraction.__name__, "EvalDataExtraction")
        self.assertEqual(correction.VotingErrorEstimation.__name__, "VotingErrorEstimation")
        self.assertEqual(
            correction_ensemble.EnsembleAnalyses.__module__,
            "redd.correction.ensemble_analyses",
        )
        self.assertEqual(proxy_pipeline.__name__, "redd.proxy.proxy_runtime.pipeline")
        self.assertEqual(proxy_types.__name__, "redd.proxy.proxy_runtime.types")
        self.assertEqual(
            finetuned_proxy.__name__,
            "redd.proxy.predicate_proxy.finetuned_proxy",
        )
        self.assertTrue(callable(gliclass_exp.extract_training_pairs))

    def test_removed_legacy_proxy_and_optimization_namespaces_are_not_importable(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.global_schema")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.schema_refinement")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.schema_tailoring")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.core.llm")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.core.embedding")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.core.doc_clustering")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.core.evaluation")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.core.proxy_runtime.pipeline")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.core.predicate_proxy.finetuned_proxy")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.core.doc_filtering")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.core.alpha_allocation")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.core.correction.ensemble_analyses")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.core.experiments.predicate_proxy.gliclass_pretrain_data")
        with self.assertRaises(ModuleNotFoundError):
            import_module("redd.core.repositories")

    def test_cli_parser_understands_subcommands(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["preprocess", "--config", "configs/examples/ground_truth_demo.yaml", "--experiment", "demo"]
        )
        self.assertEqual(args.command, "preprocess")
        self.assertEqual(args.experiment, "demo")

        args = parser.parse_args(
            ["refine", "--config", "configs/examples/ground_truth_demo.yaml", "--experiment", "demo"]
        )
        self.assertEqual(args.command, "refine")
        self.assertEqual(args.experiment, "demo")

        args = parser.parse_args(
            ["extract", "--config", "configs/examples/ground_truth_demo.yaml", "--experiment", "demo"]
        )
        self.assertEqual(args.command, "extract")
        self.assertEqual(args.experiment, "demo")

        args = parser.parse_args(
            ["run", "--config", "configs/examples/ground_truth_demo.yaml", "--experiment", "demo"]
        )
        self.assertEqual(args.command, "run")
        self.assertEqual(args.experiment, "demo")

        args = parser.parse_args(
            ["web", "--config", "configs/examples/ground_truth_demo.yaml", "--experiment", "demo"]
        )
        self.assertEqual(args.command, "web")
        self.assertEqual(args.experiment, "demo")

        args = parser.parse_args(["web"])
        self.assertEqual(args.command, "web")
        self.assertEqual(args.config, "configs/demo/demo_datasets.yaml")
        self.assertEqual(args.experiment, "demo")

    def test_cli_parser_rejects_removed_commands_and_flags(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["schemagen", "--config", "configs/examples/ground_truth_demo.yaml"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["correction", "--config", "configs/examples/ground_truth_demo.yaml"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["datapop", "--eval"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["preprocessing", "--config", "configs/examples/ground_truth_demo.yaml"])

    def test_stage_cli_modules_route_to_stage_runners(self) -> None:
        with patch("redd.runners.run_preprocessing") as mocked_preprocessing:
            preprocessing.main(["--config", "configs/examples/ground_truth_demo.yaml", "--experiment", "demo"])
            mocked_preprocessing.assert_called_once_with(
                "configs/examples/ground_truth_demo.yaml",
                "demo",
                api_key=None,
            )

        with patch("redd.runners.run_schema_refinement") as mocked_refinement:
            schema_refinement.main(["--config", "configs/examples/ground_truth_demo.yaml", "--experiment", "demo"])
            mocked_refinement.assert_called_once_with(
                "configs/examples/ground_truth_demo.yaml",
                "demo",
                api_key=None,
            )

        with patch("redd.runners.run_extract") as mocked_extract:
            extract.main(["--config", "configs/examples/ground_truth_demo.yaml", "--experiment", "demo"])
            mocked_extract.assert_called_once_with(
                "configs/examples/ground_truth_demo.yaml",
                "demo",
                api_key=None,
            )

        with patch("redd.runners.run_experiment") as mocked_run:
            run_cli.main(["--config", "configs/examples/ground_truth_demo.yaml", "--experiment", "demo"])
            mocked_run.assert_called_once_with(
                "configs/examples/ground_truth_demo.yaml",
                "demo",
                api_key=None,
            )

        with patch("redd.web_demo.serve_web_demo") as mocked_web:
            web.main([])
            mocked_web.assert_called_once_with(
                config_path="configs/demo/demo_datasets.yaml",
                experiment="demo",
                host="127.0.0.1",
                port=8000,
                reload=False,
            )

    def test_runtime_resolves_dataset_roots(self) -> None:
        config = {
            "data_main": "dataset",
            "out_main": "outputs/schema_gen",
            "exp_dn_fn_list": ["wine_1/default_task"],
        }
        dataset, data_root, out_root = resolve_dataset_roots(config)[0]
        self.assertEqual(dataset, "wine_1/default_task")
        self.assertEqual(data_root, resolve_repo_path("dataset") / dataset)
        self.assertEqual(out_root, resolve_repo_path("outputs/schema_gen") / dataset)

    def test_runtime_resolves_v2_context_output_root(self) -> None:
        config = {
            "_runtime_contexts": [
                {
                    "dataset": "wine",
                    "data_root": "dataset/canonical/spider.wine_1",
                    "out_root": "outputs/spider/wine/data_extraction/run-v1",
                }
            ]
        }
        dataset, data_root, out_root = resolve_dataset_roots(config)[0]
        self.assertEqual(dataset, "wine")
        self.assertEqual(data_root, resolve_repo_path("dataset/canonical/spider.wine_1"))
        self.assertEqual(out_root, resolve_repo_path("outputs/spider/wine/data_extraction/run-v1"))

    def test_web_demo_wrapper_returns_json_serializable_payload(self) -> None:
        with patch("redd.web_demo.run_pipeline", return_value={"data_extraction": []}) as mocked_pipeline:
            payload = run_web_demo(
                "configs/examples/ground_truth_demo.yaml",
                "demo",
            )

        self.assertEqual(payload["experiment"], "demo")
        self.assertEqual(payload["datasets"], ["demo"])
        self.assertEqual(payload["query_ids"], [])
        self.assertEqual(payload["stages"], ["data_extraction"])
        json.dumps(payload)
        mocked_pipeline.assert_called_once()

    def test_web_demo_wrapper_accepts_registry_dataset_selection(self) -> None:
        with patch("redd.web_demo.run_pipeline", return_value={"data_extraction": []}) as mocked_pipeline:
            payload = run_web_demo(
                "configs/examples/ground_truth_demo.yaml",
                "demo",
                datasets=["examples.single_doc_demo"],
                query_ids=["Q1"],
            )

        self.assertEqual(payload["datasets"], ["examples.single_doc_demo"])
        self.assertEqual(payload["query_ids"], ["Q1"])
        _, kwargs = mocked_pipeline.call_args
        self.assertEqual(kwargs["datasets"], ["examples.single_doc_demo"])
        contexts = kwargs["data_populator"].config["_runtime_contexts"]
        self.assertEqual(contexts[0]["dataset"], "examples.single_doc_demo")
        self.assertEqual(kwargs["data_populator"].config["exp_query_id_list"], ["Q1"])
        self.assertEqual(
            Path(contexts[0]["data_root"]),
            resolve_repo_path("dataset/canonical/examples.single_doc_demo"),
        )

    def test_readme_uses_current_cli_names(self) -> None:
        readme = resolve_repo_path("README.md").read_text(encoding="utf-8")

        self.assertIn("redd extract --config configs/examples/ground_truth_demo.yaml --experiment demo", readme)
        self.assertNotIn("redd datapop", readme)
        self.assertNotIn("redd schema-refinement", readme)
        self.assertNotIn("redd preprocessing", readme)

    def test_web_demo_static_resources_are_packaged(self) -> None:
        web_resources = resources.files("redd.resources.web_demo")

        self.assertIn("ReDD", web_resources.joinpath("index.html").read_text(encoding="utf-8"))
        self.assertIn("/api/run", web_resources.joinpath("app.js").read_text(encoding="utf-8"))
        self.assertIn(".app-shell", web_resources.joinpath("styles.css").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

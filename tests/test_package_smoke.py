from __future__ import annotations

import unittest
from importlib import import_module
from unittest.mock import patch

from redd import TextToSQLRequest, create_doc_filter, schema_global, schema_refine
from redd.cli import datapop, exp as exp_cli, preprocessing, schema_refinement
from redd.cli.main import build_parser
from redd.config import resolve_repo_path
from redd.runtime import resolve_dataset_roots, resolve_stage_output_root


class PackageSmokeTests(unittest.TestCase):
    def test_lazy_exports_are_available(self) -> None:
        self.assertIs(TextToSQLRequest(query="q", schema={}).query, "q")
        doc_filter = create_doc_filter({"filter_type": "noop"})
        self.assertEqual(doc_filter.__class__.__name__, "NoOpFilter")
        self.assertTrue(callable(schema_global))
        self.assertTrue(callable(schema_refine))

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
        self.assertEqual(exp.EvalDataPop.__name__, "EvalDataPop")
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
            ["preprocessing", "--config", "configs/siliconflow_qwen30B.yaml", "--exp", "demo"]
        )
        self.assertEqual(args.command, "preprocessing")
        self.assertEqual(args.exp, "demo")

        args = parser.parse_args(
            ["schema-refinement", "--config", "configs/siliconflow_qwen30B.yaml", "--exp", "demo"]
        )
        self.assertEqual(args.command, "schema-refinement")
        self.assertEqual(args.exp, "demo")

        args = parser.parse_args(
            ["datapop", "--config", "configs/siliconflow_qwen30B.yaml", "--exp", "demo"]
        )
        self.assertEqual(args.command, "datapop")
        self.assertEqual(args.exp, "demo")

        args = parser.parse_args(
            ["exp", "evaluation", "--config", "configs/siliconflow_qwen30B.yaml", "--exp", "demo"]
        )
        self.assertEqual(args.command, "exp")
        self.assertEqual(args.workflow, "evaluation")
        self.assertEqual(args.exp, "demo")

    def test_cli_parser_rejects_removed_commands_and_flags(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["schemagen", "--config", "configs/siliconflow_qwen30B.yaml"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["correction", "--config", "configs/siliconflow_qwen30B.yaml"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["datapop", "--eval"])

    def test_stage_cli_modules_route_to_stage_runners(self) -> None:
        with patch("redd.runners.run_preprocessing") as mocked_preprocessing:
            preprocessing.main(["--config", "configs/siliconflow_qwen30B.yaml", "--exp", "demo"])
            mocked_preprocessing.assert_called_once_with(
                "configs/siliconflow_qwen30B.yaml",
                "demo",
                api_key=None,
            )

        with patch("redd.runners.run_schema_refinement") as mocked_refinement:
            schema_refinement.main(["--config", "configs/siliconflow_qwen30B.yaml", "--exp", "demo"])
            mocked_refinement.assert_called_once_with(
                "configs/siliconflow_qwen30B.yaml",
                "demo",
                api_key=None,
            )

        with patch("redd.runners.run_datapop") as mocked_datapop:
            datapop.main(["--config", "configs/siliconflow_qwen30B.yaml", "--exp", "demo"])
            mocked_datapop.assert_called_once_with(
                "configs/siliconflow_qwen30B.yaml",
                "demo",
                api_key=None,
            )

        with patch("redd.runners.run_evaluation") as mocked_evaluation:
            exp_cli.main(
                ["evaluation", "--config", "configs/siliconflow_qwen30B.yaml", "--exp", "demo"]
            )
            mocked_evaluation.assert_called_once_with(
                "configs/siliconflow_qwen30B.yaml",
                "demo",
                api_key=None,
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

    def test_runtime_resolves_stage_specific_output_root(self) -> None:
        config = {
            "out_main": "outputs/spider",
            "output_layout": "module_under_task",
        }
        dataset = "wine_1/gt_schema_task"
        self.assertEqual(
            resolve_stage_output_root(config, dataset, module_name="data_pop"),
            resolve_repo_path("outputs/spider") / dataset / "data_pop",
        )
        _, _, out_root = resolve_dataset_roots(
            {
                **config,
                "data_main": "dataset/spider_sqlite",
                "exp_dn_fn_list": [dataset],
            },
            module_name="data_pop",
        )[0]
        self.assertEqual(out_root, resolve_repo_path("outputs/spider") / dataset / "data_pop")


if __name__ == "__main__":
    unittest.main()

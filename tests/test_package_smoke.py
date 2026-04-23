from __future__ import annotations

import unittest
from importlib import import_module

from redd import TextToSQLRequest, create_doc_filter, schema_global, schema_refine
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
        optimizations = import_module("redd.optimizations")
        proxy = import_module("redd.proxy")
        correction = import_module("redd.correction")
        exp = import_module("redd.exp")
        gliclass_exp = import_module("redd.exp.experiments.predicate_proxy.gliclass_pretrain_data")
        correction_ensemble = import_module("redd.correction.ensemble_analyses")
        proxy_pipeline = import_module("redd.proxy.proxy_runtime.pipeline")
        proxy_types = import_module("redd.proxy.proxy_runtime.types")
        finetuned_proxy = import_module("redd.proxy.predicate_proxy.finetuned_proxy")

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
            ["schemagen", "--config", "configs/schemagen.yaml", "--exp", "demo"]
        )
        self.assertEqual(args.command, "schemagen")
        self.assertEqual(args.exp, "demo")

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

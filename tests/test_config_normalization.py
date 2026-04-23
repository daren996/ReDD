from __future__ import annotations

import os
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from redd.config import load_experiment_config, normalize_experiment_config, resolve_api_key
from redd.core.utils.constants import PATH_TEMPLATES


class ConfigNormalizationTests(unittest.TestCase):
    def test_normalize_experiment_config_derives_dataset_targets_and_params(self) -> None:
        normalized = normalize_experiment_config(
            {
                "mode": "openai",
                "llm_model": "gpt-4o-mini",
                "exp_dn_list": ["wine_1"],
                "exp_fn_list": ["default_task"],
                "prompt": {"general_prompt_version": "4_0"},
            },
            module="schemagen",
        )

        self.assertEqual(normalized["mode"], "cgpt")
        self.assertEqual(normalized["exp_dn_fn_list"], ["wine_1/default_task"])
        self.assertEqual(normalized["general_param_str"], "mdlgpt-4o-mini_prm4_0")
        self.assertEqual(normalized["res_param_str"], "mdlgpt-4o-mini")
        self.assertIn("log_dir", normalized)

    def test_load_experiment_config_merges_shared_defaults_and_nested_module_section(self) -> None:
        config_text = textwrap.dedent(
            """
            shared:
              data_main: dataset
              out_main: outputs/schema
              prompt:
                prompt_path: prompts/schemagen_4_0.txt
            demo:
              llm_model: gpt-4o-mini
              schemagen:
                mode: openai
                exp_dn_list: [wine_1]
                exp_fn_list: [default_task]
            """
        ).strip()

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            config, resolved_path = load_experiment_config(
                config_path,
                "demo",
                module="schemagen",
            )

        self.assertEqual(resolved_path, config_path)
        self.assertEqual(config["data_main"], "dataset")
        self.assertEqual(config["out_main"], "outputs/schema")
        self.assertEqual(config["mode"], "cgpt")
        self.assertEqual(config["exp_dn_fn_list"], ["wine_1/default_task"])
        self.assertEqual(config["prompt"]["prompt_path"], "prompts/schemagen_4_0.txt")

    def test_load_experiment_config_supports_legacy_experiment_keys(self) -> None:
        config_text = textwrap.dedent(
            """
            demo_schemagen:
              mode: openai
              llm_model: gpt-4o-mini
              exp_dataset_task: wine_1/default_task
            """
        ).strip()

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "legacy.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            config, _ = load_experiment_config(
                config_path,
                "demo",
                module="schemagen",
            )

        self.assertEqual(config["mode"], "cgpt")
        self.assertEqual(config["exp_dn_fn_list"], ["wine_1/default_task"])

    def test_normalize_experiment_config_maps_legacy_chunk_filter_keys(self) -> None:
        normalized = normalize_experiment_config(
            {
                "mode": "deepseek",
                "llm_model": "deepseek-chat",
                "chunk_filter": {
                    "enabled": True,
                    "filter_type": "schema_relevance",
                    "target_recall": 0.9,
                },
            },
            module="datapop",
        )

        self.assertIn("doc_filter", normalized)
        self.assertTrue(normalized["doc_filter"]["enabled"])
        self.assertEqual(normalized["doc_filter"]["filter_type"], "schema_relevance")
        self.assertEqual(normalized["doc_filter"]["target_recall"], 0.9)

    def test_doc_filter_path_templates_keep_legacy_aliases(self) -> None:
        self.assertEqual(
            PATH_TEMPLATES.chunk_filter_result("Q1", "demo", 0.95),
            "chunk_filter_Q1_demo_recall095.json",
        )
        self.assertEqual(
            PATH_TEMPLATES.doc_filter_result("Q1", "demo", 0.95),
            "doc_filter_Q1_demo_recall095.json",
        )
        self.assertEqual(
            PATH_TEMPLATES.chunk_filter_eval("demo", 0.95),
            PATH_TEMPLATES.doc_filter_eval("demo", 0.95),
        )

    def test_resolve_api_key_honors_precedence(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            self.assertEqual(
                resolve_api_key({"api_key": "config-key"}, "cgpt", api_key="explicit-key"),
                "explicit-key",
            )
            self.assertEqual(resolve_api_key({"api_key": "config-key"}, "cgpt"), "config-key")
            self.assertEqual(resolve_api_key({}, "cgpt"), "env-key")


if __name__ == "__main__":
    unittest.main()

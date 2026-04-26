from __future__ import annotations

import os
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

from redd.config import (
    load_experiment_runtime,
    load_redd_config,
    resolve_api_key,
    resolve_repo_path,
    select_experiment,
)
from redd.stages.data_extraction import _runtime_query_ids_for_dataset

VALID_CONFIG = """
config_version: 2.1.1
project:
  name: demo
runtime:
  output_dir: outputs/demo
  log_dir: logs
  output_layout: dataset_stage
  artifact_id: run-v1
models:
  llm:
    provider: openai
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
  embedding:
    provider: openai
    model: text-embedding-3-small
datasets:
  wine:
    loader: hf_manifest
    root: dataset/canonical/spider.wine_1
    loader_options:
      manifest: manifest.yaml
    split:
      train_count: 16
stages:
  preprocessing:
    enabled: true
    prompt: schemagen_5_0
  schema_refinement:
    enabled: true
    source_stage: preprocessing
    document_filtering:
      enabled: true
      target_recall: 0.95
  data_extraction:
    enabled: true
    schema_source: schema_refinement
    oracle: llm
experiments:
  demo:
    datasets: [wine]
    stages: [preprocessing, schema_refinement, data_extraction]
"""


class ConfigV2Tests(unittest.TestCase):
    def test_load_redd_config_and_select_experiment(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(textwrap.dedent(VALID_CONFIG).strip(), encoding="utf-8")

            config, resolved_path = load_redd_config(config_path)
            runtime = select_experiment(config, "demo")

        self.assertEqual(resolved_path, config_path)
        self.assertEqual(runtime.id, "demo")
        self.assertEqual(runtime.dataset_ids(), ["wine"])
        self.assertEqual(runtime.stage_order, ["preprocessing", "schema_refinement", "data_extraction"])

    def test_stage_runtime_dict_compiles_v2_to_internal_runtime_contract(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(textwrap.dedent(VALID_CONFIG).strip(), encoding="utf-8")

            runtime, _ = load_experiment_runtime(config_path, "demo")
            stage_config = runtime.stage_runtime_dict("preprocessing")

        self.assertEqual(stage_config["mode"], "openai")
        self.assertEqual(stage_config["project_name"], "demo")
        self.assertEqual(stage_config["project_seed"], 42)
        self.assertEqual(stage_config["llm_model"], "gpt-4o-mini")
        self.assertEqual(stage_config["res_param_str"], "run-v1")
        self.assertEqual(stage_config["exp_dn_fn_list"], ["wine"])
        self.assertEqual(stage_config["training_data_count"], 16)
        self.assertEqual(stage_config["prompt"]["prompt_path"], "schemagen_5_0.txt")
        self.assertEqual(
            stage_config["_runtime_contexts"][0]["out_root"],
            str(resolve_repo_path("outputs/demo") / "wine" / "preprocessing" / "run-v1"),
        )

        refinement_config = runtime.stage_runtime_dict("schema_refinement")
        self.assertEqual(refinement_config["doc_filter"]["target_recall"], 0.95)
        self.assertEqual(refinement_config["doc_filter"]["threshold"], 0.585)
        self.assertFalse(refinement_config["doc_filter"]["enable_calibrate"])

    def test_optimizer_defaults_are_applied_when_enabled_without_thresholds(self) -> None:
        config_text = textwrap.dedent(VALID_CONFIG).replace(
            "  data_extraction:\n"
            "    enabled: true\n"
            "    schema_source: schema_refinement\n"
            "    oracle: llm\n",
            "  data_extraction:\n"
            "    enabled: true\n"
            "    schema_source: schema_refinement\n"
            "    oracle: llm\n"
            "    document_filtering:\n"
            "      enabled: true\n"
            "    proxy_runtime:\n"
            "      enabled: true\n",
        )

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(config_text.strip(), encoding="utf-8")

            runtime, _ = load_experiment_runtime(config_path, "demo")
            extraction_config = runtime.stage_runtime_dict("data_extraction")

        self.assertEqual(extraction_config["doc_filter"]["threshold"], 0.585)
        self.assertFalse(extraction_config["doc_filter"]["enable_calibrate"])
        self.assertEqual(extraction_config["proxy_runtime"]["predicate_proxy_mode"], "pretrained")
        self.assertEqual(extraction_config["proxy_runtime"]["proxy_threshold"], 0.51)

    def test_dataset_query_ids_are_preserved_for_stage_runtime(self) -> None:
        config_text = textwrap.dedent(VALID_CONFIG).replace(
            "    split:\n      train_count: 16",
            "    split:\n      train_count: 16\n    query_ids:\n      - q1",
        )

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(config_text.strip(), encoding="utf-8")

            runtime, _ = load_experiment_runtime(config_path, "demo")
            stage_config = runtime.stage_runtime_dict("data_extraction")

        self.assertEqual(stage_config["_runtime_contexts"][0]["query_ids"], ["q1"])
        self.assertEqual(_runtime_query_ids_for_dataset(stage_config, "wine"), ["q1"])

    def test_unknown_top_level_key_fails(self) -> None:
        config_text = textwrap.dedent(VALID_CONFIG).strip() + "\nmode: openai\n"

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaises(ValidationError):
                load_redd_config(config_path)

    def test_old_config_version_fails(self) -> None:
        config_text = textwrap.dedent(VALID_CONFIG).replace("config_version: 2.1.1", "config_version: 2")

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaises(ValidationError):
                load_redd_config(config_path)

    def test_legacy_flat_config_fails(self) -> None:
        config_text = textwrap.dedent(
            """
            mode: openai
            llm_model: gpt-4o-mini
            out_main: outputs/demo
            exp_dataset_task: wine_1/default_task
            """
        ).strip()

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "legacy.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaises(ValidationError):
                load_redd_config(config_path)

    def test_missing_required_artifact_id_fails(self) -> None:
        config_text = textwrap.dedent(VALID_CONFIG).replace("  artifact_id: run-v1\n", "")

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaises(ValidationError):
                load_redd_config(config_path)

    def test_resolve_api_key_honors_precedence(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            self.assertEqual(
                resolve_api_key({"api_key": "config-key"}, "openai", api_key="explicit-key"),
                "explicit-key",
            )
            self.assertEqual(resolve_api_key({"api_key": "config-key"}, "openai"), "config-key")
            self.assertEqual(resolve_api_key({}, "openai"), "env-key")


if __name__ == "__main__":
    unittest.main()

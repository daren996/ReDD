from __future__ import annotations

import json
import os
import textwrap
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from redd.core.utils.progress import emit_progress_event, progress_event_sink, tqdm
from redd.web_demo import (
    collect_web_evaluation_summary,
    collect_web_optimization_metrics,
    create_web_demo_app,
    delete_output_result,
)


class WebDemoTests(unittest.TestCase):
    def test_create_web_demo_app_reports_missing_optional_dependencies(self) -> None:
        with patch("redd.web_demo.importlib.import_module", side_effect=ModuleNotFoundError("fastapi")):
            with self.assertRaisesRegex(RuntimeError, "optional `web` dependencies"):
                create_web_demo_app()

    def test_defaults_endpoint_returns_demo_defaults(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-secret"}, clear=False):
            client = TestClient(create_web_demo_app())

            response = client.get("/api/defaults")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["config_path"], "configs/demo/demo_datasets.yaml")
        self.assertEqual(payload["experiment"], "demo")
        self.assertIn("data_extraction", payload["stages"])
        self.assertIn("openai", payload["model_catalog"]["llm"])
        self.assertEqual(payload["models"]["llm"]["provider"], "none")
        self.assertEqual(payload["models"]["llm"]["model"], "ground_truth")
        self.assertEqual(payload["models"]["llm"]["max_retries"], 5)
        self.assertEqual(payload["models"]["llm"]["wait_time"], 10)
        self.assertIsNone(payload["models"]["llm"]["temperature"])
        self.assertEqual(payload["models"]["embedding"]["provider"], "local")
        self.assertEqual(payload["models"]["embedding"]["model"], "local-hash-embedding")
        self.assertEqual(payload["models"]["embedding"]["batch_size"], 100)
        self.assertEqual(payload["models"]["embedding"]["storage_file"], "embeddings.sqlite3")
        self.assertIn("local-hash-embedding", response.text)
        self.assertEqual(payload["api_key_status"]["masked"], "****")
        self.assertEqual(payload["api_key_status"]["source"], "OPENAI_API_KEY")
        self.assertNotIn("env-secret", response.text)

    def test_web_config_uses_pretrained_proxy_default_without_recall_controls(self) -> None:
        root = Path(__file__).resolve().parents[3]
        html = (root / "src/redd/resources/web_demo/index.html").read_text(encoding="utf-8")
        app_js = (root / "src/redd/resources/web_demo/app.js").read_text(encoding="utf-8")
        proxy_mode_select = html.split('id="config-opt-proxy-mode"', 1)[1].split("</select>", 1)[0]

        self.assertNotIn('id="config-opt-proxy-recall"', html)
        self.assertNotIn('id="config-opt-doc-filter-recall"', html)
        self.assertNotIn('id="config-opt-doc-filter-calibrate"', html)
        self.assertLess(
            proxy_mode_select.index('<option value="pretrained">'),
            proxy_mode_select.index('<option value="auto">'),
        )
        self.assertIn('proxy.predicate_proxy_mode || "pretrained"', app_js)
        self.assertNotIn("configOptProxyRecall", app_js)
        self.assertNotIn("configOptDocFilterRecall", app_js)
        self.assertNotIn("configOptDocFilterCalibrate", app_js)
        self.assertNotIn("target_recall: optionalNumber", app_js)
        self.assertNotIn("enable_calibrate: Boolean", app_js)
        self.assertIn("function mergeOptimizationItem", app_js)
        self.assertIn("function formatOptimizationDetails", app_js)
        self.assertIn("function renderOptimizationActivityFeed", app_js)
        self.assertIn("function upsertOptimizationActivities", app_js)
        self.assertIn('id="evaluation-cards"', html)
        self.assertIn("function renderEvaluationCards", app_js)
        self.assertIn('event.type === "evaluation_update"', app_js)
        self.assertIn("Offline-only benchmark ablation", app_js)
        self.assertIn("offline_only_gt_guard", app_js)
        self.assertIn("dataset_consistency_audit", app_js)
        self.assertIn("conflict_ref", app_js)
        self.assertIn("text_values", app_js)
        self.assertIn("ground_truth_values", app_js)

    def test_config_inspect_returns_experiment_datasets_and_stages(self) -> None:
        client = TestClient(create_web_demo_app())

        response = client.get(
            "/api/config/inspect",
            params={
                "config_path": "configs/examples/ground_truth_demo.yaml",
                "experiment": "demo",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["experiment"], "demo")
        self.assertEqual(payload["dataset_ids"], ["demo"])
        self.assertIn("data_extraction", payload["default_stages"])
        self.assertEqual(payload["datasets"][0]["loader"], "hf_manifest")
        self.assertIn("gemini", payload["model_catalog"]["llm"])
        self.assertIn("openai", payload["model_catalog"]["embedding"])

    def test_config_inspect_falls_back_from_stale_experiment_selection(self) -> None:
        client = TestClient(create_web_demo_app())

        response = client.get(
            "/api/config/inspect",
            params={
                "config_path": "configs/demo/demo_datasets.yaml",
                "experiment": "demo_all",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["experiment"], "demo")
        self.assertEqual(payload["dataset_ids"], ["spider.college_demo", "bird.schools_demo"])
        self.assertEqual([item["id"] for item in payload["experiments"]], ["demo"])

    def test_config_inspect_masks_inline_api_key(self) -> None:
        config_text = textwrap.dedent(
            """
            config_version: 2.1.1
            project:
              name: secret-demo
            runtime:
              output_dir: outputs/demo
              log_dir: logs
              output_layout: dataset_stage
              artifact_id: secret-run
            models:
              llm:
                provider: openai
                model: gpt-4o-mini
                api_key: inline-secret
              embedding: null
            datasets:
              demo:
                loader: hf_manifest
                root: dataset/canonical/examples.single_doc_demo
                loader_options:
                  manifest: manifest.yaml
                split:
                  train_count: 0
            stages:
              data_extraction:
                enabled: true
                schema_source: ground_truth
                oracle: ground_truth
            experiments:
              demo:
                datasets: [demo]
                stages: [data_extraction]
            """
        ).strip()
        client = TestClient(create_web_demo_app())

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")
            response = client.get(
                "/api/config/inspect",
                params={"config_path": str(config_path), "experiment": "demo"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotIn("inline-secret", response.text)
        self.assertNotIn("api_key", payload["models"]["llm"])
        self.assertEqual(payload["api_key_status"]["masked"], "****")
        self.assertEqual(payload["api_key_status"]["source"], "config.models.llm.api_key")

    def test_dataset_registry_endpoints_return_registered_dataset_payloads(self) -> None:
        client = TestClient(create_web_demo_app())

        index = client.get("/api/datasets")
        detail = client.get("/api/datasets/examples.single_doc_demo")
        documents = client.get("/api/datasets/examples.single_doc_demo/documents", params={"limit": 1})
        schema = client.get("/api/datasets/examples.single_doc_demo/schema")
        queries = client.get("/api/datasets/examples.single_doc_demo/queries")

        self.assertEqual(index.status_code, 200)
        dataset_ids = [item["id"] for item in index.json()["datasets"]]
        self.assertIn("examples.single_doc_demo", set(dataset_ids))
        previous_last_position = -1
        for dataset_group in ("examples", "spider", "bird", "galois", "quest", "fda", "cuad"):
            positions = [
                position
                for position, dataset_id in enumerate(dataset_ids)
                if dataset_id.split(".", 1)[0] == dataset_group
            ]
            if positions:
                self.assertGreater(min(positions), previous_last_position)
                previous_last_position = max(positions)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["id"], "examples.single_doc_demo")
        self.assertEqual(documents.status_code, 200)
        self.assertLessEqual(len(documents.json()["documents"]), 1)
        self.assertEqual(schema.status_code, 200)
        self.assertIn("schema", schema.json())
        self.assertEqual(queries.status_code, 200)
        self.assertIn("queries", queries.json())

    def test_empty_dataset_queries_endpoint_returns_default_extraction_query(self) -> None:
        client = TestClient(create_web_demo_app())

        response = client.get("/api/datasets/quest.lcr/queries")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["default_extraction"])
        self.assertEqual(payload["queries"][0]["query_id"], "default")
        self.assertEqual(payload["queries"][0]["output_columns"], [])
        self.assertIn("every attribute", payload["queries"][0]["question"])

    def test_results_endpoint_loads_persisted_outputs(self) -> None:
        client = TestClient(create_web_demo_app())

        response = client.get("/api/results", params={"limit": 5000})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        result = next(
            item
            for item in payload["results"]
            if item["stage"] == "data_extraction" and item["records_count"]
        )
        self.assertTrue(result["relative_path"].endswith(".json"))
        self.assertEqual(result["stage"], "data_extraction")
        self.assertGreaterEqual(result["records_count"], 1)
        self.assertIsInstance(result["tables"], list)
        self.assertIsInstance(result["columns"], list)

    def test_delete_output_result_restricts_to_outputs_relative_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "project" / "demo" / "data_extraction" / "run"
            result_path.mkdir(parents=True)
            file_path = result_path / "res_tabular_data_Q1_run.json"
            file_path.write_text(json.dumps({"doc": {"res": "wine", "data": {}}}), encoding="utf-8")

            payload = delete_output_result(
                "project/demo/data_extraction/run/res_tabular_data_Q1_run.json",
                outputs_dir=root,
            )

            self.assertTrue(payload["deleted"])
            self.assertFalse(file_path.exists())
            with self.assertRaisesRegex(ValueError, "relative path"):
                delete_output_result(str(file_path), outputs_dir=root)
            with self.assertRaisesRegex(ValueError, "stay under"):
                delete_output_result("../outside.json", outputs_dir=root)

    def test_run_endpoint_delegates_to_wrapper(self) -> None:
        client = TestClient(create_web_demo_app())
        result = {
            "experiment": "demo",
            "datasets": ["demo"],
            "stages": ["data_extraction"],
            "result": {"data_extraction": []},
        }

        with patch("redd.web_demo.run_web_demo", return_value=result) as mocked:
            response = client.post(
                "/api/run",
                json={
                    "config_path": "configs/examples/ground_truth_demo.yaml",
                    "experiment": "demo",
                    "stages": ["data_extraction"],
                    "datasets": ["demo"],
                    "api_key": "secret",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["result"], {"data_extraction": []})
        self.assertIn("elapsed_seconds", payload)
        mocked.assert_called_once_with(
            "configs/examples/ground_truth_demo.yaml",
            "demo",
            stages=["data_extraction"],
            datasets=["demo"],
            query_ids=None,
            api_key="secret",
        )

    def test_run_endpoint_passes_force_rerun_when_enabled(self) -> None:
        client = TestClient(create_web_demo_app())
        result = {
            "experiment": "demo",
            "datasets": ["demo"],
            "stages": ["data_extraction"],
            "result": {"data_extraction": []},
        }

        with patch("redd.web_demo.run_web_demo", return_value=result) as mocked:
            response = client.post(
                "/api/run",
                json={
                    "config_path": "configs/examples/ground_truth_demo.yaml",
                    "experiment": "demo",
                    "stages": ["data_extraction"],
                    "datasets": ["demo"],
                    "force_rerun": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once_with(
            "configs/examples/ground_truth_demo.yaml",
            "demo",
            stages=["data_extraction"],
            datasets=["demo"],
            query_ids=None,
            api_key=None,
            force_rerun=True,
        )

    def test_run_endpoint_passes_generated_config_to_wrapper(self) -> None:
        client = TestClient(create_web_demo_app())
        result = {
            "experiment": "edited",
            "datasets": ["demo"],
            "stages": ["data_extraction"],
            "result": {"data_extraction": []},
        }
        generated_config = {
            "config_version": "2.1.1",
            "project": {"name": "edited-project", "seed": 42},
            "runtime": {
                "output_dir": "outputs/demo",
                "log_dir": "logs",
                "output_layout": "dataset_stage",
                "artifact_id": "edited-artifact",
            },
            "models": {"llm": None, "embedding": None},
            "datasets": {
                "demo": {
                    "loader": "hf_manifest",
                    "root": "dataset/canonical/examples.single_doc_demo",
                    "loader_options": {"manifest": "manifest.yaml"},
                    "split": {"train_count": 0},
                }
            },
            "stages": {
                "data_extraction": {
                    "enabled": True,
                    "schema_source": "ground_truth",
                    "oracle": "ground_truth",
                }
            },
            "experiments": {"edited": {"datasets": ["demo"], "stages": ["data_extraction"]}},
        }

        with patch("redd.web_demo.run_web_demo", return_value=result) as mocked:
            response = client.post(
                "/api/run",
                json={
                    "config_path": "configs/examples/ground_truth_demo.yaml",
                    "experiment": "edited",
                    "config": generated_config,
                    "stages": ["data_extraction"],
                    "datasets": ["demo"],
                },
            )

        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once_with(
            "configs/examples/ground_truth_demo.yaml",
            "edited",
            stages=["data_extraction"],
            datasets=["demo"],
            query_ids=None,
            api_key=None,
            config=generated_config,
        )

    def test_run_endpoint_returns_error_json(self) -> None:
        client = TestClient(create_web_demo_app())

        with patch("redd.web_demo.run_web_demo", side_effect=ValueError("bad config")):
            response = client.post("/api/run", json={"config_path": "missing.yaml"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["error"], "ValueError")

    def test_async_run_endpoint_streams_completed_events(self) -> None:
        client = TestClient(create_web_demo_app())
        result = {
            "experiment": "demo",
            "datasets": ["demo"],
            "query_ids": [],
            "stages": ["data_extraction"],
            "result": {"data_extraction": []},
        }

        with patch("redd.web_demo.run_web_demo", return_value=result):
            create_response = client.post(
                "/api/runs",
                json={
                    "config_path": "configs/examples/ground_truth_demo.yaml",
                    "experiment": "demo",
                    "stages": ["data_extraction"],
                    "datasets": ["demo"],
                },
            )
            self.assertEqual(create_response.status_code, 200)
            run_id = create_response.json()["run_id"]

            for _ in range(50):
                snapshot = client.get(f"/api/runs/{run_id}").json()
                if snapshot["status"] == "completed":
                    break
                time.sleep(0.02)

        self.assertEqual(snapshot["status"], "completed")
        events_response = client.get(f"/api/runs/{run_id}/events")
        self.assertEqual(events_response.status_code, 200)
        self.assertIn("run_completed", events_response.text)
        self.assertIn("collect_optimization_metrics", events_response.text)

    def test_async_run_endpoint_streams_failed_events(self) -> None:
        client = TestClient(create_web_demo_app())

        with patch("redd.web_demo.run_web_demo", side_effect=ValueError("bad config")):
            create_response = client.post("/api/runs", json={"config_path": "missing.yaml"})
            self.assertEqual(create_response.status_code, 200)
            run_id = create_response.json()["run_id"]

            for _ in range(50):
                snapshot = client.get(f"/api/runs/{run_id}").json()
                if snapshot["status"] == "failed":
                    break
                time.sleep(0.02)

        self.assertEqual(snapshot["status"], "failed")
        events_response = client.get(f"/api/runs/{run_id}/events")
        self.assertEqual(events_response.status_code, 200)
        self.assertIn("run_failed", events_response.text)
        self.assertIn("bad config", events_response.text)

    def test_collect_optimization_metrics_reads_filter_and_proxy_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            (out_root / "doc_filter").mkdir()
            (out_root / "doc_filter" / "doc_filter_q1.json").write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "excluded_doc_ids": ["d3"],
                        "kept_doc_ids": ["d1", "d2"],
                        "metadata": {
                            "num_docs_input": 3,
                            "num_docs_excluded": 1,
                            "num_docs_kept": 2,
                            "target_recall": 0.95,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (out_root / "res_q1_demo_adaptive_stats.json").write_text(
                json.dumps(
                    {
                        "total_documents": 3,
                        "filtered_documents": 2,
                        "documents_processed": 1,
                        "documents_saved": 1,
                        "stopped_early": True,
                    }
                ),
                encoding="utf-8",
            )
            (out_root / "table_assignment_cache.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "events": [
                            {
                                "dataset": "demo",
                                "query_id": "q1",
                                "input_docs": 3,
                                "cache_hits": 2,
                                "cache_misses": 1,
                                "source_table_metadata_misses": 1,
                                "excluded": 0,
                            },
                            {
                                "dataset": "demo",
                                "query_id": "q2",
                                "input_docs": 3,
                                "cache_hits": 1,
                                "cache_misses": 2,
                                "excluded": 0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (out_root / "res_tabular_data_q1_demo_proxy_decisions.json").write_text(
                json.dumps(
                    {
                        "students": {
                            "proxy_stats": {
                                "learned_city": {"evaluated": 3, "passed": 2, "rejected": 1},
                                "join_resolver_school": {
                                    "evaluated": 2,
                                    "passed": 1,
                                    "rejected": 1,
                                },
                            },
                            "proxy_recalls": {
                                "learned_city": {"recall": 1.0, "precision": 0.5}
                            },
                            "all_doc_ids": ["d1", "d2", "d3"],
                            "passed_doc_ids": ["d1"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (out_root / "res_tabular_data_q1_demo.json").write_text(
                json.dumps({"d1": {"res": "students", "data": {"name": "Ada"}}}),
                encoding="utf-8",
            )
            (out_root / "eval_q1_demo.json").write_text(
                json.dumps(
                    {
                        "query_aware": {
                            "query_id": "q1",
                            "summary": {"can_answer_query": True},
                            "table_assignment": {"covered": 2, "total": 2, "recall": 1.0},
                            "cell_recall": {"covered": 3, "total": 4, "recall": 0.75},
                            "answer_recall": {"covered": 1, "total": 2, "recall": 0.5},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (out_root / "eval_q2_demo.json").write_text(
                json.dumps(
                    {
                        "query_aware": {
                            "query_id": "q2",
                            "summary": {"can_answer_query": False},
                            "table_assignment": {"covered": 0, "total": 1, "recall": 0.0},
                            "cell_recall": {"covered": 0, "total": 1, "recall": 0.0},
                            "answer_recall": {"covered": 0, "total": 1, "recall": 0.0},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (out_root / "dataset_consistency_audit.json").write_text(
                json.dumps(
                    {
                        "total_conflicts": 1,
                        "conflicts_by_type": {"text_pass_gt_fail": 1},
                        "datasets": [
                            {
                                "dataset": "demo",
                                "checked_doc_predicates": 3,
                                "conflicts": [
                                    {
                                        "dataset": "demo",
                                        "query_id": "q1",
                                        "doc_id": "d3",
                                        "attribute": "score",
                                        "conflict_type": "text_pass_gt_fail",
                                        "text_values": [513],
                                        "ground_truth_values": [448],
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = {
                "datasets": ["demo"],
                "query_ids": ["q1"],
                "result": {
                    "data_extraction": [
                        {
                            "dataset": "demo",
                            "out_root": str(out_root),
                        }
                    ]
                },
            }

            metrics = {item["id"]: item for item in collect_web_optimization_metrics(payload)}
            evaluation = collect_web_evaluation_summary(payload)

        self.assertEqual(metrics["doc_filter"]["status"], "measured")
        self.assertEqual(metrics["doc_filter"]["metrics"]["excluded_docs"], 1)
        self.assertEqual(metrics["doc_filter"]["metrics"]["llm_doc_calls_before"], 3)
        self.assertEqual(metrics["doc_filter"]["metrics"]["llm_doc_calls_after"], 2)
        self.assertEqual(metrics["doc_filter"]["metrics"]["llm_doc_calls_saved"], 1)
        self.assertEqual(metrics["doc_filter"]["details"][0]["query_id"], "q1")
        self.assertEqual(metrics["doc_filter"]["details"][0]["excluded_doc_ids_preview"], ["d3"])
        self.assertEqual(metrics["schema_adaptive"]["metrics"]["documents_saved"], 1)
        self.assertEqual(metrics["table_assignment_cache"]["metrics"]["table_assignment_calls_before"], 3)
        self.assertEqual(metrics["table_assignment_cache"]["metrics"]["table_assignment_calls_after"], 1)
        self.assertEqual(metrics["table_assignment_cache"]["metrics"]["table_assignment_calls_saved"], 2)
        self.assertEqual(metrics["table_assignment_cache"]["metrics"]["source_table_metadata_misses"], 1)
        self.assertEqual(metrics["table_assignment_cache"]["details"][0]["query_id"], "q1")
        self.assertEqual(metrics["proxy_runtime"]["metrics"]["evaluated"], 5)
        self.assertEqual(metrics["proxy_runtime"]["metrics"]["llm_doc_calls_before"], 3)
        self.assertEqual(metrics["proxy_runtime"]["metrics"]["llm_doc_calls_after"], 1)
        self.assertEqual(metrics["proxy_runtime"]["metrics"]["llm_doc_calls_saved"], 2)
        self.assertEqual(metrics["proxy_runtime"]["details"][0]["query_id"], "q1")
        self.assertEqual(metrics["proxy_runtime"]["details"][0]["rejected_doc_ids_total"], 2)
        self.assertEqual(metrics["join_proxy"]["metrics"]["join_proxies"], 1)
        self.assertEqual(metrics["dataset_consistency_audit"]["status"], "measured")
        self.assertEqual(metrics["dataset_consistency_audit"]["metrics"]["total_conflicts"], 1)
        self.assertEqual(metrics["dataset_consistency_audit"]["metrics"]["text_pass_gt_fail"], 1)
        self.assertEqual(metrics["dataset_consistency_audit"]["metrics"]["affected_docs"], 1)
        self.assertEqual(
            metrics["dataset_consistency_audit"]["details"][0]["conflict_type"],
            "text_pass_gt_fail",
        )
        self.assertEqual(
            metrics["dataset_consistency_audit"]["details"][0]["conflict_ref"],
            "demo::q1::d3::score",
        )
        self.assertEqual(metrics["extraction"]["metrics"]["records"], 1)
        self.assertEqual(evaluation["status"], "measured")
        self.assertEqual(evaluation["summary"]["queries"], 1)
        self.assertEqual(evaluation["summary"]["can_answer"], 1)
        self.assertEqual(evaluation["summary"]["answer_covered"], 1)
        self.assertEqual(evaluation["summary"]["answer_total"], 2)
        self.assertEqual(evaluation["queries"][0]["query_id"], "q1")

    def test_collect_optimization_metrics_omits_disabled_optimizations(self) -> None:
        payload = {"datasets": ["demo"], "query_ids": [], "result": {"data_extraction": []}}

        metrics = collect_web_optimization_metrics(payload)

        metric_ids = {item["id"] for item in metrics}
        self.assertNotIn("doc_filter", metric_ids)
        self.assertNotIn("schema_adaptive", metric_ids)
        self.assertNotIn("table_assignment_cache", metric_ids)
        self.assertNotIn("proxy_runtime", metric_ids)
        self.assertIn("extraction", metric_ids)

    def test_collect_optimization_metrics_generates_dataset_audit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            output_dir = root / "outputs"
            out_root = output_dir / "demo" / "data_extraction" / "artifact"
            (dataset_root / "data").mkdir(parents=True)
            (dataset_root / "metadata").mkdir()
            out_root.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "dataset_id": "demo",
                        "doc_id": "d1",
                        "doc_text": "The school reached an average of 513 in math.",
                    }
                ]
            ).to_parquet(dataset_root / "data" / "documents.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset_id": "demo",
                        "doc_id": "d1",
                        "record_id": "1",
                        "table_id": "scores",
                        "column_id": "scores.avg_scr_math",
                        "column_name": "avg_scr_math",
                        "value": "448",
                        "value_type": "string",
                        "source_row_id": "1",
                    }
                ]
            ).to_parquet(dataset_root / "data" / "ground_truth.parquet", index=False)
            (dataset_root / "metadata" / "queries.json").write_text(
                """
                {
                  "schema_version": "redd.queries.v1",
                  "dataset_id": "demo",
                  "queries": [
                    {
                      "query_id": "q1",
                      "sql": "SELECT avg_scr_math FROM scores WHERE avg_scr_math >= 473;"
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            payload = {
                "datasets": ["demo"],
                "query_ids": ["q1"],
                "dataset_roots": {
                    "demo": {"root": str(dataset_root), "query_ids": ["q1"]}
                },
                "result": {
                    "data_extraction": [
                        {
                            "dataset": "demo",
                            "out_root": str(out_root),
                        }
                    ]
                },
            }

            metrics = {item["id"]: item for item in collect_web_optimization_metrics(payload)}
            audit = metrics["dataset_consistency_audit"]

            self.assertEqual(audit["status"], "measured")
            self.assertEqual(audit["metrics"]["total_conflicts"], 1)
            self.assertEqual(audit["metrics"]["text_pass_gt_fail"], 1)
            self.assertEqual(audit["details"][0]["doc_id"], "d1")
            self.assertTrue((output_dir / "dataset_consistency_audit.json").exists())
            self.assertTrue((output_dir / "dataset_consistency_audit_conflicts.jsonl").exists())
            self.assertTrue((output_dir / "dataset_consistency_audit_conflicts.csv").exists())

    def test_collect_optimization_metrics_marks_gt_guard_offline_only(self) -> None:
        with TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            (out_root / "res_tabular_data_q1_demo_proxy_decisions.json").write_text(
                json.dumps(
                    {
                        "students": {
                            "proxy_stats": {
                                "gt_text_consistency_students": {
                                    "evaluated": 3,
                                    "passed": 2,
                                    "rejected": 1,
                                }
                            },
                            "all_doc_ids": ["d1", "d2", "d3"],
                            "passed_doc_ids": ["d1", "d2"],
                            "extracted_doc_ids": ["d1", "d2"],
                            "proxy_rejected_doc_ids": {
                                "gt_text_consistency_students": ["d3"],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            payload = {
                "datasets": ["demo"],
                "query_ids": ["q1"],
                "result": {
                    "data_extraction": [
                        {
                            "dataset": "demo",
                            "out_root": str(out_root),
                        }
                    ]
                },
            }

            metrics = {item["id"]: item for item in collect_web_optimization_metrics(payload)}

        proxy = metrics["proxy_runtime"]
        self.assertIn("Offline GT Guard", proxy["title"])
        self.assertIn("Offline-only", proxy["message"])
        self.assertEqual(proxy["metrics"]["offline_only_gt_guard"], "enabled")
        self.assertEqual(proxy["metrics"]["gt_guard_rejected_doc_calls"], 1)
        self.assertTrue(proxy["details"][0]["offline_only_gt_guard"])
        self.assertEqual(proxy["details"][0]["gt_guard_rejected_doc_calls"], 1)

    def test_progress_event_sink_emits_structured_updates(self) -> None:
        events: list[dict[str, object]] = []

        with progress_event_sink(events.append):
            bar = tqdm(total=2, desc="Table Assignment demo-q1")
            bar.update(1)
            bar.update(1)
            emit_progress_event(
                {
                    "type": "optimization_update",
                    "optimization": {
                        "id": "doc_filter",
                        "status": "running",
                        "metrics": {"llm_doc_calls_saved": 1},
                    },
                }
            )
            bar.close()

        progress_events = [event for event in events if event.get("type") == "progress_update"]
        optimization_events = [
            event for event in events if event.get("type") == "optimization_update"
        ]
        self.assertGreaterEqual(len(progress_events), 2)
        final = progress_events[-1]["progress"]
        self.assertEqual(final["label"], "Table Assignment demo-q1")
        self.assertEqual(final["current"], 2)
        self.assertEqual(final["total"], 2)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(optimization_events[0]["optimization"]["id"], "doc_filter")

    def test_index_and_assets_are_packaged(self) -> None:
        client = TestClient(create_web_demo_app())

        index = client.get("/")
        script = client.get("/assets/app.js")
        styles = client.get("/assets/styles.css")
        unknown_paper = client.get("/assets/papers/unpublished.pdf")
        paper = client.get("/assets/papers/2025_ReDD.pdf")
        paper_head = client.head("/assets/papers/2025_ReDD.pdf")

        self.assertEqual(index.status_code, 200)
        self.assertIn("ReDD", index.text)
        self.assertIn("data-page-button=\"operators\"", index.text)
        self.assertIn("data-source=\"upload\"", index.text)
        self.assertIn("refresh-button", index.text)
        self.assertIn("refresh-label\">Refresh", index.text)
        self.assertIn("results-library", index.text)
        self.assertIn("optimization-feed", index.text)
        self.assertIn("force-rerun", index.text)
        self.assertIn("config-llm-provider", index.text)
        self.assertIn("config-llm-temperature", index.text)
        self.assertIn("config-llm-local-model-path", index.text)
        self.assertIn("config-embedding-model", index.text)
        self.assertIn("config-embedding-storage-file", index.text)
        self.assertNotIn("config-models-json", index.text)
        self.assertIn("upload-source-panel", index.text)
        self.assertIn("theme-toggle", index.text)
        self.assertEqual(script.status_code, 200)
        self.assertIn("/api/run", script.text)
        self.assertIn("/api/runs", script.text)
        self.assertIn('method: "DELETE"', script.text)
        self.assertIn("CUSTOM_MODEL_VALUE", script.text)
        self.assertIn("paperCatalog", script.text)
        self.assertIn("/api/results", script.text)
        self.assertIn("function renderOutputResults()", script.text)
        self.assertIn("Restart the web demo server once", script.text)
        self.assertIn("function configDatasetItems()", script.text)
        self.assertIn("renderDatasetSelect(datasetSelectItems())", script.text)
        self.assertIn('datasetSourceLabel.textContent = "From Config"', script.text)
        self.assertIn("all schema attributes", script.text)
        self.assertIn("/assets/papers/2025_ReDD.pdf?preview=1", script.text)
        self.assertIn('downloadHref: "/assets/papers/2025_ReDD.pdf"', script.text)
        self.assertEqual(styles.status_code, 200)
        self.assertIn(".app-shell", styles.text)
        self.assertIn('[data-theme="dark"]', styles.text)
        self.assertEqual(unknown_paper.status_code, 404)
        self.assertEqual(paper.status_code, 200)
        self.assertEqual(paper.headers["content-type"], "application/pdf")
        self.assertEqual(paper.headers["cache-control"], "no-store")
        self.assertNotIn("content-disposition", paper.headers)
        self.assertEqual(paper_head.status_code, 200)
        self.assertEqual(paper_head.headers["content-type"], "application/pdf")
        self.assertEqual(paper_head.headers["cache-control"], "no-store")
        self.assertNotIn("content-disposition", paper_head.headers)

    def test_not_found_routes_use_themed_html_outside_api(self) -> None:
        client = TestClient(create_web_demo_app())

        page_response = client.get("/missing-page")
        api_response = client.get("/api/missing")

        self.assertEqual(page_response.status_code, 404)
        self.assertIn("not-found-page", page_response.text)
        self.assertIn("redd-theme", page_response.text)
        self.assertEqual(api_response.status_code, 404)
        self.assertEqual(api_response.json(), {"detail": "Not Found"})


if __name__ == "__main__":
    unittest.main()

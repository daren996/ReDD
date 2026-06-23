"""Evaluation workflows for ReDD.

These modules live under `redd.exp` because they are useful workflows, but not
part of the primary runtime stage surface.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from redd.core.data_loader import DataLoaderBase, create_data_loader
from redd.core.utils import constants
from redd.core.utils.constants import (
    ATTRIBUTE_NAME_KEY,
    ATTRIBUTE_VALUE_KEY,
    ATTRIBUTES_KEY,
    GROUND_TRUTH_KEY,
    PATH_TEMPLATES,
    PREDICTION_KEY,
    RESULT_DATA_KEY,
    RESULT_RECORD_ID_KEY,
    SCHEMA_NAME_KEY,
)
from redd.core.utils.data_split import (
    resolve_training_data_count,
    resolve_training_data_split,
    resolve_training_data_split_seed,
    split_doc_ids,
)
from redd.core.utils.extraction_records import active_result_records
from redd.core.utils.sql_filter_parser import analyze_sql_predicates, get_join_graph
from redd.core.utils.utils import is_null

__all__ = [
    "EvalBasic",
    "EvalDataExtraction",
]


class EvalBasic(ABC):
    """Abstract base class for evaluation tasks."""

    def __init__(
        self,
        config: Dict[str, Any],
        data_loader: Optional[DataLoaderBase] = None,
    ):
        self.config = config
        self.data_loader = data_loader
        self.prediction_data: Optional[List[Dict[str, Any]]] = None
        self.gt_data: Optional[List[Dict[str, Any]]] = None
        logging.info(
            "[%s:__init__] Initialized evaluator with config keys: %s",
            self.__class__.__name__,
            list(config.keys()),
        )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.__call__ must be implemented by subclasses")

    def compute_stat(self) -> Optional[Tuple[int, int, int, int]]:
        if not self._validate_data():
            return None
        return self._compute_basic_stats()

    def _validate_data(self) -> bool:
        if not self.prediction_data or not self.gt_data:
            logging.error(
                "[%s:_validate_data] No data loaded. prediction_data: %s, gt_data: %s",
                self.__class__.__name__,
                bool(self.prediction_data),
                bool(self.gt_data),
            )
            return False

        pred_len = len(self.prediction_data)
        gt_len = len(self.gt_data)
        if pred_len != gt_len:
            logging.error(
                "[%s:_validate_data] Data length mismatch. Predictions: %s, Ground truth: %s",
                self.__class__.__name__,
                pred_len,
                gt_len,
            )
            return False

        logging.info(
            "[%s:_validate_data] Data validation passed. Evaluating %s instances.",
            self.__class__.__name__,
            pred_len,
        )
        return True

    def _compute_basic_stats(self) -> Tuple[int, int, int, int]:
        true_positives = false_positives = false_negatives = true_negatives = 0

        for predicted, ground_truth in zip(self.prediction_data, self.gt_data):
            pred_table = predicted.get("table")
            gt_table = ground_truth.get("table")

            is_pred_non_null = not is_null(pred_table)
            is_gt_non_null = not is_null(gt_table)

            if is_pred_non_null and is_gt_non_null:
                if pred_table == gt_table:
                    true_positives += 1
                else:
                    false_positives += 1
            elif is_pred_non_null and not is_gt_non_null:
                false_positives += 1
            elif not is_pred_non_null and is_gt_non_null:
                false_negatives += 1
            else:
                true_negatives += 1

        logging.info(
            "[%s:_compute_basic_stats] TP=%s, FP=%s, FN=%s, TN=%s",
            self.__class__.__name__,
            true_positives,
            false_positives,
            false_negatives,
            true_negatives,
        )
        return true_positives, false_positives, false_negatives, true_negatives

    def compute_recall_precision_f1(
        self,
        tp: int,
        fp: int,
        fn: int,
    ) -> Tuple[float, float, float]:
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_score = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
        logging.debug(
            "[%s:compute_recall_precision_f1] Recall=%.4f, Precision=%.4f, F1=%.4f",
            self.__class__.__name__,
            recall,
            precision,
            f1_score,
        )
        return recall, precision, f1_score

    def compute_accuracy(self, correct: int, total: int) -> float:
        if total <= 0:
            logging.warning(
                "[%s:compute_accuracy] Total count is %s, returning 0.0",
                self.__class__.__name__,
                total,
            )
            return 0.0
        accuracy = correct / total
        logging.debug(
            "[%s:compute_accuracy] Accuracy=%.4f (%s/%s)",
            self.__class__.__name__,
            accuracy,
            correct,
            total,
        )
        return accuracy

    def save_results(
        self,
        result_path: Union[str, Path],
        result_dict: Dict[str, Any],
        encoding: str = "utf-8",
    ) -> None:
        try:
            result_path = Path(result_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("w", encoding=encoding) as file:
                json.dump(result_dict, file, indent=2, ensure_ascii=False)
            file_size = result_path.stat().st_size
            logging.info(
                "[%s:save_results] Results saved to %s (%s bytes, %s entries)",
                self.__class__.__name__,
                result_path,
                file_size,
                len(result_dict),
            )
        except Exception as exc:
            logging.error(
                "[%s:save_results] Failed to save results to %s: %s",
                self.__class__.__name__,
                result_path,
                exc,
            )
            raise

    def load_json(
        self,
        file_path: Union[str, Path],
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            logging.error("[%s:load_json] File not found: %s", self.__class__.__name__, file_path)
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        try:
            with file_path.open("r", encoding=encoding) as file:
                data = json.load(file)
            entry_count = len(data) if isinstance(data, (dict, list)) else "N/A"
            logging.info(
                "[%s:load_json] Loaded %s (%s entries)",
                self.__class__.__name__,
                file_path,
                entry_count,
            )
            return data
        except json.JSONDecodeError as exc:
            logging.error(
                "[%s:load_json] Invalid JSON in file %s: %s",
                self.__class__.__name__,
                file_path,
                exc,
            )
            raise
        except Exception as exc:
            logging.error(
                "[%s:load_json] Failed to load %s: %s",
                self.__class__.__name__,
                file_path,
                exc,
            )
            raise

    def display_metrics(
        self,
        title: str,
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        correct: Optional[int] = None,
        total: Optional[int] = None,
        width: int = 80,
    ) -> None:
        recall, precision, f1 = self.compute_recall_precision_f1(tp, fp, fn)

        print("\n" + "=" * width)
        print(title)
        print("-" * width)
        print(f"{'Metric':<20}{'Value'}")
        print("-" * width)
        print(f"{'True Positives':<20}{tp}")
        print(f"{'False Positives':<20}{fp}")
        print(f"{'False Negatives':<20}{fn}")
        print(f"{'True Negatives':<20}{tn}")
        print(f"{'Recall':<20}{recall:.4f}")
        print(f"{'Precision':<20}{precision:.4f}")
        print(f"{'F1 Score':<20}{f1:.4f}")

        if correct is not None and total is not None:
            accuracy = self.compute_accuracy(correct, total)
            print(f"{'Accuracy':<20}{correct}/{total} = {accuracy:.4f}")

        print("=" * width)


@dataclass
class EvaluationMetrics:
    """Container for data-extraction evaluation metrics."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    correct_count: int = 0
    total_count: int = 0
    doc_stats: Dict[str, Dict[str, Any]] | None = None
    attr_stats: Dict[str, Dict[str, int]] | None = None

    def __post_init__(self):
        if self.doc_stats is None:
            self.doc_stats = {}
        if self.attr_stats is None:
            self.attr_stats = {}

    def to_tuple(self) -> Tuple[int, int, int, int, int, int, Dict[str, Any], Dict[str, Dict[str, int]]]:
        return (
            self.true_positives,
            self.false_positives,
            self.false_negatives,
            self.true_negatives,
            self.correct_count,
            self.total_count,
            self.doc_stats,
            self.attr_stats,
        )


@dataclass
class QueryAwareEvaluation:
    """Recall-oriented evaluation of whether extracted rows can answer a query."""

    query_id: str
    summary: Dict[str, Any]
    table_assignment: Dict[str, Any]
    cell_recall: Dict[str, Any]
    answer_recall: Dict[str, Any]
    doc_details: Dict[str, Any]
    missing_cells: List[Dict[str, Any]]
    extra_cells: List[Dict[str, Any]]
    required_cell_layers: Optional[Dict[str, Any]] = None
    semantic_cell_accuracy: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "query_id": self.query_id,
            "summary": self.summary,
            "table_assignment": self.table_assignment,
            "cell_recall": self.cell_recall,
            "answer_recall": self.answer_recall,
            "doc_details": self.doc_details,
            "missing_cells": self.missing_cells,
            "extra_cells": self.extra_cells,
        }
        if self.required_cell_layers is not None:
            payload["required_cell_layers"] = self.required_cell_layers
        if self.semantic_cell_accuracy is not None:
            payload["semantic_cell_accuracy"] = self.semantic_cell_accuracy
        return payload


class EvalDataExtraction(EvalBasic):
    """Data-extraction evaluation with optional LLM-based semantic comparison."""

    def __init__(
        self,
        config: Dict[str, Any],
        data_loader: DataLoaderBase | None = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config, data_loader)

        self.loader_type = str(config.get("data_loader_type", "hf_manifest")).lower()
        self.loader_config = deepcopy(dict(config.get("data_loader_config") or {}))

        eval_config = config.get("eval", {})
        self.res_param_str = config.get("res_param_str", "default")
        self.name_map: Optional[Dict[str, Any]] = None
        self.committee_prompts: List[Dict[str, Any]] = []
        self.prompts: Dict[str, Any] = {}
        self.eval_mode = ""
        self.eval_api_key: Optional[str] = None
        self.eval_llm_model = ""
        self.full_table_semantic_enabled = bool(eval_config.get("full_table_semantic", True))
        self._current_eval_query_id: Optional[str] = None
        semantic_context_config = eval_config.get("semantic_context", {})
        if isinstance(semantic_context_config, bool):
            semantic_context_config = {"enabled": semantic_context_config}
        if not isinstance(semantic_context_config, dict):
            semantic_context_config = {}
        self.semantic_context_enabled = bool(semantic_context_config.get("enabled", True))
        self.semantic_context_include_schema = bool(semantic_context_config.get("include_schema", True))
        self.semantic_context_include_query = bool(semantic_context_config.get("include_query", True))
        self.semantic_context_include_cell_role = bool(semantic_context_config.get("include_cell_role", True))
        self.semantic_context_include_doc_text = semantic_context_config.get("include_doc_text", False)
        try:
            self.semantic_context_doc_text_max_chars = max(
                0,
                int(semantic_context_config.get("doc_text_max_chars", 1200) or 0),
            )
        except (TypeError, ValueError):
            self.semantic_context_doc_text_max_chars = 1200

        if "committee" in eval_config:
            self._initialize_committee(eval_config, api_key)
        else:
            self._initialize_single_eval(eval_config, api_key)

    def _initialize_single_eval(self, eval_config: Dict[str, Any], api_key: Optional[str]) -> None:
        if not eval_config:
            return

        from redd.llm import get_api_key
        from redd.llm.providers import normalize_provider_name

        self.eval_mode = normalize_provider_name(eval_config.get("mode", "deepseek"))
        self.eval_api_key = get_api_key(eval_config, self.eval_mode, api_key)
        if "prompts" in eval_config:
            self._initialize_prompts(eval_config)

    def _initialize_prompts(self, eval_config: Dict[str, Any]) -> None:
        from redd.core.utils.prompt_utils import create_prompt_map

        try:
            self.eval_llm_model = eval_config.get("llm_model", "deepseek-chat")
            self.prompts = create_prompt_map(
                self.eval_mode,
                eval_config["prompts"],
                llm_model=self.eval_llm_model,
                api_key=self.eval_api_key,
                config=eval_config,
            )
        except Exception as exc:
            logging.warning(
                "[%s:_initialize_prompts] Failed to initialize prompts: %s",
                self.__class__.__name__,
                exc,
            )
            self.prompts = {}

    def _initialize_committee(self, eval_config: Dict[str, Any], api_key: Optional[str] = None) -> None:
        from redd.core.utils.prompt_utils import create_prompt_map
        from redd.llm import get_api_key
        from redd.llm.providers import normalize_provider_name

        committee_config = eval_config["committee"]
        prompt_paths = eval_config["prompts"]

        for member_config in committee_config:
            mode = normalize_provider_name(member_config["mode"])
            llm_model = member_config["llm_model"]
            member_api_key = get_api_key(member_config, mode, api_key)
            member_prompts = create_prompt_map(
                mode,
                prompt_paths,
                llm_model=llm_model,
                api_key=member_api_key,
                config=member_config,
            )
            self.committee_prompts.append(
                {
                    "mode": mode,
                    "llm_model": llm_model,
                    "prompts": member_prompts,
                }
            )

        logging.info(
            "[%s:_initialize_committee] Committee: %s members",
            self.__class__.__name__,
            len(self.committee_prompts),
        )

    def __call__(self, dataset_list: Optional[List[str]] = None) -> None:
        runtime_contexts = self.config.get("_runtime_contexts")
        if isinstance(runtime_contexts, list) and runtime_contexts:
            requested = set(str(dataset) for dataset in dataset_list) if dataset_list else None
            for context in runtime_contexts:
                if not isinstance(context, dict):
                    continue
                dataset_name = str(context["dataset"])
                if requested is not None and dataset_name not in requested:
                    continue
                self._evaluate_dataset(
                    dataset_name,
                    context["data_root"],
                    context["out_root"],
                    query_ids=context.get("query_ids"),
                    loader_config=context.get("loader_options"),
                )
            return

        if dataset_list is None:
            if "exp_dn_fn_list" in self.config:
                dataset_list = self.config["exp_dn_fn_list"]
            else:
                logging.warning(
                    "[%s:__call__] No datasets specified, using default SPIDER_DN_FN_LIST",
                    self.__class__.__name__,
                )
                dataset_list = constants.SPIDER_DN_FN_LIST

        logging.info(
            "[%s:__call__] Evaluating %s datasets: %s",
            self.__class__.__name__,
            len(dataset_list),
            dataset_list,
        )

        for dataset_name in dataset_list:
            data_root = Path(self.config.get("data_main", self.config["out_main"])) / dataset_name
            out_root = Path(self.config["out_main"]) / dataset_name
            logging.info(
                "[%s:__call__] Start evaluating dataset: data_root=%s, out_root=%s",
                self.__class__.__name__,
                data_root,
                out_root,
            )
            self._evaluate_dataset(dataset_name, data_root, out_root)

    def _evaluate_dataset(
        self,
        dataset_name: str,
        data_root: str | Path,
        out_root: str | Path,
        query_ids: Optional[List[str]] = None,
        loader_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.out_root = Path(out_root)

        if not self.data_root.exists():
            logging.error("[%s:_evaluate_dataset] Data directory not found: %s", self.__class__.__name__, self.data_root)
            return
        if not self.out_root.exists():
            logging.error("[%s:_evaluate_dataset] Output directory not found: %s", self.__class__.__name__, self.out_root)
            return

        loader = create_data_loader(
            self.data_root,
            loader_type=self.loader_type,
            loader_config=deepcopy(loader_config if loader_config is not None else self.loader_config),
        )
        logging.info(
            "[%s:_evaluate_dataset] Created %s for %s",
            self.__class__.__name__,
            loader.__class__.__name__,
            self.data_root,
        )

        try:
            query_dict = loader.load_query_dict()
        except Exception as exc:
            logging.error(
                "[%s:_evaluate_dataset] Failed to load query dict for %s: %s",
                self.__class__.__name__,
                dataset_name,
                exc,
            )
            return

        if not query_dict:
            logging.warning("[%s:_evaluate_dataset] No queries found in dataset %s", self.__class__.__name__, self.data_root)
            return

        logging.info(
            "[%s:_evaluate_dataset] Evaluating %s queries in dataset %s",
            self.__class__.__name__,
            len(query_dict),
            self.data_root.name,
        )

        selected_query_ids = [str(query_id) for query_id in query_ids] if query_ids else list(query_dict)
        for query_id in selected_query_ids:
            if query_id not in query_dict:
                logging.warning(
                    "[%s:_evaluate_dataset] Query %s not found in dataset %s; skipping",
                    self.__class__.__name__,
                    query_id,
                    dataset_name,
                )
                continue
            try:
                self._evaluate_query(loader, dataset_name, query_id, query_dict[query_id])
            except Exception as exc:
                logging.error(
                    "[%s:_evaluate_dataset] Failed to evaluate query %s in dataset %s: %s",
                    self.__class__.__name__,
                    query_id,
                    dataset_name,
                    exc,
                )

    def _evaluate_query(
        self,
        loader: DataLoaderBase,
        dataset_name: str,
        query_id: str,
        query_info: Dict[str, Any],
    ) -> None:
        query = query_info.get("query", "")
        result_path = self.out_root / PATH_TEMPLATES.data_extraction_result(query_id, self.res_param_str)

        try:
            result_dict = self.load_json(result_path)
        except Exception as exc:
            logging.error(
                "[%s:_evaluate_query] Failed to load results from %s: %s",
                self.__class__.__name__,
                result_path,
                exc,
            )
            return

        logging.info("[%s:_evaluate_query] Start evaluating query %s: %s", self.__class__.__name__, query_id, query)
        self._current_eval_query_id = query_id
        self.name_map = self._load_or_generate_mapping(query_id, loader)
        self.prediction_data, self.gt_data = self._prepare_evaluation_data(loader, result_dict)
        query_aware_eval = self.compute_query_aware_statistics(loader, result_dict, query_id, query_info)

        if not self.prediction_data and not self.gt_data:
            logging.info(
                "[%s:_evaluate_query] No legacy prediction rows for query %s; "
                "saving query-aware evaluation with empty legacy stats.",
                self.__class__.__name__,
                query_id,
            )
            stats = (0, 0, 0, 0, 0, 0, {}, {})
        else:
            stats = self.compute_statistics()
        if stats is None:
            logging.info(
                "[%s:_evaluate_query] Legacy statistics unavailable for query %s; "
                "saving query-aware evaluation with empty legacy stats.",
                self.__class__.__name__,
                query_id,
            )
            stats = (0, 0, 0, 0, 0, 0, {}, {})

        tp, fp, fn, tn, correct, total, doc_stats, attr_stats = stats
        if query_aware_eval.semantic_cell_accuracy is not None:
            query_semantic_match_path = self._save_query_aware_semantic_match_results(
                query_id,
                query_aware_eval.semantic_cell_accuracy,
            )
            query_aware_eval.semantic_cell_accuracy["match_result_path"] = str(query_semantic_match_path)

        full_table_semantic_stats = (
            self.compute_semantic_statistics()
            if self._semantic_eval_enabled() and self.full_table_semantic_enabled
            else None
        )
        if full_table_semantic_stats is not None:
            semantic_match_path = self._save_full_table_semantic_match_results(
                query_id,
                full_table_semantic_stats,
            )
            full_table_semantic_stats["match_result_path"] = str(semantic_match_path)
        optimization_summary = self._collect_query_optimization_summary(query_id)
        self._display_results(dataset_name, query_id, tp, fp, fn, tn, correct, total, attr_stats)
        self._display_query_aware_results(query_aware_eval)
        self._display_optimization_summary(optimization_summary)

        eval_results = {
            "query_aware": query_aware_eval.to_dict(),
            "legacy": {"doc_stats": doc_stats, "attr_stats": attr_stats},
            "optimization_summary": optimization_summary,
        }
        if full_table_semantic_stats is not None:
            eval_results["full_table_semantic"] = full_table_semantic_stats
            eval_results["semantic"] = {
                "scope": "full_table_all_gt_attributes",
                "deprecated_alias_for": "full_table_semantic",
                "summary": full_table_semantic_stats.get("summary", {}),
                "match_result_path": full_table_semantic_stats.get("match_result_path"),
            }
        eval_path = self.out_root / PATH_TEMPLATES.eval_result(query_id, self.res_param_str)
        self.save_results(eval_path, eval_results)
        logging.info("[%s:_evaluate_query] Evaluation results saved to %s", self.__class__.__name__, eval_path)

    def _prepare_evaluation_data(
        self,
        loader: DataLoaderBase,
        result_dict: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        prediction_data = []
        ground_truth_data = []

        for doc_id in result_dict:
            cur_info = loader.get_doc_info(doc_id)
            if not cur_info:
                logging.warning(
                    "[%s:_prepare_evaluation_data] doc-%s not found in ground truth, skipping",
                    self.__class__.__name__,
                    doc_id,
                )
                continue

            pred_records = active_result_records(result_dict.get(doc_id))
            gt_records = cur_info.get("data_records") or []
            if not gt_records and cur_info.get("table"):
                gt_records = [{"table_name": cur_info.get("table"), "data": cur_info.get("data", {})}]

            used_pred_indices: set[int] = set()
            for gt_index, gt_record in enumerate(gt_records):
                if not isinstance(gt_record, dict):
                    continue
                gt_table = gt_record.get("table_name") or gt_record.get("table")
                gt_data = gt_record.get("data", {})
                if not isinstance(gt_data, dict):
                    gt_data = {}

                candidate_indices = [
                    index
                    for index, record in enumerate(pred_records)
                    if index not in used_pred_indices and record.get("table") == gt_table
                ]
                pred_record: Dict[str, Any] = {}
                if candidate_indices:
                    pred_index = max(
                        candidate_indices,
                        key=lambda index: sum(
                            1
                            for attr, value in gt_data.items()
                            if self._compare_attribute_values(
                                pred_records[index].get(RESULT_DATA_KEY, {}).get(attr)
                                if isinstance(pred_records[index].get(RESULT_DATA_KEY, {}), dict)
                                else None,
                                value,
                            )
                        ),
                    )
                    used_pred_indices.add(pred_index)
                    pred_record = pred_records[pred_index]

                prediction_data.append(
                    {
                        "doc_id": doc_id,
                        "record_id": pred_record.get(RESULT_RECORD_ID_KEY),
                        "table": pred_record.get("table"),
                        "data": pred_record.get(RESULT_DATA_KEY, {}),
                    }
                )
                ground_truth_data.append(
                    {
                        "doc_id": doc_id,
                        "record_id": (
                            gt_record.get(RESULT_RECORD_ID_KEY)
                            or gt_record.get("record_id")
                            or gt_record.get("row_id")
                            or gt_record.get("source_row_id")
                            or gt_index
                        ),
                        "table": gt_table,
                        "data": gt_data,
                    }
                )

            if not gt_records and pred_records:
                for pred_record in pred_records:
                    prediction_data.append(
                        {
                            "doc_id": doc_id,
                            "record_id": pred_record.get(RESULT_RECORD_ID_KEY),
                            "table": pred_record.get("table"),
                            "data": pred_record.get(RESULT_DATA_KEY, {}),
                        }
                    )
                    ground_truth_data.append({"doc_id": doc_id, "table": None, "data": {}})

        return prediction_data, ground_truth_data

    def compute_query_aware_statistics(
        self,
        loader: DataLoaderBase,
        result_dict: Dict[str, Any],
        query_id: str,
        query_info: Dict[str, Any],
    ) -> QueryAwareEvaluation:
        """Evaluate the query-required extraction as recall.

        Extra extracted rows or attributes are reported but do not reduce recall.
        A required cell is covered only when the document is assigned to the
        correct required table and the extracted value matches the GT value.
        """

        required_by_table = self._required_attrs_by_table(loader, query_id, query_info)
        cell_role_by_table, layer_metadata = self._required_cell_roles(
            loader=loader,
            query_id=query_id,
            query_info=query_info,
            required_by_table=required_by_table,
        )
        layer_stats = self._init_required_cell_layer_stats()
        required_tables = set(required_by_table)
        _, eval_doc_ids = split_doc_ids(
            loader.doc_ids,
            resolve_training_data_count(self.config),
            strategy=resolve_training_data_split(self.config),
            seed=resolve_training_data_split_seed(self.config),
        )
        answer_doc_ids_by_table = self._answer_doc_ids_by_table(
            loader=loader,
            query_info=query_info,
            eval_doc_ids=eval_doc_ids,
            required_by_table=required_by_table,
        )

        table_total = table_covered = table_missing = table_wrong = 0
        cell_total = cell_covered = cell_missing = cell_mismatched = 0
        null_gt_cells_skipped = 0
        extra_cells: List[Dict[str, Any]] = []
        missing_cells: List[Dict[str, Any]] = []
        doc_details: Dict[str, Any] = {}
        relevant_doc_ids: set[str] = set()
        semantic_eval_enabled = self._semantic_eval_enabled()
        semantic_total = semantic_correct = semantic_missing = semantic_mismatched = 0
        semantic_table_mismatched = semantic_llm_judged = 0
        semantic_cells: List[Dict[str, Any]] = []

        for doc_id in eval_doc_ids:
            gt_records = self._query_required_gt_records(
                loader,
                doc_id,
                required_by_table,
                answer_doc_ids_by_table=answer_doc_ids_by_table,
            )
            pred_entry = result_dict.get(doc_id) if isinstance(result_dict.get(doc_id), dict) else {}
            pred_records = active_result_records(pred_entry)
            pred_tables = [record.get("table") for record in pred_records]
            used_pred_indices: set[int] = set()

            doc_detail = {
                "pred_table": pred_tables[0] if pred_tables else None,
                "pred_tables": pred_tables,
                "required_tables": [],
                "table_ok": True,
                "cells_total": 0,
                "cells_covered": 0,
                "missing": [],
                "mismatched": [],
            }

            for gt_record in gt_records:
                gt_table = gt_record["table"]
                gt_data = gt_record["data"]
                relevant_doc_ids.add(doc_id)
                doc_detail["required_tables"].append(gt_table)
                table_total += 1

                candidate_indices = [
                    index
                    for index, record in enumerate(pred_records)
                    if index not in used_pred_indices and record.get("table") == gt_table
                ]
                pred_index = None
                pred_data: Dict[str, Any] = {}
                if candidate_indices:
                    attrs_for_table = required_by_table.get(gt_table, [])

                    def score_candidate(index: int) -> tuple[int, int]:
                        data = pred_records[index].get(RESULT_DATA_KEY, {})
                        if not isinstance(data, dict):
                            data = {}
                        strict_matches = sum(
                            1
                            for attr in attrs_for_table
                            if self._compare_attribute_values(data.get(attr), gt_data.get(attr))
                        )
                        non_null_values = sum(
                            1 for attr in attrs_for_table if not is_null(data.get(attr))
                        )
                        return strict_matches, non_null_values

                    pred_index = max(candidate_indices, key=score_candidate)
                    used_pred_indices.add(pred_index)
                    raw_pred_data = pred_records[pred_index].get(RESULT_DATA_KEY, {})
                    pred_data = raw_pred_data if isinstance(raw_pred_data, dict) else {}

                pred_table = (
                    pred_records[pred_index].get("table")
                    if pred_index is not None
                    else (pred_tables[0] if pred_tables else None)
                )
                table_ok = pred_index is not None
                if table_ok:
                    table_covered += 1
                else:
                    doc_detail["table_ok"] = False
                    if is_null(pred_table):
                        table_missing += 1
                    else:
                        table_wrong += 1

                for attr in required_by_table.get(gt_table, []):
                    gt_value = gt_data.get(attr)
                    if is_null(gt_value):
                        null_gt_cells_skipped += 1
                        continue

                    role = cell_role_by_table.get(gt_table, {}).get(attr, "other_required")
                    layer_stat = layer_stats.setdefault(role, self._empty_required_cell_layer_stats())
                    cell_total += 1
                    layer_stat["total"] += 1
                    doc_detail["cells_total"] += 1
                    pred_value = pred_data.get(attr) if table_ok else None
                    cell_ok = table_ok and self._compare_attribute_values(pred_value, gt_value)
                    if semantic_eval_enabled:
                        semantic_total += 1
                        layer_stat["semantic_total"] += 1
                        semantic_context = None
                        if not table_ok:
                            semantic_comparison = {
                                "result": False,
                                "method": "table_mismatch",
                                "reasoning": "The predicted row is not assigned to the required table.",
                                "cached": False,
                            }
                        else:
                            semantic_context = None
                            if (
                                self.semantic_context_enabled
                                and not self._compare_attribute_values(pred_value, gt_value)
                                and not is_null(pred_value)
                                and not is_null(gt_value)
                            ):
                                semantic_context = self._semantic_cell_context(
                                    loader=loader,
                                    query_id=query_id,
                                    query_info=query_info,
                                    doc_id=doc_id,
                                    table=gt_table,
                                    attr=str(attr),
                                    cell_role=role,
                                    pred_value=pred_value,
                                    gt_value=gt_value,
                                )
                            semantic_comparison = self._semantic_match_attribute(
                                pred_attr=str(attr),
                                pred_value=pred_value,
                                gt_attr=str(attr),
                                gt_value=gt_value,
                                context=semantic_context,
                            )
                        semantic_ok = bool(semantic_comparison.get("result"))
                        if semantic_ok:
                            semantic_correct += 1
                            layer_stat["semantic_correct"] += 1
                        else:
                            semantic_method = str(semantic_comparison.get("method") or "")
                            if semantic_method == "table_mismatch":
                                semantic_table_mismatched += 1
                                layer_stat["semantic_table_mismatched"] += 1
                            elif is_null(pred_value):
                                semantic_missing += 1
                                layer_stat["semantic_missing"] += 1
                            else:
                                semantic_mismatched += 1
                                layer_stat["semantic_mismatched"] += 1
                        if semantic_comparison.get("method") in {"llm", "committee_llm"}:
                            semantic_llm_judged += 1
                            layer_stat["semantic_llm_judged"] += 1
                        semantic_cells.append(
                            {
                                "doc_id": doc_id,
                                "table": gt_table,
                                "attr": attr,
                                "cell_role": role,
                                "gt": gt_value,
                                "pred_table": pred_table,
                                "pred": pred_value,
                                **semantic_comparison,
                            }
                        )
                        if semantic_context:
                            semantic_cells[-1]["context"] = semantic_context
                    if cell_ok:
                        cell_covered += 1
                        layer_stat["covered"] += 1
                        doc_detail["cells_covered"] += 1
                        continue

                    miss = {
                        "doc_id": doc_id,
                        "table": gt_table,
                        "attr": attr,
                        "gt": gt_value,
                        "pred_table": pred_table,
                        "pred": pred_value,
                        "reason": "missing" if is_null(pred_value) else "mismatch",
                    }
                    missing_cells.append(miss)
                    if is_null(pred_value):
                        cell_missing += 1
                        layer_stat["missing"] += 1
                        doc_detail["missing"].append(miss)
                    else:
                        cell_mismatched += 1
                        layer_stat["mismatched"] += 1
                        doc_detail["mismatched"].append(miss)

            if not gt_records:
                for record in pred_records:
                    pred_table = record.get("table")
                    pred_data = record.get(RESULT_DATA_KEY, {})
                    if not isinstance(pred_data, dict) or is_null(pred_table):
                        continue
                    for attr, pred_value in pred_data.items():
                        if not is_null(pred_value):
                            extra_cells.append(
                                {
                                    "doc_id": doc_id,
                                    "table": pred_table,
                                    "attr": attr,
                                    "pred": pred_value,
                                    "reason": "doc_not_required_for_query",
                                }
                            )
            else:
                for index, record in enumerate(pred_records):
                    pred_table = record.get("table")
                    if pred_table not in required_tables:
                        continue
                    pred_data = record.get(RESULT_DATA_KEY, {})
                    if not isinstance(pred_data, dict):
                        continue
                    expected_attrs = set(required_by_table.get(str(pred_table), []))
                    if index not in used_pred_indices:
                        for attr, pred_value in pred_data.items():
                            if not is_null(pred_value):
                                extra_cells.append(
                                    {
                                        "doc_id": doc_id,
                                        "table": pred_table,
                                        "attr": attr,
                                        "pred": pred_value,
                                        "reason": "extra_record_for_query",
                                    }
                                )
                        continue
                    for attr, pred_value in pred_data.items():
                        if attr not in expected_attrs and not is_null(pred_value):
                            extra_cells.append(
                                {
                                    "doc_id": doc_id,
                                    "table": pred_table,
                                    "attr": attr,
                                    "pred": pred_value,
                                    "reason": "attr_not_required_for_query",
                                }
                            )

            if gt_records or pred_records:
                doc_details[doc_id] = doc_detail

        answer_recall = self._evaluate_answer_recall(
            loader=loader,
            result_dict=result_dict,
            query_info=query_info,
            eval_doc_ids=eval_doc_ids,
            required_by_table=required_by_table,
        )
        cell_recall_value = cell_covered / cell_total if cell_total else 1.0
        table_recall_value = table_covered / table_total if table_total else 1.0
        required_cell_layers = self._finalize_required_cell_layers(
            layer_stats=layer_stats,
            metadata=layer_metadata,
            semantic_eval_enabled=semantic_eval_enabled,
        )
        semantic_cell_accuracy = None
        if semantic_eval_enabled:
            semantic_cell_accuracy = {
                "scope": "query_required_answer_cells",
                "correct": semantic_correct,
                "total": semantic_total,
                "missing": semantic_missing,
                "mismatched": semantic_mismatched,
                "table_mismatched": semantic_table_mismatched,
                "null_gt_skipped": null_gt_cells_skipped,
                "llm_judged": semantic_llm_judged,
                "context": self._semantic_context_summary(),
                "accuracy": semantic_correct / semantic_total if semantic_total else 1.0,
                "cells": semantic_cells,
            }
        answer_recall_value = answer_recall.get("recall")
        answer_recall_ok = bool(answer_recall.get("executable", False)) and answer_recall_value == 1.0
        if answer_recall.get("reason") == "query_has_no_sql":
            answer_recall_ok = True

        summary = {
            "can_answer_query": bool(
                cell_covered == cell_total
                and table_covered == table_total
                and answer_recall_ok
            ),
            "query_required_tables": sorted(required_tables),
            "query_required_attrs": {
                table: list(attrs) for table, attrs in sorted(required_by_table.items())
            },
            "evaluated_docs": len(eval_doc_ids),
            "relevant_docs": len(relevant_doc_ids),
            "prediction_docs": len(result_dict),
            "redundant_cells": len(extra_cells),
        }

        return QueryAwareEvaluation(
            query_id=query_id,
            summary=summary,
            table_assignment={
                "covered": table_covered,
                "total": table_total,
                "missing": table_missing,
                "wrong": table_wrong,
                "recall": table_recall_value,
            },
            cell_recall={
                "covered": cell_covered,
                "total": cell_total,
                "missing": cell_missing,
                "mismatched": cell_mismatched,
                "null_gt_skipped": null_gt_cells_skipped,
                "recall": cell_recall_value,
            },
            answer_recall=answer_recall,
            doc_details=doc_details,
            missing_cells=missing_cells,
            extra_cells=extra_cells,
            required_cell_layers=required_cell_layers,
            semantic_cell_accuracy=semantic_cell_accuracy,
        )

    @staticmethod
    def _empty_required_cell_layer_stats() -> Dict[str, Any]:
        return {
            "covered": 0,
            "total": 0,
            "missing": 0,
            "mismatched": 0,
            "semantic_correct": 0,
            "semantic_total": 0,
            "semantic_missing": 0,
            "semantic_mismatched": 0,
            "semantic_table_mismatched": 0,
            "semantic_llm_judged": 0,
        }

    def _init_required_cell_layer_stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            role: self._empty_required_cell_layer_stats()
            for role in ("answer", "predicate", "join", "other_required")
        }

    def _required_cell_roles(
        self,
        *,
        loader: DataLoaderBase,
        query_id: str,
        query_info: Dict[str, Any],
        required_by_table: Dict[str, List[str]],
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
        roles: Dict[str, Dict[str, str]] = {
            table: {attr: "other_required" for attr in attrs}
            for table, attrs in required_by_table.items()
        }
        role_sources: Dict[str, Dict[str, List[str]]] = {
            role: {} for role in ("answer", "predicate", "join", "other_required")
        }

        def mark(table: str, attr: str, role: str) -> None:
            if table not in roles or attr not in roles[table]:
                return
            priority = {"other_required": 0, "join": 1, "predicate": 2, "answer": 3}
            if priority[role] >= priority.get(roles[table][attr], 0):
                roles[table][attr] = role
            role_sources.setdefault(role, {}).setdefault(table, [])
            if attr not in role_sources[role][table]:
                role_sources[role][table].append(attr)

        for column_id in query_info.get("output_columns") or []:
            table, attr = self._split_attr_ref(str(column_id))
            if table and attr:
                mark(table, attr, "answer")

        sql = str(query_info.get("sql") or "")
        predicate_safety = None
        if sql:
            try:
                schema_query = loader.load_schema_query(query_id)
                predicate_safety = analyze_sql_predicates(
                    sql,
                    schema_query,
                    query_tables=query_info.get("tables") or list(required_by_table),
                    query_info=query_info,
                )
                for table, predicates in predicate_safety.safe_predicates_by_table.items():
                    for predicate in predicates:
                        mark(table, predicate.attribute, "predicate")
                for item in predicate_safety.unsafe_predicates:
                    predicate = item.get("predicate", {})
                    table = str(item.get("table") or "")
                    attr = str(predicate.get("attribute") or "")
                    if table and attr:
                        mark(table, attr, "predicate")

                join_graph = get_join_graph(
                    sql,
                    schema_query,
                    query_tables=query_info.get("tables") or list(required_by_table),
                )
                if join_graph:
                    for condition in join_graph.conditions:
                        mark(condition.table_parent, condition.attr_parent, "join")
                        mark(condition.table_child, condition.attr_child, "join")
            except Exception as exc:
                logging.debug(
                    "[%s:_required_cell_roles] Could not derive cell roles for query %s: %s",
                    self.__class__.__name__,
                    query_id,
                    exc,
                )

        for table, attrs in roles.items():
            for attr, role in attrs.items():
                role_sources.setdefault(role, {}).setdefault(table, [])
                if attr not in role_sources[role][table]:
                    role_sources[role][table].append(attr)

        return roles, {
            "role_priority": ["answer", "predicate", "join", "other_required"],
            "columns_by_role": role_sources,
            "predicate_safety": predicate_safety.to_dict() if predicate_safety is not None else None,
        }

    @staticmethod
    def _finalize_required_cell_layers(
        *,
        layer_stats: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Any],
        semantic_eval_enabled: bool,
    ) -> Dict[str, Any]:
        finalized: Dict[str, Any] = {"metadata": metadata, "layers": {}}
        for role, stats in layer_stats.items():
            role_stats = dict(stats)
            total = int(role_stats.get("total") or 0)
            role_stats["recall"] = (role_stats["covered"] / total) if total else 1.0
            if semantic_eval_enabled:
                semantic_total = int(role_stats.get("semantic_total") or 0)
                role_stats["semantic_accuracy"] = (
                    role_stats["semantic_correct"] / semantic_total
                    if semantic_total
                    else 1.0
                )
            else:
                for key in list(role_stats):
                    if key.startswith("semantic_"):
                        role_stats.pop(key)
            finalized["layers"][role] = role_stats
        return finalized

    def _display_results(
        self,
        dataset_name: str,
        query_id: str,
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        correct: int,
        total: int,
        attr_stats: Dict[str, Dict[str, int]],
    ) -> None:
        title = f"Dataset: {dataset_name} | Query ID: {query_id}"
        self.display_metrics(title, tp, fp, fn, tn, correct, total)
        self._display_attribute_accuracy(attr_stats)

    def _display_attribute_accuracy(self, attr_stats: Dict[str, Dict[str, int]], width: int = 80) -> None:
        if not attr_stats:
            logging.warning("[%s:_display_attribute_accuracy] No attribute statistics to display", self.__class__.__name__)
            return

        table_groups: Dict[str, Dict[str, Dict[str, int]]] = {}
        for attr_key, stats in attr_stats.items():
            table_name = stats.get("table", "unknown")
            table_groups.setdefault(table_name, {})[attr_key] = stats

        total_correct = sum(stats["correct"] for stats in attr_stats.values())
        total_attrs = sum(stats["total"] for stats in attr_stats.values())
        overall_accuracy = total_correct / total_attrs if total_attrs > 0 else 0.0

        print("\n" + "=" * width)
        print("Attribute-Level Accuracy")
        print("=" * width)

        for table_name in sorted(table_groups.keys()):
            table_attrs = table_groups[table_name]
            table_correct = sum(stats["correct"] for stats in table_attrs.values())
            table_total = sum(stats["total"] for stats in table_attrs.values())
            table_accuracy = table_correct / table_total if table_total > 0 else 0.0

            print(f"\nTable: {table_name}")
            print("-" * width)
            attr_col_width = width - 32
            print(f"{'Attribute':<{attr_col_width}}{'Correct':>10}{'Total':>10}{'Accuracy':>10}")
            print("-" * width)

            for attr_key in sorted(table_attrs.keys(), key=lambda key: table_attrs[key].get("attr", "")):
                stats = table_attrs[attr_key]
                attr_name = stats.get("attr", attr_key)
                accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                print(f"{attr_name:<{attr_col_width}}{stats['correct']:>10}{stats['total']:>10}{accuracy:>10.2%}")

            print("-" * width)
            subtotal_label = f"Subtotal ({table_name})"
            print(f"{subtotal_label:<{attr_col_width}}{table_correct:>10}{table_total:>10}{table_accuracy:>10.2%}")

        print("=" * width)
        overall_label = "Overall (All Tables & Attributes)"
        attr_col_width = width - 32
        print(f"{overall_label:<{attr_col_width}}{total_correct:>10}{total_attrs:>10}{overall_accuracy:>10.2%}")
        print("=" * width)

    def _display_query_aware_results(self, result: QueryAwareEvaluation, width: int = 80) -> None:
        payload = result.to_dict()
        summary = payload["summary"]
        table = payload["table_assignment"]
        cells = payload["cell_recall"]
        semantic_cells = payload.get("semantic_cell_accuracy")
        answer = payload["answer_recall"]

        print("\n" + "=" * width)
        print("Query-Aware Recall")
        print("-" * width)
        print(f"{'Can answer query':<24}{summary['can_answer_query']}")
        print(f"{'Required tables':<24}{', '.join(summary['query_required_tables'])}")
        print(f"{'Relevant docs':<24}{summary['relevant_docs']}/{summary['evaluated_docs']}")
        print(
            f"{'Table assignment':<24}"
            f"{table['covered']}/{table['total']} = {table['recall']:.2%}"
        )
        print(
            f"{'Required cells':<24}"
            f"{cells['covered']}/{cells['total']} = {cells['recall']:.2%} "
            f"(missing={cells['missing']}, mismatched={cells['mismatched']})"
        )
        if semantic_cells is not None:
            print(
                f"{'Query semantic cells':<24}"
                f"{semantic_cells['correct']}/{semantic_cells['total']} = {semantic_cells['accuracy']:.2%} "
                f"(missing={semantic_cells['missing']}, mismatched={semantic_cells['mismatched']}, "
                f"table_mismatched={semantic_cells['table_mismatched']}, llm={semantic_cells['llm_judged']})"
            )
        if answer.get("executable"):
            print(
                f"{'SQL answer recall':<24}"
                f"{answer['covered']}/{answer['total']} = {answer['recall']:.2%} "
                f"(precision={answer['precision']:.2%})"
            )
        else:
            print(f"{'SQL answer recall':<24}not executable: {answer.get('reason')}")
        print("=" * width)

    def _display_optimization_summary(self, summary: Dict[str, Any], width: int = 80) -> None:
        if not summary.get("has_metrics"):
            return
        llm_usage = summary.get("llm_usage", {})
        doc_call = summary.get("doc_call_optimization", {})
        token_opt = summary.get("token_optimization", {})

        print("\n" + "=" * width)
        print("Optimization Summary")
        print("-" * width)
        print(
            f"{'Actual LLM calls':<24}"
            f"{llm_usage.get('calls', 0)} "
            f"(tokens={llm_usage.get('total_tokens', 0)})"
        )
        if doc_call.get("calls_before") is not None:
            print(
                f"{'Doc-call savings':<24}"
                f"{doc_call.get('calls_saved', 0)}/{doc_call.get('calls_before', 0)} = "
                f"{(doc_call.get('call_reduction') or 0):.2%}"
            )
        if token_opt.get("estimated_tokens_saved") is not None:
            print(
                f"{'Estimated tokens saved':<24}"
                f"{token_opt['estimated_tokens_saved']} "
                f"({(token_opt.get('estimated_token_reduction') or 0):.2%})"
            )
        print("=" * width)

    def _collect_query_optimization_summary(self, query_id: str) -> Dict[str, Any]:
        llm_usage = self._collect_query_llm_usage(query_id)
        doc_call_optimization = self._collect_query_doc_call_optimization(query_id)
        token_optimization = self._estimate_query_token_optimization(
            llm_usage=llm_usage,
            doc_call_optimization=doc_call_optimization,
        )
        enabled_optimizations = self._describe_enabled_query_optimizations(
            llm_usage=llm_usage,
            doc_call_optimization=doc_call_optimization,
            token_optimization=token_optimization,
        )
        has_metrics = bool(
            llm_usage.get("calls")
            or doc_call_optimization.get("by_stage")
        )
        return {
            "scope": "query",
            "query_id": query_id,
            "has_metrics": has_metrics,
            "enabled_optimizations": enabled_optimizations,
            "llm_usage": llm_usage,
            "doc_call_optimization": doc_call_optimization,
            "token_optimization": token_optimization,
            "notes": [
                "llm_usage is actual provider/runtime accounting when REDD_LLM_USAGE_LOG is available.",
                "doc_call_optimization is derived from optimization artifacts and uses document-call equivalents.",
                "token_optimization is estimated from observed tokens per call when exact skipped-call tokens are unavailable.",
            ],
        }

    @staticmethod
    def _optimization_description(stage_id: str) -> Dict[str, str]:
        descriptions = {
            "doc_filter": {
                "name": "Document Filter",
                "optimized_part": "Phase 0: query-specific document filtering before extraction",
                "optimized_what": (
                    "Reduces documents that continue into expensive table assignment and "
                    "attribute/oracle extraction."
                ),
                "usage_stage_basis": "",
            },
            "table_assignment_cache": {
                "name": "Table Assignment Cache",
                "optimized_part": "Phase 1: table assignment",
                "optimized_what": (
                    "Reuses cached or metadata-derived table assignments so fewer documents "
                    "need table-assignment LLM calls."
                ),
                "usage_stage_basis": "table_assignment",
            },
            "schema_adaptive": {
                "name": "Schema Adaptive Sampling",
                "optimized_part": "Schema refinement / schema sampling",
                "optimized_what": (
                    "Stops schema-related document processing once the adaptive sampler has "
                    "enough evidence."
                ),
                "usage_stage_basis": "schema_refinement",
            },
            "proxy_runtime": {
                "name": "Proxy Runtime",
                "optimized_part": "Phase 2: predicate proxy before oracle/attribute extraction",
                "optimized_what": (
                    "Filters documents with lightweight proxies so fewer documents reach the "
                    "expensive oracle/LLM extraction path."
                ),
                "usage_stage_basis": "proxy_runtime_oracle",
            },
        }
        return descriptions.get(
            stage_id,
            {
                "name": stage_id,
                "optimized_part": stage_id,
                "optimized_what": "Reduces document-call work in this stage.",
                "usage_stage_basis": stage_id,
            },
        )

    def _describe_enabled_query_optimizations(
        self,
        *,
        llm_usage: Dict[str, Any],
        doc_call_optimization: Dict[str, Any],
        token_optimization: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        stages = doc_call_optimization.get("by_stage")
        if not isinstance(stages, dict):
            return []
        token_by_stage = token_optimization.get("by_stage")
        if not isinstance(token_by_stage, dict):
            token_by_stage = {}
        usage_by_stage = llm_usage.get("by_stage")
        if not isinstance(usage_by_stage, dict):
            usage_by_stage = {}

        order = ["doc_filter", "table_assignment_cache", "schema_adaptive", "proxy_runtime"]
        stage_ids = [stage_id for stage_id in order if stage_id in stages]
        stage_ids.extend(sorted(stage_id for stage_id in stages if stage_id not in set(stage_ids)))
        enabled: List[Dict[str, Any]] = []
        for index, stage_id in enumerate(stage_ids, start=1):
            stage = stages.get(stage_id)
            if not isinstance(stage, dict):
                continue
            desc = self._optimization_description(stage_id)
            token_stage = token_by_stage.get(stage_id)
            if not isinstance(token_stage, dict):
                token_stage = {}
            usage_stage_id = str(token_stage.get("usage_stage_basis") or desc.get("usage_stage_basis") or "")
            actual_usage = usage_by_stage.get(usage_stage_id) if usage_stage_id else None
            if not isinstance(actual_usage, dict):
                actual_usage = {}
            details = stage.get("details") if isinstance(stage.get("details"), list) else []
            artifacts = sorted(
                {
                    str(detail.get("artifact"))
                    for detail in details
                    if isinstance(detail, dict) and detail.get("artifact")
                }
            )
            calls_before = self._metric_int(stage.get("calls_before")) or 0
            calls_after = self._metric_int(stage.get("calls_after")) or 0
            calls_saved = self._metric_int(stage.get("calls_saved")) or 0
            reduction = self._metric_float(stage.get("call_reduction"))
            token_saved = self._metric_int(token_stage.get("estimated_tokens_saved"))
            summary = (
                f"{desc['name']} optimized {desc['optimized_part']}; "
                f"saved {calls_saved}/{calls_before} document-call equivalents"
            )
            if reduction is not None:
                summary += f" ({reduction:.2%})"
            if token_saved is not None:
                summary += f", estimated {token_saved} tokens saved"
            summary += "."
            enabled.append(
                {
                    "index": index,
                    "id": stage_id,
                    "name": desc["name"],
                    "enabled": True,
                    "optimized_part": desc["optimized_part"],
                    "optimized_what": desc["optimized_what"],
                    "summary": summary,
                    "doc_call_savings": {
                        "unit": stage.get("unit", "doc_call"),
                        "calls_before": calls_before,
                        "calls_after": calls_after,
                        "calls_saved": calls_saved,
                        "call_reduction": reduction,
                    },
                    "token_savings": {
                        "status": token_optimization.get("status"),
                        "estimated_tokens_saved": token_saved,
                        "tokens_per_call_basis": token_stage.get("tokens_per_call_basis"),
                        "usage_stage_basis": usage_stage_id or None,
                    },
                    "actual_llm_usage_after": {
                        "calls": actual_usage.get("calls", 0),
                        "prompt_tokens": actual_usage.get("prompt_tokens", 0),
                        "completion_tokens": actual_usage.get("completion_tokens", 0),
                        "total_tokens": actual_usage.get("total_tokens", 0),
                    },
                    "evidence_artifacts": artifacts,
                    "details": details,
                }
            )
        return enabled

    @staticmethod
    def _empty_llm_usage_totals() -> Dict[str, Any]:
        return {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "entries_with_token_usage": 0,
            "tokens_missing_calls": 0,
        }

    @staticmethod
    def _metric_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _metric_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _add_llm_usage(self, totals: Dict[str, Any], item: Dict[str, Any]) -> None:
        usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
        totals["calls"] += 1
        has_token_usage = False
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = self._metric_int(usage.get(key))
            if value is not None:
                has_token_usage = True
                totals[key] += value
        if has_token_usage:
            totals["entries_with_token_usage"] += 1
        else:
            totals["tokens_missing_calls"] += 1

    @staticmethod
    def _usage_context(item: Dict[str, Any]) -> Dict[str, Any]:
        context = item.get("context")
        if not isinstance(context, dict):
            context = {}
        result = dict(context)
        for key in ("query_id", "stage"):
            if key not in result and key in item:
                result[key] = item.get(key)
        return result

    def _collect_query_llm_usage(self, query_id: str) -> Dict[str, Any]:
        totals = self._empty_llm_usage_totals()
        unattributed = self._empty_llm_usage_totals()
        by_stage: Dict[str, Dict[str, Any]] = {}
        unattributed_by_stage: Dict[str, Dict[str, Any]] = {}
        source_files: set[str] = set()
        matching_source_files: set[str] = set()

        for path in self._llm_usage_log_candidates():
            if not path.exists() or not path.is_file():
                continue
            source_files.add(str(path))
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                context = self._usage_context(item)
                stage = str(context.get("stage") or "unattributed")
                item_query_id = context.get("query_id")
                if item_query_id is not None and str(item_query_id) == str(query_id):
                    matching_source_files.add(str(path))
                    self._add_llm_usage(totals, item)
                    stage_totals = by_stage.setdefault(stage, self._empty_llm_usage_totals())
                    self._add_llm_usage(stage_totals, item)
                elif item_query_id in (None, ""):
                    self._add_llm_usage(unattributed, item)
                    stage_totals = unattributed_by_stage.setdefault(
                        stage,
                        self._empty_llm_usage_totals(),
                    )
                    self._add_llm_usage(stage_totals, item)

        return {
            **totals,
            "source_files": sorted(matching_source_files),
            "candidate_source_files": sorted(source_files),
            "by_stage": by_stage,
            "unattributed": {
                **unattributed,
                "by_stage": unattributed_by_stage,
            },
            "token_accounting_complete": (
                totals["calls"] > 0 and totals["tokens_missing_calls"] == 0
            ),
        }

    def _llm_usage_log_candidates(self) -> List[Path]:
        candidates: List[Path] = []

        def add(value: Any) -> None:
            if not value:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    add(item)
                return
            try:
                candidates.append(Path(str(value)).expanduser())
            except TypeError:
                return

        add(os.getenv("REDD_LLM_USAGE_LOG"))
        for container in (
            self.config,
            self.config.get("eval", {}) if isinstance(self.config.get("eval"), dict) else {},
            self.config.get("runtime", {}) if isinstance(self.config.get("runtime"), dict) else {},
        ):
            if not isinstance(container, dict):
                continue
            for key in ("llm_usage_log", "usage_log", "llm_usage_logs", "usage_logs"):
                add(container.get(key))

        roots = [
            self.out_root,
            self.out_root / "reports",
            self.out_root.parent,
            self.out_root.parent / "reports",
            self.out_root.parent.parent,
            self.out_root.parent.parent / "reports",
        ]
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for pattern in ("*llm_usage*.jsonl", "usage*.jsonl"):
                candidates.extend(root.glob(pattern))

        unique: Dict[str, Path] = {}
        for path in candidates:
            try:
                key = str(path.resolve())
            except OSError:
                key = str(path)
            unique[key] = path
        return list(unique.values())

    @staticmethod
    def _safe_read_json(path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _query_id_from_filter_name(name: str) -> Optional[str]:
        match = re.match(r"(?:doc|chunk)_filter_(?P<query>.+?)(?:_.+)?\.json$", name)
        return match.group("query") if match else None

    def _query_id_from_result_name(self, name: str) -> Optional[str]:
        prefix = "res_tabular_data_"
        suffix = f"_{self.res_param_str}.json"
        if not name.startswith(prefix) or not name.endswith(suffix):
            return None
        return name[len(prefix) : -len(suffix)]

    def _stage_call_summary(
        self,
        *,
        stage_id: str,
        title: str,
        before: int,
        after: int,
        details: List[Dict[str, Any]],
        unit: str = "doc_call",
    ) -> Dict[str, Any]:
        before = max(int(before), 0)
        after = max(int(after), 0)
        saved = max(before - after, 0)
        return {
            "id": stage_id,
            "title": title,
            "unit": unit,
            "calls_before": before,
            "calls_after": after,
            "calls_saved": saved,
            "call_reduction": saved / before if before else None,
            "llm_doc_calls_before": before,
            "llm_doc_calls_after": after,
            "llm_doc_calls_saved": saved,
            "llm_doc_call_reduction": saved / before if before else None,
            "details": details,
        }

    def _collect_query_doc_call_optimization(self, query_id: str) -> Dict[str, Any]:
        stages: Dict[str, Dict[str, Any]] = {}
        for stage in (
            self._doc_filter_doc_call_stage(query_id),
            self._table_assignment_cache_doc_call_stage(query_id),
            self._schema_adaptive_doc_call_stage(query_id),
            self._proxy_runtime_doc_call_stage(query_id),
        ):
            if stage is not None:
                stages[stage["id"]] = stage

        calls_before = sum(stage["calls_before"] for stage in stages.values())
        calls_after = sum(stage["calls_after"] for stage in stages.values())
        calls_saved = sum(stage["calls_saved"] for stage in stages.values())
        return {
            "status": "measured" if stages else "no_metrics",
            "unit": "doc_call",
            "calls_before": calls_before if stages else None,
            "calls_after": calls_after if stages else None,
            "calls_saved": calls_saved if stages else None,
            "call_reduction": calls_saved / calls_before if calls_before else None,
            "by_stage": stages,
        }

    def _doc_filter_doc_call_stage(self, query_id: str) -> Optional[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        before = after = 0
        for path in sorted((self.out_root / "doc_filter").glob("*.json")):
            content = self._safe_read_json(path)
            path_query_id = str(content.get("query_id") or self._query_id_from_filter_name(path.name) or "")
            if path_query_id != str(query_id):
                continue
            metadata = content.get("metadata") if isinstance(content.get("metadata"), dict) else {}
            input_docs = self._metric_int(metadata.get("num_docs_input"))
            kept_docs = self._metric_int(metadata.get("num_docs_kept"))
            excluded_docs = self._metric_int(metadata.get("num_docs_excluded"))
            if kept_docs is None:
                kept_docs = len(content.get("kept_doc_ids") or [])
            if excluded_docs is None:
                excluded_docs = len(content.get("excluded_doc_ids") or [])
            if input_docs is None:
                input_docs = kept_docs + excluded_docs
            before += input_docs
            after += kept_docs
            details.append(
                {
                    "artifact": str(path),
                    "query_id": path_query_id,
                    "input_docs": input_docs,
                    "kept_docs": kept_docs,
                    "excluded_docs": excluded_docs,
                }
            )
        if not details:
            return None
        return self._stage_call_summary(
            stage_id="doc_filter",
            title="Document Filter",
            before=before,
            after=after,
            details=details,
        )

    def _table_assignment_cache_doc_call_stage(self, query_id: str) -> Optional[Dict[str, Any]]:
        path = self.out_root / "table_assignment_cache.json"
        content = self._safe_read_json(path)
        events = content.get("events")
        if not isinstance(events, list):
            return None
        before = after = 0
        details: List[Dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict) or str(event.get("query_id") or "") != str(query_id):
                continue
            input_docs = self._metric_int(event.get("input_docs")) or 0
            excluded = self._metric_int(event.get("excluded")) or 0
            cache_misses = self._metric_int(event.get("cache_misses")) or 0
            calls_before = max(input_docs - excluded, 0)
            calls_after = cache_misses
            before += calls_before
            after += calls_after
            details.append(
                {
                    "artifact": str(path),
                    "query_id": str(query_id),
                    "input_docs": input_docs,
                    "excluded": excluded,
                    "cache_hits": self._metric_int(event.get("cache_hits")) or 0,
                    "cache_misses": cache_misses,
                    "source_table_metadata_hits": self._metric_int(
                        event.get("source_table_metadata_hits")
                    )
                    or 0,
                    "source_table_metadata_misses": self._metric_int(
                        event.get("source_table_metadata_misses")
                    )
                    or 0,
                }
            )
        if not details:
            return None
        return self._stage_call_summary(
            stage_id="table_assignment_cache",
            title="Table Assignment Cache",
            before=before,
            after=after,
            details=details,
        )

    def _schema_adaptive_doc_call_stage(self, query_id: str) -> Optional[Dict[str, Any]]:
        before = after = 0
        details: List[Dict[str, Any]] = []
        for path in sorted(self.out_root.rglob("*_adaptive_stats.json")):
            path_query_id = self._query_id_from_adaptive_stats_name(path.name)
            if path_query_id is not None and str(path_query_id) != str(query_id):
                continue
            content = self._safe_read_json(path)
            total = self._metric_int(content.get("filtered_documents"))
            if total is None:
                total = self._metric_int(content.get("total_documents")) or 0
            processed = self._metric_int(content.get("documents_processed"))
            if processed is None:
                processed = self._metric_int(content.get("n_processed")) or 0
            saved = self._metric_int(content.get("documents_saved"))
            if saved is not None and total is None:
                total = processed + saved
            before += total or 0
            after += processed or 0
            details.append(
                {
                    "artifact": str(path),
                    "query_id": path_query_id or query_id,
                    "documents_before": total or 0,
                    "documents_processed": processed or 0,
                    "documents_saved": max((total or 0) - (processed or 0), 0),
                }
            )
        if not details:
            return None
        return self._stage_call_summary(
            stage_id="schema_adaptive",
            title="Schema Adaptive Sampling",
            before=before,
            after=after,
            details=details,
        )

    @staticmethod
    def _query_id_from_adaptive_stats_name(name: str) -> Optional[str]:
        match = re.match(r".*?(?P<query>q\d+|Q\d+).*_adaptive_stats\.json$", name)
        return match.group("query") if match else None

    def _proxy_runtime_doc_call_stage(self, query_id: str) -> Optional[Dict[str, Any]]:
        expected = self.out_root / (
            Path(PATH_TEMPLATES.data_extraction_result(query_id, self.res_param_str)).stem
            + "_proxy_decisions.json"
        )
        files = [expected] if expected.exists() else []
        if not files:
            files = [
                path
                for path in self.out_root.rglob("*_proxy_decisions.json")
                if self._query_id_from_proxy_decision_name(path.name) == str(query_id)
            ]
        before = after = 0
        details: List[Dict[str, Any]] = []
        for path in sorted(set(files)):
            content = self._safe_read_json(path)
            for table_name, table_decision in content.items():
                if not isinstance(table_decision, dict):
                    continue
                stage_before, stage_after = self._proxy_llm_doc_call_counts(table_decision)
                before += stage_before
                after += stage_after
                details.append(
                    {
                        "artifact": str(path),
                        "query_id": query_id,
                        "table": table_name,
                        "llm_doc_calls_before": stage_before,
                        "llm_doc_calls_after": stage_after,
                        "llm_doc_calls_saved": max(stage_before - stage_after, 0),
                    }
                )
        if not details:
            return None
        return self._stage_call_summary(
            stage_id="proxy_runtime",
            title="Proxy Runtime",
            before=before,
            after=after,
            details=details,
        )

    def _query_id_from_proxy_decision_name(self, name: str) -> Optional[str]:
        prefix = "res_tabular_data_"
        suffix = f"_{self.res_param_str}_proxy_decisions.json"
        if not name.startswith(prefix) or not name.endswith(suffix):
            return None
        return name[len(prefix) : -len(suffix)]

    def _proxy_llm_doc_call_counts(self, table_decision: Dict[str, Any]) -> Tuple[int, int]:
        all_doc_ids = table_decision.get("all_doc_ids")
        passed_doc_ids = table_decision.get("passed_doc_ids")
        extracted_doc_ids = table_decision.get("extracted_doc_ids")
        if isinstance(all_doc_ids, list) and isinstance(passed_doc_ids, list):
            after_ids = extracted_doc_ids if isinstance(extracted_doc_ids, list) else passed_doc_ids
            return len(set(map(str, all_doc_ids))), len(set(map(str, after_ids)))

        proxy_stats = table_decision.get("proxy_stats")
        if not isinstance(proxy_stats, dict):
            return 0, 0
        evaluated_counts: List[int] = []
        passed_counts: List[int] = []
        for stat in proxy_stats.values():
            if not isinstance(stat, dict):
                continue
            evaluated = self._metric_int(stat.get("evaluated"))
            passed = self._metric_int(stat.get("passed"))
            if evaluated is not None:
                evaluated_counts.append(evaluated)
            if passed is not None:
                passed_counts.append(passed)
        before = max(evaluated_counts) if evaluated_counts else 0
        after = min(passed_counts) if passed_counts else before
        return before, after

    def _estimate_query_token_optimization(
        self,
        *,
        llm_usage: Dict[str, Any],
        doc_call_optimization: Dict[str, Any],
    ) -> Dict[str, Any]:
        observed_tokens = int(llm_usage.get("total_tokens") or 0)
        observed_calls = int(llm_usage.get("calls") or 0)
        calls_saved = doc_call_optimization.get("calls_saved")
        if not observed_tokens or not observed_calls or not calls_saved:
            return {
                "status": "unavailable",
                "reason": "Token savings require token-bearing usage logs and doc-call savings.",
                "observed_tokens_after": observed_tokens,
                "estimated_tokens_before": None,
                "estimated_tokens_saved": None,
                "estimated_token_reduction": None,
                "by_stage": {},
            }

        overall_avg = observed_tokens / observed_calls
        by_stage_usage = llm_usage.get("by_stage", {})
        stage_usage_map = {
            "table_assignment_cache": "table_assignment",
            "proxy_runtime": "proxy_runtime_oracle",
            "schema_adaptive": "schema_refinement",
            "doc_filter": None,
        }
        estimated_by_stage: Dict[str, Dict[str, Any]] = {}
        estimated_saved_total = 0.0
        for stage_id, stage in (doc_call_optimization.get("by_stage") or {}).items():
            saved = int(stage.get("calls_saved") or 0)
            usage_stage_id = stage_usage_map.get(stage_id, stage_id)
            stage_avg = overall_avg
            if usage_stage_id and isinstance(by_stage_usage.get(usage_stage_id), dict):
                usage_stage = by_stage_usage[usage_stage_id]
                stage_calls = int(usage_stage.get("calls") or 0)
                stage_tokens = int(usage_stage.get("total_tokens") or 0)
                if stage_calls and stage_tokens:
                    stage_avg = stage_tokens / stage_calls
            estimated_saved = saved * stage_avg
            estimated_saved_total += estimated_saved
            estimated_by_stage[stage_id] = {
                "calls_saved": saved,
                "estimated_tokens_saved": round(estimated_saved),
                "tokens_per_call_basis": stage_avg,
                "usage_stage_basis": usage_stage_id or "overall_query_average",
            }

        estimated_before = observed_tokens + estimated_saved_total
        return {
            "status": "estimated",
            "estimation_method": "observed_tokens_per_call_times_doc_calls_saved",
            "observed_tokens_after": observed_tokens,
            "estimated_tokens_before": round(estimated_before),
            "estimated_tokens_saved": round(estimated_saved_total),
            "estimated_token_reduction": (
                estimated_saved_total / estimated_before if estimated_before else None
            ),
            "by_stage": estimated_by_stage,
        }

    def compute_statistics(self) -> Optional[Tuple[int, int, int, int, int, int, Dict[str, Any], Dict[str, Dict[str, int]]]]:
        if not self._validate_data():
            return None

        metrics = EvaluationMetrics()
        for pred, gt in zip(self.prediction_data, self.gt_data):
            doc_id = pred["doc_id"]
            metrics.total_count += 1
            metrics.doc_stats[doc_id] = {"table": True, "attr": {}, "final": False}
            is_doc_correct = self._evaluate_document(pred, gt, metrics)
            if is_doc_correct:
                metrics.correct_count += 1
            metrics.doc_stats[doc_id]["final"] = is_doc_correct

        return metrics.to_tuple()

    def _evaluate_document(
        self,
        pred: Dict[str, Any],
        gt: Dict[str, Any],
        metrics: EvaluationMetrics,
    ) -> bool:
        if is_null(gt["table"]):
            return self._evaluate_irrelevant_document(pred, metrics)
        return self._evaluate_relevant_document(pred, gt, metrics)

    def _evaluate_irrelevant_document(
        self,
        pred: Dict[str, Any],
        metrics: EvaluationMetrics,
    ) -> bool:
        doc_id = pred["doc_id"]
        if not is_null(pred["table"]):
            non_null_attrs = [attr for attr in pred["data"] if not is_null(pred["data"][attr])]
            metrics.false_positives += len(non_null_attrs)
            metrics.doc_stats[doc_id]["table"] = False
            logging.info(
                "[%s:_evaluate_irrelevant_document] false_positives (doc irrelevant): %s",
                self.__class__.__name__,
                pred,
            )
            return True

        metrics.true_negatives += 1
        return True

    def _evaluate_relevant_document(
        self,
        pred: Dict[str, Any],
        gt: Dict[str, Any],
        metrics: EvaluationMetrics,
    ) -> bool:
        doc_id = pred["doc_id"]
        gt_table = gt["table"]
        attr_map_dict = self.name_map.get("attribute", {}).get(gt_table, {})
        table_correct = self._check_table_assignment(pred, gt, metrics)
        if not table_correct:
            metrics.doc_stats[doc_id]["table"] = False

        attr_correct = self._evaluate_attributes(pred, gt, gt_table, attr_map_dict, metrics)
        return table_correct and attr_correct

    def _check_table_assignment(
        self,
        pred: Dict[str, Any],
        gt: Dict[str, Any],
        metrics: EvaluationMetrics,
    ) -> bool:
        if pred["table"] == gt["table"]:
            return True

        if is_null(pred["table"]):
            metrics.false_negatives += 1
        else:
            metrics.false_positives += 1
        return False

    def _evaluate_attributes(
        self,
        pred: Dict[str, Any],
        gt: Dict[str, Any],
        gt_table: str,
        attr_map_dict: Dict[str, str],
        metrics: EvaluationMetrics,
    ) -> bool:
        doc_id = pred["doc_id"]
        all_attrs_correct = True

        for gt_attr, gt_val in gt["data"].items():
            pred_attr = attr_map_dict.get(gt_attr, gt_attr)
            pred_val = pred["data"].get(pred_attr)
            attr_key = f"{gt_table}.{gt_attr}"

            if attr_key not in metrics.attr_stats:
                metrics.attr_stats[attr_key] = {"correct": 0, "total": 0, "table": gt_table, "attr": gt_attr}
            metrics.attr_stats[attr_key]["total"] += 1

            attr_correct = self._compare_attribute_values(pred_val, gt_val)
            metrics.doc_stats[doc_id]["attr"][gt_attr] = attr_correct
            if attr_correct:
                metrics.true_positives += 1
                metrics.attr_stats[attr_key]["correct"] += 1
            else:
                all_attrs_correct = False
                if is_null(pred_val):
                    metrics.false_negatives += 1
                else:
                    metrics.false_positives += 1

        return all_attrs_correct

    def _compare_attribute_values(self, pred_val: Any, gt_val: Any) -> bool:
        if is_null(pred_val) and is_null(gt_val):
            return True
        if is_null(pred_val) or is_null(gt_val):
            return False
        pred_number = self._decimal_value(pred_val)
        gt_number = self._decimal_value(gt_val)
        if pred_number is not None and gt_number is not None:
            return pred_number == gt_number
        return self._normalize_text_value(pred_val) == self._normalize_text_value(gt_val)

    def _semantic_eval_enabled(self) -> bool:
        return bool(self.committee_prompts or self.prompts)

    def _semantic_context_summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.semantic_context_enabled,
            "include_schema": self.semantic_context_include_schema,
            "include_query": self.semantic_context_include_query,
            "include_cell_role": self.semantic_context_include_cell_role,
            "include_doc_text": self.semantic_context_include_doc_text,
            "doc_text_max_chars": self.semantic_context_doc_text_max_chars,
        }

    def _semantic_cell_context(
        self,
        *,
        loader: DataLoaderBase,
        query_id: str,
        query_info: Dict[str, Any],
        doc_id: str,
        table: str,
        attr: str,
        cell_role: str,
        pred_value: Any = None,
        gt_value: Any = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.semantic_context_enabled:
            return None

        context: Dict[str, Any] = {}
        if self.semantic_context_include_cell_role:
            context["Cell Role"] = cell_role
        if self.semantic_context_include_schema:
            schema_context = self._semantic_schema_context(loader, query_id, table, attr)
            if schema_context:
                context["Schema"] = schema_context
        if self.semantic_context_include_query:
            query_context = self._semantic_query_context(query_id, query_info)
            if query_context:
                context["Query"] = query_context
        if self._semantic_doc_text_enabled():
            doc_context = self._semantic_doc_context(
                loader=loader,
                doc_id=doc_id,
                attr=attr,
                pred_value=pred_value,
                gt_value=gt_value,
            )
            if doc_context:
                context["Document"] = doc_context

        return self._semantic_cache_context(context)

    def _semantic_schema_context(
        self,
        loader: DataLoaderBase,
        query_id: str,
        table: str,
        attr: str,
    ) -> Optional[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        try:
            schemas = loader.load_schema_query(query_id)
        except Exception as exc:
            logging.debug(
                "[%s:_semantic_schema_context] Could not load query schema for %s: %s",
                self.__class__.__name__,
                query_id,
                exc,
            )
        if not schemas:
            try:
                schemas = loader.load_schema_general()
            except Exception as exc:
                logging.debug(
                    "[%s:_semantic_schema_context] Could not load general schema: %s",
                    self.__class__.__name__,
                    exc,
                )

        for schema in schemas or []:
            if not isinstance(schema, dict):
                continue
            schema_name = str(
                schema.get(SCHEMA_NAME_KEY)
                or schema.get("table_name")
                or schema.get("name")
                or ""
            )
            if schema_name != table:
                continue

            schema_context: Dict[str, Any] = {
                "Table Name": schema_name,
                "Table Description": schema.get("Description") or schema.get("description"),
            }
            for attr_info in schema.get(ATTRIBUTES_KEY) or schema.get("columns") or []:
                if not isinstance(attr_info, dict):
                    continue
                attr_name = str(
                    attr_info.get(ATTRIBUTE_NAME_KEY)
                    or attr_info.get("name")
                    or attr_info.get("column_id")
                    or ""
                )
                column_id = str(attr_info.get("column_id") or "")
                if attr_name != attr and column_id != f"{table}.{attr}" and not column_id.endswith(f".{attr}"):
                    continue
                schema_context["Attribute"] = {
                    "Attribute Name": attr_name if attr_name else attr,
                    "Attribute Description": attr_info.get("Description") or attr_info.get("description"),
                    "Attribute Type": attr_info.get("type") or attr_info.get("Type"),
                    "Column ID": column_id,
                }
                break
            return self._semantic_cache_context(schema_context)
        return None

    @staticmethod
    def _semantic_query_context(query_id: str, query_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(query_info, dict):
            return {"Query ID": str(query_id)}
        context: Dict[str, Any] = {"Query ID": str(query_id)}
        question = (
            query_info.get("question")
            or query_info.get("query")
            or query_info.get("natural_language")
        )
        if question:
            context["Question"] = str(question)
        if query_info.get("sql"):
            context["SQL"] = str(query_info.get("sql"))
        for source_key, context_key in (
            ("tables", "Required Tables"),
            ("attributes", "Required Attributes"),
            ("required_columns", "Required Attributes"),
            ("output_columns", "Output Columns"),
        ):
            values = query_info.get(source_key)
            if values and context_key not in context:
                context[context_key] = [str(value) for value in values]
        return EvalDataExtraction._semantic_cache_context(context)

    def _semantic_doc_context(
        self,
        *,
        loader: DataLoaderBase,
        doc_id: str,
        attr: str,
        pred_value: Any,
        gt_value: Any,
    ) -> Optional[Dict[str, Any]]:
        doc_text = ""
        try:
            info = loader.get_doc_info(doc_id)
            if isinstance(info, dict):
                doc_text = str(info.get("doc") or "")
        except Exception as exc:
            logging.debug(
                "[%s:_semantic_doc_context] Could not load doc_info for %s: %s",
                self.__class__.__name__,
                doc_id,
                exc,
            )
        if not doc_text:
            try:
                doc_text = str(loader.get_doc(doc_id)[0] or "")
            except Exception as exc:
                logging.debug(
                    "[%s:_semantic_doc_context] Could not load doc text for %s: %s",
                    self.__class__.__name__,
                    doc_id,
                    exc,
                )
        if not doc_text:
            return None

        needles = [gt_value, pred_value, attr.replace("_", " "), attr]
        excerpt, truncated = self._semantic_doc_text_excerpt(doc_text, needles)
        if not excerpt:
            return None
        return {
            "Doc ID": str(doc_id),
            "Text Excerpt": excerpt,
            "Excerpt Strategy": "focused_around_value_or_attribute",
            "Excerpt Truncated": truncated,
        }

    def _semantic_doc_text_enabled(self) -> bool:
        mode = self.semantic_context_include_doc_text
        if isinstance(mode, bool):
            return mode
        return str(mode).strip().lower() in {
            "1",
            "true",
            "yes",
            "always",
            "on_mismatch",
            "mismatch",
            "focused",
            "excerpt",
        }

    def _semantic_doc_text_excerpt(self, doc_text: str, needles: List[Any]) -> Tuple[str, bool]:
        max_chars = self.semantic_context_doc_text_max_chars
        if max_chars <= 0:
            return "", False
        text = str(doc_text)
        if len(text) <= max_chars:
            return text, False

        lowered = text.casefold()
        spans: List[Tuple[int, int]] = []
        for needle in needles:
            needle_text = str(needle or "").strip()
            if len(needle_text) < 2:
                continue
            index = lowered.find(needle_text.casefold())
            if index < 0:
                continue
            half_window = max(80, max_chars // 4)
            spans.append((max(0, index - half_window), min(len(text), index + len(needle_text) + half_window)))

        if not spans:
            return text[:max_chars], True

        spans = sorted(spans)
        merged: List[Tuple[int, int]] = []
        for start, end in spans:
            if not merged or start > merged[-1][1] + 40:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        parts: List[str] = []
        used = 0
        separator = "\n...\n"
        for start, end in merged:
            remaining = max_chars - used
            if remaining <= 0:
                break
            snippet = text[start:end]
            candidate = f"{separator}{snippet}" if parts else snippet
            if len(candidate) > remaining:
                candidate = candidate[:remaining]
            parts.append(candidate)
            used += len(candidate)
        excerpt = "".join(parts).strip()
        return excerpt, True

    def compute_semantic_statistics(self) -> Dict[str, Any]:
        """Compute attribute correctness with an LLM semantic judge.

        Strict/canonical matches are accepted without an LLM call. The LLM is
        only used for non-null value pairs that fail strict comparison.
        """

        if not self.prediction_data or not self.gt_data:
            return {
                "summary": {
                    "scope": "full_table_all_gt_attributes",
                    "attr_correct": 0,
                    "attr_total": 0,
                    "attr_accuracy": None,
                    "doc_final_true": 0,
                    "doc_total": 0,
                    "doc_final_accuracy": None,
                    "llm_judged": 0,
                },
                "doc_stats": {},
                "attr_stats": {},
                "matches": [],
            }

        doc_stats: Dict[str, Dict[str, Any]] = {}
        attr_stats: Dict[str, Dict[str, Any]] = {}
        matches: List[Dict[str, Any]] = []
        attr_correct_total = 0
        attr_total = 0
        final_true = 0
        llm_judged = 0

        for pred, gt in zip(self.prediction_data, self.gt_data):
            doc_id = str(pred.get("doc_id"))
            gt_table = gt.get("table")
            pred_table = pred.get("table")
            table_correct = pred_table == gt_table
            gt_data = gt.get("data") if isinstance(gt.get("data"), dict) else {}
            pred_data = pred.get("data") if isinstance(pred.get("data"), dict) else {}
            attr_map_dict = self.name_map.get("attribute", {}).get(gt_table, {}) if isinstance(self.name_map, dict) else {}
            if not isinstance(attr_map_dict, dict):
                attr_map_dict = {}

            doc_attr_results: Dict[str, bool] = {}
            doc_all_attrs_correct = True

            if is_null(gt_table):
                doc_stats[doc_id] = {
                    "table": is_null(pred_table),
                    "attr": {},
                    "final": is_null(pred_table),
                }
                if is_null(pred_table):
                    final_true += 1
                continue

            for gt_attr, gt_value in gt_data.items():
                pred_attr = attr_map_dict.get(gt_attr, gt_attr)
                pred_value = pred_data.get(pred_attr)
                attr_key = f"{gt_table}.{gt_attr}"
                if attr_key not in attr_stats:
                    attr_stats[attr_key] = {
                        "correct": 0,
                        "total": 0,
                        "table": gt_table,
                        "attr": gt_attr,
                    }

                comparison = self._semantic_match_attribute(
                    pred_attr=str(pred_attr),
                    pred_value=pred_value,
                    gt_attr=str(gt_attr),
                    gt_value=gt_value,
                )
                attr_correct = bool(comparison["result"])
                doc_attr_results[str(gt_attr)] = attr_correct
                attr_stats[attr_key]["total"] += 1
                attr_total += 1
                if attr_correct:
                    attr_stats[attr_key]["correct"] += 1
                    attr_correct_total += 1
                else:
                    doc_all_attrs_correct = False
                if comparison.get("method") == "llm":
                    llm_judged += 1

                matches.append(
                    {
                        "doc_id": doc_id,
                        "table": gt_table,
                        "pred_table": pred_table,
                        "table_correct": table_correct,
                        "attribute": gt_attr,
                        "pred_attribute": pred_attr,
                        "gt_value": gt_value,
                        "pred_value": pred_value,
                        **comparison,
                    }
                )

            doc_final = table_correct and doc_all_attrs_correct
            if doc_final:
                final_true += 1
            doc_stats[doc_id] = {
                "table": table_correct,
                "attr": doc_attr_results,
                "final": doc_final,
            }

        doc_total = len(doc_stats)
        return {
            "summary": {
                "scope": "full_table_all_gt_attributes",
                "attr_correct": attr_correct_total,
                "attr_total": attr_total,
                "attr_accuracy": attr_correct_total / attr_total if attr_total else None,
                "doc_final_true": final_true,
                "doc_total": doc_total,
                "doc_final_accuracy": final_true / doc_total if doc_total else None,
                "llm_judged": llm_judged,
            },
            "doc_stats": doc_stats,
            "attr_stats": attr_stats,
            "matches": matches,
        }

    def _semantic_match_attribute(
        self,
        *,
        pred_attr: str,
        pred_value: Any,
        gt_attr: str,
        gt_value: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._compare_attribute_values(pred_value, gt_value):
            return {
                "result": True,
                "method": "strict",
                "reasoning": "Values match under strict numeric/text normalization.",
                "cached": False,
            }
        if is_null(pred_value) or is_null(gt_value):
            return {
                "result": False,
                "method": "null_mismatch",
                "reasoning": "One side is null or missing, so semantic comparison is not applicable.",
                "cached": False,
            }

        cache_path = self.out_root / PATH_TEMPLATES.eval_comparison_cache()
        cache = self.load_json(cache_path) if cache_path.exists() else {}
        cache_context = self._semantic_cache_context(context)
        cache_key = self._semantic_cache_key(
            pred_attr,
            pred_value,
            gt_attr,
            gt_value,
            context=cache_context,
        )
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            return {
                "result": bool(cached.get("result")),
                "method": cached.get("method", "llm"),
                "reasoning": cached.get("reasoning", ""),
                "cached": True,
                "llm_model": cached.get("llm_model"),
                "context_used": bool(cached.get("semantic_context")),
            }

        comparison = self._call_semantic_judge(
            pred_attr,
            pred_value,
            gt_attr,
            gt_value,
            context=cache_context,
        )
        if cache_context:
            comparison = {**comparison, "context_used": True}
        cache[cache_key] = {
            **comparison,
            "pred_attr": pred_attr,
            "pred_value": pred_value,
            "gt_attr": gt_attr,
            "gt_value": gt_value,
            "semantic_context": cache_context,
            "llm_model": self.eval_llm_model,
        }
        self.save_results(cache_path, cache)
        return comparison

    @staticmethod
    def _semantic_cache_context(context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        def normalize(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, dict):
                normalized_dict = {}
                for key, item in value.items():
                    normalized_value = normalize(item)
                    if normalized_value is not None:
                        normalized_dict[str(key)] = normalized_value
                return normalized_dict or None
            if isinstance(value, list):
                normalized_list = []
                for item in value:
                    normalized_value = normalize(item)
                    if normalized_value is not None:
                        normalized_list.append(normalized_value)
                return normalized_list or None
            if isinstance(value, tuple):
                return normalize(list(value))
            if isinstance(value, str):
                cleaned = re.sub(r"[ \t\r\f\v]+", " ", value).strip()
                cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
                return cleaned or None
            if isinstance(value, (bool, int, float)):
                return value
            return str(value)

        normalized = normalize(context)
        return normalized if isinstance(normalized, dict) else None

    @staticmethod
    def _semantic_cache_key(
        pred_attr: str,
        pred_value: Any,
        gt_attr: str,
        gt_value: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = {
            "pred_attr": pred_attr,
            "pred_value": "" if is_null(pred_value) else str(pred_value),
            "gt_attr": gt_attr,
            "gt_value": "" if is_null(gt_value) else str(gt_value),
        }
        if context:
            payload["semantic_context_version"] = 1
            payload["context"] = context
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _call_semantic_judge(
        self,
        pred_attr: str,
        pred_value: Any,
        gt_attr: str,
        gt_value: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cmp_input = {
            PREDICTION_KEY: {
                ATTRIBUTE_NAME_KEY: pred_attr,
                ATTRIBUTE_VALUE_KEY: str(pred_value),
            },
            GROUND_TRUTH_KEY: {
                ATTRIBUTE_NAME_KEY: gt_attr,
                ATTRIBUTE_VALUE_KEY: str(gt_value),
            },
        }
        if context:
            cmp_input["Context"] = context
        message = json.dumps(cmp_input, ensure_ascii=False)
        if self.committee_prompts:
            return self._committee_semantic_judge(message)
        return self._single_semantic_judge(message)

    def _single_semantic_judge(self, message: str) -> Dict[str, Any]:
        prompt = self._semantic_prompt()
        if prompt is None:
            return {
                "result": False,
                "method": "semantic_disabled",
                "reasoning": "No semantic comparison prompt is configured.",
                "cached": False,
            }
        for attempt in range(3):
            try:
                parsed = self._parse_semantic_response(
                    prompt(
                        msg=message,
                        max_tokens=256,
                        temperature=0,
                        usage_context={
                            "stage": "semantic_evaluation",
                            "query_id": self._current_eval_query_id,
                            "semantic_judge": "single",
                        },
                    )
                )
                if parsed is not None:
                    return {
                        "result": bool(parsed.get("Result")),
                        "method": "llm",
                        "reasoning": str(parsed.get("Reasoning") or ""),
                        "cached": False,
                        "llm_model": self.eval_llm_model,
                    }
            except Exception as exc:
                logging.warning(
                    "[%s:_single_semantic_judge] Attempt %s failed: %s",
                    self.__class__.__name__,
                    attempt + 1,
                    exc,
                )
        return {
            "result": False,
            "method": "llm_error",
            "reasoning": "Semantic judge failed after retries.",
            "cached": False,
            "llm_model": self.eval_llm_model,
        }

    def _committee_semantic_judge(self, message: str) -> Dict[str, Any]:
        votes: List[bool] = []
        results: List[Dict[str, Any]] = []
        for member in self.committee_prompts:
            prompt = self._semantic_prompt(member.get("prompts", {}))
            if prompt is None:
                continue
            try:
                parsed = self._parse_semantic_response(
                    prompt(
                        msg=message,
                        max_tokens=256,
                        temperature=0,
                        usage_context={
                            "stage": "semantic_evaluation",
                            "query_id": self._current_eval_query_id,
                            "semantic_judge": "committee",
                            "llm_model": member.get("llm_model"),
                        },
                    )
                )
                if parsed is None:
                    continue
                vote = bool(parsed.get("Result"))
                votes.append(vote)
                results.append(
                    {
                        "llm_model": member.get("llm_model"),
                        "vote": vote,
                        "reasoning": str(parsed.get("Reasoning") or ""),
                    }
                )
            except Exception as exc:
                logging.warning(
                    "[%s:_committee_semantic_judge] %s failed: %s",
                    self.__class__.__name__,
                    member.get("llm_model"),
                    exc,
                )
        true_count = sum(1 for vote in votes if vote)
        false_count = len(votes) - true_count
        result = true_count >= false_count if votes else False
        return {
            "result": result,
            "method": "committee_llm",
            "reasoning": f"Committee votes: {true_count} true, {false_count} false.",
            "cached": False,
            "votes": results,
            "llm_model": ",".join(str(member.get("llm_model")) for member in self.committee_prompts),
        }

    def _semantic_prompt(self, prompts: Optional[Dict[str, Any]] = None) -> Any | None:
        prompt_map = prompts if prompts is not None else self.prompts
        for prompt_name in ("data_extraction_cmp_str", "cmp_str"):
            if prompt_name in prompt_map:
                return prompt_map[prompt_name]
        if prompt_map:
            return next(iter(prompt_map.values()))
        return None

    @staticmethod
    def _parse_semantic_response(response: str) -> Dict[str, Any] | None:
        text = str(response or "").strip()
        candidates = [text]
        fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(match.strip() for match in fenced)
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            candidates.append(object_match.group(0))
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("Result"), bool):
                return parsed
            if isinstance(parsed, dict) and isinstance(parsed.get("result"), bool):
                return {
                    "Result": parsed["result"],
                    "Reasoning": parsed.get("reasoning") or parsed.get("Reasoning") or "",
                }
        return None

    def _save_query_aware_semantic_match_results(
        self,
        query_id: str,
        semantic_cell_accuracy: Dict[str, Any],
    ) -> Path:
        path = self.out_root / f"query_aware_semantic_matches_{query_id}_{self.res_param_str}.json"
        summary = {
            key: value
            for key, value in semantic_cell_accuracy.items()
            if key != "cells"
        }
        payload = {
            "query_id": query_id,
            "artifact": self.res_param_str,
            "scope": "query_required_answer_cells",
            "summary": summary,
            "matches": semantic_cell_accuracy.get("cells", []),
        }
        self.save_results(path, payload)
        return path

    def _save_full_table_semantic_match_results(self, query_id: str, semantic_stats: Dict[str, Any]) -> Path:
        path = self.out_root / f"full_table_semantic_matches_{query_id}_{self.res_param_str}.json"
        payload = {
            "query_id": query_id,
            "artifact": self.res_param_str,
            "scope": "full_table_all_gt_attributes",
            "summary": semantic_stats.get("summary", {}),
            "matches": semantic_stats.get("matches", []),
        }
        self.save_results(path, payload)
        return path

    @staticmethod
    def _decimal_value(value: Any) -> Decimal | None:
        text = str(value).strip()
        if not text:
            return None
        text = text.replace(",", "")
        if text.startswith("$"):
            text = text[1:]
        if text.endswith("%"):
            text = text[:-1]
        try:
            return Decimal(text)
        except (InvalidOperation, ValueError):
            return None

    @staticmethod
    def _normalize_text_value(value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value).strip())
        text = text.strip(" \t\r\n\"'")
        text = re.sub(r"[\s.。:;；，,]+$", "", text)
        return text.casefold()

    def _required_attrs_by_table(
        self,
        loader: DataLoaderBase,
        query_id: str,
        query_info: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        required_by_table: Dict[str, List[str]] = {}
        for attr_ref in query_info.get("attributes") or []:
            table, attr = self._split_attr_ref(attr_ref)
            if not table or not attr:
                continue
            required_by_table.setdefault(table, [])
            if attr not in required_by_table[table]:
                required_by_table[table].append(attr)

        for table in query_info.get("tables") or []:
            required_by_table.setdefault(str(table), [])

        if not any(required_by_table.values()):
            for schema in loader.load_schema_query(query_id):
                table = str(schema.get(SCHEMA_NAME_KEY) or "")
                if not table:
                    continue
                attrs = [
                    str(attr.get(ATTRIBUTE_NAME_KEY))
                    for attr in schema.get(ATTRIBUTES_KEY, [])
                    if isinstance(attr, dict) and attr.get(ATTRIBUTE_NAME_KEY)
                ]
                required_by_table.setdefault(table, [])
                for attr in attrs:
                    if attr not in required_by_table[table]:
                        required_by_table[table].append(attr)

        return {table: attrs for table, attrs in required_by_table.items() if table}

    @staticmethod
    def _split_attr_ref(attr_ref: Any) -> Tuple[Optional[str], Optional[str]]:
        text = str(attr_ref or "")
        if "." not in text:
            return None, text or None
        table, attr = text.split(".", 1)
        return table or None, attr or None

    def _query_required_gt_records(
        self,
        loader: DataLoaderBase,
        doc_id: str,
        required_by_table: Dict[str, List[str]],
        answer_doc_ids_by_table: Optional[Dict[str, set[Any]]] = None,
    ) -> List[Dict[str, Any]]:
        info = loader.get_doc_info(doc_id)
        if not info:
            return []
        records = info.get("data_records") or []
        if not records and info.get("table"):
            records = [{"table_name": info.get("table"), "data": info.get("data", {})}]

        required_tables = set(required_by_table)
        result = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            table = record.get("table_name") or record.get("table")
            if table not in required_tables:
                continue
            table_name = str(table)
            data = record.get("data") or {}
            if not isinstance(data, dict):
                data = {}
            row_id = (
                record.get(RESULT_RECORD_ID_KEY)
                or record.get("record_id")
                or record.get("source_row_id")
                or record.get("row_id")
                or data.get("row_id")
                or info.get("source_row_id")
            )
            if row_id is None and len(records) > 1:
                row_id = f"{doc_id}#{index}"
            if answer_doc_ids_by_table is not None:
                refs = answer_doc_ids_by_table.get(table_name, set())
                row_ref = (str(doc_id), "" if is_null(row_id) else str(row_id))
                if row_ref not in refs and str(doc_id) not in refs:
                    continue
            result.append({"table": table_name, "data": data, "row_id": row_id})
        return result

    def _answer_doc_ids_by_table(
        self,
        *,
        loader: DataLoaderBase,
        query_info: Dict[str, Any],
        eval_doc_ids: List[str],
        required_by_table: Dict[str, List[str]],
    ) -> Optional[Dict[str, set[Any]]]:
        """Return GT record refs that participate in the SQL answer rows.

        Query-aware table/cell recall should measure the rows needed to answer
        the SQL query, not every row from every table referenced by the query.
        The temporary SQL tables include ``__doc_id`` and ``row_id`` so we can
        derive this provenance for selections and joins.
        """

        sql = str(query_info.get("sql") or "").strip()
        if not sql:
            return None
        try:
            provenance_rows = self._execute_query_provenance(
                sql=sql,
                records_by_table=self._gt_records_for_sql(loader, eval_doc_ids, required_by_table),
                required_by_table=required_by_table,
            )
        except Exception as exc:
            logging.warning(
                "[%s:_answer_doc_ids_by_table] Could not compute SQL answer provenance; "
                "falling back to all required-table records: %s",
                self.__class__.__name__,
                exc,
            )
            return None

        result: Dict[str, set[Any]] = {table: set() for table in required_by_table}
        for row in provenance_rows:
            for table, refs in row.items():
                if table not in result:
                    continue
                if not isinstance(refs, list):
                    refs = [refs]
                for ref in refs:
                    if isinstance(ref, dict):
                        doc_id = ref.get("doc_id")
                        row_id = ref.get("row_id")
                    else:
                        doc_id = ref
                        row_id = None
                    if not is_null(doc_id):
                        result[table].add(
                            (
                                str(doc_id),
                                "" if is_null(row_id) else str(row_id),
                            )
                        )
        return result

    def _evaluate_answer_recall(
        self,
        *,
        loader: DataLoaderBase,
        result_dict: Dict[str, Any],
        query_info: Dict[str, Any],
        eval_doc_ids: List[str],
        required_by_table: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        sql = str(query_info.get("sql") or "").strip()
        if not sql:
            return {
                "executable": False,
                "reason": "query_has_no_sql",
                "covered": 0,
                "total": 0,
                "recall": 1.0,
                "precision": 1.0,
                "gt_rows": [],
                "pred_rows": [],
                "missing_rows": [],
                "extra_rows": [],
            }

        try:
            gt_rows = self._execute_query_over_records(
                sql=sql,
                records_by_table=self._gt_records_for_sql(loader, eval_doc_ids, required_by_table),
                required_by_table=required_by_table,
            )
            pred_rows = self._execute_query_over_records(
                sql=sql,
                records_by_table=self._prediction_records_for_sql(
                    loader,
                    result_dict,
                    eval_doc_ids,
                    required_by_table,
                ),
                required_by_table=required_by_table,
            )
        except Exception as exc:
            return {"executable": False, "reason": str(exc), "recall": None}

        gt_counter = Counter(self._canonical_row(row) for row in gt_rows)
        pred_counter = Counter(self._canonical_row(row) for row in pred_rows)
        missing_counter = gt_counter - pred_counter
        extra_counter = pred_counter - gt_counter
        covered = sum((gt_counter & pred_counter).values())
        total = sum(gt_counter.values())
        recall = covered / total if total else 1.0
        precision = covered / sum(pred_counter.values()) if pred_counter else (1.0 if not gt_counter else 0.0)

        return {
            "executable": True,
            "covered": covered,
            "total": total,
            "recall": recall,
            "precision": precision,
            "gt_rows": [list(row) for row in gt_counter.elements()],
            "pred_rows": [list(row) for row in pred_counter.elements()],
            "missing_rows": [list(row) for row in missing_counter.elements()],
            "extra_rows": [list(row) for row in extra_counter.elements()],
        }

    def _gt_records_for_sql(
        self,
        loader: DataLoaderBase,
        eval_doc_ids: List[str],
        required_by_table: Dict[str, List[str]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        records_by_table: Dict[str, List[Dict[str, Any]]] = {table: [] for table in required_by_table}
        for doc_id in eval_doc_ids:
            for record in self._query_required_gt_records(loader, doc_id, required_by_table):
                row = {attr: record["data"].get(attr) for attr in required_by_table[record["table"]]}
                row["__doc_id"] = doc_id
                row["row_id"] = record.get("row_id")
                records_by_table[record["table"]].append(row)
        return records_by_table

    def _prediction_records_for_sql(
        self,
        loader: DataLoaderBase,
        result_dict: Dict[str, Any],
        eval_doc_ids: List[str],
        required_by_table: Dict[str, List[str]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        records_by_table: Dict[str, List[Dict[str, Any]]] = {table: [] for table in required_by_table}
        for doc_id in eval_doc_ids:
            entry = result_dict.get(doc_id)
            if not isinstance(entry, dict):
                continue
            for index, record in enumerate(active_result_records(entry)):
                table = record.get("table")
                if table not in required_by_table:
                    continue
                data = record.get(RESULT_DATA_KEY, {})
                if not isinstance(data, dict):
                    data = {}
                row = {attr: data.get(attr) for attr in required_by_table[str(table)]}
                row["__doc_id"] = doc_id
                row["row_id"] = (
                    record.get(RESULT_RECORD_ID_KEY)
                    or self._doc_row_id(loader, doc_id)
                    or f"{doc_id}#{index}"
                )
                records_by_table[str(table)].append(row)
        return records_by_table

    def _execute_query_over_records(
        self,
        *,
        sql: str,
        records_by_table: Dict[str, List[Dict[str, Any]]],
        required_by_table: Dict[str, List[str]],
    ) -> List[Tuple[Any, ...]]:
        conn = sqlite3.connect(":memory:")
        try:
            self._load_query_connection(conn, records_by_table, required_by_table)
            cursor = conn.execute(sql)
            return [tuple(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def _execute_query_provenance(
        self,
        *,
        sql: str,
        records_by_table: Dict[str, List[Dict[str, Any]]],
        required_by_table: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        table_refs = self._sql_table_references(sql, set(required_by_table))
        if not table_refs:
            return []

        from_match = re.search(r"\bFROM\b", sql, re.IGNORECASE)
        if not from_match:
            return []

        selected_refs = [
            (index, ref, table)
            for index, (ref, table) in enumerate(table_refs)
            if table in required_by_table
        ]
        select_parts = []
        for index, ref, _table in selected_refs:
            select_parts.extend(
                [
                    f'"{ref}"."__doc_id" AS "__doc_id__{index}"',
                    f'"{ref}"."row_id" AS "__row_id__{index}"',
                ]
            )
        if not select_parts:
            return []

        provenance_sql = "SELECT " + ", ".join(select_parts) + " " + sql[from_match.start() :]
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            self._load_query_connection(conn, records_by_table, required_by_table)
            cursor = conn.execute(provenance_sql)
            result = []
            for row in cursor.fetchall():
                item: Dict[str, List[Dict[str, Any]]] = {}
                for index, _ref, table in selected_refs:
                    item.setdefault(table, []).append(
                        {
                            "doc_id": row[f"__doc_id__{index}"],
                            "row_id": row[f"__row_id__{index}"],
                        }
                    )
                result.append(item)
            return result
        finally:
            conn.close()

    def _load_query_connection(
        self,
        conn: sqlite3.Connection,
        records_by_table: Dict[str, List[Dict[str, Any]]],
        required_by_table: Dict[str, List[str]],
    ) -> None:
        for table, attrs in required_by_table.items():
            columns = ["__doc_id", "row_id", *(attr for attr in attrs if attr != "row_id")]
            create_sql = (
                f'CREATE TABLE "{table}" ('
                + ", ".join(
                    f'"{column}" {"TEXT" if column == "__doc_id" else "NUMERIC"}'
                    for column in columns
                )
                + ")"
            )
            conn.execute(create_sql)
            insert_sql = (
                f'INSERT INTO "{table}" ('
                + ", ".join(f'"{column}"' for column in columns)
                + ") VALUES ("
                + ", ".join("?" for _ in columns)
                + ")"
            )
            for row in records_by_table.get(table, []):
                conn.execute(insert_sql, [self._sql_value(row.get(column)) for column in columns])

    @staticmethod
    def _sql_table_references(sql: str, required_tables: set[str]) -> List[Tuple[str, str]]:
        from_match = re.search(
            r"\bFROM\s+(.+?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|;|$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if not from_match:
            return []

        refs: List[Tuple[str, str]] = []
        join_split_pattern = r"\b(?:(?:LEFT|RIGHT|FULL)(?:\s+OUTER)?\s+|INNER\s+|CROSS\s+)?JOIN\b"
        parts = re.split(join_split_pattern, from_match.group(1).strip(), flags=re.IGNORECASE)
        for part in parts:
            segment = re.split(r"\bON\b|\bUSING\s*\(", part.strip(), maxsplit=1, flags=re.IGNORECASE)[0].strip()
            match = re.match(
                r'"?([A-Za-z_][A-Za-z0-9_]*)"?\s*(?:AS\s+)?("?([A-Za-z_][A-Za-z0-9_]*)"?\s*)?$',
                segment,
                re.IGNORECASE,
            )
            if not match:
                continue
            table = match.group(1)
            alias = match.group(3) or table
            if table in required_tables:
                refs.append((alias, table))
        return refs

    def _doc_row_id(self, loader: DataLoaderBase, doc_id: str) -> Any:
        info = loader.get_doc_info(doc_id) or {}
        row_id = info.get("source_row_id")
        if not is_null(row_id):
            return row_id
        parent_doc_id = str(info.get("parent_doc_id") or "")
        if "-" in parent_doc_id:
            return parent_doc_id.rsplit("-", 1)[-1]
        return None

    @staticmethod
    def _sql_value(value: Any) -> Any:
        if is_null(value):
            return None
        text = re.sub(r"\s+", " ", str(value).strip())
        text = text.strip(" \t\r\n\"'")
        text = re.sub(r"[\s.。:;；，,]+$", "", text)
        numeric_text = text.replace(",", "")
        if numeric_text.startswith("$"):
            numeric_text = numeric_text[1:]
        if numeric_text.endswith("%"):
            numeric_text = numeric_text[:-1]
        try:
            Decimal(numeric_text)
            return numeric_text
        except (InvalidOperation, ValueError):
            return text

    @staticmethod
    def _canonical_row(row: Tuple[Any, ...]) -> Tuple[str, ...]:
        return tuple("" if is_null(value) else str(value).strip() for value in row)

    def _load_or_generate_mapping(
        self,
        query_id: str,
        loader: DataLoaderBase | None = None,
    ) -> Dict[str, Any]:
        mapping_path = self.out_root / PATH_TEMPLATES.eval_name_mapping(query_id)
        if mapping_path.exists():
            return self.load_json(mapping_path)

        if loader is not None and hasattr(loader, "load_name_map"):
            return loader.load_name_map(query_id)

        logging.warning(
            "[%s:_load_or_generate_mapping] Name map not found at %s; using empty mapping",
            self.__class__.__name__,
            mapping_path,
        )
        return {"attribute": {}}

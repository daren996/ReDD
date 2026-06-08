"""Evaluation workflows for ReDD.

These modules live under `redd.exp` because they are useful workflows, but not
part of the primary runtime stage surface.
"""

from __future__ import annotations

import json
import logging
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
    RESULT_TABLE_KEY,
    SCHEMA_NAME_KEY,
)
from redd.core.utils.data_split import (
    resolve_training_data_count,
    resolve_training_data_split,
    resolve_training_data_split_seed,
    split_doc_ids,
)
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

        full_table_semantic_stats = self.compute_semantic_statistics() if self._semantic_eval_enabled() else None
        if full_table_semantic_stats is not None:
            semantic_match_path = self._save_full_table_semantic_match_results(
                query_id,
                full_table_semantic_stats,
            )
            full_table_semantic_stats["match_result_path"] = str(semantic_match_path)
        self._display_results(dataset_name, query_id, tp, fp, fn, tn, correct, total, attr_stats)
        self._display_query_aware_results(query_aware_eval)

        eval_results = {
            "query_aware": query_aware_eval.to_dict(),
            "legacy": {"doc_stats": doc_stats, "attr_stats": attr_stats},
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

            prediction_data.append(
                {
                    "doc_id": doc_id,
                    "table": result_dict[doc_id].get("res"),
                    "data": result_dict[doc_id].get("data", {}),
                }
            )
            ground_truth_data.append(
                {
                    "doc_id": doc_id,
                    "table": loader.get_doc_table(doc_id),
                    "data": loader.get_doc_data(doc_id),
                }
            )

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
            pred_record = result_dict.get(doc_id) if isinstance(result_dict.get(doc_id), dict) else {}
            pred_table = pred_record.get(RESULT_TABLE_KEY)
            pred_data = pred_record.get(RESULT_DATA_KEY, {})
            if not isinstance(pred_data, dict):
                pred_data = {}

            doc_detail = {
                "pred_table": pred_table,
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

                table_ok = pred_table == gt_table
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

                    cell_total += 1
                    doc_detail["cells_total"] += 1
                    pred_value = pred_data.get(attr) if table_ok else None
                    cell_ok = table_ok and self._compare_attribute_values(pred_value, gt_value)
                    if semantic_eval_enabled:
                        semantic_total += 1
                        if not table_ok:
                            semantic_comparison = {
                                "result": False,
                                "method": "table_mismatch",
                                "reasoning": "The predicted row is not assigned to the required table.",
                                "cached": False,
                            }
                        else:
                            semantic_comparison = self._semantic_match_attribute(
                                pred_attr=str(attr),
                                pred_value=pred_value,
                                gt_attr=str(attr),
                                gt_value=gt_value,
                            )
                        semantic_ok = bool(semantic_comparison.get("result"))
                        if semantic_ok:
                            semantic_correct += 1
                        else:
                            semantic_method = str(semantic_comparison.get("method") or "")
                            if semantic_method == "table_mismatch":
                                semantic_table_mismatched += 1
                            elif is_null(pred_value):
                                semantic_missing += 1
                            else:
                                semantic_mismatched += 1
                        if semantic_comparison.get("method") in {"llm", "committee_llm"}:
                            semantic_llm_judged += 1
                        semantic_cells.append(
                            {
                                "doc_id": doc_id,
                                "table": gt_table,
                                "attr": attr,
                                "gt": gt_value,
                                "pred_table": pred_table,
                                "pred": pred_value,
                                **semantic_comparison,
                            }
                        )
                    if cell_ok:
                        cell_covered += 1
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
                        doc_detail["missing"].append(miss)
                    else:
                        cell_mismatched += 1
                        doc_detail["mismatched"].append(miss)

            if not gt_records and not is_null(pred_table):
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
            elif pred_table in required_tables:
                expected_attrs = set(required_by_table.get(str(pred_table), []))
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

            if gt_records or not is_null(pred_table):
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
            semantic_cell_accuracy=semantic_cell_accuracy,
        )

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
        cache_key = self._semantic_cache_key(pred_attr, pred_value, gt_attr, gt_value)
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            return {
                "result": bool(cached.get("result")),
                "method": cached.get("method", "llm"),
                "reasoning": cached.get("reasoning", ""),
                "cached": True,
                "llm_model": cached.get("llm_model"),
            }

        comparison = self._call_semantic_judge(pred_attr, pred_value, gt_attr, gt_value)
        cache[cache_key] = {
            **comparison,
            "pred_attr": pred_attr,
            "pred_value": pred_value,
            "gt_attr": gt_attr,
            "gt_value": gt_value,
            "llm_model": self.eval_llm_model,
        }
        self.save_results(cache_path, cache)
        return comparison

    @staticmethod
    def _semantic_cache_key(pred_attr: str, pred_value: Any, gt_attr: str, gt_value: Any) -> str:
        return json.dumps(
            {
                "pred_attr": pred_attr,
                "pred_value": "" if is_null(pred_value) else str(pred_value),
                "gt_attr": gt_attr,
                "gt_value": "" if is_null(gt_value) else str(gt_value),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _call_semantic_judge(
        self,
        pred_attr: str,
        pred_value: Any,
        gt_attr: str,
        gt_value: Any,
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
                parsed = self._parse_semantic_response(prompt(msg=message, max_tokens=256, temperature=0))
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
                parsed = self._parse_semantic_response(prompt(msg=message, max_tokens=256, temperature=0))
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
        answer_doc_ids_by_table: Optional[Dict[str, set[str]]] = None,
    ) -> List[Dict[str, Any]]:
        info = loader.get_doc_info(doc_id)
        if not info:
            return []
        records = info.get("data_records") or []
        if not records and info.get("table"):
            records = [{"table_name": info.get("table"), "data": info.get("data", {})}]

        required_tables = set(required_by_table)
        result = []
        for record in records:
            if not isinstance(record, dict):
                continue
            table = record.get("table_name") or record.get("table")
            if table not in required_tables:
                continue
            table_name = str(table)
            if answer_doc_ids_by_table is not None and doc_id not in answer_doc_ids_by_table.get(table_name, set()):
                continue
            data = record.get("data") or {}
            if not isinstance(data, dict):
                data = {}
            row_id = (
                record.get("source_row_id")
                or record.get("row_id")
                or data.get("row_id")
                or info.get("source_row_id")
            )
            result.append({"table": table_name, "data": data, "row_id": row_id})
        return result

    def _answer_doc_ids_by_table(
        self,
        *,
        loader: DataLoaderBase,
        query_info: Dict[str, Any],
        eval_doc_ids: List[str],
        required_by_table: Dict[str, List[str]],
    ) -> Optional[Dict[str, set[str]]]:
        """Return GT docs that participate in the SQL answer rows.

        Query-aware table/cell recall should measure the rows needed to answer
        the SQL query, not every row from every table referenced by the query.
        The temporary SQL tables include ``__doc_id`` specifically so we can
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

        result: Dict[str, set[str]] = {table: set() for table in required_by_table}
        for row in provenance_rows:
            for table, doc_id in row.items():
                if table in result and not is_null(doc_id):
                    result[table].add(str(doc_id))
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
            record = result_dict.get(doc_id)
            if not isinstance(record, dict):
                continue
            table = record.get(RESULT_TABLE_KEY)
            if table not in required_by_table:
                continue
            data = record.get(RESULT_DATA_KEY, {})
            if not isinstance(data, dict):
                data = {}
            row = {attr: data.get(attr) for attr in required_by_table[str(table)]}
            row["__doc_id"] = doc_id
            row["row_id"] = self._doc_row_id(loader, doc_id)
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

        select_parts = [
            f'"{ref}"."__doc_id" AS "__doc_id__{table}"'
            for ref, table in table_refs
            if table in required_by_table
        ]
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
                result.append({table: row[f"__doc_id__{table}"] for _, table in table_refs if table in required_by_table})
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

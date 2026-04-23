"""Evaluation workflows for ReDD.

These modules live under `redd.exp` because they are useful workflows, but not
part of the primary runtime stage surface.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from redd.core.data_loader import DataLoaderBase, create_data_loader
from redd.core.utils import constants
from redd.core.utils.constants import (
    ATTRIBUTE_NAME_KEY,
    ATTRIBUTE_VALUE_KEY,
    GROUND_TRUTH_KEY,
    PATH_TEMPLATES,
    PREDICTION_KEY,
)
from redd.core.utils.utils import is_null

__all__ = [
    "EvalBasic",
    "EvalDataPop",
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
    """Container for data-population evaluation metrics."""

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


class EvalDataPop(EvalBasic):
    """Data-population evaluation with optional LLM-based semantic comparison."""

    def __init__(
        self,
        config: Dict[str, Any],
        data_loader: DataLoaderBase | None = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config, data_loader)

        self.loader_type = str(config.get("data_loader_type", "sqlite")).lower()
        self.loader_config = deepcopy(dict(config.get("data_loader_config") or {}))

        eval_config = config.get("eval", {})
        self.res_param_str = config.get("res_param_str", "default")
        self.name_map: Optional[Dict[str, Any]] = None
        self.committee_prompts: List[Dict[str, Any]] = []
        self.prompts: Dict[str, Any] = {}

        if "committee" in eval_config:
            self._initialize_committee(eval_config, api_key)
        else:
            self._initialize_single_eval(eval_config, api_key)

    def _initialize_single_eval(self, eval_config: Dict[str, Any], api_key: Optional[str]) -> None:
        if not eval_config:
            return

        from redd.core.llm.providers import normalize_provider_name
        from redd.core.utils.prompt_utils import get_api_key

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
            )
        except Exception as exc:
            logging.warning(
                "[%s:_initialize_prompts] Failed to initialize prompts: %s",
                self.__class__.__name__,
                exc,
            )
            self.prompts = {}

    def _initialize_committee(self, eval_config: Dict[str, Any], api_key: Optional[str] = None) -> None:
        from redd.core.llm.providers import normalize_provider_name
        from redd.core.utils.prompt_utils import create_prompt_map, get_api_key

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

    def _evaluate_dataset(self, dataset_name: str, data_root: str | Path, out_root: str | Path) -> None:
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
            loader_config=deepcopy(self.loader_config),
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

        for query_id in query_dict:
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
        result_path = self.out_root / PATH_TEMPLATES.data_population_result(query_id, self.res_param_str)

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
        self.name_map = self._load_or_generate_mapping(query_id)
        self.prediction_data, self.gt_data = self._prepare_evaluation_data(loader, result_dict)

        stats = self.compute_statistics()
        if stats is None:
            logging.error("[%s:_evaluate_query] Failed to compute statistics for query %s", self.__class__.__name__, query_id)
            return

        tp, fp, fn, tn, correct, total, doc_stats, attr_stats = stats
        self._display_results(dataset_name, query_id, tp, fp, fn, tn, correct, total, attr_stats)

        eval_results = {"doc_stats": doc_stats, "attr_stats": attr_stats}
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
        return str(pred_val).strip() == str(gt_val).strip()

    def _load_or_generate_mapping(self, query_id: str) -> Dict[str, Any]:
        if self.name_map is not None:
            return self.name_map

        mapping_path = self.out_root / PATH_TEMPLATES.name_map(query_id, self.res_param_str)
        if mapping_path.exists():
            return self.load_json(mapping_path)

        logging.warning(
            "[%s:_load_or_generate_mapping] Name map not found at %s; using empty mapping",
            self.__class__.__name__,
            mapping_path,
        )
        return {"attribute": {}}

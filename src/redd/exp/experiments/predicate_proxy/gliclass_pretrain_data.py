"""Experiment-only GLiClass pretraining helpers for predicate proxies.

These utilities intentionally live under `redd.exp.experiments` rather than the
main runtime path.
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from redd.core.data_loader import create_data_loader
from redd.core.utils.sql_filter_parser import (
    group_predicates_by_table,
    parse_alias_mapping,
)
from redd.proxy.predicate_proxy.finetuned_proxy import _format_predicate_context

__all__ = [
    "extract_from_multiple_datasets",
    "extract_training_pairs",
    "pretrain_and_save_gliclass",
]


def _get_positive_doc_ids_for_query(
    loader: Any,
    sql: str,
    query_tables: List[str],
) -> Set[str]:
    """Find doc IDs that produce rows in the ground-truth query result."""
    gt_db_path = getattr(loader, "gt_db_path", None)
    if not gt_db_path or not Path(gt_db_path).exists():
        return set()

    sql_upper = sql.upper()
    if any(kw in sql_upper for kw in (" INTERSECT ", " UNION ", " EXCEPT ", " GROUP BY ", " HAVING ")):
        return set()

    table_map: Dict[str, str] = {}
    if hasattr(loader, "load_name_map"):
        try:
            name_map = loader.load_name_map()
            task_to_gt = (name_map.get("table") or {})
            for table_name in query_tables:
                table_map[table_name] = task_to_gt.get(table_name, table_name)
        except Exception:
            for table_name in query_tables:
                table_map[table_name] = table_name
    else:
        for table_name in query_tables:
            table_map[table_name] = table_name

    gt_tables = [table_map.get(table_name, table_name) for table_name in query_tables]
    alias_to_table = parse_alias_mapping(sql)
    table_to_alias = {}
    for alias, table_name in alias_to_table.items():
        if table_name in gt_tables:
            table_to_alias[table_name] = alias

    select_parts = []
    for table_name in gt_tables:
        alias = table_to_alias.get(table_name, table_name)
        select_parts.append(f'"{alias}".row_id AS _rid_{table_name}')
    if not select_parts:
        return set()

    sql_mod = re.sub(
        r"^\s*SELECT\s+.+?\s+FROM\s+",
        f"SELECT {', '.join(select_parts)} FROM ",
        sql,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if sql_mod == sql:
        return set()

    try:
        conn = sqlite3.connect(str(gt_db_path))
        cursor = conn.cursor()
        cursor.execute(sql_mod)
        rows = cursor.fetchall()
        description = cursor.description
        conn.close()
    except sqlite3.Error as exc:
        logging.debug("[_get_positive_doc_ids_for_query] SQL error: %s", exc)
        return set()

    row_id_cols = [f"_rid_{table_name}" for table_name in gt_tables]
    table_row_pairs: Set[Tuple[str, str]] = set()
    for row in rows:
        row_dict = dict(zip([item[0] for item in description], row))
        for table_name, col in zip(gt_tables, row_id_cols):
            row_id = row_dict.get(col)
            if row_id is not None:
                table_row_pairs.add((table_name, str(row_id)))

    if not table_row_pairs:
        return set()

    input_conn = getattr(loader, "_input_conn", None)
    if not input_conn:
        return set()

    doc_ids = set()
    for table_name, row_id in table_row_pairs:
        try:
            cur = input_conn.cursor()
            cur.execute(
                "SELECT doc_id FROM mapping WHERE table_name = ? AND row_id = ?",
                (table_name, row_id),
            )
            for (doc_id,) in cur.fetchall():
                doc_ids.add(doc_id)
        except sqlite3.OperationalError:
            pass

    return doc_ids


def extract_training_pairs(
    doc_dir: str,
    data_main: str = "dataset/",
    max_samples: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Extract predicate-proxy training pairs from a Spider-style dataset."""
    full_doc_dir = Path(data_main) / doc_dir if not Path(doc_dir).is_absolute() else Path(doc_dir)
    loader = create_data_loader(
        data_root=full_doc_dir,
        loader_type="hf_manifest",
        loader_config={},
    )

    query_dict = loader.load_query_dict()
    if not query_dict:
        logging.warning("[extract_training_pairs] No queries in %s", doc_dir)
        return []

    schema_list = []
    if hasattr(loader, "load_schema_query"):
        first_qid = next(iter(query_dict.keys()), None)
        if first_qid:
            schema_list = loader.load_schema_query(first_qid)

    all_pairs: List[Dict[str, Any]] = []
    doc_ids = loader.doc_ids

    for qid, query_info in query_dict.items():
        sql = query_info.get("sql", "")
        if not sql:
            continue

        query_text = query_info.get("query", "")
        query_tables = query_info.get("tables", [])

        try:
            predicates_by_table = group_predicates_by_table(
                sql,
                schema_list,
                query_tables=query_tables,
            )
        except Exception as exc:
            logging.debug("[extract_training_pairs] Skip %s: %s", qid, exc)
            continue

        positive_doc_ids = _get_positive_doc_ids_for_query(loader, sql, query_tables)
        if not positive_doc_ids:
            logging.debug("[extract_training_pairs] No positive docs for %s, skipping", qid)
            continue

        for table_name, predicates in predicates_by_table.items():
            if not predicates:
                continue

            for pred in predicates:
                predicate_context = _format_predicate_context(pred, query_text)

                for doc_id in doc_ids:
                    try:
                        doc_text, _, _ = loader.get_doc(doc_id)
                        label = 1 if doc_id in positive_doc_ids else 0
                        all_pairs.append(
                            {
                                "document": doc_text,
                                "predicate_context": predicate_context,
                                "label": label,
                                "attribute": pred.attribute,
                                "doc_id": doc_id,
                                "qid": qid,
                                "table": table_name,
                            }
                        )
                    except Exception as exc:
                        logging.debug("[extract_training_pairs] Skip %s: %s", doc_id, exc)

    if len(all_pairs) > max_samples:
        rng = random.Random(seed)
        pos_pairs = [pair for pair in all_pairs if pair["label"] == 1]
        neg_pairs = [pair for pair in all_pairs if pair["label"] == 0]
        n_pos = min(len(pos_pairs), max_samples // 2)
        n_neg = max_samples - n_pos
        sampled_pos = rng.sample(pos_pairs, n_pos) if len(pos_pairs) >= n_pos else pos_pairs
        sampled_neg = rng.sample(neg_pairs, n_neg) if len(neg_pairs) >= n_neg else neg_pairs
        all_pairs = sampled_pos + sampled_neg
        rng.shuffle(all_pairs)

    return all_pairs


def extract_from_multiple_datasets(
    task_paths: List[str],
    data_main: str = "dataset/",
    max_samples_per_dataset: Optional[int] = None,
    max_total: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Extract and sample predicate-proxy training pairs across datasets."""
    all_pairs: List[Dict[str, Any]] = []
    per_dataset = max_total // len(task_paths) if max_samples_per_dataset is None else max_samples_per_dataset

    for path in task_paths:
        try:
            pairs = extract_training_pairs(
                doc_dir=path,
                data_main=data_main,
                max_samples=per_dataset or max_total,
                seed=seed,
            )
            all_pairs.extend(pairs)
        except Exception as exc:
            logging.warning("[extract_from_multiple_datasets] Failed %s: %s", path, exc)

    if len(all_pairs) > max_total:
        rng = random.Random(seed)
        pos_pairs = [pair for pair in all_pairs if pair["label"] == 1]
        neg_pairs = [pair for pair in all_pairs if pair["label"] == 0]
        n_pos = min(len(pos_pairs), max_total // 2)
        n_neg = max_total - n_pos
        sampled_pos = rng.sample(pos_pairs, n_pos) if len(pos_pairs) >= n_pos else pos_pairs
        sampled_neg = rng.sample(neg_pairs, n_neg) if len(neg_pairs) >= n_neg else neg_pairs
        all_pairs = sampled_pos + sampled_neg
        rng.shuffle(all_pairs)

    return all_pairs


def pretrain_and_save_gliclass(
    training_pairs: List[Dict[str, Any]],
    output_dir: str | Path,
    model_name: str = "knowledgator/gliclass-instruct-large-v1.0",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> Path:
    """Fine-tune GLiClass on extracted pairs and save the model."""
    from redd.proxy.predicate_proxy.finetuned_proxy import (
        GLICLASS_AVAILABLE,
        GLICLASS_TRAINING_AVAILABLE,
        _set_reproducible_seed,
    )

    if not GLICLASS_AVAILABLE:
        raise ImportError("gliclass required. pip install gliclass")
    if not GLICLASS_TRAINING_AVAILABLE:
        raise ImportError("gliclass.training required for fine-tuning. Install from source.")

    import torch
    from gliclass import GLiClassModel
    from gliclass.data_processing import (
        AugmentationConfig,
        DataCollatorWithPadding,
        GLiClassDataset,
    )
    from gliclass.training import Trainer, TrainingArguments
    from transformers import AutoTokenizer

    _set_reproducible_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GLiClassModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    labels_gliclass = ["satisfies", "does not satisfy"]
    train_data = []
    for pair in training_pairs:
        doc = pair["document"][:4000]
        pred_ctx = pair.get("predicate_context", "Does this document satisfy the filter?")
        true_label = "satisfies" if pair["label"] == 1 else "does not satisfy"
        train_data.append(
            {
                "text": doc,
                "all_labels": labels_gliclass,
                "true_labels": [true_label],
                "prompt": pred_ctx,
            }
        )

    aug_config = AugmentationConfig(enabled=False)
    train_dataset = GLiClassDataset(
        train_data,
        tokenizer,
        aug_config,
        label2description={},
        max_length=512,
        problem_type="multi_label_classification",
        architecture_type="uni-encoder",
        prompt_first=True,
        shuffle_labels=True,
    )
    data_collator = DataCollatorWithPadding(device=device)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    model = trainer.model

    final_dir = output_path / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    logging.info("[pretrain_and_save_gliclass] Model saved to %s", final_dir)
    return final_dir

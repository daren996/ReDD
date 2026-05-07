from __future__ import annotations

from pathlib import Path

import pytest

from redd.core.data_population.strategies.proxy_runtime import (
    GTTextConsistencyProxy,
    OraclePredicateProxy,
    ProxyRuntimeExtractionStrategy,
)
from redd.core.utils.sql_filter_parser import AttributePredicate


class DummyLoader:
    doc_ids: list[str] = []


class DummyOracle:
    def extract(self, *, doc_id: str | None = None, **kwargs):
        return {"score": 10, "name": doc_id}

    def check_predicates(self, extracted_values, predicates):
        per_attr = {name: predicate(extracted_values.get(name)) for name, predicate in predicates.items()}
        return all(per_attr.values()), per_attr


def _base_config(*, oracle: str = "ground_truth", llm_mode: str = "none") -> dict:
    return {
        "oracle": oracle,
        "mode": llm_mode,
        "llm_model": "ground_truth",
        "data_main": "dataset/",
        "proxy_runtime": {
            "enabled": True,
            "predicate_proxy_mode": "pretrained",
            "use_embedding_proxies": False,
            "use_learned_proxies": True,
            "use_finetuned_learned_proxies": True,
            "finetuned_model": "heuristic",
            "use_oracle_predicate_proxy": True,
        },
    }


def test_oracle_predicate_proxy_requires_ground_truth_mode() -> None:
    with pytest.raises(ValueError, match="offline ground-truth ablations"):
        ProxyRuntimeExtractionStrategy(
            extraction_config=_base_config(oracle="llm", llm_mode="openai"),
            data_path=Path("dataset/derived/bird.schools_demo"),
            loader=DummyLoader(),
        )


def test_oracle_predicate_proxy_allows_ground_truth_ablations() -> None:
    strategy = ProxyRuntimeExtractionStrategy(
        extraction_config=_base_config(),
        data_path=Path("dataset/derived/bird.schools_demo"),
        loader=DummyLoader(),
    )

    assert strategy.proxy_runtime_config.use_oracle_predicate_proxy is True


def test_proxy_runtime_forwards_embedding_cache_dir() -> None:
    config = _base_config()
    config["proxy_runtime"]["embeddings_cache_dir"] = "outputs/demo/_embedding_cache"

    strategy = ProxyRuntimeExtractionStrategy(
        extraction_config=config,
        data_path=Path("dataset/derived/bird.schools_demo"),
        loader=DummyLoader(),
    )

    assert strategy.proxy_runtime_config.embeddings_cache_dir == "outputs/demo/_embedding_cache"


def test_gt_text_consistency_guard_requires_ground_truth_mode() -> None:
    config = _base_config(oracle="llm", llm_mode="openai")
    config["proxy_runtime"]["use_oracle_predicate_proxy"] = False
    config["proxy_runtime"]["use_gt_text_consistency_guard"] = True

    with pytest.raises(ValueError, match="offline ground-truth ablations"):
        ProxyRuntimeExtractionStrategy(
            extraction_config=config,
            data_path=Path("dataset/derived/bird.schools_demo"),
            loader=DummyLoader(),
        )


def test_train_mode_table_without_training_docs_falls_back_to_heuristic_pretrained() -> None:
    config = _base_config()
    config["proxy_runtime"]["predicate_proxy_mode"] = "train"
    config["proxy_runtime"]["use_finetuned_learned_proxies"] = False
    config["proxy_runtime"]["finetuned_model"] = "logreg"
    strategy = ProxyRuntimeExtractionStrategy(
        extraction_config=config,
        data_path=Path("dataset/derived/bird.schools_demo"),
        loader=DummyLoader(),
    )

    table_config = strategy._proxy_config_for_table(
        table_name="teaches",
        predicates=[AttributePredicate("city", "=", "Paris")],
        train_doc_ids_for_table=[],
    )

    assert table_config.predicate_proxy_mode == "pretrained"
    assert table_config.finetuned_model == "heuristic"
    assert table_config.use_finetuned_learned_proxies is True


def test_oracle_predicate_proxy_accepts_executor_metadata_kwarg() -> None:
    proxy = OraclePredicateProxy(
        name="oracle_predicate_demo",
        oracle=DummyOracle(),  # type: ignore[arg-type]
        schema={},
        attributes=["score"],
        predicate_fns={"score": lambda value: value == 10},
    )

    scores, passed = proxy.evaluate_documents(
        ["score is 10"],
        doc_ids=["doc-1"],
        metadata=[{"source_table": "demo"}],
    )

    assert scores.tolist() == [1.0]
    assert passed.tolist() == [True]


def test_gt_text_consistency_proxy_accepts_executor_metadata_kwarg() -> None:
    proxy = GTTextConsistencyProxy(
        name="gt_text_consistency_demo",
        oracle=DummyOracle(),  # type: ignore[arg-type]
        schema={},
        attributes=["score"],
        predicates=[
            AttributePredicate(attribute="score", operator=">", value=5),
        ],
    )

    scores, passed = proxy.evaluate_documents(
        ["score is 10"],
        doc_ids=["doc-1"],
        metadata=[{"source_table": "demo"}],
    )

    assert scores.tolist() == [1.0]
    assert passed.tolist() == [True]

from __future__ import annotations

from pathlib import Path

import pytest

from redd.core.data_population.strategies.proxy_runtime import ProxyRuntimeExtractionStrategy


class DummyLoader:
    doc_ids: list[str] = []


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

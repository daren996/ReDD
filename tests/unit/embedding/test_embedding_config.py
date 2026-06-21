from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from redd.embedding.config import (
    embedding_manager_kwargs,
    resolve_embedding_storage_path,
)


def test_resolve_embedding_storage_path_uses_explicit_storage_path(tmp_path: Path) -> None:
    target = tmp_path / "explicit.sqlite3"

    resolved = resolve_embedding_storage_path(
        config={
            "storage_path": target,
            "cache_dir": tmp_path / "cache",
            "storage_file": "ignored.sqlite3",
        },
        out_root=tmp_path / "artifact",
    )

    assert resolved == target.resolve()


def test_resolve_embedding_storage_path_uses_dataset_scoped_cache_dir(
    tmp_path: Path,
) -> None:
    loader = SimpleNamespace(data_root=tmp_path / "dataset" / "spider.college_demo")

    resolved = resolve_embedding_storage_path(
        config={"cache_dir": tmp_path / "_embedding_cache"},
        loader=loader,
    )

    assert resolved == (
        tmp_path / "_embedding_cache" / "spider.college_demo.embeddings.sqlite3"
    ).resolve()


def test_resolve_embedding_storage_path_accepts_legacy_cache_dir_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "legacy.sqlite3"

    resolved = resolve_embedding_storage_path(
        config={"embeddings_cache_dir": target},
        loader=SimpleNamespace(data_root=tmp_path / "dataset"),
    )

    assert resolved == target.resolve()


def test_resolve_embedding_storage_path_uses_artifact_local_storage_file(
    tmp_path: Path,
) -> None:
    resolved = resolve_embedding_storage_path(
        config={"storage_file": "schema_embeddings.sqlite3"},
        out_root=tmp_path / "artifact",
    )

    assert resolved == (tmp_path / "artifact" / "schema_embeddings.sqlite3").resolve()


def test_resolve_embedding_storage_path_uses_loader_root_storage_file(
    tmp_path: Path,
) -> None:
    loader = SimpleNamespace(data_root=tmp_path / "dataset")

    resolved = resolve_embedding_storage_path(
        config={"embedding_storage_file": "runtime_embeddings.sqlite3"},
        loader=loader,
    )

    assert resolved == (tmp_path / "dataset" / "runtime_embeddings.sqlite3").resolve()


def test_embedding_manager_kwargs_accepts_global_and_stage_key_names() -> None:
    kwargs = embedding_manager_kwargs(
        {
            "embedding_model": "local-hash-embedding",
            "embedding_api_key": "stage-key",
            "embedding_provider": "local",
            "embedding_base_url": "http://example.invalid",
        },
        default_model="text-embedding-3-small",
        fallback_api_key="fallback-key",
    )

    assert kwargs == {
        "model": "local-hash-embedding",
        "api_key": "stage-key",
        "provider": "local",
        "base_url": "http://example.invalid",
    }

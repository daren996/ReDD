from __future__ import annotations

from redd.optimizations.doc_filtering.runtime import (
    load_doc_filter_result,
    save_doc_filter_result,
)


def test_save_doc_filter_result_writes_only_doc_filter_artifact(tmp_path):
    path = save_doc_filter_result(
        query_id="q1",
        excluded_doc_ids={"d2"},
        all_doc_ids=["d1", "d2"],
        out_root=tmp_path,
        param_str="unit",
        doc_filter_config={"target_recall": 0.95},
    )

    assert path.parent == tmp_path / "doc_filter"
    assert path.exists()
    assert sorted(child.name for child in tmp_path.iterdir()) == ["doc_filter"]


def test_load_doc_filter_result_reads_doc_filter_artifact(tmp_path):
    saved_path = save_doc_filter_result(
        query_id="q1",
        excluded_doc_ids={"d2"},
        all_doc_ids=["d1", "d2"],
        out_root=tmp_path,
        param_str="unit",
        doc_filter_config={"target_recall": 0.95},
    )

    payload, path = load_doc_filter_result(tmp_path, query_id="q1")

    assert path == saved_path
    assert payload is not None
    assert payload["excluded_doc_ids"] == ["d2"]

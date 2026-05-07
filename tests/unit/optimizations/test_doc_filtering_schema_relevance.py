from __future__ import annotations

from redd.optimizations.doc_filtering.schema_relevance_filter import SchemaRelevanceFilter


class DummyLoader:
    def __init__(self) -> None:
        self.docs = {
            "d1": ("alpha", {"schema_tables": ["course"], "table_name": "course"}),
            "d2": ("beta", {"schema_table": "student", "table_name": "student_src"}),
            "d3": ("gamma", {}),
            "d4": ("delta", {"table_name": "k181915"}),
        }

    def get_doc(self, doc_id: str):
        text, metadata = self.docs[doc_id]
        return text, doc_id, metadata

    def load_schema_query(self, query_id: str):
        assert query_id == "q1"
        return [{"Schema Name": "course", "Attributes": []}]

    def load_schema_general(self):
        return [{"Schema Name": "course", "Attributes": []}]

    def load_name_map(self, query_id: str):
        assert query_id == "q1"
        return {"table": {"course": "course"}}


def test_source_table_metadata_only_filters_non_query_tables_and_keeps_unknowns() -> None:
    doc_filter = SchemaRelevanceFilter(
        {
            "filter_type": "schema_relevance",
            "use_source_table_metadata": True,
            "source_table_metadata_only": True,
        }
    )

    result = doc_filter.filter(
        query_id="q1",
        doc_ids=["d1", "d2", "d3", "d4"],
        data_loader=DummyLoader(),
    )

    assert result.excluded_doc_ids == {"d2"}
    assert result.metadata["num_docs_kept"] == 3
    assert result.metadata["source_table_known_docs"] == 2
    assert result.metadata["source_table_unknown_docs"] == 2


def test_source_table_metadata_can_exclude_unknowns_when_requested() -> None:
    doc_filter = SchemaRelevanceFilter(
        {
            "filter_type": "schema_relevance",
            "use_source_table_metadata": True,
            "source_table_metadata_only": True,
            "source_table_keep_unknown": False,
        }
    )

    result = doc_filter.filter(
        query_id="q1",
        doc_ids=["d1", "d2", "d3", "d4"],
        data_loader=DummyLoader(),
    )

    assert result.excluded_doc_ids == {"d2", "d3", "d4"}

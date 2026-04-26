from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from pydantic import BaseModel

from redd.core.data_population.data_extraction import DataExtraction
from redd.core.utils.constants import TABLE_ASSIGNMENT_KEY
from redd.core.utils.structured_outputs import AttributeExtractionOutput, TableAssignmentOutput
from redd.embedding.providers import EmbeddingProvider
from redd.llm import CompletionRequest, LLMRuntime
from redd.retrieval import build_retrieval_index


def _completion_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage={"total_tokens": 3},
        model="mock-model",
    )


class RuntimeUser(BaseModel):
    name: str
    age: int


class FakePrompt:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_model(self, msg, response_model, **kwargs):
        self.calls.append((msg, response_model, kwargs))
        return self.payload


class AIRuntimeTests(unittest.TestCase):
    def test_litellm_text_completion_uses_provider_mapping(self) -> None:
        runtime = LLMRuntime.from_config(
            "openai",
            "gpt-4o-mini",
            config={"llm_model": "gpt-4o-mini", "max_retries": 0, "wait_time": 0},
            api_key="test-key",
        )

        with patch("litellm.completion", return_value=_completion_response("hello")) as completion:
            result = runtime.complete_text(
                CompletionRequest(
                    messages=[{"role": "user", "content": "hi"}],
                    response_format="text",
                )
            )

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.usage, {"total_tokens": 3})
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertNotIn("response_format", kwargs)

    def test_litellm_completion_retries_with_tenacity(self) -> None:
        runtime = LLMRuntime.from_config(
            "openai",
            "gpt-4o-mini",
            config={"llm_model": "gpt-4o-mini", "max_retries": 1, "wait_time": 0},
            api_key="test-key",
        )

        with patch(
            "litellm.completion",
            side_effect=[RuntimeError("try again"), _completion_response("ok")],
        ) as completion:
            result = runtime.complete_text(
                CompletionRequest(messages=[{"role": "user", "content": "hi"}])
            )

        self.assertEqual(result.text, "ok")
        self.assertEqual(completion.call_count, 2)

    def test_complete_model_falls_back_to_pydantic_json_without_instructor(self) -> None:
        runtime = LLMRuntime.from_config(
            "openai",
            "gpt-4o-mini",
            config={"llm_model": "gpt-4o-mini", "max_retries": 0, "wait_time": 0},
            api_key="test-key",
        )

        with patch.dict("sys.modules", {"instructor": None}):
            with patch(
                "litellm.completion",
                return_value=_completion_response('{"name": "Ada", "age": 37}'),
            ):
                user = runtime.complete_model(
                    CompletionRequest(messages=[{"role": "user", "content": "extract"}]),
                    RuntimeUser,
                )

        self.assertEqual(user.name, "Ada")
        self.assertEqual(user.age, 37)

    def test_embedding_provider_uses_litellm_embedding_order(self) -> None:
        response = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }
        provider = EmbeddingProvider(model="text-embedding-3-small", api_key="test-key")

        with patch("litellm.embedding", return_value=response) as embedding:
            values = provider.embed_batch(["a", "b"], batch_size=2)

        self.assertEqual(values, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(provider.embedding_dim, 2)
        self.assertEqual(embedding.call_args.kwargs["model"], "openai/text-embedding-3-small")

    def test_data_extraction_table_assignment_uses_pydantic_output(self) -> None:
        extractor = DataExtraction.__new__(DataExtraction)
        extractor.schema_general = [{"Schema Name": "album"}]
        extractor.retry_params = {}
        extractor.prompt_table = cast(
            Any,
            FakePrompt(
            TableAssignmentOutput.model_validate({TABLE_ASSIGNMENT_KEY: "album"})
            ),
        )

        table, failed, reason = extractor._assign_table_single_doc(
            doc_id="1",
            doc_text="doc",
            all_tables=["album"],
            max_retries=0,
        )

        self.assertEqual(table, "album")
        self.assertFalse(failed)
        self.assertIsNone(reason)

    def test_data_extraction_attr_extraction_requires_target_attr(self) -> None:
        extractor = DataExtraction.__new__(DataExtraction)
        extractor.retry_params = {}
        extractor.prompt_attr = cast(Any, FakePrompt(AttributeExtractionOutput({"name": "Blue"})))

        value = extractor._extract_attr_single_doc(
            doc_id="1",
            doc_text="doc",
            attr="name",
            table_schema={"Schema Name": "album"},
            max_retries=0,
        )

        self.assertEqual(value, "Blue")

        extractor.prompt_attr = cast(Any, FakePrompt(AttributeExtractionOutput({"other": "Blue"})))
        missing = extractor._extract_attr_single_doc(
            doc_id="1",
            doc_text="doc",
            attr="name",
            table_schema={"Schema Name": "album"},
            max_retries=0,
        )
        self.assertIsNone(missing)

    def test_retrieval_auto_backend_falls_back_to_numpy_without_faiss(self) -> None:
        index = build_retrieval_index(
            {"a": [1.0, 0.0], "b": [0.0, 1.0]},
            model="mock-embedding",
            backend="auto",
        )

        with patch.dict("sys.modules", {"faiss": None}):
            matches = index.search([1.0, 0.0], top_k=1)

        self.assertEqual(matches[0].item_id, "a")
        self.assertAlmostEqual(matches[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()

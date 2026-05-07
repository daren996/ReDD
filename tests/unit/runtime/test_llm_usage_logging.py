from __future__ import annotations

import json
from types import SimpleNamespace

from redd.llm.providers import LLMConfig, _append_usage_log


def test_append_usage_log_writes_token_counts_without_prompt_content(tmp_path, monkeypatch) -> None:
    path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("REDD_LLM_USAGE_LOG", str(path))

    _append_usage_log(
        config=LLMConfig(mode="deepseek", model="deepseek-chat"),
        litellm_model="deepseek/deepseek-chat",
        response_model="deepseek-v4-flash",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
    )

    payload = json.loads(path.read_text().strip())
    assert payload["provider"] == "deepseek"
    assert payload["configured_model"] == "deepseek-chat"
    assert payload["response_model"] == "deepseek-v4-flash"
    assert payload["usage"]["prompt_tokens"] == 10
    assert "messages" not in payload
    assert "api_key" not in payload

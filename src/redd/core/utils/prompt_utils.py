from __future__ import annotations

import logging
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Mapping, TypeVar

from pydantic import BaseModel

from redd.config import resolve_repo_path
from redd.llm import CompletionRequest, LLMRuntime, get_api_key, normalize_provider_name

ModelT = TypeVar("ModelT", bound=BaseModel)


def _get_package_prompt_root():
    prompt_root = importlib_resources.files("redd")
    prompt_root = prompt_root.joinpath("resources")
    prompt_root = prompt_root.joinpath("prompts")
    return prompt_root


def resolve_prompt_reference(prompt_path: str | Path):
    """
    Resolve a prompt reference without relying on the current working directory.

    Resolution order:
    1. Absolute filesystem path
    2. Repository-root relative path
    3. Packaged resource under ``redd/resources/prompts``
    """
    candidate = Path(prompt_path)

    if candidate.is_absolute():
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Prompt file not found: {candidate}")

    repo_candidate = resolve_repo_path(candidate)
    if repo_candidate.is_file():
        return repo_candidate

    package_relative = candidate
    if candidate.parts and candidate.parts[0] == "prompts":
        package_relative = Path(*candidate.parts[1:])

    resource_candidate = _get_package_prompt_root()
    for part in package_relative.parts:
        resource_candidate = resource_candidate.joinpath(part)
    if resource_candidate.is_file():
        return resource_candidate

    raise FileNotFoundError(
        f"Prompt `{prompt_path}` could not be resolved from the repository root or packaged resources."
    )


def load_prompt_text(prompt_path: str | Path) -> str:
    resolved = resolve_prompt_reference(prompt_path)
    return resolved.read_text(encoding="utf-8")


class PromptTemplate:
    """Provider-agnostic prompt wrapper backed by the shared LLM runtime."""

    def __init__(
        self,
        mode: str,
        prompt_path: str | Path,
        llm_model: str,
        api_key: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.mode = normalize_provider_name(mode)
        self.prompt_path = prompt_path
        self.llm_model = llm_model
        self.prompt = load_prompt_text(prompt_path)
        self.resolved_prompt = resolve_prompt_reference(prompt_path)
        self.runtime = LLMRuntime.from_config(
            self.mode,
            self.llm_model,
            config=config,
            api_key=api_key,
        )
        logging.info(
            "[%s] Initialized prompt from %s for provider=%s model=%s",
            self.__class__.__name__,
            self.resolved_prompt,
            self.mode,
            self.llm_model,
        )

    def _messages(self, msg: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": self.prompt + "\n\n" + msg}]

    def __call__(self, msg: str, **kwargs: Any) -> str:
        request = CompletionRequest(
            messages=self._messages(msg),
            response_format=kwargs.get("response_format", "json_object"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            seed=kwargs.get("seed"),
        )
        return self.runtime.complete_text(request).text

    def complete_model(self, msg: str, response_model: type[ModelT], **kwargs: Any) -> ModelT:
        request = CompletionRequest(
            messages=self._messages(msg),
            response_format="json_object",
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            seed=kwargs.get("seed"),
        )
        return self.runtime.complete_model(request, response_model)

    def __str__(self) -> str:
        return self.prompt


def create_prompt(
    mode: str,
    prompt_path: str | Path,
    llm_model: str,
    api_key: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> PromptTemplate:
    return PromptTemplate(
        mode,
        prompt_path,
        llm_model,
        api_key=api_key,
        config=config,
    )


def create_prompt_map(
    mode: str,
    prompt_paths: Mapping[str, str | Path],
    llm_model: str,
    api_key: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, PromptTemplate]:
    return {
        prompt_name: create_prompt(
            mode,
            prompt_path,
            llm_model,
            api_key=api_key,
            config=config,
        )
        for prompt_name, prompt_path in prompt_paths.items()
    }


__all__ = [
    "PromptTemplate",
    "create_prompt",
    "create_prompt_map",
    "get_api_key",
    "load_prompt_text",
    "resolve_prompt_reference",
]

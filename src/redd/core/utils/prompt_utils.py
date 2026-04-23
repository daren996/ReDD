from __future__ import annotations

import logging
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Mapping

from redd.config import resolve_repo_path
from redd.core.llm import create_client, get_api_key, llm_completion, normalize_provider_name


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


class PromptBase:
    def __init__(
        self,
        mode,
        prompt_path,
        llm_model,
        api_key=None,
        config: Mapping[str, Any] | None = None,
    ):
        self.mode = normalize_provider_name(mode)
        self.prompt_path = prompt_path
        self.llm_model = llm_model
        self.prompt = load_prompt_text(prompt_path)
        self.client = create_client(self.mode, config=config, api_key=api_key)
        self.resolved_prompt = resolve_prompt_reference(prompt_path)
        logging.info(
            "[%s] Initialized prompt from %s for provider=%s model=%s",
            self.__class__.__name__,
            self.resolved_prompt,
            self.mode,
            self.llm_model,
        )

    def __call__(self, msg: str, **kwargs) -> str:
        attr_msg = [{"role": "user", "content": self.prompt + "\n\n" + msg}]
        return llm_completion(self.mode, self.client, attr_msg, self.llm_model, **kwargs)

    def __str__(self):
        return self.prompt


class PromptGPT(PromptBase):
    pass


class PromptDeepSeek(PromptBase):
    pass


class PromptTogether(PromptBase):
    pass


class PromptSiliconFlow(PromptBase):
    pass


class PromptGemini(PromptBase):
    pass


def create_prompt(
    mode: str,
    prompt_path: str | Path,
    llm_model: str,
    api_key: str | None = None,
    config: Mapping[str, Any] | None = None,
):
    normalized_mode = normalize_provider_name(mode)
    prompt_cls_map = {
        "cgpt": PromptGPT,
        "deepseek": PromptDeepSeek,
        "together": PromptTogether,
        "siliconflow": PromptSiliconFlow,
        "gemini": PromptGemini,
    }
    prompt_cls = prompt_cls_map.get(normalized_mode, PromptBase)
    return prompt_cls(
        normalized_mode,
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
) -> dict[str, PromptBase]:
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
    "PromptBase",
    "PromptGPT",
    "PromptDeepSeek",
    "PromptTogether",
    "PromptSiliconFlow",
    "PromptGemini",
    "create_prompt",
    "create_prompt_map",
    "get_api_key",
    "load_prompt_text",
    "resolve_prompt_reference",
]

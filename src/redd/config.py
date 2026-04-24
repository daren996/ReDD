from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

MODULE_ALIASES = {
    "schema": "schemagen",
    "schema_gen": "schemagen",
    "schema_generation": "schemagen",
    "schemagen": "schemagen",
    "data_pop": "datapop",
    "data_population": "datapop",
    "datapop": "datapop",
    "eval": "eval",
    "evaluation": "eval",
    "correction": "correction",
    "classifier": "correction",
}

SHARED_CONFIG_KEYS = {
    "default",
    "defaults",
    "default_config",
    "default_settings",
    "shared",
    "shared_config",
    "shared_settings",
    "runtime",
}

MODULE_SECTION_KEYS = {
    "schemagen",
    "schema_gen",
    "schema_generation",
    "datapop",
    "data_pop",
    "data_population",
    "eval",
    "evaluation",
    "correction",
    "classifier",
}

API_KEY_ENV_VARS = {
    "cgpt": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "together": "TOGETHER_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
}


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_yaml(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    import yaml

    resolved_path = resolve_repo_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config, resolved_path


def canonicalize_module_name(module: str | None) -> str | None:
    if module is None:
        return None
    normalized = module.strip().lower().replace("-", "_")
    return MODULE_ALIASES.get(normalized, normalized)


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _join_dataset_lists(exp_dn_list: list[str], exp_fn_list: list[str]) -> list[str]:
    if not exp_dn_list or not exp_fn_list:
        return []

    if len(exp_dn_list) == 1:
        return [f"{exp_dn_list[0]}/{fn}" for fn in exp_fn_list]
    if len(exp_fn_list) == 1:
        return [f"{dn}/{exp_fn_list[0]}" for dn in exp_dn_list]
    if len(exp_dn_list) == len(exp_fn_list):
        return [f"{dn}/{fn}" for dn, fn in zip(exp_dn_list, exp_fn_list)]

    raise ValueError(
        "Cannot derive `exp_dn_fn_list` from mismatched `exp_dn_list` and "
        f"`exp_fn_list`: {len(exp_dn_list)} vs {len(exp_fn_list)}"
    )


def _slugify_token(value: Any) -> str:
    text = str(value).strip()
    allowed = {"-", "_", "."}
    return "".join(ch for ch in text if ch.isalnum() or ch in allowed) or "default"


def _normalize_legacy_doc_filter_keys(value: Any) -> Any:
    """Map legacy doc-filter config keys to the current names recursively."""
    if isinstance(value, Mapping):
        normalized = {key: _normalize_legacy_doc_filter_keys(val) for key, val in value.items()}
        if "doc_filter" not in normalized and "chunk_filter" in normalized:
            normalized["doc_filter"] = deepcopy(normalized["chunk_filter"])
        if "doc_filtering" not in normalized and "chunk_filtering" in normalized:
            normalized["doc_filtering"] = deepcopy(normalized["chunk_filtering"])
        return normalized
    if isinstance(value, list):
        return [_normalize_legacy_doc_filter_keys(item) for item in value]
    return deepcopy(value)


def _build_res_param_str(config: Mapping[str, Any]) -> str:
    llm_model = _slugify_token(config.get("llm_model", config.get("mode", "unknown-model")))
    parts = [f"mdl{llm_model}"]

    prompt_config = config.get("prompt", {})
    prompt_version = None
    if isinstance(prompt_config, Mapping):
        prompt_version = prompt_config.get("prompt_version") or prompt_config.get("version")
    if prompt_version:
        parts.append(f"prm{_slugify_token(prompt_version)}")

    temperature = config.get("temperature")
    if temperature is None and isinstance(prompt_config, Mapping):
        temperature = prompt_config.get("temperature")
    if temperature is not None:
        parts.append(f"tmp{_slugify_token(temperature)}")

    top_p = config.get("top_p")
    if top_p is None and isinstance(prompt_config, Mapping):
        top_p = prompt_config.get("top_p")
    if top_p is not None:
        parts.append(f"tpp{_slugify_token(top_p)}")

    return "_".join(parts)


def _extract_root_defaults(full_config: Mapping[str, Any], exp: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key, value in full_config.items():
        if key == exp:
            continue
        if key in SHARED_CONFIG_KEYS and isinstance(value, Mapping):
            defaults = _merge_dicts(defaults, value)
        elif not isinstance(value, Mapping):
            defaults[key] = deepcopy(value)
    return defaults


def _extract_experiment_config(
    full_config: Mapping[str, Any],
    exp: str,
    module: str | None,
) -> dict[str, Any]:
    if exp in full_config:
        exp_config = full_config[exp]
    else:
        module_name = canonicalize_module_name(module)
        legacy_exp = f"{exp}_{module_name}" if module_name else exp
        if legacy_exp not in full_config:
            raise KeyError(f"Experiment `{exp}` not found in config")
        exp_config = full_config[legacy_exp]

    if not isinstance(exp_config, Mapping):
        raise TypeError(f"Experiment `{exp}` must map to a dictionary")

    defaults = _extract_root_defaults(full_config, exp)
    module_name = canonicalize_module_name(module)

    module_section = None
    if module_name:
        candidate_keys = [
            section_key
            for section_key in MODULE_SECTION_KEYS
            if canonicalize_module_name(section_key) == module_name
        ]
        for candidate_key in candidate_keys:
            candidate_value = exp_config.get(candidate_key)
            if isinstance(candidate_value, Mapping):
                module_section = candidate_value
                break

    if module_section is not None:
        experiment_common = {
            key: deepcopy(value)
            for key, value in exp_config.items()
            if key not in MODULE_SECTION_KEYS
        }
        return _merge_dicts(
            _merge_dicts(defaults, experiment_common),
            module_section,
        )

    return _merge_dicts(defaults, exp_config)


def normalize_experiment_config(config: Mapping[str, Any], module: str | None = None) -> dict[str, Any]:
    normalized = _normalize_legacy_doc_filter_keys(config)
    module_name = canonicalize_module_name(module)

    if "exp_dn_fn_list" not in normalized:
        exp_dataset_task = normalized.get("exp_dataset_task")
        if isinstance(exp_dataset_task, str):
            normalized["exp_dn_fn_list"] = [exp_dataset_task]
        elif isinstance(exp_dataset_task, list):
            normalized["exp_dn_fn_list"] = list(exp_dataset_task)

    if "exp_dn_fn_list" not in normalized:
        exp_dn_list = list(normalized.get("exp_dn_list") or [])
        exp_fn_list = list(normalized.get("exp_fn_list") or [])
        if exp_dn_list and exp_fn_list:
            normalized["exp_dn_fn_list"] = _join_dataset_lists(exp_dn_list, exp_fn_list)

    mode = str(normalized.get("mode", "")).strip().lower()
    if not mode and normalized.get("llm_model_path"):
        mode = "local"

    mode_aliases = {
        "openai": "cgpt",
        "gpt": "cgpt",
        "chatgpt": "cgpt",
        "deepcogito": "local",
        "cogito32b": "local",
    }
    mode = mode_aliases.get(mode, mode)
    if mode:
        normalized["mode"] = mode

    mode_settings = normalized.get(mode)
    if isinstance(mode_settings, Mapping):
        if "prompt" not in normalized and "prompt_path" in mode_settings:
            normalized["prompt"] = deepcopy(dict(mode_settings))

        normalized.setdefault("temperature", mode_settings.get("temperature"))
        normalized.setdefault("top_p", mode_settings.get("top_p"))

    prompt_config = normalized.get("prompt")
    if (
        module_name == "schemagen"
        and isinstance(prompt_config, Mapping)
        and "general_prompt_version" in prompt_config
        and "general_param_str" not in normalized
    ):
        prompt_for_general = deepcopy(dict(prompt_config))
        prompt_for_general["prompt_version"] = prompt_config["general_prompt_version"]
        normalized["general_param_str"] = _build_res_param_str(
            {"llm_model": normalized.get("llm_model"), "prompt": prompt_for_general},
        )

    if "res_param_str" not in normalized:
        normalized["res_param_str"] = _build_res_param_str(normalized)

    normalized.setdefault("log_dir", str(DEFAULT_LOG_DIR))

    return normalized


def normalize_optional_experiment_config(
    config: Mapping[str, Any] | None,
    *,
    module: str | None = None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    return normalize_experiment_config(config, module=module)


def resolve_api_key(config: Mapping[str, Any], mode: str, api_key: str | None = None) -> str:
    if api_key:
        return api_key

    if config.get("api_key"):
        return str(config["api_key"])

    env_var = API_KEY_ENV_VARS.get(mode)
    if env_var:
        env_value = os.getenv(env_var)
        if env_value:
            return env_value

    raise ValueError(
        f"API key is required for mode `{mode}`. "
        f"Provide --api-key, config.api_key, or environment variable {env_var}."
    )


def load_experiment_config(
    config_path: str | Path,
    exp: str,
    *,
    module: str | None = None,
    normalize: bool = True,
) -> tuple[dict[str, Any], Path]:
    config, resolved_path = load_yaml(config_path)
    exp_config = _extract_experiment_config(config, exp, module=module)
    if normalize:
        exp_config = normalize_experiment_config(exp_config, module=module)
    return exp_config, resolved_path

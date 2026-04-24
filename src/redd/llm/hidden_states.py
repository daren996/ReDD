"""
Utilities for matching and saving hidden states from model generation.

This module provides functions to:
- Match attribute values in token streams
- Save hidden states for classifier training
"""

from __future__ import annotations

import ast
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight environments
    torch = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required for hidden-state workflows. "
            "Install the optional local/gpu dependencies before using redd.llm.hidden_states."
        )
    return torch


class HiddenStatesManager:
    """Manage matching and persistence of hidden states from LLM generation."""

    def __init__(self, tokenizer, save_pooled: bool = False):
        self.tokenizer = tokenizer
        self.save_pooled = save_pooled

    def save_span_hs(
        self,
        span_text: str,
        token_info_pairs: List[Dict[str, Any]],
        out_path: Path,
        full_token_info: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        if self._try_span_match(span_text, token_info_pairs, out_path):
            return True
        if full_token_info is not None and full_token_info is not token_info_pairs:
            logging.debug(
                "[%s:save_span_hs] Subset match failed, trying full token stream",
                self.__class__.__name__,
            )
            if self._try_span_match(span_text, full_token_info, out_path):
                return True
        return False

    def _try_span_match(
        self,
        span_text: str,
        token_info_pairs: List[Dict[str, Any]],
        out_path: Path,
    ) -> bool:
        if not token_info_pairs:
            return False

        if span_text.startswith("[") and span_text.endswith("]"):
            try:
                parsed = ast.literal_eval(span_text)
                if isinstance(parsed, list) and len(parsed) > 0:
                    if self._try_list_element_match(parsed, token_info_pairs, out_path):
                        return True
            except (ValueError, SyntaxError):
                pass

        span_variants = self._generate_span_variants(span_text)
        gen_ids = [info["token_id"] for info in token_info_pairs]
        for variant in span_variants:
            target_ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if not target_ids:
                continue
            for idx in range(len(gen_ids) - len(target_ids) + 1):
                if gen_ids[idx : idx + len(target_ids)] == target_ids:
                    self._save_matched_hs(token_info_pairs[idx : idx + len(target_ids)], out_path)
                    return True

        token_texts = [self._decode_token_text(info["token_text"]) for info in token_info_pairs]
        normalized_span = normalize_text(span_text)
        for start in range(len(token_texts)):
            acc = ""
            for end in range(start, len(token_texts)):
                acc += token_texts[end]
                if compare_values(acc, span_text):
                    self._save_matched_hs(token_info_pairs[start : end + 1], out_path)
                    return True
                if len(normalize_text(acc)) > len(normalized_span) + 20:
                    break
        return False

    def _try_list_element_match(
        self,
        elements: List[Any],
        token_info_pairs: List[Dict[str, Any]],
        out_path: Path,
    ) -> bool:
        all_matched_tokens = []
        gen_ids = [info["token_id"] for info in token_info_pairs]
        token_texts = [self._decode_token_text(info["token_text"]) for info in token_info_pairs]
        used_indices: set[int] = set()

        for elem in elements:
            elem_str = str(elem) if not isinstance(elem, str) else elem
            elem_matched = False

            for variant in [elem_str, f'"{elem_str}"']:
                target_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if not target_ids:
                    continue
                for idx in range(len(gen_ids) - len(target_ids) + 1):
                    if any(i in used_indices for i in range(idx, idx + len(target_ids))):
                        continue
                    if gen_ids[idx : idx + len(target_ids)] == target_ids:
                        for i in range(idx, idx + len(target_ids)):
                            used_indices.add(i)
                        all_matched_tokens.extend(token_info_pairs[idx : idx + len(target_ids)])
                        elem_matched = True
                        break
                if elem_matched:
                    break

            if not elem_matched:
                normalized_elem = normalize_text(elem_str)
                for start in range(len(token_texts)):
                    if start in used_indices:
                        continue
                    acc = ""
                    for end in range(start, len(token_texts)):
                        if end in used_indices:
                            break
                        acc += token_texts[end]
                        if compare_values(acc, elem_str):
                            for i in range(start, end + 1):
                                used_indices.add(i)
                            all_matched_tokens.extend(token_info_pairs[start : end + 1])
                            elem_matched = True
                            break
                        if len(normalize_text(acc)) > len(normalized_elem) + 10:
                            break
                    if elem_matched:
                        break

            if not elem_matched:
                return False

        if all_matched_tokens:
            self._save_matched_hs(all_matched_tokens, out_path)
            return True
        return False

    def _generate_span_variants(self, span_text: str) -> List[str]:
        variants = [span_text]
        if span_text.lower() in ["true", "false"]:
            variants.extend([span_text.lower(), span_text.capitalize()])
        if span_text.startswith("[") and span_text.endswith("]"):
            try:
                parsed = ast.literal_eval(span_text)
                if isinstance(parsed, list):
                    json_variant = json.dumps(parsed, ensure_ascii=False)
                    if json_variant not in variants:
                        variants.append(json_variant)
            except (ValueError, SyntaxError):
                pass
        return variants

    def _save_matched_hs(
        self,
        matched_tokens: List[Dict[str, Any]],
        out_path: Path,
    ) -> None:
        torch_module = _require_torch()
        if self.save_pooled:
            torch_module.save(pool_hidden_states(matched_tokens), out_path)
        else:
            torch_module.save(matched_tokens, out_path)

    @staticmethod
    def _decode_token_text(token_text) -> str:
        if isinstance(token_text, bytes):
            return token_text.decode("utf-8", errors="replace")
        return str(token_text)


def pool_hidden_states(token_info_pairs: List[Dict[str, Any]]) -> Any:
    """Mean+max pool hidden states across matched tokens."""
    torch_module = _require_torch()
    all_hs = torch_module.stack([info["hidden_states"] for info in token_info_pairs])
    mean_pooled = torch_module.mean(all_hs, dim=0)
    max_pooled = torch_module.max(all_hs, dim=0)[0]
    return torch_module.cat([mean_pooled, max_pooled], dim=-1)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


def _parse_json_like(value: str) -> Any:
    value = value.strip()
    if not value:
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except (json.JSONDecodeError, ValueError, SyntaxError, TypeError):
            continue
    return value


def compare_values(val1: str, val2: str) -> bool:
    norm1 = normalize_text(val1)
    norm2 = normalize_text(val2)
    if norm1 == norm2:
        return True
    return _parse_json_like(val1) == _parse_json_like(val2)

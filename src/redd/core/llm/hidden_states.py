"""
Utilities for matching and saving hidden states from model generation.

This module provides functions to:
- Match attribute values in token streams
- Save hidden states for classifier training
"""

import ast
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class HiddenStatesManager:
    """
    Manages matching and saving of hidden states from LLM generation.
    
    Used by local extraction support workflows to save hidden states for
    downstream classifier training.
    """
    
    def __init__(self, tokenizer, save_pooled: bool = False):
        """
        Initialize the hidden states manager.
        
        Args:
            tokenizer: The tokenizer used for encoding text to token IDs
            save_pooled: If True, save pooled (mean+max) hidden states instead of raw tokens
        """
        self.tokenizer = tokenizer
        self.save_pooled = save_pooled
    
    def save_span_hs(
            self, 
            span_text: str, 
            token_info_pairs: List[Dict[str, Any]], 
            out_path: Path,
            full_token_info: Optional[List[Dict[str, Any]]] = None
        ) -> bool:
        """
        Attempt to save the hidden states corresponding to span_text.
        
        First try exact token-ID matching; if that fails, fall back to
        concatenated token_text matching to handle merged-punctuation cases.
        
        If matching fails on token_info_pairs and full_token_info is provided,
        retry on the full token stream as a fallback (in case span indices were off).
        
        Args:
            span_text: The text to match in the token stream
            token_info_pairs: Token info pairs (subset, e.g., JSON block tokens)
            out_path: Path to save the hidden states
            full_token_info: Optional full token stream for fallback matching
            
        Returns:
            True if matched and saved successfully, False otherwise
        """
        # Try matching on the provided subset first
        if self._try_span_match(span_text, token_info_pairs, out_path):
            return True
        
        # Fallback: try matching on the full token stream if provided and different
        if full_token_info is not None and full_token_info is not token_info_pairs:
            logging.debug(f"[{self.__class__.__name__}:save_span_hs] Subset match failed, trying full token stream")
            if self._try_span_match(span_text, full_token_info, out_path):
                return True
        
        return False
    
    def _try_span_match(
            self, 
            span_text: str, 
            token_info_pairs: List[Dict[str, Any]], 
            out_path: Path
        ) -> bool:
        """
        Try to match span_text in token_info_pairs and save if found.
        
        Returns:
            True if matched and saved, False otherwise
        """
        if not token_info_pairs:
            return False
        
        # Special handling for list values: match each element individually
        # This handles multi-line JSON arrays where elements are on separate lines
        if span_text.startswith("[") and span_text.endswith("]"):
            try:
                parsed = ast.literal_eval(span_text)
                if isinstance(parsed, list) and len(parsed) > 0:
                    matched = self._try_list_element_match(parsed, token_info_pairs, out_path)
                    if matched:
                        return True
            except (ValueError, SyntaxError):
                pass
            
        # Generate variants of span_text for matching
        span_variants = self._generate_span_variants(span_text)
        
        # 1) Exact token-ID matching (try all variants)
        gen_ids = [info["token_id"] for info in token_info_pairs]
        for variant in span_variants:
            target_ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if not target_ids:
                continue
            for idx in range(len(gen_ids) - len(target_ids) + 1):
                if gen_ids[idx : idx + len(target_ids)] == target_ids:
                    matched_tokens = token_info_pairs[idx : idx + len(target_ids)]
                    self._save_matched_hs(matched_tokens, out_path)
                    return True

        # 2) Concatenated token_text matching with normalization
        t_texts = [self._decode_token_text(info["token_text"]) for info in token_info_pairs]
        n = len(t_texts)
        normalized_span = normalize_text(span_text)
        for start in range(n):
            acc = ""
            for end in range(start, n):
                acc += t_texts[end]
                if compare_values(acc, span_text):
                    matched_tokens = token_info_pairs[start : end + 1]
                    self._save_matched_hs(matched_tokens, out_path)
                    return True
                # Early termination: if accumulated text is already longer than target
                if len(normalize_text(acc)) > len(normalized_span) + 20:
                    break

        return False
    
    def _try_list_element_match(
            self,
            elements: List[Any],
            token_info_pairs: List[Dict[str, Any]],
            out_path: Path
        ) -> bool:
        """
        Match each element of a list individually and collect their hidden states.
        
        Handles multi-line JSON arrays where elements are formatted on separate lines.
        
        Returns:
            True if all elements are matched, False otherwise
        """
        all_matched_tokens = []
        gen_ids = [info["token_id"] for info in token_info_pairs]
        t_texts = [self._decode_token_text(info["token_text"]) for info in token_info_pairs]
        
        used_indices = set()  # Track which tokens have been matched
        
        for elem in elements:
            elem_str = str(elem) if not isinstance(elem, str) else elem
            elem_matched = False
            
            # Try exact token-ID matching first
            # Generate variants for the element (e.g., with/without quotes)
            elem_variants = [elem_str, f'"{elem_str}"']
            for variant in elem_variants:
                target_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if not target_ids:
                    continue
                for idx in range(len(gen_ids) - len(target_ids) + 1):
                    # Skip if overlapping with already matched tokens
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
            
            # Fallback: try text matching
            if not elem_matched:
                normalized_elem = normalize_text(elem_str)
                for start in range(len(t_texts)):
                    if start in used_indices:
                        continue
                    acc = ""
                    for end in range(start, len(t_texts)):
                        if end in used_indices:
                            break
                        acc += t_texts[end]
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
                # If any element fails to match, fail the entire list match
                return False
        
        # All elements matched successfully
        if all_matched_tokens:
            self._save_matched_hs(all_matched_tokens, out_path)
            return True
        return False
    
    def _generate_span_variants(self, span_text: str) -> List[str]:
        """
        Generate multiple variants of span_text for matching.
        
        Handles: boolean case differences, list/array format differences, etc.
        """
        variants = [span_text]
        
        # Handle boolean case: JSON uses lowercase (true/false), Python str() uses titlecase (True/False)
        if span_text.lower() in ["true", "false"]:
            variants.extend([span_text.lower(), span_text.capitalize()])
        
        # Handle list/array values: convert Python list repr to JSON format
        # e.g., "['a', 'b']" -> '["a", "b"]'
        if span_text.startswith("[") and span_text.endswith("]"):
            try:
                # Try to parse as Python literal and convert to JSON
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
            out_path: Path
        ) -> None:
        """
        Save matched token hidden states to file.
        
        If self.save_pooled is True, applies pooling before saving.
        Otherwise, saves raw token-level data.
        """
        if self.save_pooled:
            pooled_hs = pool_hidden_states(matched_tokens)
            torch.save(pooled_hs, out_path)
        else:
            torch.save(matched_tokens, out_path)
    
    @staticmethod
    def _decode_token_text(token_text) -> str:
        """Decode token_text from bytes to string if needed."""
        if isinstance(token_text, bytes):
            return token_text.decode("utf-8")
        return token_text


# ---------------------- Utility Functions ----------------------

def pool_hidden_states(token_info_pairs: List[Dict[str, Any]]) -> torch.Tensor:
    """
    Applies mean pooling and max pooling across all tokens, then concatenates
    the results to produce a vector of size 2 * hidden_size.
    
    Args:
        token_info_pairs: List of token info dicts with "hidden_states" key
        
    Returns:
        Tensor of shape (num_layers, 2 * hidden_size)
    """
    # Stack hidden states: (num_tokens, num_layers, hidden_size)
    all_hs = torch.stack([info["hidden_states"] for info in token_info_pairs])
    # Mean pooling across tokens: (num_layers, hidden_size)
    mean_pooled = torch.mean(all_hs, dim=0)
    # Max pooling across tokens: (num_layers, hidden_size)
    max_pooled = torch.max(all_hs, dim=0)[0]
    # Concatenate: (num_layers, 2 * hidden_size)
    return torch.cat([mean_pooled, max_pooled], dim=-1)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by handling:
    - Unicode normalization (e.g., μ vs µ - Greek mu vs micro sign)
    - Escaped newlines (\\n) vs actual newlines
    - Whitespace normalization
    - Quote removal
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Unicode normalization (NFKC handles compatibility characters like µ -> μ)
    res = unicodedata.normalize("NFKC", text)
    
    # Handle escaped newlines: convert literal \n to actual newline for comparison
    # This handles cases where JSON has "\\n" but parsed value has actual newlines
    res = res.replace("\\n", "\n")
    res = res.replace("\\t", "\t")
    
    # Strip leading/trailing punctuation and whitespace
    punct = "\"'\n\t "
    res = res.lstrip(punct).rstrip(punct)
    
    # Remove escaped quotes and regular quotes
    res = res.replace("\\\"", "")
    res = res.replace("\"", "")
    res = res.replace("'", "")
    
    # Normalize internal whitespace (collapse multiple spaces/newlines)
    res = " ".join(res.split())
    
    return res


def compare_values(val1: str, val2: str) -> bool:
    """
    Compare two values for equality with normalization.
    
    Handles: null values, boolean case differences, Unicode normalization,
    newline escape differences, and whitespace normalization.
    
    Args:
        val1: First value to compare
        val2: Second value to compare
        
    Returns:
        True if values are considered equal, False otherwise
    """
    def is_null(val):
        return val is None or val in ["", "null", "NULL", "None", "none"]
    
    def is_bool(val):
        return val.lower() in ["true", "false"]

    # Normalize both values
    norm1 = normalize_text(val1)
    norm2 = normalize_text(val2)
    
    if is_null(norm1) and is_null(norm2):
        return True
    # Handle boolean comparison case-insensitively
    # JSON uses lowercase (true/false), Python str() uses titlecase (True/False)
    if is_bool(norm1) and is_bool(norm2):
        return norm1.lower() == norm2.lower()
    return norm1 == norm2

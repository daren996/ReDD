"""
Local GPU-based Chunk Attribute Mapper.

This module implements chunk attribute mapping using locally-hosted LLM models
with GPU inference.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import torch

from .base import ChunkAttributeMapperBase
from ..data_loader import DataLoaderBase

__all__ = ["ChunkAttributeMapper"]


class ChunkAttributeMapper(ChunkAttributeMapperBase):
    """
    Chunk attribute mapper using locally-hosted LLM models with GPU inference.
    
    Maps partial document chunks to their relevant schema attributes using
    a local LLM for inference.
    
    Config keys:
        - llm_model_path: Path to local model weights
        - llm_model: Model name for downloading if not cached
        - max_new_tokens: Maximum tokens to generate (default: 500)
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        loader: Optional[DataLoaderBase] = None
    ):
        """
        Initialize the local chunk attribute mapper.
        
        Args:
            config: Configuration dictionary
            loader: Optional pre-created DataLoader instance
        """
        super().__init__(config, loader)
        
        if not torch.cuda.is_available():
            logging.error(f"[{self.__class__.__name__}:__init__] "
                         f"CUDA unavailable. Use ChunkAttributeMapperAPI for CPU inference.")
            raise RuntimeError("CUDA is required for ChunkAttributeMapper")
        
        # Lazy imports for transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.llm_model_path = config.get("llm_model_path")
        self.llm_model_name = config.get("llm_model")
        self.max_new_tokens = config.get("max_new_tokens", 500)
        
        if not self.llm_model_path and not self.llm_model_name:
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] "
                f"Either llm_model_path or llm_model must be specified"
            )
        
        # Load model
        self._load_model()
        
        logging.info(f"[{self.__class__.__name__}:__init__] "
                    f"Initialized with model: {self.llm_model_name or self.llm_model_path}")
    
    def _load_model(self) -> None:
        """Load the LLM model for inference."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import os
        
        if self.llm_model_path and os.path.exists(self.llm_model_path):
            logging.info(f"[{self.__class__.__name__}:_load_model] "
                        f"Loading model from local path: {self.llm_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).cuda()
        else:
            logging.info(f"[{self.__class__.__name__}:_load_model] "
                        f"Downloading model: {self.llm_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).cuda()
            
            # Cache for future runs
            if self.llm_model_path:
                os.makedirs(self.llm_model_path, exist_ok=True)
                self.tokenizer.save_pretrained(self.llm_model_path)
                self.model.save_pretrained(self.llm_model_path)
    
    def _map_chunk_to_attributes(
        self, 
        chunk_text: str, 
        schema: Dict[str, Any]
    ) -> List[str]:
        """
        Use local LLM to identify which attributes are present in a chunk.
        
        Args:
            chunk_text: The text content of the chunk
            schema: Schema dict with "Schema Name" and "Attributes" keys
            
        Returns:
            List of attribute names that are present/relevant in the chunk
        """
        # Build input
        input_json = self._build_prompt_input(chunk_text, schema)
        full_prompt = self.prompt_template + "\n\n" + input_json
        
        # Generate response
        messages = [{"role": "user", "content": full_prompt}]
        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        
        attention_mask = torch.ones_like(input_tensor)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        
        gen_tokens = outputs[0][input_tensor.shape[1]:]
        response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        # Parse response
        return self._parse_response(response_text, schema)
    
    def _parse_response(
        self, 
        response_text: str, 
        schema: Dict[str, Any]
    ) -> List[str]:
        """
        Parse the LLM response to extract attribute names.
        
        Args:
            response_text: Raw LLM response
            schema: Schema dict for validation
            
        Returns:
            List of valid attribute names
        """
        # Get valid attribute names from schema
        raw_attrs = schema.get("Attributes") or schema.get("columns", [])
        valid_attrs = set()
        for attr in raw_attrs:
            if isinstance(attr, dict):
                attr_name = attr.get("Attribute Name") or attr.get("name", "")
                if attr_name:
                    valid_attrs.add(attr_name)
            elif isinstance(attr, str):
                valid_attrs.add(attr)
        
        # Try to parse JSON from response
        relevant_attrs = []
        
        # Try JSON block extraction
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.S | re.I)
        if not json_match:
            json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.S)
        
        json_str = None
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON object directly
            obj_match = re.search(r'\{[^{}]*"relevant_attributes"[^{}]*\}', response_text, re.S)
            if obj_match:
                json_str = obj_match.group(0)
        
        if json_str:
            try:
                result = json.loads(json_str)
                if isinstance(result, dict):
                    attrs = result.get("relevant_attributes", [])
                    if isinstance(attrs, list):
                        relevant_attrs = [a for a in attrs if a in valid_attrs]
            except json.JSONDecodeError:
                pass
        
        # Fallback: look for attribute names in response
        if not relevant_attrs:
            for attr_name in valid_attrs:
                # Check if attribute name appears in response (case-insensitive)
                if re.search(rf'\b{re.escape(attr_name)}\b', response_text, re.I):
                    relevant_attrs.append(attr_name)
        
        return relevant_attrs
    
    def __str__(self) -> str:
        """String representation of the mapper."""
        model_info = self.llm_model_name or self.llm_model_path
        return f"{self.__class__.__name__}(model={model_info})"

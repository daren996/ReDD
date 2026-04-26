"""
API-based Chunk Attribute Mapper.

This module implements chunk attribute mapping using remote LLM APIs
(DeepSeek, SiliconFlow, Together, OpenAI-compatible endpoints).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from redd.llm import CompletionRequest, LLMRuntime

from ..data_loader import DataLoaderBase
from ..utils.api_keys import get_api_key_for_mode
from .base import ChunkAttributeMapperBase

__all__ = ["ChunkAttributeMapperAPI"]


class ChunkAttributeMapperAPI(ChunkAttributeMapperBase):
    """
    Chunk attribute mapper using remote LLM APIs.
    
    Supports OpenAI-compatible API endpoints including:
    - DeepSeek
    - SiliconFlow
    - Together AI
    - OpenAI
    
    Config keys:
        - llm_model: Model name (e.g., "deepseek-chat", "Qwen/Qwen2.5-72B-Instruct")
        - api_key: API key for authentication (or set via environment variable)
        - base_url: API base URL (e.g., "https://api.deepseek.com/v1")
    """
    
    # Common API endpoints
    API_ENDPOINTS = {
        "deepseek": "https://api.deepseek.com/v1",
        "siliconflow": "https://api.siliconflow.cn/v1",
        "together": "https://api.together.xyz/v1",
        "openai": "https://api.openai.com/v1",
    }
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        loader: Optional[DataLoaderBase] = None
    ):
        """
        Initialize the API-based chunk attribute mapper.
        
        Args:
            config: Configuration dictionary
            loader: Optional pre-created DataLoader instance
        """
        super().__init__(config, loader)
        
        self.llm_model = config.get("llm_model")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        
        # Validate required config
        if not self.llm_model:
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] llm_model must be specified"
            )
        
        # Try to get API key from config, api_keys.json, or environment
        if not self.api_key:
            mode = config.get("mode", "deepseek").lower()
            # Map "api" mode to actual provider based on base_url or default to deepseek
            if mode == "api":
                mode = "deepseek"
            self.api_key = get_api_key_for_mode(mode)
        
        if not self.api_key:
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] api_key must be specified "
                f"in config, api_keys.json, or environment variables"
            )
        
        # Determine base URL
        if not self.base_url:
            mode = config.get("mode", "").lower()
            self.base_url = self.API_ENDPOINTS.get(mode, self.API_ENDPOINTS["deepseek"])
        
        # Initialize shared runtime
        self._init_client()
        
        logging.info(f"[{self.__class__.__name__}:__init__] "
                    f"Initialized with model: {self.llm_model}, base_url: {self.base_url}")
    
    def _init_client(self) -> None:
        """Initialize the shared LLM runtime."""
        mode = self.config.get("mode", "deepseek").lower()
        if mode == "api":
            mode = "deepseek"
        runtime_config = dict(self.config)
        runtime_config["llm_model"] = self.llm_model
        runtime_config["base_url"] = self.base_url
        self.client = LLMRuntime.from_config(
            mode,
            self.llm_model,
            config=runtime_config,
            api_key=self.api_key,
        )
    
    def _map_chunk_to_attributes(
        self, 
        chunk_text: str, 
        schema: Dict[str, Any]
    ) -> List[str]:
        """
        Use API to identify which attributes are present in a chunk.
        
        Args:
            chunk_text: The text content of the chunk
            schema: Schema dict with "Schema Name" and "Attributes" keys
            
        Returns:
            List of attribute names that are present/relevant in the chunk
        """
        # Build input
        input_json = self._build_prompt_input(chunk_text, schema)
        full_prompt = self.prompt_template + "\n\n" + input_json
        
        # Make API call
        response_text = self._call_api(full_prompt)
        
        if not response_text:
            logging.warning(f"[{self.__class__.__name__}:_map_chunk_to_attributes] "
                          f"Empty response from API")
            return []
        
        # Parse response
        return self._parse_response(response_text, schema)
    
    def _call_api(self, prompt: str) -> Optional[str]:
        """
        Call the API.
        
        Args:
            prompt: The full prompt to send
            
        Returns:
            Response text or None if call failed
        """
        try:
            return self.client.complete_text(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    response_format="text",
                )
            ).text
            
        except Exception as e:
            logging.error(f"[{self.__class__.__name__}:_call_api] "
                         f"API call failed: {e}")
            return None
    
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
        
        relevant_attrs = []
        
        # Try to parse JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.S | re.I)
        if not json_match:
            json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.S)
        
        json_str = None
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON object directly
            obj_match = re.search(r'\{[^{}]*"relevant_attributes"[^{}]*\}', response_text, re.S)
            if not obj_match:
                # Try a more permissive pattern
                obj_match = re.search(r'\{.*?\}', response_text, re.S)
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
                logging.debug(f"[{self.__class__.__name__}:_parse_response] "
                            f"Failed to parse JSON: {json_str[:100]}...")
        
        # Fallback: look for attribute names in response
        if not relevant_attrs:
            for attr_name in valid_attrs:
                # Check if attribute name appears in response
                if re.search(rf'\b{re.escape(attr_name)}\b', response_text, re.I):
                    relevant_attrs.append(attr_name)
        
        return relevant_attrs
    
    def __str__(self) -> str:
        """String representation of the mapper."""
        return f"{self.__class__.__name__}(model={self.llm_model}, base_url={self.base_url})"

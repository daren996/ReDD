"""
Oracle for proxy runtime using LLM-based data extraction or golden attributes.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from redd.core.utils.structured_outputs import AttributeExtractionOutput


class GoldenOracle:
    """
    Oracle that returns golden attributes from get_doc_info (no LLM).
    Used for filter-accuracy evaluation: proxy filtering still runs, but extraction uses ground truth.
    """

    def __init__(self, data_loader: Any):
        self.data_loader = data_loader
        self._extract_count = 0

    def extract(
        self,
        document: str,
        schema: Dict[str, Any],
        attributes: List[str],
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return golden attribute values from get_doc_info.
        Requires doc_id (passed by ProxyPipeline when use_oracle_extraction).
        """
        self._extract_count += 1
        if not doc_id:
            logging.warning("[GoldenOracle] extract called without doc_id, returning empty")
            return {attr: None for attr in attributes}

        doc_info = self.data_loader.get_doc_info(doc_id)
        if not doc_info:
            return {attr: None for attr in attributes}

        merged = {}
        for rec in doc_info.get("data_records", []) or []:
            data = rec.get("data", {})
            if isinstance(data, dict):
                merged.update(data)
        if "row_id" not in merged and doc_info.get("source_row_id") is not None:
            merged["row_id"] = doc_info.get("source_row_id")

        result = {}
        for attr in attributes:
            result[attr] = merged.get(attr)
        return result

    def check_predicates(
        self,
        extracted_values: Dict[str, Any],
        predicates: Dict[str, Callable[[Any], bool]],
    ) -> Tuple[bool, Dict[str, bool]]:
        per_attr = {}
        all_passed = True
        for attr_name, predicate_fn in predicates.items():
            value = extracted_values.get(attr_name)
            try:
                passed = predicate_fn(value) if value is not None else False
            except Exception:
                passed = False
            per_attr[attr_name] = passed
            if not passed:
                all_passed = False
        return all_passed, per_attr

    @property
    def call_count(self) -> int:
        return self._extract_count


class DataExtractionOracle:
    """
    LLM Oracle wrapper that uses the existing data-extraction runtime.
    
    Implements the LLMOracleProtocol for use with ProxyExecutor.
    """
    
    def __init__(
        self,
        mode: str = "gemini",
        llm_model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        prompt_attr_path: str = "prompts/datapop_attr_json.txt"
    ):
        """
        Initialize data-extraction oracle.
        
        Args:
            mode: LLM provider mode
            llm_model: Model name
            api_key: Optional API key
            prompt_attr_path: Path to attribute extraction prompt
        """
        self.mode = mode
        self.llm_model = llm_model
        self.api_key = api_key
        
        # Resolve prompt path
        if not Path(prompt_attr_path).exists():
            # Try finding it relative to project root (core/utils -> core -> root)
            project_root = Path(__file__).parent.parent.parent
            alt_path = project_root / prompt_attr_path
            if alt_path.exists():
                prompt_attr_path = str(alt_path)
                logging.info(f"[DataExtractionOracle] Resolved prompt path to: {prompt_attr_path}")
        
        self.prompt_attr_path = prompt_attr_path
        
        # Lazy-load prompt
        self._prompt_attr = None
        self._extract_count = 0
    
    def _ensure_prompt(self):
        """Lazy-load the prompt object."""
        if self._prompt_attr is not None:
            return
        
        from redd.core.utils.prompt_utils import create_prompt
        from redd.llm import get_api_key
        
        # Get API key
        config = {"mode": self.mode, "llm_model": self.llm_model}
        resolved_key = get_api_key(config, self.mode, self.api_key)
        
        self._prompt_attr = create_prompt(
            self.mode,
            self.prompt_attr_path,
            llm_model=self.llm_model,
            api_key=resolved_key,
            config=config,
        )
        
        logging.info(f"[DataExtractionOracle] Initialized with mode={self.mode}, model={self.llm_model}")
    
    def extract(
        self,
        document: str,
        schema: Dict[str, Any],
        attributes: List[str],
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract attribute values from a document using LLM.
        
        Args:
            document: Document text
            schema: Schema definition
            attributes: List of attributes to extract
            doc_id: Optional (ignored by LLM oracle; used by GoldenOracle)
            
        Returns:
            Dictionary mapping attribute names to extracted values
        """
        self._ensure_prompt()
        self._extract_count += 1
        
        # Build extraction input
        extract_input = {
            "Document": document,
            "Schema": schema,
            "Target Attributes": attributes
        }
        
        try:
            extracted = self._prompt_attr.complete_model(
                json.dumps(extract_input, ensure_ascii=False),
                AttributeExtractionOutput,
            ).root
            return extracted
            
        except Exception as e:
            logging.warning(f"[DataExtractionOracle] Extraction failed: {e}")
            return {attr: None for attr in attributes}
    
    def _parse_extraction_response(
        self, 
        response: str, 
        attributes: List[str]
    ) -> Dict[str, Any]:
        """Parse LLM response into attribute values."""
        import ast
        import re
        
        # Try to extract JSON from response
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    try:
                        parsed = ast.literal_eval(match)
                        if isinstance(parsed, dict):
                            return parsed
                    except (ValueError, SyntaxError):
                        pass
        
        # Try parsing entire response as JSON
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Return empty dict if parsing fails
        logging.warning(f"[DataExtractionOracle] Could not parse response: {response[:200]}...")
        return {attr: None for attr in attributes}
    
    def check_predicates(
        self,
        extracted_values: Dict[str, Any],
        predicates: Dict[str, Callable[[Any], bool]]
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if extracted values satisfy predicates.
        
        Args:
            extracted_values: Extracted attribute values
            predicates: Dictionary mapping attribute names to predicate functions
            
        Returns:
            (all_passed, per_attribute_results) tuple
        """
        per_attr = {}
        all_passed = True
        
        for attr_name, predicate_fn in predicates.items():
            value = extracted_values.get(attr_name)
            
            try:
                passed = predicate_fn(value) if value is not None else False
            except Exception as e:
                logging.debug(f"[DataExtractionOracle] Predicate check failed for {attr_name}: {e}")
                passed = False
            
            per_attr[attr_name] = passed
            if not passed:
                all_passed = False
        
        return all_passed, per_attr
    
    @property
    def call_count(self) -> int:
        """Number of extraction calls made."""
        return self._extract_count

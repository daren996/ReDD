"""
Oracle for proxy runtime using LLM-based data extraction or golden attributes.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from redd.core.utils.constants import RESULT_DATA_KEY, SCHEMA_NAME_KEY
from redd.core.utils.extraction_records import active_result_records
from redd.core.utils.prompt_registry import DATA_EXTRACTION_ATTR_PROMPT_ID
from redd.core.utils.structured_outputs import AttributeExtractionOutput
from redd.core.utils.utils import is_none_value


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

        table_name = schema.get(SCHEMA_NAME_KEY) or schema.get("table") or schema.get("table_name")
        data_records = doc_info.get("data_records", []) or []
        if table_name:
            data_records = [
                rec
                for rec in data_records
                if isinstance(rec, dict)
                and (rec.get("table_name") or rec.get("table")) == table_name
            ]

        record_values: List[Dict[str, Any]] = []
        for rec in data_records:
            data = rec.get("data", {})
            if isinstance(data, dict):
                values = {attr: data.get(attr) for attr in attributes}
                if "row_id" not in values and doc_info.get("source_row_id") is not None:
                    values["row_id"] = doc_info.get("source_row_id")
                record_values.append(values)
        if not record_values and isinstance(doc_info.get("data"), dict):
            record_values.append({attr: doc_info["data"].get(attr) for attr in attributes})

        result = {}
        primary_values = record_values[0] if record_values else {}
        for attr in attributes:
            result[attr] = primary_values.get(attr)
        result["__records"] = record_values
        return result

    def check_predicates(
        self,
        extracted_values: Dict[str, Any],
        predicates: Dict[str, Callable[[Any], bool]],
    ) -> Tuple[bool, Dict[str, bool]]:
        record_values = extracted_values.get("__records")
        if isinstance(record_values, list) and record_values:
            best_per_attr: Dict[str, bool] = {}
            for record in record_values:
                if not isinstance(record, dict):
                    continue
                passed, per_attr = self.check_predicates(
                    {key: value for key, value in record.items() if key != "__records"},
                    predicates,
                )
                for attr_name, attr_passed in per_attr.items():
                    best_per_attr[attr_name] = best_per_attr.get(attr_name, False) or attr_passed
                if passed:
                    return True, per_attr
            return False, {
                attr_name: best_per_attr.get(attr_name, False)
                for attr_name in predicates
            }

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


class MaterializedExtractionOracle:
    """
    Oracle backed by a query-independent full extraction artifact.

    This is used by experiment-run materialized mode: query execution still runs,
    but oracle extraction reads precomputed doc/table/attr values instead of
    calling an LLM.
    """

    def __init__(self, materialized_data: Dict[str, Any] | str | Path):
        if isinstance(materialized_data, (str, Path)):
            with Path(materialized_data).open("r", encoding="utf-8") as file:
                payload = json.load(file)
            self.materialized_data = payload if isinstance(payload, dict) else {}
        else:
            self.materialized_data = materialized_data
        self._extract_count = 0

    def extract(
        self,
        document: str,
        schema: Dict[str, Any],
        attributes: List[str],
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        del document
        self._extract_count += 1
        if not doc_id:
            logging.warning(
                "[MaterializedExtractionOracle] extract called without doc_id, returning empty"
            )
            return {attr: None for attr in attributes}

        entry = self.materialized_data.get(str(doc_id))
        if not isinstance(entry, dict):
            return {attr: None for attr in attributes}
        table_name = schema.get(SCHEMA_NAME_KEY) or schema.get("table") or schema.get("table_name")
        records = active_result_records(entry)
        if table_name:
            records = [record for record in records if record.get("table") == table_name]
        if not records:
            return {attr: None for attr in attributes}

        record_values = []
        for record in records:
            data = record.get(RESULT_DATA_KEY, {})
            if not isinstance(data, dict):
                continue
            record_values.append(
                {
                    attr: None if is_none_value(data.get(attr)) else data.get(attr)
                    for attr in attributes
                }
            )
        primary_values = record_values[0] if record_values else {}
        return {
            **{attr: primary_values.get(attr) for attr in attributes},
            "__records": record_values,
        }

    def check_predicates(
        self,
        extracted_values: Dict[str, Any],
        predicates: Dict[str, Callable[[Any], bool]],
    ) -> Tuple[bool, Dict[str, bool]]:
        record_values = extracted_values.get("__records")
        if isinstance(record_values, list) and record_values:
            best_per_attr: Dict[str, bool] = {}
            for record in record_values:
                if not isinstance(record, dict):
                    continue
                passed, per_attr = self.check_predicates(
                    {key: value for key, value in record.items() if key != "__records"},
                    predicates,
                )
                for attr_name, attr_passed in per_attr.items():
                    best_per_attr[attr_name] = best_per_attr.get(attr_name, False) or attr_passed
                if passed:
                    return True, per_attr
            return False, {
                attr_name: best_per_attr.get(attr_name, False)
                for attr_name in predicates
            }

        per_attr = {}
        all_passed = True
        for attr_name, predicate_fn in predicates.items():
            value = extracted_values.get(attr_name)
            try:
                passed = predicate_fn(value) if not is_none_value(value) else False
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
        prompt_attr_path: str = DATA_EXTRACTION_ATTR_PROMPT_ID,
        query_id: Optional[str] = None,
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
        self.query_id = query_id
        
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
                usage_context={
                    "stage": "proxy_runtime_oracle",
                    "query_id": self.query_id,
                    "doc_id": doc_id,
                    "table": schema.get(SCHEMA_NAME_KEY) or schema.get("table") or schema.get("table_name"),
                    "attributes": list(attributes),
                },
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

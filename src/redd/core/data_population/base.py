"""
Abstract Base Class for Data Population.

This module defines the abstract interface that all data population
implementations must follow.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Any, Optional


class DataPopulator(ABC):
    """
    Abstract base class for data population from unstructured documents.
    
    This class defines the interface and common utilities for data population.
    Subclasses must implement __call__ and __str__ methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data populator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = self._resolve_datapop_config(config)
        self.loader = None  # Data loader will be set during processing

    @staticmethod
    def _resolve_datapop_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize data population config.

        Supports both:
        - Flat config (legacy): top-level datapop keys.
        - Unified config: shared keys at top-level + `data_pop`/`datapop` section.
        """
        if not isinstance(config, dict):
            raise TypeError("DataPopulator config must be a dictionary")

        section_key = None
        for key in ("data_pop", "datapop"):
            if key in config:
                section_key = key
                break

        if section_key is None:
            merged_config = dict(config)
        else:
            section_config = config.get(section_key)
            if not isinstance(section_config, dict):
                raise TypeError(f"Config section '{section_key}' must be a dictionary")
            shared_config = {
                key: value
                for key, value in config.items()
                if key not in {"schema_gen", "schemagen", "data_pop", "datapop"}
            }
            merged_config = {**shared_config, **section_config}

        # Backward/forward compatibility for filter key naming
        if "doc_filter" not in merged_config and "doc_filtering" in merged_config:
            merged_config["doc_filter"] = merged_config["doc_filtering"]

        return merged_config
    
    @abstractmethod
    def __call__(self, dataset_task: Optional[str] = None):
        """
        Main entry point for data population.
        
        Args:
            dataset_task: Optional dataset/task path to process
        """
        logging.error(f"[{self.__class__.__name__}:__call__] Subclasses must implement __call__")
        raise NotImplementedError("Subclasses must implement __call__")
    
    @abstractmethod
    def __str__(self) -> str:
        """
        String representation of the data populator.
        
        Returns:
            String describing the data populator
        """
        logging.error(f"[{self.__class__.__name__}:__str__] Subclasses must implement __str__")
        raise NotImplementedError("Subclasses must implement __str__")
    
    # Common utility methods
    
    def save_results(self, res_path: str, res_dict: Dict[str, Any], encoding: str = "utf-8"):
        """
        Save results to JSON file.
        
        Args:
            res_path: Path to save results
            res_dict: Dictionary of results to save
            encoding: File encoding (default: utf-8)
        """
        out_path = Path(res_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path: Optional[Path] = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding=encoding,
                dir=out_path.parent,
                prefix=f".{out_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                json.dump(res_dict, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
                tmp_path = Path(f.name)
            # Retry on Windows PermissionError (file may be locked by antivirus/other process)
            for attempt in range(5):
                try:
                    os.replace(tmp_path, out_path)
                    break
                except PermissionError:
                    if attempt < 4:
                        time.sleep(0.2 * (attempt + 1))
                    else:
                        raise
        except Exception as error:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            logging.error(
                f"[{self.__class__.__name__}:save_results] Failed to save {out_path}: {error}"
            )
            raise
        logging.debug(f"[{self.__class__.__name__}:save_results] Saved results to {res_path}")
    
    def load_json(self, file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            file_path: Path to JSON file
            encoding: File encoding (default: utf-8)
            
        Returns:
            Dictionary loaded from JSON file
        """
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)
    
    def load_processed_res(self, res_path: str) -> Dict[str, Any]:
        """
        Load processed results from file.
        
        Args:
            res_path: Path to results file
            
        Returns:
            Dictionary of results, or empty dict if file doesn't exist or is invalid JSON
        """
        res_dict = dict()
        if os.path.exists(res_path):
            try:
                res_dict = self.load_json(res_path)
            except json.JSONDecodeError as error:
                logging.warning(
                    f"[{self.__class__.__name__}:load_processed_res] "
                    f"Invalid JSON in {res_path}; fallback to empty result. error={error}"
                )
        return res_dict

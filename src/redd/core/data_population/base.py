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
from typing import Any, Dict, Optional


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
        self.config = self._resolve_extraction_config(config)
        self.loader = None  # Data loader will be set during processing

    @staticmethod
    def _resolve_extraction_config(config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(config, dict):
            raise TypeError("DataPopulator config must be a dictionary")
        return dict(config)
    
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

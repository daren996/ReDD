"""
Schema Relevance Filter.

This filter determines whether a chunk is relevant to the query (schema)
using embedding-based similarity with conformal prediction guarantees.

Workflow:
1. Embed the query using a dense encoder
2. Augment the query embedding with relevant context (hook for future)
3. Embed each document chunk
4. Compute cosine similarity: sim(d_i, q) = cosine(E_query(q), E_chunk(d_i))
5. Determine a threshold τ_alpha using conformal prediction calibration
6. Filter: Keep only chunks where sim(d_i, q) >= τ_alpha

Conformal Prediction:
- Nonconformity score: NC(q, d_i) = -cos(E_q(q), E_d(d_i))
- Calibration set: Built from ground truth records (from dataloader)
- Threshold: Computed as (1-alpha) quantile of calibration scores
- Guarantee: With probability >= 1-alpha, relevant chunks are retained
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

from redd.core.data_loader.data_loader_hf_manifest import DataLoaderHFManifest
from redd.core.utils.conformal_calibration import (
    ConformalCalibrationResult,
    ConformalCalibrator,
    nonconformity_score_negative_cosine,
)
from redd.core.utils.conformal_calibration import (
    cosine_similarity as utils_cosine_similarity,
)
from redd.embedding import EmbeddingManager

from .base import DocFilterBase, FilterResult

__all__ = ["SchemaRelevanceFilter", "ConformalCalibrationResult"]


class SchemaRelevanceFilter(DocFilterBase):
    """
    Filter chunks based on their relevance to the query (schema).
    
    Uses embedding-based similarity with conformal prediction guarantees
    to ensure that relevant chunks are retained with high probability.
    
    Example:
        ```python
        from redd.doc_filtering import SchemaRelevanceFilter
        from redd.core.data_loader import DataLoaderHFManifest
        from redd.embedding import EmbeddingManager
        
        # Initialize
        loader = DataLoaderHFManifest("dataset/canonical/spider.bike_1")
        emb_manager = EmbeddingManager(loader=loader, model="text-embedding-3-small")
        
        # Split docs into train/test sets
        train_doc_ids = loader.doc_ids[:100]
        test_doc_ids = loader.doc_ids[100:]
        
        # Option 1: Pass dependencies at init time (recommended for repeated use)
        filter = SchemaRelevanceFilter(
            config={"target_recall": 0.95},
            data_loader=loader,
            embedding_manager=emb_manager,
            enable_calibrate=True,
            train_doc_ids=train_doc_ids,
        )
        
        # Filter (dependencies already set)
        result = filter.filter(
            query_id="Q1",
            doc_ids=test_doc_ids,
            query_text="Find all trips longer than 10 minutes",
        )
        
        # Option 2: Pass dependencies at filter time (flexible for one-off use)
        filter2 = SchemaRelevanceFilter(config={"target_recall": 0.95})
        result2 = filter2.filter(
            query_id="Q1",
            doc_ids=test_doc_ids,
            data_loader=loader,
            embedding_manager=emb_manager,
            query_text="Find all trips longer than 10 minutes",
            enable_calibrate=True,
            train_doc_ids=train_doc_ids,
        )
        
        # Use result
        for doc_id in test_doc_ids:
            if result.should_skip(doc_id):
                continue
            # Process document
        ```
    """
    
    # Default configuration values
    DEFAULT_TARGET_RECALL = 0.95
    DEFAULT_THRESHOLD = 0.585
    DEFAULT_THRESHOLD_GEMINI = 0.585
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_calibrate: bool = False,
        data_loader: Optional["DataLoaderHFManifest"] = None,
        embedding_manager: Optional["EmbeddingManager"] = None,
        train_doc_ids: Optional[List[str]] = None,
    ):
        """
        Initialize schema relevance filter.
        
        Args:
            config: Configuration dictionary. Expected keys:
                - target_recall (float): Target recall rate (default: 0.95)
                - threshold (float): Pre-calibrated threshold (optional)
                - embedding_model (str): Embedding model name (default: "text-embedding-3-small")
                - enable_query_augmentation (bool): Enable query augmentation (default: False)
            data_loader: Data loader for accessing document content (can be overridden in filter())
            embedding_manager: Embedding manager for computing embeddings (can be overridden in filter())
            enable_calibrate: If True, calibrate threshold using train_doc_ids (can be overridden in filter())
            train_doc_ids: List of doc IDs for calibration training set (can be overridden in filter())
        """
        super().__init__(config)
        
        # Extract configuration
        self.target_recall = self.config.get("target_recall", self.DEFAULT_TARGET_RECALL)
        self.alpha = 1.0 - self.target_recall  # Significance level for conformal prediction
        self.embedding_model = self.config.get("embedding_model", self.DEFAULT_EMBEDDING_MODEL)
        # Use model-appropriate default: Gemini embeddings tend to have higher similarity
        default_thresh = (
            self.DEFAULT_THRESHOLD_GEMINI
            if "gemini" in self.embedding_model.lower()
            else self.DEFAULT_THRESHOLD
        )
        self.threshold = self.config.get("threshold", default_thresh)
        self.enable_query_augmentation = self.config.get("enable_query_augmentation", False)
        self.use_source_table_metadata = bool(
            self.config.get("use_source_table_metadata", False)
        )
        self.source_table_metadata_only = bool(
            self.config.get("source_table_metadata_only", False)
        )
        self.source_table_keep_unknown = bool(
            self.config.get("source_table_keep_unknown", True)
        )
        
        # Store instance-level dependencies (can be overridden in filter())
        self._enable_calibrate = enable_calibrate  # TODO: load from config
        self._data_loader = data_loader
        self._embedding_manager = embedding_manager
        self._train_doc_ids = train_doc_ids
        
        # Calibration state
        self._is_calibrated = False
        self._calibration_result: Optional[ConformalCalibrationResult] = None
        
        logging.info(
            f"[{self._name}:__init__] Initialized with target_recall={self.target_recall}, "
            f"threshold={self.threshold}, embedding_model={self.embedding_model}"
        )
    
    # =========================================================================
    # Main Filtering Method
    # =========================================================================
    
    def filter(
        self,
        query_id: str,
        doc_ids: List[str],
        **kwargs,
    ) -> FilterResult:
        """
        Filter chunks based on schema relevance using embedding similarity.
        
        Workflow:
        1. Get/compute query embedding (with optional augmentation)
        2. Get/compute document embeddings
        3. Compute cosine similarity for each document
        4. (Optional) Calibrate threshold using train_doc_ids if enable_calibrate=True
        5. Filter: exclude documents where sim < threshold
        
        Args:
            query_id: The query identifier.
            doc_ids: List of document IDs to filter.
            **kwargs: Additional arguments (override instance-level settings if provided):
                - data_loader: Data loader for accessing document content
                - embedding_manager: Embedding manager for computing embeddings
                - enable_calibrate: If True, calibrate threshold using train_doc_ids
                - train_doc_ids: List of doc IDs for calibration training set.
                - query_text: Query text (required if not in data_loader)
                - threshold_override: Override the calibrated threshold (ignored if enable_calibrate=True)
                - schema_context: Schema context for query augmentation
                
        Returns:
            FilterResult with schema-irrelevant chunks excluded.
        """
        # Extract arguments (kwargs override instance-level settings)
        data_loader: Optional["DataLoaderHFManifest"] = kwargs.get("data_loader", self._data_loader)
        embedding_manager: Optional["EmbeddingManager"] = kwargs.get("embedding_manager", self._embedding_manager)
        enable_calibrate: bool = kwargs.get("enable_calibrate", self._enable_calibrate)
        train_doc_ids: Optional[List[str]] = kwargs.get("train_doc_ids", self._train_doc_ids)
        query_text: Optional[str] = kwargs.get("query_text")
        threshold_override: Optional[float] = kwargs.get("threshold_override")
        
        # Validate required arguments. Metadata-only filtering does not need embeddings.
        if data_loader is None or (
            embedding_manager is None and not self.source_table_metadata_only
        ):
            logging.error(
                f"[{self._name}:filter] data_loader and embedding_manager are required "
                f"unless source_table_metadata_only=True."
            )
            return FilterResult(
                excluded_doc_ids=set(),
                metadata={
                    "filter_name": self._name,
                    "query_id": query_id,
                    "error": "Missing required arguments: data_loader and embedding_manager"
                }
            )

        metadata_excluded_doc_ids: Set[str] = set()
        source_table_metadata: Dict[str, Any] = {}
        if self.use_source_table_metadata:
            metadata_excluded_doc_ids, source_table_metadata = (
                self._excluded_doc_ids_by_source_table(
                    query_id=query_id,
                    doc_ids=doc_ids,
                    data_loader=data_loader,
                    schema_context=kwargs.get("schema_context"),
                )
            )
            if self.source_table_metadata_only:
                kept_count = len(doc_ids) - len(metadata_excluded_doc_ids)
                return FilterResult(
                    excluded_doc_ids=metadata_excluded_doc_ids,
                    metadata={
                        "filter_name": self._name,
                        "query_id": query_id,
                        "threshold": None,
                        "is_calibrated": False,
                        "target_recall": self.target_recall,
                        "num_docs_input": len(doc_ids),
                        "num_docs_excluded": len(metadata_excluded_doc_ids),
                        "num_docs_kept": kept_count,
                        "metadata_only": True,
                        **source_table_metadata,
                        "guarantee": (
                            "Source-table metadata filter; unknown source tables "
                            "are kept unless source_table_keep_unknown=false."
                        ),
                    },
                )

        if float(self.target_recall) >= 1.0 - 1e-6:
            kept_count = len(doc_ids) - len(metadata_excluded_doc_ids)
            return FilterResult(
                excluded_doc_ids=metadata_excluded_doc_ids,
                metadata={
                    "filter_name": self._name,
                    "query_id": query_id,
                    "threshold": None,
                    "is_calibrated": False,
                    "target_recall": self.target_recall,
                    "num_docs_input": len(doc_ids),
                    "num_docs_excluded": len(metadata_excluded_doc_ids),
                    "num_docs_kept": kept_count,
                    **source_table_metadata,
                    "guarantee": "Target recall is effectively 1.0; embedding filter skipped.",
                },
            )
        
        # Get query text if not provided
        if query_text is None:
            query_info = data_loader.get_query_info(query_id)
            if query_info:
                # queries.json uses "query"; some loaders use "query_text"
                query_text = query_info.get("query_text") or query_info.get("query", "")
        
        if not query_text:
            logging.warning(
                f"[{self._name}:filter] No query text available for query_id={query_id}. "
                f"Returning empty filter result."
            )
            return FilterResult(
                excluded_doc_ids=set(),
                metadata={
                    "filter_name": self._name,
                    "query_id": query_id,
                    "error": "No query text available"
                }
            )
        
        doc_ids_to_score = [
            doc_id for doc_id in doc_ids if doc_id not in metadata_excluded_doc_ids
        ]

        # Step 1: Augment query (hook for future)
        augmented_query = self.augment_query(
            query_text,
            schema_context=kwargs.get("schema_context")
        )
        
        # Step 2: Get query embedding
        query_emb = embedding_manager.get_query_embedding(query_id, augmented_query)
        
        # Step 3: Get document embeddings (for all docs, including training ones)
        all_doc_ids_needed = set(doc_ids_to_score)
        if enable_calibrate and train_doc_ids:
            all_doc_ids_needed.update(train_doc_ids)
        
        doc_embeddings = embedding_manager.get_doc_embeddings(
            loader=data_loader,
            doc_ids=list(all_doc_ids_needed)
        )
        
        # Step 4: Compute similarities for all docs
        similarities: Dict[str, float] = {}
        for doc_id in all_doc_ids_needed:
            if doc_id not in doc_embeddings:
                logging.warning(
                    f"[{self._name}:filter] No embedding for doc_id={doc_id}. Skipping."
                )
                continue
            doc_emb = doc_embeddings[doc_id]
            sim = self.cosine_similarity(query_emb, doc_emb)
            similarities[doc_id] = sim
        
        # Step 5: Calibrate threshold if enabled
        if enable_calibrate:
            if not train_doc_ids:
                logging.warning(
                    f"[{self._name}:filter] enable_calibrate=True but no train_doc_ids provided. "
                    f"Using default threshold."
                )
            else:
                # Extract relevant doc IDs from training set based on query schema
                relevant_doc_ids = self._get_relevant_doc_ids_from_train_set(
                    data_loader=data_loader,
                    query_id=query_id,
                    train_doc_ids=train_doc_ids,
                )
                
                if not relevant_doc_ids:
                    logging.warning(
                        f"[{self._name}:filter] No relevant docs found in training set. "
                        f"Using default threshold."
                    )
                else:
                    # Collect similarities for relevant docs
                    calibration_similarities = [
                        similarities[doc_id]
                        for doc_id in relevant_doc_ids
                        if doc_id in similarities
                    ]
                    
                    if calibration_similarities:
                        # Use ConformalCalibrator to compute threshold
                        calibrator = ConformalCalibrator(
                            target_recall=self.target_recall,
                            default_threshold=self.DEFAULT_THRESHOLD,
                        )
                        self._calibration_result = calibrator.calibrate_from_similarities(
                            similarities=calibration_similarities,
                            metadata={
                                "filter_name": self._name,
                                "num_queries": 1,
                                "query_id": query_id,
                                "num_train_docs": len(train_doc_ids),
                                "num_relevant_docs": len(relevant_doc_ids),
                                "embedding_model": self.embedding_model,
                            }
                        )
                        self.threshold = self._calibration_result.threshold
                        self._is_calibrated = True
                        
                        logging.info(
                            f"[{self._name}:filter] Inline calibration complete. "
                            f"threshold={self.threshold:.4f}, "
                            f"n_train={len(train_doc_ids)}, "
                            f"n_relevant={len(calibration_similarities)}"
                        )
                    else:
                        logging.warning(
                            f"[{self._name}:filter] No valid similarities for relevant docs. "
                            f"Using default threshold."
                        )
        
        # Step 6: Determine final threshold and filter
        # If calibration was done, use calibrated threshold; otherwise use override or default
        if enable_calibrate and self._is_calibrated:
            threshold = self.threshold
        else:
            threshold = threshold_override if threshold_override is not None else self.threshold
        
        # Filter documents
        excluded_doc_ids: Set[str] = set(metadata_excluded_doc_ids)
        for doc_id in doc_ids_to_score:
            if doc_id not in similarities:
                continue
            if similarities[doc_id] < threshold:
                excluded_doc_ids.add(doc_id)
        
        # Compute statistics (only for input doc_ids)
        sim_values = [similarities[doc_id] for doc_id in doc_ids_to_score if doc_id in similarities]
        stats = {
            "min_similarity": min(sim_values) if sim_values else 0.0,
            "max_similarity": max(sim_values) if sim_values else 0.0,
            "avg_similarity": float(np.mean(sim_values)) if sim_values else 0.0,
            "std_similarity": float(np.std(sim_values)) if sim_values else 0.0,
        }
        
        logging.info(
            f"[{self._name}:filter] Filtered {len(doc_ids)} docs: "
            f"excluded={len(excluded_doc_ids)}, kept={len(doc_ids) - len(excluded_doc_ids)}, "
            f"threshold={threshold:.4f}, min_sim={stats['min_similarity']:.4f}, "
            f"avg_sim={stats['avg_similarity']:.4f}, max_sim={stats['max_similarity']:.4f}"
        )
        
        return FilterResult(
            excluded_doc_ids=excluded_doc_ids,
            metadata={
                "filter_name": self._name,
                "query_id": query_id,
                "threshold": threshold,
                "is_calibrated": self._is_calibrated,
                "target_recall": self.target_recall,
                "num_docs_input": len(doc_ids),
                "num_docs_excluded": len(excluded_doc_ids),
                "num_docs_kept": len(doc_ids) - len(excluded_doc_ids),
                **source_table_metadata,
                "similarities": {k: v for k, v in similarities.items() if k in doc_ids},
                **stats,
                "guarantee": (
                    self._calibration_result.guarantee
                    if self._calibration_result
                    else "No guarantee (not calibrated)"
                )
            }
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _excluded_doc_ids_by_source_table(
        self,
        *,
        query_id: str,
        doc_ids: List[str],
        data_loader: "DataLoaderHFManifest",
        schema_context: Any,
    ) -> tuple[Set[str], Dict[str, Any]]:
        allowed_tables = self._source_table_allowed_names(
            query_id=query_id,
            data_loader=data_loader,
            schema_context=schema_context,
        )
        if not allowed_tables:
            return set(), {
                "source_table_metadata_enabled": True,
                "source_table_metadata_error": "No allowed source tables resolved",
                "source_table_metadata_excluded": 0,
            }

        excluded_doc_ids: Set[str] = set()
        known_count = 0
        unknown_count = 0
        for doc_id in doc_ids:
            table_names = self._doc_source_tables(data_loader, doc_id, allowed_tables)
            if not table_names:
                unknown_count += 1
                if not self.source_table_keep_unknown:
                    excluded_doc_ids.add(doc_id)
                continue
            known_count += 1
            if table_names.isdisjoint(allowed_tables):
                excluded_doc_ids.add(doc_id)

        return excluded_doc_ids, {
            "source_table_metadata_enabled": True,
            "source_table_metadata_only": self.source_table_metadata_only,
            "source_table_keep_unknown": self.source_table_keep_unknown,
            "source_table_allowed": sorted(allowed_tables),
            "source_table_known_docs": known_count,
            "source_table_unknown_docs": unknown_count,
            "source_table_metadata_excluded": len(excluded_doc_ids),
        }

    def _source_table_allowed_names(
        self,
        *,
        query_id: str,
        data_loader: "DataLoaderHFManifest",
        schema_context: Any,
    ) -> Set[str]:
        schema_items = schema_context if isinstance(schema_context, list) else None
        if not schema_items:
            schema_items = data_loader.load_schema_query(query_id)
        if not schema_items:
            schema_items = data_loader.load_schema_general()

        table_names = {
            str(schema.get("Schema Name") or "").strip()
            for schema in schema_items
            if isinstance(schema, dict)
        }
        table_names = {name for name in table_names if name}

        allowed = {name.lower() for name in table_names}
        try:
            name_map = data_loader.load_name_map(query_id=query_id)
        except Exception:
            name_map = {}
        table_map = name_map.get("table", {}) if isinstance(name_map, dict) else {}
        if isinstance(table_map, dict):
            for task_table, source_table in table_map.items():
                task_name = str(task_table or "").strip()
                source_name = str(source_table or "").strip()
                if task_name in table_names and source_name:
                    allowed.add(source_name.lower())
        return allowed

    @staticmethod
    def _doc_source_tables(
        data_loader: "DataLoaderHFManifest",
        doc_id: str,
        allowed_tables: Set[str],
    ) -> Set[str]:
        if not hasattr(data_loader, "get_doc"):
            return set()
        try:
            _doc_text, _resolved_doc_id, metadata = data_loader.get_doc(doc_id)
        except Exception:
            return set()
        if not isinstance(metadata, dict):
            return set()

        schema_tables = metadata.get("schema_tables")
        if isinstance(schema_tables, list):
            normalized = {
                str(table_name or "").strip().lower()
                for table_name in schema_tables
                if str(table_name or "").strip()
            }
            if normalized:
                return normalized

        for key in ("schema_table", "table_id"):
            table_name = str(metadata.get(key) or "").strip().lower()
            if table_name:
                return {table_name}

        table_name = str(
            metadata.get("table_name") or metadata.get("source_table") or ""
        ).strip().lower()
        if table_name and table_name in allowed_tables:
            return {table_name}
        return set()

    @staticmethod
    def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
        return utils_cosine_similarity(emb1, emb2)
    
    @staticmethod
    def nonconformity_score(query_emb: List[float], doc_emb: List[float]) -> float:
        return nonconformity_score_negative_cosine(query_emb, doc_emb)
    
    @property
    def is_calibrated(self) -> bool:
        """Check if the filter has been calibrated."""
        return self._is_calibrated
    
    @property
    def calibration_result(self) -> Optional[ConformalCalibrationResult]:
        """Get the calibration result."""
        return self._calibration_result
    
    def set_threshold(self, threshold: float) -> None:
        """
        Manually set the threshold.
        
        Args:
            threshold: New threshold value
        """
        self.threshold = threshold
        logging.info(f"[{self._name}:set_threshold] Threshold set to {threshold:.4f}")
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """
        Get filter statistics and configuration.
        
        Returns:
            Dictionary with filter information
        """
        return {
            "filter_name": self._name,
            "target_recall": self.target_recall,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "is_calibrated": self._is_calibrated,
            "embedding_model": self.embedding_model,
            "enable_query_augmentation": self.enable_query_augmentation,
            "calibration_result": (
                {
                    "threshold": self._calibration_result.threshold,
                    "alpha": self._calibration_result.alpha,
                    "guarantee": self._calibration_result.guarantee,
                    "num_samples": self._calibration_result.num_calibration_samples,
                }
                if self._calibration_result
                else None
            )
        }
    
    # =========================================================================
    # Calibration Helpers
    # =========================================================================
    
    def _get_relevant_doc_ids_from_train_set(
        self,
        data_loader: "DataLoaderHFManifest",
        query_id: str,
        train_doc_ids: List[str],
    ) -> List[str]:
        """
        Extract relevant document IDs from the training set based on query schema.
        
        A document is considered relevant if its ground-truth records contain
        one of the query's tables.
        
        Args:
            data_loader: Data loader with ground truth records
            query_id: Query identifier to get schema information
            train_doc_ids: List of document IDs in the training set
            
        Returns:
            List of relevant document IDs (subset of train_doc_ids)
        """
        # Get query schema information (task schema names)
        query_schemas = data_loader.load_schema_query(query_id)
        if not query_schemas:
            query_schemas = data_loader.load_schema_general()
        
        # Collect task schema table names
        schema_names: Set[str] = set()
        for schema in query_schemas:
            schema_name = schema.get("Schema Name", "")
            if schema_name:
                schema_names.add(schema_name)
        
        if not schema_names:
            logging.warning(
                f"[{self._name}:_get_relevant_doc_ids_from_train_set] "
                f"No schema/table names found for query_id={query_id}"
            )
            return []
        
        # Keep this name-map hook for older loaders; current datasets use
        # consistent table IDs across schema, queries, and ground truth.
        name_map = data_loader.load_name_map(query_id=query_id)
        table_map = name_map.get("table", {})
        
        query_gt_table_names: Set[str] = set()
        for schema_name in schema_names:
            gt_name = table_map.get(schema_name, schema_name)
            query_gt_table_names.add(gt_name.lower())
        
        # Filter training docs: keep those whose GT table is in our set.
        train_doc_ids_set = set(train_doc_ids)
        relevant_doc_ids: List[str] = []
        
        for doc_id in train_doc_ids_set:
            doc_info = data_loader.get_doc_info(doc_id)
            if not doc_info:
                continue
            
            data_records = doc_info.get("data_records") or []
            for record in data_records:
                table_name = record.get("table_name", "")
                if table_name.lower() in query_gt_table_names:
                    relevant_doc_ids.append(doc_id)
                    break
        
        logging.debug(
            f"[{self._name}:_get_relevant_doc_ids_from_train_set] "
            f"Found {len(relevant_doc_ids)} relevant docs out of {len(train_doc_ids)} train docs "
            f"for tables: {query_gt_table_names}"
        )
        
        return relevant_doc_ids
    
    # =========================================================================
    # Query Augmentation (Hook for future implementation)
    # =========================================================================
    
    def augment_query(
        self,
        query_text: str,
        schema_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Augment the query with relevant context from ISD Phase I.
        
        This is a hook for future implementation. Currently returns the
        original query text unchanged.
        
        Args:
            query_text: Original query text
            schema_context: Schema context information
            **kwargs: Additional augmentation parameters
            
        Returns:
            Augmented query text
        """
        if not self.enable_query_augmentation:
            return query_text
        
        # TODO: Implement query augmentation
        # Potential strategies:
        # 1. Append schema attribute names to query
        # 2. Add extracted partial evidence
        # 3. Use LLM to rephrase/expand query
        
        logging.debug(
            f"[{self._name}:augment_query] Query augmentation not yet implemented. "
            f"Returning original query."
        )
        return query_text

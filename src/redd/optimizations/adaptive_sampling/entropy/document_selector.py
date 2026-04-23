"""
Document Selector for Adaptive Sampling.

This module implements efficient document selection using SQLite-backed embeddings.
Uses farthest-from-mean strategy with an indexed embedding structure for O(1) lookups.
Embeddings are loaded/generated via ``core.embedding.EmbeddingManager`` cache.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Set

import numpy as np

from redd.core.embedding import EmbeddingManager


class DocumentSelector:
    """
    Efficiently selects documents for processing based on embedding similarity.
    Uses SQLite-cached embeddings with an indexed structure for fast lookups.
    Implements incremental farthest-from-mean selection during adaptive sampling.
    """
    
    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        """
        Initialize the document selector.
        
        Args:
            config: Configuration dictionary containing adaptive_sampling settings
            api_key: Optional API key used when missing embeddings are generated
        """
        adaptive_config = config.get("adaptive_sampling", {})
        self.enabled = adaptive_config.get("use_embedding_selection", True)
        self.embedding_model = adaptive_config.get("embedding_model", "gemini-embedding-001")
        
        # Deprecated: legacy JSON embedding path (ignored in SQLite flow).
        self.embedding_file = adaptive_config.get("embedding_file")
        self.dataset_db_path = (
            adaptive_config.get("dataset_db_path")
            or config.get("dataset_db_path")
            or "dataset/adaptive_sampling/default_task.db"
        )
        
        # SQLite-backed embedding manager.
        self._embedding_manager = EmbeddingManager(
            dataset_db_path=self.dataset_db_path,
            model=self.embedding_model,
            api_key=api_key,
        )
        
        # Embedding index structure for efficient lookups
        # embedding_matrix: (N, D) numpy array of all embeddings
        # doc_id_to_idx: dict mapping doc_id (str) -> index in embedding_matrix
        # idx_to_doc_id: dict mapping index -> doc_id (str) for reverse lookup
        self.embedding_matrix = None  # np.ndarray of shape (N, D)
        self.doc_id_to_idx = {}  # doc_id -> row index in matrix
        self.idx_to_doc_id = {}  # row index -> doc_id
        self.embedding_dim = None
        
        self._index_built = False
        
        if self.enabled:
            logging.info(f"[{self.__class__.__name__}:__init__] Initialized with model: {self.embedding_model}")

    @staticmethod
    def _build_doc_loader(doc_dict: Dict[str, Any]) -> Any:
        """Build a minimal loader adapter around in-memory doc_dict."""
        class _DocDictLoader:
            def __init__(self, docs: Dict[str, Any]):
                self._docs = docs
                self.doc_ids = list(docs.keys())

            def get_doc_text(self, doc_id: str) -> str:
                doc_content = self._docs.get(doc_id)
                if isinstance(doc_content, list) and doc_content:
                    return str(doc_content[0])
                if isinstance(doc_content, str):
                    return doc_content
                return ""

        return _DocDictLoader(doc_dict)

    def build_index(self, doc_dict: Dict[str, Any]) -> bool:
        """
        Build an efficient index structure from SQLite-cached embeddings.
        Must be called before using incremental selection.
        
        Args:
            doc_dict: Dictionary of documents {doc_id: [doc_text, ...]}
                     Used to get list of all document IDs
            
        Returns:
            True if index built successfully, False otherwise
        """
        if not self.enabled:
            return False
            
        if self._index_built:
            logging.debug(f"[{self.__class__.__name__}:build_index] Index already built, skipping")
            return True
        
        if self.embedding_file:
            logging.info(
                f"[{self.__class__.__name__}:build_index] "
                "adaptive_sampling.embedding_file is ignored; using SQLite embedding cache."
            )

        loader = self._build_doc_loader(doc_dict)
        embeddings_dict = self._embedding_manager.get_doc_embeddings(
            loader=loader,
            doc_ids=list(doc_dict.keys()),
        )
        
        if not embeddings_dict:
            logging.warning(
                f"[{self.__class__.__name__}:build_index] No embeddings loaded/generated. "
                f"Falling back to random selection."
            )
            return False

        # Filter embeddings to only include documents in doc_dict.
        all_doc_ids = set(doc_dict.keys())
        filtered_embeddings = {}
        
        for did in all_doc_ids:
            if did in embeddings_dict:
                filtered_embeddings[did] = embeddings_dict[did]
        
        if not filtered_embeddings:
            logging.warning(
                f"[{self.__class__.__name__}:build_index] No matching embeddings found for documents in doc_dict. "
                f"Expected {len(all_doc_ids)} documents, found {len(embeddings_dict)} embeddings."
            )
            return False
        
        missing_ids = all_doc_ids - set(filtered_embeddings.keys())
        if missing_ids:
            logging.warning(
                f"[{self.__class__.__name__}:build_index] {len(missing_ids)}/{len(all_doc_ids)} documents missing embeddings. "
                f"Will use random selection for those documents."
            )
        
        # Build index structure: create matrix and mappings
        doc_ids = list(filtered_embeddings.keys())
        embeddings_list = [filtered_embeddings[doc_id] for doc_id in doc_ids]
        
        # Stack into matrix: shape (N, D) where N=num_docs, D=embedding_dim
        self.embedding_matrix = np.stack(embeddings_list, axis=0).astype(np.float32)
        self.embedding_dim = self.embedding_matrix.shape[1]
        
        # Build bidirectional mappings
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        self.idx_to_doc_id = {idx: doc_id for idx, doc_id in enumerate(doc_ids)}
        
        self._index_built = True
        
        logging.info(
            f"[{self.__class__.__name__}:build_index] Built index for {len(doc_ids)} documents "
            f"(embedding_dim={self.embedding_dim})"
        )
        return True

    def select_next_farthest_from_mean(
        self,
        available_doc_ids: Set[str],
        selected_doc_ids: List[str]
    ) -> Optional[str]:
        """
        Select the next document that is farthest from the mean of already selected documents.
        This is an incremental operation - called each time we need the next document.
        
        Args:
            available_doc_ids: Set of document IDs that haven't been selected yet
            selected_doc_ids: List of document IDs already selected (in order)
            
        Returns:
            Document ID of the next document to select, or None if index not built or no candidates
        """
        if not self._index_built or self.embedding_matrix is None:
            # Fallback to random if index not available
            if available_doc_ids:
                return random.choice(list(available_doc_ids))
            return None
        
        # If no documents selected yet, pick random from available
        if not selected_doc_ids:
            return random.choice(list(available_doc_ids)) if available_doc_ids else None
        
        # Get indices of selected documents in the embedding matrix
        selected_indices = [self.doc_id_to_idx[doc_id] for doc_id in selected_doc_ids if doc_id in self.doc_id_to_idx]
        
        if not selected_indices:
            # Selected documents not in index, fallback to random
            return random.choice(list(available_doc_ids)) if available_doc_ids else None
        
        # Compute mean of selected embeddings efficiently using numpy
        # selected_embeddings: (M, D) where M=len(selected_indices), D=embedding_dim
        selected_embeddings = self.embedding_matrix[selected_indices, :]  # Shape: (M, D)
        mean_embedding = np.mean(selected_embeddings, axis=0)  # Shape: (D,)
        
        # Get indices of available documents that are in our index
        available_indices = []
        available_doc_ids_in_index = []
        for doc_id in available_doc_ids:
            if doc_id in self.doc_id_to_idx:
                idx = self.doc_id_to_idx[doc_id]
                available_indices.append(idx)
                available_doc_ids_in_index.append(doc_id)
        
        if not available_indices:
            # No available documents in index, fallback to random
            return random.choice(list(available_doc_ids)) if available_doc_ids else None
        
        # Compute distances from mean to all available embeddings efficiently
        # available_embeddings: (K, D) where K=len(available_indices)
        available_embeddings = self.embedding_matrix[available_indices, :]  # Shape: (K, D)
        
        # Compute L2 distances: ||emb - mean|| for each available embedding
        # Use broadcasting: (K, D) - (D,) -> (K, D), then norm along axis=1 -> (K,)
        distances = np.linalg.norm(available_embeddings - mean_embedding, axis=1)  # Shape: (K,)
        
        # Find index of maximum distance
        max_dist_idx = np.argmax(distances)
        farthest_doc_id = available_doc_ids_in_index[max_dist_idx]
        
        return farthest_doc_id

    def get_document_order(self, doc_dict: Dict[str, Any]) -> List[str]:
        """
        DEPRECATED: Pre-orders all documents. 
        For adaptive sampling, use build_index() and select_next_farthest_from_mean() incrementally.
        
        This method is kept for backward compatibility but is inefficient for adaptive sampling
        since it orders all documents upfront even if we might stop early.
        
        Args:
            doc_dict: Dictionary of documents
            
        Returns:
            Pre-ordered list of document IDs
        """
        all_ids = list(doc_dict.keys())
        
        if not self.enabled:
            logging.info(f"[{self.__class__.__name__}] Embedding selection disabled, using random order")
            random.shuffle(all_ids)
            return all_ids
        
        # Try to build index
        if not self.build_index(doc_dict):
            # Fallback to random
            logging.warning(f"[{self.__class__.__name__}:get_document_order] Index build failed, using random order")
            random.shuffle(all_ids)
            return all_ids
        
        # Pre-order all documents (inefficient for adaptive sampling, but kept for compatibility)
        ordered_ids = []
        available_ids = set(all_ids)
        
        logging.info(f"[{self.__class__.__name__}:get_document_order] Pre-ordering {len(all_ids)} documents using Farthest-From-Mean strategy...")
        
        while available_ids:
            next_id = self.select_next_farthest_from_mean(available_ids, ordered_ids)
            if next_id is None:
                # Should not happen, but handle gracefully
                next_id = random.choice(list(available_ids))
            
            ordered_ids.append(next_id)
            available_ids.remove(next_id)
            
            if len(ordered_ids) % 50 == 0:
                logging.debug(f"[{self.__class__.__name__}:get_document_order] Pre-ordered {len(ordered_ids)}/{len(all_ids)} documents")
        
        return ordered_ids

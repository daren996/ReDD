"""
Fixed Size Chunker

Implements fixed-length document chunking with optional overlap.
Splits documents into chunks of a specified character length.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..data_loader import DataLoaderBase
from .base_chunker import BaseChunker


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size document chunking with optional overlap.
    
    Splits documents into chunks of approximately `chunk_size` characters,
    with optional overlap between consecutive chunks.
    
    Config keys (in addition to BaseChunker):
    - chunk_size: Target size of each chunk in characters (default: 4000)
    - overlap: Number of overlapping characters between chunks (default: 200)
    
    Example:
        >>> from redd.core.doc_chunking import create_chunker
        >>> config = {
        ...     "data_main": "dataset/fda_sqlite/",
        ...     "exp_dataset_task_list": ["no_chunk"],
        ...     "data_loader_type": "sqlite",
        ...     "chunker_type": "fixed_size",
        ...     "chunk_size": 4000,
        ...     "overlap": 200,
        ... }
        >>> chunker = create_chunker(config)
        >>> chunker()  # Creates: dataset/fda_sqlite/fixed_size_4000_overlap_200.db
    """
    
    def __init__(self, config: Dict[str, Any], loader: Optional[DataLoaderBase] = None):
        """
        Initialize the fixed-size chunker.
        
        Args:
            config: Configuration dictionary containing:
                - data_main: Base data directory
                - exp_dataset_task_list: List of data paths to process
                - chunk_size: Target size of each chunk in characters (default: 4000)
                - overlap: Number of overlapping characters (default: 200)
            loader: Optional pre-created DataLoader instance
        """
        super().__init__(config, loader)
        
        self.chunk_size = config.get("chunk_size", 200000)
        self.overlap = config.get("overlap", 4000)
        
        if self.chunk_size <= 0:
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] chunk_size must be positive, got {self.chunk_size}"
            )
        if self.overlap < 0:
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] overlap must be non-negative, got {self.overlap}"
            )
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] overlap ({self.overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        logging.info(f"[{self.__class__.__name__}:__init__] "
                    f"chunk_size={self.chunk_size}, overlap={self.overlap}")
    
    @property
    def chunker_name(self) -> str:
        """Return the chunker identifier for output file naming."""
        if self.overlap > 0:
            return f"fixed_size_{self.chunk_size}_overlap_{self.overlap}"
        else:
            return f"fixed_size_{self.chunk_size}"
    
    def chunk_document(
        self, 
        doc_text: str, 
        doc_id: str, 
        metadata: Dict[str, Any]
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Split a document into fixed-size chunks.
        
        Args:
            doc_text: The document text to chunk
            doc_id: Original document ID
            metadata: Original document metadata
            
        Returns:
            List of (chunk_text, chunk_id, chunk_metadata) tuples
        """
        if not doc_text:
            # Empty document - return single empty chunk
            return [(
                "",
                f"{doc_id}-0",
                {
                    **metadata,
                    "parent_doc_id": doc_id,
                    "chunk_index": 0,
                }
            )]
        
        # If document is smaller than chunk_size, return as single chunk
        if len(doc_text) <= self.chunk_size:
            return [(
                doc_text,
                f"{doc_id}-0",
                {
                    **metadata,
                    "parent_doc_id": doc_id,
                    "chunk_index": 0,
                }
            )]
        
        chunks = []
        step = self.chunk_size - self.overlap
        chunk_index = 0
        start = 0
        
        while start < len(doc_text):
            end = min(start + self.chunk_size, len(doc_text))
            chunk_text = doc_text[start:end]
            
            # Create chunk ID: original_doc_id-N (e.g., 0-0, 1-0, 2-1)
            chunk_id = f"{doc_id}-{chunk_index}"
            
            # Create chunk metadata
            chunk_metadata = {
                **metadata,
                "parent_doc_id": doc_id,
                "chunk_index": chunk_index,
            }
            
            chunks.append((chunk_text, chunk_id, chunk_metadata))
            
            chunk_index += 1
            start += step
            
            # Avoid creating tiny trailing chunks
            if start >= len(doc_text):
                break
            # If remaining text would be less than overlap, include it in last chunk
            if len(doc_text) - start < self.overlap:
                break
        
        return chunks


class FixedSizeTokenChunker(BaseChunker):
    """
    Fixed-size document chunking based on token count (instead of character count).
    
    Useful when working with LLMs that have token limits.
    Requires tiktoken to be installed.
    
    Config keys (in addition to BaseChunker):
    - chunk_size: Target size of each chunk in tokens (default: 1024)
    - overlap: Number of overlapping tokens between chunks (default: 100)
    - tokenizer_name: Name of the tiktoken encoding (default: "cl100k_base")
    
    Example:
        >>> from redd.core.doc_chunking import create_chunker
        >>> config = {
        ...     "data_main": "dataset/fda_sqlite/",
        ...     "exp_dataset_task_list": ["no_chunk"],
        ...     "data_loader_type": "sqlite",
        ...     "chunker_type": "fixed_token",
        ...     "chunk_size": 1024,
        ...     "overlap": 100,
        ...     "tokenizer_name": "cl100k_base",
        ... }
        >>> chunker = create_chunker(config)
        >>> chunker()  # Creates: dataset/fda_sqlite/fixed_token_1024_overlap_100.db
    """
    
    def __init__(self, config: Dict[str, Any], loader: Optional[DataLoaderBase] = None):
        """
        Initialize the token-based chunker.
        
        Args:
            config: Configuration dictionary containing:
                - data_main: Base data directory
                - exp_dataset_task_list: List of data paths to process
                - chunk_size: Target size of each chunk in tokens (default: 1024)
                - overlap: Number of overlapping tokens (default: 100)
                - tokenizer_name: Name of the tiktoken encoding (default: "cl100k_base")
            loader: Optional pre-created DataLoader instance
        """
        super().__init__(config, loader)
        
        self.chunk_size = config.get("chunk_size", 50000)
        self.overlap = config.get("overlap", 1000)
        self.tokenizer_name = config.get("tokenizer_name", "cl100k_base")
        
        if self.chunk_size <= 0:
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] chunk_size must be positive, got {self.chunk_size}"
            )
        if self.overlap < 0:
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] overlap must be non-negative, got {self.overlap}"
            )
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] overlap ({self.overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        # Lazy load tokenizer
        self._tokenizer = None
        
        logging.info(f"[{self.__class__.__name__}:__init__] "
                    f"chunk_size={self.chunk_size} tokens, overlap={self.overlap}, "
                    f"tokenizer={self.tokenizer_name}")
    
    @property
    def tokenizer(self):
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding(self.tokenizer_name)
            except ImportError:
                raise ImportError(
                    f"[{self.__class__.__name__}] tiktoken is required for token-based chunking. "
                    f"Install with: pip install tiktoken"
                )
            except Exception as e:
                raise ValueError(
                    f"[{self.__class__.__name__}] Failed to load tokenizer '{self.tokenizer_name}': {e}"
                )
        return self._tokenizer
    
    @property
    def chunker_name(self) -> str:
        """Return the chunker identifier for output file naming."""
        if self.overlap > 0:
            return f"fixed_token_{self.chunk_size}_overlap_{self.overlap}"
        else:
            return f"fixed_token_{self.chunk_size}"
    
    def chunk_document(
        self, 
        doc_text: str, 
        doc_id: str, 
        metadata: Dict[str, Any]
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Split a document into fixed-size token chunks.
        
        Args:
            doc_text: The document text to chunk
            doc_id: Original document ID
            metadata: Original document metadata
            
        Returns:
            List of (chunk_text, chunk_id, chunk_metadata) tuples
        """
        if not doc_text:
            return [(
                "",
                f"{doc_id}-0",
                {
                    **metadata,
                    "parent_doc_id": doc_id,
                    "chunk_index": 0,
                }
            )]
        
        # Tokenize the document
        tokens = self.tokenizer.encode(doc_text)
        
        # If document is smaller than chunk_size, return as single chunk
        if len(tokens) <= self.chunk_size:
            return [(
                doc_text,
                f"{doc_id}-0",
                {
                    **metadata,
                    "parent_doc_id": doc_id,
                    "chunk_index": 0,
                }
            )]
        
        chunks = []
        step = self.chunk_size - self.overlap
        chunk_index = 0
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk ID: original_doc_id-N (e.g., 0-0, 1-0, 2-1)
            chunk_id = f"{doc_id}-{chunk_index}"
            
            # Create chunk metadata
            chunk_metadata = {
                **metadata,
                "parent_doc_id": doc_id,
                "chunk_index": chunk_index,
            }
            
            chunks.append((chunk_text, chunk_id, chunk_metadata))
            
            chunk_index += 1
            start += step
            
            if start >= len(tokens):
                break
            if len(tokens) - start < self.overlap:
                break
        
        return chunks

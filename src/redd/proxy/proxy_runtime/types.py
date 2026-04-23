"""Configuration and result types for the proxy runtime."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging

from redd.core.utils.sql_filter_parser import AttributePredicate

from .executor import ExecutionStats, HardNegative


@dataclass
class ProxyPipelineConfig:
    """
    Configuration for the proxy runtime pipeline.
    
    Attributes:
        dataset_path: Path to dataset directory (e.g., "spider_sqlite/college_2")
        query_id: Query ID to process (e.g., "Q1")
        task_db_name: Optional specific task database name
        
        # LLM Configuration
        llm_mode: LLM provider mode ("gemini", "deepseek", "cgpt", etc.)
        llm_model: LLM model name
        api_key: Optional API key (defaults to environment variable)
        
        # Embedding Configuration
        embedding_model: Model for document embeddings
        embedding_api_key: Optional API key for embeddings
        
        # Predicate proxy configuration
        use_embedding_proxies: If True, use embedding-based similarity proxies
        use_classifier_proxies: If True, use pre-trained classifier proxies
        use_finetuned_learned_proxies: If True (default), use pretrained LM classifier (DeBERTa fine-tuned);
            else use LogisticRegression on embeddings for learned proxies
        classifier_paths: Dict mapping attribute names to classifier checkpoint paths
        proxy_threshold: Default threshold for predicate proxies
        target_recall: Target recall for calibration (if calibration data available)
        
        # Processing Configuration
        batch_size: Documents per batch for processing
        max_documents: Maximum documents to process (None = all)
        
        # Output Configuration
        output_dir: Directory to save results
        save_hard_negatives: If True, save hard negatives for retraining
    """
    # Dataset
    dataset_path: str = ""
    query_id: str = ""
    task_db_name: Optional[str] = None
    data_main: str = "dataset/"
    
    # LLM
    llm_mode: str = "gemini"
    llm_model: str = "gemini-2.5-flash-lite"
    api_key: Optional[str] = None
    
    # Embeddings (uses Gemini embedding API by default)
    embedding_model: str = "gemini-embedding-001"
    embedding_api_key: Optional[str] = None
    embeddings_cache_dir: Optional[str] = None  # Directory to cache embeddings
    
    # Predicate proxies
    use_embedding_proxies: bool = True
    use_classifier_proxies: bool = False
    use_learned_proxies: bool = True
    use_finetuned_learned_proxies: bool = True  # Default: pretrained LM classifier (DeBERTa); else LogisticRegression on embeddings
    training_data_count: int = 100
    # Kept for backward compatibility; prefix-only split policy no longer uses these minima.
    min_training_data: int = 0
    min_calibration_data: int = 0
    classifier_paths: Dict[str, str] = field(default_factory=dict)
    proxy_threshold: float = 0.5
    target_recall: float = 0.95
    random_seed: int = 42
    
    # Predicate proxy costs (relative)
    embedding_proxy_cost: float = 0.1
    classifier_proxy_cost: float = 0.05
    
    # Processing
    batch_size: int = 32
    max_documents: Optional[int] = None
    
    # Output
    output_dir: Optional[str] = None
    save_hard_negatives: bool = True
    verbose: bool = True
    
    # Join support
    use_join_resolution: bool = True
    join_extractor: str = "llm"

    # When True, train_doc_ids may overlap with doc_ids (table-relevant training)
    allow_train_test_overlap: bool = False

    # Model for learned guards (GLiClass). Can be HuggingFace model name or path to pretrained model.
    finetuned_model: str = "knowledgator/gliclass-instruct-large-v1.0"
    finetuned_epochs: int = 3
    finetuned_learning_rate: float = 2e-5

    # GLiClass in-context learning (few-shot examples, no fine-tuning)
    use_gliclass_icl: bool = False
    gliclass_icl_examples_per_class: int = 3


@dataclass
class PipelineResults:
    """
    Results from proxy-runtime execution.
    """
    # Query info
    query_id: str = ""
    query_text: str = ""
    sql: str = ""
    
    # Predicates parsed from SQL
    predicates: List[AttributePredicate] = field(default_factory=list)
    
    # Document processing stats
    total_documents: int = 0
    documents_processed: int = 0
    documents_passed_proxies: int = 0
    documents_extracted: int = 0
    
    # Extracted data
    extractions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Execution stats
    execution_stats: Optional[ExecutionStats] = None
    
    # Hard negatives for feedback
    hard_negatives: List[HardNegative] = field(default_factory=list)
    
    # Timing
    total_time_seconds: float = 0.0
    embedding_time_seconds: float = 0.0
    proxy_time_seconds: float = 0.0
    extraction_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "sql": self.sql,
            "predicates": [p.to_dict() for p in self.predicates],
            "total_documents": self.total_documents,
            "documents_processed": self.documents_processed,
            "documents_passed_proxies": self.documents_passed_proxies,
            "documents_extracted": self.documents_extracted,
            "extractions": self.extractions,
            "hard_negatives_count": len(self.hard_negatives),
            "timing": {
                "total_seconds": self.total_time_seconds,
                "embedding_seconds": self.embedding_time_seconds,
                "proxy_seconds": self.proxy_time_seconds,
                "extraction_seconds": self.extraction_time_seconds,
            },
        }
        if self.execution_stats is not None:
            safe_proxy_stats = {}
            for name, stat in self.execution_stats.proxy_stats.items():
                safe_proxy_stats[name] = {
                    k: int(v) if hasattr(v, "item") else v
                    for k, v in stat.items()
                }
            result["proxy_decisions"] = {
                "proxy_stats": safe_proxy_stats,
                "proxy_rejected_doc_ids": dict(self.execution_stats.proxy_rejected_doc_ids),
                "passed_doc_ids": list(self.execution_stats.passed_doc_ids),
                "all_doc_ids": list(self.execution_stats.all_doc_ids),
            }
        return result
    
    def save(self, path: Union[str, Path]):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logging.info(f"[PipelineResults] Saved results to {path}")

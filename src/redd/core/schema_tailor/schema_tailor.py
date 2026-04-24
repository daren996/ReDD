"""
Schema Tailor: Query-Specific Schema Refinement with Adaptive Sampling.

This module implements the second pass of schema discovery:
1. Load general schema from first pass
2. Filter documents relevant to a specific query using embeddings
3. Apply adaptive sampling with entropy-based stopping
4. Refine the schema for the query using LLM prompts

The SchemaTailor class uses adaptive sampling similar to the general schema pass:
- Documents are processed one by one
- Schema entropy is tracked after each document
- Early stopping when entropy stabilizes
"""

import json
import logging
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from redd.embedding import EmbeddingManager
from ..utils.prompt_utils import (
    PromptDeepSeek,
    PromptGPT,
    PromptGemini,
    PromptSiliconFlow,
    PromptTogether,
)
from ..utils.constants import PATH_TEMPLATES
from ..utils.progress import tqdm
from redd.optimizations.adaptive_sampling.entropy.sampler import AdaptiveSampler
from redd.optimizations.adaptive_sampling.schema_entropy import SchemaEntropyCalculator


class QueryDocumentFilter:
    """
    Filter documents based on query relevance using embeddings.
    
    Uses cosine similarity between query embedding and document embeddings
    to select the most relevant documents for schema refinement.
    """
    
    def __init__(
        self,
        embedding_model: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: float = 0.0,
        dataset_db_path: Optional[str] = None,
    ):
        """
        Initialize the document filter.
        
        Args:
            embedding_model: Model for generating query embedding
            api_key: API key for embedding generation
            top_k: Select top K most similar documents (if None, use threshold)
            similarity_threshold: Minimum cosine similarity to include document
            dataset_db_path: Dataset DB anchor path for SQLite embedding cache
        """
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.dataset_db_path = Path(dataset_db_path) if dataset_db_path else None
        self._embedding_manager: Optional[EmbeddingManager] = None

    def set_dataset_db_path(self, dataset_db_path: str | Path) -> None:
        """Set dataset DB anchor path and reset embedding manager."""
        self.dataset_db_path = Path(dataset_db_path)
        self._embedding_manager = None

    @property
    def embedding_manager(self) -> EmbeddingManager:
        """Get or create SQLite-backed embedding manager."""
        if self._embedding_manager is None:
            db_anchor = self.dataset_db_path or Path("dataset/schema_tailor/default_task.db")
            self._embedding_manager = EmbeddingManager(
                dataset_db_path=db_anchor,
                model=self.embedding_model,
                api_key=self.api_key,
            )
        return self._embedding_manager

    def get_query_embedding(self, query: str, query_id: str) -> Optional[np.ndarray]:
        """Generate embedding for a query with SQLite cache."""
        try:
            embedding = self.embedding_manager.get_query_embedding(
                query_id=query_id,
                query_text=query,
            )
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logging.error(
                f"[QueryDocumentFilter:get_query_embedding] "
                f"Error generating query embedding: {e}"
            )
            return None

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
    
    def filter_and_rank_documents(
        self,
        query: str,
        doc_dict: Dict[str, Any],
        query_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Filter and rank documents by query relevance.
        
        Handles doc_id format mismatch (e.g., "0-0" vs "0").
        
        Args:
            query: The query to filter documents for
            doc_dict: Dictionary of documents {doc_id: [doc_text, source_info]}
        
        Returns:
            List of (doc_id, similarity_score) tuples, sorted by score descending
        """
        cache_query_id = query_id
        if cache_query_id is None:
            digest = hashlib.md5(query.encode("utf-8")).hexdigest()[:16]
            cache_query_id = f"schema_tailor_{digest}"

        # Get query embedding
        query_embedding = self.get_query_embedding(query, query_id=cache_query_id)
        if query_embedding is None:
            logging.warning(
                "[QueryDocumentFilter:filter_and_rank_documents] "
                "Could not generate query embedding, returning documents in random order"
            )
            doc_ids = list(doc_dict.keys())
            random.shuffle(doc_ids)
            return [(doc_id, 0.0) for doc_id in doc_ids]
        
        # Load/generate doc embeddings from SQLite cache.
        doc_loader = self._build_doc_loader(doc_dict)
        doc_embeddings = self.embedding_manager.get_doc_embeddings(
            loader=doc_loader,
            doc_ids=list(doc_dict.keys()),
        )

        # Calculate similarity scores for all documents.
        scores: List[Tuple[str, float]] = []
        missing_count = 0
        
        for doc_id in doc_dict:
            emb = doc_embeddings.get(doc_id)
            if emb is not None:
                sim = EmbeddingManager.cosine_similarity(query_embedding.tolist(), emb)
                scores.append((doc_id, float(sim)))
            else:
                # Document embedding not found, assign low score
                scores.append((doc_id, -1.0))
                missing_count += 1
        
        if missing_count > 0:
            logging.warning(
                f"[QueryDocumentFilter:filter_and_rank_documents] "
                f"{missing_count}/{len(doc_dict)} documents missing embeddings"
            )
        
        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter based on top_k or threshold
        if self.top_k is not None:
            scores = scores[:self.top_k]
        elif self.similarity_threshold > 0:
            scores = [(doc_id, sim) for doc_id, sim in scores if sim >= self.similarity_threshold]
        
        logging.info(
            f"[QueryDocumentFilter:filter_and_rank_documents] "
            f"Filtered {len(doc_dict)} documents to {len(scores)} "
            f"(top_k={self.top_k}, threshold={self.similarity_threshold})"
        )
        
        return scores


class SchemaTailor:
    """
    Query-Specific Schema Refinement with Adaptive Sampling.
    
    Implements the second pass of schema discovery:
    1. Load general schema from first pass
    2. Filter/rank documents by query relevance using embeddings
    3. Process documents one by one with LLM to refine schema
    4. Use entropy-based adaptive sampling for early stopping
    """
    
    # Map mode to Prompt class
    PROMPT_CLASS_MAP = {
        "cgpt": PromptGPT,
        "deepseek": PromptDeepSeek,
        "together": PromptTogether,
        "siliconflow": PromptSiliconFlow,
        "gemini": PromptGemini,
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        api_key: Optional[str] = None
    ):
        """
        Initialize the Schema Tailor.
        
        Args:
            config: Configuration dictionary
            api_key: API key for LLM
        """
        self.config = config
        self.api_key = api_key
        self.mode = config.get("mode", "deepseek")
        self.param_str = config.get("res_param_str", "schema_tailor")
        
        # Initialize prompt for tailoring
        tailor_prompt_path = config.get("tailor_prompt_path", "prompts/schema_tailor_1_0.txt")
        PromptClass = self.PROMPT_CLASS_MAP.get(self.mode, PromptGPT)
        
        self.tailor_prompt = PromptClass(
            self.mode,
            tailor_prompt_path,
            llm_model=config.get("llm_model", "gpt-4o-mini"),
            api_key=api_key
        )
        
        # Initialize document filter
        adaptive_config = config.get("adaptive_sampling", {})
        embedding_model = adaptive_config.get("embedding_model", "gemini-embedding-001")
        dataset_db_path = config.get("dataset_db_path")
        
        # Document filtering configuration
        filter_config = config.get("document_filter", {})
        self.doc_filter = QueryDocumentFilter(
            embedding_model=embedding_model,
            api_key=api_key,
            top_k=filter_config.get("top_k"),
            similarity_threshold=filter_config.get("similarity_threshold", 0.0),
            dataset_db_path=dataset_db_path,
        )
        
        # Adaptive sampling parameters
        theta = adaptive_config.get("entropy_threshold", 0.05)
        m = adaptive_config.get("streak_limit", 5)
        n_min = adaptive_config.get("min_docs", 10)
        delta = adaptive_config.get("failure_probability", 0.05)
        epsilon = adaptive_config.get("epsilon", 0.05)
        
        # For schema tailoring, we disable probabilistic stopping by default
        # because we start with a complete schema (high feature count),
        # which makes the probabilistic condition too strict.
        # Use stability-based stopping (entropy streak) only.
        enable_probabilistic = adaptive_config.get("enable_probabilistic_stop", False)
        
        # Initialize adaptive sampler with entropy-based stopping
        self.adaptive_sampler = AdaptiveSampler(
            theta=theta,
            m=m,
            n_min=n_min,
            delta=delta,
            epsilon=epsilon,
            enable_probabilistic_stop=enable_probabilistic
        )
        
        logging.info(
            f"[SchemaTailor:__init__] Initialized with mode={self.mode}, "
            f"theta={theta}, m={m}, n_min={n_min}"
        )
    
    def load_general_schema(self, schema_path: Path) -> Optional[List[Dict]]:
        """Load general schema from a JSON file (list format)."""
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            logging.info(
                f"[SchemaTailor:load_general_schema] "
                f"Loaded general schema with {len(schema)} tables"
            )
            return schema
        except Exception as e:
            logging.error(
                f"[SchemaTailor:load_general_schema] "
                f"Error loading schema: {e}"
            )
            return None
    
    def load_general_schema_from_results(self, results_path: Path) -> Optional[List[Dict]]:
        """
        Extract general schema from schema_gen output results file.
        """
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            if not results:
                return None
            
            doc_indices = [int(k) for k in results.keys() if k.isdigit()]
            if not doc_indices:
                return None
            
            last_doc_idx = str(max(doc_indices))
            schema = results[last_doc_idx].get("log", [])
            
            logging.info(
                f"[SchemaTailor:load_general_schema_from_results] "
                f"Extracted schema from {results_path} (doc {last_doc_idx}): {len(schema)} tables"
            )
            return schema
            
        except Exception as e:
            logging.error(
                f"[SchemaTailor:load_general_schema_from_results] "
                f"Error loading schema from results: {e}"
            )
            return None
    
    def load_queries(self, query_path: Path) -> Dict[str, Dict]:
        """Load queries from queries.json file."""
        try:
            with open(query_path, "r", encoding="utf-8") as f:
                queries = json.load(f)
            logging.info(
                f"[SchemaTailor:load_queries] "
                f"Loaded {len(queries)} queries from {query_path}"
            )
            return queries
        except Exception as e:
            logging.error(
                f"[SchemaTailor:load_queries] "
                f"Error loading queries: {e}"
            )
            return {}
    
    def prepare_tailor_input(
        self,
        schema: List[Dict],
        query: str,
        document: str
    ) -> Dict[str, Any]:
        """
        Prepare input for the tailoring LLM prompt.
        
        Args:
            schema: Current schema state
            query: The query to tailor for
            document: The document to process
        
        Returns:
            Input dictionary for the LLM prompt
        """
        # Build schema with document as example
        schema_with_examples = []
        for table in schema:
            schema_with_examples.append({
                "Schema Name": table.get("Schema Name", ""),
                "Attributes": table.get("Attributes", []),
                "Example Documents": [document]  # Include current document as example
            })
        
        return {
            "Schema": schema_with_examples,
            "Query": query
        }
    
    def process_single_document(
        self,
        current_schema: List[Dict],
        query: str,
        document: str,
        retry_count: int = 0
    ) -> Optional[List[Dict]]:
        """
        Process a single document to refine the schema.
        
        Args:
            current_schema: Current schema state
            query: The query to tailor for
            document: The document to process
            retry_count: Current retry count
        
        Returns:
            Updated schema, or None on error
        """
        input_json = self.prepare_tailor_input(current_schema, query, document)
        
        try:
            result_str = self.tailor_prompt(
                msg="New Input:\n" + json.dumps(input_json, indent=2)
            ).strip()
            
            result = json.loads(result_str)
            updated_schema = result.get("Updated Schema", result.get("Schema", None))
            
            if updated_schema is None:
                logging.warning(
                    f"[SchemaTailor:process_single_document] "
                    f"No 'Updated Schema' in response"
                )
                return None
            
            return updated_schema
            
        except json.JSONDecodeError as e:
            logging.warning(
                f"[SchemaTailor:process_single_document] "
                f"JSON parse error (retry {retry_count}): {e}"
            )
            return None
        except Exception as e:
            logging.warning(
                f"[SchemaTailor:process_single_document] "
                f"Error (retry {retry_count}): {e}"
            )
            return None
    
    def process_query_with_adaptive_sampling(
        self,
        doc_dict: Dict[str, Any],
        query: str,
        general_schema: List[Dict],
        out_root: Path,
        query_id: str
    ) -> Dict[str, Any]:
        """
        Process a query with adaptive sampling using entropy-based stopping.
        
        This is the main processing loop that:
        1. Filters documents by query relevance
        2. Processes documents one by one with LLM
        3. Tracks schema entropy after each document
        4. Stops early when entropy stabilizes
        
        Args:
            doc_dict: Dictionary of documents
            query: The query to process
            general_schema: Starting schema from first pass
            out_root: Output directory for results
            query_id: Query identifier
        
        Returns:
            Dictionary with results and statistics
        """
        # Reset adaptive sampler
        self.adaptive_sampler.reset()
        
        # Step 1: Filter and rank documents by query relevance
        total_docs = len(doc_dict)
        ranked_docs = self.doc_filter.filter_and_rank_documents(
            query,
            doc_dict,
            query_id=query_id,
        )
        
        # Log filter results
        logging.info(
            "[SchemaTailor:process_query_with_adaptive_sampling] "
            "Document filtering results: total=%s filtered=%s top_k=%s threshold=%s",
            total_docs,
            len(ranked_docs),
            self.doc_filter.top_k,
            self.doc_filter.similarity_threshold,
        )
        
        logging.info(
            f"[SchemaTailor:process_query_with_adaptive_sampling] "
            f"Starting adaptive sampling with {len(ranked_docs)} ranked documents (filtered from {total_docs})"
        )
        
        # Initialize state
        current_schema = general_schema.copy() if general_schema else []
        results = {}
        stopped_early = False
        docs_processed = 0
        retry_count = 0
        max_retries = 5
        
        # Progress bar
        progress_bar = tqdm(total=len(ranked_docs), desc=f"Processing {query_id} (Adaptive)")
        
        # Step 2: Process documents one by one with adaptive sampling
        for doc_id, similarity_score in ranked_docs:
            # Get document text
            doc_content = doc_dict.get(doc_id)
            if doc_content is None:
                progress_bar.update(1)
                continue
            
            if isinstance(doc_content, list) and len(doc_content) > 0:
                doc_text = doc_content[0]
            elif isinstance(doc_content, str):
                doc_text = doc_content
            else:
                progress_bar.update(1)
                continue
            
            # Process document with LLM
            updated_schema = self.process_single_document(
                current_schema, query, doc_text, retry_count
            )
            
            # Handle errors with retry
            if updated_schema is None:
                retry_count += 1
                if retry_count > max_retries:
                    logging.warning(
                        f"[SchemaTailor:process_query_with_adaptive_sampling] "
                        f"Max retries exceeded for doc {doc_id}, skipping"
                    )
                    retry_count = 0
                    progress_bar.update(1)
                    continue
                else:
                    # Retry same document
                    continue
            
            # Successful processing
            retry_count = 0
            current_schema = updated_schema
            docs_processed += 1
            
            # Store result
            results[doc_id] = {
                "schema": updated_schema,
                "similarity_score": similarity_score
            }
            
            # Step 3: Check adaptive stopping condition using entropy
            should_continue = self.adaptive_sampler.should_continue(current_schema)
            
            if not should_continue:
                stopped_early = True
                logging.info(
                    f"[SchemaTailor:process_query_with_adaptive_sampling] "
                    f"Early stopping at document {docs_processed}/{len(ranked_docs)}: "
                    f"{self.adaptive_sampler.get_stop_reason()}"
                )
                progress_bar.update(1)
                break
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Get statistics
        stats = self.adaptive_sampler.get_statistics()
        stats["total_documents"] = len(doc_dict)
        stats["filtered_documents"] = len(ranked_docs)
        stats["documents_processed"] = docs_processed
        stats["stopped_early"] = stopped_early
        
        # Save final tailored schema
        tailored_schema_path = out_root / PATH_TEMPLATES.schema_query_tailored(query_id, self.param_str)
        with open(tailored_schema_path, "w", encoding="utf-8") as f:
            json.dump(current_schema, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_path = out_root / f"{tailored_schema_path.stem}_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logging.info(
            f"[SchemaTailor:process_query_with_adaptive_sampling] "
            f"Completed: {docs_processed} docs processed, "
            f"stopped_early={stopped_early}, "
            f"final schema has {len(current_schema)} tables"
        )
        
        return {
            "query_id": query_id,
            "query": query,
            "general_schema_tables": len(general_schema),
            "tailored_schema_tables": len(current_schema),
            "tailored_schema": current_schema,
            "documents_processed": docs_processed,
            "stopped_early": stopped_early,
            "stats": stats
        }
    
    def __call__(
        self,
        data_root: Path,
        out_root: Path,
        general_schema: List[Dict],
        queries_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run schema tailoring for all queries in a dataset.
        
        Args:
            data_root: Dataset root directory
            out_root: Output directory
            general_schema: General schema from first pass
            queries_path: Path to queries.json (default: data_root/queries.json)
        
        Returns:
            Dictionary with results for all queries
        """
        data_root = Path(data_root)
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        
        if general_schema is None:
            return {"error": "No general schema provided"}
        
        # Load queries
        if queries_path is None:
            queries_path = data_root / "queries.json"
        queries = self.load_queries(queries_path)
        if not queries:
            return {"error": "No queries found"}
        
        # Load document dictionary
        doc_dict_path = data_root / "doc_dict.json"
        if doc_dict_path.exists():
            with open(doc_dict_path, "r", encoding="utf-8") as f:
                doc_dict = json.load(f)
        else:
            logging.error(f"[SchemaTailor] doc_dict.json not found at {doc_dict_path}")
            return {"error": "doc_dict.json not found"}
        
        # Process each query with adaptive sampling
        self.doc_filter.set_dataset_db_path(data_root / "__schema_tailor_anchor__.db")
        results = {}
        for query_id, query_info in queries.items():
            query_text = query_info.get("query", "")
            
            logging.info(
                f"[SchemaTailor] Processing query {query_id}: {query_text[:100]}..."
            )
            
            result = self.process_query_with_adaptive_sampling(
                doc_dict, query_text, general_schema, out_root, query_id
            )
            results[query_id] = result
        
        # Save summary
        summary_path = out_root / "tailor_summary.json"
        summary = {
            qid: {
                "query_id": r["query_id"],
                "query": r["query"],
                "general_schema_tables": r["general_schema_tables"],
                "tailored_schema_tables": r["tailored_schema_tables"],
                "documents_processed": r["documents_processed"],
                "stopped_early": r["stopped_early"]
            }
            for qid, r in results.items()
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logging.info(
            f"[SchemaTailor] Completed tailoring for {len(results)} queries"
        )
        
        return results

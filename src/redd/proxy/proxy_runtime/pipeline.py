"""
Proxy Pipeline for cost-optimized data extraction.

This module provides the end-to-end predicate-proxy pipeline:

1. Load ground truth schema and SQL query from dataset
2. Parse SQL WHERE clause into individual attribute filters
3. Create predicate proxies for each filter (or use pre-trained classifiers)
4. Run cascaded filtering on documents
5. Use DataPop (LLM) to extract attributes for documents that pass all filters

Example Usage:
    ```python
    from redd.proxy.proxy_runtime.pipeline import ProxyPipeline, ProxyPipelineConfig
    
    # Configure pipeline
    config = ProxyPipelineConfig(
        dataset_path="spider_sqlite/college_2",
        query_id="Q1",
        llm_mode="gemini",
        llm_model="gemini-2.5-flash-lite",
    )
    
    # Create and run pipeline
    pipeline = ProxyPipeline(config)
    results = pipeline.run()
    
    # Access results
    for doc_id, extraction in results.extractions.items():
        print(f"Document {doc_id}: {extraction}")
    ```
"""

from __future__ import annotations

import logging
import math
import time
import json
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Tuple
)

import numpy as np

from redd.core.data_loader import DataLoaderBase, create_data_loader
from redd.core.embedding import EmbeddingManager
from redd.core.utils.sql_filter_parser import (
    SQLFilterParser,
    AttributePredicate,
    predicates_to_filter_dict,
)
from redd.proxy.predicate_proxy.factory import PredicateProxyFactory
from redd.proxy.proxy_runtime.oracle import DataPopOracle, GoldenOracle
from redd.proxy.proxy_runtime.types import PipelineResults, ProxyPipelineConfig
from .executor import (
    ProxyExecutor,
    ProxyRuntimeConfig,
    ConformalProxy,
    EmbeddingProxy,
    DocumentBatch,
    HardNegative,
)

__all__ = [
    "ProxyPipeline",
    "ProxyPipelineConfig",
    "PipelineResults",
    "DataPopOracle",
    "GoldenOracle",
]


class ProxyPipeline:
    """
    Complete predicate-proxy pipeline for cost-optimized data extraction.
    
    Orchestrates:
    1. Loading dataset and query
    2. Parsing SQL predicates
    3. Creating proxies (embedding-based or classifier-based)
    4. Running proxy filtering
    5. LLM extraction for passing documents
    """
    
    def __init__(self, config: ProxyPipelineConfig):
        """
        Initialize proxy pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Components (lazy-loaded)
        self._data_loader: Optional[DataLoaderBase] = None
        self._sql_parser: Optional[SQLFilterParser] = None
        self._oracle: Optional[DataPopOracle] = None
        self._embedding_manager: Optional[EmbeddingManager] = None
        self._proxy_factory: Optional[PredicateProxyFactory] = None
        
        # Cached data
        self._query_info: Optional[Dict[str, Any]] = None
        self._schema: Optional[List[Dict[str, Any]]] = None
        self._predicates: Optional[List[AttributePredicate]] = None
        
        logging.info(f"[ProxyPipeline] Initialized for dataset={config.dataset_path}, "
                    f"query={config.query_id}")
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def data_loader(self) -> DataLoaderBase:
        """Get or create data loader."""
        if self._data_loader is None:
            doc_dir = self.config.dataset_path
            if self.config.task_db_name:
                doc_dir = f"{doc_dir}/{self.config.task_db_name}"
            self._data_loader = create_data_loader(
                doc_dir=doc_dir,
                loader_type="sqlite",
                config={"data_main": self.config.data_main}
            )
            logging.info(f"[ProxyPipeline] Created data loader with "
                        f"{self._data_loader.num_docs} documents")
        return self._data_loader
    
    @property
    def sql_parser(self) -> SQLFilterParser:
        """Get or create SQL parser."""
        if self._sql_parser is None:
            self._sql_parser = SQLFilterParser(strip_table_aliases=True)
        return self._sql_parser
    
    @property
    def oracle(self) -> DataPopOracle | GoldenOracle:
        """Get or create extraction oracle (LLM or ground truth)."""
        if self._oracle is None:
            if str(self.config.llm_mode).lower() in {"ground_truth", "gt"}:
                self._oracle = GoldenOracle(self.data_loader)
            else:
                self._oracle = DataPopOracle(
                    mode=self.config.llm_mode,
                    llm_model=self.config.llm_model,
                    api_key=self.config.api_key
                )
        return self._oracle
    
    @property
    def embedding_manager(self) -> EmbeddingManager:
        """Get or create embedding manager."""
        if self._embedding_manager is None:
            self._embedding_manager = EmbeddingManager(
                loader=self.data_loader,
                model=self.config.embedding_model,
                api_key=self.config.embedding_api_key,
            )
        return self._embedding_manager

    @property
    def proxy_factory(self) -> PredicateProxyFactory:
        """Get or create proxy factory."""
        if self._proxy_factory is None:
            self._proxy_factory = PredicateProxyFactory(
                config=self.config,
                embedding_manager=self.embedding_manager
            )
        return self._proxy_factory

    @property
    def guard_factory(self) -> PredicateProxyFactory:
        """Backward-compatible alias for legacy proxy-factory naming."""
        return self.proxy_factory
    
    # ========================================================================
    # Loading Methods
    # ========================================================================
    
    def load_query(self) -> Dict[str, Any]:
        """Load query information from dataset."""
        if self._query_info is not None:
            return self._query_info
        
        # Try loading from data loader
        query_info = self.data_loader.get_query_info(self.config.query_id)
        
        if query_info is None:
            # Try loading from generated_queries.json
            gen_queries_path = self.data_loader.data_root / "generated_queries.json"
            if gen_queries_path.exists():
                with open(gen_queries_path, 'r', encoding='utf-8') as f:
                    gen_data = json.load(f)
                
                queries = gen_data.get("queries", [])
                for q in queries:
                    if q.get("id") == self.config.query_id:
                        query_info = {
                            "query": q.get("natural_language", ""),
                            "sql": q.get("sql", ""),
                            "attributes": q.get("golden_schema", {}).get("columns", []),
                            "tables": q.get("golden_schema", {}).get("tables", []),
                            "difficulty": q.get("difficulty", ""),
                        }
                        break
        
        if query_info is None:
            raise ValueError(f"Query {self.config.query_id} not found in dataset")
        
        self._query_info = query_info
        logging.info(f"[ProxyPipeline] Loaded query: {query_info.get('query', '')[:100]}...")
        
        return self._query_info
    
    def load_schema(self) -> List[Dict[str, Any]]:
        """Load schema for the query."""
        if self._schema is not None:
            return self._schema
        
        # Try query-specific schema first
        self._schema = self.data_loader.load_schema_query(self.config.query_id)
        
        # Fall back to general schema
        if not self._schema:
            self._schema = self.data_loader.load_schema_general()
        
        logging.info(f"[ProxyPipeline] Loaded schema with {len(self._schema)} tables")
        return self._schema
    
    def parse_predicates(self) -> List[AttributePredicate]:
        """Parse SQL query into attribute predicates."""
        if self._predicates is not None:
            return self._predicates
        
        query_info = self.load_query()
        sql = query_info.get("sql", "")
        
        if not sql:
            logging.warning("[ProxyPipeline] No SQL query found, returning empty predicates")
            self._predicates = []
            return self._predicates
        
        self._predicates = self.sql_parser.parse(sql)
        
        logging.info(f"[ProxyPipeline] Parsed {len(self._predicates)} predicates from SQL:")
        for pred in self._predicates:
            logging.info(f"  - {pred}")
        
        return self._predicates
    
    # ========================================================================
    # Embedding Methods
    # ========================================================================

    def _fallback_embedding_dim(self) -> int:
        """Best-effort fallback dim when no embedding vector is available."""
        model = str(self.config.embedding_model).lower()
        if "text-embedding-3-large" in model:
            return 3072
        if "text-embedding" in model or "ada-002" in model:
            return 1536
        return 768
    
    def compute_embeddings(
        self, 
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
        show_progress: bool = True,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Compute or load embeddings for a list of texts.

        Uses the SQLite-backed EmbeddingManager from core.embedding.
        """
        del show_progress, use_cache  # kept for backward-compatible signature

        if not texts:
            return np.zeros((0, self._fallback_embedding_dim()), dtype=np.float32)

        # For ad-hoc texts (no stable doc IDs), embed directly.
        if not doc_ids:
            vectors = []
            for text in texts:
                vectors.append(
                    np.array(self.embedding_manager.embed_single(text), dtype=np.float32)
                )
            return np.stack(vectors, axis=0)

        if len(doc_ids) != len(texts):
            logging.warning(
                f"[ProxyPipeline] doc_ids/texts length mismatch: {len(doc_ids)} vs {len(texts)}"
            )

        emb_dict = self.embedding_manager.get_doc_embeddings(
            loader=self.data_loader,
            doc_ids=doc_ids,
        )

        inferred_dim = None
        if emb_dict:
            first_vec = next(iter(emb_dict.values()))
            inferred_dim = len(first_vec)
        else:
            inferred_dim = self._fallback_embedding_dim()

        vectors = []
        for doc_id, text in zip(doc_ids, texts):
            emb = emb_dict.get(doc_id)
            if emb is None:
                try:
                    emb = self.embedding_manager.embed_single(text)
                except Exception as e:
                    logging.warning(
                        f"[ProxyPipeline] Failed to embed doc_id={doc_id}; using zeros: {e}"
                    )
                    emb = [0.0] * inferred_dim
            vectors.append(np.array(emb, dtype=np.float32))

        return np.stack(vectors, axis=0)

    # ========================================================================
    # Main Run Method
    # ========================================================================
    
    def run(self) -> PipelineResults:
        """
        Run the complete proxy pipeline.
        
        Returns:
            PipelineResults with extracted data and statistics
        """
        start_time = time.time()
        results = PipelineResults()
        
        # 1. Load query and schema
        query_info = self.load_query()
        schema = self.load_schema()
        
        results.query_id = self.config.query_id
        results.query_text = query_info.get("query", "")
        results.sql = query_info.get("sql", "")
        
        # 2. Parse SQL predicates
        predicates = self.parse_predicates()
        results.predicates = predicates
        
        # Create predicate functions for validation
        predicate_fns = predicates_to_filter_dict(predicates)
        
        # 3. Get attributes to extract
        attributes = self._get_extraction_attributes(query_info, predicates, schema)
        logging.info(f"[ProxyPipeline] Will extract attributes: {attributes}")
        # 4. Apply global split: first N docs are training, remaining docs are test.
        all_doc_ids = list(self.data_loader.doc_ids)
        results.total_documents = len(all_doc_ids)
        train_size = max(int(getattr(self.config, "training_data_count", 100)), 0)
        train_doc_ids = all_doc_ids[:train_size]
        test_doc_ids = all_doc_ids[train_size:]

        if self.config.max_documents:
            test_doc_ids = test_doc_ids[:self.config.max_documents]

        logging.info(
            f"[ProxyPipeline] Global split for run(): training={len(train_doc_ids)}, "
            f"test={len(test_doc_ids)}, training_data_count={train_size}"
        )

        subset_results = self.run_for_documents(
            doc_ids=test_doc_ids,
            train_doc_ids=train_doc_ids,
            predicates=predicates,
            table_schema=schema,
            query_text=results.query_text,
            data_loader=self.data_loader,
            extra_proxies=None,
        )

        results.documents_processed = subset_results.documents_processed
        results.documents_passed_proxies = subset_results.documents_passed_proxies
        results.documents_extracted = subset_results.documents_extracted
        results.extractions = subset_results.extractions
        results.execution_stats = subset_results.execution_stats
        results.hard_negatives = subset_results.hard_negatives
        results.embedding_time_seconds = subset_results.embedding_time_seconds
        results.proxy_time_seconds = subset_results.proxy_time_seconds
        results.extraction_time_seconds = subset_results.extraction_time_seconds
        results.total_time_seconds = time.time() - start_time
        
        # 10. Save results if output dir configured
        if self.config.output_dir:
            output_path = Path(self.config.output_dir) / f"ccg_results_{self.config.query_id}.json"
            results.save(output_path)
        
        # Log summary
        logging.info(f"[ProxyPipeline] Complete!")
        logging.info(f"  - Documents: {results.total_documents}")
        logging.info(f"  - Passed proxies: {results.documents_passed_proxies}")
        logging.info(f"  - Extracted: {results.documents_extracted}")
        logging.info(f"  - Hard negatives: {len(results.hard_negatives)}")
        logging.info(f"  - Total time: {results.total_time_seconds:.2f}s")
        logging.info(f"  - Oracle calls: {self.oracle.call_count}")
        
        return results

    def run_for_documents(
        self,
        doc_ids: List[str],
        train_doc_ids: Optional[List[str]],
        predicates: List[AttributePredicate],
        table_schema: Union[Dict[str, Any], List[Dict[str, Any]]],
        query_text: Optional[str] = None,
        data_loader: Optional[DataLoaderBase] = None,
        extra_proxies: Optional[List] = None,
    ) -> PipelineResults:
        """
        Run the proxy runtime for a subset of documents with table-specific predicates.
        
        Used by the unified datapop proxy-runtime strategy to process each table's documents sequentially.
        
        Args:
            doc_ids: Test document IDs to process
            train_doc_ids: Training document IDs for learned-proxy fitting
            predicates: Predicates for this table (from group_predicates_by_table)
            table_schema: Schema for this table (single table dict or list with one)
            query_text: Natural language query (for predicate proxies). Uses config query if None.
            data_loader: Data loader for fetching doc text. Uses pipeline's loader if None.
            extra_proxies: Optional list of additional proxies (e.g., JoinResolver). Prepended
                to standard predicate proxies for high rejection efficiency.
            
        Returns:
            PipelineResults with extractions keyed by doc_id
        """
        start_time = time.time()
        results = PipelineResults()
        
        loader = data_loader or self.data_loader
        results.query_id = self.config.query_id
        results.query_text = query_text if query_text is not None else self.load_query().get("query", "")
        
        # Normalize table_schema to list
        schema_list = [table_schema] if isinstance(table_schema, dict) else table_schema
        if not schema_list:
            schema_list = self.load_schema()
        
        results.predicates = predicates
        predicate_fns = predicates_to_filter_dict(predicates)
        
        # Get attributes from table schema
        attributes = []
        for s in schema_list:
            for attr_info in s.get("Attributes", []):
                attr_name = attr_info.get("Attribute Name", "") if isinstance(attr_info, dict) else str(attr_info)
                if attr_name:
                    attributes.append(attr_name)
        attributes = list(dict.fromkeys(attributes))  # dedupe preserving order
        
        if not attributes:
            # Fallback: use predicate attributes
            attributes = [p.attribute for p in predicates]
        
        # Ensure deterministic order and deduplicate IDs.
        doc_ids = list(dict.fromkeys(doc_ids))
        train_doc_ids = list(dict.fromkeys(train_doc_ids or []))
        if not getattr(self.config, "allow_train_test_overlap", False):
            test_doc_id_set = set(doc_ids)
            train_doc_ids = [
                doc_id for doc_id in train_doc_ids if doc_id not in test_doc_id_set
            ]
        fit_train_doc_ids, calibration_doc_ids = self._split_training_doc_ids(train_doc_ids)

        logging.info(
            f"[ProxyPipeline] run_for_documents: test={len(doc_ids)}, "
            f"train_fit={len(fit_train_doc_ids)}, train_calib={len(calibration_doc_ids)}, "
            f"predicates={len(predicates)}, attrs={attributes}"
        )
        
        results.total_documents = len(doc_ids)
        
        if not doc_ids:
            results.total_time_seconds = time.time() - start_time
            return results
        
        # Load test document texts
        doc_texts = []
        for doc_id in doc_ids:
            doc_texts.append(self._load_doc_text(loader, doc_id))
        
        # Determine whether learned proxies can use a fine-tuned classifier.
        use_finetuned = False
        if self.config.use_learned_proxies:
            try:
                from redd.proxy.predicate_proxy.factory import FINETUNED_AVAILABLE
                use_finetuned = (
                    getattr(self.config, "use_finetuned_learned_proxies", True)
                    and FINETUNED_AVAILABLE
                )
            except ImportError:
                use_finetuned = False

        # Compute test embeddings only when needed.
        need_embeddings = self.config.use_embedding_proxies or self.config.save_hard_negatives
        if self.config.use_learned_proxies and fit_train_doc_ids and not use_finetuned:
            # LogisticRegression learned proxies need training embeddings.
            need_embeddings = True

        emb_start = time.time()
        if need_embeddings:
            embeddings = self.compute_embeddings(doc_texts, doc_ids=doc_ids)
        else:
            dim = self._fallback_embedding_dim()
            embeddings = np.zeros((len(doc_texts), dim), dtype=np.float32)
            logging.info("[ProxyPipeline] Skipping embedding API for test docs")
        results.embedding_time_seconds = time.time() - emb_start
        
        # Create or train predicate proxies.
        proxies = []
        batch = DocumentBatch(doc_ids=doc_ids, documents=doc_texts, embeddings=embeddings)

        if predicates:
            if self.config.use_learned_proxies and fit_train_doc_ids:
                label_doc_ids = list(dict.fromkeys(fit_train_doc_ids + calibration_doc_ids))
                label_doc_texts: Dict[str, str] = {
                    doc_id: self._load_doc_text(loader, doc_id) for doc_id in label_doc_ids
                }

                train_docs = [label_doc_texts[doc_id] for doc_id in fit_train_doc_ids]
                train_extractions: Dict[str, Dict[str, Any]] = {}
                calibration_extractions: Dict[str, Dict[str, Any]] = {}
                schema_dict = self._prepare_schema_dict(schema_list, attributes)
                for label_doc_id in label_doc_ids:
                    label_doc_text = label_doc_texts[label_doc_id]
                    try:
                        ext = self.oracle.extract(
                            document=label_doc_text,
                            schema=schema_dict,
                            attributes=attributes,
                            doc_id=label_doc_id,
                        )
                        if label_doc_id in fit_train_doc_ids:
                            train_extractions[label_doc_id] = ext
                        if label_doc_id in calibration_doc_ids:
                            calibration_extractions[label_doc_id] = ext
                    except Exception as e:
                        logging.warning(f"[ProxyPipeline] Training extraction failed for {label_doc_id}: {e}")

                if train_extractions:
                    if use_finetuned:
                        model_name = getattr(
                            self.config,
                            "finetuned_model",
                            "knowledgator/gliclass-instruct-large-v1.0",
                        )
                        logging.info(
                            f"[ProxyPipeline] Training learned proxies (model={model_name})"
                        )
                        proxies = self.proxy_factory.train_proxies_finetuned(
                            predicates=predicates,
                            query_text=results.query_text,
                            documents=train_docs,
                            doc_ids=fit_train_doc_ids,
                            extractions=train_extractions,
                            model_name=model_name,
                            epochs=getattr(self.config, "finetuned_epochs", 3),
                            learning_rate=getattr(self.config, "finetuned_learning_rate", 2e-5),
                        )
                    else:
                        if getattr(self.config, "use_finetuned_learned_proxies", True):
                            logging.warning(
                                "[ProxyPipeline] Fine-tuned proxies unavailable; "
                                "using embedding-based LogisticRegression proxies."
                            )
                        train_embeddings = self.compute_embeddings(
                            train_docs, doc_ids=fit_train_doc_ids
                        )
                        proxies = self.proxy_factory.train_proxies(
                            predicates=predicates,
                            query_text=results.query_text,
                            doc_embeddings=train_embeddings,
                            doc_ids=fit_train_doc_ids,
                            extractions=train_extractions,
                        )
                    if proxies and calibration_doc_ids:
                        calibration_docs = [
                            label_doc_texts[doc_id] for doc_id in calibration_doc_ids
                        ]
                        calibration_embeddings = None
                        if any(not getattr(g, "uses_documents", False) for g in proxies):
                            calibration_embeddings = self.compute_embeddings(
                                calibration_docs, doc_ids=calibration_doc_ids
                            )
                        self._recalibrate_learned_proxies(
                            proxies=proxies,
                            predicate_fns=predicate_fns,
                            calibration_doc_ids=calibration_doc_ids,
                            calibration_docs=calibration_docs,
                            calibration_extractions=calibration_extractions,
                            calibration_embeddings=calibration_embeddings,
                        )
                else:
                    logging.warning(
                        "[ProxyPipeline] No training labels extracted; learned proxies not trained."
                    )

            # Fallback to non-learned embedding proxies when configured.
            if not proxies and self.config.use_embedding_proxies:
                expected_dim = int(embeddings.shape[1]) if embeddings.size > 0 else None
                proxies = self.proxy_factory.create_proxies(
                    predicates, results.query_text, expected_embedding_dim=expected_dim
                )
            elif self.config.use_learned_proxies and not fit_train_doc_ids:
                logging.warning(
                    "[ProxyPipeline] No training docs for learned proxies; running without learned proxy training."
                )
        
        # Merge join-resolution proxies ahead of predicate proxies for fail-fast execution.
        if extra_proxies:
            proxies = list(extra_proxies) + proxies
            logging.info(f"[ProxyPipeline] Added {len(extra_proxies)} extra proxies (e.g., join resolution)")
        
        # Run proxy filtering
        proxy_start = time.time()
        if proxies:
            executor = ProxyExecutor(guards=proxies, llm_oracle=None, config=ProxyRuntimeConfig(
                collect_hard_negatives=self.config.save_hard_negatives, verbose=self.config.verbose
            ))
            proxy_results, exec_stats = executor.process_batch(batch)
            results.execution_stats = exec_stats
            # Only documents that passed all configured proxies are in proxy_results.
            passed_doc_ids = [r["doc_id"] for r in proxy_results if r.get("passed_guards", False)]
            learned_proxy_prefixes = ("learned_", "finetuned_", "llm_")
            has_learned_proxy = any(
                str(getattr(proxy, "name", "")).startswith(learned_proxy_prefixes)
                for proxy in proxies
            )
            if has_learned_proxy and len(passed_doc_ids) == 0 and len(batch.doc_ids) > 0:
                logging.warning(
                    "[ProxyPipeline] All documents rejected by learned proxies "
                    "(table/query-level all-zero risk). Falling back to conservative filtering."
                )
                fallback_proxies: List[Any] = []
                if self.config.use_embedding_proxies:
                    expected_dim = int(embeddings.shape[1]) if embeddings.size > 0 else None
                    fallback_proxies = self.proxy_factory.create_proxies(
                        predicates, results.query_text, expected_embedding_dim=expected_dim
                    )
                if extra_proxies:
                    fallback_proxies = list(extra_proxies) + fallback_proxies

                if fallback_proxies:
                    fallback_executor = ProxyExecutor(
                        guards=fallback_proxies,
                        llm_oracle=None,
                        config=ProxyRuntimeConfig(
                            collect_hard_negatives=self.config.save_hard_negatives,
                            verbose=self.config.verbose,
                        ),
                    )
                    fallback_results, fallback_stats = fallback_executor.process_batch(batch)
                    results.execution_stats = fallback_stats
                    passed_doc_ids = [
                        r["doc_id"] for r in fallback_results if r.get("passed_guards", False)
                    ]
                    logging.warning(
                        f"[ProxyPipeline] Fallback filtering kept {len(passed_doc_ids)}/{len(batch.doc_ids)} docs."
                    )
                else:
                    passed_doc_ids = list(batch.doc_ids)
                    logging.warning(
                        "[ProxyPipeline] No conservative fallback proxies available; passing all docs."
                    )
            current_exec_stats = results.execution_stats
            if current_exec_stats and len(passed_doc_ids) != current_exec_stats.passed_to_oracle:
                raise AssertionError(
                    f"Guard filter mismatch: passed_doc_ids={len(passed_doc_ids)} "
                    f"!= exec_stats.passed_to_oracle={current_exec_stats.passed_to_oracle}"
                )
            if current_exec_stats and current_exec_stats.proxy_stats:
                for proxy_name, stat in current_exec_stats.proxy_stats.items():
                    evaluated = int(stat.get("evaluated", 0))
                    passed = int(stat.get("passed", 0))
                    rejected = int(stat.get("rejected", 0))
                    pass_rate = (passed / evaluated) if evaluated > 0 else 0.0
                    logging.info(
                        f"[ProxyPipeline] Proxy stats {proxy_name}: "
                        f"evaluated={evaluated}, passed={passed}, rejected={rejected}, "
                        f"pass_rate={pass_rate:.4f}"
                    )
        else:
            passed_doc_ids = batch.doc_ids
        results.proxy_time_seconds = time.time() - proxy_start
        results.documents_processed = len(doc_ids)
        results.documents_passed_proxies = len(passed_doc_ids)
        
        # Extract for passing documents
        extract_start = time.time()
        schema_dict = self._prepare_schema_dict(schema_list, attributes)
        batch_lookup = {did: (txt, emb) for did, txt, emb in zip(batch.doc_ids, batch.documents, batch.embeddings)}
        
        for doc_id in passed_doc_ids:
            doc_text, doc_emb = batch_lookup.get(doc_id, ("", None))
            try:
                extracted = self.oracle.extract(document=doc_text, schema=schema_dict, attributes=attributes, doc_id=doc_id)
                all_passed, per_attr = self.oracle.check_predicates(extracted_values=extracted, predicates=predicate_fns)
                if all_passed:
                    results.extractions[doc_id] = extracted
                    results.documents_extracted += 1
                elif self.config.save_hard_negatives:
                    failed_attrs = [k for k, v in per_attr.items() if not v]
                    for attr in failed_attrs:
                        results.hard_negatives.append(HardNegative(
                            doc_id=doc_id, document=doc_text, embedding=doc_emb,
                            failed_attribute=attr, extracted_value=extracted.get(attr),
                            expected_predicate=str(predicate_fns.get(attr)), proxy_scores={}
                        ))
            except Exception as e:
                logging.warning(f"[ProxyPipeline] Extraction failed for {doc_id}: {e}")
        
        results.extraction_time_seconds = time.time() - extract_start
        results.total_time_seconds = time.time() - start_time
        return results
    
    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _load_doc_text(self, loader: DataLoaderBase, doc_id: str) -> str:
        """Best-effort document text loader for a single doc ID."""
        doc_text = loader.get_doc_text(doc_id) if hasattr(loader, "get_doc_text") else ""
        if doc_text:
            return doc_text
        tup = loader.get_doc(doc_id) if hasattr(loader, "get_doc") else None
        return tup[0] if tup else ""

    def _split_training_doc_ids(self, train_doc_ids: List[str]) -> Tuple[List[str], List[str]]:
        """
        Split training docs into fit/calibration subsets.

        Web/main datapop policy: all proxy fit/calibration data must come from
        the global training prefix (first ``training_data_count`` documents).
        To fully use this prefix without extra minimum constraints, use the
        same training pool for both fit and calibration.
        """
        ids = list(train_doc_ids)
        if not ids:
            return [], []
        return ids, ids

    def _proxy_attribute_name(self, proxy_name: str) -> str:
        """Extract attribute name from proxy identifier."""
        name = str(proxy_name or "")
        for prefix in ("learned_", "finetuned_", "llm_"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def _guard_attribute_name(self, proxy_name: str) -> str:
        """Backward-compatible alias for the legacy helper name."""
        return self._proxy_attribute_name(proxy_name)

    def _compute_per_proxy_target_recall(self, proxy_count: int) -> float:
        """
        Convert global target recall to per-proxy target recall.

        Uses Bonferroni-style alpha allocation:
          alpha_global = 1 - target_recall
          alpha_i = alpha_global / proxy_count
          target_i = 1 - alpha_i
        """
        count = max(int(proxy_count), 1)
        try:
            target_global = float(self.config.target_recall)
        except (TypeError, ValueError):
            target_global = 0.95
        target_global = min(max(target_global, 0.0), 0.999)
        alpha_global = 1.0 - target_global
        alpha_i = alpha_global / count
        target_i = 1.0 - alpha_i
        return min(max(target_i, 0.0), 0.999)

    def _compute_per_guard_target_recall(self, guard_count: int) -> float:
        """Backward-compatible alias for the legacy helper name."""
        return self._compute_per_proxy_target_recall(guard_count)

    def _recalibrate_learned_proxies(
        self,
        proxies: List[Any],
        predicate_fns: Dict[str, Any],
        calibration_doc_ids: List[str],
        calibration_docs: List[str],
        calibration_extractions: Dict[str, Dict[str, Any]],
        calibration_embeddings: Optional[np.ndarray],
    ) -> None:
        """Re-calibrate learned proxy thresholds on held-out calibration docs."""
        if not proxies or not calibration_doc_ids:
            return

        predicate_proxies = [
            proxy
            for proxy in proxies
            if self._proxy_attribute_name(getattr(proxy, "name", "")) in predicate_fns
        ]
        if not predicate_proxies:
            return

        per_proxy_target = self._compute_per_proxy_target_recall(len(predicate_proxies))
        global_target = 1.0 - (1.0 - per_proxy_target) * max(len(predicate_proxies), 1)
        global_target = min(max(global_target, 0.0), 0.999)
        quantile = 1.0 - per_proxy_target
        quantile = min(max(quantile, 0.0), 1.0)
        logging.info(
            "[ProxyPipeline] Re-calibrating %d predicate proxies on %d held-out docs "
            "(global_target=%.4f, per_proxy_target=%.4f)",
            len(predicate_proxies),
            len(calibration_doc_ids),
            global_target,
            per_proxy_target,
        )

        for proxy in predicate_proxies:
            attr = self._proxy_attribute_name(getattr(proxy, "name", ""))
            pred_fn = predicate_fns.get(attr)
            if pred_fn is None:
                continue

            try:
                if getattr(proxy, "uses_documents", False):
                    try:
                        scores, _ = proxy.evaluate_documents(
                            calibration_docs, doc_ids=calibration_doc_ids
                        )
                    except TypeError:
                        scores, _ = proxy.evaluate_documents(calibration_docs)
                else:
                    if calibration_embeddings is None:
                        continue
                    scores, _ = proxy.evaluate(calibration_embeddings)
            except Exception as e:
                logging.warning(
                    f"[ProxyPipeline] Calibration scoring failed for proxy {proxy.name}: {e}"
                )
                continue

            y_calib: List[int] = []
            for doc_id in calibration_doc_ids:
                value = calibration_extractions.get(doc_id, {}).get(attr)
                try:
                    y_calib.append(1 if (value is not None and pred_fn(value)) else 0)
                except Exception:
                    y_calib.append(0)
            y_calib_arr = np.array(y_calib, dtype=np.int32)
            scores_arr = np.array(scores, dtype=np.float32)
            pos_scores = scores_arr[y_calib_arr == 1]
            if len(pos_scores) <= 0:
                logging.info(
                    f"[ProxyPipeline] Proxy {proxy.name}: no positive calibration samples; "
                    "keeping existing threshold."
                )
                continue

            old_threshold = float(getattr(proxy, "threshold", 0.5))
            new_threshold = max(float(np.quantile(pos_scores, quantile)), 0.01)
            proxy.threshold = new_threshold
            logging.info(
                f"[ProxyPipeline] Proxy {proxy.name}: threshold recalibrated "
                f"{old_threshold:.4f} -> {new_threshold:.4f} "
                f"(positives={len(pos_scores)}, quantile={quantile:.4f})"
            )

    def _recalibrate_learned_guards(
        self,
        guards: List[Any],
        predicate_fns: Dict[str, Any],
        calibration_doc_ids: List[str],
        calibration_docs: List[str],
        calibration_extractions: Dict[str, Dict[str, Any]],
        calibration_embeddings: Optional[np.ndarray],
    ) -> None:
        """Backward-compatible alias for the legacy helper name."""
        self._recalibrate_learned_proxies(
            proxies=guards,
            predicate_fns=predicate_fns,
            calibration_doc_ids=calibration_doc_ids,
            calibration_docs=calibration_docs,
            calibration_extractions=calibration_extractions,
            calibration_embeddings=calibration_embeddings,
        )
    
    def _get_extraction_attributes(
        self,
        query_info: Dict[str, Any],
        predicates: List[AttributePredicate],
        schema: List[Dict[str, Any]]
    ) -> List[str]:
        """Get list of attributes to extract."""
        attributes = set()
        
        # From query info
        query_attrs = query_info.get("attributes", [])
        for attr in query_attrs:
            if isinstance(attr, str):
                # Handle table.column format
                if '.' in attr:
                    attr = attr.split('.')[-1]
                attributes.add(attr)
        
        # From predicates
        for pred in predicates:
            attributes.add(pred.attribute)
        
        # From schema
        for table in schema:
            for attr_info in table.get("Attributes", []):
                attr_name = attr_info.get("Attribute Name", "")
                if attr_name:
                    attributes.add(attr_name)
        
        return list(attributes)
    
    def _prepare_schema_dict(
        self,
        schema: List[Dict[str, Any]],
        attributes: List[str]
    ) -> Dict[str, Any]:
        """Prepare schema dictionary for LLM extraction."""
        if not schema:
            return {"attributes": attributes}
        
        # Use first table for now (could be smarter about table selection)
        if len(schema) > 0:
            return schema[0]
        
        return {"Schema Name": "extraction", "Attributes": attributes}

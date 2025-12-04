"""
Mixin class for integrating adaptive sampling into schema generators.

This module provides a mixin class that can be added to existing schema
generator classes to enable adaptive sampling functionality without
major refactoring.
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional
from .adaptive_sampler import AdaptiveSampler
from .schema_entropy import SchemaEntropyCalculator


class AdaptiveSamplingMixin:
    """
    Mixin class to add adaptive sampling capabilities to schema generators.
    
    This mixin provides:
    - Adaptive sampler initialization from config
    - Modified document processing loop with early stopping
    - Statistics tracking and reporting
    
    Usage:
        class SchemaGenWithAdaptive(AdaptiveSamplingMixin, SchemaGenGPT):
            pass
    
    Or simply add to existing class:
        class SchemaGenGPT(SchemaGenBasic, AdaptiveSamplingMixin):
            ...
    """
    
    def init_adaptive_sampling(self, config: Dict[str, Any]):
        """
        Initialize adaptive sampling from configuration.
        
        Expected config structure:
            adaptive_sampling:
                enabled: true
                theta: 0.05          # Entropy threshold
                m: 3                 # Stability streak threshold
                n_min: 5             # Minimum samples
                delta: 0.1           # Failure probability
                epsilon: 0.05        # Minimum feature frequency
                probabilistic_stop: true
        
        Args:
            config: Configuration dictionary
        """
        adaptive_config = config.get("adaptive_sampling", {})
        self.adaptive_enabled = adaptive_config.get("enabled", False)
        
        if not self.adaptive_enabled:
            logging.info(
                f"[{self.__class__.__name__}:init_adaptive_sampling] "
                "Adaptive sampling disabled"
            )
            self.adaptive_sampler = None
            return
        
        # Extract parameters with defaults
        theta = adaptive_config.get("theta", 0.05)
        m = adaptive_config.get("m", 3)
        n_min = adaptive_config.get("n_min", 5)
        delta = adaptive_config.get("delta", 0.1)
        epsilon = adaptive_config.get("epsilon", 0.05)
        probabilistic_stop = adaptive_config.get("probabilistic_stop", True)
        
        # Initialize adaptive sampler
        self.adaptive_sampler = AdaptiveSampler(
            theta=theta,
            m=m,
            n_min=n_min,
            delta=delta,
            epsilon=epsilon,
            enable_probabilistic_stop=probabilistic_stop
        )
        
        logging.info(
            f"[{self.__class__.__name__}:init_adaptive_sampling] "
            f"Adaptive sampling enabled with parameters: "
            f"theta={theta}, m={m}, n_min={n_min}, delta={delta}, "
            f"epsilon={epsilon}, probabilistic_stop={probabilistic_stop}"
        )
    
    def process_documents_adaptive(
        self, 
        doc_dict, 
        query, 
        res_dict, 
        log_init, 
        general_schema, 
        res_path, 
        pgbar_name
    ):
        """
        Process documents with adaptive sampling (replacement for process_documents).
        
        This method replaces the standard process_documents method with one that
        implements adaptive early stopping based on schema entropy.
        
        Args:
            doc_dict: Dictionary of documents {doc_id: [doc_text, source_info]}
            query: Query string (if applicable)
            res_dict: Existing results dictionary
            log_init: Initial log/schema state
            general_schema: General schema context
            res_path: Path to save results
            pgbar_name: Name for progress bar
        """
        from tqdm import tqdm
        
        if not self.adaptive_enabled or self.adaptive_sampler is None:
            # Fall back to regular processing
            logging.warning(
                f"[{self.__class__.__name__}:process_documents_adaptive] "
                "Adaptive sampling not enabled, using standard processing"
            )
            return self.process_documents(
                doc_dict, query, res_dict, log_init, general_schema, 
                res_path, pgbar_name
            )
        
        # Reset adaptive sampler for this query/dataset
        self.adaptive_sampler.reset()
        
        num_doc = len(doc_dict)
        i, cnt = 0, 0
        progress_bar = tqdm(total=num_doc, desc=f"Processing {pgbar_name} (Adaptive)")
        
        logging.info(
            f"[{self.__class__.__name__}:process_documents_adaptive] "
            f"Start adaptive processing: query={query}"
        )
        logging.info(
            f"[{self.__class__.__name__}:process_documents_adaptive] "
            f"Documents: processed={len(res_dict)}/{num_doc}"
        )
        
        stopped_early = False
        
        while i < num_doc:
            # Skip already processed documents
            if str(i) in res_dict:
                i, cnt = i + 1, 0
                progress_bar.update(1)
                
                # Still check adaptive stopping even for skipped docs
                # Use the existing result's log as current schema
                if "log" in res_dict[str(i-1)]:
                    current_schema = res_dict[str(i-1)]["log"]
                    if not self.adaptive_sampler.should_continue(current_schema):
                        stopped_early = True
                        logging.info(
                            f"[{self.__class__.__name__}:process_documents_adaptive] "
                            f"Early stopping triggered at document {i}/{num_doc}"
                        )
                        break
                
                continue
            
            # Prepare and process document
            log = log_init if i == 0 else res_dict[str(i-1)]["log"]
            input_json = self.prepare_input_json(
                doc_dict, i, query, log, general_schema
            )
            out_dict = self.process_single_document(input_json, cnt, i)
            result_data = self.extract_result_data(out_dict)
            
            # Handle processing errors
            if not result_data or len(result_data["log"]) < len(log):
                cnt += 1
                if cnt > 10:
                    if not result_data:
                        logging.warning(
                            f"[{self.__class__.__name__}:process_documents_adaptive] "
                            f"Failed to process document {i} after {cnt} retries!"
                        )
                    else:
                        logging.warning(
                            f"[{self.__class__.__name__}:process_documents_adaptive] "
                            f"Schema num decrease, retry_count {cnt}, doc_index {i}"
                        )
                    exit()
                continue
            
            # Save result
            res_dict[str(i)] = result_data
            self.save_results(res_path, res_dict)
            
            # Check adaptive stopping criterion
            current_schema = result_data["log"]
            if not self.adaptive_sampler.should_continue(current_schema):
                stopped_early = True
                i += 1  # Count this document as processed
                progress_bar.update(1)
                logging.info(
                    f"[{self.__class__.__name__}:process_documents_adaptive] "
                    f"Early stopping triggered at document {i}/{num_doc}"
                )
                break
            
            i, cnt = i + 1, 0
            progress_bar.update(1)
            logging.info(
                f"[{self.__class__.__name__}:process_documents_adaptive] "
                f"Processed document {i}/{num_doc}"
            )
        
        progress_bar.close()
        
        # Save adaptive sampling statistics
        stats = self.adaptive_sampler.get_statistics()
        stats["total_documents"] = num_doc
        stats["stopped_early"] = stopped_early
        stats["documents_saved"] = num_doc - i if stopped_early else 0
        
        self._save_adaptive_stats(res_path, stats)
        
        # Log summary
        if stopped_early:
            logging.info(
                f"[{self.__class__.__name__}:process_documents_adaptive] "
                f"Finished with early stopping: processed {i}/{num_doc} documents "
                f"(saved {num_doc - i} documents)"
            )
            logging.info(
                f"[{self.__class__.__name__}:process_documents_adaptive] "
                f"Stop reason: {self.adaptive_sampler.get_stop_reason()}"
            )
        else:
            logging.info(
                f"[{self.__class__.__name__}:process_documents_adaptive] "
                f"Finished processing all {num_doc} documents (no early stopping)"
            )
        
        logging.info(
            f"[{self.__class__.__name__}:process_documents_adaptive] "
            f"Final entropy statistics: {stats['entropy_statistics']}"
        )
    
    def _save_adaptive_stats(self, res_path: Path, stats: Dict[str, Any]):
        """
        Save adaptive sampling statistics alongside results.
        
        Args:
            res_path: Path to results file
            stats: Statistics dictionary
        """
        res_path = Path(res_path)
        stats_path = res_path.parent / f"{res_path.stem}_adaptive_stats.json"
        
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            
            logging.info(
                f"[{self.__class__.__name__}:_save_adaptive_stats] "
                f"Saved adaptive statistics to {stats_path}"
            )
        except Exception as e:
            logging.error(
                f"[{self.__class__.__name__}:_save_adaptive_stats] "
                f"Failed to save adaptive statistics: {e}"
            )
    
    def get_adaptive_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get current adaptive sampling statistics.
        
        Returns:
            Statistics dictionary, or None if adaptive sampling not enabled
        """
        if not self.adaptive_enabled or self.adaptive_sampler is None:
            return None
        
        return self.adaptive_sampler.get_statistics()


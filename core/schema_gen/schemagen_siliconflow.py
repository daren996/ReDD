import logging

from .schemagen_gpt import SchemaGenGPT
from ..utils.prompt_utils import PromptSiliconFlow, get_api_key


class SchemaGenSiliconFlow(SchemaGenGPT):
    
    def __init__(self, config, api_key=None):
        # Initialize basic config
        self.config = config
        self.mode = config["mode"]
        
        # Data loader configuration (inherited from parent)
        self.loader_type = config.get("data_loader_type", "spider")
        self.loader_config = config.get("data_loader_config", {})
        
        # Initialize adaptive sampling
        self.init_adaptive_sampling(config)
        
        if self.mode == "siliconflow":
            try:
                self.log_init_file = config.get("log_init_file")
                self.doc_cluster_file = config.get("doc_cluster_file")
                
                prompt_config = config["prompt"]
                self.param_str = config["res_param_str"]
                self.general_param_str = config.get("general_param_str")
            
                api_key = get_api_key(config, self.mode, api_key)
                self.prompt = PromptSiliconFlow(
                        self.mode,
                        prompt_config["prompt_path"],
                        llm_model=config["llm_model"], 
                        api_key=api_key
                    )
            except Exception as e:
                logging.error(f"[{self.__class__.__name__}:__init__] When initializing, Error: {e}")
                exit()

    def apply_prompt(self, attr_msg):
        return self.prompt(msg=attr_msg).strip()

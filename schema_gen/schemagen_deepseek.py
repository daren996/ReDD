
import logging

from schema_gen.schemagen_gpt import SchemaGenGPT
from utils.prompt_utils import PromptDeepSeek


class SchemaGenDeepSeek(SchemaGenGPT):

    def __init__(self, config, api_key=None):

        super().__init__(config)

        if api_key:
            self.api_key = api_key
        elif "api_key" in config:
            self.api_key = config["api_key"]
        else: 
            logging.error("API key is required for DeepSeek mode.")
            exit()

        if config["mode"] == "deepseek":
            try:
                self.log_init_file = config["log_init_file"] if "log_init_file" in config else None
                self.doc_cluster_file = config["doc_cluster_file"] if "doc_cluster_file" in config else None
                param_str_tmp = "mdl{model}_prm%s".format(model=config["llm_model"])
                self.param_str = param_str_tmp % config["prompt"]["prompt_version"]
                self.general_param_str = param_str_tmp % config["prompt"]["general_prompt_version"] if "general_prompt_version" in config["prompt"] else None
                self.prompt = PromptDeepSeek(
                        config["mode"],
                        config["prompt"]["prompt_path"],
                        llm_model=config["llm_model"], 
                        api_key=self.api_key
                    )
            except Exception as e:
                logging.error(f"When initializing {self.__class__.__name__}, Error: {e}")
                exit()

    def apply_prompt(self, attr_msg):
        return self.prompt(msg=attr_msg).strip()
    
import os
import logging

from openai import OpenAI

from utils.utils import gpt_completion


class PromptGPT:
    def __init__(self, mode, prompt_path, llm_model, api_key=None):
        self.mode = mode
        self.prompt_path = prompt_path
        self.llm_model = llm_model
        self.prompt = None
        self.load_prompts()
        if mode == "cgpt":
            self.client = OpenAI()
        elif mode == "deepseek":
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    def load_prompts(self):        
        if not os.path.exists(self.prompt_path):
            logging.error(f"Prompt `{self.prompt_path}` does not exist.")
            exit()
        try:
            self.prompt = open(self.prompt_path, "r").read()
        except FileNotFoundError:
            raise FileNotFoundError("prompts load failed")

    def __call__(self, msg: str, **kwargs) -> str:
        """
        This method sends the messages with a prompt to a GPT model and returns the generated completion.

        - arg msg: Input message, serialized as a JSON string.
        - output: The generated completion, serialized as a JSON string.
        The input msg and output both adhere to a JSON format template specified in the configuration file.
        """
        attr_msg = [{"role": "user", "content": self.prompt + "\n\n" + msg}]
        return gpt_completion(self.mode, self.client, attr_msg, self.llm_model, **kwargs)

    def __str__(self):
        return self.prompt


class PromptDeepSeek(PromptGPT):
    def __init__(self, mode, prompt_path, llm_model, api_key=None):
        super().__init__(mode, prompt_path, llm_model, api_key=api_key)


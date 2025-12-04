import os
import logging
import time

from openai import OpenAI, RateLimitError, APIError


def llm_completion(mode, client, messages, model, **wargs):
    """
    Generate chat completion using LLM with automatic retry on rate limits.
    
    Uses default model parameters (temperature=1.0, top_p=1.0, seed=0, max_tokens=8192)
    for consistent behavior across different LLM providers.
    
    Params:
        mode:               API mode ("cgpt", "deepseek", "together", "siliconflow")
        client:             OpenAI client instance
        messages:           [{"role": ..., "content": ...}, ...]
        model:              Model name (e.g., gpt-4o, deepseek-chat)
        response_format:    "json_object" or "text" (default: "json_object")
        max_retries:        Maximum number of retries for rate limit errors (default: 5)
        wait_time:          Wait time in seconds between retries (default: 61.0)
    """
    max_retries = wargs.get("max_retries", 5)
    wait_time = wargs.get("wait_time", 61.0)
    response_format = wargs.get("response_format", "json_object")
    
    for attempt in range(max_retries + 1):
        try:
            if mode == "cgpt":
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": response_format},
                )
            elif mode == "deepseek":
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": response_format},
                )
            elif mode == "together":
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": response_format},
                )
            elif mode == "siliconflow":
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": response_format},
                )
            else:
                logging.error(f"Invalid mode: {mode}")
                raise ValueError(f"Invalid mode: {mode}")
            
            return completion.choices[0].message.content
            
        except RateLimitError as e:
            if attempt == max_retries:
                logging.error(f"[llm_completion] Rate limit exceeded after {max_retries} retries. Error: {e}")
                raise
            
            # Calculate wait time with exponential backoff
            logging.info(
                f"[llm_completion] Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                f"Waiting {wait_time:.1f}s before retry... Error: {e}"
            )
            time.sleep(wait_time)
            
        except APIError as e:
            # For other API errors, retry once with a short wait
            if attempt < max_retries:
                logging.warning(
                    f"[llm_completion] API error (attempt {attempt + 1}). "
                    f"Waiting {wait_time:.1f}s before retry... Error: {e}"
                )
                time.sleep(wait_time)
            else:
                logging.error(f"[llm_completion] API error after retries: {e}")
                raise


def get_api_key(config, mode, api_key=None):
    """
    Get API key with priority: provided api_key > config file > environment variable > error
    
    Args:
        config: Configuration dictionary
        mode: API mode ("cgpt", "deepseek", "together", "siliconflow")
        api_key: Explicitly provided API key
    
    Returns:
        str: API key
        
    Raises:
        SystemExit: If API key cannot be found
    """
    # Priority 1: Explicitly provided api_key
    if api_key:
        return api_key
    
    # Priority 2: Config file
    if "api_key" in config:
        return config["api_key"]
    
    # Priority 3: Environment variables
    env_var_map = {
        "cgpt": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY", 
        "together": "TOGETHER_API_KEY",
        "siliconflow": "SILICONFLOW_API_KEY"
    }
    
    env_var = env_var_map.get(mode)
    if env_var:
        env_api_key = os.getenv(env_var)
        if env_api_key:
            return env_api_key
    
    # Priority 4: Return error
    logging.error(f"API key is required for {mode} mode. Please provide it via:")
    logging.error(f"  1. Command line argument --api-key")
    logging.error(f"  2. Config file 'api_key' field")
    logging.error(f"  3. Environment variable {env_var}")
    exit(1)


class PromptBase:
    def __init__(self, mode, prompt_path, llm_model, api_key=None):
        self.mode = mode
        self.prompt_path = prompt_path
        self.llm_model = llm_model
        self.prompt = None
        self.load_prompts()
        resolved_api_key = get_api_key({}, mode, api_key)
        
        if mode == "cgpt":
            self.client = OpenAI(api_key=resolved_api_key)
        elif mode == "deepseek":
            self.client = OpenAI(api_key=resolved_api_key, base_url="https://api.deepseek.com")
        elif mode == "together":
            self.client = OpenAI(api_key=resolved_api_key, base_url="https://api.together.ai/v1")
        elif mode == "siliconflow":
            self.client = OpenAI(api_key=resolved_api_key, base_url="https://api.siliconflow.com/v1")
        else:
            logging.error(f"Invalid mode: {mode}")
    
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
        This method sends the messages with a prompt to a LLM and returns the generated completion.

        - arg msg: Input message, serialized as a JSON string.
        - output: The generated completion, serialized as a JSON string.
        The input msg and output both adhere to a JSON format template specified in the configuration file.
        """
        attr_msg = [{"role": "user", "content": self.prompt + "\n\n" + msg}]
        return llm_completion(self.mode, self.client, attr_msg, self.llm_model, **kwargs)

    def __str__(self):
        return self.prompt


class PromptGPT(PromptBase):
    def __init__(self, mode, prompt_path, llm_model, api_key=None):
        super().__init__(mode, prompt_path, llm_model, api_key=api_key)


class PromptDeepSeek(PromptBase):
    def __init__(self, mode, prompt_path, llm_model, api_key=None):
        super().__init__(mode, prompt_path, llm_model, api_key=api_key)


class PromptTogether(PromptBase):
    def __init__(self, mode, prompt_path, llm_model, api_key=None):
        super().__init__(mode, prompt_path, llm_model, api_key=api_key)


class PromptSiliconFlow(PromptBase):
    def __init__(self, mode, prompt_path, llm_model, api_key=None):
        super().__init__(mode, prompt_path, llm_model, api_key=api_key)



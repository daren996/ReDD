import json
import re
import ast
import logging

from openai import OpenAI


def gpt_completion(mode, client, messages, model, **wargs):
    """
    Generate chat completion using ChatGPT models.
    models: 
        gpt-4o        $2.50 / 1M input tokens
                      $10.00 / 1M output tokens
        gpt-4o-mini   $0.150 / 1M input tokens
                      $0.600 / 1M output tokens
    Params:
        model:              e.g., gpt-4o, gpt-4o-mini
        messages:           [{"role": ..., "content": ...}, ...]
        temperature:        output randomness
        top_p:              nucleus sampling
        response_format:    "json_object" or "text".
        seed:               random seed
    """
    if mode == "cgpt":
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=wargs["temperature"] if "temperature" in wargs else 1.0,
            top_p=wargs["top_p"] if "top_p" in wargs else 1.0,
            frequency_penalty=wargs["frequency_penalty"] if "frequency_penalty" in wargs else 0,
            presence_penalty=wargs["presence_penalty"] if "presence_penalty" in wargs else 0,
            max_tokens=wargs["max_tokens"] if "max_tokens" in wargs else 8192,
            response_format={"type": wargs["response_format"]} if "response_format" in wargs else {"type": "json_object"},
            seed=wargs["seed"] if "seed" in wargs else 0
        )
    elif mode == "deepseek":
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": wargs["response_format"]} if "response_format" in wargs else {"type": "json_object"},
        )
    else:
        logging.error(f"Invalid mode: {mode}")
    return completion.choices[0].message.content

def gpt_embeddings(string, model="text-embedding-3-small"):
    """
    Generate Text embeddings using ChatGPT models.
    models: 
        text-embedding-3-small    $0.020 / 1M tokens
        text-embedding-3-large    $0.130 / 1M tokens
        ada v2                    $0.100 / 1M tokens
    """
    client_gpt = OpenAI()
    embeddings = client_gpt.embeddings.create(
        input=string,
        model=model
    )  # TODO: make it support DeekSeek
    return embeddings.data[0].embedding

def extract_json_block(raw_text):
    """
    Extracts JSON block from a string.
    """
    pattern = rf"```json(.*?)```"
    match = re.search(pattern, raw_text, re.S | re.I)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(match.group(1).strip())
            except:
                return None
    return None

def compute_tp_fp_fn(golden: set, output: set):
    golden = set(golden)
    output = set(output)
    _tp = len(golden & output)    # Correctly extracted attributes
    _fp = len(output - golden)    # Incorrectly extracted attributes
    _fn = len(golden - output)    # Missing attributes
    return _tp, _fp, _fn

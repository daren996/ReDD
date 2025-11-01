import os
import json


class SchemaGenBasic:
    def __init__(self, config):
        self.config = config

    def __call__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def save_results(self, res_path, res_dict):
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=2)
    
    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
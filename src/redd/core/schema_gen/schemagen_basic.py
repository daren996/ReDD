import os
import json
import logging


class SchemaGenBasic:
    def __init__(self, config):
        self.config = config
        
        # Data loader will be set by subclasses
        self.loader = None

    def __call__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def save_results(self, res_path, res_dict, encoding="utf-8"):
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        with open(res_path, "w", encoding=encoding) as f:
            json.dump(res_dict, f, indent=2)
    
    def load_json(self, file_path, encoding="utf-8"):
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)
    
    def load_processed_res(self, res_path):
        """Load processed results from <res_path>"""
        res_dict = dict()
        if os.path.exists(res_path):
            res_dict = self.load_json(res_path)
        return res_dict
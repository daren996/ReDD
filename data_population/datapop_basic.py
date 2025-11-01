import os
import json


class DataPopBasic:
    def __init__(self, config):
        self.config = config

    def __call__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def load_processed_res(self, res_path):
        """ Load Processed Results from <res_path> """
        res_dict = dict()
        if os.path.exists(res_path):
            res_dict = self.load_json(res_path)
        return res_dict
    
    def save_results(self, res_path, res_dict):
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=2)
    
    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __str__(self):
        return f"{self.__class__.__name__}: \n{self.param_str}"
    
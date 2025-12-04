

from .datapop_deepseek import DataPopDeepSeek


class DataPopGPT(DataPopDeepSeek):

    def __init__(self, config, api_key=None):
        super().__init__(config)
        raise NotImplementedError("DataPopGPT is not implemented yet.")




from data_population.datapop_deepseek import DataPopDeepSeek


class DataPopGPT(DataPopDeepSeek):

    def __init__(self, config):
        super().__init__(config)
        raise NotImplementedError("DataPopGPT is not implemented yet.")


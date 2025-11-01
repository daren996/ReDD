import os
import re
import string
import logging
import random
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, Sampler

from data_loader.data_loader_spider import SpiderDatasetLoader


# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LazyHiddenStatesDataset(Dataset):
    """
    按需加载隐藏状态数据的 Dataset
    在初始化时根据 eval_output 和 res_data_dict 构建一个样本索引, 每个样本对应于: 
      - 表 (table) 样本: doc id、"table" 类型、无属性名、标签由 eval_output["table"] 决定
      - 属性 (attr) 样本: doc id、"attr" 类型、属性名称、标签由 eval_output["attr"][attr] 决定
    在 __getitem__ 中, 根据样本信息按需加载磁盘上的 .pt 文件, 并提取指定 layer_index 对应的 hidden state
    分块缓存隐藏状态的 Dataset, 实现有限大小的块缓存 (例如最多缓存 2 个 block)
    每个 block (例如每 100 个 did) 首次计算后将结果保存到文件, 下次直接从文件加载
    """
    def __init__(self, trainer, loader: SpiderDatasetLoader, res_data_dict, states_dir, eval_output, 
                 layer_index, sample_dids, qid, cache_batch_size=100):
        self.trainer = trainer
        self.loader = loader
        self.res_data_dict = res_data_dict          # 从 res_tabular_data_xxx.json 加载的结果
        self.states_dir = states_dir                # 隐藏状态文件所在目录
        self.eval_output = eval_output              # 评价信息, 决定标签
        self.layer_index = layer_index              # 要抽取的层索引
        self.sample_dids = set(sample_dids)         # 仅处理指定 doc id 集合
        self.row_size = len(sample_dids)            # 用于构造缓存文件名的参数
        self.cache_batch_size = cache_batch_size    # 每个缓存文件包含的 did 数量 (默认 100)

        # 根据 schema_general 构建表名称到 token 序列的映射
        self.table2tokens = {}
        for item in loader.load_schema_general():
            table = item["Schema Name"]
            self.table2tokens[table] = self.trainer.tokenizer.encode(table, add_special_tokens=False)
        # 根据 schema_query 构建表内各属性到 token 序列的映射
        self.table2attr2tokens = {}
        for item in loader.load_schema_query(qid):
            table = item["Schema Name"]
            if table in self.table2tokens:
                self.table2attr2tokens[table] = {}
                for attr_item in item["Attributes"]:
                    attr = attr_item["Attribute Name"]
                    self.table2attr2tokens[table][attr] = self.trainer.tokenizer.encode(attr, add_special_tokens=False)
        
        # 构建样本索引列表
        self.samples = []
        # 注意: eval_output 的 key 为字符串形式的 doc id
        for did_str, info in self.eval_output.items():
            did = int(did_str)
            if did not in self.sample_dids:
                continue
            # 表样本: 标签约定为 0 表示 eval_output[did]["table"] 为 True, 否则为 1
            table_label = 0 if info["table"] else 1
            self.samples.append((did, "table", None, table_label))
            # 属性样本: 每个属性单独作为一个样本, 标签同理
            for attr, attr_val in info["attr"].items():
                attr_label = 0 if attr_val else 1
                self.samples.append((did, "attr", attr, attr_label))
        self.samples = sorted(self.samples, key=lambda x: x[0])
        self.all_dids = sorted([int(did_str) for did_str in self.eval_output.keys()])

        self.did_to_sampleindices = {}
        for idx, sample in enumerate(self.samples):
            did = sample[0]
            self.did_to_sampleindices.setdefault(did, []).append(idx)

        self.cache = OrderedDict()
        self.max_cache_blocks = 2

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        visited = set()  # 用于记录已访问的索引, 防止无限循环
        while True:
            if index in visited:
                raise ValueError("所有样本都没有有效的 hidden_state!")
            visited.add(index)
            
            did, sample_type, attr, label = self.samples[index]
            block_index = did // self.cache_batch_size
            # logging.info(f"[LazyDataset:__getitem__] 获取样本: did={did}, index={index}, block={block_index}")  # TODO

            # 检查当前 block 是否在缓存中, 若在则移到末尾
            if block_index in self.cache:
                doc_cache = self.cache.pop(block_index)
                self.cache[block_index] = doc_cache
            else:
                if len(self.cache) >= self.max_cache_blocks:
                    self.cache.popitem(last=False)
                doc_cache = self.load_block_cache(block_index)
                self.cache[block_index] = doc_cache
                # logging.info(f"[LazyDataset:__getitem__] 加载 block {block_index} 的缓存. did: {did}")  # TODO

            # 从缓存中获取对应 doc 的 hidden state
            if did not in doc_cache:
                # 如果该 did 不在缓存中 (异常情况), 则直接调用计算函数
                if sample_type == "table":
                    hidden_state = self.get_table_hidden_state(did)
                else:
                    hidden_state = self.get_attr_hidden_state(did, attr)
            else:
                if sample_type == "table":
                    hidden_state = doc_cache[did].get("table", None)
                elif sample_type == "attr":
                    hidden_state = doc_cache[did].get("attr", {}).get(attr, None)
                else:
                    raise ValueError(f"Unknown sample type: {sample_type}")

            # 如果 hidden_state 获取不到, 则跳过, 使用下一个 index
            if hidden_state is None:
                index = (index + 1) % len(self.samples)
                continue
            else:
                return hidden_state.float(), torch.tensor(label, dtype=torch.long)

    def load_block_cache(self, block_index):
        """
        加载指定 block 的缓存:
          1. 根据 all_dids, 获取当前 block 范围内的所有 did
          2. 检查对应缓存文件是否存在, 若存在则加载; 否则计算后保存
        返回一个字典, 结构为 { did: {"table": hidden_state, "attr": { attr: hidden_state, ... } } }
        """
        cache_file = os.path.join(self.states_dir, f"cache_hstates_{block_index}.pt")
        if os.path.exists(cache_file):
            return torch.load(cache_file)

        # 根据 all_dids, 获取当前 block 内的所有 did (假设 all_dids 已经排好序)
        block_start = block_index * self.cache_batch_size
        block_end = block_start + self.cache_batch_size
        block_dids = [did for did in self.all_dids if block_start <= did < block_end]

        doc_cache = {}
        for did in block_dids:
            doc_cache[did] = {}
            doc_cache[did]["table"] = self.get_table_hidden_state(did)
            doc_cache[did]["attr"] = {}
            did_str = str(did)
            if did_str in self.eval_output:
                for attr in self.eval_output[did_str]["attr"]:
                    doc_cache[did]["attr"][attr] = self.get_attr_hidden_state(did, attr)
        torch.save(doc_cache, cache_file)
        logging.info(f"[LazyDataset:load_block_cache] 缓存文件 {cache_file} 不存在, 已计算并保存.")
        return doc_cache
    
    def get_did_hidden_state(self, did):
        """ 根据给定的 did, 返回该 did 对应的所有 hidden states """
        if did not in self.did_to_sampleindices:
            logging.warning(f"did {did} 不存在于数据集中")
            return []
        hidden_states = []
        for idx in self.did_to_sampleindices[did]:
            hs, _ = self[idx]
            hidden_states.append(hs)
        return torch.stack(hidden_states)

    def get_table_hidden_state(self, did):
        state_file = os.path.join(self.states_dir, f"doc-{did}-table.pt")
        if not os.path.exists(state_file):
            logging.warning(f"[LazyDataset:get_table_hidden_state] state file {state_file} does not exist (did: {did}.")
            return None
        state_dict_table = torch.load(state_file, weights_only=False)
        hidden_state = torch.stack([state_dict_table[i]["hidden_states"][self.layer_index] 
                                    for i in range(len(state_dict_table))])
        hidden_state = hidden_state.squeeze(1)  # N x D
        mean_pooled = torch.mean(hidden_state, dim=0)  # D
        max_pooled = torch.max(hidden_state, dim=0)[0]  # D
        return torch.cat([mean_pooled, max_pooled])  # 2D
        
    def get_attr_hidden_state(self, did, attr):
        state_file = os.path.join(self.states_dir, f"doc-{did}-attr-{attr}.pt")
        if not os.path.exists(state_file):
            logging.warning(f"[LazyDataset:get_attr_hidden_state] state file {state_file} does not exist (did: {did}, attr: {attr}).")
            return None
        state_dict_attr = torch.load(state_file, weights_only=False)
        hidden_state = torch.stack([state_dict_attr[i]["hidden_states"][self.layer_index] 
                                    for i in range(len(state_dict_attr))])
        hidden_state = hidden_state.squeeze(1)  # N x D
        mean_pooled = torch.mean(hidden_state, dim=0)  # D
        max_pooled = torch.max(hidden_state, dim=0)[0]  # D
        return torch.cat([mean_pooled, max_pooled])  # 2D

    def get_table_hidden_state_old(self, did):
        res_data = self.res_data_dict[str(did)]
        table_name = res_data["res"]
        if table_name not in self.table2tokens:
            # logging.info(f"[LazyDataset:get_table_hidden_state] Table name {table_name} 未在 token 映射中找到 (did: {did}).")
            return None
        res_table_tokens = self.table2tokens[table_name]
        state_file = os.path.join(self.states_dir, f"doc-{did}-table.pt")
        if not os.path.exists(state_file):
            logging.warning(f"[LazyDataset:get_table_hidden_state] 状态文件 {state_file} 不存在 (did: {did}).")
            return None
        state_dict_table = torch.load(state_file, weights_only=False)
        # 遍历 state_dict_table 找到 token 序列匹配的位置
        for i in range(len(state_dict_table) - len(res_table_tokens) + 1):
            i_tokens = [item["token_id"] for item in state_dict_table[i:i+len(res_table_tokens)]]
            if i_tokens == res_table_tokens:
                return state_dict_table[i]["hidden_states"][self.layer_index]
        # logging.info(f"[LazyDataset:get_table_hidden_state] 未在文件中找到匹配的 table tokens (did: {did}).")
        return None
    
    def get_attr_hidden_state_old(self, did, attr):
        res_data = self.res_data_dict[str(did)]
        table_name = res_data["res"]
        if table_name not in self.table2attr2tokens or attr not in self.table2attr2tokens[table_name]:
            # logging.info(f"[LazyDataset:get_attr_hidden_state] 属性 {attr} 在表 {table_name} 中未找到 token 映射 (did: {did}).")
            return None
        if attr not in res_data["data"]:
            # logging.info(f"[LazyDataset:get_attr_hidden_state] res_data 中缺失属性 {attr} (did: {did}).")
            return None
        res_value = res_data["data"][attr].strip().strip(string.punctuation)
        res_value = re.sub(r'^€', '', res_value)
        res_value_tokens = self.trainer.tokenizer.encode(res_value, add_special_tokens=False)
        state_file = os.path.join(self.states_dir, f"doc-{did}-attr-{attr}.pt")
        if not os.path.exists(state_file):
            logging.warning(f"[LazyDataset:get_attr_hidden_state] 状态文件 {state_file} 不存在 (did: {did}, attr: {attr}).")
            return None
        state_dict_attr = torch.load(state_file, weights_only=False)
        # 遍历 state_dict_attr 查找匹配 token 序列
        for i in range(len(state_dict_attr) - len(res_value_tokens) + 1):
            i_tokens = [item["token_id"] for item in state_dict_attr[i:i+len(res_value_tokens)]]
            if len(res_value_tokens) > 0 and i_tokens[0] == res_value_tokens[0]:
                if i_tokens == res_value_tokens:
                    return state_dict_attr[i]["hidden_states"][self.layer_index]
                # 或者对比 token_text (解码后比较) 
                i_text = ''.join([self.trainer.decode_str(item["token_text"]) for item in state_dict_attr[i:i+len(res_value_tokens)]])
                i_text = i_text.strip().strip(string.punctuation)
                if i_text == res_value:
                    return state_dict_attr[i]["hidden_states"][self.layer_index]
        # logging.info(f"[LazyDataset:get_attr_hidden_state] 未找到匹配的 attribute tokens (did: {did}, attr: {attr}).")
        return None


class SequentialOversampler(Sampler):
    """
    自定义采样器: 对每个类别按原顺序取样, 并对样本较少的类别做 oversample, 
    最终返回一个按顺序排列的索引列表
    """
    def __init__(self, indices, labels):
        self.indices = indices
        self.labels = labels
        # 根据标签分组索引, 保持原来的顺序
        self.class_to_indices = {}
        for idx, label in zip(indices, labels):
            self.class_to_indices.setdefault(label, []).append(idx)
        # 每个类别需要采样的样本数: 取所有类别中样本最多的数量
        self.max_count = max(len(lst) for lst in self.class_to_indices.values())
        
        # 构建 oversampled 索引列表: 对每个类别重复扩充, 使其样本数达到 max_count
        oversampled_indices = []
        for label in sorted(self.class_to_indices.keys()):
            lst = self.class_to_indices[label]
            repeat_factor = self.max_count // len(lst)
            remainder = self.max_count % len(lst)
            new_indices = lst * repeat_factor + lst[:remainder]
            oversampled_indices.extend(new_indices)
        
        # 为保证最终索引顺序与传入的 train_indices 保持一致, 这里对 oversampled_indices 直接排序
        self.oversampled_indices = sorted(oversampled_indices)
        # random.shuffle(self.oversampled_indices)

    def __iter__(self):
        return iter(self.oversampled_indices)

    def __len__(self):
        return len(self.oversampled_indices)
    
import logging
import os
import re
import string
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, Sampler

from redd.core.data_loader import DataLoaderBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LazyHiddenStatesDataset(Dataset):
    """
    Dataset for lazy loading of hidden state data.
    """

    def __init__(
        self,
        trainer,
        loader: DataLoaderBase,
        res_data_dict,
        states_dir,
        eval_output,
        layer_index,
        sample_dids,
        qid,
        cache_batch_size=100,
    ):
        self.trainer = trainer
        self.loader = loader
        self.res_data_dict = res_data_dict
        self.states_dir = states_dir
        self.eval_output = eval_output
        self.layer_index = layer_index
        self.sample_dids = set(sample_dids)
        self.row_size = len(sample_dids)
        self.cache_batch_size = cache_batch_size

        self.table2tokens = {}
        for item in loader.load_schema_general():
            table = item["Schema Name"]
            self.table2tokens[table] = self.trainer.tokenizer.encode(table, add_special_tokens=False)

        self.table2attr2tokens = {}
        for item in loader.load_schema_query(qid):
            table = item["Schema Name"]
            if table in self.table2tokens:
                self.table2attr2tokens[table] = {}
                for attr_item in item["Attributes"]:
                    attr = attr_item["Attribute Name"]
                    self.table2attr2tokens[table][attr] = self.trainer.tokenizer.encode(
                        attr,
                        add_special_tokens=False,
                    )

        self.samples = []
        for did_str, info in self.eval_output.items():
            did = int(did_str)
            if did not in self.sample_dids:
                continue
            table_label = 0 if info["table"] else 1
            self.samples.append((did, "table", None, table_label))
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
        visited = set()
        while True:
            if index in visited:
                raise ValueError("All samples have no valid hidden_state!")
            visited.add(index)

            did, sample_type, attr, label = self.samples[index]
            block_index = did // self.cache_batch_size

            if block_index in self.cache:
                doc_cache = self.cache.pop(block_index)
                self.cache[block_index] = doc_cache
            else:
                if len(self.cache) >= self.max_cache_blocks:
                    self.cache.popitem(last=False)
                doc_cache = self.load_block_cache(block_index)
                self.cache[block_index] = doc_cache

            if did not in doc_cache:
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

            if hidden_state is None:
                index = (index + 1) % len(self.samples)
                continue
            return hidden_state.float(), torch.tensor(label, dtype=torch.long)

    def load_block_cache(self, block_index):
        cache_file = os.path.join(self.states_dir, f"cache_hstates_{block_index}.pt")
        if os.path.exists(cache_file):
            return torch.load(cache_file)

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
        logging.info(
            "[LazyDataset:load_block_cache] Cache file %s does not exist, computed and saved.",
            cache_file,
        )
        return doc_cache

    def get_did_hidden_state(self, did):
        if did not in self.did_to_sampleindices:
            logging.warning("did %s does not exist in dataset", did)
            return []
        hidden_states = []
        for idx in self.did_to_sampleindices[did]:
            hs, _ = self[idx]
            hidden_states.append(hs)
        return torch.stack(hidden_states)

    def get_table_hidden_state(self, did):
        state_file = os.path.join(self.states_dir, f"doc-{did}-table.pt")
        if not os.path.exists(state_file):
            logging.warning(
                "[LazyDataset:get_table_hidden_state] state file %s does not exist (did: %s).",
                state_file,
                did,
            )
            return None
        state_dict_table = torch.load(state_file, weights_only=False)
        hidden_state = torch.stack(
            [state_dict_table[i]["hidden_states"][self.layer_index] for i in range(len(state_dict_table))]
        )
        hidden_state = hidden_state.squeeze(1)
        mean_pooled = torch.mean(hidden_state, dim=0)
        max_pooled = torch.max(hidden_state, dim=0)[0]
        return torch.cat([mean_pooled, max_pooled])

    def get_attr_hidden_state(self, did, attr):
        state_file = os.path.join(self.states_dir, f"doc-{did}-attr-{attr}.pt")
        if not os.path.exists(state_file):
            logging.warning(
                "[LazyDataset:get_attr_hidden_state] state file %s does not exist (did: %s, attr: %s).",
                state_file,
                did,
                attr,
            )
            return None
        state_dict_attr = torch.load(state_file, weights_only=False)
        hidden_state = torch.stack(
            [state_dict_attr[i]["hidden_states"][self.layer_index] for i in range(len(state_dict_attr))]
        )
        hidden_state = hidden_state.squeeze(1)
        mean_pooled = torch.mean(hidden_state, dim=0)
        max_pooled = torch.max(hidden_state, dim=0)[0]
        return torch.cat([mean_pooled, max_pooled])

    def get_table_hidden_state_old(self, did):
        res_data = self.res_data_dict[str(did)]
        table_name = res_data["res"]
        if table_name not in self.table2tokens:
            return None
        res_table_tokens = self.table2tokens[table_name]
        state_file = os.path.join(self.states_dir, f"doc-{did}-table.pt")
        if not os.path.exists(state_file):
            logging.warning(
                "[LazyDataset:get_table_hidden_state] State file %s does not exist (did: %s).",
                state_file,
                did,
            )
            return None
        state_dict_table = torch.load(state_file, weights_only=False)
        for i in range(len(state_dict_table) - len(res_table_tokens) + 1):
            i_tokens = [item["token_id"] for item in state_dict_table[i : i + len(res_table_tokens)]]
            if i_tokens == res_table_tokens:
                return state_dict_table[i]["hidden_states"][self.layer_index]
        return None

    def get_attr_hidden_state_old(self, did, attr):
        res_data = self.res_data_dict[str(did)]
        table_name = res_data["res"]
        if table_name not in self.table2attr2tokens or attr not in self.table2attr2tokens[table_name]:
            return None
        if attr not in res_data["data"]:
            return None
        res_value = res_data["data"][attr].strip().strip(string.punctuation)
        res_value = re.sub(r"^€", "", res_value)
        res_value_tokens = self.trainer.tokenizer.encode(res_value, add_special_tokens=False)
        state_file = os.path.join(self.states_dir, f"doc-{did}-attr-{attr}.pt")
        if not os.path.exists(state_file):
            logging.warning(
                "[LazyDataset:get_attr_hidden_state] State file %s does not exist (did: %s, attr: %s).",
                state_file,
                did,
                attr,
            )
            return None
        state_dict_attr = torch.load(state_file, weights_only=False)
        for i in range(len(state_dict_attr) - len(res_value_tokens) + 1):
            i_tokens = [item["token_id"] for item in state_dict_attr[i : i + len(res_value_tokens)]]
            if len(res_value_tokens) > 0 and i_tokens[0] == res_value_tokens[0]:
                if i_tokens == res_value_tokens:
                    return state_dict_attr[i]["hidden_states"][self.layer_index]
                i_text = "".join(
                    [self.trainer.decode_str(item["token_text"]) for item in state_dict_attr[i : i + len(res_value_tokens)]]
                )
                i_text = i_text.strip().strip(string.punctuation)
                if i_text == res_value:
                    return state_dict_attr[i]["hidden_states"][self.layer_index]
        return None


class SequentialOversampler(Sampler):
    """
    Custom sampler: samples each class in original order, and oversamples classes with fewer samples.
    """

    def __init__(self, indices, labels):
        self.indices = indices
        self.labels = labels
        self.class_to_indices = {}
        for idx, label in zip(indices, labels):
            self.class_to_indices.setdefault(label, []).append(idx)
        self.max_count = max(len(lst) for lst in self.class_to_indices.values())

        oversampled = []
        for _, class_indices in sorted(self.class_to_indices.items()):
            repeated = []
            while len(repeated) < self.max_count:
                needed = self.max_count - len(repeated)
                repeated.extend(class_indices[:needed] if needed < len(class_indices) else class_indices)
            oversampled.extend(repeated[: self.max_count])

        self.oversampled_indices = sorted(oversampled)

    def __iter__(self):
        return iter(self.oversampled_indices)

    def __len__(self):
        return len(self.oversampled_indices)


__all__ = [
    "LazyHiddenStatesDataset",
    "SequentialOversampler",
]

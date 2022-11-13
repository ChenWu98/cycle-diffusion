import math
from typing import Dict
from copy import deepcopy
import numpy as np
from pprint import pprint
from random import shuffle
from torch.utils.data import Dataset


def upsample(data, weight):
    n_data = len(data)
    assert weight >= 1

    integral = list(range(n_data)) * int(math.floor(weight))
    residual = list(range(n_data))
    shuffle(residual)
    residual = residual[:int(n_data * (weight - int(math.floor(weight))))]
    return [deepcopy(data[idx]) for idx in integral + residual]


class MultiTaskWrapper(Dataset):
    def __init__(self, name2dataset, meta_args, split):

        # Raw data and size.
        name2data = dict()
        for name, dataset in name2dataset.items():
            name2data[name] = [dataset[idx] for idx in range(len(dataset))]

        # Up-weight.
        temp = meta_args.raw_data.upsample_temp
        if temp and temp != 1 and split == 'train':
            # Dataset statistics.
            name2size = dict()
            for name, data in name2data.items():
                name2size[name] = len(data)

            # Compute resampling weights.
            name2upsample = dict()
            sum_tau_size = sum([np.exp(np.log(size) / temp) for size in name2size.values()])
            sum_size = sum(name2size.values())
            for name, size in name2size.items():
                tau_size = np.exp(np.log(size) / temp)
                name2upsample[name] = tau_size / sum_tau_size * sum_size / size

            # Compute upsampling weights.
            largest_arg_path, _ = max(name2size.items(), key=lambda x: x[1])
            norm_coef = name2upsample[largest_arg_path]
            for name in name2upsample.keys():
                name2upsample[name] = name2upsample[name] / norm_coef

            # Upsample.
            for name in sorted(name2data.keys()):
                name2data[name] = upsample(name2data[name], name2upsample[name])

            print('--- Before upsampling')
            pprint(name2size)
            print('--- Upsampling weights')
            pprint(name2upsample)
            print('--- After upsampling')
            pprint({name: len(data) for name, data in name2data.items()})

        # Add split and name.
        for name, data in name2data.items():
            for item in data:
                item['split'] = split
                item['name'] = name

        # Subset for dev.
        if split == 'dev' and meta_args.raw_data.eval_num:
            for name in name2data.keys():
                full_data = name2data[name]
                eval_num = meta_args.raw_data.eval_num
                if eval_num < len(full_data):
                    stride = 1.0 * len(full_data) / eval_num
                    name2data[name] = [full_data[int(idx * stride)] for idx in range(eval_num)]

        # Concatenate.
        self.dataset = []
        for name in sorted(name2data.keys()):
            self.dataset.extend(name2data[name])

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class StrideWrapper(Dataset):
    def __init__(self, dataset, stride):
        self.dataset = dataset
        self.index2old_index = [idx * stride for idx in range(len(self.dataset) // stride)]

    def __getitem__(self, index):
        old_index = self.index2old_index[index]
        return self.dataset[old_index]

    def __len__(self):
        return len(self.index2old_index)


class SplitArgpathWrapper(Dataset):
    def __init__(self, dataset, split, name):
        self.dataset = dataset
        self.split = split
        self.name = name

    def __getitem__(self, index):
        item = self.dataset[index]
        item['split'] = self.split
        item['name'] = self.name
        return item

    def __len__(self):
        return len(self.dataset)


def get_multi_task_dataset_splits(meta_args, name2dataset_splits):

    name2train_dataset, name2dev_dataset, name2test_dataset = dict(), dict(), dict()
    for name, dataset_splits in name2dataset_splits.items():
        name2train_dataset[name] = dataset_splits['train']
        name2dev_dataset[name] = dataset_splits['dev']
        name2test_dataset[name] = dataset_splits.get('test', dataset_splits['dev'])

    return {
        'train': MultiTaskDataset(meta_args, name2train_dataset, split='train'),
        'dev': MultiTaskDataset(meta_args, name2dev_dataset, split='dev'),
        'test': MultiTaskDataset(meta_args, name2test_dataset, split='test'),
    }


class MultiTaskDataset(Dataset):

    def __init__(self, meta_args, name2dataset: Dict[str, Dataset], split: str):
        self.meta_args = meta_args

        self.data = MultiTaskWrapper(name2dataset=name2dataset, meta_args=meta_args, split=split)

    def __getitem__(self, index):
        data = self.data[index]
        model_inputs = {
            k: data[k] for k in data['model_kwargs']
        }
        return model_inputs

    def __len__(self):
        return len(self.data)

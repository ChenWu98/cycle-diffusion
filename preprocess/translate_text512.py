import os
import torch
import json
from datasets import DatasetDict
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.file_utils import pil_loader
from utils.transform_utils import CenterCropLongEdge


class Preprocessor(object):

    def __init__(self, args, meta_args):
        self.args = args
        self.meta_args = meta_args

    def preprocess(self, raw_datasets: DatasetDict, cache_root: str):
        assert len(raw_datasets) == 3  # Not always.
        train_dataset = TrainDataset(self.args, self.meta_args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, self.meta_args, raw_datasets['validation'], cache_root)

        return {
            'train': train_dataset,
            'dev': dev_dataset,
        }


class TrainDataset(Dataset):

    def __init__(self, args, meta_args, raw_datasets, cache_root):
        self.data = []

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)


class DevDataset(Dataset):

    def __init__(self, args, meta_args, raw_datasets, cache_root):

        raw_data = json.load(open('./data/translate-text.json'))
        self.transform = transforms.Compose([
            CenterCropLongEdge(),
            transforms.Resize(512),
            transforms.ToTensor()
        ])

        start, end = meta_args.raw_data.range

        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "meta": meta,
                "model_kwargs": ["sample_id", ]
            }
            for idx, meta in enumerate(raw_data[start: end])
        ]

    def __getitem__(self, index):
        data = {k: v for k, v in self.data[index].items()}

        # Add image.
        filename = data['meta']['img_path']
        img = pil_loader(filename)
        img = self.transform(img)

        # Add encode text.
        data["encode_text"] = data['meta']['encode_text']
        data["model_kwargs"] = data["model_kwargs"] + ["encode_text", ]

        # Add decode text.
        data["decode_text"] = data['meta']['decode_text']
        data["model_kwargs"] = data["model_kwargs"] + ["decode_text", ]

        # Add original image.
        data['original_image'] = img
        data["model_kwargs"] = data["model_kwargs"] + ["original_image", ]

        return data

    def __len__(self):
        return len(self.data)

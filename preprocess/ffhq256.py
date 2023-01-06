# Created by Chen Henry Wu
import os
import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


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

        self.root_dir = './data/images1024x1024'  # TODO
        self.transform = transforms.Compose([
            transforms.Resize(256),  # TODO: use the 256x256 version or follow the original processing.
            transforms.ToTensor()
        ])

        self.metas = []
        # for i in range(1, 21):
        for i in [1, 11, 15]:  # TODO
        # for i in [15]:
            self.metas.append(str(i).zfill(5) + '.png')

        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "meta": meta,
                "model_kwargs": ["sample_id", ]
            }
            for idx, meta in enumerate(self.metas)
        ]

    def __getitem__(self, index):
        data = {k: v for k, v in self.data[index].items()}

        # Add image.
        filename = self.root_dir + '/' + data['meta']
        img = Image.open(filename)
        img = self.transform(img)

        # Add image.
        data['original_image'] = img
        data["model_kwargs"] = data["model_kwargs"] + ["original_image", ]

        return data

    def __len__(self):
        return len(self.data)

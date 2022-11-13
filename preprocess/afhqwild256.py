import os
import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

from utils.file_utils import list_image_files_recursively


INTERPOLATION = TF.InterpolationMode.BILINEAR


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

        self.root_dir = './stargan-v2/data/test/wild'
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=INTERPOLATION),  # 512 -> 256
            transforms.ToTensor()
        ])

        self.file_names = list_image_files_recursively(self.root_dir)

        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "file_name": file_name,
                "model_kwargs": ["sample_id", ]
            }
            for idx, file_name in enumerate(self.file_names)
        ]

    def __getitem__(self, index):
        data = {k: v for k, v in self.data[index].items()}

        # Add image.
        img = Image.open(data['file_name'])
        assert img.size == (512, 512)
        img = self.transform(img)

        # Add image.
        data['original_image'] = img
        data["model_kwargs"] = data["model_kwargs"] + ["original_image", ]

        return data

    def __len__(self):
        return len(self.data)
        # return 32

import os
import blobfile as bf

import torch
from torchvision import utils
from PIL import Image


def save_images(images: torch.Tensor, output_dir: str, file_prefix: str, nrows: int, iteration: int) -> None:
    utils.save_image(
        images,
        os.path.join(output_dir, f"{file_prefix}_{str(iteration).zfill(6)}.png"),
        nrow=nrows,
    )


def list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(list_image_files_recursively(full_path))
    return results


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

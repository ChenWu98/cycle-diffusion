import os
import torch
from cleanfid import fid
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image

from .utils import save_image, calculate_ssim, calculate_psnr
from utils.file_utils import list_image_files_recursively
from preprocess.afhqwild256 import INTERPOLATION


class Evaluator(object):

    def __init__(self, args, meta_args):
        self.args = args
        self.meta_args = meta_args

        self.ref_transform = transforms.Compose([
            transforms.Resize(256, interpolation=INTERPOLATION),  # 512 -> 256
            transforms.ToTensor()
        ])

    def evaluate(self, images, model, weighted_loss, losses, data, split):
        """

        Args:
            images: list of images, or list of tuples of images
            model: model to evaluate
            weighted_loss: list of scalar tensors
            losses: dictionary of lists of scalar tensors
            data: list of dictionary
            split: str

        Returns:

        """
        assert split in ['eval', 'test']

        # Add metrics here.
        f_gen = os.path.join(self.meta_args.output_dir, 'temp_gen')
        f_ref = os.path.join(self.meta_args.output_dir, 'temp_ref')
        if os.path.exists(f_gen):
            os.remove(f_gen)
        os.mkdir(f_gen)
        if os.path.exists(f_ref):
            os.remove(f_ref)
        os.mkdir(f_ref)

        # Resize reference images.
        root_dir = './stargan-v2/data/test/dog'
        for idx, file_name in tqdm(enumerate(list_image_files_recursively(root_dir))):
            ref_img = Image.open(file_name)
            assert ref_img.size == (512, 512)
            ref_img = self.ref_transform(ref_img)
            save_image(os.path.join(f_ref, '{}.png'.format(idx)), ref_img)

        n = len(images)
        all_psnr, all_ssim, l2 = 0, 0, 0
        idx = 0
        for original_img, img in tqdm(images):
            assert img.dim() == original_img.dim() == 3

            img = img.clamp(0, 1)
            original_img = original_img.clamp(0, 1)

            all_psnr += calculate_psnr(img, original_img)
            all_ssim += calculate_ssim(
                (img.numpy() * 255).transpose((1, 2, 0)),
                (original_img.numpy() * 255).transpose((1, 2, 0)),
            )
            l2 += torch.sqrt(
                ((img - original_img) ** 2).sum(2).sum(1).sum(0)
            ).item()

            assert img.shape == original_img.shape
            save_image(os.path.join(f_gen, '{}.png'.format(idx)), img)
            idx += 1

        kid_score = fid.compute_kid(
            fdir1=f_gen, fdir2=f_ref,
            batch_size=32,
        )
        fid_score = fid.compute_fid(
            fdir1=f_gen, fdir2=f_ref,
            batch_size=32,
        )

        summary = {
            "psnr": all_psnr / n,
            "ssim": all_ssim / n,
            "l2": l2 / n,
            "kid": kid_score,
            "fid": fid_score,
        }

        return summary

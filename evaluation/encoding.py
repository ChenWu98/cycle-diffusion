# Created by Chen Henry Wu
import os
import numpy as np
import torch
import cv2
from cleanfid import fid
from tqdm import tqdm
from torchvision import utils


def save_image(image_path, image):
    assert image.dim() == 3 and image.shape[0] == 3

    utils.save_image(image, image_path)


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class Evaluator(object):

    def __init__(self, args, meta_args):
        self.args = args
        self.meta_args = meta_args

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
        n = len(images)
        all_psnr = 0
        all_diff0, all_diff1, all_diff2 = 0, 0, 0  # TODO: remove this.
        for img, original_img in tqdm(images):
            assert img.dim() == original_img.dim() == 3
            mse = ((img - original_img) ** 2).mean(2).mean(1).mean(0)
            all_diff0 += (img - original_img)[0, :, :].mean(1).mean(0)  # TODO: remove this.
            all_diff1 += (img - original_img)[1, :, :].mean(1).mean(0)  # TODO: remove this.
            all_diff2 += (img - original_img)[2, :, :].mean(1).mean(0)  # TODO: remove this.
            psnr = 10 * torch.log10(1 / mse)

            all_psnr += psnr

            assert img.shape == original_img.shape

        summary = {
            "psnr": all_psnr / n,
            "diff0": all_diff0 / n,  # TODO: remove this.
            "diff1": all_diff1 / n,  # TODO: remove this.
            "diff2": all_diff2 / n,  # TODO: remove this.
        }

        return summary

import os
import math
import torch
import torch.nn.functional as F

from utils.file_utils import save_images


class Visualizer(object):

    def __init__(self, args):
        self.args = args

    def visualize(self,
                  images,
                  model,
                  description: str,
                  save_dir: str,
                  step: int,
                  ):
        # Merge.
        k = len(images)
        assert k >= 2
        bsz, c, h, w = images[0].shape
        if k == 3:
            bsz2, c2, h2, w2 = images[2].shape
            assert bsz == bsz2 and c == c2
            assert h2 == w2
            assert h == w
            if h == h2:
                pass
            else:
                assert h2 < h
                images = (
                    images[0],
                    images[1],
                    F.interpolate(images[2], size=(h, w), mode='nearest'),
                )
        images = torch.stack(images, dim=1).view(bsz * k, c, h, w)

        # Just visualize the first 64 images.
        images = images[:100 * k, :, :, :]

        save_images(
            images,
            output_dir=save_dir,
            file_prefix=description,
            nrows=8,
            iteration=step,
        )

        # Lower resolution
        images_256 = F.interpolate(
            images,
            (256, 256),
            mode='bicubic',
        )
        save_images(
            images_256,
            output_dir=save_dir,
            file_prefix=f'{description}_256',
            nrows=8,
            iteration=step,
        )



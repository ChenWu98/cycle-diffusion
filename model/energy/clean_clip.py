import torch
import torchvision.transforms as transforms

import clip


class DirectionalCLIP(object):
    def __init__(self):

        self.model, clip_preprocess = clip.load("ViT-B/32", device="cpu")  # cpu allows for fp32 loading.
        self.model = self.model.to('cuda')
        self.model.eval()

        self.clip_preprocess = transforms.Compose(  # Already un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            clip_preprocess.transforms[:2] +  # Skip ToRGB and ToTensor
            clip_preprocess.transforms[4:]
        )

    def __call__(self, img, original_img, encode_text, decode_text):
        assert len(decode_text) == img.shape[0]
        assert len(encode_text) == original_img.shape[0]

        with torch.no_grad():
            encode_text_feature = self.model.encode_text(clip.tokenize(encode_text).to('cuda'))
            encode_text_feature /= encode_text_feature.norm(dim=-1, keepdim=True)
            decode_text_feature = self.model.encode_text(clip.tokenize(decode_text).to('cuda'))
            decode_text_feature /= decode_text_feature.norm(dim=-1, keepdim=True)
            img_feature = self.model.encode_image(self.clip_preprocess(img).to('cuda'))
            img_feature /= img_feature.norm(dim=-1, keepdim=True)
            original_img_feature = self.model.encode_image(self.clip_preprocess(original_img).to('cuda'))
            original_img_feature /= original_img_feature.norm(dim=-1, keepdim=True)

            img_direction = img_feature - original_img_feature
            img_direction /= img_direction.norm(dim=-1, keepdim=True)
            text_direction = decode_text_feature - encode_text_feature
            text_direction /= text_direction.norm(dim=-1, keepdim=True)

            clip_score = torch.einsum('bz,bz->b', img_feature, decode_text_feature)
            dclip_score = torch.einsum('bz,bz->b', img_direction, text_direction)

        return clip_score, dclip_score


class CLIP(object):
    def __init__(self):

        self.model, clip_preprocess = clip.load("ViT-B/32", device="cpu")  # cpu allows for fp32 loading.
        self.model = self.model.to('cuda')
        self.model.eval()

        self.clip_preprocess = transforms.Compose(  # Already un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            clip_preprocess.transforms[:2] +  # Skip ToRGB and ToTensor
            clip_preprocess.transforms[4:]
        )

    def __call__(self, img, text):
        assert len(text) == img.shape[0]

        with torch.no_grad():
            text_feature = self.model.encode_text(clip.tokenize(text).to('cuda'))
            text_feature /= text_feature.norm(dim=-1, keepdim=True)

            img_feature = self.model.encode_image(self.clip_preprocess(img).to('cuda'))
            img_feature /= img_feature.norm(dim=-1, keepdim=True)

            clip_score = torch.einsum('bz,bz->b', img_feature, text_feature)

        return clip_score

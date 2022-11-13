import os
import argparse
import sys
sys.path.append(os.path.abspath('model/lib/latentdiff'))
import glob
from omegaconf import OmegaConf
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from model.energy.clean_clip import DirectionalCLIP
from txt2img import load_model_from_config, DDIMSampler
from ..model_utils import requires_grad


def prepare_latentdiff_text(source_model_type):
    print('First of all, when the code changes, make sure that no part in the model is under no_grad!')

    if source_model_type == "text2img-large":
        config = OmegaConf.load(os.path.join('ckpts', 'ldm_models', source_model_type, 'txt2img-1p4B-eval.yaml'))

        latentdiff_ckpt = os.path.join('ckpts', 'ldm_models', source_model_type, 'model.ckpt')
    else:
        raise ValueError()

    return config, latentdiff_ckpt


def get_condition(model, text, bs):
    assert isinstance(text, list)
    assert isinstance(text[0], str)
    uc = model.get_learned_conditioning(bs * [""])
    print("model.cond_stage_key: ", model.cond_stage_key)
    c = model.get_learned_conditioning(text)
    print("c.shape: ", c.shape)
    print('-' * 50)
    return c, uc


def convsample_ddim_conditional(model, steps, shape, x_T, skip_steps, eta, eps_list, scale, text):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    c, uc = get_condition(model, text, bs)
    samples, intermediates = ddim.sample_with_eps(steps,
                                                  eps_list,
                                                  conditioning=c,
                                                  batch_size=bs,
                                                  shape=shape,
                                                  eta=eta,
                                                  verbose=False,
                                                  x_T=x_T,
                                                  skip_steps=skip_steps,
                                                  unconditional_guidance_scale=scale,
                                                  unconditional_conditioning=uc
                                                  )
    return samples, intermediates


def make_convolutional_sample_with_eps_conditional(model, custom_steps, eta, x_T, skip_steps, eps_list,
                                                   scale, text):
    with model.ema_scope("Plotting"):
        sample, intermediates = convsample_ddim_conditional(model,
                                                            steps=custom_steps,
                                                            shape=x_T.shape,
                                                            x_T=x_T,
                                                            skip_steps=skip_steps,
                                                            eta=eta,
                                                            eps_list=eps_list,
                                                            scale=scale,
                                                            text=text)

    x_sample = model.decode_first_stage(sample)

    return x_sample


def ddpm_ddim_encoding_conditional(model, steps, shape, eta, white_box_steps, skip_steps, x0, scale, text):
    with model.ema_scope("Plotting"):
        ddim = DDIMSampler(model)
        bs = shape[0]
        shape = shape[1:]
        c, uc = get_condition(model, text, bs)

        z_list = ddim.ddpm_ddim_encoding(steps,
                                         conditioning=c,
                                         batch_size=bs,
                                         shape=shape,
                                         eta=eta,
                                         white_box_steps=white_box_steps,
                                         skip_steps=skip_steps,
                                         verbose=True,
                                         x0=x0,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         )

    return z_list


class LatentDiffStochasticTextWrapper(torch.nn.Module):

    def __init__(self, source_model_type, custom_steps, eta, white_box_steps, skip_steps,
                 encoder_unconditional_guidance_scales=None, decoder_unconditional_guidance_scales=None,
                 n_trials=None):
        super(LatentDiffStochasticTextWrapper, self).__init__()

        self.encoder_unconditional_guidance_scales = encoder_unconditional_guidance_scales
        self.decoder_unconditional_guidance_scales = decoder_unconditional_guidance_scales
        self.n_trials = n_trials

        # Set up generator
        self.config, self.ckpt = prepare_latentdiff_text(source_model_type)

        print(self.config)

        self.generator = load_model_from_config(self.config, self.ckpt, verbose=True)

        print(75 * "-")

        self.eta = eta
        self.custom_steps = custom_steps
        self.white_box_steps = white_box_steps
        self.skip_steps = skip_steps

        self.resolution = self.generator.first_stage_model.encoder.resolution
        print(f"resolution: {self.resolution}")

        print(f'Using DDIM sampling with {self.custom_steps} sampling steps and eta={self.eta}')

        # Freeze.
        # requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

        # Directional CLIP score.
        self.directional_clip = DirectionalCLIP()

    def generate(self, z_ensemble, decode_text):
        img_ensemble = []
        for i, z in enumerate(z_ensemble):
            skip_steps = self.skip_steps[i % len(self.skip_steps)]
            bsz = z.shape[0]
            if self.white_box_steps != -1:
                eps_list = z.view(bsz, (self.white_box_steps - skip_steps), self.generator.channels, self.generator.image_size, self.generator.image_size)
            else:
                eps_list = z.view(bsz, 1, self.generator.channels, self.generator.image_size, self.generator.image_size)
            x_T = eps_list[:, 0]
            eps_list = eps_list[:, 1:]

            for decoder_unconditional_guidance_scale in self.decoder_unconditional_guidance_scales:
                img = make_convolutional_sample_with_eps_conditional(self.generator,
                                                                     custom_steps=self.custom_steps,
                                                                     eta=self.eta,
                                                                     x_T=x_T,
                                                                     skip_steps=skip_steps,
                                                                     eps_list=eps_list,
                                                                     scale=decoder_unconditional_guidance_scale,
                                                                     text=decode_text)
                img_ensemble.append(img)

        return img_ensemble

    def encode(self, image, encode_text):
        # Eval mode for the generator.
        self.generator.eval()

        # Normalize.
        image = (image - 0.5) * 2.0
        # Resize.
        assert image.shape[2] == image.shape[3] == self.resolution
        with torch.no_grad():
            # Encode.
            encoder_posterior = self.generator.encode_first_stage(image)
            z = self.generator.get_first_stage_encoding(encoder_posterior)
            x0 = z

        bsz = image.shape[0]
        z_ensemble = []
        for trial in range(self.n_trials):
            for encoder_unconditional_guidance_scale in self.encoder_unconditional_guidance_scales:
                for skip_steps in self.skip_steps:
                    with torch.no_grad():
                        # DDIM forward.
                        z_list = ddpm_ddim_encoding_conditional(self.generator,
                                                                steps=self.custom_steps,
                                                                shape=x0.shape,
                                                                eta=self.eta,
                                                                white_box_steps=self.white_box_steps,
                                                                skip_steps=skip_steps,
                                                                x0=x0,
                                                                scale=encoder_unconditional_guidance_scale,
                                                                text=encode_text)
                        z = torch.stack(z_list, dim=1).view(bsz, -1)
                        z_ensemble.append(z)

        return z_ensemble

    def forward(self, z_ensemble, original_img, encode_text, decode_text):
        # Eval mode for the generator.
        self.generator.eval()

        img_ensemble = self.generate(z_ensemble, decode_text)
        assert len(img_ensemble) == len(self.decoder_unconditional_guidance_scales) * len(self.encoder_unconditional_guidance_scales) * len(self.skip_steps) * self.n_trials

        # Post process.
        img_ensemble = [self.post_process(img) for img in img_ensemble]

        # Rank with directional CLIP score.
        score_ensemble = []
        for img in img_ensemble:
            _, dclip_score = self.directional_clip(img, original_img, encode_text, decode_text)
            assert dclip_score.shape == (img.shape[0],)

            score_ensemble.append(dclip_score)
        score_ensemble = torch.stack(score_ensemble, dim=1)  # (bsz, n_trials)
        assert score_ensemble.shape == (img_ensemble[0].shape[0], len(img_ensemble))

        best_idx = torch.argmax(score_ensemble, dim=1)  # (bsz,)
        bsz = score_ensemble.shape[0]
        img = torch.stack(
            [
                img_ensemble[best_idx[b].item()][b] for b in range(bsz)
            ],
            dim=0,
        )
        print('best scales:')
        best_idx = best_idx % (len(self.decoder_unconditional_guidance_scales) * len(self.encoder_unconditional_guidance_scales) * len(self.skip_steps))
        print(
            [
                (
                    self.encoder_unconditional_guidance_scales[_best_idx // (len(self.decoder_unconditional_guidance_scales) * len(self.skip_steps))],
                    self.decoder_unconditional_guidance_scales[_best_idx % (len(self.decoder_unconditional_guidance_scales) * len(self.skip_steps)) // len(self.skip_steps)],
                    self.skip_steps[_best_idx % len(self.skip_steps)],
                )
                for _best_idx in best_idx
            ]
        )

        return img

    @property
    def device(self):
        return next(self.parameters()).device





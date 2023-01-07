# Created by Chen Henry Wu
import os
import sys
sys.path.append(os.path.abspath('model/lib/latentdiff'))
import glob
from omegaconf import OmegaConf
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from sample_diffusion import get_parser, load_model, DDIMSampler
from ..model_utils import requires_grad


def prepare_latentdiff(source_model_type, custom_steps):
    print('First of all, when the code changes, make sure that no part in the model is under no_grad!')
    latentdiff_ckpt = os.path.join('ckpts', 'ldm_models', 'ldm', source_model_type, 'model.ckpt')

    parser = get_parser()
    opt, unknown = parser.parse_known_args(
        [
            '-r', latentdiff_ckpt,
            '-c', str(custom_steps),
            '-e', str(0),
        ]
    )
    print('unknown args: ', unknown)

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    return config, opt, ckpt


def get_condition(model, class_label, bs):
    uc = model.get_learned_conditioning(
        {model.cond_stage_key: torch.tensor(bs * [1000]).to(model.device)}
    )
    print("model.cond_stage_key: ", model.cond_stage_key)
    c = model.get_learned_conditioning({model.cond_stage_key: class_label})
    print("c.shape: ", c.shape)
    print('-' * 50)
    return c, uc


def convsample_ddim_conditional(model, steps, shape, x_T, eta, scale, class_label):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    c, uc = get_condition(model, class_label, bs)
    samples, intermediates = ddim.sample(steps,
                                         conditioning=c,
                                         batch_size=bs,
                                         shape=shape,
                                         eta=eta,
                                         verbose=True,  # TODO: verbose.
                                         x_T=x_T,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         )
    return samples, intermediates


def convsample_ddim(model, steps, shape, x_T, eta):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=True, x_T=x_T)  # TODO: verbose.
    return samples, intermediates


def convsample_ddim_forward(model, steps, shape, x0):
    with model.ema_scope("Plotting"):   # TODO: important
        ddim = DDIMSampler(model)
        bs = shape[0]
        shape = shape[1:]
        samples, intermediates = ddim.sample_forward(steps, batch_size=bs, shape=shape, eta=0, verbose=True, x0=x0)  # TODO: verbose.
        return samples, intermediates


def convsample_ddim_forward_conditional(model, steps, shape, x0, scale, class_label):
    with model.ema_scope("Plotting"):   # TODO: important
        ddim = DDIMSampler(model)
        bs = shape[0]
        shape = shape[1:]
        c, uc = get_condition(model, class_label, bs)

        samples, intermediates = ddim.sample_forward(steps,
                                                     conditioning=c,
                                                     batch_size=bs,
                                                     shape=shape,
                                                     eta=0,
                                                     verbose=True,  # TODO: verbose.
                                                     x0=x0,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,
                                                     )

        return samples, intermediates


def make_convolutional_sample(model, vanilla, custom_steps, eta, x_T):
    with model.ema_scope("Plotting"):
        if vanilla:
            raise NotImplementedError()
        else:
            sample, intermediates = convsample_ddim(model,
                                                    steps=custom_steps,
                                                    shape=x_T.shape,
                                                    x_T=x_T,
                                                    eta=eta)

    x_sample = model.decode_first_stage(sample)

    return x_sample


def make_convolutional_sample_conditional(model, vanilla, custom_steps, eta, x_T, scale, class_label):
    assert scale is not None
    with model.ema_scope("Plotting"):
        if vanilla:
            raise NotImplementedError()
        else:
            sample, intermediates = convsample_ddim_conditional(model,
                                                                steps=custom_steps,
                                                                shape=x_T.shape,
                                                                x_T=x_T,
                                                                eta=eta,
                                                                scale=scale,
                                                                class_label=class_label)

    x_sample = model.decode_first_stage(sample)

    return x_sample


class LatentDiffWrapper(torch.nn.Module):

    def __init__(self, source_model_type, custom_steps, custom_steps_train=None,
                 enforce_class_input=None, unconditional_guidance_scale=None):
        super(LatentDiffWrapper, self).__init__()

        self.enforce_class_input = enforce_class_input
        self.custom_steps_train = custom_steps_train
        self.unconditional_guidance_scale = unconditional_guidance_scale

        # Set up generator
        self.config, self.opt, self.ckpt = prepare_latentdiff(source_model_type, custom_steps)

        gpu = True
        eval_mode = True

        print(self.config)
        print(vars(self.opt))

        self.generator, global_step = load_model(self.config, self.ckpt, gpu, eval_mode)

        print(f"global step: {global_step}")
        print(75 * "-")

        self.eta = self.opt.eta
        self.vanilla = self.opt.vanilla_sample
        self.custom_steps = self.opt.custom_steps

        self.resolution = self.generator.first_stage_model.encoder.resolution
        print(f"resolution: {self.resolution}")

        if self.vanilla:
            print(f'Using Vanilla DDPM sampling with {self.generator.num_timesteps} sampling steps.')
        else:
            print(f'Using DDIM sampling with {self.custom_steps} sampling steps and eta={self.eta}')

        self.latent_dim = self.generator.image_size ** 2 * self.generator.channels
        # Freeze.
        # requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    def generate(self, z, class_label):
        if not self.training:
            custom_steps = self.custom_steps
        else:
            assert self.custom_steps_train is not None
            custom_steps = self.custom_steps_train

        bsz = z.shape[0]
        x_T = z.view(bsz, self.generator.channels, self.generator.image_size, self.generator.image_size)
        if self.enforce_class_input:
            assert class_label is not None
            img = make_convolutional_sample_conditional(self.generator,
                                                        vanilla=self.vanilla,
                                                        custom_steps=custom_steps,
                                                        eta=self.eta,
                                                        x_T=x_T,
                                                        scale=self.unconditional_guidance_scale,
                                                        class_label=class_label)
        else:
            img = make_convolutional_sample(self.generator,
                                            vanilla=self.vanilla,
                                            custom_steps=custom_steps,
                                            eta=self.eta,
                                            x_T=x_T)

        return img

    def encode(self, image, class_label=None):
        # Eval mode for the generator.
        self.generator.eval()

        with torch.no_grad():
            # Normalize.
            image = (image - 0.5) * 2.0

            # Resize.
            assert image.shape[2] == image.shape[3] == self.resolution

            # Encode.
            encoder_posterior = self.generator.encode_first_stage(image)
            x0 = self.generator.get_first_stage_encoding(encoder_posterior)

            # DDIM forward.
            if self.enforce_class_input:
                assert class_label is not None
                samples, intermediates = convsample_ddim_forward_conditional(self.generator,
                                                                             steps=self.custom_steps,
                                                                             shape=x0.shape,
                                                                             x0=x0,
                                                                             scale=self.unconditional_guidance_scale,
                                                                             class_label=class_label)
            else:
                samples, intermediates = convsample_ddim_forward(self.generator,
                                                                 steps=self.custom_steps,
                                                                 shape=x0.shape,
                                                                 x0=x0)
            z = samples.view(samples.shape[0], -1)
            assert z.shape[1] == self.latent_dim

            return z

    def forward(self, z, class_label=None):
        # Eval mode for the generator.
        self.generator.eval()

        img = self.generate(z, class_label)

        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device





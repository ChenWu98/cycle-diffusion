import os
import sys
sys.path.append(os.path.abspath('model/lib/latentdiff'))
import glob
from omegaconf import OmegaConf
import torch
import torchvision.transforms as transforms

from ..lib.latentdiff.sample_diffusion import get_parser, load_model, DDIMSampler
from .latentdiff_wrapper import get_condition
from ..model_utils import requires_grad


def prepare_latentdiff(source_model_type, custom_steps, eta):
    print('First of all, when the code changes, make sure that no part in the model is under no_grad!')
    latentdiff_ckpt = os.path.join('ckpts', 'ldm_models', 'ldm', source_model_type, 'model.ckpt')

    parser = get_parser()
    opt, unknown = parser.parse_known_args(
        [
            '-r', latentdiff_ckpt,
            '-c', str(custom_steps),
            '-e', str(eta),
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


def convsample_ddim(model, steps, shape, x_T, eta, eps_list, refine_steps):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample_with_eps(steps,
                                                  eps_list,
                                                  batch_size=bs,
                                                  shape=shape,
                                                  eta=eta,
                                                  verbose=False,
                                                  x_T=x_T,
                                                  )

    refine_eta = 1
    if refine_steps > 0:
        samples, intermediates = ddim.refine(steps, refine_steps=refine_steps,
                                             batch_size=bs,
                                             shape=shape,
                                             eta=refine_eta,
                                             verbose=False,
                                             x0=samples,
                                             )

    return samples, intermediates


def convsample_ddim_conditional(model, steps, shape, x_T, eta, eps_list, scale, class_label, refine_steps):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    c, uc = get_condition(model, class_label, bs)
    samples, intermediates = ddim.sample_with_eps(steps,
                                                  eps_list,
                                                  conditioning=c,
                                                  batch_size=bs,
                                                  shape=shape,
                                                  eta=eta,
                                                  verbose=False,
                                                  x_T=x_T,
                                                  unconditional_guidance_scale=scale,
                                                  unconditional_conditioning=uc
                                                  )

    refine_eta = 1
    if refine_steps > 0:
        samples, intermediates = ddim.refine(steps, refine_steps=refine_steps,
                                             conditioning=c,
                                             batch_size=bs,
                                             shape=shape,
                                             eta=refine_eta,
                                             verbose=False,
                                             x0=samples,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc
                                             )

    return samples, intermediates


def make_convolutional_sample_with_eps(model, vanilla, custom_steps, eta, x_T, eps_list, refine_steps):
    with model.ema_scope("Plotting"):
        if vanilla:
            raise NotImplementedError()
        else:
            sample, intermediates = convsample_ddim(model,
                                                    steps=custom_steps,
                                                    shape=x_T.shape,
                                                    x_T=x_T,
                                                    eta=eta,
                                                    eps_list=eps_list,
                                                    refine_steps=refine_steps)

    x_sample = model.decode_first_stage(sample)

    return x_sample


def make_convolutional_sample_with_eps_conditional(model, vanilla, custom_steps, eta, x_T, eps_list,
                                                   scale, class_label, refine_steps):
    with model.ema_scope("Plotting"):
        if vanilla:
            raise NotImplementedError()
        else:
            sample, intermediates = convsample_ddim_conditional(model,
                                                                steps=custom_steps,
                                                                shape=x_T.shape,
                                                                x_T=x_T,
                                                                eta=eta,
                                                                eps_list=eps_list,
                                                                scale=scale,
                                                                class_label=class_label,
                                                                refine_steps=refine_steps)

    x_sample = model.decode_first_stage(sample)

    return x_sample


def ddpm_ddim_encoding(model, steps, shape, eta, white_box_steps, x0):
    with model.ema_scope("Plotting"):
        ddim = DDIMSampler(model)
        bs = shape[0]
        shape = shape[1:]
        z_list = ddim.ddpm_ddim_encoding(steps, batch_size=bs, shape=shape, eta=eta, white_box_steps=white_box_steps, verbose=False, x0=x0)
        return z_list


def ddpm_ddim_encoding_conditional(model, steps, shape, eta, white_box_steps, x0, scale, class_label):
    with model.ema_scope("Plotting"):
        ddim = DDIMSampler(model)
        bs = shape[0]
        shape = shape[1:]
        c, uc = get_condition(model, class_label, bs)

        z_list = ddim.ddpm_ddim_encoding(steps,
                                         conditioning=c,
                                         batch_size=bs,
                                         shape=shape,
                                         eta=eta,
                                         white_box_steps=white_box_steps,
                                         verbose=True,
                                         x0=x0,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         )

    return z_list


class LatentDiffStochasticWrapper(torch.nn.Module):

    def __init__(self, source_model_type, custom_steps, eta, white_box_steps, refine_steps=0,
                 enforce_class_input=None, unconditional_guidance_scale=None):
        super(LatentDiffStochasticWrapper, self).__init__()

        self.enforce_class_input = enforce_class_input
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.refine_steps = refine_steps

        # Set up generator
        self.config, self.opt, self.ckpt = prepare_latentdiff(source_model_type, custom_steps, eta)

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
        self.white_box_steps = white_box_steps

        self.resolution = self.generator.first_stage_model.encoder.resolution
        print(f"resolution: {self.resolution}")

        if self.vanilla:
            print(f'Using Vanilla DDPM sampling with {self.generator.num_timesteps} sampling steps.')
        else:
            print(f'Using DDIM sampling with {self.custom_steps} sampling steps and eta={self.eta}')

        if self.generator.cond_stage_model is None:
            pass
        else:
            raise NotImplementedError('Currently only sampling for unconditional models supported.')

        self.latent_dim = self.generator.image_size ** 2 * self.generator.channels * self.white_box_steps
        # Freeze.
        # requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    def generate(self, z, class_label):
        bsz = z.shape[0]
        eps_list = z.view(bsz, self.white_box_steps, self.generator.channels, self.generator.image_size, self.generator.image_size)
        x_T = eps_list[:, 0]
        eps_list = eps_list[:, 1:]
        if self.enforce_class_input:
            assert class_label is not None
            img = make_convolutional_sample_with_eps_conditional(self.generator,
                                                                 vanilla=self.vanilla,
                                                                 custom_steps=self.custom_steps,
                                                                 eta=self.eta,
                                                                 x_T=x_T,
                                                                 eps_list=eps_list,
                                                                 scale=self.unconditional_guidance_scale,
                                                                 class_label=class_label,
                                                                 refine_steps=self.refine_steps)
        else:
            img = make_convolutional_sample_with_eps(self.generator,
                                                     vanilla=self.vanilla,
                                                     custom_steps=self.custom_steps,
                                                     eta=self.eta,
                                                     x_T=x_T,
                                                     eps_list=eps_list,
                                                     refine_steps=self.refine_steps)

        return img

    def encode(self, image, class_label=None):
        # Eval mode for the generator.
        self.generator.eval()

        bsz = image.shape[0]
        with torch.no_grad():
            # Normalize.
            image = (image - 0.5) * 2.0

            # Resize.
            assert image.shape[2] == image.shape[3] == self.resolution

            # Encode.
            encoder_posterior = self.generator.encode_first_stage(image)
            x0 = self.generator.get_first_stage_encoding(encoder_posterior)

            # DPM-Encoder.
            if self.enforce_class_input:
                assert class_label is not None
                z_list = ddpm_ddim_encoding_conditional(self.generator,
                                                        steps=self.custom_steps,
                                                        shape=x0.shape,
                                                        eta=self.eta,
                                                        white_box_steps=self.white_box_steps,
                                                        x0=x0,
                                                        scale=self.unconditional_guidance_scale,
                                                        class_label=class_label)
            else:
                z_list = ddpm_ddim_encoding(self.generator,
                                            steps=self.custom_steps,
                                            shape=x0.shape,
                                            eta=self.eta,
                                            white_box_steps=self.white_box_steps,
                                            x0=x0)

            z = torch.stack(z_list, dim=1).view(bsz, -1)
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





import os
import argparse
import yaml
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from ..lib.ddpm_ddim.models.ddpm.diffusion import DDPM
from ..lib.ddpm_ddim.models.improved_ddpm.script_util import i_DDPM
from ..lib.ddpm_ddim.utils.diffusion_utils import (
    get_beta_schedule, denoising_step, extract
)
from ..model_utils import requires_grad


def prepare_ddpm_ddim(source_model_type, source_model_path):
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # Default
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')

    # Train & Test
    parser.add_argument('--model_path', type=str, default=None, help='Test model path')

    if source_model_type == 'celeba256':
        assert source_model_path is None
        ddim_args = parser.parse_args(
            [
                '--config', 'celeba.yml',
                '--model_path', 'ckpts/ddpm/celeba_hq.ckpt',
            ]
        )
    elif source_model_type == 'afhqdog256':
        assert source_model_path is not None
        ddim_args = parser.parse_args(
            [
                '--config', 'afhq.yml',
                '--model_path', source_model_path,
            ]
        )
    elif source_model_type == 'afhqcat256':
        assert source_model_path is not None
        ddim_args = parser.parse_args(
            [
                '--config', 'afhq.yml',
                '--model_path', source_model_path,
            ]
        )
    elif source_model_type == 'afhqwild256':
        assert source_model_path is not None
        ddim_args = parser.parse_args(
            [
                '--config', 'afhq.yml',
                '--model_path', source_model_path,
            ]
        )
    elif source_model_type == 'ffhq256':
        assert source_model_path is None
        ddim_args = parser.parse_args(
            [
                '--config', 'ffhq.yml',
                '--model_path', 'ckpts/ddpm/ffhq_10m.pt',
            ]
        )
    elif source_model_type == 'bedroom256':
        assert source_model_path is None
        ddim_args = parser.parse_args(
            [
                '--config', 'bedroom.yml',
                '--model_path', 'ckpts/ddpm/bedroom.ckpt',
            ]
        )
    elif source_model_type == 'church_outdoor256':
        assert source_model_path is None
        ddim_args = parser.parse_args(
            [
                '--config', 'church.yml',
                '--model_path', 'ckpts/ddpm/church_outdoor.ckpt',
            ]
        )
    elif source_model_type == 'imagenet256':
        raise NotImplementedError()  # Find other checkpoints.
    elif source_model_type == 'imagenet512':
        assert source_model_path is None
        ddim_args = parser.parse_args(
            [
                '--config', 'imagenet.yml',
                '--model_path', 'ckpts/ddpm/512x512_diffusion.pt',
            ]
        )
    else:
        raise NotImplementedError()

    # parse config file
    with open(os.path.join('ckpts/ddpm/configs', ddim_args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return ddim_args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def denoising_step_with_eps(xt, eps, t, t_next, *,
                            models,
                            logvars,
                            b,
                            sampling_type='ddpm',
                            eta=0.0,
                            learn_sigma=False,
                            hybrid=False,
                            hybrid_config=None,
                            ratio=1.0,
                            out_x0_t=False,
                            ):

    assert eps.shape == xt.shape

    # Compute noise and variance
    if type(models) != list:
        model = models
        et = model(xt, t)
        if et.shape != xt.shape:
            et, model_var_values = torch.split(et, et.shape[1] // 2, dim=1)
        if learn_sigma:
            et, model_var_values = torch.split(et, et.shape[1] // 2, dim=1)
            # calculations for posterior q(x_{t-1} | x_t, x_0)
            bt = extract(b, t, xt.shape)
            at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)
            at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)
            posterior_variance = bt * (1.0 - at_next) / (1.0 - at)
            # log calculation clipped because the posterior variance is 0 at the
            # beginning of the diffusion chain.
            min_log = torch.log(posterior_variance.clamp(min=1e-6))
            max_log = torch.log(bt)
            frac = (model_var_values + 1) / 2
            logvar = frac * max_log + (1 - frac) * min_log
        else:
            logvar = extract(logvars, t, xt.shape)
    else:
        if not hybrid:
            et = 0
            logvar = 0
            if ratio != 0.0:
                et_i = ratio * models[1](xt, t)
                if learn_sigma:
                    raise NotImplementedError()
                    et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                    logvar += logvar_learned
                else:
                    logvar += ratio * extract(logvars, t, xt.shape)
                et += et_i

            if ratio != 1.0:
                et_i = (1 - ratio) * models[0](xt, t)
                if learn_sigma:
                    raise NotImplementedError()
                    et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                    logvar += logvar_learned
                else:
                    logvar += (1 - ratio) * extract(logvars, t, xt.shape)
                et += et_i

        else:
            for thr in list(hybrid_config.keys()):
                if t.item() >= thr:
                    et = 0
                    logvar = 0
                    for i, ratio in enumerate(hybrid_config[thr]):
                        ratio /= sum(hybrid_config[thr])
                        et_i = models[i+1](xt, t)
                        if learn_sigma:
                            raise NotImplementedError()
                            et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                            logvar_i = logvar_learned
                        else:
                            logvar_i = extract(logvars, t, xt.shape)
                        et += ratio * et_i
                        logvar += ratio * logvar_i
                    break

    # Compute the next x
    bt = extract(b, t, xt.shape)  # bt is the \beta_t
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)

    if t_next.sum() == -t_next.shape[0]:  # if t_next is -1
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)  # at_next is the \hat{\alpha}_{t_next}

    xt_next = torch.zeros_like(xt)
    if sampling_type == 'ddpm':
        weight = bt / torch.sqrt(1 - at)

        mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)
        noise = eps
        mask = 1 - (t == 0).float()
        mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
        xt_next = xt_next.float()

    elif sampling_type == 'ddim':
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # predicted x0_t
        if eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        elif at > (at_next):
            print('Inversion process is only possible with eta = 0')
            raise ValueError
        else:
            c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()  # sigma_t
            c2 = ((1 - at_next) - c1 ** 2).sqrt()  # direction pointing to x_t
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * eps

    if out_x0_t == True:
        return xt_next, x0_t
    else:
        return xt_next


def compute_eps(xt, xt_next, t, t_next, models, sampling_type, b, logvars, eta, learn_sigma):

    assert eta is None or eta > 0
    # Compute noise and variance
    if type(models) != list:
        model = models
        et = model(xt, t)
        if et.shape != xt.shape:
            et, model_var_values = torch.split(et, et.shape[1] // 2, dim=1)
        if learn_sigma:
            # calculations for posterior q(x_{t-1} | x_t, x_0)
            bt = extract(b, t, xt.shape)
            at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)
            at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)
            posterior_variance = bt * (1.0 - at_next) / (1.0 - at)
            # log calculation clipped because the posterior variance is 0 at the
            # beginning of the diffusion chain.
            min_log = torch.log(posterior_variance.clamp(min=1e-6))
            max_log = torch.log(bt)
            frac = (model_var_values + 1) / 2
            logvar = frac * max_log + (1 - frac) * min_log
        else:
            logvar = extract(logvars, t, xt.shape)
    else:
        raise NotImplementedError()

    # Compute the next x
    bt = extract(b, t, xt.shape)  # bt is the \beta_t
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)

    assert not t_next.sum() == -t_next.shape[0]  # t_next should never be -1
    assert not t.sum() == 0  # t should never be 0
    at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)  # at_next is the \hat{\alpha}_{t_next}

    if sampling_type == 'ddpm':
        weight = bt / torch.sqrt(1 - at)

        mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)
        print('torch.exp(0.5 * logvar).sum()', torch.exp(0.5 * logvar).sum())
        eps = (xt_next - mean) / torch.exp(0.5 * logvar)

    elif sampling_type == 'ddim':
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # predicted x0_t

        c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()  # sigma_t
        c2 = ((1 - at_next) - c1 ** 2).sqrt()  # direction pointing to x_t
        eps = (xt_next - at_next.sqrt() * x0_t - c2 * et) / c1
    else:
        raise ValueError()

    return eps


def sample_xt_next(x0, xt, t, t_next, sampling_type, b, eta):
    bt = extract(b, t, xt.shape)  # bt is the \beta_t
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)

    assert not t_next.sum() == -t_next.shape[0]  # t_next should never be -1
    assert not t.sum() == 0  # t should never be 0
    at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)  # at_next is the \hat{\alpha}_{t_next}

    if sampling_type == 'ddpm':
        w0 = at_next.sqrt() * bt / (1 - at)
        wt = (1 - bt).sqrt() * (1 - at_next) / (1 - at)
        mean = w0 * x0 + wt * xt

        var = bt * (1 - at_next) / (1 - at)

        xt_next = mean + var.sqrt() * torch.randn_like(x0)
    elif sampling_type == 'ddim':
        et = (xt - at.sqrt() * x0) / (1 - at).sqrt()  # posterior et given x0 and xt
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()  # sigma_t
        c2 = ((1 - at_next) - c1 ** 2).sqrt()  # direction pointing to x_t
        xt_next = at_next.sqrt() * x0 + c2 * et + c1 * torch.randn_like(x0)
    else:
        raise ValueError()

    return xt_next


def sample_xt(x0, t, b):
    at = extract((1.0 - b).cumprod(dim=0), t, x0.shape)  # at is the \hat{\alpha}_t
    print('at', at)
    xt = at.sqrt() * x0 + (1 - at).sqrt() * torch.randn_like(x0)
    return xt


class DDPMDDIMWrapper(torch.nn.Module):

    def __init__(self, source_model_type, sample_type, custom_steps, es_steps, source_model_path=None,
                 refine_steps=0, refine_iterations=1, eta=None, t_0=None, enforce_class_input=None):
        super(DDPMDDIMWrapper, self).__init__()

        self.enforce_class_input = enforce_class_input
        self.custom_steps = custom_steps
        self.refine_steps = refine_steps
        self.refine_iterations = refine_iterations
        self.sample_type = sample_type
        self.eta = eta
        self.t_0 = t_0 if t_0 is not None else 999
        self.es_steps = es_steps

        if self.sample_type == 'ddim':
            assert self.eta > 0
        elif self.sample_type == 'ddpm':
            assert self.eta is None
        else:
            raise ValueError()

        # Set up generator
        self.ddim_args, config = prepare_ddpm_ddim(source_model_type, source_model_path)

        print(self.ddim_args)
        print(config)

        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.register_buffer(
            'betas', torch.from_numpy(betas).float()
        )
        self.num_timesteps = betas.shape[0]

        # ----------- Model -----------#
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if config.data.dataset in ["CelebA_HQ", "LSUN"]:
            self.generator = DDPM(config)
            self.learn_sigma = False
            if config.model.var_type == "fixedlarge":
                self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))
            elif config.model.var_type == 'fixedsmall':
                self.logvar = np.log(np.maximum(posterior_variance, 1e-20))
            else:
                raise ValueError()
            print("Original diffusion Model loaded.")
        elif config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            self.generator = i_DDPM(config.data.dataset)
            self.learn_sigma = False
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise NotImplementedError()
        init_ckpt = torch.load(self.ddim_args.model_path)
        self.generator.load_state_dict(init_ckpt)

        self.resolution = config.data.image_size
        self.channels = config.data.channels
        self.latent_dim = self.resolution ** 2 * self.channels * self.es_steps
        # Freeze.
        requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    def generate(self, z, class_label):
        if (self.t_0 + 1) % self.custom_steps == 0:
            seq_inv = range(0, self.t_0 + 1, (self.t_0 + 1) // self.custom_steps)
            assert len(seq_inv) == self.custom_steps
        else:
            seq_inv = np.linspace(0, 1, self.custom_steps) * self.t_0
        seq_inv = [int(s) for s in list(seq_inv)][:self.es_steps]  # 0, 1, ..., t_0
        seq_inv_next = ([-1] + list(seq_inv[:-1]))[:self.es_steps]  # -1, 0, 1, ..., t_0-1

        bsz = z.shape[0]
        eps_list = z.view(bsz, self.es_steps, self.channels, self.resolution, self.resolution)
        x_T = eps_list[:, 0]
        eps_list = eps_list[:, 1:]

        x = x_T
        if self.enforce_class_input:
            assert class_label is not None
            raise NotImplementedError()
        else:
            for it, (i, j) in enumerate(zip(reversed(seq_inv), reversed(seq_inv_next))):
                t = (torch.ones(bsz) * i).to(self.device)
                t_next = (torch.ones(bsz) * j).to(self.device)

                if it < self.es_steps - 1:
                    eps = eps_list[:, it]
                    x = denoising_step_with_eps(x, eps=eps, t=t, t_next=t_next, models=self.generator,
                                                logvars=self.logvar,
                                                sampling_type=self.sample_type,
                                                b=self.betas,
                                                eta=self.eta,
                                                learn_sigma=self.learn_sigma)
                else:
                    x = denoising_step(x, t=t, t_next=t_next, models=self.generator,
                                       logvars=self.logvar,
                                       sampling_type=self.sample_type,
                                       b=self.betas,
                                       eta=self.eta,
                                       learn_sigma=self.learn_sigma)

            if self.refine_steps == 0:
                img = x
            else:
                for r in range(self.refine_iterations):
                    refine_eta = 1
                    # Sample xt
                    t = (torch.ones(bsz) * self.refine_steps - 1).to(self.device)
                    xt = sample_xt(x0=x, t=t, b=self.betas)
                    # Denoise
                    x = xt
                    assert self.refine_steps < self.custom_steps
                    seq_inv_refine = seq_inv[:self.refine_steps]
                    seq_inv_next_refine = seq_inv_next[:self.refine_steps]
                    for i, j in zip(reversed(seq_inv_refine), reversed(seq_inv_next_refine)):
                        t = (torch.ones(bsz) * i).to(self.device)
                        t_next = (torch.ones(bsz) * j).to(self.device)
                        x = denoising_step(x, t=t, t_next=t_next, models=self.generator,
                                           logvars=self.logvar,
                                           sampling_type=self.sample_type,
                                           b=self.betas,
                                           eta=refine_eta,
                                           learn_sigma=self.learn_sigma)
                img = x

        return img

    def encode(self, image, class_label=None):
        # Eval mode for the generator.
        self.generator.eval()

        if (self.t_0 + 1) % self.custom_steps == 0:
            seq_inv = range(0, self.t_0 + 1, (self.t_0 + 1) // self.custom_steps)
            assert len(seq_inv) == self.custom_steps
        else:
            seq_inv = np.linspace(0, 1, self.custom_steps) * self.t_0
        seq_inv = [int(s) for s in list(seq_inv)][:self.es_steps]
        seq_inv_next = ([-1] + list(seq_inv[:-1]))[:self.es_steps]

        # Normalize.
        image = (image - 0.5) * 2.0
        # Resize.
        assert image.shape[2] == image.shape[3] == self.resolution

        with torch.no_grad():
            x0 = image
            bsz = x0.shape[0]

            # DPM-Encoder.
            if self.enforce_class_input:
                assert class_label is not None
                raise NotImplementedError()
            else:
                T = (torch.ones(bsz) * (self.es_steps - 1)).to(self.device)
                xT = sample_xt(x0=x0, t=T, b=self.betas)
                z_list = [xT, ]

                xt = xT
                for it, (i, j) in enumerate(zip(reversed(seq_inv), reversed(seq_inv_next))):
                    t = (torch.ones(bsz) * i).to(self.device)
                    t_next = (torch.ones(bsz) * j).to(self.device)

                    if it < self.es_steps - 1:
                        xt_next = sample_xt_next(
                            x0=x0,
                            xt=xt,
                            t=t,
                            t_next=t_next,
                            sampling_type=self.sample_type,
                            b=self.betas,
                            eta=self.eta,
                        )
                        eps = compute_eps(
                            xt=xt,
                            xt_next=xt_next,
                            t=t,
                            t_next=t_next,
                            models=self.generator,
                            sampling_type=self.sample_type,
                            b=self.betas,
                            logvars=self.logvar,
                            eta=self.eta,
                            learn_sigma=self.learn_sigma,
                        )
                        print(it, (eps ** 2).sum().item())
                        xt = xt_next
                        z_list.append(eps)
                    else:
                        break

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





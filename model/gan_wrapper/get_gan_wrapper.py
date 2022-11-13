

def get_gan_wrapper(args, target=False):

    kwargs = {}
    for kw, arg in args:
        if kw != 'gan_type':
            if (not kw.startswith('source_')) and (not kw.startswith('target_')):
                kwargs[kw] = arg
            else:
                if target and kw.startswith('target_'):
                    final = kw[len('target_'):]
                    kwargs[f'source_{final}'] = arg
                elif (not target) and kw.startswith('source_'):
                    kwargs[kw] = arg

    if args.gan_type == "LatentDiffStochastic":
        from .latentdiff_stochastic_wrapper import LatentDiffStochasticWrapper
        return LatentDiffStochasticWrapper(**kwargs)
    elif args.gan_type == "DDPM_DDIM":
        from .ddpm_ddim_wrapper import DDPMDDIMWrapper
        return DDPMDDIMWrapper(**kwargs)
    elif args.gan_type == "LatentDiffStochasticText":
        from .latentdiff_stochastic_text_wrapper import LatentDiffStochasticTextWrapper
        return LatentDiffStochasticTextWrapper(**kwargs)
    elif args.gan_type == "SDStochasticText":
        from .stable_diffusion_stochastic_text_wrapper import SDStochasticTextWrapper
        return SDStochasticTextWrapper(**kwargs)
    else:
        raise ValueError()


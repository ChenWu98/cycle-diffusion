

def get_energy(name, energy_kwargs, gan_wrapper):

    if name == "PriorZEnergy":
        from .prior_z import PriorZEnergy
        return PriorZEnergy()
    else:
        raise ValueError()


def parse_key(key):
    if key.endswith('1'):
        return key[:-1], 1
    elif key.endswith('2'):
        return key[:-1], 2
    elif key.endswith('Pair'):
        return key[:-len('Pair')], 'Pair'
    else:
        return key, None

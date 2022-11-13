import torch.nn as nn


class PriorZEnergy(nn.Module):
    def __init__(self):
        super(PriorZEnergy, self).__init__()

    @ staticmethod
    def prepare_inputs(**kwargs):
        return {
            'z': kwargs['z'],
        }

    def forward(self, z):
        if z.ndim == 2:
            prior_z_loss = 0.5 * (z ** 2).sum(1)
        elif z.ndim == 3:
            prior_z_loss = 0.5 * (z ** 2).sum(2).sum(1)
        else:
            raise ValueError()

        return prior_z_loss

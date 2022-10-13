# Created by Chen Henry Wu
MAX_SAMPLE_SIZE = 4096


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

import torch
import numpy as np


def truncated_gumbel(logit, truncation):
    """truncated_gumbel
    :param logit: Location of the Gumbel variable (e.g., log probability)
    :param truncation: Value of Maximum Gumbel
    """
    # Note: In our code, -inf shows up for zero-probability events, which is
    # handled in the topdown function
    assert not np.isneginf(logit)

    gumbel = np.random.gumbel(size=(truncation.shape[0])) + logit
    trunc_g = -np.log(np.exp(-gumbel) + np.exp(-truncation))
    return trunc_g


def _topdown(logits, k, nsamp=1):
    """topdown
    Top-down sampling from the Gumbel posterior
    :param logits: log probabilities of each outcome
    :param k: Index of observed maximum
    :param nsamp: Number of samples from gumbel posterior
    """
    np.testing.assert_approx_equal(np.sum(np.exp(logits)), 1, significant=5), "Probabilities do not sum to 1"
    ncat = logits.shape[0]

    gumbels = np.zeros((nsamp, ncat))

    # Sample top gumbels
    topgumbel = np.random.gumbel(size=(nsamp))

    for i in range(ncat):
        # This is the observed outcome
        if i == k:
            gumbels[:, k] = topgumbel - logits[i]
        # These were the other feasible options (p > 0)
        elif not(np.isneginf(logits[i])):
            gumbels[:, i] = truncated_gumbel(logits[i], topgumbel) - logits[i]
        # These have zero probability to start with, so are unconstrained
        else:
            gumbels[:, i] = np.random.gumbel(size=nsamp)
    print('gumbels.max():', gumbels.max())
    print('gumbels.min():', gumbels.min())

    return gumbels


def topdown(logits, k, nsamp=1):
    """topdown
    Top-down sampling from the Gumbel posterior
    :param logits: log probabilities of each outcome
    :param k: Index of observed maximum
    :param nsamp: Number of samples from gumbel posterior
    """
    np.testing.assert_approx_equal(np.sum(np.exp(logits)), 1, significant=5), "Probabilities do not sum to 1"
    ncat = logits.shape[0]

    # Sample top gumbels
    topgumbel = np.random.gumbel(size=(nsamp))

    gumbel = np.random.gumbel(size=(nsamp, ncat)) + logits[None, :]
    trunc_g = -np.log(np.exp(-gumbel) + np.exp(-topgumbel)[:, None])

    gumbels = np.where(np.isneginf(logits)[None, :], np.random.gumbel(size=(nsamp, ncat)), trunc_g - logits[None, :])
    gumbels[:, k] = topgumbel - logits[k]

    print('gumbels.max():', gumbels.max())
    print('gumbels.min():', gumbels.min())

    return gumbels
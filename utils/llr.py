"""
Utilities for computing log-likelihood ratios for discontinuity search.
"""
import numpy as np
from scipy.optimize import fsolve

def beta_partial(beta, Ps, Ts):
    """
    The partial derivative of the log likelihood with respect to
    the beta parameter we wish to maximize.

    This is equation 10 in the Herlands et al. paper. This equation
    applies to solving for all three beta parameters.

    Note that for beta_0 the sum is taken over all data points (does not change),
    while for beta_g0 and beta_g2 the sum is taken over their corresponding group partitions.

    The goal is to solve for beta * partial LLR = 0

    Args:
        beta (float): the scalar beta value that we want to maximize
        Ps (np.ndarray): the p(x_i) for all the data points
        Ts (np.ndarray): the T_i for all the data points

    Returns:
        float: the result of beta_0 * partial LLR
    """
    assert Ps.shape[0] == Ts.shape[0]

    inner_sum = Ts - ((Ps*beta) / (1 - Ps + (beta * Ps)))
    return np.sum(inner_sum)


def bernoulli_llr(beta_0, beta_g0, beta_g1, Ps, Ts, Gs):
    """
    Computes the LLR of the null and alternative models for testing
    for the existence of a discontinuity under binary treatment.

    This is equation 9 in the Herlands et al. paper.

    Args:
        beta_0 (float): MLE beta_0 parameter
        beta_g0 (float): MLE beta_g0 parameter
        beta_g1 (float): MLE beta_g1 parameter
        Ps (np.ndarray): the p(x_i) for all the data points
        Ts (np.ndarray): the T_i for all the data points
        Gs (np.ndarray): the group assignments g_i for all the data points

    Returns:
        float: the log likelihood ratio
    """

    mus = ((1 - Gs) * beta_g0) + (Gs * beta_g1)
    epsilon = 1e-7
    inner_sum = (Ts * np.log(epsilon + mus / beta_0)) \
        + np.log(epsilon + 1 - Ps + (beta_0 * Ps))  \
        - np.log(epsilon + 1 - Ps + (mus * Ps))

    assert inner_sum.shape[0] == mus.shape[0]

    return np.sum(inner_sum)


def compute_llr(Ps,Ts,Gs):
    """
    Wrapper function for finding the MLEs for the betas and computing the LLR.

    Args:
        Ps (np.ndarray): the p(x_i) for all the data points
        Ts (np.ndarray): the T_i for all the data points
        Gs (np.ndarray): the group assignments g_i for all the data points


    Returns:
        float: the log likelihood ratio
    """

    assert Gs.shape[0] == Ps.shape[0]
    assert Gs.shape[0] == Ts.shape[0]

    beta_0_hat = fsolve(beta_partial, 0, args=(Ps, Ts), xtol=1e-06, maxfev=500)

    g0_Ps = Ps[np.argwhere(Gs == 0).flatten()]
    g0_Ts = Ts[np.argwhere(Gs == 0).flatten()]

    g1_Ps = Ps[np.argwhere(Gs == 1).flatten()]
    g1_Ts = Ts[np.argwhere(Gs == 1).flatten()]

    beta_g0_hat = fsolve(beta_partial, 0, args=(g0_Ps, g0_Ts), xtol=1e-06, maxfev=500)
    beta_g1_hat = fsolve(beta_partial, 0, args=(g1_Ps, g1_Ts), xtol=1e-06, maxfev=500)

    return bernoulli_llr(beta_0_hat,
                         beta_g0_hat,
                         beta_g1_hat,
                         Ps,
                         Ts,
                         Gs)

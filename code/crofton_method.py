import numpy as np
import macroeco.models as mod
import scipy.optimize as opt

"""
Implement Crofton method for estimating parasite-induce mortality
"""

def nbd_fixed_mean(data, mean):

    loglike = lambda k: -np.sum(mod.nbinom.logpmf(data, mean, k))
    return opt.fmin(loglike, 1, disp=0)[0]




def crofton_method1(data, bin_edges, guess=None):
    """
    Crofton's method for estimating parasite induced mortality

    Parameters
    ----------
    data : array
        Parasite data across hosts
    bin_boundaries : list of tuples
        List of tuples specifying upper and lower bin boundaries. The final
        value in bin_edges is an inclusive upper bound!
    guess : None or tuple
        If tuple, should be a guess for mu and k of the negative binomial

    """


    obs = np.histogram(data, bins=bin_edges)[0]

    splits = []
    for i, b in enumerate(bin_edges):

        if i != len(bin_edges) - 1:
            splits.append(np.arange(bin_edges[i], bin_edges[i + 1]))

    N_guess = len(data)

    if not guess:
        mu_guess, k_guess = mod.nbinom.fit_mle(data)
    else:
        mu_guess = guess[0]
        k_guess = guess[1]

    def opt_fxn(params):
        N, mu, k = params
        pred = [N * np.sum(mod.nbinom.pmf(s, mu, k)) for s in splits]
        return np.sum((obs - pred)**2)

    opt_params = opt.fmin(opt_fxn, np.array([N_guess, mu_guess, k_guess]),
        xtol=0.01, ftol=0.01)
    pred = [opt_params[0] * np.sum(mod.nbinom.pmf(s, opt_params[1],
        opt_params[2])) for s in splits]

    return opt_params, obs, pred, splits


def crofton_method2(data, trun_value):
    """
    Crofton's method for estimating parasite induced mortality

    Parameters
    ----------
    data : array
        Parasite data across hosts
    bin_boundaries : list of tuples
        List of tuples specifying upper and lower bin boundaries

    """

    vals = np.arange(0, trun_value + 1)
    obs = [np.sum(data == i) for i in vals]

    N_guess = len(data)
    mu_guess, k_guess = mod.nbinom.fit_mle(data)

    def opt_fxn(params):
        N, mu, k = params
        pred = N * mod.nbinom.pmf(vals, mu, k)

        return np.sum((obs - pred)**2)

    opt_params = opt.fmin(opt_fxn, np.array([N_guess, mu_guess, k_guess]))
    pred = opt_params[0] * mod.nbinom.pmf(vals, opt_params[1], opt_params[2])

    return opt_params, obs, pred

apo = np.repeat((50, 150,250, 350, 450, 550, 650, 750), (79, 10, 7, 7, 1, 4, 2,
    1))

n_sal = np.repeat((30, 80, 200, 250, 350, 390, 475, 550, 650, 700),
    (32, 27, 8, 10, 8, 6, 5, 1, 2, 1))

n_sal_bins = [0, 76, 151, 226, 301, 376, 451, 526, 601, 676, 751]






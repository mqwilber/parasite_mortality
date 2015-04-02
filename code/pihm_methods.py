from __future__ import division
import numpy as np
import macroeco.models as mod
import scipy.optimize as opt
import pandas as pd
import statsmodels.api as sm
import pymc as pm
import matplotlib.pyplot as plt

"""
Module contains the updated functions and classes for assessing
parasite-induced host mortality (PIHM)

Functions
---------

"""

class PIHM:
    """
    Class PIHM (parasite-induced host morality) provides an interface for
    analyzing the effect of parasite-induced host mortality on a given dataset

    """

    def __init__(self, data=None):
        """
        """

        self.data = data
        self.Np = None
        self.mup = None
        self.kp = None
        self.a = None
        self.b = None

    def crofton_method(self, bin_edges, guess=None, full_output=False):
        """
        Crofton's method for estimating the pre-mortality parameters N_p
        (population size before mortality),
        mu_p (mean parasite load before mortality), and
        k_p (parasite aggregation before mortality)

        Parameters
        ----------
        data : array
            Parasite data across hosts
        bin_edges : list
            List specifying upper and lower bin boundaries. Upper right most bin
            boundary is inclusive; all other upper bounds are exclusive.  Example:
            You want to group as follows [0, 1, 3], [4, 5, 6], [7, 8, 9] you would
            specify [0, 4, 7, 9.1].  This can be a little weird, but if you want to
            bin from [0], [1], [2], you would specify [0, 1, 2, 2.9].  You have to
            be a bit careful with the upper bound because it is INCLUSIVE.
        guess : None or tuple
            If tuple, should be a guess for mu and k of the negative binomial
        full_output : bool
            If full_output is False, just returns N_p, mu_p, and k_p. If True,
            return opt_params : (N_p, mu_p, k_p)
            obs : observed number of hosts in groups
            pred: predicted number of hosts in groups
            splits: the group boundaries
            np.sum((obs - pred)**2 / pred): minimized chi-squared statistic

        Returns
        -------
        See full_output


        """

        data = self.data

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

            # Minimizing chi-squared
            return np.sum((obs - pred)**2 / pred)

        # Fixing the boundary of guesses for k, can't be above 5

        # bounds = [(0, 2*N_guess), (0, 2*mu_guess), (0.001, 5)]
        # full_out = opt.fmin_l_bfgs_b(opt_fxn,
        #                 np.array([N_guess, mu_guess, k_guess]),
        #                 bounds=bounds, approx_grad=True, disp=0)
        # opt_params = full_out[0]

        full_out = opt.fmin(opt_fxn, np.array([N_guess, mu_guess, k_guess]),
                                        disp=0)

        opt_params = full_out
        pred = [opt_params[0] * np.sum(mod.nbinom.pmf(s, opt_params[1],
            opt_params[2])) for s in splits]

        self.Np, self.mup, self.kp = opt_params

        if not full_output:
            return opt_params
        else:
            return opt_params, obs, pred, splits, np.sum((obs - pred)**2 / pred)


    def adjei_method(self, full_bin_edges, crof_bin_edges, no_bins=True,
                    crof_params=None):
        """
        Implements adjei method.  Truncated negative binomial method is fit using
        Crofton Method and Adjei method is used to estimate morality function.

        Parameters
        -----------
        data : array
            Host-parasite data
        bin_edges : array-like
            List specifying upper and lower bin boundaries. Upper right most bin
            boundary is inclusive; all other upper bounds are exclusive.  Example:
            You want to group as follows [0, 1, 3], [4, 5, 6], [7, 8, 9] you would
            specify [0, 4, 7, 9.1].
        bin_data : array-like
            List specifying how you want to bin all the data. This binning is done
            on the full data after the truncated negative binomial has been
            estimated on the truncated data. If you don't want any binning specify
            np.arange(0, np.max(data)  + 2) or set no_bins=True.
        crof_params : tuple
            Must be a tuple of (N, mu, k).  Parameters of the population before
            mortality.

        """
        data = self.data
        num_hosts = len(data)
        max_worms = np.max(data)

        if no_bins:
            full_bin_edges = np.arange(0, np.max(data) + 2)

        # Use Crofton Method to truncate data.


        if crof_params:
            N, mu, k = crof_params
        else:
            (N, mu, k), obs, pred, split, _ = crofton_method(data, crof_bin_edges)

        # 2. Bin empirical data and theoretical data
        empirical_binned = np.histogram(data, bins=full_bin_edges)[0]
        theor_data = np.round(mod.nbinom.pmf(np.arange(0, max_worms + 1), mu, k) *
                        N, decimals=0)
        full_theor_data = np.repeat(np.arange(0, max_worms + 1),
                theor_data.astype(int))
        theor_binned = np.histogram(full_theor_data, bins=full_bin_edges)[0]

        # Put everything into dataframe
        if no_bins:
            para_num = np.arange(0, np.max(data) + 1)

        else:
            para_num = [np.mean((full_bin_edges[i], full_bin_edges[i + 1])) for i in
                                    xrange(len(full_bin_edges) - 1)]

        all_data = pd.DataFrame({"para": para_num, 'emp': empirical_binned,
                            'pred': theor_binned})

        # If emp is greater than pred set to pred
        all_data['emp_pure'] = all_data['emp']
        ind = all_data['emp'] > all_data['pred']
        all_data['emp'][ind] = all_data['pred'][ind]

        # Drop all zeros
        ind_nz = all_data['pred'] != 0
        all_data = all_data[ind_nz]

        # Drop the zero class if not binned
        if no_bins:
            all_data = all_data.ix[all_data.index[1:]]

        params = fit_glm(all_data)
        a = params[0]
        b = params[1]

        return (N, mu, k, a, b), all_data



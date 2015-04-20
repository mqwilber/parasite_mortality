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

    Description
    ------------
    The class PIHM has two primary functions: 1) To test whether an observed
    host-parasite distribution is experiencing parasite induced host morality.

    """

    def __init__(self, data=None):

        self.data = data  # Dataset to fit
        self.Np = None  # Total hosts pre-mortality
        self.mup = None  # Mean parasites per host pre-mortality
        self.kp = None  # Aggretaion of parasites pre-moratlity
        self.a = None  # a parameter of the survival function
        self.b = None  # b parameter of the survival function


    def set_premort_params(self, Np, mup, kp):
        """
        Set the premortality parameters
        """

        self.Np, self.mup, self.kp = Np, mup, kp

    def set_all_params(self, Np, mup, kp, a, b):
        """
        Set the premortality parameters
        """

        self.Np, self.mup, self.kp, self.a, self.b = Np, mup, kp, a, b

    def get_premort_params(self):
        """
        Return the premortality params
        """

        return self.Np, self.mup, self.kp

    def get_all_params(self):
        """
        Return the premortality params
        """

        return self.Np, self.mup, self.kp, self.a, self.b


    def crofton_method(self, crof_bin_edges, guess=None):
        """
        Crofton's method for estimating the pre-mortality parameters N_p
        (population size before mortality),


        mu_p (mean parasite load before mortality), and k_p (parasite
        aggregation before mortality).  Uses the iterative technique proposed
        by Lester (1977), which minimizes the chi-squared statistic

        Parameters
        ----------
        data : array
            Parasite data across hosts
        crof_bin_edges : list
            List specifying upper and lower bin boundaries. Upper right most bin
            boundary is inclusive; all other upper bounds are exclusive.  Example:
            You want to group as follows [0, 1, 3], [4, 5, 6], [7, 8, 9] you would
            specify [0, 4, 7, 9.1].  This can be a little weird, but if you want to
            bin from [0], [1], [2], you would specify [0, 1, 2, 2.9].  You have to
            be a bit careful with the upper bound because it is INCLUSIVE.
        guess : None or tuple
            If tuple, should be a guess for mu and k of the negative binomial.
            If None, uses fitted mu and k for a guess.

        Returns
        -------
        : Np, mup, kp


        """

        data = self.data

        obs = np.histogram(data, bins=crof_bin_edges)[0]

        splits = split_data(crof_bin_edges)

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

        full_out = opt.fmin(opt_fxn, np.array([N_guess, mu_guess, k_guess]),
                                        disp=0)
        # Calculate predicted in bins
        opt_params = full_out
        # pred = [opt_params[0] * np.sum(mod.nbinom.pmf(s, opt_params[1],
        #     opt_params[2])) for s in splits]

        self.set_premort_params(*opt_params)

        return self.get_premort_params()


    def obs_pred(self, full_bin_edges):
        """
        Returns the observed and predicted vectors for the preset values of
        Np, mup, and kp.  Will throw an error in Np, mup and kp are not defined.
        You can either define these manually or run the crofton_method first.

        Parameters
        ----------
        full_bin_edges : array-like
            List specifying upper and lower bin boundaries. Upper right most bin
            boundary is inclusive; all other upper bounds are exclusive.  Example:
            You want to group as follows [0, 1, 3], [4, 5, 6], [7, 8, 9] you would
            specify [0, 4, 7, 9.1]

        Returns
        --------
        : DataFrame
            Three columns. obs: The observed number of hosts in a given range
            pred: the predicted number of hosts in a given range
            range: The range of the data with the upper bound being exlusive.
            For example, (1, 2) specifies a host with 1 parasites. (1, 3)
            specifies a host with 1 or 2 parasites.


        """

        # Check that all parameters are assigned
        if None in [self.Np, self.mup, self.kp]:
            raise TypeError("Np, mup, and kp must all be assigned. See docstring")

        obs, edges = np.histogram(self.data, bins=full_bin_edges)

        splits = split_data(full_bin_edges)

        pred = [self.Np * np.sum(mod.nbinom.pmf(s, self.mup,
            self.kp)) for s in splits]

        min_max = [str((np.min(s), np.max(s) + 1)) for s in splits]

        res_df = pd.DataFrame({"observed": obs, "predicted": pred,
                                "range": min_max})

        return res_df


    def adjei_method(self, full_bin_edges, crof_bin_edges, no_bins=True,
                    run_crof=False):
        """
        Implements adjei method.  Truncated negative binomial method is fit using
        Crofton Method and Adjei method is used to estimate morality function.

        Parameters
        -----------
        full_bin_edges : array-like
            List specifying upper and lower bin boundaries. Upper right most bin
            boundary is inclusive; all other upper bounds are exclusive.  Example:
            You want to group as follows [0, 1, 3], [4, 5, 6], [7, 8, 9] you would
            specify [0, 4, 7, 9.1].
        crof_bin_edges : list
            List specifying upper and lower bin boundaries. Upper right most bin
            boundary is inclusive; all other upper bounds are exclusive.  Example:
            You want to group as follows [0, 1, 3], [4, 5, 6], [7, 8, 9] you would
            specify [0, 4, 7, 9.1].  This can be a little weird, but if you want to
            bin from [0], [1], [2], you would specify [0, 1, 2, 2.9].  You have to
            be a bit careful with the upper bound because it is INCLUSIVE.
        run_crof : bool
            If True, runs the crofton method. Otherwise does not.

        Returns
        -------
        : Np, mup, kp, a, b

        Also stores the data used to fit the GLM in self.adjei_all_data

        """
        num_hosts = len(self.data)
        max_worms = np.max(self.data)

        if no_bins:
            full_bin_edges = np.arange(0, np.max(self.data) + 2)

        if run_crof:
            # Only run crofton method if necessary
            _, _, _ = self.crofton_method(crof_bin_edges)

        # 2. Bin empirical self.data and theoretical self.data
        empirical_binned = np.histogram(self.data, bins=full_bin_edges)[0]
        theor_data = np.round(mod.nbinom.pmf(np.arange(0, max_worms + 1),
                        self.mup, self.kp) * self.Np, decimals=0)
        full_theor_data = np.repeat(np.arange(0, max_worms + 1),
                theor_data.astype(int))
        theor_binned = np.histogram(full_theor_data, bins=full_bin_edges)[0]


        # Put everything into self.dataframe
        if no_bins:
            para_num = np.arange(0, np.max(self.data) + 1)

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

        self.adjei_all_data = all_data

        # Fit the glm model
        params = fit_glm(all_data)
        self.a, self.b = params
        a = params[0]
        b = params[1]

        return self.get_all_params()


    def likelihood_method(self, full_fit=True, max_sum=1000, guess=[10, -5]):
        """
        Using likelihood method to estimate parameter of truncated NBD and survival
        fxn

        Parameters
        ----------
        data : array-like
            Observed host parasite data
        full_fit : bool
            If True, fits mup, kp, a, b. If False, used preset Np,
        max_sum: int
            Upper bound on normalizing constant.  Technically bounded by positive
            infinity, but in practice a lower upper bound works fine.
        guess : list
            Guess for a and b of the survival function

        Returns
        -------
        : tuple
            (mu, k, a, b)

        Notes
        -----

        """

        if not full_fit:

            if None in list(self.get_premort_params())[1:]:
                raise TypeError("mup, kp must be preset")

            params = opt.fmin(likefxn2, guess, args=(self.data, self.mup,
                                    self.kp), disp=1)
            out_params = [self.mup, self.kp] + list(params)

        else:

            mu_guess = np.mean(self.data)
            k_guess = 1
            out_params = opt.fmin(likefxn1, [mu_guess, k_guess] + guess,
                        args=(self.data,))

        self.mup, self.kp, self.a, self.b = out_params

        return tuple(out_params)


    def get_pihm_samples(self, other_death=False, percent=0.1):
        """

        Simulate parasite induced host_mortality. Uses self.Np, self.mup,
        self.kp, self.a,  and self.b to run simulation.

        These must be specified for method to run.
        An error will be thrown otherwise.

        Parameters
        ----------
        other_death : bool
            True if random death occurs. False if parasite mortality is the only
            death
        percent : float
            Between 0 and 1.  The random mortality percentage

        Return
        ------
        : tuple
            (Post-portality random sample, pre-mortality random sample)

        """

        if None in list(self.get_all_params()):
            raise TypeError("Np, mup, kp, a, b must all be pre-set")

        init_pop = mod.nbinom.rvs(self.mup, self.kp, size=self.Np)

        survival_probs = surv_prob(init_pop, self.a, self.b)
        survival_probs[np.isnan(survival_probs)] = 1

        # Determine stochastic survival
        rands = np.random.rand(len(survival_probs))
        surv = rands < survival_probs
        alive_hosts = init_pop[surv]

        # Additional random death
        if other_death:
            rands = np.random.rand(len(alive_hosts))
            surv = rands > percent
            alive_hosts = alive_hosts[surv]

        #true_death = 1 - len(alive_hosts) / float(len(init_pop))

        return alive_hosts, init_pop


    # def simulate(self, Nps, mups, kps, as, bs, samp_size=150, methods="both"):
    #     """
    #     Method for


    #     Parameters
    #     -----------
    #     Nps : list
    #         List of values of N_p to use for simulation
    #     mups : float
    #         List of mean of the pre-mortality NBD
    #     kps : float
    #         List of k of the pre-mortality NBD
    #     as : float
    #         List of a parameters of the survival function
    #     bs : float
    #         List of b parameters of the survival function
    #     samp_size : int
    #         Number of simlations to run or each parameter combination

    #     """
    #     pass



def pihm_pmf(x, mup, kp, a, b, max_sum=1000):
    """
    The probability masss function for parasite induced host mortality

    Parameters
    ----------
    x : int
        Parasite load
    mup : float
        Mean parasite load pre-mortality
    kp : float
        Aggregation pre-mortality
    a : float
        a of the survival function
    b : float
        b of the survival function
    max_sum : int
        Upper bound for normalization

    Returns
    -------
    : float or array, pmf value for x

    """


    kern = lambda x: surv_prob(x, a, b) * \
                                    mod.nbinom.pmf(x, mup, kp)
    return(kern(x) / sum(kern(np.arange(0, 1000))))



def likefxn1(params, x):
    """ Likelihood fxn for all parameters for pihm pmf
    Returns negative log-likelihood

    """

    mu, k, a, b = params
    return -np.sum(np.log(pihm_pmf(x, mu, k, a, b)))


def likefxn2(params, x, mu, k):
    """

    Likelihood fxn for just a and b for pihm pmf.
    Returns negative log-likelihood

    """

    a, b = params
    return -np.sum(np.log(pihm_pmf(x, mu, k, a, b)))


def extract_simulation_results(sim_results, keys, method_name, param):
    """

    Method to extract parameters from a simulation dictionary as generated
    by analysis scripts

    Parameters
    ----------
    sim_results : dict
        A dictionary with analysis results

    keys : list
        First item is a value for mup, second item is a tuple specify the
        (a, b) pair, third iterm is kp.

    method_name: str
        Either likelihood or adjei

    param: str
        either "a", "b", or "ld50"

    Returns
    : tuple
        samp_sizes, biases, precisions


    """

    sims = sim_results[keys[0]][keys[1]][keys[2]]

    Np_vals = list(sims.viewkeys())

    if method_name == "likelihood":
        index = 0
    else:
        index = 1

    samp_sizes = []
    biases = []
    precisions = []

    for Np in Np_vals:

        res = sims[Np]

        try:
            a_vals, b_vals = zip(*res[index])
        except:
            continue

        N_vals = res[2]

        samp_sizes.append(np.mean(N_vals))

        if param == "a":

            biases.append(scaled_bias(np.array(a_vals), keys[1][0]))
            precisions.append(scaled_precision(np.array(a_vals)))

        elif param == "b":

            biases.append(scaled_bias(np.array(b_vals), keys[1][1]))
            precisions.append(scaled_precision(np.array(b_vals)))

        elif param == "ld50":

            ld50_vals = np.exp(np.array(a_vals) / np.abs(b_vals))
            truth = np.exp(keys[1][0] / np.abs(keys[1][1]))

            biases.append(scaled_bias(ld50_vals, truth))
            precisions.append(scaled_precision(ld50_vals))
        else:
            raise KeyError("Don't recognize parameter: should be a, b, or ld50")

    return samp_sizes, biases, precisions


def split_data(bin_edges):
    """
    Given bin edges, split data in fully enumerate bins. Example: [1, 4, 8]
    becomes [(1, 2, 3), (4, 5, 6, 8)]

    """

    splits = []
    for i, b in enumerate(bin_edges):

        if i != len(bin_edges) - 1:
            splits.append(np.arange(bin_edges[i], bin_edges[i + 1]))

    return splits

def surv_prob(x, a, b):
    """
    Calculate survival probability given a parasite load x
    """
    x = np.atleast_1d(x)

    probs = np.empty(len(x))
    probs[x == 0] = 1
    ind = x != 0
    probs[ind] = np.exp(a + b * np.log(x[ind])) / (1 + np.exp(a + b * np.log(x[ind])))

    return probs

def fit_glm(all_data):
    """
    Fitting the GLM in statsmodels

    Returns
    -------
    : a and b parameters
    """

    all_data = all_data[['emp', 'pred', 'para']].copy()
    all_data['diff'] = np.array(all_data.pred) - np.array(all_data.emp)
    all_data['log_para'] = np.log(all_data['para'])
    new_data = sm.add_constant(all_data)
    glm_fit = sm.GLM(new_data[['emp', 'diff']],
                    new_data[['const', 'log_para']], family=sm.families.Binomial())
    res = glm_fit.fit()

    return res.params

def scaled_bias(data, truth):
    """
    Aboluste value of scaled bias calculation from Walther and Moore 2005
    """

    return np.abs(np.sum(data - truth) / (len(data) * float(truth)))


def scaled_precision(data):
    """
    Just the Coefficient of Variation
    """

    return 100 * np.std(data, ddof=1) / np.mean(data)

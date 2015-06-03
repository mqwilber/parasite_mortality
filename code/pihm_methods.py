from __future__ import division
import numpy as np
import macroeco.models as mod
import macroeco.compare as comp
import scipy.optimize as opt
import pandas as pd
import statsmodels.api as sm
import pymc as pm
import matplotlib.pyplot as plt
from scipy.stats import chisqprob

"""
Module contains the updated functions and classes for assessing
parasite-induced host mortality (PIHM)

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

    def fit_nbd(self, k_array=np.linspace(0.01, 20, 100)):
        """
        Fit a NBD to the parasite data
        """

        return mod.nbinom.fit_mle(self.data, k_array=k_array)


    def crofton_method(self, crof_bin_edges, guess=None, bounded=False,
        upper=[2, 2, 5]):
        """
        Crofton's method for estimating the pre-mortality parameters N_p
        (population size before mortality),


        mu_p (mean parasite load before mortality), and k_p (parasite
        aggregation before mortality).  Uses the iterative technique proposed
        by Lester (1977), which minimizes the chi-squared statistic

        Parameters
        ----------
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

        bounded : bool
            If True, uses a bounded minimizer.  Else uses an unbounded
        upper : list
            A list of multipliers/upper bounds. The first multiplies the N upper
            bound, the second item multiplies the mu upper bound, and the third
            item specifies the k upper bound.
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

        if not bounded:

            full_out = opt.fmin(opt_fxn, np.array([N_guess, mu_guess, k_guess]),
                                            disp=0)

            opt_params = full_out

        else:

            bounds = [(0, upper[0]*N_guess), (0, upper[1]*mu_guess),
                                (0.001, upper[2])]
            full_out = opt.fmin_l_bfgs_b(opt_fxn,
                            np.array([N_guess, mu_guess, k_guess]),
                            bounds=bounds, approx_grad=True, disp=0)
            opt_params = full_out[0]


        # Calculate predicted in bins
        # pred = [opt_params[0] * np.sum(mod.nbinom.pmf(s, opt_params[1],
        #     opt_params[2])) for s in splits]

        self.set_premort_params(*opt_params)

        return self.get_premort_params()


    def obs_pred(self, full_bin_edges):
        """
        Returns the observed and predicted vectors for the preset values of
        Np, mup, and kp.  Will throw an error in Np, mup and kp are not defined.
        You can either define these manually or run the crofton_method first.
        Note that the predicted values are the predicted values WITHOUT mortality

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
                    run_crof=False, test_sig=False):
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
        test_sig : bool
            If True, returns an additional tuple with the null_deviance, the
            deviance, the chi-squared value,  and the p-value of full model.
            A p-value of less than 0.05 (or alpha) indicates that PIHM is
            significant.

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

        # Note that if Np is too low and mup is too high the Adjei method
        # Won't work because the probability is too disperesed!  There won't be
        # a complete individual in any of the classes so you can't fit the
        # binomial. Need to try different bin sizes

        split_edges = split_data(full_bin_edges)


        theor_binned = np.array([np.round(np.sum(mod.nbinom.pmf(s, self.mup, self.kp)
                        * self.Np), decimals=0) for s in split_edges])

        # Group the theoretical data by bin edges

        if np.all(theor_binned == 0):
            raise AttributeError("With Np = %.2f and mup = %.2f,the are"  % (self.Np, self.mup) +
                                  " no\ncomplete individuals in the given " +
                                  "bins. Try different bin sizes")

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
        params, null_deviance, deviance = fit_glm(all_data)
        self.a, self.b = params
        a = params[0]
        b = params[1]

        if test_sig:
            return (self.get_all_params(), (null_deviance, deviance))

        else:
            return self.get_all_params()


    def likelihood_method(self, full_fit=True, max_sum=1000, continuous=False,
                guess=[10, -5], disp=True):
        """
        Using likelihood method to estimate parameter of truncated NBD and survival
        fxn

        Parameters
        ----------
        data : array-like
            Observed host parasite data
        full_fit : bool
            If True, fits mup, kp, a, b. If False, used preset mup and kp
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
        When dealing with small sample sizes and trying to fit the full
        distribution (i.e., not using the Crofton Method), the convergence of
        the likelihood method will often fail or multiple likelihood peaks
        might exist.  One way around this is to use the approach of Ferguson
        and fix k and then just try to estimate


        """

        if not full_fit:

            if None in list(self.get_premort_params())[1:]:
                raise TypeError("mup, kp must be preset")

            params = opt.fmin(likefxn2, guess, args=(self.data, self.mup,
                        self.kp), disp=disp, maxiter=10000, maxfun=10000)
            out_params = [self.mup, self.kp] + list(params)

        else:

            mu_guess = np.mean(self.data)
            k_guess = 1
            out_params = opt.fmin(likefxn1, [mu_guess, k_guess] + guess,
                        args=(self.data,), disp=disp, maxiter=10000, maxfun=10000)

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

    def test_for_pihm_w_adjei(self, full_bin_edges, crof_bin_edges,
                        no_bins=True, run_crof=False):
        """

        Test a dataset for parasite induced host mortality using the likelihood
        method

        This method compares a reduced model (negative binomial distribution)
        to a full model (Negative binomail with PIHM).  The two models are
        nested and differ by two parameters: a and b.  This amounts to fitting
        the model a negative binomial distribution to the data, then fitting
        the full model to the data and comparing likelihoods using a likelihood
        ratio test.  The likelihood ratio should be approximately chi-squared
        with the dof equal to the difference in the parameters.

        """

        params, (null_deviance, deviance) = self.adjei_method(full_bin_edges, crof_bin_edges,
                            no_bins=no_bins, run_crof=run_crof, test_sig=True)

        chi_sq = null_deviance - deviance
        prob = chisqprob(chi_sq, 1)

        return chi_sq, prob, null_deviance, deviance, params


    def test_for_pihm_w_likelihood(self, guess=[10, -5],
                        k_array=np.linspace(0.05, 2, 100),
                        fixed_pre=False, disp=True):
        """
        Test a dataset for parasite induced host mortality using the likelihood
        method

        This method compares a reduced model (negative binomial distribution)
        to a full model (Negative binomail with PIHM).  The two models are
        nested and differ by two parameters: a and b.  This amounts to fitting
        the model a negative binomial distribution to the data, then fitting
        the full model to the data and comparing likelihoods using a likelihood
        ratio test.  The likelihood ratio should be approximately chi-squared
        with the dof equal to the difference in the parameters.

        Parameters
        ----------
        guess : list
            Guesses for a and b
        k_array : array
            Array of values over which to search to best fit k
        fixed_pre : bool
            If True, the premortality parameters are fixed (mup and kp). Else,
            they are jointly estimated from the data
        disp : bool
            If True, convergence message is printed.  If False, no convergence
            method is printed


        Returns
        -------
        : chi_squared valued, p-value, full nll, reduced nll
        """

        # No params are known
        if not fixed_pre:

            # Get full nll
            params = self.likelihood_method(full_fit=True, guess=guess,
                        disp=disp)
            full_nll = likefxn1(params, self.data)

            mle_fit = mod.nbinom.fit_mle(self.data, k_array=k_array)

            red_nll = comp.nll(self.data, mod.nbinom(*mle_fit))

        # Params are known
        else:

            params = self.likelihood_method(full_fit=False, guess=guess,
                        disp=disp)
            full_nll = likefxn2(params[2:], self.data, self.mup, self.kp)

            red_nll = comp.nll(self.data, mod.nbinom(self.mup, self.kp))

        # Approximately chi-squared...though this is a large sample size approx
        chi_sq = 2 * (-full_nll - (-red_nll))
        prob = chisqprob(chi_sq, 2)

        return chi_sq, prob, full_nll, red_nll, params


def pihm_pmf(x, mup, kp, a, b, max_sum=5000):
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
    return(kern(x) / sum(kern(np.arange(0, max_sum))))

def likefxn_def(mu, k, a, b, x):
    """ Likelihood fxn for all parameters for pihm pmf
    Returns negative log-likelihood

    """

    return -np.sum(np.log(pihm_pmf(x, mu, k, a, b)))



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


def extract_simulation_results(sim_results, keys, method_name, param,
                                alpha=0.05):
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
        either "a", "b", "ld50" or "p".  If the param is p, then the type I
        and type II errors are extracted.  This is only applicable for
        results from the type I and type II error simulations

    alpha : float
        The significance level.  Only used if param == "p"

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
            # If the param is p, a_vals are type I p_vals
            # and b_vals are type II p_vals
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

        elif param == "p":

            # Type I error
            typeIp, typeIb = zip(*a_vals)
            powerp, powerb = zip(*b_vals)

            biases.append(np.sum(np.array(typeIp) < alpha) / len(typeIp))

            # Power
            precisions.append(np.sum(np.array(powerp) < alpha) / len(powerp))

        else:
            raise KeyError("Don't recognize parameter: should be a, b, or ld50")

    return samp_sizes, biases, precisions

def extract_full_likelihood_simulation(sim_results, keys, param,
                                alpha=0.05):
    """

    Method to extract parameters from the simulation dicationary from the full
    likelihood simulation

    Parameters
    ----------
    sim_results : dict
        A dictionary with analysis results

    keys : list
        First item is a value for mup, second item is a tuple specify the
        (a, b) pair, third iterm is kp.

    param: str
        either "ld50" or "p".  If the param is p, then the type I error is
        extracted. If the parameter is ld50 than the bias and precision for the
        ld50 estimates are extracted.

    alpha : float
        The significance level.  Only used if param == "p"

    Returns
    : tuple
        samp_sizes, results

        Results could contain either type I errors of a tuple (bias, precision)
        depending on the value of param


    """

    sims = sim_results[keys[0]][keys[1]][keys[2]]

    Np_vals = list(sims.viewkeys())

    samp_sizes = []
    biases = []
    precisions = []

    for Np in Np_vals:

        res = sims[Np]

        p_vals, ab_vals = zip(*res[0])
        a_vals, b_vals  = zip(*ab_vals)

        N_vals = res[1]

        samp_sizes.append(np.mean(N_vals))

        if param == "ld50":

            ld50_vals = np.exp(np.array(a_vals) / np.abs(b_vals))
            truth = np.exp(keys[1][0] / np.abs(keys[1][1]))

            biases.append(scaled_bias(ld50_vals, truth))
            precisions.append(scaled_precision(ld50_vals))

        elif param == "p":

            powerp = p_vals

            # Power
            precisions.append(np.sum(np.array(powerp) < alpha) / len(powerp))
            biases.append(np.nan)


        else:
            raise KeyError("Don't recognize parameter: should be p or ld50")

    return samp_sizes, biases, precisions


def split_data(bin_edges):
    """
    Given bin edges, split data in fully enumerate bins. Example: [1, 4, 8.1]
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

    all_data['const'] = 1

    glm_fit = sm.GLM(all_data[['emp', 'diff']],
                    all_data[['const', 'log_para']],
                    family=sm.families.Binomial())
    res = glm_fit.fit()

    return res.params, res.null_deviance, res.deviance

def scaled_bias(data, truth):
    """
    Aboluste value of scaled bias calculation from Walther and Moore 2005
    """

    return np.abs(np.sum(data - truth) / (len(data) * float(truth)))


def scaled_precision(data):
    """
    Just the Coefficient of Variation
    """

    return np.abs(100 * np.std(data, ddof=1) / np.mean(data))

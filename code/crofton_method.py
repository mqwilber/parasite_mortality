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
Implement Crofton method for estimating parasite-induced mortality
"""

def nbd_fixed_mean(data, mean):
    """ MLE estimation of NBD with a fixed mean """

    loglike = lambda k: -np.sum(mod.nbinom.logpmf(data, mean, k))
    return opt.fmin(loglike, 1, disp=0)[0]


def crofton_method(data, bin_edges, guess=None):
    """
    Crofton's method for estimating parasite induced mortality

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

    return opt_params, obs, pred, splits, np.sum((obs - pred)**2 / pred)


def w2_method(data, full_bin_edges, crof_bin_edges, guess=(10, -2),
                no_bins=False, crof_params=None):
    """

    This method first uses the Crofton method to find the best N, mu and k and
    then uses our own method to find a and b.  This seems to work much better
    than the adjei method and it makes a lot less assumptions!

    Parameters
    ----------
    data : array-like
        Full data
    full_bin_edges : array-like
        Bin edges for the full data.  The bins should over the full data
    crof_bin_edges : array-like
        The bin edges for crofton.  Should only include the data range that
        does not have parasite induced mortality
    guess : tuple
        Initial guess for a and b respectively
    no_bins : bool
        If True, over writes full_bin_edges with np.arange(0, max(data) + 2)
        This is useful when mean parasites per host is relatively low.  When
        mean parasites per host is high, use pre-defined bins.
    crof_params : tuple
        Must be a tuple of (N, mu, k).  Parameters of the population before
        mortality.

    Return
    ------
    : tuple

    Notes
    ------
    Decided to minimize the chisquared statistic because that is what
    historically is done.  np.abs works well too

    """

    # First use the crofton method

    if crof_params:
        N, mu, k = crof_params
    else:
        crof_res = crofton_method(data, crof_bin_edges)
        N, mu, k = crof_res[0]

    if no_bins:
        full_bin_edges = np.arange(0, np.max(data) + 2)

    obs = np.histogram(data, bins=full_bin_edges)[0]

    splits = []
    for i, b in enumerate(full_bin_edges):

        if i != len(full_bin_edges) - 1:
            splits.append(np.arange(full_bin_edges[i], full_bin_edges[i + 1]))

    # Fix these parameters
    def opt_fxn(params):

        a, b = params
        pred = np.array([N * np.sum(mod.nbinom.pmf(s, mu, k) *
                    surv_prob(s, a, b)) for s in splits])

        #pred = N * pred / np.sum(pred)
        #return np.sum(np.abs(obs - pred))
        return np.sum((obs - pred)**2 / pred)

    # A bounded search seems to work better.  Though there are still problems
    opt_params = opt.fmin_l_bfgs_b(opt_fxn, np.array(guess),
               bounds=[(0, 100), (-30, 0)], approx_grad=True)[0]
    #opt_params = opt.fmin(opt_fxn, np.array(guess))
    # opt_params = opt.brute(opt_fxn, ((0, 30), (-30, 0)), Ns=20)
    a, b = opt_params

    pred = [N * np.sum(mod.nbinom.pmf(s, mu, k) *
            surv_prob(s, a, b)) for s in splits]

    return tuple([N, mu, k] + list(opt_params)), obs, pred, splits


def likelihood_method(data, crof_params=None, max_sum=1000, guess=[10, -5]):
    """
    Using likelihood method to estimate parameter of truncated NBD and survival
    fxn

    Parameters
    ----------
    data : array-like
        Observed host parasite data
    crof_params : tuple or None
        If tuple contains (N_p, mu_p, k_p): Pre-mortality abundance,
        pre-mortality mean parasites per host, pre-mortality k.  If None will
        to estimate all of the parameters from the data.
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
    # kern = lambda x, mu, k, a, b: surv_prob(x, a, b) * \
    #                                     mod.nbinom.pmf(x, mu, k)
    # pmf = lambda x, mu, k, a, b: kern(x, mu, k, a, b) / \
    #                 sum(kern(np.arange(0, max_sum), mu, k, a, b))

    # def likefxn1(params, x):
    #     """ Likelihood fxn for all parameters """

    #     mu, k, a, b = params
    #     return -np.sum(np.log(pmf(x, mu, k, a, b)))

    # def likefxn2(params, x, mu, k):
    #     """ Likelihood fxn for just a and b """

    #     a, b = params
    #     return -np.sum(np.log(pmf(x, mu, k, a, b)))

    if crof_params:
        N, mu, k = crof_params

        params = opt.fmin(likefxn2, guess, args=(data, mu, k), disp=1)
        out_params = [mu, k] + list(params)

    else:

        mu_guess = np.mean(data)
        k_guess = 1
        out_params = opt.fmin(likefxn1, [mu_guess, k_guess] + guess,
                    args=(data,))

    return tuple(out_params)

# PMF for host mortality
kern = lambda x, mu, k, a, b: surv_prob(x, a, b) * \
                                    mod.nbinom.pmf(x, mu, k)
pmf = lambda x, mu, k, a, b: kern(x, mu, k, a, b) / \
                sum(kern(np.arange(0, 1000), mu, k, a, b))

def likefxn1(params, x):
    """ Likelihood fxn for all parameters """

    mu, k, a, b = params
    return -np.sum(np.log(pmf(x, mu, k, a, b)))

def likefxn2(params, x, mu, k):
    """ Likelihood fxn for just a and b """

    a, b = params
    return -np.sum(np.log(pmf(x, mu, k, a, b)))


def adjei_fitting_method(data, full_bin_edges, crof_bin_edges, no_bins=True,
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


def fit_glm(all_data):
    """
    Trying out the GLM in statsmodels
    """

    all_data = all_data[['emp', 'pred', 'para']]
    all_data['diff'] = np.array(all_data.pred) - np.array(all_data.emp)
    all_data['log_para'] = np.log(all_data['para'])
    new_data = sm.add_constant(all_data)
    glm_fit = sm.GLM(new_data[['emp', 'diff']],
                    new_data[['const', 'log_para']], family=sm.families.Binomial())
    res = glm_fit.fit()

    return res.params


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


def get_pred(N, mu, k, bin_edges):
    """
    Function for getting predicted parasites in bins given just N, mu, and k
    """

    splits = []
    for i, b in enumerate(bin_edges):

        if i != len(bin_edges) - 1:
            splits.append(np.arange(bin_edges[i], bin_edges[i + 1]))

    pred = [N * np.sum(mod.nbinom.pmf(s, mu, k)) for s in splits]
    return pred, splits


def get_alive_and_dead(num_hosts, a, b, k, mu, other_death=False, percent=0.1):
    """
    Simulate hosts and get alive and get alive and dead

    Parameters
    ----------
    num_hosts : int
        Number of pre-mortality hosts
    a : float
        a parameter of host survival function
    b : float
        b parameter of host survival function
    k : float
        Clumping parameter of pre-mortality hosts
    mu : float
        Mean of pre-mortality hosts
    other_death : bool
        True if random death occurs. False is parasite mortality is the only
        death
    percent : float
        Between 0 and 1.  The random mortality percentage

    Return
    ------
    : tuple


    """

    init_pop = mod.nbinom.rvs(mu, k, size=num_hosts)

    survival_probs = surv_prob(init_pop, a, b)
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

    true_death = 1 - len(alive_hosts) / float(len(init_pop))



    return alive_hosts, init_pop, true_death






def scaled_bias(data, truth):
    """
    Scaled bias calculation from Walther and Moore 2005
    """

    return np.abs(np.sum(data - truth) / (len(data) * float(truth)))


def scaled_precision(data):
    """
    Just the Coefficient of Variation
    """

    return 100 * np.std(data, ddof=1) / np.mean(data)

def percent_dead(a, b, mu, k, max_num=1000):
    """
    Calculate percent of population killed by parasites

    """

    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    vals = np.arange(0, max_num + 1)
    prob_dead_res = np.empty(len(a))

    for i in xrange(len(a)):
        prob_dead_res[i] =  1 - np.sum(surv_prob(vals, a[i], b[i]) *
                                mod.nbinom(mu, k).pmf(vals))
    return prob_dead_res


def simulate_over_np(N_vals, mu, k, a, b, SAMP, crof_params=True,
                fit_bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                crof_bins=[0, 1, 2, 3, 4, 4.9]):
    """

    Function for simulating dataset and testing it against the W2 method and
    the Adjei Method

    Parameters
    -----------
    N_vals : list
        List of values of N_p to use for the simulatino (pre-mortality
        population size)
    mu : float
        Mean of the pre-mortality NBD
    k : float
        k of the pre-mortality NBD
    a : float
        a parameter of the survival function
    b : float
        b parameter of the survival function
    SAMP : int
        Number of samples for each test
    crof_params : bool
        If True, uses known crof params.  If False, uses the crofton method
        to estimate the parameters

    Returns
    --------
    : tuple
        First element is a list of all W2, Adjei, and likelihood results.
        Second element is the average Ns over the samples

    """

    plot_vals_full_sim = []
    N_avgs_full_sim = []

    for N in N_vals:

        print("Starting N = %i" % N)
        adjei_params = []  # Store parameters from adjei method
        w2_params = []  # Store parameters from w2 method
        like_params = []  # Store parameters from likelihood method

        Ns = []
        for i in xrange(SAMP):

            alive, initial, td = get_alive_and_dead(N, a, b, k, mu)
            Ns.append(len(alive))

            if not crof_params:
                params = tuple(crofton_method(alive, crof_bins)[0])

            # Try First alternative method
            try:
                if crof_params:
                    tw2 = w2_method(alive,
                        fit_bins + [np.max(alive) + 0.9], crof_bins,
                        no_bins=False, crof_params=(N, mu, k))[0][-2:]
                else:
                    tw2 = w2_method(alive,
                        fit_bins + [np.max(alive) + 0.9], crof_bins,
                        no_bins=False, crof_params=params)[0][-2:]

                w2_convg = True
            except:
                w2_convg = False

            # Try adjei method
            try:
                if crof_params:
                    tadjei = adjei_fitting_method(alive, [],
                                    crof_bins, no_bins=True,
                                    crof_params=(N, mu, k))[0][-2:]
                else:
                    tadjei = adjei_fitting_method(alive, [],
                                        crof_bins, no_bins=True,
                                      crof_params=params)[0][-2:]

                adjei_convg = True
            except:
                adjei_convg = False

            try:
                if crof_params:
                    tlike = likelihood_method(alive,
                                    crof_params=(N, mu, k), guess=[a, b])[-2:]
                else:
                    tlike = likelihood_method(alive,
                                      crof_params=params, guess=[a, b])[-2:]
                like_convg = True
            except:
                like_convg = False

            # Only add if all of them converged
            if np.all([w2_convg, adjei_convg, like_convg]):

                w2_params.append(tw2)
                adjei_params.append(tadjei)
                like_params.append(tlike)

        N_avgs_full_sim.append(np.mean(Ns))
        plot_vals_full_sim.append(((N, a, b, k, mu), w2_params, adjei_params,
                            like_params))

    return plot_vals_full_sim, N_avgs_full_sim


def extract_parameters(plot_vals_full_sim, a, b, mu=None, k=None):
    """
    Extract parameters from simulations and calcuate bias and precision

    Parameters
    ----------
    plot_vals_full_sim : list
        Results of simulation outputted from simulation_over_np
    a : float
        a parameter of survival function
    b : float
        b parameter of survival function
    mu : float
        mean of the pre-mortality negative binomial distribution
    k : float
        aggregation of the pre-mortality negative binomial distribution

    Returns
    -------
    : tuple
        Tuple has 7 objects. 1) Bias and Precision for W2 a 2) Bias and Precision for W2 b
        3) Bias and Precision for Adjei a 4) Bias and Precision for Adjei b 5) Bias and precision
        for likelihood method a 6) Bias and precision for likelihood method b 7) Bias for
        the LD50 (w2, adjei, like)

    """

    # Get scaled bias and precision for estimates
    estimates_w2_a = []
    estimates_w2_b = []

    estimates_adjei_a = []
    estimates_adjei_b = []

    estimates_like_a = []
    estimates_like_b = []

    ld50_extract = []
    pd_extract = []

    # pred_pd = percent_dead(a, b, mu, k)

    for i in xrange(len(plot_vals_full_sim)):

        w2_a, w2_b = zip(*plot_vals_full_sim[i][1])
        adjei_a, adjei_b = zip(*plot_vals_full_sim[i][2])
        like_a, like_b = zip(*plot_vals_full_sim[i][3])

        # Drop the fits that didn't converge...also try not dropping: W2
        ind_b = ~np.bitwise_or(np.array(w2_b) == -30, np.array(w2_b) >= -0.1)

        # Drop for adjei
        ind_b_adjei = np.bitwise_or(np.array(adjei_b) <= -100,
                        np.array(adjei_b) >= -0.1)

        # Drop for like
        ind_b_like = np.bitwise_or(np.array(like_b) <= -100,
                        np.array(like_b) >= -0.1)

        #full_ind = np.bitwise_or(ind_b, ind_b_adjei)

        w2_a = np.array(w2_a)[ind_b]
        w2_b = np.array(w2_b)[ind_b]
        adjei_a = np.array(adjei_a)[~ind_b_adjei]
        adjei_b = np.array(adjei_b)[~ind_b_adjei]
        like_a = np.array(like_a)[~ind_b_like]
        like_b = np.array(like_b)[~ind_b_like]

        w2_a_bias = scaled_bias(w2_a, a)
        w2_a_prec = scaled_precision(w2_a)
        w2_b_bias = scaled_bias(w2_b, b)
        w2_b_prec = scaled_precision(w2_b)
        w2_ld50_bias = scaled_bias(np.exp(w2_a / np.abs(w2_b)),
                                  np.exp(a / np.abs(b)))

        # w2_pd_bias = scaled_bias(percent_dead(w2_a, w2_b, mu, k), pred_pd)

        adjei_a_bias = scaled_bias(adjei_a, a)
        adjei_a_prec = scaled_precision(adjei_a)
        adjei_b_bais = scaled_bias(adjei_b, b)
        adjei_b_prec = scaled_precision(adjei_b)
        adjei_ld50_bias = scaled_bias(np.exp(adjei_a / np.abs(adjei_b)),
                                    np.exp(a / np.abs(b)))
        # adjei_pd_bias = scaled_bias(percent_dead(adjei_a, adjei_b, mu, k),
        #                                 pred_pd)

        like_a_bias = scaled_bias(like_a, a)
        like_a_prec = scaled_precision(like_a)
        like_b_bais = scaled_bias(like_b, b)
        like_b_prec = scaled_precision(like_b)
        like_ld50_bias = scaled_bias(np.exp(like_a / np.abs(like_b)),
                                    np.exp(a / np.abs(b)))

        # like_pd_bias = scaled_bias(percent_dead(like_a, like_b, mu, k),
        #                                 pred_pd)

        estimates_w2_a.append((w2_a_bias, w2_a_prec))
        estimates_w2_b.append((w2_b_bias, w2_b_prec))

        estimates_adjei_a.append((adjei_a_bias, adjei_a_prec))
        estimates_adjei_b.append((adjei_b_bais, adjei_b_prec))

        estimates_like_a.append((like_a_bias, like_a_prec))
        estimates_like_b.append((like_b_bais, like_b_prec))

        ld50_extract.append((w2_ld50_bias, adjei_ld50_bias, like_ld50_bias))
        #pd_extract.append((w2_pd_bias, adjei_pd_bias, like_pd_bias))

    return (estimates_w2_a, estimates_w2_b,
                estimates_adjei_a, estimates_adjei_b,
                estimates_like_a, estimates_like_b, ld50_extract)

# Define plotting function
def plot_bias(sims, a_b_vec, file_name, N_vals, ylims=(-2, 0.5),
                ks=[1, 1, 0.5, 0.5], figsize=(12, 8), ld50=False, index=6):
    """
    Function plots the bais of the a parameter from the thre different PIHM
    methods

    Parameters
    ----------
    sims : list of length 3
        Each element is a list containing three different ratios of LD50
    a_b_vec : list of tuples
        Length of three (a, b) values. Different LD50 values
    file_name : str
        Name of the file to which to save
    ld50: bool
        If True, plots the LD50 bias.  If False, plots a and b bias

    """

    method = ["Chi-squared Method", "Likelihood Method", "Chi-squared Method",
             "Likelihood Method",  "Chi-squared Method", "Likelihood Method"]
    colors = ["black", "grey", "#B8B8B8"]
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.ravel()
    for ax in axes:
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
    alpha = 1
    lwd = 2

    for i, sim in enumerate(sims[0]):

        if not ld50:

            w2_a_bias_vec, w2_a_prec_vec = zip(*sim[0])
            w2_b_bias_vec, w2_b_prec_vec = zip(*sim[1])

            adjei_a_bias_vec, adjei_a_prec_vec = zip(*sim[2])
            adjei_b_bias_vec, adjei_b_prec_vec = zip(*sim[3])

            like_a_bias_vec, like_a_prec_vec = zip(*sim[4])
            like_b_bias_vec, like_b_prec_vec = zip(*sim[5])

        else:

            w2_a_bias_vec, adjei_a_bias_vec, like_a_bias_vec = zip(*sim[index])



        axes[0].semilogx(N_vals, adjei_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
                                     label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)

        axes[1].semilogx(N_vals, adjei_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
                                      label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)
#         axes[1].semilogx(N_vals, adjei_b_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
#                                      label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)
#         axes[0].semilogx(N_vals, adjei_b_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
#                                      label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)


        axes[0].semilogx(N_vals, w2_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls="-",
                                     label=r"Chi. Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)
        axes[1].semilogx(N_vals, like_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls="--" ,
                                     label=r"Like. Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)

    for i, sim in enumerate(sims[1]):

        if not ld50:

            w2_a_bias_vec, w2_a_prec_vec = zip(*sim[0])
            w2_b_bias_vec, w2_b_prec_vec = zip(*sim[1])

            adjei_a_bias_vec, adjei_a_prec_vec = zip(*sim[2])
            adjei_b_bias_vec, adjei_b_prec_vec = zip(*sim[3])

            like_a_bias_vec, like_a_prec_vec = zip(*sim[4])
            like_b_bias_vec, like_b_prec_vec = zip(*sim[5])

        else:

            w2_a_bias_vec, adjei_a_bias_vec, like_a_bias_vec = zip(*sim[index])

        axes[2].semilogx(N_vals, adjei_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
                                     label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)

        axes[3].semilogx(N_vals, adjei_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
                                     label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)
#         axes[2].semilogx(N_vals, adjei_b_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
#                                      label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)
#         axes[3].semilogx(N_vals, adjei_b_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
#                                      label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)


        axes[2].semilogx(N_vals, w2_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls="-",
                                     label=r"Chi. Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)
        axes[3].semilogx(N_vals, like_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls="--" ,
                                     label=r"Like. Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)

    for i, sim in enumerate(sims[2]):

        if not ld50:

            w2_a_bias_vec, w2_a_prec_vec = zip(*sim[0])
            w2_b_bias_vec, w2_b_prec_vec = zip(*sim[1])

            adjei_a_bias_vec, adjei_a_prec_vec = zip(*sim[2])
            adjei_b_bias_vec, adjei_b_prec_vec = zip(*sim[3])

            like_a_bias_vec, like_a_prec_vec = zip(*sim[4])
            like_b_bias_vec, like_b_prec_vec = zip(*sim[5])

        else:

            w2_a_bias_vec, adjei_a_bias_vec, like_a_bias_vec = zip(*sim[index])

        axes[4].semilogx(N_vals, adjei_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
                                     label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)

        axes[5].semilogx(N_vals, adjei_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
                                     label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)
#         axes[2].semilogx(N_vals, adjei_b_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
#                                      label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)
#         axes[3].semilogx(N_vals, adjei_b_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls=":",
#                                      label=r"Adjei Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)


        axes[4].semilogx(N_vals, w2_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls="-",
                                     label=r"Chi. Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)
        axes[5].semilogx(N_vals, like_a_bias_vec, lw=lwd, alpha=alpha, color=colors[i], ls="--" ,
                                     label=r"Like. Method $a = %.1f$, $b = %.1f$" % a_b_vec[i], basex=10)

    for i, ax in enumerate(axes):

        ax.text(0.5, 0.95, method[i] + r", $k_p = %.2f$" % ks[i], horizontalalignment="center", transform=ax.transAxes)
        ax.set_ylabel("Standardized Bias")
        ax.set_xlabel(r"$\log{N_p}$")
        ax.set_ylim(ylims)
        ax.set_xlim((10**2, 10**4.5))
        ax.hlines(0, 10**2, 10**4.5, linestyle="--", color="red", alpha=0.5)
        ax.legend(loc="lower right", prop={'size' : 6})

    plt.tight_layout()
    fig.savefig("../results/" + file_name, dpi=300)


def test_for_pihm(data, guess=[10, -5], crof_params=None):
    """
    Test a dataset for parasite induced host mortality

    Parameters
    ----------
    data : array
        Hosts with a given parasite loads

    Returns
    -------
    :
    """

    # Get full nll
    params = likelihood_method(data, crof_params=crof_params, guess=guess)
    full_nll = likefxn1(params, data)


    if crof_params:
        mu, k = np.array(crof_params)[1:]
        red_nll = comp.nll(data, mod.nbinom(mu, k))

    else:
        mle_fit = mod.nbinom.fit_mle(data, k_array=np.linspace(.1, 2, 100))
        red_nll = comp.nll(data, mod.nbinom(*mle_fit))

    chi_sq = 2 * (-full_nll - (-red_nll))
    prob = chisqprob(chi_sq, 2)
    return chi_sq, prob, full_nll, red_nll





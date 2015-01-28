import numpy as np
import macroeco.models as mod
import scipy.optimize as opt
import pandas as pd
import statsmodels.api as sm
import pymc as pm


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


def w1_method(data, bin_edges, fix_params=None, guess=(10, 2)):
    """

    Improved fitting method using abs(obs - pred).  Tries to fit N, mu, k, a
    and b simultanesouly.  fmin has a bit of trouble with this one because
    there seem to be multiple parameter combinations that lead to the same
    minimum.  Fixing some parameters can help.

    """

    obs = np.histogram(data, bins=bin_edges)[0]

    splits = []
    for i, b in enumerate(bin_edges):

        if i != len(bin_edges) - 1:
            splits.append(np.arange(bin_edges[i], bin_edges[i + 1]))

    # Fix these parameters
    N_guess = len(data)
    mu_guess = np.mean(data)
    _, k_guess = mod.nbinom.fit_mle(data)

    guess_vec = np.array([N_guess, mu_guess, k_guess, guess[0], guess[1]])

    def opt_fxn(params):

        N, mu, k, a, b = params
        pred = np.array([N * np.sum(mod.nbinom.pmf(s, mu, k) *
                    surv_prob(s, a, b)) for s in splits])
        # Renormalize
        # pred = N * pred / np.sum(pred)
        return np.sum(np.abs(obs - pred))

    opt_params = opt.fmin(opt_fxn, guess_vec, xtol=0.001, ftol=0.001)
    N, mu, k, a, b = opt_params

    pred = [N * np.sum(mod.nbinom.pmf(s, mu, k) *
            surv_prob(s, a, b)) for s in splits]

    return opt_params, obs, pred, splits, np.sum(np.abs(obs - pred))


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


def w3_method(data, full_bin_edges, crof_bin_edges, guess=(10, -2)):
    """

    This method first uses the Crofton method to find the best N, mu and k and
    then uses our own method to find a and b.  Similar to method to but,
    minimizes the log-likelihood of the data rather than np.abs(obs - pred)

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

    Return
    ------
    : tuple


    """

    crof_res = crofton_method(data, crof_bin_edges)
    N, mu, k = crof_res[0]

    obs = np.histogram(data, bins=full_bin_edges)[0]

    splits = []
    for i, b in enumerate(full_bin_edges):

        if i != len(full_bin_edges) - 1:
            splits.append(np.arange(full_bin_edges[i], full_bin_edges[i + 1]))

    # Fix these parameters

    def opt_fxn(params):

        a, b = params
        p_vec = np.array([np.sum(mod.nbinom.pmf(s, mu, k) *
                    surv_prob(s, a, b)) for s in splits])
        p_vec = p_vec / np.sum(p_vec)

        return -pm.multinomial_like(obs, N, p_vec)

    opt_params = opt.fmin(opt_fxn, np.array(guess), xtol=0.001, ftol=0.001)
    a, b = opt_params

    pred = [N * np.sum(mod.nbinom.pmf(s, mu, k) *
            surv_prob(s, a, b)) for s in splits]

    return opt_params, obs, pred, splits, np.sum(np.abs(obs - pred))


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


def infected_lost(params, all_data, full_para=True):
    """
    Use formula from Adjei et al to calculate percent infected lost.

    Parameters
    ----------
    params : array-like
        Length 2. First value is a parameter of GLM and second parameter is
        b parameter of the GLM
    all_data : dataframe
        DataFrame returned from adjei_fitting_method.
        Columns 'emp', 'pred', 'para'
    full_para : bool
        If the column para includes the zero class
    """

    a = params[0]
    b = params[1]
    surv_prob = lambda x, a, b: np.exp(a + b * np.log(x)) / \
                                    (1 + np.exp(a + b * np.log(x)))
    pred = np.array(all_data['pred'])
    para = np.array(all_data['para'])

    # Proportion infected lost
    if full_para:

        surv_vals = surv_prob(para[1:], a, b)
        inf_lost = 1 - np.sum(surv_vals * pred[1:]) / np.sum(pred[1:])

        numer = pred[0] + np.sum(surv_vals * pred[1:])
        denom = np.sum(pred)
        tot_lost = 1 - numer / denom
        return inf_lost, tot_lost

    else:
        return (1 - np.sum(surv_prob(para, a, b) * pred) / np.sum(pred), None)


def simulate_model(num_hosts, a, b, k, mu, percent=0.5):
    """
    Simulation of model
    """

    alive_hosts, init_pop, true_death = \
                    get_alive_and_dead(num_hosts, a, b, k, mu, percent=0.5)

    LD50 = np.exp(a / np.abs(b))
    bin_edges = np.arange(0, percent*LD50)
    all_data, params, (N, mu, k) = \
                adjei_fitting_method(alive_hosts, bin_edges, [], no_bins=True)

    inf_lost, tot_lost = infected_lost(params, all_data, full_para=True)

    return params, bin_edges, true_death, tot_lost, (N, mu, k), alive_hosts, init_pop


def surv_prob(x, a, b):
    """
    Calculate survival probability given a parasite load x
    """
    x = np.atleast_1d(x)

    probs = np.empty(len(x))

    for i, val in enumerate(x):
        if val == 0:
            probs[i] = 1
        else:
            probs[i] = np.exp(a + b * np.log(val)) / (1 + np.exp(a + b * np.log(val)))

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


def get_alive_and_dead(num_hosts, a, b, k, mu, percent=0.5):
    """
    Simulate hosts and get alive and get alive and dead
    """

    init_pop = mod.nbinom.rvs(mu, k, size=num_hosts)

    survival_probs = surv_prob(init_pop, a, b)
    survival_probs[np.isnan(survival_probs)] = 1

    # Determine stochastic survival
    rands = np.random.rand(len(survival_probs))
    surv = rands < survival_probs
    alive_hosts = init_pop[surv]

    true_death = 1 - np.sum(surv) / float(len(surv))

    return alive_hosts, init_pop, true_death


# def w2_method(data, bin_edges, fix_params=None, guess=(10, 2)):
#     """

#     Possible fitting method using full maximum likelihood.  Using the
#     the chi-squared statistic.  Minimize chi-squared

#     """

#     obs = np.histogram(data, bins=bin_edges)[0]


#     splits = []
#     for i, b in enumerate(bin_edges):

#         if i != len(bin_edges) - 1:
#             splits.append(np.arange(bin_edges[i], bin_edges[i + 1]))

#     N_guess = len(data)
#     mu = np.mean(data)
#     _, k_guess = mod.nbinom.fit_mle(data)
#     guess_vec = np.array([N_guess, k_guess, guess[0], guess[1]])

#     # minimize the log-likelihood
#     def opt_fxn(params):

#         N, k, a, b = params

#         p_vec =  [np.sum(mod.nbinom.pmf(s, mu, k)) *
#                             surv_prob(np.mean(s), a, b) for s in splits[:-1]]
#         p_vec.append(1 - np.sum(p_vec))

#         return -pm.multinomial_like(obs, N, p_vec)

#     opt_params = opt.fmin(opt_fxn, guess_vec, xtol=0.01, ftol=0.01)

#     N, k, a, b = opt_params

#     #pred = [N * np.sum(mod.nbinom.pmf(s, mu, k)) *
#     #        surv_prob(np.mean(s), a, b) for s in splits]

#     pred = np.array([N * np.sum(mod.nbinom.pmf(s, mu, k) *
#                     surv_prob(s, a, b)) for s in splits])

#     return opt_params, obs, pred, splits


def full_params_method(data, full_bin_edges, crof_bin_edges, guess=(10, -2),
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

    Return
    ------
    : tuple

    Notes
    ------
    Decided to minimize the chisquared statistic because that is what
    historically is done.  np.abs works well too

    """

    # First use the crofton method

    # if crof_params:
    #     N, mu, k = crof_params
    # else:
    #     crof_res = crofton_method(data, crof_bin_edges)
    #     N, mu, k = crof_res[0]

    if no_bins:
        full_bin_edges = np.arange(0, np.max(data) + 2)

    obs = np.histogram(data, bins=full_bin_edges)[0]

    splits = []
    for i, b in enumerate(full_bin_edges):

        if i != len(full_bin_edges) - 1:
            splits.append(np.arange(full_bin_edges[i], full_bin_edges[i + 1]))

    # Fix these parameters
    def opt_fxn(params):

        N, k, mu, a, b = params
        pred = np.array([N * np.sum(mod.nbinom.pmf(s, mu, k) *
                    surv_prob(s, a, b)) for s in splits])

        return np.sum((obs - pred)**2 / pred)

    # A bounded search seems to work better.  Though there are still problems
    # opt_params = opt.fmin_l_bfgs_b(opt_fxn, np.array(guess),
    #             bounds=[(0, 100), (-30, 0)], approx_grad=True)[0]
    guess = [len(data), np.mean(data), 1, 10, -2]
    opt_params = opt.fmin(opt_fxn, np.array(guess))
    # opt_params = opt.brute(opt_fxn, ((0, 30), (-30, 0)), Ns=20)
    N, k, mu, a, b = opt_params

    pred = [N * p.sum(mod.nbinom.pmf(s, mu, k) *
            surv_prob(s, a, b)) for s in splits]

    return [N, mu, k] + list(opt_params), obs, pred, splits


def scaled_bias(data, truth):
    """
    Scaled bias calculation from Walther and Moore 2005
    """

    return np.sum(data - truth) / (len(data) * float(truth))


def scaled_precision(data):
    """
    Just the Coefficient of Variation
    """

    return 100 * np.std(data, ddof=1) / np.mean(data)

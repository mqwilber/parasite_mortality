import pihm_methods as pm
reload(pm)
import numpy as np
import pandas as pd

"""
Description
-----------

Full simulation for type I and type II error section of the parasite-induced
host mortality manuscript.

This script looks at all combinations of parameters that I am using in the
manuscript and performs 150 simulations for each parameter combination.
Looking at three levels of premortality parameters

The results are saved to a pickled dictionary in the results section this project
Repo.

See the document ../doc/parasite_mortality_manuscript.tex for a more thorough
description of the methods.

Author : Mark Wilber


"""


# Parameters
ks = [1, 0.5, .1]

mu_ld50 = {10: ([(5, -2.5), (10, -5), (20, -10)], []),
            100: ([(5, -1.15), (10, -2.3), (20, -4.6)], []),
            500: ([(10, -1.8), (20, -3.6), (60, -10.8)], [])}

Nps = [300, 500, 1000, 2000, 5000, 7500, 10000]
N_samp = 150

# ks = [1]

# mu_ld50 = {10: ([(5, -2.5), (10, -5), (20, -10)], [])}

#             # 100: ([(5, -1.15), (10, -2.3), (20, -4.6)], []),
#             # 500: ([(10, -1.8), (20, -3.6), (60, -10.8)], [])}

# Nps = [300, 500, 1000, 2000, 5000, 7500, 10000]
# N_samp = 1

scenario_1_results = {}

pihm = pm.PIHM()

for mup in mu_ld50.iterkeys():

  scenario_1_results[mup] = {}

  for ld50 in mu_ld50[mup][0]:

    scenario_1_results[mup][ld50] = {}

    for kp in ks:

      scenario_1_results[mup][ld50][kp] = {}

      for Np in Nps:

        # Print to see progress
        print((mup, ld50, kp, Np))

        results_like = []
        results_adjei = []
        samp_sizes = []

        # Set the parameters

        for i in xrange(N_samp):

          pihm.set_all_params(Np, mup, kp, ld50[0], ld50[1])

          data_post, data_pre = pihm.get_pihm_samples()

          samp_sizes.append(len(data_post))

          try:

            # Test on pre-mort
            pihm.data = data_pre

            # Likelihood method
            like_p_typeI = pihm.test_for_pihm_w_likelihood(fixed_pre=True,
                                  disp=False)[1]

            # Adjei results
            adjei_p_typeI = pihm.test_for_pihm_w_adjei([], [], no_bins=True)[1]

            # Test on post-mort
            pihm.data = data_post

            # Likelihood method
            like_p_typeII = pihm.test_for_pihm_w_likelihood(fixed_pre=True,
                                disp=False)[1]

            # Adjei results
            adjei_p_typeII = pihm.test_for_pihm_w_adjei([], [], no_bins=True)[1]

            results_like.append((like_p_typeI, like_p_typeII))
            results_adjei.append((adjei_p_typeI, adjei_p_typeII))

          except:
            pass

        scenario_1_results[mup][ld50][kp][Np] = \
                        (results_like, results_adjei, samp_sizes)


pd.to_pickle(scenario_1_results, "../results/typeIandII_analysis_results.pkl")

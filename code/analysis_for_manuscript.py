import pihm_methods as pm
reload(pm)
import numpy as np
import configparser
import pandas as pd

"""
Description
-----------

Full simulation for the manuscript

MORE

"""


# Parameters
ks = [1, 0.5, .1]
# mu_ld50 = {10: ([(5, -2.5), (10, -5), (20, -10)], []),
#            100: ([(5, -1.15), (10, -2.3), (20, -4.6)], []),
mu_ld50 =  {500: ([(10, -1.8), (20, -3.6), (60, -10.8)], [])}
Nps = [300, 500, 1000, 2000, 5000, 7500, 10000]
N_samp = 150

# ks = [1]
# mu_ld50 = {10: ([(20, -10)], [0, 1, 2, 3, 4, 4.9])}
# Nps = [5000]
# N_samp = 50


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
          pihm.data = pihm.get_pihm_samples()[0]

          samp_sizes.append(len(pihm.data))

          try:

            # Likelihood results
            like_out = pihm.likelihood_method(full_fit=False, disp=False)[-2:]

            # Adjei results
            adjei_out = pihm.adjei_method([], [], no_bins=True,
                                                run_crof=False)[-2:]

            results_like.append(like_out)
            results_adjei.append(adjei_out)

          except:
            pass

        scenario_1_results[mup][ld50][kp][Np] = \
                        (results_like, results_adjei, samp_sizes)


pd.to_pickle(scenario_1_results, "../results/scenario1_analysis_results2.pkl")

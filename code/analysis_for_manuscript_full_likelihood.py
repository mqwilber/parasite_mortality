import pihm_methods as pm
reload(pm)
import numpy as np
import pandas as pd

"""
Description
-----------



Author : Mark Wilber


"""


# Parameters
ks = [1]

mu_ld50 = {10: ([(5, -2.5), (10, -5), (20, -10)], [])}

Nps = np.round(np.logspace(1.8, 3.5, 10, base=10)).astype(np.int)#[50, 100, 200, 300, 400, 500] #[300, 500, 1000, 2000, 5000, 7500, 10000]
N_samp = 300

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
        samp_sizes = []

        # Set the parameters

        for i in xrange(N_samp):

          pihm.set_all_params(Np, mup, kp, ld50[0], ld50[1])

          data_post, data_pre = pihm.get_pihm_samples()

          samp_sizes.append(len(data_post))

          # Test on post_mort
          pihm.data = data_post

          # Likelihood method. Don't fix the premortatliy
          like_res = pihm.test_for_pihm_w_likelihood(fixed_pre=False, disp=False)

          # Pull out p-val and a and b
          results_like.append((like_res[1], like_res[-1][-2:]))

        scenario_1_results[mup][ld50][kp][Np] = (results_like, samp_sizes)


# This saves the likelihood results with different means
#pd.to_pickle(scenario_1_results, "../results/full_likelihood_analysis_results.pkl")

# This saves the likelihood results with different means
pd.to_pickle(scenario_1_results, "../results/full_likelihood_analysis_results_just_10.pkl")

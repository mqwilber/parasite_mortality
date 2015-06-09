import crofton_method as cm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
reload(cm)

"""
Description
------------
This script compares the Adjei method and our methods using a different set of

For 500, and 1000 individuals in the community before mortality

Prelim: Our method does better for (40, -12), (20, -6) they are about the same
(10, -3) with our method doing a little better.

"""

# Simulate data


params1 = {}  # Parameters from pure method with bins
params2 = {}  # Parameters from pure method without bins
params3 = {}  # Parameters from Adjei method with bins
params4 = {}  # Parameters from Adjei method without bins

# Parameters

SIMS = 500

# Same LD50, but different regions of the curve
sim_vals1 = [(20, -6), (10, -3), (5, -1.5)]
num_hosts = [500, 1000]
ks = [1, 0.5]
mu = 20

for svals in sim_vals1:

    print("Running with parameter %.2f, %.2f" % (svals[0], svals[1]))

    params1[svals] = {}
    params2[svals] = {}
    params3[svals] = {}
    params4[svals] = {}

    for num in num_hosts:

        print("Running with num_hosts %i" % num)

        params1[svals][num] = {}
        params2[svals][num] = {}
        params3[svals][num] = {}
        params4[svals][num] = {}

        for k in ks:

            print("Running with k %i" % k)

            params1[svals][num][k] = []
            params2[svals][num][k] = []
            params3[svals][num][k] = []
            params4[svals][num][k] = []

            for i in xrange(SIMS):

                print("Simulation %i" % i)
                alive, full, true_death = cm.get_alive_and_dead(num,
                                            svals[0], svals[1], k, mu)

                bin_edges = [0, 1, 6, 11, 16, 21, 31, np.max(alive) + 1]
                crof_bins = [0, 1, 6, 11, 16, 21]

                try:
                    out1 = cm.w2_method(alive, bin_edges, crof_bins,
                                                guess=(svals[0], svals[1]))
                except:
                    out1 = [(np.nan, np.nan)]

                try:
                    out2 = cm.w2_method(alive, bin_edges, crof_bins, no_bins=True,
                                        guess=(svals[0], svals[1]))
                except:
                    out2 = [(np.nan, np.nan)]

                try:

                    out3 = cm.adjei_fitting_method(alive, crof_bins, bin_edges,
                                no_bins=False)
                except:
                    out3 = [(np.nan, np.nan)]

                try:

                    out4 = cm.adjei_fitting_method(alive, crof_bins, bin_edges,
                                        no_bins=True)
                except:
                    out4 = [(np.nan, np.nan)]

                params1[svals][num][k].append(tuple(out1[0]))
                params2[svals][num][k].append(tuple(out2[0]))
                params3[svals][num][k].append(tuple(out3[0]))
                params4[svals][num][k].append(tuple(out4[0]))

results_sim1 = (params1, params2, params3, params4)
pd.to_pickle(results_sim1, "../results/simulation2_parameters.pkl")


import crofton_method as cm
import numpy as np
import matplotlib.pyplot as plt
reload(cm)

# alive, full, true_death = cm.get_alive_and_dead(1000, 20, -6, 1, 20)

# bin_edges = [0, 1, 6, 11, 16, 21, 26, np.max(alive) + 1]
# crof_bins = [0, 1, 6, 11, 21]

alive, full, true_death = cm.get_alive_and_dead(1000, 5, -1.5, 1, 28)

bin_edges = [0, 1, 2, 3, np.max(alive) + 1]
crof_bins = [0, 1, 2, 3, 6, 11, 21]

# bin_edges = [0, 1, 2, 3, 4, 5, 8, np.max(alive) + 1]
# crof_bins = [0, 1, 2, 3, 5.1]

#out1 = cm.w1_method(alive, bin_edges, guess=(5, -0.5))
out2 = cm.w2_method(alive, bin_edges, crof_bins, guess=(10, -1), no_bins=False)
out3 = cm.w2_method(alive, bin_edges, crof_bins, guess=(10, -1), no_bins=True)
#out3 = cm.w3_method(alive, bin_edges, crof_bins, guess=(10, -2))
#crof = cm.crofton_method(alive, bin_edges)
adjei = cm.adjei_fitting_method(alive, crof_bins, [], no_bins=True)
adjei_no_bins = cm.adjei_fitting_method(alive, crof_bins, bin_edges,
                no_bins=False)

print(out2[0], out3[0], adjei[0], adjei_no_bins[0])

# plt.hist(full, bins=bin_edges)
# plt.hist(alive, bins=bin_edges, alpha=0.5)
# plt.show()
#out2 = cm.w2_method(alive, bin_edges, guess=(8, -0.5))

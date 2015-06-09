import numpy as np
import pihm_methods as pm

"""
Description
------------
This script compares the validity of our crofton method to those given in
Crofton 1971.  Our implementation of the crofton method gives equivalent
answers to the answers obtained by Crofton in his 1971 paper.

Run the script and you can compare the k values, the Ns and the chi-squared
values minimized for the first three stations in the crofton data.

"""

chi_sq = lambda pred, obs: np.sum((obs - pred)**2 / pred)

para_crof = np.arange(0, 9)

# Data from Crofton 1971
st1_obs = np.array([161, 111, 67, 65, 50, 30, 33, 13, 8])
st2_obs = np.array([189, 129, 86, 51, 27, 14, 8, 1, 2])
st3_obs = np.array([458, 81, 40, 22, 19, 4, 6, 3, 0])
st4_obs = np.array([164, 147, 92, 43, 25, 11, 3, 0, 1])
st5_obs = np.array([140, 77, 30, 14, 10, 3, 2, 0, 0])
st6_obs = np.array([153, 29, 6, 2, 1, 0, 0, 0, 0])

# Raw data
st1_raw = np.repeat(para_crof, st1_obs)
st2_raw = np.repeat(para_crof, st2_obs)
st3_raw = np.repeat(para_crof, st3_obs)
st4_raw = np.repeat(para_crof, st4_obs)
st5_raw = np.repeat(para_crof, st5_obs)
st6_raw = np.repeat(para_crof, st6_obs)

# Crofton predicted with truncations!
st1_pred = {
    3: np.array([162.2, 103.9, 77.4, 60.4, 48.2, 38.9, 31.7, 26.0, 21.4]),
    4: np.array([162.4, 103.4, 77.6, 61.2, 49.4, 40.5, 33.5, 27.9, 23.3]),
    5: np.array([161.2, 107.0, 78.0, 58.7, 44.7, 34.4, 26.6, 20.7, 16.1]),
    6: np.array([162.1, 104.6, 77.3, 59.5, 46.7, 37.1, 29.7, 23.9, 19.3]),
    7: np.array([160.2, 108.8, 79.2, 58.9, 44.3, 33.6, 25.5, 19.5, 14.9]),
    8: np.array([158.4, 111.7, 80.8, 59.0, 43.2, 31.7, 23.3, 17.2, 12.7])}

st2_pred = {
    3: np.array([188.6, 130.7, 83.6 , 52.0, 31.9, 19.4, 11.7, 7.0, 4.2]),
    4: np.array([187.9, 133.0, 83.0, 49.4, 28.7, 16.5, 9.3, 5.3, 2.9,]),
    5: np.array([187.3, 134.5, 83.0, 48.4, 27.4, 15.3, 8.4, 4.6, 2.5]),
    6: np.array([187.2, 134.8, 83.1, 48.3, 27.3, 15.1, 8.3, 4.5, 2.4]),
    7: np.array([185.3, 138.1, 84.2, 47.5, 25.7, 13.6, 7.0, 3.6, 1.8]),
    8: np.array([185.5, 137.9, 84.1, 47.5, 25.8, 13.7, 7.1, 3.7, 1.9])}

st3_pred = {
    3: np.array([457.9, 81.8, 39.0, 22.5, 14.1, 9.2, 6.3, 4.3, 3.0]),
    4: np.array([458.1, 79.5, 40.3, 25.0, 16.9, 12.0, 8.8, 6.5, 5.0]),
    5: np.array([457.5, 83.7, 39.0, 21.9, 13.3, 8.5, 5.5, 3.7, 2.5]),
    6: np.array([457.0, 83.4, 39.1, 22.0, 13.5, 8.6, 5.7, 3.8, 2.2]),
    7: np.array([457.4, 84.1, 39.1, 21.8, 13.2, 8.3, 5.4, 3.6, 2.4])}

# Station 1 comparison

trunc = np.arange(3, 9)

crof_st1_ks = [0.754, 0.736, 0.834, 0.776, 0.874, 0.951]
crof_st1_N = [677, 708, 606, 646, 594, 573]
our_st1_N = []
our_st1_ks = []
crof_st1_compare_ss = []

for i in trunc:

    bin_edges = list(np.arange(0, i + 1)) + [i + 0.9]
    pm_obj = pm.PIHM(st1_raw)



    N, mu, k = pm_obj.crofton_method(bin_edges)
    our_st1_ks.append(k)
    our_st1_N.append(N)

    our_pred = pm_obj.obs_pred(np.arange(0, 10))['predicted']
    crof_ss = chi_sq(st1_pred[i], st1_obs)
    our_ss = chi_sq(our_pred, st1_obs)
    crof_st1_compare_ss.append((crof_ss, our_ss))

# Station 2 comparison

crof_st2_ks = [1.182, 1.313, 1.392, 1.407, 1.574, 1.563]
crof_st2_N = [535, 520, 514, 513, 508, 509]
our_st2_N = []
our_st2_ks = []
crof_st2_compare_ss = []

for i in trunc:

    bin_edges = list(np.arange(0, i + 1)) + [i + 0.9]
    pm_obj = pm.PIHM(st2_raw)

    N, mu, k = pm_obj.crofton_method(bin_edges)
    our_st2_ks.append(k)
    our_st2_N.append(N)

    our_pred = pm_obj.obs_pred(np.arange(0, 10))['predicted']
    crof_ss = chi_sq(st2_pred[i], st2_obs)
    our_ss = chi_sq(our_pred, st2_obs)
    crof_st2_compare_ss.append((crof_ss, our_ss))

# Station 3 comparison

trunc = np.arange(3, 8)

crof_st3_ks = [.229, .206, .244, .242, .247]
crof_st3_N = [646, 670, 642, 643, 641]
our_st3_N = []
our_st3_ks = []
crof_st3_compare_ss = []

for i in trunc:

    bin_edges = list(np.arange(0, i + 1)) + [i + 0.9]
    pm_obj = pm.PIHM(st3_raw)

    N, mu, k = pm_obj.crofton_method(bin_edges)
    our_st3_ks.append(k)
    our_st3_N.append(N)

    our_pred = pm_obj.obs_pred(np.arange(0, 10))['predicted']
    crof_ss = chi_sq(st3_pred[i], st3_obs)
    our_ss = chi_sq(our_pred, st3_obs)
    crof_st3_compare_ss.append((crof_ss, our_ss))

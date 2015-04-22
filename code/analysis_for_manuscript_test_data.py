import pihm_methods as pm
reload(pm)
import numpy as np
import pandas as pd

"""
Description
-----------

Testing the Adjei Method and the Likelihood Method against known data

A couple different tests

1. Run Crofton Method and then test for PIHM (Adjei and Likelihood)
2. For Likelihood, fit with all parameters

"""

### Crofton Data for 6 different stations ###

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

all_crof_data = [st1_raw, st2_raw, st3_raw, st4_raw, st5_raw, st6_raw]

bins = [0, 1, 2, 3, 4, 4.9]

crof = pm.PIHM()

test_like_full_crof = []
test_like_full_ld50_crof = []
test_like_crof = []
test_like_ld50_crof = []
test_adjei_crof = []
test_adjei_ld50_crof = []

for raw in all_crof_data:

    crof.data = raw
    crof.crofton_method(bins)
    test_like_crof.append(crof.test_for_pihm_w_likelihood(fixed_pre=True)[1])

    # Calculate the LD50 for the like
    res1 = crof.likelihood_method(full_fit=False)
    test_like_ld50_crof.append(np.exp(res1[-2] / np.abs(res1[-1])))

    test_adjei_crof.append(crof.test_for_pihm_w_adjei([], [])[1])

    # Calculate the LD50 with the adjei method
    res2 = crof.adjei_method([], [])
    test_adjei_ld50_crof.append(np.exp(res2[-2] / np.abs(res2[-1])))


    test_like_full_crof.append(crof.test_for_pihm_w_likelihood(fixed_pre=False)[1])

    res3 = crof.likelihood_method(full_fit=True)
    test_like_full_ld50_crof.append(np.exp(res3[-2] / np.abs(res3[-1])))

### Adjei Data : 2 species of fish, both male and female ###

# Specify the Adjei Method Data
st_females = np.repeat((0, 1, 2, 3, 4, 5, 6, 7),
                        (201, 114, 63, 37, 19, 5, 3, 4))
st_males = np.repeat((0, 1, 2, 3, 4, 5), (226, 128, 62, 30, 3, 3))

su_females = np.repeat(np.arange(0, 8), (2311, 180, 66, 8, 5, 2, 0, 1))
su_males = np.repeat((0, 1, 2, 3, 4), (2257, 146, 29, 7, 1))

all_adjei_data = [st_females, st_males, su_females, su_males]

crof = pm.PIHM()

test_like_full_adjei = []
test_like_full_ld50_adjei = []
test_like_adjei = []
test_like_ld50_adjei = []
test_adjei_adjei = []
test_adjei_ld50_adjei = []

### Test for st fish

# Fit females

crof.data = st_females

crof.crofton_method([0, 1, 2, 2.9])
test_like_adjei.append(crof.test_for_pihm_w_likelihood(fixed_pre=True)[1])

res1 =  crof.likelihood_method(full_fit=False)
test_like_ld50_adjei.append(np.exp(res1[-2] / np.abs(res1[-1])))


test_adjei_adjei.append(crof.test_for_pihm_w_adjei([], [])[1])

res2 = crof.adjei_method([], [])
test_adjei_ld50_adjei.append(np.exp(res2[-2] / np.abs(res2[-1])))


# Fit males with female parameters

crof.data = st_males

# Use the parameters already set for the females
test_like_adjei.append(crof.test_for_pihm_w_likelihood(fixed_pre=True)[1])
test_adjei_adjei.append(crof.test_for_pihm_w_adjei([], [])[1])

res3 =  crof.likelihood_method(full_fit=False)
test_like_ld50_adjei.append(np.exp(res3[-2] / np.abs(res3[-1])))

res4 = crof.adjei_method([], [])
test_adjei_ld50_adjei.append(np.exp(res4[-2] / np.abs(res4[-1])))


# Full likelihood for females
crof.data = st_females
test_like_full_adjei.append(crof.test_for_pihm_w_likelihood(fixed_pre=False)[1])

res5 = crof.likelihood_method(full_fit=True)
test_like_full_ld50_adjei.append(np.exp(res5[-2] / np.abs(res5[-1])))

# Fit full likelihood for males using female parameters
crof.data = st_males
test_like_full_adjei.append(crof.test_for_pihm_w_likelihood(fixed_pre=True)[1])

res6 = crof.likelihood_method(full_fit=False)
test_like_full_ld50_adjei.append(np.exp(res6[-2] / np.abs(res6[-1])))

### Test for su fish

# Fit females

crof.data = su_females
crof.crofton_method([0, 1, 2, 2.9])
test_like_adjei.append(crof.test_for_pihm_w_likelihood(fixed_pre=True)[1])

res1 =  crof.likelihood_method(full_fit=False)
test_like_ld50_adjei.append(np.exp(res1[-2] / np.abs(res1[-1])))


test_adjei_adjei.append(crof.test_for_pihm_w_adjei([], [])[1])

res2 = crof.adjei_method([], [])
test_adjei_ld50_adjei.append(np.exp(res2[-2] / np.abs(res2[-1])))


# Fit males with female parameters

crof.data = su_males
test_like_adjei.append(crof.test_for_pihm_w_likelihood(fixed_pre=True)[1])
test_adjei_adjei.append(crof.test_for_pihm_w_adjei([], [])[1])

res3 =  crof.likelihood_method(full_fit=False)
test_like_ld50_adjei.append(np.exp(res3[-2] / np.abs(res3[-1])))

res4 = crof.adjei_method([], [])
test_adjei_ld50_adjei.append(np.exp(res4[-2] / np.abs(res4[-1])))

# Full likelihood for females
crof.data = su_females
test_like_full_adjei.append(crof.test_for_pihm_w_likelihood(fixed_pre=False, guess=[5, -5])[1])

res5 = crof.likelihood_method(full_fit=True, guess=[5, -5])
test_like_full_ld50_adjei.append(np.exp(res5[-2] / np.abs(res5[-1])))

# Fit males with female parameters
crof.data = su_males
test_like_full_adjei.append(crof.test_for_pihm_w_likelihood(fixed_pre=True)[1])

res6 = crof.likelihood_method(full_fit=False)
test_like_full_ld50_adjei.append(np.exp(res6[-2] / np.abs(res6[-1])))







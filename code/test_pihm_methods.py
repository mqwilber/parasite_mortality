from __future__ import division

from numpy.testing import (TestCase, assert_equal, assert_array_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_allclose, assert_, assert_raises)

import numpy as np
from pihm_methods import *
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import macroeco.models as mod

class TestPIHM(TestCase):

    def test_crofton_method(self):
        pass

    def test_obs_pred(self):
        pass

    def test_adjei_method(self):
        """
        Description
        -----------

        Testing the the Adjei method that we coded gives the same answer as the Adjei
        method in the book.  We know that the Crofton Method works pretty well.

        These results show that, as far as I can tell, our implementation of the Adjei
        method is returning pretty similar results to what Adjei is showing for St
        males, SU males and SU females.  We are getting very different answers with ST
        females and looking at their answers, it doesn't seem like they should be
        getting what they are getting.  They calculate the LD50 for the ST females to be
        around 5.7

        """

        # Specify the Adjei Method Data
        st_females = np.repeat((0, 1, 2, 3, 4, 5, 6, 7),
                                (201, 114, 63, 37, 19, 5, 3, 4))
        st_males = np.repeat((0, 1, 2, 3, 4, 5), (226, 128, 62, 30, 3, 3))

        su_females = np.repeat(np.arange(0, 8), (2311, 180, 66, 8, 5, 2, 0, 1))
        su_males = np.repeat((0, 1, 2, 3, 4), (2257, 146, 29, 7, 1))


        # Fit the adjei method
        st_fem = PIHM(st_females)
        st_fem_fit = st_fem.adjei_method([], [0, 1, 2, 2.9], no_bins=True,
                                run_crof=True)
        # Fit males using females
        st_male = PIHM(st_males)
        st_male.set_premort_params(st_fem_fit[0], st_fem_fit[1], st_fem_fit[2])
        st_male_fit = st_male.adjei_method([], [], no_bins=True, run_crof=False)

        # Fit the females
        su_fem = PIHM(su_females)
        su_fem_fit = su_fem.adjei_method([], [0, 1, 2, 2.9], no_bins=True,
                            run_crof=True)

        su_male = PIHM(su_males)
        su_male.set_premort_params(su_fem_fit[0], su_fem_fit[1], su_fem_fit[2])
        su_male_fit = su_male.adjei_method([], [], no_bins=True, run_crof=False)

        fits = [st_fem_fit, st_male_fit, su_fem_fit, su_male_fit]
        ld50 = lambda x: np.exp(x[3] / np.abs(x[4]))

        pred_ld50 = np.round([ld50(f) for f in fits], decimals=1)
        adjei_ld50 = [5.7, 3.4, 3.2, 1.8]

        print(zip(pred_ld50, adjei_ld50))

    def test_likelihood_method(self):

        tobj = PIHM()
        tobj.set_all_params(30000, 10, 1, 20, -10)

        alive, init = tobj.get_pihm_samples()

        tobj.data = alive
        tobj.likelihood_method()

        rounded = np.round(tobj.get_all_params(), decimals=0)

        # Won't always pass but should be close
        assert_array_equal(np.array([20, -10]), rounded[3:])

    def test_get_pihm_samples(self):

        # Test random sample works
        tobj = PIHM()
        tobj.set_all_params(1000, 10, 1, 10, -5)

        alive, init = tobj.get_pihm_samples()

        assert_equal(len(init), 1000)

        # Test that setting with likelihood method works
        rand = mod.nbinom(10, 1).rvs(1000)
        tobj = PIHM(rand)
        tobj.likelihood_method()

        assert_raises(TypeError, tobj.get_pihm_samples)

        tobj.Np = 1000
        alive, init = tobj.get_pihm_samples()
        assert_equal(len(init), 1000)




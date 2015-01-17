import crofton_method as cm
import numpy as np

"""
Description
-----------

Testing the the Adjei method that we coded gives the same answer as the Adjei
method in the book.  We know that the Crofton Method works pretty well.

These results show that, as far as I can tell, our implementation of the Adjei
method is returning pretty similar results to what Adjei is showing for St
males, SU males and SU females.  We are getting very different answers with ST
females and looking at their answers, it doesn't seem like they should be
getting what they are getting.  The calculate the LD50 for the ST females to be
around 5.7

"""

# Looking at Crofton Data
st_females = np.repeat((0, 1, 2, 3, 4, 5, 6, 7),
                        (201, 114, 63, 37, 19, 5, 3, 4))
st_males = np.repeat((0, 1, 2, 3, 4, 5), (226, 128, 62, 30, 3, 3))

su_females = np.repeat(np.arange(0, 8), (2311, 180, 66, 8, 5, 2, 0, 1))
su_males = np.repeat((0, 1, 2, 3, 4), (2257, 146, 29, 7, 1))

st_fem_fit = cm.adjei_fitting_method(st_females, [], [0, 1, 2, 2.9],
    no_bins=True)
st_male_fit = cm.adjei_fitting_method(st_males, [], [], no_bins=True,
                crof_params=tuple(np.array(st_fem_fit[0])[0:3]))

su_fem_fit = cm.adjei_fitting_method(su_females, [], [0, 1, 2, 2.9],
    no_bins=True)
su_male_fit = cm.adjei_fitting_method(su_males, [], [], no_bins=True,
                crof_params=tuple(np.array(su_fem_fit[0])[0:3]))

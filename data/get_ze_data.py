import pandas as pd
import numpy as np


"""
Script for converting Andrea Jani's enzootic Bd data to something with which
we can estimate mortality.

"""

bd_data = pd.read_csv("../../nsf_sloan_model/data/formatted/OTUdata_formatted.csv")

# Subset by swab
swabs = bd_data[bd_data['sample type']  == 'swab']

# Bd Site
bd_site = swabs[swabs.bd_status == 1]

# Only look at enzootic sites
ind = ~((bd_site["Site Name"] == 'Marmot Lake') | (bd_site["Site Name"] ==
    "Dusy Basin"))
enzootic = bd_site[ind]

# Only adults
adults = enzootic[enzootic['Life Stage'] == 'adult']

ze_data = adults[["ZE1"]]
ze_data.to_csv("ze_adults_enzootic.csv", index=False)



import pandas as pd
import numpy as np


"""
Joining all columns of raccoon data
"""

rac_data = pd.read_csv("raccoon_data10_24.csv")

# Glue columns together and drop nas
agg_rac_data = []
for col in rac_data.columns:
    agg_rac_data.append(np.array(rac_data[col].dropna()))

full_data = np.concatenate(agg_rac_data)

full_df = pd.DataFrame({'worms' : full_data})
full_df.to_csv("all_raccoon_data.csv", index=False)



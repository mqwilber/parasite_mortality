# Repo containing parasite analyses

`docs/`: Contains the latex manuscript and supplementary materials

`code/`: Contains all the code necessary for exactly reproducing the 
analyses and figures in the manuscript. The work horse behind all of the analysis is the module `pihm_methods.py` which provides the functions for implementing the Crofton Method, Adjei Method, and the Likelihood Method. 

    - All of the analyses in the manuscript can be re-run

    - To run the simulations simply type the following commands at the command line

    - `python analysis_for_manuscript_full_likelihood.py`

    - `python analysis_for_manuscript_hostsurvivalfxn.py`

    - `python analysis_for_manuscript_typeIandII.py`

    - All figures in the plot can be remade by opening `manuscript_plots.ipynb` in the ipython notebook and running all the cells

    - The results for table 2 are generated by `analysis_for_manuscript_test_data.py`

`results/`: Directory where the analysis results are stored

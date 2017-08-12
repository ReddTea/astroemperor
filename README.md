# EMPEROR
EMPEROR Exoplanet Mcmc Parallel tEmpering Radial velOcity fitteR

# Brief Description

EMPEROR (Exoplanet Mcmc Parallel tEmpering Radial velOcity fitteR), is a new Python-based algorithm that automatically searches for signals in radial velocity timeseries, employing Markov chains and parallel tempering methods, convergence 
tests and Bayesian statistics, along with various noise models.  A number of posterior sampling routines are available, focused on efficiently searching for signals in highly multi-modal posteriors.  The code allows the analysis of multi-instrument and multi-planet data sets and performs model comparisons automatically to return the optimum model that best describes the data.

# Dependencies
This code makes use of:
  - Numpy
  - Scipy
  - emcee (http://dan.iel.fm/emcee/current/)
  - PyAstronomy (http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/index.html)
  - pygame (https://www.pygame.org/)
  - tqdm (https://pypi.python.org/pypi/tqdm)
  - termcolor (https://pypi.python.org/pypi/termcolor)

All of them can be easily installed with pip.

# Installation
Soon to be pip installable! Pardon us for the 'manual installation' and read below!

# Easy Setup
Download folder and run test_emperor.py to make sure everything works!

# Usage
The code is really simple to use! You just need to have your data under /datafiles folder.
And then, as shown in test_emperor.py, append the names of the datasets you want to use in an array, set the configuration of the chain in another single array, then call the method and conquer!

# Example
```sh
import scipy as sp
import emperor

stardat = sp.array(['starname_1_telescopename1.vels', 'starname_2_telescopename2.vels'])
setup = sp.array([5, 300, 12000])  # temperatures, walkers, steps

DT = emperor.EMPIRE(stardat, setup)  # EMPIRE(data_to_read, chain_parameters)
DT.conquer(0, 5)
```

# Outputs
They go under /datalogs folder. Following the name convention for the datasets, you should have them properly clasiffied as /datalogs/starname/<date_i>, where i is the ith run in that date.

You will see chain plots, posterior plots, histograms, phasefolded curve, the chain sample and more!!! 


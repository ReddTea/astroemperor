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
DT.conquer(0, 5)  # run from 0 to 5 signals !
```

# Outputs
They go under /datalogs folder. Following the name convention for the datasets, you should have them properly clasiffied as /datalogs/starname/<date_i>, where i is the ith run in that date.

You will see chain plots, posterior plots, histograms, phasefolded curves, the chain sample and more!!! 

# Why EMPEROR?

  -It's really simple to use
  -It has a series of configuration commands that will amaze you
  -Advanced Noise Model
  -Quite Flexible!
  
# List of Commands
Assuming  you have already set up the chain:
```sh
DT = emperor.EMPIRE(stardat, setup)  # EMPIRE(data_to_read, chain_parameters)
```
You have:
| Command | Action |
| ------ | ------ |
| Setup | ------ |
| ------ | ------ |
| DT.cores | changes the number of cores you are using, default is maximum! **Let's up the tempo.** |
| DT.burn_out | to configure the steps for the burn out phase. Default is half the steps for the chainlength. |
| DT.CONSTRAIN | This constrain the search according to the results of the previous analyzed model. Default is True. (should be always True) |
| DT.thin | Thins the chain. Default is 1. |
| ------ | ------ |
| Statistical Tools | ------ |
| ------ | ------ |
| DT.bayes_factor | changes the in-chain factor, default is sp.log(150) |
| DT.model_comparison | Changes the posterior comparison between models with k signals, when this doesn't comply emperor stops running, default is 5 |
| DT.BIC | This is the BIC used to compare models. Default is 5. |
| DT.AIC | This is the AIC used to compare models. Default is 5. |
| ------ | ------ |
| Model | ------ |
| ------ | ------ |
| DT.MOAV | Sets the Moving Average Order for the rednoise model. Default is 1. (can be 0) |
| DT.eccprior | Sets the sigma for the Prior Normal Distribution for eccentricity. Default is 0.3 |
| DT.jittprior | Sets the sigma for the Prior Normal Distribution for jitter. Default is 5.0 |
| DT.jittmean | Sets the mu for the Prior Normal Distribution for jitter. Default is 5.0 as well |
| DT.STARMASS | Outputs the Minimum Mass and Semi-Major Axis. Should be put in solar masses. |
| DT.HILL | Enables the fact that the Hill Stability Criteria has to comply as a prior |
| ------ | ------ |
| Easter | ------ |
| DT.MUSIC | True/False. Short sounds that tell you what is emperor doing! So you can work on other things while it's ready. |
| ------ | ------ |
| Plotting | ------ |
| DT.PLOT | Enables Plotting. Default is True |
| DT.CORNER | Enables Corner plot. Default is True |
| DT.HISTOGRAMS | Enables Beautiful Histograms. Default is True |
| DT.PNG | Enables PNG plots. Default is True |
| DT.PDF | Enables PDF plots. Default is False |
| DT.draw_every_n | draws 1 every n points in the plots, without thining the chain for the statistics. Default is 1 |
| ------ | ------ |



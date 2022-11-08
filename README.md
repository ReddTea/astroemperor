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
  - corner (https://pypi.python.org/pypi/corner)
  - reddutils (https://github.com/ReddTea/reddutils)
  - tabulate (https://pypi.org/project/tabulate/)
  - kepler (https://github.com/dfm/kepler.py)
  - arviz (https://arviz-devs.github.io/arviz/)

All of them can be easily installed with pip.

# Installation

## Pip
Just try out
```sh
pip3 install astroEMPEROR
```

## Manual installation
For the 'manual installation' read below!!

In the console type in your work folder
```sh
git clone https://github.com/ReddTea/astroEMPEROR.git
```


# Easy Setup
Download the tests folder and run python_file.py to make sure everything works!

```sh
ipython  # open python environment
run python_file
```
or just
```sh
python python_file.py
```


# Usage
The code is really simple to use! You just need to have your data under /datafiles folder.
And then, as shown in python_file.py, append the names of the datasets you want to use in an array, set the configuration of the chain in another single array, then call the method and conquer!

# Example
```sh
import astroEMPEROR as emp
import numpy as np

sim = emp.Simulation()
sim._set_engine__('emcee')
setup = np.array([3, 50, 500])
sim._data__('GJ876')  # Target folder name in /datafiles/
sim._run_auto__(setup, 2, param=0, acc=1, moav=0)

sim.save_chain([0])  # to save more chains, [0, 1, 2, ...]
sim.save_posteriors([0])

```

# Outputs
They go under /datalogs folder. Following the name convention for the datasets, you should have them properly classified as /datalogs/starname/<date_i>, where i is the ith run in that date.

You will see chain plots, posterior plots, histograms, phasefolded curves, the chain sample and more!!!

# Why EMPEROR?

  - It's really simple to use
  - It has a series of configuration commands that will amaze you
  - Advanced Noise Model
  - Quite Flexible!

# List of Commands
I'll update this soon, promise.


| Command           | Action                                                                                                                                                 | Input Type  | Default                                        |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|------------------------------------------------|
| Setup             |                                                                                                                                                        |             |                                                |
| cores             | Sets the number of threads you are using.                                                                                                              | Int         | Default is maximum! **Let's up the tempo.**    |
| burn_out          | The steps for the burn-in phase.                                                                                                                       | Int         | Default is half the steps for the chainlength. |
| CONSTRAIN         | Constrains the search according to the results of the previous analyzed model.                                                                         | Boolean     | Default is True. (should be always True)       |
| thin              | Thins the chain.                                                                                                                                       | Int         | 1                                              |
| betas             | Sets the beta factor for each temperature.                                                                                                             | Array       | None                                           |
| Statistical Tools |                                                                                                                                                        |             |                                                |
| bayes_factor      | Changes the in-chain comparison factor.                                                                                                                | float       | sp.log(150)                                    |
| model_comparison  | Changes the posterior comparison between models with k signals, when this doesn't comply emperor stops running.                                        | float       | 5.0                                            |
| BIC               | This is the BIC used to compare models. Default is 5.                                                                                                  | float       | 5.0                                            |
| AIC               | This is the AIC used to compare models. Default is 5.                                                                                                  | float       | 5.0                                            |
| Model             |                                                                                                                                                        |             |                                                |
| MOAV              | Sets the Moving Average Order for the rednoise model (can be 0).                                                                                       | Int         | 1                                              |
| eccprior          | Sets the sigma for the Prior Normal Distribution for eccentricity.                                                                                     | float       | 0.3                                            |
| jittprior         | Sets the sigma for the Prior Normal Distribution for jitter.                                                                                           | float       | 5.0                                            |
| jittmean          | Sets the mu for the Prior Normal Distribution for jitter.                                                                                              | float       | 5.0 as well                                    |
| STARMASS          | Outputs the Minimum Mass and Semi-Major Axis. Should be put in solar masses.                                                                           | float/False | False                                          |
| HILL              | Enables the fact that the Hill Stability Criteria has to comply as a prior (requires STARMASS)                                                         | boolean     | False                                          |
| Plotting          |                                                                                                                                                        |             |                                                |
| PLOT              | Enables Plotting.                                                                                                                                      | boolean     | True                                           |
| CORNER            | Enables Corner plot.                                                                                                                                   | boolean     | True                                           |
| HISTOGRAMS        | Enables Beautiful Histograms for the keplerian parameters.                                                                                             | boolean     | True                                           |
| PNG               | Enables PNG plots. (light weight and fast)                                                                                                             | boolean     | True                                           |
| PDF               | Enables PDF plots. (heavy and slow, maximum quality!)                                                                                                  | boolean     | False                                          |
| draw_every_n      | Draws 1 every n points in the plots, without thining the chain for the statistics. So it takes shorter on printing the plots (never necessary in .PNG) | int         | 1                                              |
| Easter            |                                                                                                                                                        |             |                                                |
| MUSIC             | Sounds so you don't have to explicitly check EMPEROR to know what is it doing.                                                                         | boolean     | False                                          |

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

## Autorun
```sh
_run_auto__(setup, up_to_k, param=int, acc=int, moav=int)
```

setup: Sets the number of temperatures, walkers and steps you will use, respectively

up_to_k: Up to the k-th keplerian signal

param: Sets the parameterisation to use for the keplerian model.

param = 0 sets the vanilla configuration, with [period, amplitude, phase, eccentricity and longitude of periastron (w)].
param = 1 uses the Hou parameterisation, as seen on Sec. 2 in https://arxiv.org/pdf/1104.2612.pdf
param = 2 uses the time of inferior conjunction instead of phase. [period, amplitude, t0, eccentricity, w]
param = 3 uses both the time of inferior conjuction and Hou's parameterisation [per, amp, t0, sqrt(e)sin(w), sqrt(e)cos(w)]

acc: Sets the acceleration order. 0 for no acceleration, 1 for linear acceleration, 2 for linear plus quadratic...

moav: Sets the moving average order.

## conditions
```sh
.conds.append([param_name, attribute, value])
```

This modifies the corresponding parameter. It's equivalent to do in the proper part of the run:
Parameter[param_name].attribute = value

param_name: str with the name of the parameter.
attribute: str with the name of any attribute of the Parameter object.
value: The value you want to change it to.

## Others


| Command           | Action                                                                                                                                                 | Input Type  | Default                                        |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|------------------------------------------------|
| Setup             | Sets the number of temperatures, walkers and steps you will use, respectively                                                                          | Array       | Input by user                                  |
| cores__           | Sets the number of threads you are using.                                                                                                              | Int         | Default is maximum! **Let's up the tempo.**    |
| *burn_out         | Deprecated! The steps for the burn-in phase.                                                                                                           | Int         | Default is half the steps for the chainlength. |
| switch_constrain  | Constrains the search according to the results of the previous analyzed model.                                                                         | Boolean     | Default is False (True is recommended)         |
| switch_staract    | Adds the star activity model, as linear correlations.                                                                                                  | Boolean     | Default is False (True is recommended)         |
| *thin             | Deprecated. Thins the chain.                                                                                                                           | Int         | 1                                              |
| betas             | Sets the beta factor for each temperature.                                                                                                             | Array       | None (inputs [1/sqrt{2}^i for i in ntemps])    |
| Statistical Tools |                                                                                                                                                        |             |                                                |
| bayes_factor      | Changes the in-chain comparison factor.                                                                                                                | float       | np.log(10000)                                  |
| *model_comparison | Deprecated. Changes the posterior comparison between models with k signals, when this doesn't comply emperor stops running.                            | float       | 5.0                                            |
| BIC               | This is the BIC used to compare models. Default is 5.                                                                                                  | float       | 5.0                                            |
| AIC               | This is the AIC used to compare models. Default is 5.                                                                                                  | float       | 5.0                                            |
| Model             |                                                                                                                                                        |             |                                                |
| *MOAV             | Deprecated. Sets the Moving Average Order for the rednoise model (can be 0).                                                                           | Int         | 1                                              |
| eccentricity_prargs | Sets the mean and sigma for the Prior Normal Distribution for eccentricity.                                                                                   | float       | [0, 0.1]                                       |
| jitter_prargs     | Sets the mean and sigma for the Prior Normal Distribution for jitter.                                                                                           | float       | [5, 5]                                         |
| starmass          | Outputs the Minimum Mass and Semi-Major Axis. Should be put in solar masses.                                                                           | float/False | False                                          |
| *HILL             | Deprecated. Enables the fact that the Hill Stability Criteria has to comply as a prior (requires STARMASS)                                             | boolean     | False                                          |
| Plotting          |                                                                                                                                                        |             |                                                |
| plot_show         | Displays plots after run.                                                                                                                              | boolean     | False                                          |
| plot_save         | Saves plots after run.                                                                                                                                 | boolean     | False                                          |
| *CORNER           | Deprecated. Enables Corner plot.                                                                                                                       | boolean     | True                                           |
| *HISTOGRAMS       | Deprecated. Enables Beautiful Histograms for the keplerian parameters.                                                                                 | boolean     | True                                           |
| plot_fmt          | Choose the plotting format.                                                                                                                            | str         | 'png'                                          |

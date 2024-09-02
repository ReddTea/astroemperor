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
Download the tests folder and run emp_test.py to make sure everything works!

```sh
ipython  # open python environment
run emp_test
```
or just
```sh
python emp_test.py
```


# Usage
The code is really simple to use! You just need to have your data under /datafiles folder.
And then, as shown in python_file.py, append the names of the datasets you want to use in an array, set the configuration of the chain in another single array, then call the method and conquer!

# Example
```sh
import astroEMPEROR as emp
import numpy as np

sim = emp.Simulation()
sim.set_engine('emcee')
setup = np.array([3, 50, 250, 2])
sim.load_data('51Peg')  # Target folder name in /datafiles/
sim.run_auto(setup, k_start=0, k_end=2)

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
I'll update this soon, promise, in the meanwhile, there is a reference guide at the bottom. See the example file for the know-how.

## Autorun
```sh
run_auto(setup, k_start=2, k_end=2, parameterisation=int, moav=int, accel=int)
```

setup: Sets the number of temperatures, walkers and steps you will use, respectively

up_to_k: Up to the k-th keplerian signal

parameterisation: Sets the parameterisation to use for the keplerian model.


| parameterisation = | Description                                                                                                                          | Parameters                                                            |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| 0                  | Sets the vanilla configuration.                                                                                                      | period, amplitude, phase, eccentricity and longitude of periastron    |
| 1                  | Sets the Hou parameterisation, as seen on Sec. 2 in https://arxiv.org/pdf/1104.2612.pdf                                              | period, A_s, A_c, e_s, e_c                                            |
| 2                  | Uses the time of inferior conjunction instead of phase                                                                               | period, amplitude, t0, eccentricity, w                                |
| 3                  | Uses both the time of inferior conjuction and Hou's parameterisation                                                                 | per, amp, t0, e_s, e_c                                                |


acc: Sets the acceleration order. 0 for no acceleration, 1 for linear acceleration, 2 for linear plus quadratic...

moav: Sets the moving average order.

## conditions
```sh
.add_condition([param_name, attribute, value])
```

This modifies the corresponding parameter. It's equivalent to do in the proper part of the run:
Parameter[param_name].attribute = value

param_name: str with the name of the parameter.

attribute: str with the name of any attribute of the Parameter object.

value: The value you want to change it to.

## Others


| Command              | Action                                                                                                                                                 | Input Type  | Default                                        |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|------------------------------------------------|
| set_engine           | Sets the sampling engine.                                                                                                                              | str         | Input by user                                  |
| setup                | Sets the number of temperatures, walkers and steps you will use, respectively for MCMC runs.                                                           | Array       | Input by user                                  |
| load_data            | Loads data from the read directory from the folder with this name.                                                                                     | str         | Input by user                                  |
| switch_constrain     | Sets priors to the search according to the posterior of the previous run.                                                                              | Boolean     | Default is False (True is recommended)         |
| constrain_method     | Method for the prior model. Gaussian Mixtures is default.                                                                                              | Str         | 'GM'                                           |
| constrain_sigma      | Number of standard deviations used for the constrain if method is 'sigma'.                                                                             | Int         | 1                                              |
| switch_SA            | Adds the star activity model, as linear correlations.                                                                                                  | Boolean     | Default is False                               |
| ModelSelection       |                                                                                                                                                        |             |                                                |
| set_criteria         | Selects the model comparison criteria that is being used.                                                                                              | str         | 'BIC'                                          |
| set_tolerance        | Sets the tolerance for the criteria.                                                                                                                   | float       | 5.0                                            |
| update()             | Pushes the changes for the model selection criteria.                                                                                                   | None        |                                                |
| Utils                |                                                                                                                                                        |             |                                                |
| read_loc             | Changes the load root folder.                                                                                                                          | str         | ''                                             |
| save_loc             | Changes the save folder.                                                                                                                               | str         | ''                                             |
| instrument_names     | Changes the names of the instruments for all prints and plots                                                                                          | list        | Filenames                                      |
| debug_mode           | Adds some extra logging info.                                                                                                                          | Boolean     | False                                          |
| Multiprocessing      |                                                                                                                                                        |             |                                                |
| multiprocess_method  | Sets the multiprocessing method from 7 available options                                                                                               | str         | 1                                              |
| cores__              | Sets the number of threads you are using.                                                                                                              | Int         | Default is maximum! **Let's up the tempo.**    |
| Dynamics             |                                                                                                                                                        |             |                                                |
| starmass             | Outputs the Minimum Mass and Semi-Major Axis. Should be put in solar masses.                                                                           | float/False | False                                          |
| Plots                |                                                                                                                                                        |             |                                                |
| save_plots_fmt       | Format for the plot outputs.                                                                                                                           | str         | 'pdf'                                          |

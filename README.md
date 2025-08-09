# EMPEROR
Exoplanet Mcmc Parallel tEmpering for Rv Orbit Retrieval

# Overview
`EMPEROR` (Exoplanet Mcmc Parallel tEmpering for Rv Orbit Retrieval), is a Python-based algorithm that automatically searches for signals in Radial Velocity timeseries, employing Markov chains and parallel tempering methods, convergence tests and Bayesian statistics, along with various noise models. A number of posterior sampling routines are available, focused on efficiently searching for signals in highly multi-modal posteriors. The code allows the analysis of multi-instrument and multi-planet data sets and performs model comparisons automatically to return the optimum model that best describes the data.

Make sure to check the [documentation!](https://astroemperor.readthedocs.io/en/latest/)

## Why `EMPEROR`?

  - It's really simple to use
  - It has a series of configuration commands that will amaze you
  - Advanced Noise Model
  - Quite Flexible!


# Dependencies
This code makes use of:

  - [Numpy](https://numpy.org)
  - [Scipy](https://scipy.org)
  - [pandas](https://pandas.pydata.org)
  - [matplotlib>=3.5.1](https://matplotlib.org)

  - [kepler](https://github.com/dfm/kepler.py)
  - [reddemcee](https://github.com/ReddTea/reddemcee/)
  - [reddcolors](https://github.com/ReddTea/reddcolors/)
  - [tabulate](https://pypi.org/project/tabulate/)
  - [termcolor](https://pypi.python.org/pypi/termcolor)
  - [tqdm](https://pypi.python.org/pypi/tqdm)

All of them can be easily installed with pip.

For additional capabilities, you can install:

  - [arviz](https://arviz-devs.github.io/arviz/)
  - [celerite2](https://celerite2.readthedocs.io/en/latest/)
  - [corner](https://pypi.python.org/pypi/corner)
  - [dynesty](https://dynesty.readthedocs.io/en/stable/)
  - [emcee](http://dan.iel.fm/emcee/current/)
  - [scikit-learn](https://scikit-learn.org/stable/)



# Installation

## Pip
In the console type
```sh
pip3 install astroEMPEROR
```

## From Source
In the console type
```sh
git clone https://github.com/ReddTea/astroEMPEROR.git
```


## Installation Verification
Download the [tests folder](https://github.com/ReddTea/astroemperor/tree/main/tests) and run `test_basic.py` to make sure everything works!

In terminal:

```sh
python test_basic.py
```


# Quick Usage
We need to set up our working directory with two subfolders, `datafiles` and `datalogs`, the former for data input, the later for output.

```
📂working_directory
 ┣ 📜mini_test.py
 ┣ 📂datafiles
 ┃ ┣ 📂51Peg
 ┃ ┃ ┗ 📂RV
 ┃ ┃ ┃ ┗ 📜51peg.vels
 ┣ 📂datalogs
 ┃ ┣ 📂51Peg
 ┃ ┃ ┗ 📂run_1
```

Running the code is as simple as:

```python
import astroemperor

sim = astroemperor.Simulation()

sim.set_engine('reddemcee')
sim.engine_config['setup'] = [2, 100, 500, 1]
sim.load_data('51Peg')  # read from ./datafiles/

sim.plot_trace['plot'] = False  # deactivate arviz plots
sim.autorun(1, 1)  # (from=1, to=1): just 1 keplerian

```

# Outputs
All results can be found in the `datalogs` folder. You will see chain plots, posterior plots, histograms, phasefolded curves, the chain sample and more!

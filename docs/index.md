# EMPEROR
Exoplanet Mcmc Parallel tEmpering for Rv Orbit Retrieval.


## Overview
`EMPEROR` (Exoplanet Mcmc Parallel tEmpering for Rv Orbit Retrieval), is a Python-based algorithm that automatically searches for signals in Radial Velocity timeseries (and also joint modelling with Astrometry), employing Markov chains and parallel tempering methods, convergence tests and Bayesian statistics, along with various noise models. A number of posterior sampling routines are available, focused on efficiently searching for signals in highly multi-modal posteriors. The code allows the analysis of multi-instrument and multi-planet data sets and performs model comparisons automatically to return the optimum model that best describes the data.


## Why `EMPEROR`?

  - It's really simple to use
  - It has a series of configuration commands that will amaze you
  - Advanced Noise Models
  - Quite Flexible!

## Quick Install
### Dependencies
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

### Pip
In the console type
```sh
pip install astroemperor
```

### From Source
In the console type
```sh
git clone https://github.com/ReddTea/astroemperor/
cd astroemperor
python -m pip install -e .
```

### Installation Verification
Download the [tests folder](https://github.com/ReddTea/astroemperor/tree/main/tests) and run `test_basic.py` to make sure everything works!

In terminal:

```sh
python test_basic.py
```

## Quick Usage
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


## User Guide

In depth utilities of `EMPEROR`.

- [Installation](user-guide/ug-installation.md)
- [Getting Started](user-guide/ug-getting-started.md)
- [Seeting the Sampler](user-guide/ug-the-samplers.md)
- [Setting up the Model](user-guide/ug-the-model.md)
- [Model Comparison and Next Run](user-guide/ug-model-comparison.md)
- [The Results](user-guide/ug-the-results.md)
- [FAQ](user-guide/ug-FAQ.md)


## License and Attribution

Copyright (c) 2024 Pablo Peña Rojas.

`EMPEROR` is open software under the MIT License (see [License](https://github.com/ReddTea/astroemperor/blob/main/LICENSE)).

If you use `EMPEROR`, please cite the paper ([arXiv](https://arxiv.org/abs/2511.05331), [ADS](https://ui.adsabs.harvard.edu/abs/2025A%26A...704A.323P/abstract), [SciX](https://scixplorer.org/abs/2025A%26A...704A.323P/abstract), [aanda](https://www.aanda.org/articles/aa/full_html/2025/12/aa54336-25/aa54336-25.html)).



BibTeX:
```
@ARTICLE{2025A&A...704A.323P,
       author = {{Pe{\~n}a R.}, Pablo A. and {Jenkins}, James S.},
        title = "{EMPEROR: I. Exoplanet MCMC parallel tempering for RV orbit retrieval}",
      journal = {A\&A},
     keywords = {methods: data analysis, methods: numerical, techniques: radial velocities, planets and satellites: detection, planets and satellites: individual: HD 55693, planets and satellites: individual: Barnard's Star, Earth and Planetary Astrophysics, Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = dec,
       volume = {704},
          eid = {A323},
        pages = {A323},
          doi = {10.1051/0004-6361/202554336},
archivePrefix = {arXiv},
       eprint = {2511.05331},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025A&A...704A.323P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

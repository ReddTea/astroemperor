# Installation

## Requirements
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

## Pip
In the console type
```sh
pip install astroemperor
```

## From Source
In the console type
```sh
git clone https://github.com/ReddTea/astroemperor
cd astroemperor
python -m pip install -e .
```

## Recommendation

If you are using conda, I recommend creating an environment for running the `EMPEROR`:

```sh
conda create -n emperor python=3.12
```


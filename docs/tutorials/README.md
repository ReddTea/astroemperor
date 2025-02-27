# User Guide

In depth utilities of reddemcee

---

asd

---


## Installation

### Requirements
Section in construction!

### Pip
In the console type
```sh
pip install astroemperor
```

### From Source
In the console type
```sh
git clone https://github.com/ReddTea/astroemperor
cd astroemperor
python -m pip install -e .
```

---
## APT Sampler
View the object detail

---
## FAQ

### Why `EMPEROR`?
- APT is really well suited for multimodal posteriors, as well as very large prior-volumes.
- APT provides a reliable estimation of the bayesian evidence (Z)

### What is APT?
Adaptative Parallel Tempering. I highly recommend checking proper literature as the ones suggested in the homepage!

In very few sentences, Adaptive Parallel Tempering MCMC is a variant of Markov Chain Monte Carlo (MCMC) designed to improve sampling efficiency, especially for multimodal distributions. It runs multiple chains in parallel at different "temperatures". Higher temperatures explore the distribution more broadly, while lower temperatures focus on finer details. Periodically, the chains swap states, enabling better exploration of the distribution's modes. 

The "adaptive" component refers to dynamically adjusting the temperature levels based on past performance, making the method more efficient over time. This combination enhances convergence and reduces the risk of getting stuck in local minima.


### How many walkers?
[In construction!]

- At least double the dimensions.
- I recommend a multiple of the threads you are using to minimise idle-core time.
- I like to scale them as an exponential of ndim.
- You can never go wrong with more
- More walkers means more concurrent memory usage.


### How many temps?
[In construction!]

- Really problem dependant.
- At least 5, more than 20 seems overkill.



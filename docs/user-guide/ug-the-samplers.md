# Sampler Settings

You can select which Bayesian Sampling algorithm–the engine–by using `set_engine`:

```python
sim.set_engine('reddemcee')
```

Each engine is managed through two distinct dictionaries. `engine_config` is passed as named arguments to the sampler itself, and `run_config` sets options within `EMPEROR` to interact with the sampler.


## reddemcee

An Adaptative Parallel Tempering MCMC algorithm, based on the excellent `emcee`.

Chain tempering has been shown to be necessary to efficiently sample highly multi-modal posteriors, where instead of sampling the posterior of the distribution, an artificially dampened posterior is sampled. The dampening factor, is the inverse temperature \\(\beta\\).

The benefit from this is that now the samplers at different temperatures can build proposal densities that are based on chains with other temperatures, and since the walkers in the hotter chains are less constrained, they are less likely to get stuck in regions of the posterior that are much higher than others, bringing confidence to the fact that cold chain (\\(\beta\\)=1) members have sampled the actual maximum of the posterior and have not gotten trapped in a region of high probability that is not the global maximum.

Another benefit from this method is that with multiple chains at different temperatures, one is able to approximate the Bayesian Evidence, through thermodynamic integration.

More information on `reddemcee`'s options can be found on [reddemcee's documentation](https://reddemcee.readthedocs.io/en/latest/).

| engine_config | type       | description                                    |
|---------------|------------|------------------------------------------------|
| setup         | list       | [ntemps, nwalkers, nsweeps, nsteps]            |
| betas         | (opt) list | The specific temperature ladder to start with. |
| moves         | (opt) list | moves used by the sampler                      |
| tsw_history   | (opt) bool | Saves the temperature swap rate. |
| smd_history   | (opt) bool | Saves the swap mean distance.    |
| adapt_mode    | int        | Uses different adaptation schemes.             |
| adapt_tau     | float      | Ladder adaptation decay timescale.             |
| adapt_nu      | float      | Ladder adaptation decay rate.                  |
| progress      | bool       | Wheter to display the progress bar.            |

As an example, after setting the engine, you could change the number of [ntemps, nwalkers, nsweeps, nsteps] to use in the run, the temperature adaptation rate, and the initial inverse temperatures:

```python
sim.engine_config['setup'] = [5, 200, 1000, 1]
sim.engine_config['adapt_nu'] = 0.5
sim.engine_config['betas'] = [1.0, 0.624, 0.3673, 0.3414, 0.]
```




`EMPEROR` also has the option to run in batches. After a batch is done, it will check if a pre-determined convergence criteria over the auto-correlation time is met. When this criteria is met, it will cease to adapt it's ladder, and sample a final time. Options pertaining this mode start with 'adaptation' in the following list:


| run_config         | type  | description                                  |
|--------------------|-------|----------------------------------------------|
| adaptation_batches | int   | Batches where the ladder adaptation is done. |
| adaptation_nsweeps | int   | Length of each adaptation batch.             |
| adaptation_tol     | int   | Minimum chain length in tau units.           |
| adaptation_tau_diff| int   | Difference in estimated tau between batches. |
| burnin             | float | Drops the first part of the chain.           |
| thin               | int   | Thins the samples.                           |
| logger_level       | str   | 'ERROR', 'CRITICAL', 'DEBUG'                 |

Let's picture we want to run `emperor` in batches. We will use a maximum of 12 batches, each of length 500. After the stopping criteria has been met, we run the chain for an additional 1000 sweeps, and we burn-in half of those, leaving us with a total of 100,000 samples for the cold-chain (\\(200 \cdot 1000 \cdot 0.5\\)), we simply add:

```python

sim.run_config['adaptation_batches'] = 12
sim.run_config['adaptation_nsweeps'] = 500
sim.run_config['burnin'] = 0.5  # it is in niter
```




## dynesty

Although the APT method is highly recommended for broad searches in multi-modal phase-spaces, `dynesty` is an alternative Bayesian posterior sampling engine which uses DNS, Dynamic Nested Sampling, a generalisation of the Standard Nested Sampling (SNS) algorithm where the live-points (akin to MCMC walkers) vary in number to improve sampling efficiency.


`dynesty`'s sampler options can be found on its [documentation](https://dynesty.readthedocs.io/). In this case, `engine_config` is used when setting up the sampler, while `run_config` when running the sampler. Some useful options are:

| engine_config | type       | description                                         |
|---------------|------------|-----------------------------------------------------|
| nlive         | int        | N livepoints                                        |
| queue_size    | int        | for parallelization                                 |
| bound         | str        | 'none', 'single', 'multi', 'balls', 'cubes'         |
| sample        | str        | 'rwalk', 'auto', 'unif', 'rwalk', 'slice', 'rslice' |

For example, if we wanted to use the dynamic nested sampler, with 1500 live-points, multiprocessing, utilising the `slice` sampling method, with a maximum number of likelihood calls of 100,000:

```python
sim.set_engine('dynesty_dynamic')

sim.engine_config['nlive'] = 1500
sim.engine_config['queue_size'] = sim.cores__  # for multiprocessing
sim.engine_config['sample'] = 'slice'  # 'auto', 'unif', 'rwalk', 'slice', 'rslice'

sim.run_config['maxcall'] = 100000
```
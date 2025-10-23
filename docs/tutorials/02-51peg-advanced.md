# 51 Peg - advanced

This section is a hands-on tutorial on how to make a run taking advantage of `EMPEROR`'s options.
Similarly to the previous tutorial, we use the 51 Peg RV data available on [GitHub](https://github.com/ReddTea/astroemperor/tree/main/tests/datafiles/51Peg/RV).

First, we start our simulation:


```python
import astroemperor as emp
import numpy as np
np.random.seed(1234)


sim = emp.Simulation()
sim.load_data('51Peg')  # folder read from /datafiles/
```

## General Options
We take a deeper look into the Emperor options, starting with parallelisation, and sampler backend.

### Parallelisation
Parallelisation can be done with several different libraries. The available options are:

```multiprocess_method```: To change the parallelisation scheme.

```cores__```: Misnomer for how many threads (and not cores) to use.


| Code | Library         | Pool           |
|------|-----------------|----------------|
| 0    | None            | None           |
| 1    | multiprocessing | Pool           |
| 2    | multiprocess    | Pool           |
| 3    | multiprocessing | ThreadPool     |
| 4    | pathos          | ProcessingPool |
| 5    | schwimmbad      | SerialPool     |
| 6    | schwimmbad      | JoblibPool     |
| 7    | schwimmbad      | MultiPool      |

### Sampler backend
`reddemcee` has two different backends. `PTBackend` is the default, working exclusively with RAM, and `HDFBackend`, stores the chain in an HDF5 file using `h5py`, saving it there step-by-step.

The first one is faster, but requires a higher RAM usage.

The second one is safer and less memory-hungry. To use the h5 backend, simply change the `backend_bool` attribute to `True`.


```python
sim.multiprocess_method = 1  # multiprocessing Pool
sim.cores__ = 12  # threads for the run
sim.backend_bool = False  # True for h5py backend
```

## Engine configuration

We add ```set_engine``` with some custom options. We will use a different starting temperature ladder. A good initial ladder increases the efficiency of the adaptation, in the same way a good initial guess for the parameters increases the convergence time.

We will use a different adaptation algorithm, based on the specific heat of the system. We also change the adaptation rate, and decay timescale.


```python
ntemps, nwalkers, nsweeps, nsteps = 10, 256, 2048, 1
sim.set_engine('reddemcee')


sim.engine_config['setup'] = [ntemps, nwalkers, nsweeps, nsteps]

sim.engine_config['betas'] = list(np.linspace(1, 0, ntemps))

sim.engine_config['tsw_history'] = True  # save temperature swaps per sweep
sim.engine_config['smd_history'] = True  # save swap mean distance per sweep

sim.engine_config['adapt_tau'] = 100
sim.engine_config['adapt_nu'] = 1.5
sim.engine_config['adapt_mode'] = 2  # Specific Heat
```

## Run Configuration
We will set up a 10\% burn-in phase.

```python
sim.run_config['burnin'] = 0.1
```

### Model Comparison
In our example, instead of comparing BIC we will compare Evidences directly.
We can change the evidence estimation method between Curvature-aware Thermodynamic Integration (TI+), Geometric-Bridge Stepping Stones (SS+), and a Hybrid algorithm (for more details see the [reddemcee paper](https://arxiv.org/abs/2509.24870)).


```python
sim.set_comparison_criteria('Evidence')
sim.set_tolerance(10)  # difference between models

sim.evidence_method = 'ss'  # stepping stones
```

## Model Configuration

We feed the name of the instrument (optional), as well as the stellar mass for calculating the minimum-mass and semi-major axis, and stellar mass error (optional). We will use the Keplerian parameterisation $(P, K, \\(\phi\\), \\(e\\), \\\omega\\). We add some boundaries to speed up the process, and add some initial positions:

```python
sim.instrument_names_RV = ['LICK']
sim.starmass = 1.12
sim.starmass_err = 0.04
sim.keplerian_parameterisation = 0


sim.add_condition(['Period 1', 'limits', [3, 5]])
sim.add_condition(['Amplitude 1', 'limits', [45, 60]])
sim.add_condition(['Eccentricity 1', 'limits', [0, 0.5]])

sim.add_condition(['Offset 1', 'limits', [-10., 10.]])

sim.add_condition(['Period 1', 'init_pos', [4.1, 4.3]])
sim.add_condition(['Amplitude 1', 'init_pos', [50, 60]])
sim.add_condition(['Eccentricity 1', 'init_pos', [0, 0.1]])
```


### Plotting Options
We add some plotting options to speed up this test a little. We will only plot the posteriors for the cold chain, and two intermediate chains. Also, we won't use the `arviz` optional plots.


```python
sim.plot_posteriors['temps'] = [0]
sim.plot_trace['plot'] = False
sim.plot_gaussian_mixtures['plot'] = False
```

Finally, we run our simulation (it will take some minutes):


```python
sim.autorun(0, 1)
```
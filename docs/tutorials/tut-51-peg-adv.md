# 51 Peg

This section is a hands-on tutorial on how to make a simple run.
We use the 51 Peg RV data available on [GitHub](https://github.com/ReddTea/astroemperor/tree/main/tests/datafiles/51Peg/RV).

## Data
We need to set up our working directory with two subfolders, `datafiles` and `datalogs`. 

`datafiles` will contain our RV catalogues. For each target or system we create a subfolder with the system name. In this case, `51Peg`. Inside, we create a second subfolder, named `RV`, which will contain the data to be read.

We copy-paste the file downloaded from GitHub into `/datafiles/51Peg/RV/`.


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

## Setting up EMPEROR

Under our working directory, we create a python file named `mini_test`.
First, we import the library and start our simulation:

```python
import astroemperor as emp
import numpy as np
np.random.seed(1234)


sim = emp.Simulation()
sim.load_data('51Peg')  # folder read from /datafiles/
```

### Setting the engine

For this example, we will use `reddemcee`, with 10 temperatures, 500 walkers, 3000 sweeps each of 1 step:

```python
sim.set_engine('reddemcee')
sim.engine_config['setup'] = [10, 500, 3000, 1]
```

### Setting the model
We feed the name of the instrument (optional), as well as the starmass for calculating the minimum-mass and semi-major axis. We will use the Keplerian parameterisation (P, K, \\(\phi\\), \\(e_{s}\\), \\(e_{c}\\)). We add some boundaries to speed up the process, and add some initial positions:

```python
sim.instrument_names_RV = ['LICK']
sim.starmass = 1.12
sim.keplerian_parameterisation = 1


sim.add_condition(['Period 1', 'limits', [3, 5]])
sim.add_condition(['Amplitude 1', 'limits', [45, 60]])

sim.add_condition(['Offset 1', 'limits', [-10., 10.]])

sim.add_condition(['Period 1', 'init_pos', [4.1, 4.3]])
sim.add_condition(['Amplitude 1', 'init_pos', [50, 60]])
```

### Plotting Options
We add some plotting options to speed up this test a little. We will only plot the posteriors for the cold chain, and two intermediate chains. Also, we won't use the `arviz` optional plots.

```python
sim.plot_posteriors['temps'] = [0, 2, 6]
sim.plot_trace['plot'] = False

```

Finally, we run our simulation (it will take some minutes):

```python
sim.autorun(0, 1)
```


## Advanced example

```python
import multiprocessing
sim = emp.Simulation()
sim.load_data('51Peg')  # folder read from /datafiles/
sim.use_c = True  # (OPTIONAL) uses the fast-kepler c library, requires installation

sim.multiprocess_method = 1  # for the multiprocessing library
sim.cores__ = multiprocessing.cpu_count()  # all the available cores

sim.save_backends = False  # OPTIONAL: don't save the chains for this test
```

### Engine Setup
This time, we set additional options for both `emperor` and the sampler.
We tweak the adaptation parameters in `engine_config`, and we select the starting betas.

```python
sim.set_engine('reddemcee')
sim.engine_config['setup'] = [12, 500, 3000, 1]

sim.engine_config['adapt_tau'] = 500
sim.engine_config['adapt_nu'] = 1
sim.engine_config['adapt_mode'] = 0

sim.engine_config['betas'] = list(np.linspace(1, 0, 12))  # 12 temperatures
```



### Run Setup
We set up a burn-in period of half the steps. For the following runs, we can add constrains on the solution based on previous results.
We change the comparison criteria to the 'Evidence' and ask for a difference of 10:

```python
sim.run_config['burnin'] = 0.5  # half the steps

sim.switch_constrain = True
sim.constrain_method = 'range'  # based on the HDI
sim.constrain_sigma = 3  # equivalent to the 99 HDI range

sim.set_comparison_criteria('Evidence')
sim.set_tolerance(10)
```


### Model Setup
Here we can add more to the model.


```python
sim.instrument_names_RV = ['LICK']
sim.starmass = 1.12
sim.keplerian_parameterisation = 1

sim.sinusoid = False  # a simple sinusoid, turned off
sim.magnetic_cycle = False  # a magnetic cycle model, turned off
sim.acceleration = 1  # adds a linear acceleration


sim.add_condition(['Period 1', 'limits', [3, 5]])
sim.add_condition(['Amplitude 1', 'limits', [45, 60]])

sim.add_condition(['Offset 1', 'limits', [-10., 10.]])

sim.add_condition(['Period 1', 'init_pos', [4.1, 4.3]])
sim.add_condition(['Amplitude 1', 'init_pos', [50, 60]])
```

### Plot Options

```python
sim.plot_posteriors['temps'] = [0]  # only cold chain
sim.plot_posteriors['modes'] = [0]  # only scatter plot

sim.plot_histograms['temps'] = [0]  # only cold chain
sim.plot_rates['window'] = 10  # takes averages in window, for visual simplicity

sim.plot_trace['plot'] = True  # optional, requires arviz
sim.plot_trace['modes'] = [0]  # only the corner plot

```

Finally, we run our simulation:

```python
sim.autorun(0, 1)
```




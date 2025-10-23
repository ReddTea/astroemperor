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
```

### Setting the engine

For this example, we will use `reddemcee`, with 10 temperatures, 256 walkers, 2048 sweeps each of 1 step:

```python
sim.set_engine('reddemcee')
sim.engine_config['setup'] = [10, 256, 2048, 1]
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

Finally, we read the data and run our simulation (it will take some minutes):

```python
sim.load_data('51Peg')  # folder read from /datafiles/
sim.autorun(0, 1)
```



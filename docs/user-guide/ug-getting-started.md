```python
from IPython.display import Image, display
import numpy as np
import astroemperor as emp

np.random.seed(1234)
```

# Quickstart

We will start with a simple evaluation of 51 Peg. The data for this system can be downloaded from the [GitHub](https://github.com/ReddTea/astroemperor/tree/main/tests/datafiles/51Peg/RV).


## Working Directory
The working directory must have a folder named `datafiles`, where the RVs will be read, and a folder called `datalogs`, where all relevant results will be stored.
Below is an example of a work directory:
```
📂working_directory
 ┣ 📜mini_test.py
 ┣ 📜my_scripts.py
 ┣ 📂datafiles
 ┃ ┣ 📂51Peg
 ┃ ┃ ┗ 📂RV
 ┃ ┃ ┃ ┗ 📜51peg.vels
 ┣ 📂datalogs
 ┃ ┣ 📂51Peg
 ┃ ┃ ┗ 📂run_1
```

## Running the code
We set up the basics for the run to check that everything is working as intended. The results won't be conclussive since this is a very short run. Feel free to explore the results in the `run_1` folder!

```python
sim = emp.Simulation()

sim.set_engine('reddemcee')  # sets the engine
sim.engine_config['setup'] = [2, 100, 500, 1]  # ntemps, nwalkers, nsweeps, nsteps
sim.load_data('51Peg')  # read from ./datafiles/
sim.plot_trace['plot'] = False  # to speed up the test

sim.autorun(1, 1)  # (from=1, to=1): just 1 keplerian
```
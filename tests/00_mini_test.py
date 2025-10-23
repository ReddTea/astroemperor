# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import astroemperor as emp

np.random.seed(1234)

## Engine Setup
sim = emp.Simulation()
sim.load_data('51Peg')  # folder read from /datafiles/

sim.set_engine('reddemcee')
sim.engine_config['setup'] = [8, 128, 512, 1]  # ntemps, nwalkers, nsweeps, nsteps

sim.cores__ = 12  # threads to use

## Model setup
sim.instrument_names_RV = ['LICK']
sim.starmass = 1.12
sim.keplerian_parameterisation = 1


sim.add_condition(['Period 1', 'limits', [3, 5]])
sim.add_condition(['Amplitude 1', 'limits', [45, 60]])

sim.add_condition(['Offset 1', 'limits', [-10., 10.]])

sim.add_condition(['Period 1', 'init_pos', [4.1, 4.3]])
sim.add_condition(['Amplitude 1', 'init_pos', [50, 60]])


## Plot Options
sim.plot_posteriors['temps'] = [0, 2, 8]
sim.plot_trace['plot'] = False


sim.autorun(1, 1)

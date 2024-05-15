# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import astroemperor as emp

np.random.seed(1234)


sim = emp.Simulation()
target_folder = '51Peg'
sim.load_data(target_folder)


setup = np.array([5, 100, 500])  # temps, walkers, steps

sim.set_engine('reddemcee')
sim.reddemcee_config['burnin'] = 0.25


# constrains to speed up the test
if target_folder == '51Peg':
    sim.instrument_names = ['LICK']
    sim.starmass = 1.12

    sim.add_condition(['Period 1', 'limits', [3, 5]])
    sim.add_condition(['Amplitude 1', 'limits', [45, 60]])
    sim.add_condition(['Phase 1', 'limits', [2.5, 3.5]])
    sim.add_condition(['Eccentricity 1', 'limits', [0, 0.1]])
    sim.add_condition(['Longitude 1', 'limits', [1.1, 1.5]])

    sim.add_condition(['Offset 1', 'limits', [-10., 10.]])

# some options to speed up the test
# PLOT OPTIONS
sim.plot_paper_mode = True  # superseeds other options
sim.save_plots_fmt = 'png'  # lower memory usage


sim.run_auto(setup, k_start=1, k_end=1)

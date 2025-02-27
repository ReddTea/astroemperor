# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
# -*- coding: utf-8 -*-

import astroemperor

sim = astroemperor.Simulation()

sim.set_engine('reddemcee')
sim.engine_config['setup'] = [2, 100, 500, 1]
sim.load_data('51Peg')  # read from ./datafiles/

sim.plot_trace['plot'] = False  # deactivate arviz plots

sim.autorun(1, 1)  # (from=1, to=1): just 1 keplerian


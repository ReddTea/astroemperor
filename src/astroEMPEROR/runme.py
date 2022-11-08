# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.1.0
# date 21 jul 2021

import numpy as np
import main_body as mb




sim = Simulation()
# data
sim._add_data__('Test', label='Test Data 1')
sim._sort_data__()

# model
#sim._mk_sinusoid__()
sim._mk_keplerian__()
sim._mk_keplerian__()
#sim._mk_acceleration__(1)
sim._mk_noise_instrumental__()

# engine
'''
import emcee
sim._set_engine__(emcee)

# ntemps, nwalkers, nsteps
setup = np.array([2, 100, 200])


sim._run__(setup)
sim._post_process__(setup)

sim.plot2()
sim.plot3()
'''

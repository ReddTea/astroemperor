# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.3
# date 18 nov 2022

import numpy as np
import astroemperor

np.random.seed(1234)

sim = astroemperor.Simulation()

'''
sim.set_engine('dynesty_dynamic') # reddemcee, dynesty, dynesty_dynamic, pymc3
setup = np.array([200, 20])  # nlive, nlive_batch
sim.dynesty_config['dlogz_init'] = 0.05
sim.dynesty_config['nlive_init'] = 200
sim.dynesty_config['nlive_batch'] = 20
sim.dynesty_config['maxiter'] = 800
'''
sim.set_engine('reddemcee')

sim.reddemcee_config['burnin'] = 'half'
sim.reddemcee_config['thinby'] = 1
setup = np.array([3, 50, 100])
    
sim.ModelSelection.set_criteria = 'BIC'  # default is BIC
sim.ModelSelection.set_tolerance = 5

#sim.save_loc = '../otrafolder/'

my_parameterisation = 0
# evidence -169 +- 0.2
# max posterior -137

sim.load_data('synth')
sim.instrument_names = ['Synth Data 1', 'Synth Data 2']

if True:
    if my_parameterisation == 0:
        sim.add_condition(['Period 1', 'limits', [58, 62]])
        sim.add_condition(['Period 2', 'limits', [23, 25]])

        sim.add_condition(['Amplitude 1', 'limits', [198, 202]])
        sim.add_condition(['Amplitude 2', 'limits', [98, 102]])

        sim.add_condition(['Phase 1', 'limits', [np.pi/4-0.1, np.pi/4+0.1]])
        sim.add_condition(['Phase 2', 'limits', [np.pi/3-0.1, np.pi/3+0.1]])

        sim.add_condition(['Eccentricity 1', 'fixed', 0])
        sim.add_condition(['Eccentricity 2', 'fixed', 0])

        sim.add_condition(['Longitude 1', 'fixed', 0])
        sim.add_condition(['Longitude 2', 'fixed', 0])

        #sim.add_condition(['Offset 1', 'limits', [99, 101]])
        #sim.add_condition(['Offset 2', 'limits', [-101, -99]])

        sim.add_condition(['Acceleration', 'fixed', 0])

    if my_parameterisation == 1:
        sim.add_condition(['lPeriod 1', 'limits', [np.log(55), np.log(65)]])
        sim.add_condition(['lPeriod 2', 'limits', [np.log(20), np.log(30)]])

        sim.add_condition(['Amp_sin 1', 'limits', [9, 11]])
        sim.add_condition(['Amp_sin 2', 'limits', [7, 10]])

        sim.add_condition(['Amp_cos 1', 'limits', [9, 11]])
        sim.add_condition(['Amp_cos 2', 'limits', [5.5, 8]])


        sim.add_condition(['Ecc_sin 1', 'fixed', 0])
        sim.add_condition(['Ecc_sin 2', 'fixed', 0])

        sim.add_condition(['Ecc_cos 1', 'fixed', 0])
        sim.add_condition(['Ecc_cos 2', 'fixed', 0])

    if my_parameterisation == 2:
        sim.add_condition(['Period 1', 'limits', [55, 65]])
        sim.add_condition(['Period 2', 'limits', [20, 30]])

        #sim.add_condition(['T_0 1', 'limits', [7.3, 7.7]])
        #sim.add_condition(['T_0 2', 'limits', [9.7, 10.3]])

        sim.add_condition(['Amplitude 1', 'limits', [195, 205]])
        sim.add_condition(['Amplitude 2', 'limits', [95, 105]])

        sim.add_condition(['Eccentricity 1', 'fixed', 0])
        sim.add_condition(['Eccentricity 2', 'fixed', 0])

        sim.add_condition(['Longitude 1', 'fixed', 0])
        sim.add_condition(['Longitude 2', 'fixed', 0])

    if my_parameterisation == 3:
        pass


sim.multiprocess_method = 1  # 0 no mp, 1 multiprocessing


sim.starmass = 0.3  # None or 0 for nothing done here
sim.switch_dynamics = True
sim.switch_constrain = True
sim.constrain_method = 'GM'   # 'sigma', 'GM'
sim.constrain_sigma = 1

sim.gaussian_mixtures_fit = True

sim.plot_gaussian_mixtures['plot'] = True

sim.plot_keplerian_model['plot'] = True  # hist, uncertain
sim.plot_keplerian_model['hist'] = True
sim.plot_keplerian_model['errors'] = True
sim.plot_keplerian_model['uncertain'] = False
sim.plot_keplerian_model['format'] = 'pdf'
sim.plot_keplerian_model['logger_level'] = 'CRITICAL'  # ERROR, CRITICAL


sim.plot_trace['plot'] = True
sim.plot_trace['modes'] = [0,1,2,3]  # 0:trace, 1:norm_post, 2:dens_interv, 3:corner


#sim.run_auto(setup, k_start=2, k_end=2, parameterisation=my_parameterisation, moav=0, pool=1)
sim.run_auto(setup, k_start=1, k_end=2,
                       parameterisation=my_parameterisation,
                       moav=0, accel=0)








'''
# Offset 100 plotting tests

truths = np.array([60, 200, np.pi/4, 0, 0])
truths2 = np.array([25, 100, np.pi/3, 0, 0])
mu1, sigma1 = 0, 15
mu2, sigma2 = 0, 25

offset1 = 100
offset2 = -100




# Benchmarks
macos k1, i1, [5, 50, 500], 16.84s
macos k2, i2, [5, 50, 500], 34s, 34s

prior tests
# pass string
# 5, 50, 200
t = [9.454, 9.354, 9.047, 9.361, 9.355, 9.0426, 9.3798]

# pass func
5, 50, 200
[9.547, 9.088, 9.342, 9.340, 9.116, 9.454]

# pass obj
9.411 9.381 9.072

# Saddie
synth_ins1_2k_moav.vels
5, 50, 200
k2, i1, moav1

##############
clean:  5.099
pool1:
pool2:
pool3:  11.23
pool4:  5.52
pool5:  5.10    5.21
pool6:  5.7523

#################
MOAV2
clean:  16.504
pool1:
pool2:
pool3:  30.105
pool4:  17.013
pool5:  16.448
pool6:  18.166


# Truths
synth_ins1_2k_moav.vels
truths = np.array([60, 200, np.pi/4, 0, 0])     0.7853
truths2 = np.array([25, 100, np.pi/3, 0, 0])    1.0471
mu1, sigma1 = 0, 15
mu2, sigma2 = 0, 25

offset1 = 50
jitter1 = 10
theta_ma = 0.9, 20
'''

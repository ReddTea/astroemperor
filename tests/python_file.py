# @auto-fGJ9827old regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.1.2
# date 3 sept 2021

import astroEMPEROR as emp
import numpy as np
import sys
import matplotlib.pyplot as pl

if True:
    # Simulation setup

    sim = emp.Simulation()  # Creates the instance
    #sim.developer_mode = True  # displays a couple of extra messages

    # workaround for OSX systems
    if sys.platform == 'darwin':
        sim.cores__ = 1
    # ENGINE
    sim._set_engine__('emcee')

    # engine things

    # ntemps, nwalkers, nsteps
    setup = np.array([3, 50, 500])
    #sim.betas = np.array([1, 0.7, 0.5, 0.3, 0.1])

    # DATA
    # sim.save_loc = '/loc/'
    # sim.read_loc = '/loc/'
    sim._data__('GJ876')  # Target folder name in /datafiles/


    # PLOT OPTIONS, defaults  are False
    sim.plot_save = True
    sim.run_save = True  # this is overrided by individual chain saves
    #sim.plot_show = True
    #self.plot_fmt = 'png'  # default is .png

    sim.starmass = 0.37  # optional, required for planet signatures
    # GJ876 stars mass 0.37

    # MODEL
    # wanna add extra stuff? just add it before hand
    # sim._mk_acceleration__()  # this is default now in _run_auto__

    # sim.jitter_limits = [[0, 100] for i in range(7)]  # dont move really
    sim.jitter_prargs = [[5, 5] for i in range(7)]


    ### CONDITIONS HERE

    sim.conds.append(['Period 1', 'limits', [55, 65]])
    sim.conds.append(['Period 2', 'limits', [25, 35]])

    sim.conds.append(['Amplitude 1', 'limits', [190, 210]])
    sim.conds.append(['Amplitude 2', 'limits', [95, 105]])

    sim.conds.append(['Eccentricity 1', 'fixed', 0])
    sim.conds.append(['Eccentricity 2', 'fixed', 0])

    #sim._mk_staract__()  # run with star index
    sim.switch_constrain = True
    #sim.switch_cherry = True

    #sim.bayes_factor = np.log(10000)  # switch_cherry
    #sim.minimum_samples = 1000  # switch_cherry

    sim._run_auto__(setup, 2, param=0, acc=1, moav=0)  # how many signals
    # auto instanciates the noise parameters and tries to find the best signal model

    # kwargs
    # param=0 *read below
    # moav=0
    # SAVE STUFF
    sim.save_chain([0])  # to save more chains, [0, 1, 2, ...]
    sim.save_posteriors([0])

    pl.close('all')  # close all instances of pl open

    # Alternatively, hand knit the model and run
    if False:
        sim._mk_keplerian__()  # for parametrization use __(param=xxx*)
        sim._mk_noise_instrumental__()  # for MOAV order n use __(n)
        sim._mk_acceleration__()  # for order n use __(n)

        # * param = 0 or param = 'vanilla' uses per, amp, phase, ecc and w
        # param = 1 or param = 'hou' uses per, sqrt(a)sin(ph), sqrt(a)cos(ph), sqrt(e)sin(w), sqrt(e)cos(w)
        # param = 2 or param = 't0' uses per, amp, t0, ecc, w
        # param = 3 or param = 'hout0' uses per, amp, t0, sqrt(e)sin(w), sqrt(e)cos(w)

        # RUN
        #sim._run__(setup)

        # post processing, stats and plots
        #sim._post_process__(setup)

        # PLOTS
        #sim.plotmodel()
        #sim.plottrace()
        #sim.plotpost()
        #sim.plotcorner()

        # SAVE
        #sim.save_chain([0])  # to save more chains, [0, 1, 2, ...]
        #sim.save_posteriors([0])











        #

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# EMPEROR Exoplanet Mcmc Parallel tEmpering Radial velOcity fitteR
# version 0.-432,12.78

# DEPENDENCIES
if True:
    from __future__ import division
    import os
    import sys
    import numpy as np
    import scipy as sp
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab

    import emcee
    from emcee import PTSampler
    import multiprocessing
    from PyAstronomy.pyasl import MarkleyKESolver
    import time as chrono
    import datetime as dt

    from decimal import Decimal  # histograms
    import corner

    import emperors_library as emplib
    import emperors_mirror as empmir


    try:
        from tqdm import tqdm
    except ImportError:
        raise ImportError('You don t have the package tqdm installed.\
                           Try pip install tqdm.')
    try:
        from termcolor import colored
    except:
        print('You are missing the most cool package in Python!\
               Try pip install termcolor')
    try:
        from pygame import mixer
        mixer.init()
        imperial = mixer.Sound('mediafiles/imperial_march.wav')
        thybiding = mixer.Sound('mediafiles/swvader04.wav')
        technological_terror = mixer.Sound('mediafiles/technological.wav')
        alerted = mixer.Sound('mediafiles/alerted.wav')
        junk = mixer.Sound('mediafiles/piece_o_junk.wav')
        technical = mixer.Sound('mediafiles/technical.wav')
        fault = mixer.Sound('mediafiles/yourfault.wav')

    except:
        print('You are missing the most cool package in Python!\
               Try pip install pygame or set MUSIC=False')
else:
    print('You are missing some libraries :/')

# DUMMY FUNCTIONS
def logp(theta, func_logp, args):
    return func_logp(theta, args)


def logl(theta, func_logl, args):
    return func_logl(theta, args)


class EMPIRE:
    def __init__(self, stardat, setup, file_type='rv_file'):
        assert len(stardat) >= 1, 'stardat has to contain at least 1 file ! !'
        assert len(setup) == 3, 'setup has to be [ntemps, nwalkers, nsteps]'
        #  Setup
        self.cores = multiprocessing.cpu_count()
        self.setup = setup
        self.ntemps, self.nwalkers, self.nsteps = setup
        self.betas = None

        self.burn_out = self.nsteps // 2
        self.RV = False
        self.PM = False

        # initialize flat model, this should go elsewhere
        # name  # units     # prior     # lims  # args
        self.theta = sp.array([])
        self.ld = {'uniform':0,
                    'linear':2,
                    'quadratic':2,
                    'square-root':2,
                    'logarithmic':2,
                    'exponential':2,
                    'power2':2,
                    'nonlinear':2}

        #  Reading data


        if False:  # this will contain rv+pm
            pass

        elif file_type=='rv_file':
            self.rvfiles = stardat
            rvdat = emplib.read_data(stardat)
            self.time, self.rv, self.err, self.ins = rvdat[0]  # time, radial velocities, error and instrument flag
            self.all_data = rvdat[0]
            self.staract, self.starflag = rvdat[1], rvdat[2]  # time, star activity index and flag
            self.totcornum = rvdat[3]  # quantity if star activity indices

            self.nins = len(self.rvfiles)  # number of instruments autodefined
            self.ndat = len(self.time)  # number of datapoints
            self.RV = True
            # PM
            self.time_pm, self.rv_pm, self.err_pm, self.ins_pm = 0., 0., 0., 0.
            self.totcornum_pm = 0.

        elif file_type=='pm_file':
            self.pmfiles = stardat
            pmdat = emplib.read_data(stardat)
            self.time_pm, self.rv_pm, self.err_pm, self.ins_pm = pmdat[0]
            self.all_data_pm = pmdat[0]
            self.staract_pm, self.starflag_pm = pmdat[1], pmdat[2]  # time, star activity index and flag
            self.totcornum_pm = pmdat[3]  # ?

            self.nins_pm = len(self.pmfiles)
            self.ndat_pm = len(self.time_pm)
            self.MOAV_pm = 0  # for flat model
            self.PM = True

            self.params_pm = sp.array([])
            self.lenppm = len(self.params_pm)
            #  Correlate with rv's
            self.fsig = 5
            self.f2k = None
            #  acceleration quadratic
            self.PACC_pm = False
        else:
            print('You sure you wrote the filetype correctly mate?'')
        #  Statistical Tools
        self.bayes_factor = sp.log(150)  # inside chain comparison (smaller = stricter)
        self.model_comparison = 5  # between differet k configurations
        self.BIC = 5
        self.AIC = 5

        #  Menudencies
        self.thin = 1
        self.starname = self.rvfiles[0].split('_')[0]
        self.STARMASS = False
        self.HILL = False
        self.CHECK = False
        self.RAW = False
        self.MUSIC = False

        # Plotting stuff
        self.INPLOT = True
        self.draw_every_n = 1
        self.PNG = True
        self.PDF = False
        self.CORNER = True
        self.HISTOGRAMS = True
        self.breakFLAG = False

        # About the search parameters
        self.ACC = 1  # Acceleration order
        self.WN = True  # jitter fitting (dont touch)
        self.MOAV = 1  # MOAV order

        # EXTERMINATE
        self.eccprior = 0.3
        self.jittprior = 5.0
        self.jittmean = 5.0

        # auxiliary for later
        self.sampler = 0.0
        ########################################

        pass

    def conquer(self, from_k, to_k, logl=logl, logp=logp, BOUND=sp.array([])):
        # 1 handle data
        # 2 set adecuate model
        # 3 generate values for said model, different step as this should allow configuration
        # 4 run chain
        # 5 get stats (and model posterior)
        # 6 compare, exit or next
        # 7 remodel prior, go back to step 2

        assert self.cores >= 1, 'Cores is set to 0 ! !'
        assert self.thin * self.draw_every_n < self.nsteps, 'You are thining way too hard ! !'
        if self.betas is not None:
            assert len(self.betas) == self.ntemps, 'Betas array and ntemps dont match ! !'

        if self.MUSIC:
            imperial.play()

        pass














#

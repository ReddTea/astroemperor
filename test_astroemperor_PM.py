#!/usr/bin/env python
# -*- coding: utf-8 -*-
import astroemperor_pm
import scipy as sp


# SETUP

BOUNDARY = sp.array([[4.11098843e+00, 4.11105404e+00, -1.01515928e+01, -6.33312469e+00, 1.00520806e+01,   1.35949748e+01, 0.004, 0.009],
                     [3.40831451e+00, 3.40863545e+00, -5.69294095e+00, 6.05896817e-02, -9.33328013e+00, -8.07370401e+00, 0.01,   0.05]])

stardat = sp.array(['GJ876_2_KECK.vels'])  # same data
setup = sp.array([1, 80, 200])  # ntemps, nwalkers, nsteps, now real

em = astroemperor_pm.EMPIRE(stardat, setup)  # EMPIRE(data_to_read, chain_parameters)
em.betas = sp.array([1.00])
#em.PACC = True
em.STARMASS = False  # known mass for this particular star GJ876
em.HILL = False
em.CORNER = False  # corner plot disabled
em.eccprior = 0.3  # sigma for the eccentricity prior!
em.jittprior = 5.0  # sigma for the jitter prior
em.jittmean = 5.0
em.MOAV = 0  # Moving Average Order, works from 0 to the number of datapoints
# em.MUSIC= True  # Music ON, False for OFF

em.conquer(2, 3, BOUND=BOUNDARY)  # up to 5 signals!?

'''
#
[  2.17026166e-02,   4.89399730e+00  -2.14956999e+01   7.95963710e-01
   9.77157796e+00, -1.08881758e-01   3.48063468e-02  -6.87520004e-02
  -1.18413130e-01,   1.59788110e-02]
'''

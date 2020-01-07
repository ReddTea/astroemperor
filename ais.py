# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*elif/ /^\s*def/ /^\s*class spec/
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# EMPEROR Exoplanet Mcmc Parallel tEmpering Radial velOcity fitteR
# version 0.572.-47/31,64 Pluto, Ceres
'''
Na fone Eyfelevoy bashni
S Ayfona selfi zayeboshim
A nakhuya zh yeshche nam nash voyazh?
VOYAAAAAAAAAAZH, VOYAAAAAAAAAAZH 1.41
'''


# DEPENDENCIES
from __future__ import division

if True:
    import os
    import sys
    import numpy as np
    import scipy as sp
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    from functools import reduce as ft_red
    from operator import iconcat as op_ic

    import batman

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
        raise ImportError("You don't have the package tqdm installed.\
                           Try pip install tqdm.")
    try:
        from tabulate import tabulate
    except ImportError:
        raise ImportError("You don't have the package tabulate installed.\
                           Try pip install tabulate.")
    try:
        from termcolor import colored
    except:
        print('You are missing cool-ors in your life!\
               Try pip install termcolor'

               )
    try:
        from pygame import mixer
        mixer.init()
        imperial = mixer.Sound('mediafiles/imperial_march.wav')
        thybiding = mixer.Sound('mediafiles/swvader04.wav')
        technological_terror = mixer.Sound('mediafiles/technological.wav')
        alerted = mixer.Sound('mediafiles/alerted.wav')
        junk = mixer.Sound('mediafiles/piece_o_junk.wav')
        technical = mixer.Sound('mediafiles/technical.wav')
        fault = mixer.Sound('mediafiles/your_fault.wav')

    except:
        imperial = False
        thybiding = False
        technological_terror = False
        alerted = False
        junk = False
        technical = False
        fault = False
        print('You are missing the most cool package in Python!\
               Try pip install pygame or set MUSIC=False')

# DUMMY FUNCTIONS

def logp(theta, func_logp, args):
    '''
    Dummy function for emcee. Shouldn't be touched.
    '''
    return func_logp(theta, args)


def logl(theta, func_logl, args):
    '''
    Dummy function for emcee. Shouldn't be touched.
    '''
    return func_logl(theta, args)


def neo_init_batman(t, ld_mod, ldn):
    '''
    initializes batman
    delete this after deleting the if true at the end
    '''
    n = {'t0': min(t), 'per': 1., 'rp': 0.1, 'a': 15.,
         'inc': 87., 'ecc':0., 'w':90.}
    params = batman.TransitParams()
    for x in n:
        setattr(params, x, n[x])
    params.limb_dark = ld_mod  # limb darkening model
    ld_coefs = sp.ones(ldn) * 0.5  # dummy coefficients  # not 1 # DEL

    params.u = ld_coefs
    model = batman.TransitModel(params, t)
    return model, params


class spec_list:
    """Class wrapping spec utilities in a list format.

    A simple list for spec objects with inbuilt functions for ease of use.

    Attributes
    ----------
    list_ : list
        This is the list object containing the spec objects.
    ndim_ : int
        Returns the real dimentionality of the problem (doesn't count either
        fixed parameters nor joined twice).
    gral_priors : List of functions.
        List containing all the extra priors that are not directly attached to
        a parameter.
    C : List
        Coordinator. It's a simple list with the positions of the parameters to
        fit.
    A : List
        Anticoordinator. It's a simple list with the positions of the fixed
        parameters. It's the complement of the coordinator, hencec the name.

    """

    def __init__(self):
        self.list_ = sp.array([])
        self.ndim_ = 0
        self.gral_priors = sp.array([])

        self.C = []  # coordinator
        self.A = []  # anticoordinator
        self.B = []  # joinator (!?)

        self.CV = sp.array([[0,0,0] for _ in range(10)])

    def __repr__(self):
        """Human readable representation."""
        rep = [str(o) for o in self.list_]
        return str(rep)

    def __str__(self):
        rep = [str(o) for o in self.list_]
        return str(rep)

    def __len__(self):
        """Overload len() method.

        To get length of speclist now do: len(speclist)
        """
        return len(self.list_)

    def __getitem__(self, item):
        return self.list_[item]

    def _update_list_(self):
        """Update the ndim_, C and A attributes.

        -------
        function
            Really, it just updates.

        """
        self.A = []
        self.C = []
        ndim = len(self)
        priors = self.list('prior')
        for i in range(len(self)):
            if priors[i] == 'fixed':
                ndim -= 1
                self.A.append(i)
                self.list_[i].lims = [sp.nan, sp.nan]
            elif priors[i] == 'joined':
                for j in range(len(self)):
                    if i==j:
                        pass
                    elif self[i].name.split('_')[:-1][0] == self[j].name:
                        break
                    else:
                        pass

                ndim -= 1
                self.B.append([i, j])
                self.list_[i].lims = [sp.nan, sp.nan]

            else:
                self.C.append(i)
        self.ndim_ = ndim
        pass

    def _update_CV(self, update):
        for c in range(len(update)):
            self.CV[c] = update[c]

    def change_val(self, commands):
        """Change an attribute of a spec object matching the input name.

        Parameters
        ----------
        commands : list
            Should be a list containing the object name, the attribute to
            change and the value you want.

        Returns
        -------
        boolean
            Returns a boolean, to check if the action was done or not.

        """
        object_id, action = commands[:2]
        whato = commands[2:]
        if len(whato) == 1:  # dictionary quickfix
            whato = commands[2]  # dictionary quickfix
        for theta in self.list_:
            if theta.name == object_id:
                setattr(theta, action, whato)
                return True
        return False

    def apply_changes_list(self, changes_list):
        """Apply a list of changes.

        Input is a list with the 'command' format for the change_val
        function. If the change is applied, it deletes that command from the
        input list.

        Parameters
        ----------
        changes_list : list
            List containing the commands to apply to the spec_list.

        """
        used = []
        for j in changes_list.keys():
            if self.change_val(changes_list[j]):
                print('\nFollowing condition has been applied: ',
                      changes_list[j])
                used.append(j)
        for j in used[::-1]:
            del changes_list[j]
        print('\n')
        pass

    def list(self, *call):
        """In-built function to call a list containing the same attribute for
        each spec object.

        Parameters
        ----------
        *call : type
            The attribute or attributes that you want to retrieve, in a list.

        Returns
        -------
        list
            List with the attributes for each spec object.
            If more than one attribute is requested in call, it returns a
            list of lists as the describes before. Well, arrays really.

        """
        if len(call) == 1:
            return sp.array([getattr(self.list_[i], call[0]) for i in range(len(self))])
        else:
            return sp.array([sp.array([getattr(self.list_[i], c) for i in range(len(self))]) for c in call])


class spec:
    """Metadata for each parameter.

    Spec object contains metadata corresponding to a parameter to fit with
    emcee.

    Attributes
    ----------
    name : string
        Unique identifier for your parameter.
    units : string
        Units that you want displayed in your plots. They DONT change anything
        else.
    prior : string
        String that calls a function with the same name in emperors_mirror.py.
        Alternatively it is a marker for fixed parameters.
    lims : list
        List containing the lower and upper boundary for the parameter.
    val : float
        Value for a fixed parameter. Default for the others is -inf.
    type : string
        A 'general' information from where this parameter is. As 'keplerian' or
        'gaussian process'
    args : any
        Any additional argument that is needed. Has no real use for now.

    """

    def __init__(self, name, units, prior, lims, val, type, args=[]):
        self.name = name
        self.units = units
        self.prior = prior
        self.lims = lims
        self.val = -sp.inf
        self.args = args
        self.type = type
        self.true_val = -sp.inf

        self.cv = False
        self.sigmas = {}
        self.other = []
        if self.type == 'keplerian':
            self.cv = True

    def __repr__(self):
        """Human readable representation."""
        try:
            rep = self.name + self.units
        except TypeError:
            rep = self.name + self.units[0]
        return rep

    def __str__(self):
        """Machine readable representation."""
        return self.name

    def tag(self):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        return self.name.split('_')[0]
    pass


class EMPIRE:
    def __init__(self, stardat, setup, file_type='rv_file'):
        """Contains all the data, functions and setup necesary for the run
        of the MCMC chain.

        Parameters
        ----------
        stardat : list
            Contains the paths to the data files.
        setup : list
            Goes as [ntemps, nwalkers, nsteps].
        file_type : str
            Fyle type, should be 'rv_file' or 'pm_file'.

        """
        emplib.ensure(len(stardat) >= 1,
                      'stardat has to contain at least 1 file ! !', fault)
        emplib.ensure(len(setup) == 3,
                      'setup has to be [ntemps, nwalkers, nsteps]', fault)

        #  Setup
        self.cores = multiprocessing.cpu_count()  # number of available threads
        self.setup = setup  # contains the ntemp, nwalkers and nsteps
        self.ntemps, self.nwalkers, self.nsteps = setup  # unpacks
        self.betas = None  # default temperatures if no input is given

        self.changes_list = {}  # list containing user-made parameter changes
        self.coordinator = sp.array([])  # coordinator, check class spec_list
        self.anticoor = sp.array([])  # anticoordinator, idem

        self.burn_out = self.nsteps // 2  # default lenght for burnin phase
        self.RV = False  # marker for the data
        self.PM = False  # marker for the data
        self.START = chrono.time()  # total time counter
        self.VINES = False  # jaja

        self.theta = spec_list()  # parameters metadata, initializes spec_list

        self.ld = {'uniform': 0,
                    'linear': 1,
                    'quadratic': 2,
                    'square-root': 2,
                    'logarithmic': 2,
                    'exponential': 2,
                    'power2': 2,
                    'nonlinear': 2
                    }  # dictionary with limb darkening dimentionality

        self.lk = {'Constant': 1,
                    'RealTerm':2,
                    'ComplexTerm':4,
                    'SHOTerm':3,
                    'Matern32Term':2,
                    'JitterTerm':1}

        self.CV = sp.array([(True, True, True) for _ in range(10)])
        self.kplan = 0
        self.fplan = 0


        ###  READING DATA  ###

        if file_type == 'rvpm_file':  # this will contain rv+pm
            self.rvfiles, self.pmfiles = stardat
            rvdat = emplib.read_data(self.rvfiles)
            pmdat = emplib.read_data(self.pmfiles, data_type='pm_file')
            # time, radial velocities, error and instrument flag
            self.all_data, self.all_data_pm = rvdat[0], pmdat[0]
            self.time, self.rv, self.err, self.ins = self.all_data
            self.time_pm, self.rv_pm, self.err_pm, self.ins_pm = pmdat[0]

            # star activity index and flag
            self.staract, self.starflag = rvdat[1], rvdat[2]
            self.staract_pm, self.starflag_pm = pmdat[1], pmdat[2]

            self.totcornum = rvdat[3]  # quantity if star activity is given
            self.totcornum_pm = pmdat[3]  # ?



            self.nins = len(self.rvfiles)  # number of instruments
            self.nins_pm = len(self.pmfiles)
            self.ndat = len(self.time)  # number of datapoints
            self.ndat_pm = len(self.time_pm)
            self.RV, self.PM = True, True  # setup data type marker


            # About the search parameters
            self.ACC = 0  # Acceleration polynomial order, default is 1, a line
            self.MOAV_STAR = 0  # Moving Average for the star activity

            self.ACC_pm = 0  # Acceleration order
            self.MOAV_STAR_pm = 0  #

            self.WN = True  # white noise, jitter fitting (dont touch) # DEL
            self.MOAV = sp.array([0 for _ in range(self.nins)])  # MOAV order for each instrument

            self.WN_pm = True  # jitter fitting (dont touch)
            self.MOAV_pm = sp.array([0, 0])  # MOAV order for each instrument

            self.starname = self.rvfiles[0].split('_')[0]
            self.ins_names = [self.rvfiles[i].split('_')[1].split('.')[0] for i in range(self.nins)]
            self.ins_names_pm = [self.pmfiles[i].split('_')[1].split('.')[0] for i in range(self.nins_pm)]


            # batmans assistance dictionaries
            self.batman_m = {}  # holds batmans model
            self.batman_p = {}  # holds batmans parameters
            self.batman_ld = []  # holds batmans limb darkening
            self.batman_ldn = []  # holds batmans ld total number of parameters

            # gp support dictionaries
            self.george_gp = {}  # not needed i guess  # DEL
            self.george_k = {}  # not needed i guess

            self.gaussian_processor = ''
            self.george_kernels = sp.array([])
            self.george_jitter = True

            self.celerite_kernels = sp.array([])
            self.celerite_jitter = False

            self.emperors_gp = []

            #  Correlate with rv's

            self.neo_logl = empmir.neo_logl_rvpm
            self.neo_logp = empmir.neo_logp_rvpm


        elif file_type == 'rv_file':  # for RV data
            self.rvfiles = stardat
            rvdat = emplib.read_data(stardat)
            # time, radial velocities, error and instrument flag
            self.all_data = rvdat[0]
            self.time, self.rv, self.err, self.ins = self.all_data
            # star activity index and flag
            self.staract, self.starflag = rvdat[1], rvdat[2]
            self.totcornum = rvdat[3]  # quantity if star activity is given
            self.nins = len(self.rvfiles)  # number of instruments
            self.ndat = len(self.time)  # number of datapoints
            self.RV = True  # setup data type marker


            # About the search parameters
            self.ACC = 1  # Acceleration polynomial order, default is 1, a line
            self.MOAV_STAR = 0  # Moving Average for the star activity

            self.WN = True  # white noise, jitter fitting (dont touch)
            self.MOAV = sp.array([0 for _ in range(self.nins)])  # MOAV order for each instrument

            # PM
            self.time_pm, self.rv_pm, self.err_pm, self.ins_pm = 0., 0., 0., 0.
            self.totcornum_pm = 0.

            self.starname = self.rvfiles[0].split('_')[0]
            self.ins_names = [stardat[i].split('_')[1].split('.')[0] for i in range(self.nins)]
            #from emperors_mirror import neo_logp_rv as neo_logp
            #from emperors_mirror import neo_logl_rv as neo_logl

            self.neo_logl = empmir.neo_logl_rv
            self.neo_logp = empmir.neo_logp_rv

        elif file_type == 'pm_file':
            self.pmfiles = stardat
            pmdat = emplib.read_data(stardat, data_type=file_type)
            self.all_data = pmdat[0]
            self.time_pm, self.rv_pm, self.err_pm, self.ins_pm = pmdat[0]
            self.all_data_pm = pmdat[0]
            # time, star activity index and flag
            self.staract_pm, self.starflag_pm = pmdat[1], pmdat[2]
            self.totcornum_pm = pmdat[3]  # ?

            self.nins_pm = len(self.pmfiles)
            self.ndat_pm = len(self.time_pm)
            self.PM = True


            self.params_pm = sp.array([])
            self.lenppm = len(self.params_pm)

            # About the search parameters
            self.ACC_pm = 1  # Acceleration order
            self.MOAV_STAR_pm = 0

            self.WN_pm = True  # jitter fitting (dont touch)
            self.MOAV_pm = sp.array([0, 0])  # MOAV order for each instrument

            self.batman_m = {}
            self.batman_p = {}
            self.batman_ld = []
            self.batman_ldn = []

            self.george_gp = {}  # not needed i guess
            self.george_k = {}  # not needed i guess

            self.gaussian_processor = ''
            self.george_kernels = sp.array([])
            self.george_jitter = True

            self.celerite_kernels = sp.array([])
            self.celerite_jitter = False

            self.emperors_gp = []

            #  Correlate with rv's
            self.time, self.rv, self.err, self.ins = 0., 0., 0., 0.
            self.totcornum = 0.


            self.starname = self.pmfiles[0].split('_')[0]
            self.ins_names = [stardat[i].split('_')[1].split('.')[0] for i in range(self.nins_pm)]

            self.neo_logl = empmir.neo_logl_pm
            self.neo_logp = empmir.neo_logp_pm


        else:
            emplib.ensure(False, 'Did you write the filetype correctly?', fault)
        #  Statistical Tools
        # inside chain comparison (smaller = stricter)
        self.bayes_factor = sp.log(150)
        self.model_comparison = 5  # between different k configurations
        self.BIC = 5  # Bayes Information Criteria factor
        self.AIC = 5  # Akaike Information Criteria factor

        #  Menudencies
        self.thin = 1  # thins the chain taking one every self.thin samples
        self.STARMASS = False  # if this is included, calculates extra stuff
        self.HILL = False  # Use Hill Stability Criteria as a prior too
        self.CHECK = False  # prints stuff, for developers use
        self.RAW = False  # saves the chain without the in-model cutoff
        self.MUSIC = True  # hehe

        # Plotting stuff
        self.INPLOT = False  # plots during the run
        self.draw_every_n = 1  # draws only 1 every draw_every_n samples
        self.PNG = True  # saved plots format. Short time, so and so quality
        self.PDF = False  #  saved plots format. Long time, high quality
        self.CORNER = True  # does a corner plot.
        self.HISTOGRAMS = True  # plots histograms
        self.breakFLAG = False  # not really sure
        self.NoWarnings = True  # Ignores all warnings
        self.ushallnotpass = True  # constrains k_n+1 according to k_n


        # EXTERMINATE  # DEL
        self.eccprior = 0.3
        self.jittprior = 5.0
        self.jittmean = 5.0

        # auxiliary for later
        self.sampler = 0.0
        ########################################

        pass

    def _theta_rv(self, limits, conditions, kplanets, from_k):
        names = ["Period", "Amplitude", "Phase", "Eccentricity", "Longitude"]
        if kplanets >= 2:
            names = [str(name) + '_' + str(kplanets) for name in names]

        units = [" [Days]", r" $[\frac{m}{s}]$", " $[rad]$", "",
                 r" $[\frac{rad}{s}]$"]
        priors = ['uniform', 'uniform', 'uniform', 'uniform', 'uniform']
        new = sp.array([])
        for i in range(5):
            t = spec(names[i], units[i], priors[i], limits[i],
                     -sp.inf, 'keplerian', args=conditions[i])
            new = sp.append(new, t)

        self.theta.list_ = sp.insert(self.theta.list_, (kplanets - 1) * 5, new)
        pass

    def _theta_ins(self, limits, conditions, nin, MOAV):
        units = [' $[\\frac{m}{s}]$', ' $[\\frac{m}{s}]$', ' [Days]', '']
        priors = ['normal', 'uniform', 'uniform', 'uniform']

        for n_ in range(nin):
            names = ['Jitter', 'Offset', 'MACoefficient', 'MATimescale']
            names = [str(name) + '_' + self.ins_names[n_] for name in names]
            new = sp.array([])
            # APPENDS JITTER AND OFFSET
            t = spec(names[0], units[0], priors[0], limits[0], -sp.inf, 'instrumental', args=[5.0, 5.0])
            new = sp.append(new, t)
            t = spec(names[1], units[1], priors[1], limits[1], -sp.inf, 'instrumental')
            new = sp.append(new, t)
            # APPENDS MOAV COEF AND TIMESCALE
            for j in range(2 * MOAV[n_]):
                names1 = names
                if MOAV[n_] > 1:
                    names1 = [str(name) + '_' + str(j // 2 + 1) for name in names]
                t = spec(names1[j % 2 + 2], units[j % 2 + 2], priors[j % 2 + 2],
                         limits[j % 2 + 2], -sp.inf, 'instrumental_moav')
                new = sp.append(new, t)
            self.theta.list_ = sp.append(self.theta.list_, new)
        pass

    def _theta_star(self, limits, conditions):
        name = 'Stellar Activity'
        new = sp.array([])
        sa_ins = self.starflag+1
        units = ''
        prior = 'uniform'

        for sa in range(self.totcornum):
            na = name + '_%i' %sa
            t = spec(na, units, prior, limits[sa], -sp.inf, 'stellar')
            self.theta.list_ = sp.append(self.theta.list_, t)
        pass

    def _theta_gen(self, limits, conditions):

        new = []
        names = ['Acceleration', 'MACoefficient_star', 'MATimescale_star']
        for i in range(self.ACC):
            if self.ACC == 1:
                aux = ''
            else:
                aux = '_%i' % i+1
            if i > 0:
                units = [r' $[\frac{m}{s^{%i}}]$' % (i+1)]
            else:
                units = [r' $[\frac{m}{s}]$']
            t = spec(names[0]+aux, units, 'uniform', limits[0], -sp.inf, 'general')
            new = sp.append(new, t)

        #limits = limits[1:]
        for j in range(2*self.MOAV_STAR):
            name_ = names[1+j%2]+'_'+str(j//2+1)  # in which moav of this ins
            unit_ = ['','[Days]']
            t = spec(name_, unit_[j%2], 'uniform', limits[j%2+1], -sp.inf, 'general')
            new = sp.append(new, t)
        if len(new) > 0:
            self.theta.list_ = sp.append(new, self.theta.list_)
        pass

    def _theta_photo(self, limits, conditions, kplanets, ldn):
        names = ['t0', 'Period', 'Planet Radius', 'SemiMajor Axis', 'Inclination',
                 'Eccentricity', 'Longitude']
        names_ld = ['coef1', 'coef2', 'coef3', 'coef4']
        if kplanets >= 2:
            names = [str(name) + '_' + str(kplanets) for name in names]
            names_ld = [str(name_ld) + '_' + str(kplanets)
                        for name_ld in names_ld]
        units = [" [Days]", " $[\\frac{m}{s}]$", " $[Stellar Radii]$", "Stellar Radii",
                 " $[rads]$", '', '$[rads]$']
        units_ld = ['', '', '', '']
        priors = ['uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform',
                  'uniform']
        priors_ld = ['uniform', 'uniform', 'uniform', 'uniform']
        new = sp.array([])
        # for parameters other than limb darkening
        for i in range(7):
            t = spec(names[i], units[i], priors[i],
                     limits[i], -sp.inf, 'photometric')
            new = sp.append(new, t)
        for l in range(ldn):
            t = spec(names_ld[l], units_ld[l], priors_ld[l], [-1., 1.], -sp.inf, 'photometric')
            new = sp.append(new, t)
        if kplanets == 1:
            self.theta.list_ = sp.append(new, self.theta.list_)
        else:
            self.theta.list_ = sp.insert(
                self.theta.list_, (kplanets - 1) * (7 + ldn), new)
        pass

    def _theta_photo_rvpm(self, limits, conditions, kplanets, ldn):
        names = ['t0', 'Period_j', 'Planet Radius', 'SemiMajor Axis', 'Inclination',
                 'Eccentricity_j', 'Longitude_j']
        names_ld = ['coef1', 'coef2', 'coef3', 'coef4']
        if kplanets >= 2:
            names = [str(name) + '_' + str(kplanets) for name in names]
            names_ld = [str(name_ld) + '_' + str(kplanets)
                        for name_ld in names_ld]
        units = [" [Days]", " $[\\frac{m}{s}]$", " $[Stellar Radii]$", "Stellar Radii",
                 " $[rads]$", '', '$[rads]$']
        units_ld = ['', '', '', '']
        priors = ['uniform', 'joined', 'uniform', 'uniform', 'uniform', 'joined',
                  'joined']
        priors_ld = ['uniform', 'uniform', 'uniform', 'uniform']
        new = sp.array([])
        # for parameters other than limb darkening
        for i in range(7):
            t = spec(names[i], units[i], priors[i],
                     limits[i], -sp.inf, 'photometric')
            new = sp.append(new, t)
        for l in range(ldn):
            t = spec(names_ld[l], units_ld[l], priors_ld[l], [-1., 1.], -sp.inf, 'photometric')
            new = sp.append(new, t)
        '''
        if kplanets == 1:
            self.theta.list_ = sp.append(self.theta.list_, new)
        else:
            self.theta.list_ = sp.insert(
                self.theta.list_, (kplanets - 1) * (7 + ldn), new)
        '''
        c = kplanets * 5 + (sp.sum(self.MOAV) + self.nins + self.MOAV_STAR ) * 2 + self.ACC
        self.theta.list_ = sp.insert(
                self.theta.list_, (kplanets - 1) * (7 + ldn) + c, new)
        pass

    def _theta_george_pm(self, limits, conditions, kplanets):
        names = sp.array(
            ['kernel%i' % kn for kn in range(len(self.george_kernels))])

        if self.george_jitter:
            t = spec('Jitter_pm', 'm/s', 'uniform',
                     [0., 10.], -sp.inf, 'georgian_wn')
            self.theta.list_ = sp.append(self.theta.list_, t)

        for kn in range(len(self.george_kernels)):
            for c in range(len(self.george_kernels[kn]) + 1):
                t = spec(names[kn] + '_' + str(c), '',
                         'uniform', limits, -sp.inf, 'georgian')
                self.theta.list_ = sp.append(self.theta.list_, t)

        pass

    def _theta_celerite_pm(self, limits, conditions, kplanets):
        names = sp.array(
            ['kernel%i' % kn for kn in range(len(self.celerite_kernels))])

        if self.celerite_jitter:
            t = spec('Jitter_pm', 'm/s', 'uniform', limits, -sp.inf, 'celeritian')
            self.theta.list_ = sp.append(self.theta.list_, t)

        for kn in range(len(self.celerite_kernels)):
            for km in range(len(self.celerite_kernels[kn])):
                for kt in range(self.lk[self.celerite_kernels[kn, km]]):
                    t = spec(names[kn] + '_' + str(km) + '_' + str(kt), '',
                            'uniform', limits, -sp.inf, 'celeritian')
                    self.theta.list_ = sp.append(self.theta.list_, t)
        #t = spec('Jitter', 'm/s', 'uniform', [0, 10], -sp.inf, 'celeritian')
        #self.theta.list_ = sp.append(self.theta.list_, t)
        pass

    def _theta_gen_pm(self, limits, conditions):
        new = []
        names = ['Acceleration_pm', 'MACoefficient_star_pm', 'MATimescale_star_pm']
        for i in range(self.ACC_pm):
            if self.ACC_pm == 1:
                aux = ''
            else:
                aux = '_%s' % str(i+1)
            units = [' $[\\frac{m}{s%s}]$' % str(i+1)]
            t = spec(names[0]+aux, units, 'uniform', limits[0], -sp.inf, 'general')
            new = sp.append(new, t)

        #limits = limits[1:]
        for j in range(2*self.MOAV_STAR_pm):
            name_ = names[1+j%2]+'_'+str(j//2+1)  # in which moav of this ins
            unit_ = ['','[Days]']
            t = spec(name_, unit_[j%2], 'uniform', limits[j%2+1], -sp.inf, 'general_pm')
            new = sp.append(new, t)
        if len(new) > 0:
            self.theta.list_ = sp.append(new, self.theta.list_)
        pass

    def _theta_joined(self, limits, conditions, ldn):
        names = ["Periodj", "Amplitudej", "Phasej", "Eccentricityj", "Longitudej"]
        units = [" [Days]", r" $[\frac{m}{s}]$", " $[rad]$", "",
                 r" $[\frac{rad}{s}]$"]
        names += ['t0j', 'Planet Radiusj', 'SemiMajor Axisj', 'Inclinationj']

        priors = ['joined', 'uniform', 'uniform', 'joined', 'joined', 'uniform',
                  'uniform', 'uniform', 'uniform']

        names_ld = ['coef1j', 'coef2j', 'coef3j', 'coef4j']
        priors_ld = ['uniform', 'uniform', 'uniform', 'uniform']
        units += [" [Days]", " $[Stellar Radii]$", "Stellar Radii", " $[rads]$"]
        if self.kplan >= 2:
            names = [str(name) + '_' + str(self.kplan) for name in names]

        new = sp.array([])
        for i in range(9):
            t = spec(names[i], units[i], priors[i], limits[i], -sp.inf, 'joint')
            new = sp.append(new, t)
        for l in range(ldn):
            t = spec(names_ld[l], '', priors_ld[l], [-1., 1.], -sp.inf, 'photometric')
            new = sp.append(new, t)

        if self.kplan == 1:
            self.theta.list_ = sp.append(new, self.theta.list_)
        else:
            self.theta.list_ = sp.insert(
                self.theta.list_, (self.kplan - 1) * (9 + ldn), new)
        pass

    def mklogfile(self, kplanets):
        '''
        BROKEN
        '''
        dayis = dt.date.today()  # This is for the folder name

        def ensure_dir(date='datalogs/' + self.starname + '/' + str(dayis.month) + '.' + str(dayis.day) + '.' + str(dayis.year)[2:]):
            if not os.path.exists(date):
                os.makedirs(date)
                return date
            else:
                if len(date.split('_')) == 2:
                    aux = int(date.split('_')[1]) + 1
                    date = date.split('_')[0] + '_' + str(aux)
                else:
                    date = date + '_1'
            return ensure_dir(date)

        def timer():
            timing = chrono.time() - self.START
            #insec = sp.array([604800, 86400, 3600, 60])
            weeks, rest0 = timing // 604800, timing % 604800
            days, rest1 = rest0 // 86400, rest0 % 86400
            hours, rest2 = rest1 // 3600, rest1 % 3600
            minutes, seconds = rest2 // 60, rest2 % 60
            if weeks == 0:
                if days == 0:
                    if hours == 0:
                        if minutes == 0:
                            return '%i seconds' % seconds
                        else:
                            return '%i minutes and %i seconds' % (minutes, seconds)
                    else:
                        return '%i hours, %i minutes and %i seconds' % (hours, minutes, seconds)
                else:
                    return '%i days, %i hours, %i minutes and %i seconds' % (days, hours, minutes, seconds)
            else:
                return '%i weeks, %i days, %i hours, %i minutes and %i seconds' % (weeks, days, hours, minutes, seconds)

        def mklogdat():
            G = 39.5
            days_in_year = 365.242199
            logdat = '\nStar Name                         : ' + self.starname
            if self.RV:
                for i in range(self.nins):
                    if i == 0:
                        logdat += '\nUsed datasets                     : ' + \
                            self.rvfiles[i]
                    else:
                        logdat += '\n                                  : ' + \
                            self.rvfiles[i]
            if self.PM:
                for i in range(self.nins_pm):
                    if i == 0:
                        logdat += '\nUsed datasets                     : ' + \
                            self.pmfiles[i]
                    else:
                        logdat += '\n                                  : ' + \
                            self.pmfiles[i]

            logdat += '\n--------------------------------------------------------------------'
            logdat += '\nThe sample sizes are        :    ' + \
                str(self.sample_sizes)
            logdat += '\nThe maximum posterior is    :    ' + \
                str(self.post_max)
            logdat += '\nThe BIC is                  :    ' + str(self.NEW_BIC)
            logdat += '\nThe AIC is                  :    ' + str(self.NEW_AIC)
            # logdat += '\nThe RMS is                  :    ' + str(sp.sum(residuals**2))  # get this # DEL
            '''
            logdat += '\nThe most probable chain values are as follows...'
            for t in self.theta.list_:
                logdat += '\n' + str(t.name) + \
                    str(t.units) + ':   ' + str(t.val)
            '''
            if self.RV:
                logdat += '\n------------------------------ RV DATA ------------------------------'
                logdat += '\nTemperatures, Walkers, Steps      : ' + \
                    str((self.ntemps, self.nwalkers, self.nsteps))
                logdat += '\nN Instruments, K planets, N data  : ' + \
                    str((self.nins, kplanets, self.ndat))
                logdat += '\nNumber of Dimensions              : ' + \
                    str(self.theta.ndim_)
                logdat += '\nN Moving Average                  : ' + str(self.MOAV)
                logdat += '\nBeta Detail                       : ' + \
                    str(self.betas)
                logdat += '\n--------------------------------------------------------------------'
            if self.PM:
                logdat += '\n------------------------------ PM DATA ------------------------------'
                logdat += '\nN Instruments, N signals, N data  : ' + \
                    str((self.nins_pm, self.kplan, self.ndat_pm))
                if kplanets > 0:
                    ndim_rv = 5 * kplanets + self.nins * 2 * \
                        (self.MOAV + 1) + self.ACC + self.totcornum
                    logdat += '\nNumber of Dimensions              : ' + \
                        str(ndim_rv + self.kplan * self.lenppm)
                else:
                    pass
                #logdat += '\nN Moving Average                  : '+str(self.MOAV_pm)
                #logdat += '\nBeta Detail                       : '+str(self.betas)
                logdat += '\n--------------------------------------------------------------------'

            logdat += '\nRunning Time                      : ' + timer()
            print(logdat)
            return logdat

        name = str(ensure_dir())
        #logdat = mklogdat()
        #sp.savetxt(name + '/log.dat', sp.array([logdat]), fmt='%100s')
        #sp.savetxt(name+'/residuals.dat', sp.c_[self.time, residuals])
        return name

    def MCMC(self, *args):
        if args:
            kplan, logl, logp = args

        # ndat = len(self.time)  # DEL
        ndim = self.theta.ndim_

        def starinfo():  # in this format so developers can wrap
            colors = ['red', 'green', 'blue', 'yellow',
                      'grey', 'magenta', 'cyan', 'white']
            c = sp.random.randint(0, 7)
            print(
                colored('\n    ###############################################', colors[c]))
            print(
                colored('    #                                             #', colors[c]))
            print(
                colored('    #                                             #', colors[c]))
            print(
                colored('    #                 E M P E R 0 R               #', colors[c]))
            print(
                colored('    #                                             #', colors[c]))
            print(
                colored('    #                                             #', colors[c]))
            print(
                colored('    ###############################################', colors[c]))
            print(colored('Exoplanet Mcmc Parallel tEmpering Radial vel0city fitteR',
                          colors[sp.random.randint(0, 7)]))
            logdat = '\n\nStar Name                         : ' + self.starname
            logdat += '\nTemperatures, Walkers, Steps      : ' + \
                str((self.ntemps, self.nwalkers, self.nsteps))
            if self.RV:
                logdat += '\nN Instruments, K planets, N data  : ' + \
                    str((self.nins, kplan, self.ndat))
                logdat += '\nN Moving Average per instrument   : ' + \
                    str(self.MOAV)
            if self.PM:
                logdat += '\nN Instruments, K planets, N data  : ' + \
                    str((self.nins_pm, kplan, self.ndat_pm))
                logdat += '\nN Moving Average per instrument   : ' + \
                    str(self.MOAV_pm)
                logdat += '\nN of data for Photometry          : ' + \
                    str(self.ndat_pm)
            logdat += '\nN Number of Dimensions            : ' + str(ndim)
            logdat += '\nBeta Detail                       : ' + \
                str(self.betas)
            logdat += '\n-----------------------------------------------------'
            print(logdat)
            pass

        starinfo()


        ###  SETUPS THE SAMPLER  ###
        if self.RV:
            self.sampler = PTSampler(self.ntemps, self.nwalkers, ndim, logl, logp,
                                     loglargs=[self.neo_logl, self.logl_params],
                                     logpargs=[self.neo_logp, self.logp_params],
                                     threads=self.cores, betas=self.betas)

        if self.PM:
            logl_params_aux = sp.array([self.time_pm, self.rv_pm, self.err_pm,
                                        self.ins_pm, kplan, self.nins_pm,
                                        self.batman_ldn, self.batman_m, self.batman_p,
                                        self.emperors_gp, self.gaussian_processor])

            logl_params = [self.theta.list_, self.anticoor, logl_params_aux]

            self.sampler = PTSampler(self.ntemps, self.nwalkers, ndim, logl, logp,
                                    loglargs=[self.neo_logl, self.logl_params],
                                    logpargs=[self.neo_logp, self.logp_params],
                                    threads=self.cores, betas=self.betas)


        print('\n --------------------- BURN IN --------------------- \n')

        pbar = tqdm(total=self.burn_out)
        for p, lnprob, lnlike in self.sampler.sample(self.pos0, iterations=self.burn_out):
            pbar.update(1)
            pass
        pbar.close()

        p0, lnprob0, lnlike0 = p, lnprob, lnlike

        print('\n ------------ Mean acceptance fraction ------------- \n')
        tab_h = ['Chain', 'Mean', 'Minimum', 'Maximum']
        tab_all = []
        for t in range(self.ntemps):
            maf = self.sampler.acceptance_fraction[t]
            tab_all.append([t, sp.mean(maf), sp.amin(maf), sp.amax(maf)])
        print(tabulate(tab_all, headers=tab_h))
        print('\n --------------------------------------------------- \n')


        self.sampler.reset()
        print('\n ---------------------- CHAIN ---------------------- \n')
        pbar = tqdm(total=self.nsteps)
        for p, lnprob, lnlike in self.sampler.sample(p0, lnprob0=lnprob0,
                                                     lnlike0=lnlike0,
                                                     iterations=self.nsteps,
                                                     thin=self.thin):
            pbar.update(1)
            pass
        pbar.close()


        emplib.ensure(self.sampler.chain.shape == (self.ntemps, self.nwalkers, self.nsteps / self.thin, ndim),
                      'something really weird happened', fault)

        print('\n ------------ Mean acceptance fraction ------------- \n')
        tab_h = ['Chain', 'Mean', 'Minimum', 'Maximum']
        tab_all = []
        for t in range(self.ntemps):
            maf = self.sampler.acceptance_fraction[t]
            tab_all.append([t, sp.mean(maf), sp.amin(maf), sp.amax(maf)])
        print(tabulate(tab_all, headers=tab_h))
        print('\n --------------------------------------------------- \n')


        pass


    def conquer(self, from_k, to_k, from_k_pm=0, to_k_pm=5, logl=logl, logp=logp):
        # 1 handle data
        # 2 set adecuate model
        # 3 generate values for said model, different step as this should allow configuration
        # 4 run chain
        # 5 get stats (and model posterior)
        # 6 compare, exit or next
        # 7 remodel prior, go back to step 2

        ##########################################
        # 1 Check inputs are properly input (haha)
        ##########################################
        emplib.ensure(self.cores >= 1, 'Cores is set to 0 ! !', fault)
        emplib.ensure(self.thin * self.draw_every_n < self.nsteps,
                      'You are thining way too hard ! !', fault)
        if self.betas is not None:
            emplib.ensure(len(self.betas) == self.ntemps,
                          'Betas array and ntemps dont match ! !', fault)

        if self.MUSIC:
            imperial.play()

        if self.NoWarnings:
            import warnings
            warnings.filterwarnings("ignore")

        ##########################################
        # 3 Set model to run (reads inputs)
        ##########################################
        #from also import Accumulator
        #prepo1 = Accumulator()
        #also = prepo1.also



        ##########################################
        # 3a Sets the boundaries of the EMPIRE
        ##########################################
        if self.RV:
            # for instruments in rv
            acc_lims = sp.array([-1., 1.])
            jitt_limiter = sp.amax(abs(self.rv))
            jl_ = 2 * jitt_limiter  # or 3?
            ol_ = jitt_limiter
            jit_lims = sp.array([0.0001, jl_])
            off_lims = sp.array([-ol_, ol_])

            # for the keplerian signals
            self.kplan = from_k
            sqrta, sqrte = jl_, 1.
            sqrta, sqrte = sqrta ** 0.5, sqrte ** 0.5
            free_lims = sp.array([sp.log(0.1), sp.log(
                max(self.time)), -sqrta, sqrta, -sqrta, sqrta, -sqrte, sqrte, -sqrte, sqrte])

            p_lims = sp.array([sp.log(0.1), sp.log(max(self.time))])
            a_lims = sp.array([-sqrta, sqrta])
            e_lims = sp.array([-sqrte, sqrte])
            free_lims_rv = [p_lims, a_lims, a_lims, e_lims, e_lims]

        if self.PM:
            # create limits for instruments
            acc_bnd = sp.array([-1., 1.])
            jitt_bounder = sp.amax(abs(self.rv_pm))
            jlpm_ = 2 * jitt_bounder
            olpm_ = jitt_bounder
            jitoff_bnd = sp.array([0.0001, jlpm_, -olpm_, olpm_])

            jit_lims_pm = sp.array([0.0001, jlpm_])
            off_lims_pm = sp.array([-olpm_, olpm_])

            # for the photometric signals
            self.kplan = from_k
            t0bnd = sp.array([min(self.time_pm), max(self.time_pm)])  # maybe +-10
            periodbnd = sp.array([0.1, max(self.time_pm)])
            prbnds = sp.array([0.00001, 1])
            smabnds = sp.array([0.00001, 1000])
            incbnds = sp.array([0., 360.])
            eccbnds = sp.array([0., 1])
            longbnds = sp.array([0., 360.])
            ldcbnds = sp.array([-1., 1.])

            free_lims_pm = [t0bnd, periodbnd, prbnds, smabnds, incbnds,
                            eccbnds, longbnds]

            # should add to ^ the ldcbnds

            pass

        if (self.RV and self.PM):  # Here goes the rvpm
            self.fplan = from_k_pm
            self.joined_signals = 5
            pass

        # if prepo1.none:
        #     raise Exception('Mark RV or PM')
        #     pass

        self.oldlogpost = -sp.inf

        ##########################################
        # 3b Initializes model params
        ##########################################

        if self.RV:
            # INITIALIZE GENERAL PARAMS
            # acc, moavcoef, moavtimescale (moav for star)
            gen_lims = sp.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 6.]])
            self._theta_gen(gen_lims, None)

            # INITIALIZE INSTRUMENT PARAMS

            # this lim values are experimental
            moav_coef_lims = sp.array([-0.3, 0.3])
            moav_time_lims = sp.array([0.1, 6.])
            ins_lims = [jit_lims, off_lims, moav_coef_lims, moav_time_lims]
            self._theta_ins(ins_lims, None, self.nins, self.MOAV)

            #sa_lims = sp.array([[-sp.amax(self.staract[x]), sp.amax(self.staract[x])] for x in range(self.totcornum)])
            sa_lims = sp.array([[-1., 1.] for i in range(self.totcornum)])
            self._theta_star(sa_lims, None)

        if self.PM:
            # INITIALIZE GENERAL PARAMS
            gen_lims_pm = sp.array([[-1., 1.], [-0.2, 0.2], [0.1, 6.]])
            self._theta_gen_pm(gen_lims_pm, None)
            # INITIALIZE INSTRUMENT PARAMS
            # INITIALIZE GEORGE
            # for n in range(len(self.george_kernels)):
            if self.gaussian_processor == 'george':
                ins_bnd = sp.array([0., 10.])
                self._theta_george_pm(ins_bnd, None, 0)
                try:  # put somewhere else # DEL
                    import george
                except ImportError:
                    raise ImportError('You don t have the package george installed.\
                                       Try pip install george.')
                # this is a general gp, not per instrument, so jitter is for staract
                self.george_k = empmir.neo_init_george(self.george_kernels)

                # always jitter?  # DEL
                # jitter is first one in the kernel
                if self.george_jitter:
                    self.george_gp = george.GP(self.george_k,
                                               white_noise=sp.log(0.1**2),
                                               fit_white_noise=True)
                else:
                    self.george_gp = george.GP(self.george_k)
                self.emperors_gp = self.george_gp
                # DEL combinar lo de abajo con el p0 aleatorio
                self.emperors_gp.compute(
                    self.time_pm, self.err_pm)  # DEL  que ondi esto

                #raise Exception('Debug')

            if self.gaussian_processor == 'celerite':
                ins_bnd = sp.array([-10, 10])
                self._theta_celerite_pm(ins_bnd, None, 0)
                try:  # put somewhere else # DEL
                    import celerite
                except ImportError:
                    raise ImportError('You don t have the package george installed.\
                                       Try pip install celerite.')
                self.celerite_k = empmir.neo_init_terms(self.celerite_kernels)
                if self.celerite_jitter:
                    self.celerite_k += empmir.neo_init_terms([['JitterTerm']])
                    # self.celerite_gp = celerite.GP(self.celerite_k,
                    #                                mean=0., fit_mean=False,
                    #                                white_noise=sp.log(0.1**2),
                    #                                fit_white_noise=True)

                self.celerite_gp = celerite.GP(self.celerite_k, mean=0.)
                self.emperors_gp = self.celerite_gp

                self.emperors_gp.compute(self.time_pm)

        self.first_run = True
        while self.kplan <= to_k:
            if self.kplan > 0:
                # INITIALIZE KEPLERIAN PARAMS
                if self.RV and self.PM:
                    '''
                    ax = [True, False, True, True, True, False, False]
                    rvpm_lims = sp.r_[sp.array(free_lims_rv), sp.array(free_lims_pm)[ax]]

                    self.batman_ldn.append(self.ld[self.batman_ld[self.kplan - 1]])
                    self._theta_joined(rvpm_lims, [],
                                       self.batman_ldn[self.kplan - 1])
                    if self.kplan == self.fplan:
                        conds_ = [[], [0.0001, jl_], [], [0, 1], []]
                    self.batman_m[self.kplan - 1], self.batman_p[self.kplan - 1] = empmir.neo_init_batman(
                        self.time_pm, self.batman_ld[self.kplan - 1], self.batman_ldn[self.kplan - 1])
                    '''
                    pass
                if self.RV:
                    conds_ = [[], [0.0000001, jl_], [], [0, 1], []]
                    if self.first_run and from_k > 1:
                        for i in sp.arange(from_k)+1:
                            self._theta_rv(free_lims_rv, conds_, i, from_k)
                        self.first_run = False
                    else:
                        self._theta_rv(free_lims_rv, conds_, self.kplan, from_k)
                    pass
                if self.PM:
                    self.batman_ldn.append(self.ld[self.batman_ld[self.kplan - 1]])
                    # INITIALIZE PHOTOMETRIC PARAMS
                    if self.RV:
                        self._theta_photo_rvpm(free_lims_pm, None, self.kplan,
                                          self.batman_ldn[self.kplan - 1])
                    else:
                        self._theta_photo(free_lims_pm, None, self.kplan,
                                          self.batman_ldn[self.kplan - 1])
                    # INITIALIZE BATMAN
                    self.batman_m[self.kplan - 1], self.batman_p[self.kplan - 1] = empmir.neo_init_batman(
                        self.time_pm, self.batman_ld[self.kplan - 1], self.batman_ldn[self.kplan - 1])
                    pass
                #else:
                #    raise Exception('Something really weird happened!!')

        ##########################################
        # 3c Apply user commands
        ##########################################

            self.theta.apply_changes_list(self.changes_list)


            if self.RV:
                self.theta.CV_ = sp.array([[True, True, True] for _ in range(self.kplan)])
                for j in range(len(self.theta.list_)):
                    # change prior and lims for amplitude and eccentricity
                    # if phase or w are fixed

                    # fixed amplitude or ecc
                    if (self.theta.list_[j].prior == 'fixed' and
                        self.theta.list_[j+1].prior != 'fixed'):

                        if self.theta.list_[j].tag() == 'Amplitude':
                            self.theta.list_[j].cv = False  # amp or ecc
                            self.theta.list_[j+1].cv = False  # phase or w
                            self.theta.list_[j+1].lims = [0., 2*sp.pi]  # phase or ecc
                            self.theta.CV_[j//5][1] = False  # no CV on amp-pha
                        if self.theta.list_[j].tag() == 'Eccentricity':
                            self.theta.list_[j].cv = False  # amp or ecc
                            self.theta.list_[j+1].cv = False  # phase or w
                            self.theta.list_[j+1].lims = [0., 2*sp.pi]  # phase or ecc
                            self.theta.CV_[j//5][2] = False  # no CV on ecc-w
                    # fixed pha or w
                    if (self.theta.list_[j].prior == 'fixed' and
                        self.theta.list_[j-1].prior != 'fixed'):

                        if self.theta.list_[j].tag() == 'Phase':
                            self.theta.list_[j].cv = False  # amp or ecc
                            self.theta.list_[j-1].cv = False  # phase or w
                            self.theta.list_[j-1].lims = [0.1, jl_]  # for amp
                            self.theta.CV_[j//5][1] = False  # no CV on amp-pha
                        if self.theta.list_[j].tag() == 'Longitude':
                            self.theta.list_[j].cv = False  # amp or ecc
                            self.theta.list_[j-1].cv = False  # phase or w
                            self.theta.list_[j-1].lims = [0., 1]  # for ecc
                            self.theta.CV_[j//5][2] = False  # no CV on ecc-w

            if self.PM:
                self.theta.CV_ = sp.array([[False, False, False] for _ in range(self.kplan)])

            if self.RV and self.PM:  # should be different but for speeds sake
                self.theta.CV_ = sp.array([[False, False, False] for _ in range(self.kplan)])
                for j in range(len(self.theta.list_)):
                    if self.theta.list_[j].tag() == 'Amplitude':
                        self.theta.list_[j].cv = False  # amp or ecc
                        self.theta.list_[j].lims = [0.1, jl_]  # for amp

                    if self.theta.list_[j].tag() == 'Eccentricity':
                        self.theta.list_[j].cv = False  # amp or ecc
                        self.theta.list_[j].lims = [0., 1]  # for ecc

                    if self.theta.list_[j].tag() == 'Phase':
                        self.theta.list_[j].cv = False  # amp or ecc
                        self.theta.list_[j].lims = [0., 2*sp.pi]  # phase or ecc

                    if self.theta.list_[j].tag() == 'Longitude':
                        self.theta.list_[j].cv = False  # amp or ecc
                        self.theta.list_[j].lims = [0., 2*sp.pi]  # phase or ecc


                sub_ind = ((sp.arange(self.kplan * 5) % 5) + 1) // 2
                if self.kplan > 0:
                    for j in range(self.kplan*5):
                        if self.theta.list_[j].cv == False:
                            self.theta.CV_[j//5][((j%5)+1)//2] = False

            self.theta._update_list_()
            #self.theta._update_CV(self.CV)  # should be before.... ctm
            # show the initialized params and priors, developers
            tab_all = []
            tab_h = ['Name', 'Prior', 'Limits', 'Change of Var','Value']
            for t in self.theta.list_:
                tab_one = []
                tab_one.append(t.name)
                tab_one.append(t.prior)
                tab_one.append(sp.around(t.lims, 5))
                tab_one.append(t.cv)
                tab_one.append(t.val)
                tab_all.append(tab_one)
            print('\n\n------------ Initial Setup ------------\n\n')
            print('__(developer s use)')
            print(tabulate(tab_all, headers=tab_h))

            ### COORDINATOR
            self.coordinator = self.theta.C
            self.anticoor = self.theta.A

        ##########################################
        # 3 generate p0
        ##########################################
            self.pos0 = emplib.neo_p0(
                self.setup, self.theta.list_, self.theta.ndim_, self.coordinator)
        ####################################
        # 4 Test p0 and run chain
        ####################################
            # rv and pm testing, reroll p0 if not, should be in 3 rlly...
            if self.RV:
                self.logl_params_aux = sp.array([self.time, self.rv, self.err,
                                            self.ins,self.staract, self.starflag,
                                            self.kplan, self.nins, self.MOAV,
                                            self.MOAV_STAR, self.totcornum,self.ACC])
                indexer = self.anticoor
            if self.PM:
                self.logl_params_aux = sp.array([self.time_pm, self.rv_pm, self.err_pm,
                                        self.ins_pm, self.kplan, self.nins_pm,
                                        self.batman_ldn, self.batman_m, self.batman_p,
                                        self.emperors_gp, self.gaussian_processor])
                indexer = self.anticoor
            if (self.RV and self.PM):
                self.logl_params_aux = sp.array([[self.time, self.rv, self.err,
                                        self.ins,self.staract, self.starflag,
                                        self.kplan, self.nins, self.MOAV,
                                        self.MOAV_STAR, self.totcornum,self.ACC],
                                        [self.time_pm, self.rv_pm, self.err_pm,
                                        self.ins_pm, self.kplan, self.nins_pm,
                                        self.batman_ldn, self.batman_m, self.batman_p,
                                        self.emperors_gp, self.gaussian_processor]])
                indexer = [self.anticoor, self.theta.B, self.theta.CV_]
            self.logl_params = [self.theta, indexer,
                                self.logl_params_aux]

            self.logp_params = [self.theta.list_, self.theta.ndim_, self.coordinator]

            if True:
                self.autodestruction = 0
                self.adc = 0
                self.bad_bunnies = []
                self.mad_hatter = 0
                for i in range(self.nwalkers):
                    self.a = self.neo_logp(self.pos0[0][i], self.logp_params, CHECK=True)
                    self.b = self.neo_logl(self.pos0[0][i], self.logl_params)
                    if self.a == -sp.inf:
                        self.adc += 1
                        self.bad_bunnies.append(i)
                        print('a')  # experimental feature # DEL
                    elif self.b == -sp.inf:
                        self.adc += 1
                        self.bad_bunnies.append(i)
                        print('b')  # experimental feature # DEL

                self.autodestruction = (self.nwalkers - self.adc) / self.nwalkers
                self.adc = 0
                print('\nInitial Position acceptance rate', self.autodestruction)
                while self.autodestruction <= 0.98:
                    print('Reinitializing walkers')
                    self.mad_hatter += 1
                    self.pos0 = emplib.neo_p0(self.setup, self.theta.list_, self.theta.ndim_, self.coordinator)
                    for i in range(self.nwalkers):
                        self.a = self.neo_logp(self.pos0[0][i], self.logp_params, CHECK=True)
                        self.b = self.neo_logl(self.pos0[0][i], self.logl_params)
                        if self.a == -sp.inf or self.b ==-sp.inf:
                            self.adc += 1
                    self.autodestruction = (self.nwalkers - self.adc) / self.nwalkers
                    print('\nInitial Position acceptance rate', self.autodestruction)
                    self.adc = 0
                    if self.mad_hatter == 10:
                        raise Exception('asdasd')

            print('kplan', self.kplan)
            self.MCMC(self.kplan, logl, logp)
            print('kplan', self.kplan)
        ###################################
        # 5 get stats (and model posterior)
        ###################################

            # posterior and chain handling

            chains = self.sampler.flatchain
            self.posteriors = sp.array(
                [self.sampler.lnprobability[i].reshape(-1) for i in range(self.ntemps)])
            self.post_max = sp.amax(self.posteriors[0])
            self.ajuste = chains[0][sp.argmax(self.posteriors[0])]
            self.like_max = self.neo_logl(self.ajuste, self.logl_params)
            self.prior_max = self.neo_logp(self.ajuste, [self.theta.list_, self.theta.ndim_, self.coordinator])
            # TOP OF THE POSTERIOR
            if self.RAW:
                self.cherry_chain = sp.array([chains[temp] for temp in sp.arange(self.ntemps)])
                self.cherry_post = sp.array([self.posteriors[temp] for temp in range(self.ntemps)])
                self.sigmas = sp.array([sp.std(self.cherry_chain[0][:, i]) for i in range(self.theta.ndim_)])
            else:
                cherry_locat = sp.array([max(self.posteriors[temp]) - self.posteriors[temp] < self.bayes_factor for temp in sp.arange(self.ntemps)])

                self.cherry_chain = sp.array([chains[temp][cherry_locat[temp]] for temp in sp.arange(self.ntemps)])
                self.cherry_post = sp.array([self.posteriors[temp][cherry_locat[temp]] for temp in range(self.ntemps)])
                self.sigmas = sp.array([sp.std(self.cherry_chain[0][:, i]) for i in range(self.theta.ndim_)])

            import copy
            if (self.RV and self.PM):
                self.cherry_chain_h = copy.deepcopy(self.cherry_chain)
                self.ajuste_h = self.cherry_chain_h[0][sp.argmax(self.cherry_post[0])]
                self.sigmas_h = sp.array([sp.std(self.cherry_chain_h[0][:, i]) for i in range(self.theta.ndim_)])

                for b in self.theta.B:
                    self.ajuste = sp.insert(self.ajuste, b[0], self.ajuste[b[1]])

                self.theta.list_[self.coordinator[i]].true_val = self.ajuste[i]
                self.theta.list_[self.coordinator[i]].val = self.ajuste_h[self.coordinator[i]]
                pass
            elif self.RV:  # henshin
                self.cherry_chain_h = empmir.henshin_hou(copy.deepcopy(self.cherry_chain), self.kplan, self.theta.CV_, self.theta.list('val'), self.anticoor)
                self.ajuste_h = self.cherry_chain_h[0][sp.argmax(self.cherry_post[0])]
                self.sigmas_h = sp.array([sp.std(self.cherry_chain_h[0][:, i]) for i in range(self.theta.ndim_)])

                for i in range(self.theta.ndim_):
                    self.theta.list_[self.coordinator[i]].true_val = self.ajuste[i]
                    self.theta.list_[self.coordinator[i]].val = self.ajuste_h[self.coordinator[i]]

            elif self.PM:
                ## will need changes after combined
                self.cherry_chain_h = copy.deepcopy(self.cherry_chain)
                self.ajuste_h = self.cherry_chain_h[0][sp.argmax(self.cherry_post[0])]
                self.sigmas_h = sp.array([sp.std(self.cherry_chain_h[0][:, i]) for i in range(self.theta.ndim_)])

                for i in range(self.theta.ndim_):
                    self.theta.list_[self.coordinator[i]].true_val = self.ajuste[i]
                    self.theta.list_[self.coordinator[i]].val = self.ajuste_h[i]


            self.sample_sizes = sp.array(
                [len(self.cherry_chain[i]) for i in range(self.ntemps)])


            # updates values in self.theta.list_ with best of emcee run

            # dis goes at the end
            print('\n\n--------------- Best Fit ---------------\n\n')
            print(tabulate(self.theta.list('name', 'val', 'prior').T, headers=['Name', 'Value', 'Prior']))


            #residuals = empmir.RV_residuals(ajuste, self.rv, self.time,
                         #self.ins, self.staract, self.starflag, kplan,
                         #self.nins, self.MOAV, self.totcornum, self.ACC)
            #alt_res = self.alt_results(cherry_chain[0], kplan)
            if self.MUSIC:
                thybiding.play()
        ##############################
        # 6 compare, exit or next
        ##############################

        #BIC & AIC
            if self.RV:
                setup_ = sp.hstack([self.setup, self.ACC, self.MOAV_STAR, self.MOAV])
                self.NEW_BIC = sp.log(self.ndat) * \
                    self.theta.ndim_ - 2 * self.like_max
                self.OLD_BIC = sp.log(self.ndat) * \
                    self.theta.ndim_ - 2 * self.oldlogpost
                self.NEW_AIC = 2 * self.theta.ndim_ - 2 * self.post_max
                self.OLD_AIC = 2 * - 2 * self.oldlogpost

                if self.VINES:  # saves chains, posteriors and log
                    self.saveplace = self.mklogfile(self.kplan)
                    emplib.instigator(
                                    setup_, self.theta,
                                    self.cherry_chain_h[:, :, self.coordinator],
                                    self.cherry_post,
                                    self.all_data, self.saveplace)
                    if self.INPLOT:
                        print('printing')
                        from emperors_canvas import CourtPainter
                        vangogh = CourtPainter(self.kplan, self.saveplace+'/', self.PDF, self.PNG)
                        try:
                            print('printing')
                        except:
                            raise Exception('debug')
                            pass
                        try:
                            vangogh.paint_timeseries()
                        except:
                            pass
                        try:
                            vangogh.paint_fold()
                        except:
                            pass
                        try:
                            vangogh.paint_chains()
                        except:
                            pass
                        try:
                            vangogh.paint_posteriors()
                        except:
                            pass
                        try:
                            vangogh.paint_histograms()
                        except:
                            pass

                        pass


            if self.PM:
                setup_ = sp.hstack([self.setup, 0, 0, 0])
                self.NEW_BIC = sp.log(self.ndat_pm) * \
                    self.theta.ndim_ - 2 * self.post_max
                self.OLD_BIC = sp.log(self.ndat_pm) * \
                    self.theta.ndim_ - 2 * self.oldlogpost
                self.NEW_AIC = 2 * self.theta.ndim_ - 2 * self.post_max
                self.OLD_AIC = 2 * - 2 * self.oldlogpost
                if self.VINES:  # saves chains, posteriors and log
                    self.saveplace = self.mklogfile(self.kplan)

                    emplib.instigator(
                                    setup_, self.theta,
                                    self.cherry_chain_h[:, :, self.coordinator],
                                    self.cherry_post,
                                    self.all_data, self.saveplace)
                    if self.INPLOT:
                        print('printing')
                        from emperors_canvas import CourtPainter
                        vangogh = CourtPainter(self.kplan, self.saveplace+'/', self.PDF, self.PNG)
                        try:
                            print('printing')
                        except:
                            raise Exception('debug')
                            pass
                        try:
                            vangogh.paint_timeseries()
                        except:
                            pass
                        try:
                            vangogh.paint_fold()
                        except:
                            pass
                        try:
                            vangogh.paint_chains()
                        except:
                            pass
                        try:
                            vangogh.paint_posteriors()
                        except:
                            pass
                        try:
                            vangogh.paint_histograms()
                        except:
                            pass

                        pass


            if self.MUSIC:
                thybiding.play()

            h = ['Criteria', 'This Run', 'Previous Run', 'Difference', 'Requirement', 'Condition']

            bic_c = self.OLD_BIC - self.NEW_BIC
            aic_c = self.OLD_AIC - self.NEW_AIC
            post_c = self.post_max - self.oldlogpost


            bic = ['BIC', self.NEW_BIC, self.OLD_BIC, bic_c, self.BIC, bool(bic_c > self.BIC)]
            aic = ['AIC', self.NEW_AIC, self.OLD_AIC, aic_c, self.AIC, bool(aic_c > self.AIC)]
            pos = ['Posterior', self.post_max, self.oldlogpost, post_c, self.model_comparison, bool(post_c>self.model_comparison)]
            print('\n\n\n------------- S T A T S -------------\n\n\n')
            print(tabulate([bic, aic, pos], headers = h))

            if post_c < self.model_comparison:
                print('\nBayes Factor of %.2f requirement not met ! !' %
                      self.model_comparison)
                # break
            self.oldlogpost = self.post_max

        ########################################
        # 7 remodel prior, go back to step 2
        ########################################

            if self.ushallnotpass:
                #self.constrain = [38.15, 61.85]
                self.constrain = [15.9, 84.1]
                if self.kplan > 0 and self.sigmas.all():
                    print('Priors remodeled successfully!')
                    for i in range(self.theta.ndim_):
                        __t = self.theta.list_[self.coordinator[i]]
                        if __t.type == 'keplerian':
                            __t.lims = sp.percentile(
                                self.cherry_chain[0][:, i], self.constrain)
                            #__t.prior = 'normal'
                            #__t.args = [self.ajuste[i], self.sigmas[i]]
                            pass

            self.first_run = False
            self.kplan += 1

        if self.MUSIC:  # end music
            technological_terror.play()
        pass

#

# stardat = sp.array(
#     ['GJ357_1_HARPS.dat', 'GJ357_2_UVES.dat', 'GJ357_3_KECK.vels'])

#stardat = sp.array(['LTT9779_harps.fvels', 'LTT9779_ESPRESSO.fvels'])

# stardat = sp.array(['TOI1047_CORALIE_sanz.dat', 'TOI1047_FEROS_sanz.dat'])
#rvfiles = sp.array(['synth_RV.vels'])

stardat = sp.array(['GJ876_LICK.vels', 'GJ876_KECK.vels'])
setup = sp.array([5, 150, 15000])
#setup = sp.array([2, 50, 1000])
em = EMPIRE(stardat, setup)
em.ACC = 1
em.MOAV = sp.array([1, 1])  # not needed
#setup = sp.array([5, 150, 15000])


####pmfiles = sp.array(['flux/transit_ground_r.flux'])
# pmfiles = sp.array(['synth_KHAN2.flux', 'synth_GENGHIS2.flux'])
# stardat = pmfiles
# em = EMPIRE(stardat, setup, file_type='pm_file')  # ais.empire

###rvpm
# rvfiles = sp.array(['synth_RV.vels'])
# pmfiles = sp.array(['synth_KHAN2.flux', 'synth_GENGHIS2.flux'])
# stardat = [rvfiles, pmfiles]
# em = EMPIRE(stardat, setup, file_type='rvpm_file')  # ais.empire
# em.ACC = 1
# em.MOAV = sp.array([0, 0])  # not needed
# em.ACC_pm = 0
# em.batman_ld = ['quadratic', 'quadratic']
# em.gaussian_processor = 'celerite'
# em.celerite_kernels = sp.array([['RealTerm']])
# em.celerite_jitter = True
# em.CV = sp.array([False, False, False])

#em.betas = None
#em.betas = sp.array([1.0, 0.55, 0.3025, 0.1663, 0.0915])
em.bayes_factor = 5

#em.MOAV_STAR = 0

#em.burn_out = 1

em.RAW = True  # no bayes cut
em.CORNER = False  # corner plot disabled as it takes some time to plot
em.ushallnotpass = True  # constrain for next run
em.VINES = True
em.INPLOT = True

# em.ACC_pm = 0
# em.batman_ld = ['quadratic', 'quadratic']
# em.gaussian_processor = 'celerite'
# em.celerite_kernels = sp.array([['RealTerm']])
# em.celerite_jitter = True

#
# #[[rt,rt]] = rt*rt
# #[[rt],[rt]] = rt+rt


#

em.MUSIC = False

# rv test gj876
if True:

    #em.changes_list = { 0:['Period', 'lims', [sp.log(3.515), sp.log(3.525)]]
    #                    }

    # GJ876 constrained around the signal
    '''
    em.changes_list = {0:['Period', 'lims', [4.11098843e+00, 4.11105404e+00]],
                       1:['Amplitude', 'lims', [205, 207]],
                       2:['Phase', 'lims', [5.64, 5.655]],
                       3:['Eccentricity', 'lims', [0., 0.001]],
                       4:['Longitude', 'lims', [0.0, 0.001]],
                       5:['Amplitude', 'cv', False],
                       6:['Phase', 'cv', False],
                       7:['Eccentricity', 'cv', False],
                       8:['Longitude', 'cv', False],
                       9:['Offset_LICK', 'prior', 'uniform'],
                       10:['Offset_LICK', 'lims', [-24.9, -24.7]],
                       11:['Offset_KECK', 'prior', 'uniform'],
                       12:['Offset_KECK', 'lims', [7.14, 7.15]],
                       13:['Jitter_LICK', 'prior', 'uniform'],
                       14:['Jitter_LICK', 'lims', [12.3, 12.4]],
                       15:['Jitter_KECK', 'prior', 'uniform'],
                       16:['Jitter_KECK', 'lims', [35, 36]]
                       }



    em.changes_list = {0:['Period', 'lims', [4.11098843e+00, 4.11105404e+00]],
                       1:['Amplitude', 'lims', [-1.01515928e+01, -6.33312469e+00]],
                       2:['Phase', 'lims', [1.00520806e+01,   1.35949748e+01]],
                       3:['Eccentricity', 'lims', [-1.24823254e-02,   2.27443388e-02]],
                       4:['Longitude', 'lims', [4.14811179e-02, 1.38568310e-01]],
                       5:['Period_2', 'lims', [3.40831451e+00, 3.40863545e+00]],
                       6:['Amplitude_2', 'lims', [-5.69294095e+00, 6.05896817e-02]],
                       7:['Phase_2', 'lims', [-9.33328013e+00, -8.07370401e+00]],
                       8:['Eccentricity_2', 'lims', [-2.48303912e-01,  -7.10641857e-02]],
                       9:['Longitude_2', 'lims', [3.47811764e-02,   1.79877743e-01]]
                       }
    '''

    em.conquer(0,3)
    pass

# pm test khan
if False:
    # for synth2_KHAN
    # # true params
    #t_ = [2458042.0, 3.52, 0.102, 8.06, 86.1, 0.0, 90., 0.1, 0.3]
    #t1_= [2458046.0, 7, 0.12, 12, 86.2, 0.0, 90., 0.1, 0.2, 0.4]
    em.changes_list = {0:['t0', 'lims', [2458041.98, 2458042.02]],
                       1:['Period', 'lims', [3.518, 3.522]],
                       2:['Planet Radius', 'lims', [0.1018, 0.1022]],
                       3:['SemiMajor Axis', 'lims', [8.058, 8.062]],
                       4:['Inclination', 'lims', [86.098, 86.102]],
                       5:['Eccentricity', 'lims', [0, 0.0001]],
                       6:['Longitude', 'lims', [89.9999, 90]],
                       7:['coef1', 'lims', [0.098,0.102]],
                       8:['coef2', 'lims', [0.298,0.302]],
                       9:['t0_2', 'lims', [2458045.98, 2458046.02]],
                       10:['Period_2', 'lims', [6.998, 7.002]],
                       11:['Planet Radius_2', 'lims', [0.1198, 0.1202]],
                       12:['SemiMajor Axis_2', 'lims', [11.998, 12.002]],
                       13:['Inclination_2', 'lims', [86.198, 86.202]],
                       14:['Eccentricity_2', 'lims', [0, 0.0001]],
                       15:['Longitude_2', 'lims', [89.9999, 90]],
                       16:['coef1_2', 'lims', [0.198,0.202]],
                       17:['coef2_2', 'lims', [0.398,0.402]]
                       }
    em.conquer(1, 2)
    pass

PLOT_PM = False

if PLOT_PM:
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 20,
            }
    x,y,y_error = em.time_pm, em.rv_pm, em.err_pm



    if PLOT_PM:
        import batman
        import george
        from george import kernels
        plt.subplots(figsize=(16,8))
        plt.grid(True)
        plt.xlim( (min(x)-0.01) , (max(x+0.01)))
        delt_y = max(y)-min(y)
        plt.ylim(sp.amin(y)- 0.003, 1.005)

        font = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 20,
                }

        # central black line
        plt.plot((x[0], x[-1]), (1., 1.), 'k', linewidth=4)
        # data
        plt.errorbar(x, y, yerr=y_error, fmt='b.', alpha=1/1.)

        theta__ = em.ajuste[:9*(em.kplan-1)]
        T0_r, P_r, r_r, sma_r, inc_r, ecc_r, w_r, c1_r, c2_r = em.ajuste[:9]
        theta_gp = em.ajuste[9*(em.kplan-1):]
        # if len(em.ajuste)>10:  # not relevant thou
        #     T0_r2, P_r2, r_r2, sma_r2, inc_r2, ecc_r2, w_r2, c1_r2, c2_r2=em.ajuste[9:18]
        #theta_gp__ = em.ajuste[9:]

        # model
        params__ = [x, em.kplan-1, em.batman_ldn, em.batman_m, em.batman_p]
        y_transit = empmir.neo_lightcurve(theta__, params__)
        plt.plot(x, y_transit, 'r',  linewidth=2)


        plt.ylabel('Normalized Flux', fontsize=15)
        plt.xlabel('JD', fontsize=15)
        plt.title(em.ins_names[0]+' data', fontsize=40)
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        plt.subplots_adjust(left=0.15)

        plt.show()

    if True:
        plt.subplots(figsize=(16,8))
        plt.grid(True)
        plt.xlim( (min(x)-0.01) , (max(x+0.01)))
        delt_y = max(y)-min(y)
        plt.ylim(sp.amin(y)- 0.003, 1.005)

        font = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 20,
                }

        # central black line
        plt.plot((x[0], x[-1]), (1., 1.), 'k', linewidth=4)
        # data
        plt.errorbar(x, y, yerr=y_error, fmt='b.', alpha=1/1.)
        samples = em.sampler.flatchain[0, :]
        for s in samples[sp.random.randint(len(samples), size=24), 9*(em.kplan-1):]:
            em.emperors_gp.set_parameter_vector(s)
            mu = em.emperors_gp.predict(y, x, return_cov=False)
            plt.plot(x, mu, color='orange', alpha=0.3)
        em.emperors_gp.set_parameter_vector(em.ajuste[9*(em.kplan-1):])
        mu0 = em.emperors_gp.predict(y,x,return_cov=False)
        plt.plot(x, mu0, color='red')
        plt.show()



    #M1, P1 = neo_init_batman(x2)




#

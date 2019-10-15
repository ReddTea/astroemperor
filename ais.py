# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*elif/ /^\s*def/ /^\s*class spec/
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# EMPEROR Exoplanet Mcmc Parallel tEmpering Radial velOcity fitteR
# version 0.572.-47/31,64 Pluto, Ceres
'''
Na fone Eyfelevoy bashni
S Ayfona selfi zayeboshim
A nakhuya zh yeshche nam nash voyazh?
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
            if priors[i] == 'fixed' or priors[i] == 'joined':
                ndim -= 1
                self.A.append(i)
            else:
                self.C.append(i)
        self.ndim_ = ndim
        pass

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
            return sp.array([getattr(self.list_[i], call[0]) for i in range(len(self.list_))])
        else:
            return sp.array([sp.array([getattr(self.list_[i], c) for i in range(len(self.list_))]) for c in call])


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

        self.cv = None
        self.sigmas = {}
        self.other = []

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

        ###  READING DATA  ###

        if False:  # this will contain rv+pm
            pass

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

        elif file_type == 'pm_file':
            self.pmfiles = stardat
            pmdat = emplib.read_data(stardat)
            self.time_pm, self.rv_pm, self.err_pm, self.ins_pm = pmdat[0]
            self.all_data_pm = pmdat[0]
            # time, star activity index and flag
            self.staract_pm, self.starflag_pm = pmdat[1], pmdat[2]
            self.totcornum_pm = pmdat[3]  # ?

            self.nins_pm = len(self.pmfiles)
            self.ndat_pm = len(self.time_pm)
            self.MOAV_pm = 0  # for flat model
            self.PM = True

            self.params_pm = sp.array([])
            self.lenppm = len(self.params_pm)

            # About the search parameters
            self.ACC_pm = 1  # Acceleration order
            self.WN_pm = True  # jitter fitting (dont touch)
            self.MOAV_pm = sp.array([0, 0])  # MOAV order for each instrument

            self.batman_m = {}
            self.batman_p = {}
            self.batman_ld = []
            self.batman_ldn = []

            self.george_gp = {}  # not needed i guess
            self.george_k = {}  # not needed i guess

            self.gaussian_processor = 'george'
            self.george_kernels = sp.array([])
            self.george_jitter = True

            self.celerite_kernels = sp.array([])
            self.celerite_jitter = True

            self.emperors_gp = 0
            #  Correlate with rv's
            self.time, self.rv, self.err, self.ins = 0., 0., 0., 0.
            self.totcornum = 0.
            #self.fsig = 5
            # self.f2k = None  # EXTERMINATE

            self.starname = self.pmfiles[0].split('_')[0]
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
        priors = ['uniform', 'uniform_spe_a',
                  'uniform_spe_b', 'uniform_spe_a', 'uniform_spe_b']
        new = sp.array([])
        for i in range(5):
            if priors[i] == 'uniform_spe_a':
                if names[i][:3] == 'Amp':
                    t = spec(names[i], units[i], priors[i], [
                             limits[2 * i], limits[2 * i + 1]], -sp.inf, 'keplerian', args=[0.0001, conditions[0]])
                if names[i][:3] == 'Ecc':
                    t = spec(names[i], units[i], priors[i], [
                             limits[2 * i], limits[2 * i + 1]], -sp.inf, 'keplerian', args=conditions[1])
            else:
                t = spec(names[i], units[i], priors[i], [
                         limits[2 * i], limits[2 * i + 1]], -sp.inf, 'keplerian')
            new = sp.append(new, t)
        if kplanets == from_k:
            self.theta.list_ = sp.append(new, self.theta.list_)
        else:
            self.theta.list_ = sp.insert(
                self.theta.list_, (kplanets - 1) * 5, new)
        pass

    def _theta_ins(self, limits, conditions, nin, MOAV):
        names = ['Jitter', 'Offset', 'MACoefficient_ins', 'MATimescale_ins']
        if nin > 0:
            names = [str(name) + '_' + str(nin + 1) for name in names]
        # print(names)
        units = [' $[\\frac{m}{s}]$', ' $[\\frac{m}{s}]$', ' [Days]', '']
        priors = ['normal', 'uniform', 'uniform', 'uniform']
        new = sp.array([])
        # APPENDS JITTER AND OFFSET
        for i in range(2):
            if i == 0:
                t = spec(names[i], units[i], priors[i], [
                         limits[2 * i], limits[2 * i + 1]], -sp.inf, 'instrumental', args=[5.0, 5.0])
                new = sp.append(new, t)
            else:
                t = spec(names[i], units[i], priors[i], [
                         limits[2 * i], limits[2 * i + 1]], -sp.inf, 'instrumental')
                new = sp.append(new, t)
        # APPENDS MOAV COEF AND TIMESCALE
        for j in range(2 * MOAV):
            # if MOAV > 1:
            names1 = [str(name) + '_' + str(j // 2 + 1)
                      for name in names]  # in which moav of this ins
            # else:
            #    names1 = names
            t = spec(names1[j % 2 + 2], units[j % 2 + 2], priors[j % 2 + 2],
                     [limits[(j + 2) * 2], limits[(j + 2) * 2 + 1]], -sp.inf, 'instrumental')
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
            na = name + ' %i' %sa
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
            unit_ = ['[Days]', '']
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
            t = spec(names[i], units[i], priors[i], [
                     limits[2 * i], limits[2 * i + 1]], -sp.inf, 'photometric')
            new = sp.append(new, t)
        for l in range(ldn):
            t = spec(names_ld[l], units_ld[l], priors_ld[l], [-1., 1.], -sp.inf, 'photometric')
            new = sp.append(new, t)
        if kplanets == 1:
            self.theta.list_ = sp.append(new, self.theta.list_)
        else:
            self.theta.list_ = sp.insert(
                self.theta.list_, (kplanets - 1) * (7 + limb_dark), new)
        pass

    def _theta_george_pm(self, limits, conditions, kplanets):
        names = sp.array(
            ['kernel%i' % kn for kn in range(len(self.george_kernels))])

        if self.george_jitter:
            t = spec('Jitter', 'm/s', 'uniform',
                     [0., 10.], -sp.inf, 'georgian_j')
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

        for kn in range(len(self.celerite_kernels)):
            for c in range(len(self.celerite_kernels[kn]) + 1):
                t = spec(names[kn] + '_' + str(c), '',
                         'uniform', limits, -sp.inf, 'celeritian')
                self.theta.list_ = sp.append(self.theta.list_, t)
        t = spec('Jitter', 'm/s', 'uniform', [0, 10], -sp.inf, 'celeritian')
        self.theta.list_ = sp.append(self.theta.list_, t)
        pass

    def _theta_gen_pm(self, limits, conditions):
        priors = 'uniform'
        new = []
        for i in range(self.ACC_pm):
            name = 'Acceleration'
            if self.ACC_pm == 1:
                aux = ''
            else:
                aux = '_%i' % i + 1
            units = [' $[\\frac{m}{s%i}]$' % (i + 1)]
            t = spec(name + aux, units, priors,
                     [limits[0], limits[1]], -sp.inf, 'general')
            new = sp.append(new, t)
        self.theta.list_ = sp.append(new, self.theta.list_)
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
            for i in range(self.nins):
                if i == 0:
                    logdat += '\nUsed datasets                     : ' + \
                        self.rvfiles[i]
                else:
                    logdat += '\n                                  : ' + \
                        self.rvfiles[i]
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
                    str((self.nins_pm, self.fsig, self.ndat_pm))
                if kplanets > 0:
                    ndim_rv = 5 * kplanets + self.nins * 2 * \
                        (self.MOAV + 1) + self.ACC + self.totcornum
                    logdat += '\nNumber of Dimensions              : ' + \
                        str(ndim_rv + self.fsig * self.lenppm)
                else:
                    pass
                #logdat += '\nN Moving Average                  : '+str(self.MOAV_pm)
                #logdat += '\nBeta Detail                       : '+str(self.betas)
                logdat += '\n--------------------------------------------------------------------'

            logdat += '\nRunning Time                      : ' + timer()
            print(logdat)
            return logdat

        name = str(ensure_dir())
        logdat = mklogdat()
        sp.savetxt(name + '/log.dat', sp.array([logdat]), fmt='%100s')
        #sp.savetxt(name+'/residuals.dat', sp.c_[self.time, residuals])
        return name

    def MCMC(self, *args):
        if args:
            pos0, kplan, sigmas_raw, logl, logp = args

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
        # '''
        #from emperors_library import logp_rv
        #print(str(self.PM), ndim, 'self.pm y ndim')  # PMPMPM  # DEL

        logp_params = [self.theta.list_, self.theta.ndim_, self.coordinator]

        ###  SETUPS THE SAMPLER  ###
        if self.RV:
            logl_params_aux = sp.array([self.time, self.rv, self.err, self.ins,
                                    self.staract, self.starflag, kplan, self.nins,
                                    self.MOAV, self.MOAV_STAR, self.totcornum,
                                    self.ACC, self.anticoor])  # anticoor here too? DEL

            logl_params = [self.theta.list_, self.anticoor, logl_params_aux]

            self.sampler = PTSampler(self.ntemps, self.nwalkers, ndim, logl, logp,
                                     loglargs=[
                                         empmir.neo_logl_rv, logl_params],
                                     logpargs=[
                                         empmir.neo_logp_rv, logp_params],
                                     threads=self.cores, betas=self.betas)

        if self.PM:
            logl_params_aux = sp.array([self.time_pm, self.rv_pm, self.err_pm,
                                        self.ins_pm, kplan, self.nins_pm,
                                        self.batman_ldn, self.batman_m, self.batman_p,
                                        self.emperors_gp, self.gaussian_processor])

            logl_params = [self.theta.list_, self.anticoor, logl_params_aux]

            self.sampler = PTSampler(self.ntemps, self.nwalkers, ndim, logl, logp,
                                     loglargs=[
                                         empmir.neo_logl_pm, logl_params],
                                     logpargs=[
                                         empmir.neo_logp_pm, logp_params],
                                     threads=self.cores, betas=self.betas)

        # RVPM THINGY

        print('\n --------------------- BURN IN --------------------- \n')

        pbar = tqdm(total=self.burn_out)
        for p, lnprob, lnlike in self.sampler.sample(pos0, iterations=self.burn_out):
            pbar.update(1)
            pass
        pbar.close()
        #raise Exception('debug')
        p0, lnprob0, lnlike0 = p, lnprob, lnlike
        print("\nMean acceptance fraction: {0:.3f}".format(
            sp.mean(self.sampler.acceptance_fraction)))
        emplib.ensure(sp.mean(self.sampler.acceptance_fraction) !=
                      0, 'Mean acceptance fraction = 0 ! ! !', fault)
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
        # '''

        emplib.ensure(self.sampler.chain.shape == (self.ntemps, self.nwalkers, self.nsteps / self.thin, ndim),
                      'something really weird happened', fault)
        print("\nMean acceptance fraction: {0:.3f}".format(
            sp.mean(self.sampler.acceptance_fraction)))

        pass


    def conquer(self, from_k, to_k, logl=logl, logp=logp, BOUND=sp.array([])):
        # 1 handle data
        # 2 set adecuate model
        # 3 generate values for said model, different step as this should allow configuration
        # 4 run chain
        # 5 get stats (and model posterior)
        # 6 compare, exit or next
        # 7 remodel prior, go back to step 2

        # 1 is currently being done upstairs (in __init__ i mean)
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

        # Here should be how to run! Where does it start? Full auto?

        from also import Accumulator
        prepo1 = Accumulator()
        also = prepo1.also

        if also(self.RV):
            # for instruments in rv
            acc_lims = sp.array([-1., 1.])
            jitt_limiter = sp.amax(abs(self.rv))
            jitt_lim = 2 * jitt_limiter  # or 3?
            offs_lim = jitt_limiter
            jitoff_lim = sp.array([0.0001, jitt_lim, -offs_lim, offs_lim])

            # for the keplerian signals
            kplan = from_k
            sqrta, sqrte = jitt_lim, 1.
            sqrta, sqrte = sqrta ** 0.5, sqrte ** 0.5
            free_lims = sp.array([sp.log(0.1), sp.log(
                3 * max(self.time)), -sqrta, sqrta, -sqrta, sqrta, -sqrte, sqrte, -sqrte, sqrte])

        if also(self.PM):
            # create limits for instruments
            acc_bnd = sp.array([-1., 1.])
            jitt_bounder = sp.amax(abs(self.rv_pm))
            jitt_bnd = 3 * jitt_bounder
            offs_bnd = jitt_bounder
            jitoff_bnd = sp.array([0.0001, jitt_bnd, -offs_bnd, offs_bnd])
            # for the photometric signals

            kplan = from_k
            t0bnd = sp.array(
                [min(self.time_pm), max(self.time_pm)])  # maybe +-10
            periodbnd = sp.array([0.1, 3 * max(self.time_pm)])
            prbnds = sp.array([0.00001, 1])
            smabnds = sp.array([0.00001, 1000])
            incbnds = sp.array([0., 360.])
            eccbnds = sp.array([0., 1])
            longbnds = sp.array([0., 360.])
            ldcbnds = sp.array([-1., 1.])
            free_lims_pm = sp.array([t0bnd, periodbnd, prbnds, smabnds, incbnds,
                                     eccbnds, longbnds]).reshape(-1)

            # should add to ^ the ldcbnds

            pass

        if also(self.RV and self.PM):  # Here goes the rvpm
            pass

        if prepo1.none:
            raise Exception('Mark RV or PM')
            pass

        # sigmas, sigmas_raw = sp.zeros(self._ndim), sp.zeros(self._ndim)  # should go in param object?
        pos0 = 0.
        self.oldlogpost = -sp.inf

        if self.RV:
        # INITIALIZE GENERAL PARAMS
            gen_lims = sp.array([[-1., 1.], [-0.2, 0.2], [0.1, 6.]])
            self._theta_gen(gen_lims, None)

            # INITIALIZE INSTRUMENT PARAMS

            for nin in range(self.nins):
                moav_lim = sp.array([(-0.2, 0.2, 0.1, 6.) for _ in range(self.MOAV[nin])]).reshape(-1)
                ins_lims = sp.append(jitoff_lim, moav_lim).reshape(-1)
                self._theta_ins(ins_lims, None, nin, self.MOAV[nin])


            sa_lims = sp.array([[-sp.amax(self.staract[x]), sp.amax(self.staract[x])] for x in range(self.totcornum)])
            self._theta_star(sa_lims, None)

        if self.PM:
            # INITIALIZE GENERAL PARAMS
            # INITIALIZE INSTRUMENT PARAMS
            # INITIALIZE GEORGE
            # for n in range(len(self.george_kernels)):
            if self.gaussian_processor == 'george':
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
                ins_bnd = sp.array([0., 10.])
                self._theta_george_pm(ins_bnd, None, 0)

            if self.gaussian_processor == 'celerite':
                import celerite
                self.celerite_k = empmir.neo_term(self.celerite_kernels)
                if self.celerite_jitter:
                    self.celerite_gp = celerite.GP(self.celerite_k,
                                                   mean=0., fit_mean=False,
                                                   white_noise=sp.log(0.1**2),
                                                   fit_white_noise=True)
                else:
                    self.celerite_gp = celerite.GP(self.celerite_k)
                self.emperors_gp = self.celerite_gp

                self.emperors_gp.compute(self.time_pm, self.err_pm)
                ins_bnd = sp.array([-10, 10])
                self._theta_celerite_pm(ins_bnd, None, 0)


        ##################################################
        self.first_run = True
        while kplan <= to_k:
            if kplan > 0:
                if self.RV:
                    # INITIALIZE KEPLERIAN PARAMS
                    ## ARREGLAR ESTO POR EL AMOR DE DIOS
                    if self.first_run and from_k>1:
                        for i in sp.arange(from_k)+1:
                            self._theta_rv(free_lims, [jitt_lim, [0, 1]], i, from_k)
                        self.first_run = False
                    else:
                        self._theta_rv(free_lims, [jitt_lim, [0, 1]], kplan, from_k)
                    pass
                if self.PM:
                    # INITIALIZE PHOTOMETRIC PARAMS
                    self.batman_ldn.append(self.ld[self.batman_ld[kplan - 1]])
                    self._theta_photo(free_lims_pm, None, kplan,
                                      self.batman_ldn[kplan - 1])
                    # INITIALIZE BATMAN
                    self.batman_m[kplan - 1], self.batman_p[kplan - 1] = empmir.neo_model_pm(
                        self.time_pm, self.batman_ld[kplan - 1], self.batman_ldn[kplan - 1])
                    pass

        # FINAL MODEL STEP, apply commands
            # '''

            self.theta.apply_changes_list(self.changes_list)

            if self.RV:
                self.cv = sp.array([[1, 1] for _ in range(kplan)])
                for j in range(len(self.theta.list_)):
                    # change prior and lims for amplitude and eccentricity
                    # if phase or w are fixed
                    if (self.theta.list_[j].prior == 'uniform_spe_a' and
                        self.theta.list_[j + 1].prior == 'fixed'):
                        self.theta.list_[j].prior = 'uniform'
                        if self.theta.list_[j].tag == 'Amplitude':
                            self.cv[(j + 1)//5][0] = 0
                            self.theta.list_[j].lims = [0.1, jitt_lim]
                        elif self.theta.list_[j].tag == 'Eccentricity':
                            self.cv[(j + 1)//5][1] = 0
                            self.theta.list_[j].lims = [0., 1]
                        self.theta.list_[j].lims = []

                    # if amplitude or ecc are fixed
                    if (self.theta.list_[j].prior == 'uniform_spe_b' and
                        self.theta.list_[j-1].prior == 'fixed'):  # amplitude fixed, so phase
                        self.theta.list_[j].prior = 'uniform'
                        self.theta.list_[j].lims = [0., 2*sp.pi]
                        if self.theta.list_[j].tag == 'Phase':
                            self.cv[(j+1)//5][0] = 0  # amplitude fixed
                        if self.theta.list_[j].tag == 'Longitude':
                            self.cv[(j+1)//5][1] = 0  # ecc fixed

            # show the initialized params and priors
            tab_all = []
            tab_h = ['Name', 'Prior', 'Limits', 'Value']
            for t in self.theta.list_:
                tab_one = []
                tab_one.append(t.name)
                tab_one.append(t.prior)
                tab_one.append(sp.around(t.lims, 5))
                tab_one.append(t.val)
                tab_all.append(tab_one)
            print('\n\n------------ Initial Setup ------------\n\n')
            print(tabulate(tab_all, headers=tab_h))

            self.theta._update_list_()
            ### COORDINATOR
            self.coordinator = self.theta.C
            self.anticoor = self.theta.A

        # 3 generate values for said model, different step as this should allow configuration
            self.pos0 = emplib.neo_p0(
                self.setup, self.theta.list_, self.theta.ndim_, self.coordinator)

        # 4 run chain
            if self.RV:
                from emperors_mirror import neo_logp_rv as neo_logp
                from emperors_mirror import neo_logl_rv as neo_logl

            if self.PM:
                from emperors_mirror import neo_logp_pm, neo_logl_pm
                logl_params = sp.array([self.time_pm, self.rv_pm, self.err_pm,
                                        self.ins_pm, kplan, self.nins_pm,
                                        self.batman_ldn, self.batman_m, self.batman_p,
                                        self.emperors_gp, self.gaussian_processor])

            # rv and pm testing, reroll p0 if not, should be in 3 rlly...
            self.autodestruction = 0
            self.adc = 0
            logl_params_aux = sp.array([self.time, self.rv, self.err, self.ins,
                                    self.staract, self.starflag, kplan, self.nins,
                                    self.MOAV, self.MOAV_STAR, self.totcornum,
                                    self.ACC, self.theta.A])  # anticoor here too? DEL

            logl_params = [self.theta.list_, self.theta.A, logl_params_aux]

            self.bad_bunnies = []
            self.mad_hatter = 0
            if self.RV:
                for i in range(self.nwalkers):
                    self.a = neo_logp(self.pos0[0][i], [self.theta.list_, self.theta.ndim_, self.coordinator])
                    self.b = neo_logl(self.pos0[0][i], logl_params)
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
                        self.a = neo_logp(self.pos0[0][i], [self.theta.list_, self.theta.ndim_, self.coordinator])
                        self.b = neo_logl(self.pos0[0][i], logl_params)
                        if self.a == -sp.inf or self.b ==-sp.inf:
                            self.adc += 1
                    self.autodestruction = (self.nwalkers - self.adc) / self.nwalkers
                    print('\nInitial Position acceptance rate', self.autodestruction)
                    self.adc = 0
                    if self.mad_hatter == 2:
                        raise Exception('asdasd')

            if self.PM:
                for i in range(self.nwalkers):
                    self.c = neo_logp_pm(self.pos0[0][i], [self.theta.list_, self.theta.ndim_, self.coordinator])
                    if self.c == -sp.inf:
                        self.adc += 1
                    self.autodestruction = (self.nwalkers - self.adc) / self.nwalkers
                    self.adc = 0
                print('autodestruction', self.autodestruction)

                while self.autodestruction <= 0.98:
                    print('Reinitializing walkers')
                    print('autodestruction', self.autodestruction)
                    self.pos0 = emplib.neo_p0(
                        self.setup, self.theta.list_, self.theta.ndim_, self.coordinator)
                    for i in range(self.nwalkers):
                        self.c = neo_logp_pm(self.pos0[0][i], [self.theta.list_, self.theta.ndim_, self.coordinator])
                        if self.c == -sp.inf:
                            self.adc += 1
                    self.autodestruction = (
                        self.nwalkers - self.adc) / self.nwalkers
                    self.adc = 0


            if self.PM:
                self.c = neo_logp_pm(
                    p, [self.theta.list_, self.theta.ndim_, self.coordinator])
                self.d = neo_logl_pm(
                    p, [self.theta.list_, self.anticoor, logl_params])

            # real chain
            sigmas, sigmas_raw = sp.zeros(
                self.theta.ndim_), sp.zeros(self.theta.ndim_)
            self.MCMC(self.pos0, kplan, sigmas_raw, logl, logp)

        ###################################
        # 5 get stats (and model posterior)
        ###################################

            # posterior and chain handling

            chains = self.sampler.flatchain
            self.posteriors = sp.array(
                [self.sampler.lnprobability[i].reshape(-1) for i in range(self.ntemps)])
            self.post_max = sp.amax(self.posteriors[0])
            self.ajuste = chains[0][sp.argmax(self.posteriors[0])]
            self.like_max = neo_logl(self.ajuste, logl_params)
            self.prior_max = neo_logp(self.ajuste, [self.theta.list_, self.theta.ndim_, self.coordinator])
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

            if True:  # henshin!!!!!!
                import copy
                self.cherry_chain_h = empmir.henshin_hou(copy.deepcopy(self.cherry_chain), kplan, self.cv, self.theta.list('val'), self.anticoor)
                self.ajuste_h = self.cherry_chain_h[0][sp.argmax(self.cherry_post[0])]
                self.sigmas_h = sp.array([sp.std(self.cherry_chain_h[0][:, i]) for i in range(self.theta.ndim_)])

            self.sample_sizes = sp.array(
                [len(self.cherry_chain[i]) for i in range(self.ntemps)])


            # updates values in self.theta.list_ with best of emcee run
            for i in range(self.theta.ndim_):
                # is ajuste_h
                self.theta.list_[self.coordinator[i]].val = self.ajuste_h[self.coordinator[i]]

            # dis goes at the end
            print('\n\n--------------- Best Fit ---------------\n\n')
            print(tabulate(self.theta.list('name', 'val', 'prior').T, headers=['Name', 'Value', 'Prior']))



            #residuals = empmir.RV_residuals(ajuste, self.rv, self.time,
                         #self.ins, self.staract, self.starflag, kplan,
                         #self.nins, self.MOAV, self.totcornum, self.ACC)
            #alt_res = self.alt_results(cherry_chain[0], kplan)
            if self.MUSIC:
                thybiding.play()

        # 6 compare, exit or next
            # BIC & AIC

            if self.RV:
                self.NEW_BIC = sp.log(self.ndat) * \
                    self.theta.ndim_ - 2 * self.like_max
                self.OLD_BIC = sp.log(self.ndat) * \
                    self.theta.ndim_ - 2 * self.oldlogpost
                self.NEW_AIC = 2 * self.theta.ndim_ - 2 * self.post_max
                self.OLD_AIC = 2 * - 2 * self.oldlogpost

                #self.NEW_DIC =
            if self.PM:
                self.NEW_BIC = sp.log(self.ndat_pm) * \
                    self.theta.ndim_ - 2 * self.post_max
                self.OLD_BIC = sp.log(self.ndat_pm) * \
                    self.theta.ndim_ - 2 * self.oldlogpost
                self.NEW_AIC = 2 * self.theta.ndim_ - 2 * self.post_max
                self.OLD_AIC = 2 * - 2 * self.oldlogpost

            if self.VINES:  # saves chains, posteriors and log
                self.saveplace = self.mklogfile(kplan)
                setup = sp.hstack(
                        [self.setup, self.ACC, self.MOAV_STAR, self.MOAV]
                        )
                emplib.instigator(
                                setup, self.theta,
                                self.cherry_chain_h[:, :, self.coordinator],
                                self.cherry_post,
                                self.all_data, self.saveplace
                                )

            if self.MUSIC:
                thybiding.play()

            if self.INPLOT:
                from emperors_canvas import CourtPainter
                vangogh = CourtPainter(kplan, self.saveplace+'/',
                                       self.PDF, self.PNG)
                vangogh.paint_chains()
                vangogh.paint_posteriors()
                vangogh.paint_timeseries()
                vangogh.paint_fold()

                pass

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

        # 7 remodel prior, go back to step 2

            #self.constrain = [38.15, 61.85]
            if self.ushallnotpass:
                self.constrain = [15.9, 84.1]
                print('FLAME OF UDUN')
                if kplan > 0:
                    for i in range(self.theta.ndim_):
                        __t = self.theta.list_[self.coordinator[i]]
                        if __t.type == 'keplerian':
                            if (__t.prior != 'fixed'
                                    and __t.type == 'keplerian'):
                                __t.lims = sp.percentile(
                                    self.cherry_chain[0][:, i], self.constrain)
                                __t.prior = 'normal'
                                __t.args = [self.ajuste[i], self.sigmas[i]]
                                pass


            kplan += 1

        if self.MUSIC:  # end music
            technological_terror.play()
        pass

#

# stardat = sp.array(
#     ['GJ357_1_HARPS.dat', 'GJ357_2_UVES.dat', 'GJ357_3_KECK.vels'])


stardat = sp.array(['GJ876_1_LICK.vels', 'GJ876_2_KECK.vels'])
# stardat = sp.array(['LTT9779_harps.fvels', 'LTT9779_ESPRESSO.fvels'])

#pmfiles = sp.array(['flux/transit_ground_r.flux'])
#pmfiles = sp.array(['flux/synth2.flux'])

#stardat = pmfiles
setup = sp.array([1, 60, 120])
# setup = sp.array([3, 120, 6000])
em = EMPIRE(stardat, setup)
# em = EMPIRE(stardat, setup, file_type='pm_file')  # ais.empire
em.CORNER = False  # corner plot disabled as it takes some time to plot
# array([1.0])  # beta factor for each temperature, None for automatic
em.betas = None
#em.betas = sp.array([1.0, 0.55, 0.3025, 0.1663, 0.0915])

# we actually run the chain from 0 to 2 signals
# em.RAW = True  # no bayes cut
em.bayes_factor = 5

em.ACC = 1
em.MOAV = sp.array([1, 1])  # not needed
# em.burn_out = 1
#em.MOAV_STAR = 1

em.VINES = True
em.ushallnotpass = False  # constrain for next run
em.INPLOT = True
#em.batman_ld = ['quadratic']
#em.gaussian_processor = 'george'
#em.gaussian_processor = 'celerite'

#em.george_kernels = sp.array([['Matern32Kernel']])
#em.george_jitter = False

#em.celerite_kernels = sp.array([['Matern32Term', 'RealTerm']])
#em.celerite_jitter = False

em.MUSIC = False

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


em.conquer(2, 2)

# from emperors_canvas import CourtPainter
# vangogh = CourtPainter(2, 'datalogs/GJ876/10.4.19/', False, True)
# vangogh.paint_chains()
# vangogh.paint_posteriors()
# vangogh.paint_histograms()
# vangogh.paint_timeseries()
# vangogh.paint_fold()
# vangogh.paint_corners()


if False:
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 20,
            }
    x,y,y_error = em.time_pm, em.rv_pm, em.err_pm

    T0_f, P_f, r_f, ka_f, kr_f = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*sp.percentile(em.sampler.flatchain[0], [16, 50, 84],
                                                    axis=0)))
    print('T0 = ' + str(T0_f))
    print('P = ' + str(P_f))
    print('r = ' + str(r_f))
    print('k a = ' + str(ka_f))
    print('k r = ' + str(kr_f))
    '''
    T0_f = (2456915.696655258, 0.0010478422045707703, 0.0018055629916489124)
    r_f = (0.06377535753897773, 0.007796892949455764, 0.00786433202933419)
    ka_f = (0.0278890718957462, 0.014619445296243011, 0.014502978882819123)
    kr_f = (0.22656735905414685, 0.23992662025386238, 0.16059694335564756)
    '''
    PLOT3 = False
    PLOT4 = True

    import batman
    import george
    from george import kernels

    if PLOT3:
        plt.subplots(figsize=(16,8))
        plt.grid(True)
        plt.xlim( (min(x)-0.01) , (max(x+0.01)))
        plt.ylim(0.99, 1.015)

        font = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 20,
                }

        #std = np.std(y[-5:])

        #plt.plot((x[0], x[-1]), (1.0 - std, 1.0 - std), 'k--', linewidth=2, alpha = 0.5)
        #plt.plot((x[0], x[-1]), (1.0 + std , 1. + std), 'k--', linewidth=2, alpha = 0.5)
        plt.plot((x[0], x[-1]), (1., 1.), 'k', linewidth=4)

        plt.errorbar(x, y, yerr=y_error, fmt='b.', alpha=1/1.)

        def transit_lightCurve(time, t0, radius, dist, P, inc):
            params = batman.TransitParams()
            params.t0 = t0                       #time of inferior conjunction
            params.per = P                       #orbital period in days
            params.rp = radius                   #planet radius (in units of stellar radii)
            params.a = dist                      #semi-major axis (in units of stellar radii)
            params.inc = inc                     #orbital inclination (in degrees)
            params.ecc = 0.                      #eccentricity
            params.w = 0.                        #longitude of periastron (in degrees)
            params.u = [0.1, 0.3]                #limb darkening coefficients [u1, u2]
            params.limb_dark = "quadratic"       #limb darkening model

            m = batman.TransitModel(params, time)    #initializes model
            flux = m.light_curve(params)          #calculates light curve

            return (flux)

        def Model(param, x):
            T0, r = param
            transit = transit_lightCurve(x, T0, r, 101.1576001138329, 24.73712, 89.912)
            return transit

        T0_f, r_f, k_a, k_r = em.ajuste
        y_transit = transit_lightCurve(x, T0_f, r_f, 101.1576001138329, 24.73712, 89.912)
        plt.plot(x, y_transit, 'r',  linewidth=2)

        #temp = x-x[0]
        #y_model = y_transit + a_f[0] + b_f[0]*temp + c_f[0]*temp*temp
        #plt.plot(x, y_model, 'g',  linewidth=2, alpha = 0.5)

        plt.ylabel('Normalized Flux', fontsize=15)
        plt.xlabel('JD', fontsize=15)
        plt.title('GROND in i band', fontsize=40)
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        plt.subplots_adjust(left=0.15)

        #plt.text(x[0]+0.2, 1.005, r'STD = '+str(np.around(std, decimals=4)), fontdict=font)

        r_t = str(np.around(r_f, decimals=4))
        #r_tp = str(np.around(r_f[1], decimals=4))
        #r_tm = str(np.around(r_f[2], decimals=4))
        plt.text(x[0]+0.06, 1.007, 'r = '+ r_t , fontdict=font)
        #plt.text(x[0]+0.10, 1.0075, '+ '+ r_tp, fontdict=font)
        #plt.text(x[0]+0.102, 1.0065, '-  '+ r_tm, fontdict=font)

        x2 = np.linspace(min(x), max(x), 1000)

        em.sampler.flatchain[0] = em.sampler.flatchain[0]
        if True:
            for s in em.sampler.flatchain[0][np.random.randint(len(em.sampler.flatchain[0]), size=24)]:
                radius = 10.**s[-1]
                gp = george.GP(s[-2]* kernels.Matern32Kernel(radius))
                gp.compute(x, y_error)
                m = gp.sample_conditional(y - Model(s[:-2], x), x2) + Model(s[:-2], x2)
                plt.plot(x2, m, '-', color="#4682b4", alpha=0.2)

        plt.show()

    x2 = np.linspace(min(x), max(x), 1200)
    #M1, P1 = neo_init_batman(x2)



    if PLOT4:
        plt.subplots(figsize=(12,6))
        plt.xlim( (min(x)-0.01) , (max(x+0.01)))
        plt.ylim(min(y)-0.01, 2-(min(y)-0.01))
        plt.plot((x[0], x[-1]), (1., 1.), 'k', linewidth=4)

        plt.errorbar(x, y, yerr=y_error, fmt='b.', alpha=1/1.)

        print('tag1')
        t_ = [2458042.0, 30.30, 0.1, 100.120, 90.3, 0.314, 120.120, 0.1, 0.3]
        t_1 = [T0_f, P_f, r_f, 100.120, 90.3, 0.314, 120.120, 0.1, 0.3]
        #t_1 = [T0_f, P_f, r_f, 111.111, 89.9, 0., 90., 0.1, 0.3]

        #M, P = em.batman_m[0], em.batman_p[0]
        M, P = neo_init_batman(x, 'quadratic', 2)
        print('tag2')
        paramis = [x, 1, [2], [M], [P]]
        y_transit = empmir.neo_lightcurve(t_, paramis)

        M1, P1 = neo_init_batman(x2, 'quadratic', 2)
        paramis1 = [x2, 1, [2], [M1], [P1]]
        y_transit1 = empmir.neo_lightcurve(t_, paramis1)
        #raise Exception('debug')
        print('tag3')
        plt.plot(x2, y_transit1, 'r',  linewidth=2)
        plt.ylabel('Normalized Flux', fontsize=22)
        plt.xlabel('JD', fontsize=22)
        plt.title('Synth Data 1', fontsize=40)
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        plt.subplots_adjust(left=0.15)

        r_t = str(np.around(r_f[0], decimals=4))
        r_tp = str(np.around(r_f[1], decimals=4))
        r_tm = str(np.around(r_f[2], decimals=4))
        #plt.text(x[0]+0.06, 1.007, 'r = '+ r_t , fontdict=font)
        #plt.text(x[0]+0.10, 1.0075, '+ '+ r_tp, fontdict=font)
        #plt.text(x[0]+0.102, 1.0065, '-  '+ r_tm, fontdict=font)
        print('tag4')
        if True:
            for s in em.sampler.flatchain[0][np.random.randint(len(em.sampler.flatchain[0]), size=24)]:
                radius = 10.**s[-1]
                t_g = sp.array([s[-2], radius])
                gp = george.GP(s[-2]* kernels.Matern32Kernel(radius))
                #G.set_parameter_vector(t_g)
                #gp = george.GP(0.2* kernels.Matern32Kernel(radius))
                gp.compute(x, y_error)

                t_ = [s[0],s[1],s[2], 111.111, 89.9, 0., 90., 0.1, 0.3]
                t_ = [s[0],s[1],s[2], 100.120, 90.3, 0.314, 120.120, 0.1, 0.3]
                paramis = [x, 1, em.batman_ldn, em.batman_m, em.batman_p]

                ldn = [2]
                M_, P_ = neo_init_batman(x2, 'quadratic', 2)
                paramis1 = [x2, 1, ldn, [M_], [P_]]

                m = gp.sample_conditional(y - empmir.neo_lightcurve(t_, paramis), x2) + empmir.neo_lightcurve(t_, paramis1)
                #raise Exception('debug')

                plt.plot(x2, m, '-', color="#4682b4", alpha=0.2)
                print('tag5')
            plt.show()
#

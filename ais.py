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
    """
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

        self.CV = []

    def len(self):
        return len(self.list_)

    def _update_list_(self):
        """Updates the ndim_, C and A attributes.

        -------
        function
            Really, it just updates.

        """
        self.A = []
        self.C = []
        ndim = self.len()
        priors = self.list('prior')
        for i in range(self.len()):
            if priors[i] == 'fixed' or priors[i]=='joined':
                ndim -= 1
                self.A.append(i)
                self.list_[i].lims = [sp.nan, sp.nan]
            else:
                self.C.append(i)
        self.ndim_ = ndim
        pass

    def change_val(self, commands):
        """Changes an attribute of a spec object matching the input name.

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
        """Input is a list with the 'command' format for the change_val
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
    """
    Spec object contains metadata corresponding to a parameter to fit with
    emcee.

    Parameters
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

    Attributes
    ----------
    Just the same as stated above.
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

    def identify(self):
        """Out goes the string containing the name and units. Used for the
        displays on terminal.

        Returns
        -------
        string
            Name and units of the spec.

        """
        return self.name + '    ' + self.units

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

        self.kplan = 0
        self.kplan_pm = 0


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
            self.celerite_jitter = True

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

        units = [" [Days]", " $[\\frac{m}{s}]$", " $[rad]$", "", " $[rads]$"]
        priors = ['uniform', 'uniform_spe_a',
                  'uniform_spe_b', 'uniform_spe_a', 'uniform_spe_b']
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
                         limits[j % 2 + 2], -sp.inf, 'instrumental')
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
                aux = '_%s' % str(i+1)
            units = [' $[\\frac{m}{s%s}]$' % str(i+1)]
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
        '''
        should change limits format to match _theta_rv
        '''
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

        for kn in range(len(self.celerite_kernels)):
            for c in range(len(self.celerite_kernels[kn]) + 1):
                t = spec(names[kn] + '_' + str(c), '',
                         'uniform', limits, -sp.inf, 'celeritian')
                self.theta.list_ = sp.append(self.theta.list_, t)
        t = spec('Jitter', 'm/s', 'uniform', [0, 10], -sp.inf, 'celeritian')
        self.theta.list_ = sp.append(self.theta.list_, t)
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


    def conquer(self, from_k, to_k, from_k_pm=0, to_k_pm=0, logl=logl, logp=logp):
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
        from also import Accumulator
        prepo1 = Accumulator()
        also = prepo1.also

        ##########################################
        # 3a Sets the boundaries of the EMPIRE
        ##########################################
        if also(self.RV):
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

        if also(self.PM):
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

            free_lims_pm = sp.array([t0bnd, periodbnd, prbnds, smabnds, incbnds,
                                     eccbnds, longbnds]).reshape(-1)

            # should add to ^ the ldcbnds

            pass

        if also(self.RV and self.PM):  # Here goes the rvpm
            pass

        if prepo1.none:
            raise Exception('Mark RV or PM')
            pass

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

            sa_lims = sp.array([[-sp.amax(self.staract[x]), sp.amax(self.staract[x])] for x in range(self.totcornum)])
            self._theta_star(sa_lims, None)

        if self.PM:
            # INITIALIZE GENERAL PARAMS
            gen_lims_pm = sp.array([[-1., 1.], [-0.2, 0.2], [0.1, 6.]])
            self._theta_gen_pm(gen_lims_pm, None)
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


        self.first_run = True
        while self.kplan <= to_k:
            if self.kplan > 0:
                # INITIALIZE KEPLERIAN PARAMS
                if self.RV:
                    conds_ = [[], [0.0001, jl_], [], [0, 1], []]
                    if self.first_run and from_k > 1:
                        for i in sp.arange(from_k)+1:
                            self._theta_rv(free_lims_rv, conds_, i, from_k)
                        self.first_run = False
                    else:
                        self._theta_rv(free_lims_rv, conds_, self.kplan, from_k)
                    pass
                if self.PM:
                    # INITIALIZE PHOTOMETRIC PARAMS
                    self.batman_ldn.append(self.ld[self.batman_ld[self.kplan - 1]])
                    self._theta_photo(free_lims_pm, None, self.kplan,
                                      self.batman_ldn[self.kplan - 1])
                    # INITIALIZE BATMAN
                    self.batman_m[self.kplan - 1], self.batman_p[self.kplan - 1] = empmir.neo_init_batman(
                        self.time_pm, self.batman_ld[self.kplan - 1], self.batman_ldn[self.kplan - 1])
                    pass

        ##########################################
        # 3c Apply user commands
        ##########################################

            self.theta.apply_changes_list(self.changes_list)

            if self.RV:
                self.theta.CV = sp.array([[True, True, True] for _ in range(self.kplan)])
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
                            self.theta.CV[j//5][1] = False  # no CV on amp-pha
                        if self.theta.list_[j].tag() == 'Eccentricity':
                            self.theta.list_[j].cv = False  # amp or ecc
                            self.theta.list_[j+1].cv = False  # phase or w
                            self.theta.list_[j+1].lims = [0., 2*sp.pi]  # phase or ecc
                            self.theta.CV[j//5][2] = False  # no CV on ecc-w
                    # fixed pha or w
                    if (self.theta.list_[j].prior == 'fixed' and
                        self.theta.list_[j-1].prior != 'fixed'):

                        if self.theta.list_[j].tag() == 'Phase':
                            self.theta.list_[j].cv = False  # amp or ecc
                            self.theta.list_[j-1].cv = False  # phase or w
                            self.theta.list_[j-1].lims = [0.1, jl_]  # for amp
                            self.theta.CV[j//5][1] = False  # no CV on amp-pha
                        if self.theta.list_[j].tag() == 'Longitude':
                            self.theta.list_[j].cv = False  # amp or ecc
                            self.theta.list_[j-1].cv = False  # phase or w
                            self.theta.list_[j-1].lims = [0., 1]  # for ecc
                            self.theta.CV[j//5][2] = False  # no CV on ecc-w

                    '''

                    # if amplitude or ecc are fixed
                    if (self.theta.list_[j].prior == 'uniform_spe_b' and
                        self.theta.list_[j-1].prior == 'fixed'):  # amplitude fixed, so phase
                        self.theta.list_[j].prior = 'uniform'
                        self.theta.list_[j].cv = False
                        self.theta.list_[j-1].cv = False

                        self.theta.list_[j].lims = [0., 2*sp.pi]  # for both
                        if self.theta.list_[j].tag() == 'Phase':
                            self.theta.CV[j//5][1] = False  # amplitude fixed
                        if self.theta.list_[j].tag() == 'Longitude':
                            self.theta.CV[j//5][2] = False  # ecc fixed
                    '''
            self.theta._update_list_()
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
            if self.PM:
                self.logl_params_aux = sp.array([self.time_pm, self.rv_pm, self.err_pm,
                                        self.ins_pm, self.kplan, self.nins_pm,
                                        self.batman_ldn, self.batman_m, self.batman_p,
                                        self.emperors_gp, self.gaussian_processor])

            self.logl_params = [self.theta.list_, self.anticoor,
                                self.logl_params_aux]

            self.logp_params = [self.theta.list_, self.theta.ndim_, self.coordinator]

            if True:
                self.autodestruction = 0
                self.adc = 0
                self.bad_bunnies = []
                self.mad_hatter = 0
                for i in range(self.nwalkers):
                    self.a = self.neo_logp(self.pos0[0][i], self.logp_params)
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
                        self.a = self.neo_logp(self.pos0[0][i], [self.theta.list_, self.theta.ndim_, self.coordinator])
                        self.b = self.neo_logl(self.pos0[0][i], self.logl_params)
                        if self.a == -sp.inf or self.b ==-sp.inf:
                            self.adc += 1
                    self.autodestruction = (self.nwalkers - self.adc) / self.nwalkers
                    print('\nInitial Position acceptance rate', self.autodestruction)
                    self.adc = 0
                    if self.mad_hatter == 2:
                        raise Exception('asdasd')

            self.MCMC(self.kplan, logl, logp)

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

            if self.RV:  # henshin
                import copy
                self.cherry_chain_h = empmir.henshin_hou(copy.deepcopy(self.cherry_chain), self.kplan, self.theta.CV, self.theta.list('val'), self.anticoor)
                self.ajuste_h = self.cherry_chain_h[0][sp.argmax(self.cherry_post[0])]
                self.sigmas_h = sp.array([sp.std(self.cherry_chain_h[0][:, i]) for i in range(self.theta.ndim_)])

                for i in range(self.theta.ndim_):
                    self.theta.list_[self.coordinator[i]].true_val = self.ajuste[i]
                    self.theta.list_[self.coordinator[i]].val = self.ajuste_h[self.coordinator[i]]

            if self.PM:
                for i in range(self.theta.ndim_):
                    self.theta.list_[self.coordinator[i]].true_val = self.ajuste[i]
                    self.theta.list_[self.coordinator[i]].val = self.ajuste[self.coordinator[i]]




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
                self.NEW_BIC = sp.log(self.ndat) * \
                    self.theta.ndim_ - 2 * self.like_max
                self.OLD_BIC = sp.log(self.ndat) * \
                    self.theta.ndim_ - 2 * self.oldlogpost
                self.NEW_AIC = 2 * self.theta.ndim_ - 2 * self.post_max
                self.OLD_AIC = 2 * - 2 * self.oldlogpost

                if self.VINES:  # saves chains, posteriors and log
                    self.saveplace = self.mklogfile(self.kplan)
                    emplib.instigator(self.setup, self.theta, self.cherry_chain_h,
                                      self.cherry_post, self.all_data, self.saveplace)
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
                self.NEW_BIC = sp.log(self.ndat_pm) * \
                    self.theta.ndim_ - 2 * self.post_max
                self.OLD_BIC = sp.log(self.ndat_pm) * \
                    self.theta.ndim_ - 2 * self.oldlogpost
                self.NEW_AIC = 2 * self.theta.ndim_ - 2 * self.post_max
                self.OLD_AIC = 2 * - 2 * self.oldlogpost
                if self.VINES:  # saves chains, posteriors and log
                    self.saveplace = self.mklogfile(self.kplan)
                    emplib.instigator(self.setup, self.theta, self.cherry_chain,
                                      self.cherry_post, self.all_data, self.saveplace)
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
                            __t.prior = 'normal'
                            __t.args = [self.ajuste[i], self.sigmas[i]]
                            pass

            self.first_run = False
            self.kplan += 1

        if self.MUSIC:  # end music
            technological_terror.play()
        pass

#

#stardat = sp.array(['GJ357_1_HARPS.dat', 'GJ357_2_UVES.dat', 'GJ357_3_KECK.vels'])

stardat = sp.array(['LTT9779_harps.fvels', 'LTT9779_ESPRESSO.fvels'])
stardat = sp.array(['GJ876_LICK.vels', 'GJ876_KECK.vels'])

setup = sp.array([2, 60, 120])
#setup = sp.array([5, 300, 10000])
#em = EMPIRE(stardat, setup)

#pmfiles = sp.array(['flux/transit_ground_r.flux'])
pmfiles = sp.array(['synth_KHAN2.flux', 'synth_GENGHIS2'])
#stardat = pmfiles
em = EMPIRE(stardat, setup, file_type='pm_file')  # ais.empire

#em.betas = None
#em.betas = sp.array([1.0, 0.55, 0.3025, 0.1663, 0.0915])
em.bayes_factor = 5
#em.ACC = 1
#em.MOAV = sp.array([1, 1])  # not needed

#em.burn_out = 1
#em.MOAV_STAR = 2

em.RAW = True  # no bayes cut
em.CORNER = False  # corner plot disabled as it takes some time to plot
em.VINES = False
em.ushallnotpass = False  # constrain for next run
em.INPLOT = True


em.ACC_pm = 0
em.batman_ld = ['quadratic']
#em.gaussian_processor = 'george'
#em.gaussian_processor = 'celerite'

#em.george_kernels = sp.array([['Matern32Kernel']])
#em.george_jitter = False

#ignore this
PLOT_PM = True
PLOT_PM1 = True
#em.celerite_kernels = sp.array([['Matern32Term', 'RealTerm']])
#em.celerite_jitter = False

em.MUSIC = False
if True:
    '''
    em.changes_list = {0:['Period', 'lims', [4.11098843e+00, 4.11105404e+00]],
                       1:['Period_2', 'lims', [3.40831451e+00, 3.40863545e+00]]
                       }
    '''
    # GJ876 constrained around the signal
    '''
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
# for synth2_KHAN
# # true params
# t_ = [2458042.0, 3.3, 0.015, 15., 89.8, 0.0, 90., 0.1, 0.3]
# t1_= [2458046.0, 4.5, 0.02, 20, 89.8, 0.0, 90., 0.2, 0.4]
'''
em.changes_list = {0:['t0', 'lims', [2458041.9, 2458042.1]],
                   1:['Period', 'lims', [3.29, 3.31]],
                   2:['Planet Radius', 'lims', [0.014, 0.016],],
                   3:['SemiMajor Axis', 'lims', [14.9, 15.1]],
                   4:['Inclination', 'lims', [89.7, 89.9]],
                   5:['Eccentricity', 'prior', 'fixed'],
                   6:['Eccentricity', 'val', 0],
                   7:['Longitude', 'prior', 'fixed',
                   8:['Longitude', 'val', '90'],
                   9:['coef1', 'lims', [0.09,0.11]],
                   10:['coef2', 'lims', [0.29,0.31]],
                    }
'''
em.conquer(1, 1)




if PLOT_PM:
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 20,
            }
    x,y,y_error = em.time_pm, em.rv_pm, em.err_pm

    T0_f, P_f, r_f, sma_f, inc_f, ecc_f, w_f, c1_f, c2_f = map(
        lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*sp.percentile(
            em.sampler.flatchain[0], [16, 50, 84], axis=0)))



    if PLOT_PM1:
        import batman
        import george
        from george import kernels
        plt.subplots(figsize=(16,8))
        plt.grid(True)
        plt.xlim( (min(x)-0.01) , (max(x+0.01)))
        delt_y = max(y)-min(y)
        plt.ylim(sp.mean(y)-delt_y*1.1, sp.mean(y)+delt_y*0.9)

        font = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 20,
                }

        # central black line
        plt.plot((x[0], x[-1]), (1., 1.), 'k', linewidth=4)
        # data
        plt.errorbar(x, y, yerr=y_error, fmt='b.', alpha=1/1.)

        T0_r, P_r, r_r, sma_r, inc_r, ecc_r, w_r, c1_r, c2_r = em.ajuste
        theta__ = em.ajuste[:9]
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


        #r_t = str(np.around(r_f, decimals=4))
        #r_tp = str(np.around(r_f[1], decimals=4))
        #r_tm = str(np.around(r_f[2], decimals=4))
        #plt.text(x[0]+0.06, 1.007, 'r = '+ r_t , fontdict=font)
        #plt.text(x[0]+0.10, 1.0075, '+ '+ r_tp, fontdict=font)
        #plt.text(x[0]+0.102, 1.0065, '-  '+ r_tm, fontdict=font)

        x2 = np.linspace(min(x), max(x), 1000)
        # gps
        params2__ = [x2, em.kplan-1, em.batman_ldn, em.batman_m, em.batman_p]
        '''
        if True:  # dont use this
            for s in em.sampler.flatchain[0][np.random.randint(len(em.sampler.flatchain[0]), size=24)]:
                radius = 10.**s[-1]  # k_r
                gp = george.GP(s[-2]* kernels.Matern32Kernel(radius))
                gp.compute(x, y_error)
                res = y - empmir.neo_lightcurve(s[:-2], params__)
                m = gp.sample_conditional(res, x2) + empmir.neo_lightcurve(s[:-2], params2__)
                plt.plot(x2, m, '-', color="#4682b4", alpha=0.2)
        '''
        plt.show()

    x2 = np.linspace(min(x), max(x), 1200)
    #M1, P1 = neo_init_batman(x2)




#

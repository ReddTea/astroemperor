# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*elif/ /^\s*def/
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# EMPEROR Exoplanet Mcmc Parallel tEmpering Radial velOcity fitteR
# version 0.572.-47/31,64

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

    try:  # put somewhere else # DEL
        import george
    except ImportError:
        raise ImportError("You don't have the package george installed.\
                           Try pip install george.")

    try:
        from tqdm import tqdm
    except ImportError:
        raise ImportError("You don't have the package tqdm installed.\
                           Try pip install tqdm.")
    try:
        from termcolor import colored
    except:
        print('You are missing the coolest package in Python!\
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
        fault = mixer.Sound('mediafiles/your_fault.wav')

    except Exception as e:
        imperial = False
        thybiding = False
        technological_terror = False
        alerted = False
        junk = False
        technical = False
        fault = False
        print('You are missing the most cool package in Python!\
               Try pip install pygame or set MUSIC=False')
else:
    print('You are missing some libraries :/')

# DUMMY FUNCTIONS


def logp(theta, func_logp, args):
    return func_logp(theta, args)


def logl(theta, func_logl, args):
    return func_logl(theta, args)


class spec_list:
    def __init__(self):
        self.list_ = sp.array([])
        self.ndim_ = 0
        self.gral_priors = sp.array([])

    def _update_list_(self):
        ndim = len(self.list_)
        for t in self.list_:
            if t.prior == 'fixed' or t.prior == 'joined':
                ndim -= 1
            else:
                pass
        self.ndim_ = ndim

    def change_val(self, commands):
        '''
        To change values only knowing the name!
        '''
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
        used = []
        for j in changes_list.keys():
            if self.change_val(changes_list[j]):
                print('Following condition has been applied: ',
                      changes_list[j])
                used.append(j)
        for j in used[::-1]:
            del changes_list[j]
        pass


class spec:
    def __init__(self, name, units, prior, lims, val, type, args=[]):
        self.name = name
        self.units = units
        self.prior = prior  # d[str(prior)]
        self.lims = lims
        self.val = -sp.inf
        self.args = args
        self.type = type

    def __prior(self, x, *args):
        return self.__prior(x, args)

    def identify(self):
        return self.name + '    ' + self.units

    def tag(self):
        return self.name.split('_')[0]
    pass


class EMPIRE:
    def __init__(self, stardat, setup, file_type='rv_file'):
        emplib.ensure(len(stardat) >= 1,
                      'stardat has to contain at least 1 file ! !', fault)
        emplib.ensure(len(setup) == 3,
                      'setup has to be [ntemps, nwalkers, nsteps]', fault)

        #  Setup
        self.cores = multiprocessing.cpu_count()
        self.setup = setup
        self.ntemps, self.nwalkers, self.nsteps = setup
        self.betas = None

        self.changes_list = {}
        self.coordinator = sp.array([])
        self.anticoor = sp.array([])

        self.burn_out = self.nsteps // 2
        self.RV = False
        self.PM = False

        self.START = chrono.time()
        self.VINES = True  # jaja
        # initialize flat model, this should go elsewhere
        # name  # units     # prior     # lims  # args
        self.theta = spec_list()
        self.ld = {
            'uniform': 0,
            'linear': 1,
            'quadratic': 2,
            'square-root': 2,
            'logarithmic': 2,
            'exponential': 2,
            'power2': 2,
            'nonlinear': 2
        }

        #  Reading data

        if False:  # this will contain rv+pm
            pass

        elif file_type == 'rv_file':
            self.rvfiles = stardat
            rvdat = emplib.read_data(stardat)
            # time, radial velocities, error and instrument flag
            self.time, self.rv, self.err, self.ins = rvdat[0]
            self.all_data = rvdat[0]
            # time, star activity index and flag
            self.staract, self.starflag = rvdat[1], rvdat[2]
            self.totcornum = rvdat[3]  # quantity if star activity indices

            self.nins = len(self.rvfiles)  # number of instruments autodefined
            self.ndat = len(self.time)  # number of datapoints
            self.RV = True

            # About the search parameters
            self.ACC = 1  # Acceleration order
            self.WN = True  # jitter fitting (dont touch)
            self.MOAV = sp.array([1, 1])  # MOAV order for each instrument

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
            raise Exception('You sure you wrote the filetype correctly mate?')
        #  Statistical Tools
        # inside chain comparison (smaller = stricter)
        self.bayes_factor = sp.log(150)
        self.model_comparison = 5  # between differet k configurations
        self.BIC = 5
        self.AIC = 5

        #  Menudencies
        self.thin = 1
        self.STARMASS = False
        self.HILL = False
        self.CHECK = False
        self.RAW = False
        self.MUSIC = True

        # Plotting stuff
        self.INPLOT = True
        self.draw_every_n = 1
        self.PNG = True
        self.PDF = False
        self.CORNER = True
        self.HISTOGRAMS = True
        self.breakFLAG = False

        # EXTERMINATE  # DEL
        self.eccprior = 0.3
        self.jittprior = 5.0
        self.jittmean = 5.0

        # auxiliary for later
        self.sampler = 0.0
        ########################################

        pass

    def _theta_rv(self, limits, conditions, kplanets):
        names = ["Period", "Amplitude", "Phase", "Eccentricity", "Longitude"]
        if kplanets >= 2:
            names = [str(name) + '_' + str(kplanets) for name in names]
        units = [" [Days]", " $[\\frac{m}{s}]$", " $[rad]$", "", " $[rads]$"]
        priors = ['uniform', 'uniform_spe_a',
                  'uniform_spe_b', 'uniform_spe_c', 'uniform_spe_d']
        new = sp.array([])
        for i in range(5):
            if (priors[i] == 'uniform_spe_a' or priors[i] == 'uniform_spe_c'):
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
        if kplanets == 1:
            self.theta.list_ = sp.append(new, self.theta.list_)
        else:
            self.theta.list_ = sp.insert(
                self.theta.list_, (kplanets - 1) * 5, new)
        pass

    def _theta_ins(self, limits, conditions, nin, MOAV):
        names = ['Jitter', 'Offset', 'MACoefficient', 'MATimescale']
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

    def _theta_star(self, limits, conditions, instruments):
        name = 'Stellar Activity'
        pass

    def _theta_gen(self, limits, conditions):
        priors = 'uniform'
        new = []
        for i in range(self.ACC):
            name = 'Acceleration'
            if self.ACC == 1:
                aux = ''
            else:
                aux = '_%i' % i + 1
            units = [' $[\\frac{m}{s%i}]$' % (i + 1)]
            t = spec(name + aux, units, priors,
                     [limits[0], limits[1]], -sp.inf, 'general')
            new = sp.append(new, t)
        self.theta.list_ = sp.append(new, self.theta.list_)
        pass

    def _theta_photo(self, limits, conditions, kplanets, limb_dark):
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
        for l in range(limb_dark):
            t = spec(names_ld[l], units_ld[l], priors_ld[l],
                     [-1., 1.], -sp.inf, 'photometric')
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
            logdat += '\nThe most probable chain values are as follows...'
            for t in self.theta.list_:
                logdat += '\n' + str(t.name) + \
                    str(t.units) + ':   ' + str(t.val)

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
                        (self.MOAV + 1) + self.PACC + 1 + self.totcornum
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

        def starinfo():
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
        print(str(self.PM), ndim, 'self.pm y ndim')  # PMPMPM

        logp_params = [self.theta.list_, self.theta.ndim_, self.coordinator]

        if self.RV:
            logl_params_aux = sp.array([self.time, self.rv, self.err, self.ins,
                                        self.staract, self.starflag, kplan, self.nins,
                                        self.MOAV, self.totcornum, self.ACC, self.anticoor])  # anticoor here too? DEL

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
        '''
        ln_post = self.sampler.lnprobability

        posteriors = sp.array([ln_post[i].reshape(-1) for i in range(self.ntemps)])
        chains = self.sampler.flatchain

        best_post = posteriors[0] == np.max(posteriors[0])
        #raise ImportError

        thetas_raw = sp.array([chains[i] for i in range(self.ntemps)])
        thetas_hen = sp.array([empmir.henshin(chains[i], kplan) for i in sp.arange(self.ntemps)])

        ajuste_hen = thetas_hen[0][best_post][0]
        ajuste_raw = thetas_raw[0][best_post][0]

        interesting_loc = sp.array([max(posteriors[temp]) - posteriors[temp] < self.bayes_factor for temp in sp.arange(self.ntemps)])
        interesting_thetas = sp.array([thetas_hen[temp][interesting_loc[temp]] for temp in sp.arange(self.ntemps)])
        thetas_hen = sp.array([thetas_hen[temp] for temp in sp.arange(self.ntemps)])
        interesting_thetas_raw = sp.array([thetas_raw[temp][interesting_loc[temp]] for temp in sp.arange(self.ntemps)])
        interesting_posts = sp.array([posteriors[temp][interesting_loc[temp]] for temp in range(self.ntemps)])
        sigmas = sp.array([ sp.std(interesting_thetas[0][:, i]) for i in range(ndim) ])
        sigmas_raw = sp.array([ sp.std(interesting_thetas_raw[0][:, i]) for i in range(ndim) ])
        #print('sigmas', sigmas)  # for testing
        #print('sigmas_raw', sigmas_raw)
        #print('mod_lims', boundaries)
        print('ALL RIGHT ALL RIGHT ALL RIGHT ALL RIGHT ALL RIGHT ALL RIGHT ALL RIGHT ALL RIGHT ')
        return thetas_raw, ajuste_raw, thetas_hen, ajuste_hen, p, lnprob, lnlike, posteriors, self.sampler.betas, interesting_thetas, interesting_posts, sigmas, sigmas_raw
        '''

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

        # Here should be how to run! Where does it start? Full auto?

        from also import Accumulator
        prepo1 = Accumulator()
        also = prepo1.also

        if also(self.RV):
            # for instruments in rv
            acc_lims = sp.array([-1., 1.])
            jitt_limiter = sp.amax(abs(self.rv))
            jitt_lim = 3 * jitt_limiter
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
        thetas_hen, ajuste_hen = 0., 0.
        ajuste_raw = sp.array([0])
        self.oldlogpost = -sp.inf
        interesting_thetas, interesting_posts = sp.array([]), sp.array([])
        thetas_raw = sp.array([])

        if self.RV:
            # INITIALIZE GENERAL PARAMS
            self._theta_gen(acc_lims, None)

            # INITIALIZE INSTRUMENT PARAMS

            for nin in range(self.nins):
                moav_lim = sp.array([(-1.0, 1.0, 0.1, 10)
                                     for _ in range(self.MOAV[nin])]).reshape(-1)
                ins_lims = sp.append(jitoff_lim, moav_lim).reshape(-1)
                self._theta_ins(ins_lims, None, nin, self.MOAV[nin])

        if self.PM:
            '''
            # INITIALIZE GENERAL PARAMS
            self._theta_gen_pm(acc_bnd, None)

            # INITIALIZE INSTRUMENT PARAMS

            for nin in range(self.nins_pm):
                moav_bnd = sp.array([(-1.0, 1.0, 0.1, 10) for _ in range(self.MOAV_pm[nin])]).reshape(-1)
                ins_bnd = sp.append(jitoff_bnd, moav_bnd).reshape(-1)
                self._theta_ins(ins_bnd, None, nin, self.MOAV_pm[nin])
            '''
            # INITIALIZE GEORGE
            # for n in range(len(self.george_kernels)):
            if self.gaussian_processor == 'george':
                # import george here? # DEL
                # this is a general gp, not per instrument, so jitter is for staract
                self.george_k = empmir.neo_kernel(self.george_kernels)

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

                ins_bnd = sp.array([-1, 1])
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

        # raise Exception('DEBUG')  # DEL
        while kplan <= to_k:
            if kplan > 0:
                if self.RV:
                    # INITIALIZE KEPLERIAN PARAMS
                    self._theta_rv(free_lims, [jitt_lim, [0, 1]], kplan)
                    pass
                if self.PM:
                    # INITIALIZE PHOTOMETRIC PARAMS
                    #ld_d = {'uniform':0, 'linear':1, 'quadratic':2, 'nonlinear':4}
                    self.batman_ldn.append(self.ld[self.batman_ld[kplan - 1]])
                    self._theta_photo(free_lims_pm, None, kplan,
                                      self.batman_ldn[kplan - 1])
                    # INITIALIZE BATMAN
                    # ncb = sp.ones(self.ld[self.batman_ld[kplan-1]])  # dummy coefficients
                    self.batman_m[kplan - 1], self.batman_p[kplan - 1] = empmir.neo_model_pm(
                        self.time_pm, self.batman_ld[kplan - 1], self.batman_ldn[kplan - 1])
                    # raise Exception('DEBUG')  # DEL
                    pass
        # FINAL MODEL STEP, apply commands
            # '''

            '''
            for j in range(len(self.changes_list))[::-1]:
                if self.theta.change_val(self.changes_list[j]):
                    print('Following condition has been applied: ', self.changes_list[j])
                    self.changes_list = sp.append(self.changes_list[:j], self.changes_list[j+1:])
                    self.changes_list = self.changes_list.reshape((len(self.changes_list)//3, 3))
            '''
            self.theta.apply_changes_list(self.changes_list)

            for j in range(len(self.theta.list_)):
                if (self.theta.list_[j].prior == 'uniform_spe_a'
                        and self.theta.list_[j + 1].prior == 'fixed'):
                    # phase fixed, so amplitude
                    self.changes_list[len(self.changes_list)] = \
                        [str(self.theta.list_[j].name), 'prior', 'uniform']
                    l1, l2 = self.theta.list_[j].args
                    self.changes_list[len(self.changes_list)] = \
                        [str(self.theta.list_[j].name), 'lims', 0., l1, l2]

                if (self.theta.list_[j].prior == 'uniform_spe_b'
                        and self.theta.list_[j - 1].prior == 'fixed'):
                    # amplitude fixed, so phase
                    self.changes_list[len(self.changes_list)] = \
                        [str(self.theta.list_[j].name), 'prior', 'uniform']
                    self.changes_list[len(self.changes_list)] = \
                        [str(self.theta.list_[j].name), 'lims', 0., 2 * sp.pi]

            # print('changes_list.shape es ', self.changes_list.shape) DEL

            # '''  # DEL
            # show the initialized params and priors
            for t in self.theta.list_:
                print(t.name, t.prior, t.val, t.lims)
            print('____')
            # '''
            # raise Exception('DEBUG')  # DEL
            # COORDINATOR
            self.coordinator = []
            self.anticoor = []
            for i in range(len(self.theta.list_)):
                if self.theta.list_[i].prior == 'fixed':
                    self.anticoor.append(i)
                else:
                    self.coordinator.append(i)

            '''
            for j in range(len(self.theta.list_)-1):
                if (self.theta.list_[j].prior == 'uniform_spe' and
                    self.theta.list_[j+1].prior != 'fixed'):
                    self.theta.gral_priors = sp.append(self.theta.gral_priors, 'hou_cov')
            '''
            self.theta._update_list_()

        # 3 generate values for said model, different step as this should allow configuration
            self.pos0 = emplib.neo_p0(
                self.setup, self.theta.list_, self.theta.ndim_, self.coordinator)

            '''
            # idea here is to reroll if nan on ll or lp on p0
            trynum = 2
            while sp.isnan(self.pos0).any() == True:
                print('Optimizing starting point, try number %i', % trynum)
                trynum += 1
                self.pos0 = emplib.neo_p0(self.setup, self.theta.list_, self.theta.ndim_, self.coordinator)
            '''
        # 4 run chain

            p = self.pos0[0][1]

            # raise Exception('DEBUG')  # DEL
            if self.RV:
                from emperors_mirror import neo_logp_rv, neo_logl_rv
                logl_params = sp.array([self.time, self.rv, self.err, self.ins,
                                        self.staract, self.starflag, kplan, self.nins,
                                        self.MOAV, self.totcornum, self.ACC])
            if self.PM:
                from emperors_mirror import neo_logp_pm, neo_logl_pm
                logl_params = sp.array([self.time_pm, self.rv_pm, self.err_pm,
                                        self.ins_pm, kplan, self.nins_pm,
                                        self.batman_ldn, self.batman_m, self.batman_p,
                                        self.emperors_gp, self.gaussian_processor])

            # rv and pm testing
            self.autodestruction = 0
            self.adc = 0
            if self.RV:
                for i in range(self.nwalkers):
                    self.a = neo_logp_rv(
                        self.pos0[0][i], [self.theta.list_, self.theta.ndim_, self.coordinator])
                    if self.a == -sp.inf:
                        self.adc += 1
                    self.autodestruction = (
                        self.nwalkers - self.adc) / self.nwalkers
                    self.adc = 0
                print('autodestruction', self.autodestruction)

                while self.autodestruction <= 0.98:
                    print('Reinitializing walkers')
                    print('autodestruction', self.autodestruction)
                    self.pos0 = emplib.neo_p0(
                        self.setup, self.theta.list_, self.theta.ndim_, self.coordinator)
                    for i in range(self.nwalkers):
                        self.a = neo_logp_rv(
                            self.pos0[0][i], [self.theta.list_, self.theta.ndim_, self.coordinator])
                        if self.a == -sp.inf:
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
            # '''
            # raise Exception('DEBUG')  # DEL
        # 5 get stats (and model posterior)

            # posterior handling

            self.posteriors = sp.array(
                [self.sampler.lnprobability[i].reshape(-1) for i in range(self.ntemps)])
            self.post_max = sp.amax(self.posteriors[0])

            self.ajuste = self.sampler.flatchain[0][sp.argmax(
                self.posteriors[0])]

            # updates values in self.theta.list_ with best of emcee run
            for i in range(self.theta.ndim_):
                self.theta.list_[self.coordinator[i]].val = self.ajuste[i]
                print(self.theta.list_[self.coordinator[i]].name, self.theta.list_[
                      self.coordinator[i]].val)

            # TOP OF THE POSTERIOR
            cherry_locat = sp.array([max(self.posteriors[temp]) - self.posteriors[temp]
                                     < self.bayes_factor for temp in sp.arange(self.ntemps)])
            self.cherry_chain = sp.array(
                [self.sampler.flatchain[temp][cherry_locat[temp]] for temp in sp.arange(self.ntemps)])
            self.cherry_post = sp.array(
                [self.posteriors[temp][cherry_locat[temp]] for temp in range(self.ntemps)])

            # sigmas are taken from cold chain
            self.sigmas = sp.array(
                [sp.std(self.cherry_chain[0][:, i]) for i in range(self.theta.ndim_)])

            self.sample_sizes = sp.array(
                [len(self.cherry_chain[i]) for i in range(self.ntemps)])

            # ojo esto
            if self.RV and self.VINES:
                '''
                henshin actual asume los change of variable (cov) de hou
                para todos los parametros... si fixeas cosas no respondo
                '''
                self.cherry_chain_h = sp.array(
                    [empmir.henshin(self.cherry_chain[i], kplan) for i in sp.arange(self.ntemps)])
                # self.cherry_chain_h = sp.array([self.cherry_chain[temp] for temp in sp.arange(self.ntemps)])  # no se pq doble pero no lo voy a mirar
                self.ajuste_h = self.cherry_chain_h[0][sp.argmax(
                    self.cherry_post[0])]

            # residuals = empmir.RV_residuals(ajuste, self.rv, self.time,
                #self.ins, self.staract, self.starflag, kplan,
                # self.nins, self.MOAV, self.totcornum, self.ACC)
            #alt_res = self.alt_results(cherry_chain[0], kplan)
            if self.MUSIC:
                thybiding.play()

        # 6 compare, exit or next
            # BIC & AIC
            if self.RV:
                self.NEW_BIC = sp.log(self.ndat) * \
                    self.theta.ndim_ - 2 * self.post_max
                self.OLD_BIC = sp.log(self.ndat) * \
                    self.theta.ndim_ - 2 * self.oldlogpost
                self.NEW_AIC = 2 * self.theta.ndim_ - 2 * self.post_max
                self.OLD_AIC = 2 * - 2 * self.oldlogpost
            if self.PM:
                self.NEW_BIC = sp.log(self.ndat_pm) * \
                    self.theta.ndim_ - 2 * self.post_max
                self.OLD_BIC = sp.log(self.ndat_pm) * \
                    self.theta.ndim_ - 2 * self.oldlogpost
                self.NEW_AIC = 2 * self.theta.ndim_ - 2 * self.post_max
                self.OLD_AIC = 2 * - 2 * self.oldlogpost

            saveplace = self.mklogfile(kplan)
            if self.VINES:  # saves chains, posteriors, rv data and log
                emplib.instigator(
                    self.cherry_chain, self.cherry_post, self.all_data,
                    saveplace
                )

            if self.MUSIC:
                thybiding.play()

            if self.INPLOT:
                pass

            if self.OLD_BIC - self.NEW_BIC < self.BIC:
                print(
                    '\nBayes Information Criteria of %.2f requirement not met ! !' % self.BIC)
            if self.OLD_AIC - self.NEW_AIC < self.AIC:
                print(
                    '\nAkaike Information Criteria of %.2f requirement not met ! !' % self.AIC)

            print('Max logpost vs. Past max logpost', self.post_max,
                  self.oldlogpost, self.post_max - self.oldlogpost)
            print('Old BIC vs New BIC', self.OLD_BIC,
                  self.NEW_BIC, self.OLD_BIC - self.NEW_BIC)
            print('Old AIC vs New AIC', self.OLD_AIC,
                  self.NEW_AIC, self.OLD_AIC - self.NEW_AIC)

            if self.post_max - self.oldlogpost < self.model_comparison:
                print('\nBayes Factor of %.2f requirement not met ! !' %
                      self.model_comparison)
                # break

            self.oldlogpost = self.post_max

        # 7 remodel prior, go back to step 2

            self.constrain = [15.9, 84.1]
            self.constrain = [30.15, 69.85]
            self.constrain = [38.15, 61.85]
            if kplan > 0:
                for i in range(self.theta.ndim_):
                    if (self.theta.list_[self.coordinator[i]].prior != 'fixed'
                            and self.theta.list_[self.coordinator[i]].type == 'keplerian'):
                        self.theta.list_[self.coordinator[i]].lims = sp.percentile(
                            self.cherry_chain[0][:, i], self.constrain)
                        #self.theta.list_[self.coordinator[i]].args = [ajuste[i], sigmas[i]]
                        pass

            # '''
            kplan += 1

        if self.MUSIC:  # end music
            technological_terror.play()
        pass  # end CONQUER
#


# stardat = sp.array(['GJ357_1_HARPS3.dat',
#                    'GJ357_2_UVES3.dat',
#                    'GJ357_3_KECK3.vels'])
stardat = sp.array(['GJ876_1_LICK.vels', 'GJ876_2_KECK.vels'])
#pmfiles = sp.array(['flux/GJ357_tess_pdcflux_flat_4clip.flux'])
#pmfiles = sp.array(['flux/transit_ground_r.flux'])
#stardat = pmfiles
#stardat = sp.array([])
setup = sp.array([2, 60, 120])
em = EMPIRE(stardat, setup)
# em = EMPIRE(stardat, setup, file_type='pm_file')  # ais.empire
em.CORNER = False  # corner plot disabled as it takes some time to plot
# array([1.0])  # beta factor for each temperature, None for automatic
em.betas = None
#em.betas = sp.array([1.0, 0.55, 0.3025, 0.1663, 0.0915])

# we actually run the chain from 0 to 2 signals
#em.RAW = True
#em.ACC = 1
em.MOAV = sp.array([0, 0])  # not needed


#em.batman_ld = ['quadratic']
#em.gaussian_processor = 'george'
#em.gaussian_processor = 'celerite'

#em.george_kernels = sp.array([['Matern32Kernel']])
#em.george_jitter = False

#em.celerite_kernels = sp.array([['Matern32Term', 'RealTerm']])
#em.celerite_jitter = False

em.MUSIC = False


# this is for you vines
# gj876 lims
em.changes_list = {0: ['Period', 'lims', 4.11098843, 4.11105404],
                   1: ['Amplitude', 'lims', -10.1515928, -6.33312469],
                   2: ['Phase', 'lims', 1.00520806e+01,   1.35949748e+01],
                   3: ['Eccentricity', 'lims', 4.14811179e-02, 1.38568310e-01],
                   4: ['Longitude', 'lims', -1.24823254e-02,   2.27443388e-02],
                   5: ['Period_2', 'lims', 3.40831451, 3.40863545],
                   6: ['Amplitude_2', 'lims', -5.69294095, 0.0605896817],
                   7: ['Phase_2', 'lims', -9.33328013e+00, -8.07370401e+00],
                   8: ['Eccentricity_2', 'lims', 3.47811764e-02,   1.79877743e-01],
                   9: ['Longitude_2', 'lims', -2.48303912e-01,  -7.10641857e-02]
                   }

'''
em.changes_list = {0:['t0', 'lims', 2456915.67, 2456915.73],
                   1:['Planet Radius', 'lims', 0.05, 0.09],
                   2:['Period', 'prior', 'fixed'],
                   3:['Period', 'val', 24.73712],
                   4:['SemiMajor Axis', 'prior', 'fixed'],
                   5:['SemiMajor Axis', 'val', 101.1576001138329],
                   6:['Inclination', 'prior', 'fixed'],
                   7:['Inclination', 'val', 89.912],
                   8:['Eccentricity', 'prior', 'fixed'],
                   9:['Eccentricity', 'val', 0.],
                   10:['Longitude', 'prior', 'fixed'],
                   11:['Longitude', 'val', 90.],
                   12:['coef1', 'prior', 'fixed'],
                   13:['coef1', 'val', 0.1],
                   14:['coef2', 'prior', 'fixed'],
                   15:['coef2', 'val', 0.3]}
'''
em.conquer(1, 1)

'''
em.changes_list = {:['t0', 'prior', 'fixed'],
                   :['t0', 'val', 2456915.6997],
                   :['Planet Radius', 'prior', 'fixed'],
                   :['Planet Radius', 'val', 0.0704],
                   :['Period', 'prior', 'fixed'],
                   :['Period', 'val', 24.73712],
                   :['SemiMajor Axis', 'prior', 'fixed'],
                   :['SemiMajor Axis', 'val', 101.1576001138329],
                   :['Inclination', 'prior', 'fixed'],
                   :['Inclination', 'val', 89.912],
                   :['Eccentricity', 'prior', 'fixed'],
                   :['Eccentricity', 'val', 0.],
                   :['Longitude', 'prior', 'fixed'],
                   :['Longitude', 'val', 90.],
                   :['coef1', 'prior', 'fixed'],
                   :['coef1', 'val', 0.1],
                   :['coef2', 'prior', 'fixed'],
                   :['coef2', 'val', 0.3]}
'''

'''
em.changes_list = {0:['Period', 'lims', 4.0943445, 4.1271343],
                   1:['Period_2', 'lims', 3.38, 3.42]
                   }

em.changes_list = {0:['Period', 'prior', 'fixed'],
                   1:['Period', 'val', 3.93],
                   2:['Inclination', 'prior', 'fixed'],
                   3:['Inclination', 'val', 88.946],
                   4:['Eccentricity', 'prior', 'fixed'],
                   5:['Eccentricity', 'val', 0.],
                   6:['Longitude', 'prior', 'fixed'],
                   7:['Longitude', 'val', 0.],
                   8:['t0', 'prior', 'fixed'],
                   9:['t0', 'val', 2458517.99]}
'''


#

#x, y, y_error = em.time_pm, em.rv_pm, em.err_pm

'''
array(['t0', 'Period', 'Planet Radius', 'SemiMajor Axis', 'Inclination',
       'Eccentricity', 'Longitude', 'coef1', 'coef2', 'Jitter',
       'kernel0_0', 'kernel0_1'], dtype='<U14')

array(['213482777044.7721', '24.73712', '1167549.8787819578', '-inf',
       '89.912', '0.0', '0.0', '0.1', '0.3', '-11.316116337221487',
       '24.096249121805645', '32.28733028116761']
'''

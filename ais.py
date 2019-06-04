#!/usr/bin/env python
# -*- coding: utf-8 -*-
# EMPEROR Exoplanet Mcmc Parallel tEmpering Radial velOcity fitteR
# version 0.-432,12.78

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


class spec:
    def __init__(self, name, units, prior, lims, val, args=[]):
        self.name = name
        self.units = units
        self.prior = prior  #d[str(prior)]
        self.lims = lims
        self.val = -sp.inf
        self.args = args

    def __prior(self, x, *args):
        return self.__prior(x, args)
    def identify(self):
        return self.name+'    '+self.units
    def tag(self):
        return self.name.split('_')[0]
    pass

class EMPIRE:
    def __init__(self, stardat, setup, file_type='rv_file'):
        assert len(stardat) >= 1, 'stardat has to contain at least 1 file ! !'
        assert len(setup) == 3, 'setup has to be [ntemps, nwalkers, nsteps]'
        #  Setup
        self.cores = multiprocessing.cpu_count()
        self.setup = setup
        self.ntemps, self.nwalkers, self.nsteps = setup
        self.betas = None

        self.changes_list = sp.array([])
        self.coordinator = sp.array([])

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
        self.ndim = len(self.theta)
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
            self.f2k = None  # EXTERMINATE
            #  acceleration quadratic
            self.PACC_pm = False
        else:
            raise Exception('You sure you wrote the filetype correctly mate?')
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
        self.MOAV = sp.array([1, 1])  # MOAV order for each instrument

        # EXTERMINATE
        self.eccprior = 0.3
        self.jittprior = 5.0
        self.jittmean = 5.0

        # auxiliary for later
        self.sampler = 0.0
        ########################################

        pass


    def change_val(self, commands):
        object_id, action, whato = commands
        for theta in self.theta:
            if theta.name == object_id:
                setattr(theta, action, whato)
                return True
        return False


    def _ndim(self):
        dim = 0
        for t in self.theta:
            if t.prior=='fixed' or t.prior=='joined':
                dim -= 1
        return len(self.theta) + dim

    def _theta_rv(self, limits, conditions, kplanets):
        names = ["Period", "Amplitude", "Phase", "Eccentricity", "Longitude"]
        if kplanets >= 2:
            names = [str(name)+'_'+str(kplanets) for name in names]
        units = [" [Days]", " $[\\frac{m}{s}]$", " $[rad]$", "", " $[rads]$"]
        priors = ['uniform', 'uniform', 'uniform', 'uniform', 'uniform']
        new = sp.array([])
        for i in range(5):
            t = spec(names[i], units[i], priors[i], [limits[2*i], limits[2*i+1]], -sp.inf)
            new = sp.append(new, t)
        if kplanets == 1:
            self.theta = sp.append(new, self.theta)
        else:
            self.theta = sp.insert(self.theta, (kplanets-1)*len(names), new)
        pass

    def _theta_ins(self, limits, conditions, nin, MOAV):
        names = ['Jitter', 'Offset', 'MACoefficient', 'MATimescale']
        if nin > 0:
            names = [str(name)+'_'+str(nin+1) for name in names]
        #print(names)
        units = [' $[\\frac{m}{s}]$', ' $[\\frac{m}{s}]$', ' [Days]', '']
        priors = ['uniform', 'uniform', 'uniform', 'uniform']
        new = sp.array([])
        for i in range(2):
            t = spec(names[i], units[i], priors[i], [limits[2*i], limits[2*i+1]], -sp.inf)
            new = sp.append(new, t)
        for j in range(2*MOAV):
            if MOAV > 1:
                names1 = [str(name)+'_'+str(j//2+1) for name in names]
            else:
                names1 = names
            t = spec(names1[j%2+2], units[j%2+2], priors[j%2+2], [limits[j%2+2], limits[j%2+2]], -sp.inf)
            new = sp.append(new, t)
        self.theta = sp.append(self.theta, new)
        pass

    def _theta_star(self, limits, conditions, instruments):
        name = 'Stellar Activity'

    def _theta_gen(self, limits, conditions):
        priors = 'uniform'
        new = []
        for i in range(self.ACC):
            name = 'Acceleration'
            if self.ACC == 1:
                aux = ''
            else:
                aux = '_%i' % i
            units = [' $[\\frac{m}{s%i}]$' % (i+1)]
            t = spec(name+aux, units, priors, [limits[0], limits[1]], -sp.inf)
            new = sp.append(new, t)
        self.theta = sp.append(new, self.theta)
        pass

    def MCMC(self, *args):
        if args:
            #kplan, mod_lims, ins_lims, acc_lims, sigmas_raw, pos0, logl, logp
            kplanets, boundaries, inslims = args[0], args[1], args[2]
            acc_lims, sigmas_raw, pos0 = args[3], args[4], args[5]
            logl, logp = args[6], args[7]
        #ndim = 1 + 5 * kplanets + self.nins*2*(self.MOAV+1) + self.totcornum + self.PACC
        #print(str(self.PM)), 'self.pm!!'  # PMPMPM
        if kplanets > 0:
            if self.PM:
                pm_lims = args[8]
                ndim += self.lenppm*self.fsig
                print('checkpoint 1')  # PMPMPM
        ndat = len(self.time)
        def starinfo():
            colors = ['red', 'green', 'blue', 'yellow', 'grey', 'magenta', 'cyan', 'white']
            c = sp.random.randint(0,7)
            print(colored('\n    ###############################################', colors[c]))
            print(colored('    #                                             #', colors[c]))
            print(colored('    #                                             #', colors[c]))
            print(colored('    #                 E M P E R 0 R               #', colors[c]))
            print(colored('    #                                             #', colors[c]))
            print(colored('    #                                             #', colors[c]))
            print(colored('    ###############################################', colors[c]))
            print(colored('Exoplanet Mcmc Parallel tEmpering Radial vel0city fitteR', colors[sp.random.randint(0,7)]))
            logdat = '\n\nStar Name                         : '+self.starname
            logdat += '\nTemperatures, Walkers, Steps      : '+str((self.ntemps, self.nwalkers, self.nsteps))
            logdat += '\nN Instruments, K planets, N data  : '+str((self.nins, kplanets, self.ndat))
            if self.PM:
                logdat += '\nN of data for Photometry          : '+str(self.ndat_pm)
            logdat += '\nN Number of Dimensions            : '+str(ndim)
            logdat += '\nN Moving Average                  : '+str(self.MOAV)
            logdat += '\nBeta Detail                       : '+str(self.betas)
            logdat += '\n-----------------------------------------------------'
            print(logdat)
            pass

        starinfo()
        #'''
        #from emperors_library import logp_rv
        print(str(self.PM), ndim, 'self.pm y ndim')  # PMPMPM
        if self.PM:
            if kplanets > 0:
                logp_params = sp.array([sp.array([self.time, kplanets, self.nins, self.MOAV,
                                        self.totcornum, boundaries, inslims, acc_lims,
                                        sigmas_raw, self.eccprior, self.jittprior,
                                        self.jittmean, self.STARMASS, self.HILL,
                                        self.PACC, self.CHECK]),
                               sp.array([self.time_pm, self.fsig, self.lenppm,
                                         self.nins_pm, self.MOAV_pm,
                                         self.totcornum_pm, boundaries, sigmas_raw,
                                         self.PACC_pm])])

                logl_params = sp.array([sp.array([self.time, self.rv, self.err, self.ins,
                                        self.staract, self.starflag, kplanets, self.nins,
                                        self.MOAV, self.totcornum, self.PACC]),
                               sp.array([self.time_pm, self.rv_pm, self.err_pm, self.ins_pm,
                                        self.staract_pm, self.starflag_pm, self.fsig,
                                        self.f2k, self.nins_pm, self.MOAV_pm,
                                        self.totcornum_pm, self.PACC_pm, kplanets])])
                self.sampler = PTSampler(self.ntemps, self.nwalkers, ndim, logl, logp,
                                         loglargs=[empmir.logl_rvpm, logl_params],
                                         logpargs=[empmir.logp_rvpm, logp_params],
                                         threads=self.cores, betas=self.betas)
                #raise ImportError('xd dale al debug mejor')
            else:
                logp_params = sp.array([self.time, kplanets, self.nins, self.MOAV,
                                        self.totcornum, boundaries, inslims, acc_lims,
                                        sigmas_raw, self.eccprior, self.jittprior,
                                        self.jittmean, self.STARMASS, self.HILL,
                                        self.PACC, self.CHECK])
                logl_params = sp.array([self.time, self.rv, self.err, self.ins,
                                        self.staract, self.starflag, kplanets, self.nins,
                                        self.MOAV, self.totcornum, self.PACC])
                self.sampler = PTSampler(self.ntemps, self.nwalkers, ndim, logl, logp,
                                         loglargs=[empmir.logl_rv, logl_params],
                                         logpargs=[empmir.logp_rv, logp_params],
                                         threads=self.cores, betas=self.betas)
            # raise ImportError
        else:
            logp_params = sp.array([self.time, kplanets, self.nins, self.MOAV,
                                    self.totcornum, boundaries, inslims, acc_lims,
                                    sigmas_raw, self.eccprior, self.jittprior,
                                    self.jittmean, self.STARMASS, self.HILL,
                                    self.PACC, self.CHECK])
            logl_params = sp.array([self.time, self.rv, self.err, self.ins,
                                    self.staract, self.starflag, kplanets, self.nins,
                                    self.MOAV, self.totcornum, self.PACC])
            self.sampler = PTSampler(self.ntemps, self.nwalkers, ndim, logl, logp,
                                     loglargs=[empmir.logl_rv, logl_params],
                                     logpargs=[empmir.logp_rv, logp_params],
                                     threads=self.cores, betas=self.betas)
        # RVPM THINGY

        print('\n --------------------- BURN IN --------------------- \n')

        pbar = tqdm(total=self.burn_out)

        for p, lnprob, lnlike in self.sampler.sample(pos0, iterations=self.burn_out):
            pbar.update(1)
            pass
        pbar.close()

        p0, lnprob0, lnlike0 = p, lnprob, lnlike
        print("\nMean acceptance fraction: {0:.3f}".format(sp.mean(self.sampler.acceptance_fraction)))
        assert sp.mean(self.sampler.acceptance_fraction) != 0, 'Mean acceptance fraction = 0 ! ! !'
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
        #'''

        assert self.sampler.chain.shape == (self.ntemps, self.nwalkers, self.nsteps/self.thin, ndim), 'something really weird happened'
        print("\nMean acceptance fraction: {0:.3f}".format(sp.mean(self.sampler.acceptance_fraction)))

        ln_post = self.sampler.lnprobability

        posteriors = sp.array([ln_post[i].reshape(-1) for i in range(self.ntemps)])
        chains = self.sampler.flatchain
        best_post = posteriors[0] == np.max(posteriors[0])
        #raise ImportError

        thetas_raw = sp.array([chains[i] for i in range(self.ntemps)])
        thetas_hen = sp.array([empmir.henshin(chains[i], kplanets) for i in sp.arange(self.ntemps)])

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


    def conquer(self, from_k, to_k, logl=logl, logp=logp, BOUND=sp.array([])):
        # 1 handle data
        # 2 set adecuate model
        # 3 generate values for said model, different step as this should allow configuration
        # 4 run chain
        # 5 get stats (and model posterior)
        # 6 compare, exit or next
        # 7 remodel prior, go back to step 2


        # 1 is currently being done upstairs (in __init__ i mean)

        assert self.cores >= 1, 'Cores is set to 0 ! !'
        assert self.thin * self.draw_every_n < self.nsteps, 'You are thining way too hard ! !'
        if self.betas is not None:
            assert len(self.betas) == self.ntemps, 'Betas array and ntemps dont match ! !'

        if self.MUSIC:
            imperial.play()

        pass

        #Here should be how to run! Where does it start? Full auto?

        from also import Accumulator
        prepo1 = Accumulator()
        also = prepo1.also

        if also(self.RV):
            # for instruments in rv
            acc_lims = sp.array([-1., 1.])
            jitt_limiter = sp.amax(abs(self.rv))
            jitt_lim = 3 * jitt_limiter
            offs_lim = jitt_limiter


            # for the keplerian signals
            kplan = from_k
            sqrta, sqrte = jitt_lim, 1.
            sqrta, sqrte = sqrta ** 0.5, sqrte ** 0.5
            free_lims = sp.array([sp.log(0.1), sp.log(3 * max(self.time)), -sqrta, sqrta, -sqrta, sqrta, -sqrte, sqrte, -sqrte, sqrte])




        if also(self.PM):
            pass

        if also(self.RV and self.PM):  # Here goes the rvpm
            pass

        if prepo1.none:
            raise Exception('Mark RV or PM')
            pass

        #sigmas, sigmas_raw = sp.zeros(self._ndim), sp.zeros(self._ndim)  # should go in param object?
        pos0 = 0.
        thetas_hen, ajuste_hen = 0., 0.
        ajuste_raw = sp.array([0])
        oldlogpost = -999999999.
        interesting_thetas, interesting_posts = sp.array([]), sp.array([])
        thetas_raw = sp.array([])
        START = chrono.time()

        # INITIALIZE GENERAL PARAMS
        self._theta_gen(acc_lims, None)

        # INITIALIZE INSTRUMENT PARAMS
        for nin in range(self.nins):
            ins_lims = sp.append(sp.array([0.0001, jitt_lim, -offs_lim, offs_lim]), sp.array([(-1.0, 1.0, 0.1, 10) for _ in range(self.MOAV[nin])])).reshape(-1)
            self._theta_ins(ins_lims, None, nin, self.MOAV[nin])

        while kplan <= to_k:
            if kplan > 0:
                # INITIALIZE KEPLERIAN PARAMS
                self._theta_rv(free_lims, None, kplan)
                pass


        # FINAL MODEL STEP, apply commands
            #'''



            for j in range(len(self.changes_list))[::-1]:
                if self.change_val(self.changes_list[j]):
                    print('Following condition has been applied: ', self.changes_list[j])
                    self.changes_list = sp.append(self.changes_list[:j], self.changes_list[j+1:])
                    self.changes_list = self.changes_list.reshape((len(self.changes_list)//3, 3))

            print('asdasdasd', self.changes_list.shape)
            for t in self.theta:
                print(t.name, t.prior, t.val)


            ### COORDINATOR
            self.coordinator = []
            self.ac = []
            for i in range(len(self.theta)):
                if self.theta[i].prior == 'fixed':
                    self.ac.append(i)
                else:
                    self.coordinator.append(i)
            ##########
            #self.change_val(['Acceleration', 'prior', 'fixed'])
            #self.change_val(['Acceleration', 'val', '0.1'])
        # 3 generate values for said model, different step as this should allow configuration
            self.pos0 = emplib.neo_p0(self.setup, self.theta, self._ndim(), self.coordinator)
        # 4 run chain
            #thetas_raw, ajuste_raw, thetas_hen, ajuste_hen, p, lnprob, lnlike, posteriors, betas, interesting_thetas, interesting_posts, sigmas, sigmas_raw = self.MCMC(kplan, mod_lims, ins_lims, acc_lims, sigmas_raw, pos0, logl, logp)

            from emperors_mirror import neo_logp_rv
            p=self.pos0[0][1]



            self.a = neo_logp_rv(p, [self.theta, self._ndim(), self.coordinator])


            logl_params = sp.array([self.time, self.rv, self.err, self.ins,
                                    self.staract, self.starflag, kplanets, self.nins,
                                    self.MOAV, self.totcornum, self.PACC])

            self.b = neo_logl_rv(p, [self.theta, self.anticoordinator(), logl_params])

            #em.aa = neo_logl_rv(p, [em.theta, em._ndim()])



            '''
            s0 = chrono.time()
            for _ in range(10000):
                neo_logp_rv(p, [em.theta, em._ndim()])
            print('________', chrono.time()-s0)
            '''
            kplan += 1






        pass  # end CONQUER
#




# import ais
stardat = sp.array(['GJ876_1_LICK.vels', 'GJ876_2_KECK.vels'])
setup = sp.array([2, 50, 100])


em = EMPIRE(stardat, setup)  # ais.empire
em.CORNER = False  # corner plot disabled as it takes some time to plot
#em.betas = None #array([1.0])  # beta factor for each temperature, None for automatic
#em.MOAV = 0
# em.MUSIC= True
# we actually run the chain from 0 to 2 signals
#em.RAW = True
#em.ACC = 3
em.MOAV = sp.array([1,1])
em.MUSIC = False
#'''
em.changes_list = sp.array([['Acceleration', 'prior', 'fixed'],
                            ['Acceleration', 'val', 0.1],
                            ['Eccentricity', 'prior', 'fixed'],
                            ['Eccentricity', 'val', 0.0],
                            ['Period_2', 'prior', 'fixed'],
                            ['Period_2', 'val', 31.]])

#'''


em.conquer(0, 2)
#

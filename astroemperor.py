#!/usr/bin/env python
# -*- coding: utf-8 -*-
# EMPEROR Exoplanet Mcmc Parallel tEmpering Radial velOcity fitteR
# version 0.10.0
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
from PyAstronomy.pyasl import MarkleyKESolver
from decimal import Decimal  # histograms
import corner
import time as chrono
import multiprocessing
import datetime as dt

import emperors_library as emplib
import emperors_mirror as empmir

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError('You don t have the package tqdm installed. Try pip install tqdm.')
try:
    from termcolor import colored
except:
    print('You are missing the most cool package in Python! Try pip install termcolor')
try:
    from pygame import mixer
except:
    print('You are missing the most cool package in Python! Try pip install pygame or set MUSIC=False')



def logp(theta, func_logp, args):
    return func_logp(theta, args)


def logl(theta, func_logl, args):
    return func_logl(theta, args)


class EMPIRE:
    def __init__(self, stardat, setup):
        assert len(stardat) >= 1, 'stardat has to contain at least 1 file ! !'
        assert len(setup) == 3, 'setup has to be [ntemps, nwalkers, nsteps]'
        #  Setup
        self.cores = multiprocessing.cpu_count()
        self.setup = setup
        self.ntemps, self.nwalkers, self.nsteps = setup
        self.betas = None
        self.MOAV = 1

        self.burn_out = self.nsteps // 2
        self.PM = False
        #  Reading data

        if len(stardat.shape) > 1:
            # RV
            self.rvfiles = stardat[0]

            rvdat = emplib.read_data(self.rvfiles)
            self.time, self.rv, self.err, self.ins = rvdat[0]  # time, radial velocities, error and instrument flag
            self.all_data = rvdat[0]
            self.staract, self.starflag = rvdat[1], rvdat[2]  # time, star activity index and flag
            self.totcornum = rvdat[3]  # quantity if star activity indices

            self.nins = len(self.rvfiles)  # number of instruments autodefined
            self.ndat = len(self.time)  # number of datapoints

            # PM
            self.pmfiles = stardat[1]
            pmdat = emplib.read_data(self.pmfiles)
            self.time_pm, self.rv_pm, self.err_pm, self.ins_pm = pmdat[0]  # just the fname its pm rly
            self.all_data_pm = pmdat[0]
            self.staract_pm, self.starflag_pm = pmdat[1], pmdat[2]  # time, star activity index and flag
            self.totcornum_pm = pmdat[3]  # ?

            self.nins_pm = len(self.pmfiles)
            self.ndat_pm = len(self.time_pm)
            self.MOAV_pm = 1  # for flat model
            self.fsig = 1
            self.f2k = sp.array([0, 0])
            self.PM = True

            self.params_pm = sp.array([1, 2, 3, 4])
            self.lenppm = len(self.params_pm)

            self.PACC_pm = False

        else:  # RV

            self.rvfiles = stardat
            rvdat = emplib.read_data(stardat)
            self.time, self.rv, self.err, self.ins = rvdat[0]  # time, radial velocities, error and instrument flag
            self.all_data = rvdat[0]
            self.staract, self.starflag = rvdat[1], rvdat[2]  # time, star activity index and flag
            self.totcornum = rvdat[3]  # quantity if star activity indices

            self.nins = len(self.rvfiles)  # number of instruments autodefined
            self.ndat = len(self.time)  # number of datapoints

            # PM
            self.time_pm, self.rv_pm, self.err_pm, self.ins_pm = 0., 0., 0., 0.
            self.totcornum_pm = 0.

        #  Statistical Tools
        self.bayes_factor = sp.log(150)  # inside chain comparison (smaller = stricter)
        self.model_comparison = 5
        self.BIC = 5
        self.AIC = 5

        #  Menudencies
        self.thin = 1
        self.PLOT = True
        self.draw_every_n = 1
        self.PNG = True
        self.PDF = False
        self.CORNER = True
        self.HISTOGRAMS = True
        self.starname = self.rvfiles[0].split('_')[0]
        self.MUSIC = False
        self.breakFLAG = False
        self.STARMASS = False
        self.HILL = False
        self.CHECK = False
        self.RAW = False
        '''
        self.CORNER_MASK = True
        self.CORNER_K = True
        self.CORNER_I =
        '''

        # About the search parameters
        self.PACC = False  # parabolic Acceleration

        self.CONSTRAIN = True
        self.eccprior = 0.3
        self.jittprior = 5.0
        self.jittmean = 5.0

        self.sampler = 0.0
        ########################################
        # los IC sirven para los mod_lims? NO!
        # mod_lims sólo acotan el espacio donde buscar, excepto para periodo
        pass


    def mklogfile(self, *args):
        if args:
            theta_max, best_post, sample_sizes = args[0], args[1], args[2]
            sigmas, kplanets, modlims = args[3], args[4], args[5]
            BIC, AIC, alt_res = args[6], args[7], args[8]
            START, residuals = args[9], args[10]



        dayis = dt.date.today()  # This is for the folder name
        def ensure_dir(date='datalogs/'+self.starname+'/'+str(dayis.month)+'.'+str(dayis.day)+'.'+str(dayis.year)[2:]):
            if not os.path.exists(date):
                os.makedirs(date)
                return date
            else:
                if len(date.split('_')) == 2:
                    aux = int(date.split('_')[1]) + 1
                    date = date.split('_')[0]+'_'+str(aux)
                else:
                    date = date + '_1'
            return ensure_dir(date)

        def timer():
            timing = chrono.time() - START
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

        def mklogdat(theta, best_post):
            G = 39.5
            days_in_year = 365.242199
            sigmas_hen = sigmas
            logdat = '\nStar Name                         : '+self.starname
            for i in range(self.nins):
                if i==0:
                    logdat += '\nUsed datasets                     : '+self.rvfiles[i]
                else:
                    logdat += '\n                                  : '+self.rvfiles[i]
            logdat += '\n--------------------------------------------------------------------'
            logdat += '\nThe sample sizes are        :    ' + str(sample_sizes)
            logdat += '\nThe maximum posterior is    :    ' + str(best_post)
            logdat += '\nThe BIC is                  :    ' + str(BIC)
            logdat += '\nThe AIC is                  :    ' + str(AIC)
            logdat += '\nThe RMS is                  :    ' + str(sp.sum(residuals**2))
            logdat += '\nThe most probable chain values are as follows...'
            for i in range(kplanets):
                #MP = sp.sqrt(1 - theta[i*5+4] ** 2) * theta[i*5] * theta[i*5+1] ** (1/3.) * STARMASS ** (2/3.) / 203.
                #SMA = (((theta[i*5+1]/365.242199) ** 2) * G * STARMASS0 / (4 * sp.pi ** 2)) ** (1/3.)
                if self.STARMASS:
                    SMA = (((theta[i*5]*24.*3600.)**2.0) / ( (4.0*sp.pi**2.0) / (6.67e-11 * self.STARMASS * 1.99e30) ))**(1./3) / 1.49598e11
                    MP = theta[i*5+1] / ( (28.4/sp.sqrt(1. - theta[i*5+4]**2.)) * (self.STARMASS**(-0.5)) * (SMA**(-0.5)) ) * 317.8

                logdat += '\n--------------------------------------------------------------------'
                logdat += '\nPeriod   '+str(i+1)+'[days] :   ' + str(theta[i*5]) + ' +- ' + str(sigmas_hen[i*5])
                logdat += '\nAmplitude  '+str(i+1)+'[m/s]:   ' + str(theta[i*5+1]) + ' +- ' + str(sigmas_hen[i*5+1])
                logdat += '\nPhase   '+str(i+1)+'        :   ' + str(theta[i*5+2]) + ' +- ' + str(sigmas_hen[i*5+2])
                logdat += '\nLongitude   '+str(i+1)+'    :   ' + str(theta[i*5+3]) + ' +- ' + str(sigmas_hen[i*5+3])
                logdat += '\nEccentricity   '+str(i+1)+' :   ' + str(theta[i*5+4]) + ' +- ' + str(sigmas_hen[i*5+4])
                if self.STARMASS:
                    logdat += '\nMinimum Mass   '+str(i+1)+' :   ' + str(MP)
                    logdat += '\nSemiMajor Axis '+str(i+1)+' :   ' + str(SMA)
            logdat += '\n--------------------------------------------------------------------'
            logdat += '\nAcceleration [m/s/(year)]:'+str(theta[5*kplanets]/(days_in_year*24*60*60)) + ' +- ' + str(sigmas_hen[5*kplanets]/(days_in_year*24*60*60))
            if self.PACC:
                logdat += '\nQuadratic Acceleration [m/s/(year)]:'+str(theta[5*kplanets + self.PACC]/(days_in_year*24*60*60)) + ' +- ' + str(sigmas_hen[5*kplanets + self.PACC]/(days_in_year*24*60*60))

            for i in range(self.nins):
                logdat += '\n--------------------------------------------------------------------'
                logdat += '\nJitter '+str(i+1)+'    [m/s]:   ' + str(theta[5*kplanets + i*2*(self.MOAV+1) + self.PACC + 1]) + ' +- ' + str(sigmas_hen[5*kplanets + i*2*(self.MOAV+1) + self.PACC + 1])
                logdat += '\nOffset '+str(i+1)+'    [m/s]:   ' + str(theta[5*kplanets + i*2*(self.MOAV+1) + self.PACC + 2]) + ' +- ' + str(sigmas_hen[5*kplanets + i*2*(self.MOAV+1) + self.PACC + 2])
                for j in range(self.MOAV):
                    logdat += '\nMA coef '+str(i+1)+'_'+str(j+1)+'        : ' + str(theta[5*kplanets + i*2*(self.MOAV+1) + 2*(j+1) + 1 + self.PACC]) + ' +- ' + str(sigmas_hen[5*kplanets + i*2*(self.MOAV+1) + 2*(j+1) + 1 + self.PACC])
                    logdat += '\nTimescale '+str(i+1)+'_'+str(j+1)+'[days]: ' + str(theta[5*kplanets + i*2*(self.MOAV+1) + 2*(j+1) + 2 + self.PACC]) + ' +- ' + str(sigmas_hen[5*kplanets + i*2*(self.MOAV+1) + 2*(j+1) + 2 + self.PACC])
            for h in range(self.totcornum):
                logdat += '\n--------------------------------------------------------------------'
                logdat += '\nStellar Activity'+str(h+1)+':   ' + str(theta[5*kplanets + self.nins*2*(self.MOAV+1) + self.PACC + 1 + h]) + ' +- ' + str(sigmas_hen[5*kplanets + self.nins*2*(self.MOAV+1) + self.PACC + 1 + h])

            if self.PM:
                if kplanets > 0:
                    for i in range(self.fsig):
                        ndim_rv = 5*kplanets + self.nins*2*(self.MOAV+1) + self.PACC + 1 + self.totcornum
                        logdat += '\n--------------------------------------------------------------------'
                        for ii in range(self.lenppm):  # BATMAN PARAMS
                            logdat += '\nParam_pm'+str(i+1)+str(ii+1)+'[m/s]:   ' + str(theta[ndim_rv + i*self.lenppm + ii]) + ' +- ' + str(sigmas_hen[ndim_rv + i*self.lenppm + ii])

                    logdat += '\n--------------------------------------------------------------------'
                    fdim = ndim_rv + self.lenppm*self.fsig
                '''
                else:
                    logdat += '\nAcceleration [m/s/(year)]:'+str(theta[ndim_rv]/(days_in_year*24*60*60)) + ' +- ' + str(sigmas_hen[ndim_rv]/(days_in_year*24*60*60))
                    for i in range(self.nins_pm):
                        logdat += '\n--------------------------------------------------------------------'
                        logdat += '\nJitter_pm '+str(i+1)+' [m/s]:   ' + str(theta[ndim_rv + i*2*(self.MOAV_pm+1) + self.PACC_pm + 1]) + ' +- ' + str(sigmas_hen[ndim_rv + i*2*(self.MOAV_pm+1) + self.PACC_pm + 1])
                        logdat += '\nOffset_pm '+str(i+1)+' [m/s]:   ' + str(theta[ndim_rv + i*2*(self.MOAV_pm+1) + self.PACC_pm + 2]) + ' +- ' + str(sigmas_hen[ndim_rv + i*2*(self.MOAV_pm+1) + self.PACC_pm + 2])
                        for j in range(self.MOAV_pm):
                            logdat += '\nMA coef '+str(i+1)+'_'+str(j+1)+'        : ' + str(theta[ndim_rv + i*2*(self.MOAV_pm+1) + 2*(j+1) + 1 + self.PACC_pm]) + ' +- ' + str(sigmas_hen[ndim_rv + i*2*(self.MOAV_pm+1) + 2*(j+1) + 1 + self.PACC_pm])
                            logdat += '\nTimescale '+str(i+1)+'_'+str(j+1)+'[days]: ' + str(theta[ndim_rv + i*2*(self.MOAV_pm+1) + 2*(j+1) + 2 + self.PACC_pm]) + ' +- ' + str(sigmas_hen[ndim_rv + i*2*(self.MOAV_pm+1) + 2*(j+1) + 2 + self.PACC_pm])
                '''

            logdat += '\n------------------------------ RV DATA ------------------------------'
            logdat += '\nTemperatures, Walkers, Steps      : '+str((self.ntemps, self.nwalkers, self.nsteps))
            logdat += '\nN Instruments, K planets, N data  : '+str((self.nins, kplanets, self.ndat))
            logdat += '\nNumber of Dimensions              : '+str(5 * kplanets + self.nins*2*(self.MOAV+1) + self.totcornum + self.PACC + 1)
            logdat += '\nN Moving Average                  : '+str(self.MOAV)
            logdat += '\nBeta Detail                       : '+str(self.betas)
            logdat += '\n--------------------------------------------------------------------'
            if self.PM:
                logdat += '\n------------------------------ PM DATA ------------------------------'
                logdat += '\nN Instruments, N signals, N data  : '+str((self.nins_pm, self.fsig, self.ndat_pm))
                if kplanets > 0:
                    ndim_rv = 5*kplanets + self.nins*2*(self.MOAV+1) + self.PACC + 1 + self.totcornum
                    logdat += '\nNumber of Dimensions              : '+str(ndim_rv + self.fsig*self.lenppm)
                else:
                    pass
                #logdat += '\nN Moving Average                  : '+str(self.MOAV_pm)
                #logdat += '\nBeta Detail                       : '+str(self.betas)
                logdat += '\n--------------------------------------------------------------------'


            logdat += '\nRunning Time                      : '+timer()
            print(logdat)
            logdat += '\n -------------------------- ADVANCED --------------------------'
            logdat += '\n raw fit'
            logdat += '\n'+str(theta)
            logdat += '\n raw boundaries'
            logdat += '\n '+ str(modlims)
            logdat += '\n alt_res'
            logdat += '\n '+ str(alt_res)
            return logdat


        name = str(ensure_dir())
        logdat = mklogdat(theta_max, best_post)
        sp.savetxt(name+'/log.dat', sp.array([logdat]), fmt='%100s')
        sp.savetxt(name+'/residuals.dat', sp.c_[self.time, residuals])
        return name


    def instigator(self, chain, post, saveplace, kplanets):
        '''
        Automatically saves chains and posteriors.
        '''

        def mk_header(kplanets):
            h = []
            kepler = ['Period                  ', 'Amplitude               ', 'Phase                   ', 'Longitude               ', 'Eccentricity            ', 'Minimum Mass            ', 'SemiMajor Axis          ']
            telesc = ['Jitter                  ', 'Offset                  ']
            mov_ave = ['MA Coef ', 'Timescale ']

            photo = ['param']

            for i in range(kplanets):
                for item in kepler:
                    h.append(item)
            if self.PACC:
                h.append('Linear Acceleration     ')
                h.append('Quadratic Acceleration  ')
            else:
                h.append('Acceleration            ')
            for j in range(self.nins):
                for item in telesc:
                    h.append(item)
                    for c in range(self.MOAV):
                        h.append(mov_ave[0]+str(c)+'               ')
                        h.append(mov_ave[1]+str(c)+'             ')
            for jj in range(self.totcornum):
                h.append('Stellar Activity'+str(jj))

            if self.PM:
                for k in range(self.nins_pm):
                    for kk in range(self.lenppm):
                        h.append('Photometry Parameter '+str(k)+'_'+str(kk))
            h = ' '.join(h)
            return h

        def savechain(chain):
            for i in range(self.ntemps):
                sp.savetxt(saveplace + '/chain_'+str(i)+'.dat', chain[i], header=mk_header(kplanets))
            pass
        def savepost(post):
            for i in range(self.ntemps):
                sp.savetxt(saveplace + '/posterior_'+str(i)+'.dat', post[i], header=mk_header(kplanets))
            pass
        savechain(chain)
        savepost(post)
        pass


    def alt_results(self, samples, kplanets):
        titles = sp.array(["Period","Amplitude","Longitude", "Phase","Eccentricity", 'Acceleration', 'Jitter', 'Offset', 'MACoefficient', 'MATimescale', 'Stellar Activity'])
        namen = sp.array([])
        ndim = kplanets * 5 + self.nins*2*(self.MOAV+1) + self.totcornum + 1 + self.PACC

        RESU = sp.zeros((ndim, 5))
        for k in range(kplanets):
            namen = sp.append(namen, [titles[i] + '_'+str(k) for i in range(5)])
        namen = sp.append(namen, titles[5])  # for acc
        if self.PACC:
            namen = sp.append(namen, 'Parabolic Acceleration')
        for i in range(self.nins):
            namen = sp.append(namen, [titles[ii] + '_'+str(i+1) for ii in sp.arange(2)+6])
            for c in range(self.MOAV):
                namen = sp.append(namen, [titles[ii] + '_'+str(i+1) + '_'+str(c+1) for ii in sp.arange(2)+8])
        for h in range(self.totcornum):
            namen = sp.append(namen, titles[-1]+'_'+str(h+1))

        if self.PM:
            for g in range(self.nins_pm):
                for gg in range(self.lenppm):
                    namen = sp.append(namen, 'Photometry param'+str(g)+'_'+str(gg+1))

        alt_res = list(map(lambda v: (v[2], v[3]-v[2], v[2]-v[1], v[4]-v[2], v[2]-v[0]),
                      zip(*np.percentile(samples, [2, 16, 50, 84, 98], axis=0))))
        logdat = '\nAlternative results with uncertainties based on the 2nd, 16th, 50th, 84th and 98th percentiles of the samples in the marginalized distributions'
        logdat = '\nFormat is like median +- 1-sigma, +- 2-sigma'
        for res in range(ndim):
            logdat += '\n'+namen[res]+'     : '+str(alt_res[res][0])+' +- '+str(alt_res[res][1:3]) +'    2%   +- '+str(alt_res[res][3:5])
            RESU[res] = sp.percentile(samples, [2, 16, 50, 84, 98], axis=0)[:, res]
        print(logdat)
        return RESU


    def MCMC(self, *args):
        if args:
            #kplan, mod_lims, ins_lims, acc_lims, sigmas_raw, pos0, logl, logp
            kplanets, boundaries, inslims = args[0], args[1], args[2]
            acc_lims, sigmas_raw, pos0 = args[3], args[4], args[5]
            logl, logp = args[6], args[7]
        ndim = 1 + 5 * kplanets + self.nins*2*(self.MOAV+1) + self.totcornum + self.PACC
        print(str(self.PM)), 'self.pm!!'  # PMPMPM
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

            s0 = chrono.time()
            for _ in range(10000):
                empmir.logp_rv(pos0[0][0], logp_params)
            print('________', chrono.time()-s0)

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
        burn_out = self.burn_out
        assert self.cores >= 1, 'Cores is set to 0 ! !'
        assert self.thin * self.draw_every_n < self.nsteps, 'You are thining way too hard ! !'
        if self.betas is not None:
            assert len(self.betas) == self.ntemps, 'Betas array and ntemps dont match ! !'

        if self.MUSIC:
            mixer.init()
            s = mixer.Sound('mediafiles/imperial_march.wav')
            thybiding = mixer.Sound('mediafiles/swvader04.wav')
            technological_terror = mixer.Sound('mediafiles/technological.wav')
            s.play()

        kplan = from_k
        # for k = 0
        mod_lims = sp.array([])
        acc_lims = sp.array([-1., 1.])
        jitt_limiter = sp.amax(abs(self.rv))
        jitt_lim = 3 * jitt_limiter  # review this
        offs_lim = jitt_limiter  # review this
        ins_lims = sp.array([sp.append(sp.array([0.0001, jitt_lim, -offs_lim, offs_lim]), sp.array([sp.array([-1.0, 1.0, 0.1, 10]) for j in range(self.MOAV)])) for i in range(self.nins)]).reshape(-1)
        sqrta, sqrte = jitt_lim, 1.
        sqrta, sqrte = sqrta ** 0.5, sqrte ** 0.5
        ndim = kplan * 5 + self.nins*2*(self.MOAV+1) + self.totcornum + 1 + self.PACC
        free_lims = sp.array([sp.log(0.1), sp.log(3 * max(self.time)), -sqrta, sqrta, -sqrta, sqrta, -sqrte, sqrte, -sqrte, sqrte])
        sigmas, sigmas_raw = sp.zeros(ndim), sp.zeros(ndim)
        pos0 = 0.
        thetas_hen, ajuste_hen = 0., 0.
        ajuste_raw = sp.array([0])
        oldlogpost = -999999999.
        interesting_thetas, interesting_posts = sp.array([]), sp.array([])
        thetas_raw = sp.array([])
        #####################################################
        #if self.PM:
        #    pm_lims = sp.array([min(self.time_pm), max(self.time_pm), 0.0, 1.0, min(self.rv_pm), max(self.rv_pm), 0.1, 10])
            #                           t0min           t0max      ratiomin ratiomax kamin          ka_max      kr_min   kr_max
        #####################################################
        START = chrono.time()
        while kplan <= to_k:

            mod_lims = sp.array([free_lims for i in range(kplan)]).reshape(-1)
            ins_lims = sp.array([sp.append(sp.array([0.0001, jitt_lim, -offs_lim, offs_lim]), sp.array([sp.array([-1.0, 1.0, 0.1, 10]) for j in range(self.MOAV)])) for i in range(self.nins)]).reshape(-1)
            #if LIL_JITT:
            #    ins_lims = ins_lims = sp.array([sp.append(sp.array([0.0001, 10.0, -offs_lim, offs_lim]), sp.array([sp.array([-1.0, 1.0, 0.1, 10]) for j in range(self.MOAV)])) for i in range(self.nins)]).reshape(-1)
            if kplan > 0:
                if self.PM:
                    ndim += self.fsig*self.lenppm
                    t0min, t0max = min(self.time_pm), max(self.time_pm)
                    pm_lims = sp.array([t0min, t0max, 0.0, 1.0, min(self.rv_pm), max(self.rv_pm), -10, 10])  # makes sense????
                    #                   t0min  t0max ratiomin ratiomax kamin       ka_max       kr_min  kr_max
                    print(pm_lims, pm_lims.shape, 'pm_lims and shape\n\n')  # PMPMPM
                    print(self.fsig*self.lenppm, 'ndim en pm\n\n')  # PMPMPM

                if self.CONSTRAIN and ajuste_raw[0]:
                    constrained = sp.array([])
                    for k in range(kplan - 1):
                        amp = 2.0
                        Pk, Ask, Ack, Sk, Ck = ajuste_raw[k*5:(k+1)*5]
                        Pk_std, Ask_std, Ack_std, Sk_std, Ck_std = sigmas_raw[5*k:5*(k+1)]
                        Ask_std, Ack_std, Sk_std, Ck_std = amp * sp.array([Ask_std, Ack_std, Sk_std, Ck_std])
                        aux = sp.array([Pk-Pk_std, Pk+Pk_std, Ask - Ask_std, Ask + Ask_std, Ack - Ack_std, Ack + Ack_std, Sk - Sk_std, Sk + Sk_std, Ck - Ck_std, Ck + Ck_std])

                        constrained = sp.append(constrained, aux)

                    for nin in range(self.nins):  # only constrains
                        ins_lims[0 + nin*4*(self.MOAV+1)] = 0.0001
                        ins_lims[1 + nin*4*(self.MOAV+1)] = ajuste_raw[(kplan - 1) * 5 + nin*2*(self.MOAV+1) + 1 + self.PACC]

                    mod_lims = sp.append(constrained, free_lims)  # only planets limits

            if len(BOUND) != 0:
                nn = len(BOUND)
                for j in range(len(BOUND[:kplan].reshape(-1))):
                    if BOUND[:kplan].reshape(-1)[j] != -sp.inf:
                        mod_lims[j] = BOUND[:kplan].reshape(-1)[j]
                if nn <= kplan:
                    BOUND = sp.array([])

            #print('ins_lims', ins_lims)  # testing purposes
            #print('mod_lims', mod_lims)
            #print('ajuste_raw', ajuste_raw)

            if self.breakFLAG==True:
                break

            if self.PM:
                if kplan > 0:
                    pos0 = emplib.pt_pos_rvpm(self.setup, kplan, self.nins, mod_lims, ins_lims,
                                              acc_lims, self.MOAV, self.totcornum, self.PACC,
                                              self.fsig, self.lenppm, self.nins_pm, pm_lims)
                    print(pos0, pos0.shape, 'pos0 y shape\n\n')  # PMPMPM
                    thetas_raw, ajuste_raw, thetas_hen, ajuste_hen, p, lnprob, lnlike, posteriors, betas, interesting_thetas, interesting_posts, sigmas, sigmas_raw = self.MCMC(kplan, mod_lims, ins_lims, acc_lims, sigmas_raw, pos0, logl, logp, pm_lims)
                else:
                    pos0 = emplib.pt_pos(self.setup, kplan, self.nins, mod_lims, ins_lims,
                                         acc_lims, self.MOAV, self.totcornum, self.PACC)
                    thetas_raw, ajuste_raw, thetas_hen, ajuste_hen, p, lnprob, lnlike, posteriors, betas, interesting_thetas, interesting_posts, sigmas, sigmas_raw = self.MCMC(kplan, mod_lims, ins_lims, acc_lims, sigmas_raw, pos0, logl, logp)
            else:
                pos0 = emplib.pt_pos(self.setup, kplan, self.nins, mod_lims, ins_lims,
                                     acc_lims, self.MOAV, self.totcornum, self.PACC)
                thetas_raw, ajuste_raw, thetas_hen, ajuste_hen, p, lnprob, lnlike, posteriors, betas, interesting_thetas, interesting_posts, sigmas, sigmas_raw = self.MCMC(kplan, mod_lims, ins_lims, acc_lims, sigmas_raw, pos0, logl, logp)
            chain = thetas_hen
            fit = ajuste_hen
            sample_sizes = sp.array([len(interesting_thetas[i]) for i in range((len(interesting_thetas)))])
            bestlogpost = max(posteriors[0])

            # BIC
            NEW_BIC = sp.log(self.ndat) * ndim - 2 * bestlogpost
            OLD_BIC = sp.log(self.ndat) * ndim - 2 * oldlogpost
            NEW_AIC = 2 * ndim - 2 * bestlogpost
            OLD_AIC = 2 * ndim - 2 * oldlogpost
                # theta, time, kplanets, ins, staract, starflag, kplanets, nins, MOAV, totcornum, PACC

            residuals = empmir.RV_residuals(ajuste_raw, self.rv, self.time, self.ins, self.staract, self.starflag, kplan, self.nins, self.MOAV, self.totcornum, self.PACC)
            alt_res = self.alt_results(interesting_thetas[0], kplan)
            saveplace = self.mklogfile(fit, bestlogpost, sample_sizes, sigmas, kplan, mod_lims, NEW_BIC, NEW_AIC, alt_res, START, residuals)
            self.instigator(interesting_thetas, interesting_posts, saveplace, kplan)
            if self.MUSIC:
                thybiding.play()
            if self.PLOT:
                from emperors_canvas import plot1, plot2
                plug = sp.array([self.setup, kplan, self.nins, self.totcornum,
                                 saveplace, self.MOAV, self.PACC])

                plot2(self.all_data, plug, fit, self.starflag, self.staract,
                      self.ndat)

                plug2 = sp.array([self.HISTOGRAMS, self.CORNER, self.STARMASS,
                                  self.PNG, self.PDF, self.thin, self.draw_every_n])
                plot1(interesting_thetas, interesting_posts, plug, plug2, '0')

                if self.RAW:
                    rawplace = str(saveplace)+'/RAW'
                    os.makedirs(rawplace)
                    self.instigator(thetas_hen, posteriors, rawplace, kplan)
                    plug[4] = rawplace
                    plot1(thetas_hen, posteriors, plug, plug2, '0')

            if OLD_BIC - NEW_BIC < self.BIC:
                print('\nBayes Information Criteria of %.2f requirement not met ! !' % self.BIC)
            if OLD_AIC - NEW_AIC < self.AIC:
                print('\nAkaike Information Criteria of %.2f requirement not met ! !' % self.AIC)
                print(OLD_AIC, NEW_AIC, OLD_AIC - NEW_AIC)

            print('New logpost vs. Old logpost', bestlogpost, oldlogpost, bestlogpost - oldlogpost)
            print('Old BIC vs New BIC', OLD_BIC, NEW_BIC, OLD_BIC - NEW_BIC)
            print('Old AIC vs New AIC', OLD_AIC, NEW_AIC, OLD_AIC - NEW_AIC)

            if bestlogpost - oldlogpost < self.model_comparison:
                print('\nBayes Factor of %.2f requirement not met ! !' % self.model_comparison)
                break

            oldlogpost = bestlogpost
            kplan += 1
        if self.MUSIC:
            technological_terror.play()
        return pos0, chain, fit, thetas_raw, ajuste_raw, mod_lims, posteriors, bestlogpost, interesting_thetas, interesting_posts, sigmas, sigmas_raw















#

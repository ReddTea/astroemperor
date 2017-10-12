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
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

import emcee
from emcee import PTSampler
from PyAstronomy.pyasl import MarkleyKESolver
from PyAstronomy.pyasl import foldAt
from decimal import Decimal  # histograms
import corner
import time as chrono
import multiprocessing
import datetime as dt
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

def read_data(instruments):
    '''
    Data pre-processing
    '''
    nins = len(instruments)
    instruments = sp.array([sp.loadtxt('datafiles/'+x) for x in instruments])
    def data(data, ins_no):
        Time, Radial_Velocity, Err = data.T[:3]  # el error de la rv
        Radial_Velocity -= sp.mean(Radial_Velocity)
        Flag = sp.ones(len(Time)) * ins_no  # marca el instrumento al q pertenece
        Staract = data.T[3:]
        return sp.array([Time, Radial_Velocity, Err, Flag, Staract])

    def sortstuff(tryin):
        t, rv, er, flag = tryin
        order = sp.argsort(t)
        return sp.array([x[order] for x in [t, rv, er, flag]])

    fd = sp.array([]), sp.array([]), sp.array([]), sp.array([])

    for k in range(len(instruments)):  # appends all the data in megarg
        t, rv, er, flag, star = data(instruments[k], k)
        fd = sp.hstack((fd, [t, rv, er, flag] ))  # ojo this, list not array

    fd[0] = fd[0] - min(fd[0])
    alldat = sp.array([])
    try:
        staract = sp.array([data(instruments[i], i)[4] for i in range(nins)])
    except:
        staract = sp.array([sp.array([]) for i in range(nins)])
    starflag = sp.array([sp.array([i for k in range(len(staract[i]))]) for i in range(len(staract))])
    tryin = sortstuff(fd)
    for i in range(len(starflag)):
        for j in range(len(starflag[i])):
            staract[i][j] -= sp.mean(staract[i][j])
    totcornum = 0
    for correlations in starflag:
        if len(correlations) > 0:
            totcornum += len(correlations)

    return fd, staract, starflag, totcornum

def logp(theta, time, kplanets, nins, MOAV, totcornum, boundaries, inslims, acc_lims, sigmas, eccprior, jittprior, jittmean, STARMASS, HILL, CHECK):
    G = 39.5 ##6.67408e-11 * 1.9891e30 * (1.15740741e-5) ** 2  # in Solar Mass-1 s-2 m3
    lp_flat_fix, lp_flat_ins, lp_ecc, lp_jitt = 0., 0., 0., 0.
    lp_correl = 0.
    lp_jeffreys = 0.

    model_params = kplanets * 5
    ins_params = nins * 4
    MP = sp.zeros(kplanets)
    SMA, GAMMA = sp.zeros(kplanets), sp.zeros(kplanets)
    for k in range(kplanets):
        Ask, Pk, Ack, Sk, Ck = theta[k*5:(k+1)*5]

        if (boundaries[k*10] <= Ask <= boundaries[k*10+1] and
            boundaries[k*10+4] <= Ack <= boundaries[k*10+5] and
            boundaries[k*10+6] <= Sk <= boundaries[k*10+7] and
            boundaries[k*10 + 8] <= Ck <= boundaries[k*10 + 9]):
            lp_flat_fix += 0.0
            #print('GOOD_ONE_1')
        else:
            if CHECK:
                print('mark 1', boundaries[k*10],Ask, boundaries[k*10+1],
                      boundaries[k*10+4], Ack, boundaries[k*10+5],
                      boundaries[k*10+6], Sk, boundaries[k*10+7],
                      boundaries[k*10 + 8], Ck, boundaries[k*10 + 9])
            return -sp.inf

        Pmin, Pmax = boundaries[k*10+2], boundaries[k*10+3]
        if Pmin <= Pk <= Pmax:
            #lp_jeffreys += sp.log(Pk**-1 / (sp.log(Pmax/Pmin)))
            lp_jeffreys += 0.  # Because it is logp ??
            #print('GOOD_ONE_2', Pmin, Pk, Pmax)
        else:
            if CHECK:
                print('mark 2', Pmin, Pk, Pmax)
            return -sp.inf

        Ak = Ask ** 2 + Ack ** 2
        ecck = Sk ** 2 + Ck ** 2

        if 0.0 <= ecck <= 1.0:
            lp_ecc += normal_pdf(ecck, 0, eccprior**2)
        else:
            if CHECK:
                print('ecc')
            return -sp.inf



        if kplanets > 1:
            if HILL:
                per_s = Pk * 24. * 3600.
                SMA[k] = ((per_s**2.0) / ( (4.0*sp.pi**2.0) / (6.67e-11 * STARMASS * 1.99e30) ))**(1./3) / 1.49598e11
                MP[k] = Ak / ( (28.4/sp.sqrt(1. - ecck**2.)) * (STARMASS**(-0.5)) * (SMA[k]**(-0.5)) ) * 317.8
                #SMA[k] = (((sp.exp(Pk)/365.242199) ** 2) * G * STARMASS / (4 * sp.pi ** 2)) ** (1/3.)
                #MP[k] = sp.sqrt(1 - ecck ** 2) * Ak * sp.exp(Pk) ** (1/3.) * STARMASS ** (2/3.) / 203.
                GAMMA[k] = sp.sqrt(1 - ecck)
    if kplanets > 1:
        if HILL:
            orden = sp.argsort(SMA)
            SMA = SMA[orden]  # in AU
            MP = MP[orden]  # in Earth Masses
            #M = STARMASS * 1047.56 + sp.sum(MP)  # to Jupyter masses
            M = STARMASS * 332946 + sp.sum(MP)  # to Earth Masses

            mu = MP / M
            for kk in range(kplanets-1):
                alpha = mu[kk] + mu[kk+1]
                delta = sp.sqrt(SMA[kk+1] / SMA[kk])

                if alpha ** -3 * (mu[kk] + (mu[kk+1] / (delta ** 2))) * (mu[kk] * GAMMA[kk] + mu[kk+1] * GAMMA[kk+1] * delta)**2 < 1 + (3./alpha)**(4./3) * (mu[kk] * mu[kk+1]):
                    if CHECK:
                        print('HILL UNSTABLE')
                    return -sp.inf
                else:
                    pass

    acc_k = theta[kplanets * 5]
    if acc_lims[0] <= acc_k <= acc_lims[1]:
        lp_flat_fix += 0
    else:
        if CHECK:
            print('ACCEL')
        return -sp.inf

    j = 0
    lp_flat_ins = 0.0
    lp_jitt = 0.0
    for j in range(nins):
        for c in range(MOAV):
            macoef_j = theta[model_params + j*2*(MOAV+1) + 2*(c+1) + 1]
            timescale_j = theta[model_params + j*2*(MOAV+1) + 2*(c+1) + 2]
            bookmark = 4 * (j*MOAV + j + c + 1)
            if (inslims[bookmark] <= macoef_j <= inslims[bookmark+1] and
                inslims[bookmark+2] <= timescale_j <= inslims[bookmark+3]):
                lp_flat_ins += 0.0
            else:
                if CHECK:
                    print('MOVING AVERAGE')
                return -sp.inf
        jitt_j = theta[model_params + j*2*(MOAV+1) + 1]
        offset_j = theta[model_params + j*2*(MOAV+1) + 2]
        jittmin, jittmax = inslims[4*(j*MOAV+j)], inslims[4*(j*MOAV+j)+1]
        if jittmin <= jitt_j <= jittmax:
            lp_jitt += normal_pdf(jitt_j, jittmean, jittprior**2)
        else:
            if CHECK:
                print('SI JITT')
            return -sp.inf
        if inslims[4*(j*MOAV+j)+2] <= offset_j <= inslims[4*(j*MOAV+j)+3]:
            lp_flat_ins += 0.0
        else:
            if CHECK:
                print('NO JITT')
            return -sp.inf
    for h in range(totcornum):
        cork = theta[model_params + ins_params + 1 + h]
        if acc_lims[0] <= cork <= acc_lims[1]:
            lp_correl += 0
        else:
            return -sp.inf

    return lp_ecc + lp_flat_fix + lp_flat_ins + lp_jitt + lp_correl

def logl(theta, time, rv, err, ins, staract, starflag, kplanets, nins, MOAV, totcornum):
    i, lnl = 0, 0
    ndat = len(time)
    model_params = kplanets * 5
    ins_params = nins * 2 * (MOAV + 1)
    jitter, offset, macoef, timescale = sp.zeros(ndat), sp.zeros(ndat), sp.array([sp.zeros(ndat) for i in range(MOAV)]), sp.array([sp.zeros(ndat) for i in range(MOAV)])
    ACC = theta[model_params] * (time - time[0])

    residuals = sp.zeros(ndat)
    for i in range(ndat):
        jitpos = int(model_params + ins[i] * 2 * (MOAV+1) + 1)
        jitter[i], offset[i] = theta[jitpos], theta[jitpos + 1]  # jitt
        for j in range(MOAV):
            macoef[j][i], timescale[j][i] = theta[jitpos + 2*(j+1)], theta[jitpos + 2*(j+1) + 1]
    a1 = (theta[:model_params])

    if totcornum:
        COR = sp.array([sp.array([sp.zeros(ndat) for k in range(len(starflag[i]))]) for i in range(len(starflag))])
        SA = theta[model_params+ins_params+1:]

        assert len(SA) == totcornum, 'error in correlations'
        AR = 0.0  # just to remember to add this
        counter = -1

        for i in range(nins):
            for j in range(len(starflag[i])):
                counter += 1
                passer = -1
                for k in range(ndat):
                    if starflag[i][j] == ins[k]:  #
                        passer += 1
                        COR[i][j][k] = SA[counter] * staract[i][j][passer]

        FMC = 0
        for i in range(len(COR)):
            for j in range(len(COR[i])):
                FMC += COR[i][j]
    else:
        FMC = 0

    MODEL = model(a1, time, kplanets) + offset + ACC + FMC


    for i in range(ndat):
        residuals[i] = rv[i] - MODEL[i]
        for c in range(MOAV):
            if i > c:
                MA = macoef[c][i] * sp.exp(-sp.fabs(time[i-1] - time[i]) / timescale[c][i]) * residuals[i-1]
                residuals[i] -= MA

    inv_sigma2 = 1.0 / (err**2 + jitter**2)
    lnl = sp.sum(residuals ** 2 * inv_sigma2 - sp.log(inv_sigma2)) + sp.log(2*sp.pi) * ndat
    return -0.5 * lnl

def normal_pdf(x, mean, variance):
    var = 2 * variance
    return ( - (x - mean) ** 2 / var)

def model(THETA, time, kplanets):
    modelo = 0.0
    if kplanets == 0:
        return 0.0
    for i in range(kplanets):
        As, P, Ac, S, C = THETA[5*i:5*(i+1)]
        A = As ** 2 + Ac ** 2
        ecc = S ** 2 + C ** 2
        w = sp.arccos(C / (ecc ** 0.5))  # longitude of periastron
        phase = sp.arccos(Ac / (A ** 0.5))
        ### test
        if S < 0:
            w = 2 * sp.pi - sp.arccos(C / (ecc ** 0.5))
        if As < 0:
            phase = 2 * sp.pi - sp.arccos(Ac / (A ** 0.5))
        ###
        per = sp.exp(P)
        freq = 2. * sp.pi / per
        M = freq * time + phase
        E = sp.array([MarkleyKESolver().getE(m, ecc) for m in M])
        f = (sp.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * sp.tan(E / 2.)) * 2.)
        modelo += A * (sp.cos(f + w) + ecc * sp.cos(w))
    return  modelo

class EMPIRE:
    def __init__(self, stardat, setup):
        assert len(stardat) >= 1, 'stardat has to contain at least 1 file ! !'
        assert len(setup) == 3, 'setup has to be [ntemps, nwalkers, nsteps]'
        #  Setup
        self.cores = multiprocessing.cpu_count()
        self.ntemps, self.nwalkers, self.nsteps = setup

        self.burn_out = self.nsteps // 2
        #  Reading data
        tryout = read_data(stardat)
        self.time, self.rv, self.err, self.ins = tryout[0]  # time, radial velocities, error and instrument flag
        self.betas = None
        self.staract, self.starflag = tryout[1], tryout[2]  # star activity index and flag
        self.totcornum = tryout[3]  # quantity if star activity indices
        self.stardat = stardat

        #  Statistical Tools
        self.bayes_factor = sp.log(150)  # inside chain comparison (smaller = stricter)
        self.model_comparison = 5
        self.BIC = 5
        self.AIC = 5

        self.nins = len(stardat)  # number of instruments autodefined
        self.ndat = len(self.time)  # number of datapoints
        self.MOAV = 1

        #  Menudencies
        self.thin = 1
        self.PLOT = True
        self.draw_every_n = 1
        self.PNG = True
        self.PDF = False
        self.CORNER = True
        self.HISTOGRAMS = True
        self.starname = stardat[0].split('_')[0]
        self.MUSIC = False
        self.breakFLAG = False
        self.STARMASS = False
        self.HILL = False
        self.CHECK = False

        self.CONSTRAIN = True
        self.eccprior = 0.3
        self.jittprior = 5.0
        self.jittmean = 5.0

    def semimodel(self, params, time):
        A, P, phase, w, ecc = params
        freq = 2. * sp.pi / P
        M = freq * time + phase
        E = sp.array([MarkleyKESolver().getE(m, ecc) for m in M])
        f = (sp.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * sp.tan(E / 2.)) * 2.)
        modelo = A * (sp.cos(f + w) + ecc * sp.cos(w))
        return modelo

    def henshin(self, thetas, kplanets):
        for i in range(kplanets):
            Ask = thetas[:, i*5]
            Pk = thetas[:, i*5 + 1]
            Ack = thetas[:, i*5 + 2]
            Sk = thetas[:, i*5 + 3]
            Ck = thetas[:, i*5 + 4]

            ecck  = Sk ** 2 + Ck ** 2
            Ak = Ask ** 2 + Ack ** 2
            wk = sp.arccos(Ck / (ecck ** 0.5))
            Phasek = sp.arccos(Ack / (Ak ** 0.5))
            for j in range(len(Sk)):
                if Sk[j] < 0:
                    wk[j] = 2 * sp.pi - sp.arccos(Ck[j] / (ecck[j] ** 0.5))
                if Ask[j] < 0:
                    Phasek[j] = 2 * sp.pi - sp.arccos(Ack[j] / (Ak[j] ** 0.5))

            thetas[:, i*5] = Ak
            thetas[:, i*5 + 1] = sp.exp(Pk)
            thetas[:, i*5 + 2] = Phasek
            thetas[:, i*5 + 3] = wk
            thetas[:, i*5 + 4] = ecck
        return thetas

    ########################################
    # los IC sirven para los mod_lims? NO!
    # mod_lims sólo acotan el espacio donde buscar, excepto para periodo
    def mklogfile(self, theta_max, best_post, sample_sizes, sigmas, kplanets, modlims, BIC, AIC, alt_res, START):
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
            sigmas_hen = sigmas
            logdat = '\nStar Name                         : '+self.starname
            for i in range(self.nins):
                if i==0:
                    logdat += '\nUsed datasets                     : '+self.stardat[i]
                else:
                    logdat += '\n                                  : '+self.stardat[i]
            logdat += '\n--------------------------------------------------------------------'
            logdat += '\nThe sample sizes are        :    ' + str(sample_sizes)
            logdat += '\nThe maximum posterior is    :    ' + str(best_post)
            logdat += '\nThe BIC is                  :    ' + str(BIC)
            logdat += '\nThe AIC is                  :    ' + str(AIC)
            logdat += '\nThe most probable chain values are as follows...'
            for i in range(kplanets):
                #MP = sp.sqrt(1 - theta[i*5+4] ** 2) * theta[i*5] * theta[i*5+1] ** (1/3.) * STARMASS ** (2/3.) / 203.
                #SMA = (((theta[i*5+1]/365.242199) ** 2) * G * STARMASS0 / (4 * sp.pi ** 2)) ** (1/3.)
                if self.STARMASS:
                    SMA = (((theta[i*5+1]*24.*3600.)**2.0) / ( (4.0*sp.pi**2.0) / (6.67e-11 * self.STARMASS * 1.99e30) ))**(1./3) / 1.49598e11
                    MP = theta[i*5] / ( (28.4/sp.sqrt(1. - theta[i*5+4]**2.)) * (self.STARMASS**(-0.5)) * (SMA**(-0.5)) ) * 317.8

                logdat += '\n--------------------------------------------------------------------'
                logdat += '\nAmplitude   '+str(i+1)+'[JD]:   ' + str(theta[i*5]) + ' +- ' + str(sigmas_hen[i*5])
                logdat += '\nPeriod   '+str(i+1)+'[days] :   ' + str(theta[i*5+1]) + ' +- ' + str(sigmas_hen[i*5+1])
                logdat += '\nPhase   '+str(i+1)+'        :   ' + str(theta[i*5+2]) + ' +- ' + str(sigmas_hen[i*5+2])
                logdat += '\nLongitude   '+str(i+1)+'    :   ' + str(theta[i*5+3]) + ' +- ' + str(sigmas_hen[i*5+3])
                logdat += '\nEccentricity   '+str(i+1)+' :   ' + str(theta[i*5+4]) + ' +- ' + str(sigmas_hen[i*5+4])
                if self.STARMASS:
                    logdat += '\nMinimum Mass   '+str(i+1)+' :   ' + str(MP)
                    logdat += '\nSemiMajor Axis '+str(i+1)+' :   ' + str(SMA)
            logdat += '\n--------------------------------------------------------------------'
            logdat += '\nAcceleration [m/(s days)]:'+str(theta[5*kplanets]) + ' +- ' + str(sigmas_hen[5*kplanets])

            for i in range(self.nins):
                logdat += '\n--------------------------------------------------------------------'
                logdat += '\nJitter '+str(i+1)+'    [m/s]:   ' + str(theta[5*kplanets + i*2*(self.MOAV+1) + 1]) + ' +- ' + str(sigmas_hen[5*kplanets + i*2*(self.MOAV+1) + 1])
                logdat += '\nOffset '+str(i+1)+'    [m/s]:   ' + str(theta[5*kplanets + i*2*(self.MOAV+1) + 2]) + ' +- ' + str(sigmas_hen[5*kplanets + i*2*(self.MOAV+1) + 2])
                for j in range(self.MOAV):
                    logdat += '\nMA coef '+str(i+1)+'_'+str(j+1)+'        : ' + str(theta[5*kplanets + i*2*(j+1) + 3]) + ' +- ' + str(sigmas_hen[5*kplanets + i*2*(j+1) + 3])
                    logdat += '\nTimescale '+str(i+1)+'_'+str(j+1)+'[days]: ' + str(theta[5*kplanets + i*2*(j+1) + 4]) + ' +- ' + str(sigmas_hen[5*kplanets + i*2*(j+1) + 4])

            for h in range(self.totcornum):
                logdat += '\n--------------------------------------------------------------------'
                logdat += '\nStellar Activity'+str(h+1)+':   ' + str(theta[5*kplanets + self.nins*2*(self.MOAV+1) + 1 + h]) + ' +- ' + str(sigmas_hen[5*kplanets + self.nins*2*(self.MOAV+1) + 1 + h])

            logdat += '\n--------------------------------------------------------------------'
            logdat += '\nTemperatures, Walkers, Steps      : '+str((self.ntemps, self.nwalkers, self.nsteps))
            logdat += '\nN Instruments, K planets, N data  : '+str((self.nins, kplanets, self.ndat))
            logdat += '\nNumber of Dimensions              : '+str(1 + 5 * kplanets + self.nins*2*(self.MOAV+1) + self.totcornum)
            logdat += '\nN Moving Average                  : '+str(self.MOAV)
            logdat += '\nBeta Detail                       : '+str(self.betas)
            logdat += '\n--------------------------------------------------------------------'
            logdat += '\nRunning Time                      : '+timer()
            print(logdat)
            logdat += '\n -------------------------- A D V A N C E D --------------------------'
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
        return name

    def pt_pos(self, kplanets, boundaries, inslims, acc_lims):
        ndim = 1 + 5 * kplanets + self.nins*2*(self.MOAV+1) + self.totcornum
        pos = sp.array([sp.zeros(ndim) for i in range(self.nwalkers)])
        k = -2
        l = -2
        ll = -2  ##
        for j in range(ndim):
            if j < 5 * kplanets:
                k += 2
                if j%5==1:
                    fact = sp.absolute(boundaries[k] - boundaries[k+1]) / self.nwalkers
                else:
                    #fact = sp.absolute(boundaries[k]) / (self.nwalkers)
                    fact = (sp.absolute(boundaries[k] - boundaries[k+1]) * 2) / (5 * self.nwalkers)
                dif = sp.arange(self.nwalkers) * fact * np.random.uniform(0.9, 0.999)
                for i in range(self.nwalkers):
                    if j%5==1:
                        pos[i][j] = boundaries[k] + (dif[i] + fact/2.0)
                    else:
                        #pos[i][j] = boundaries[k] * 0.5 + (dif[i] + fact/2.0)
                        pos[i][j] = (boundaries[k+1]+3*boundaries[k])/4 + (dif[i] + fact/2.0)
            if j == 5 * kplanets:  # acc
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / self.nwalkers
                dif = sp.arange(self.nwalkers) * fact * np.random.uniform(0.9, 0.999)
                for i in range(self.nwalkers):
                    pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)

            if 5 * kplanets < j < 5*kplanets + self.nins*2*(self.MOAV+1) + 1:
                l += 2
                fact = sp.absolute(inslims[l] - inslims[l+1]) / self.nwalkers
                dif = sp.arange(self.nwalkers) * fact * np.random.uniform(0.9, 0.999)

                if (j-5*kplanets-1) % self.nins*2*(self.MOAV+1) == 0:  # ojo aqui
                    jitt_ini = sp.sort(sp.fabs(sp.random.normal(0, 1, self.nwalkers))) * 0.1
                    dif = jitt_ini * np.random.uniform(0.9, 0.999)
                for i in range(self.nwalkers):
                    pos[i][j] = inslims[l] + (dif[i] + fact/2.0)
            if self.totcornum:
                if j > 5*kplanets + self.nins*2*(self.MOAV+1):
                    fact = sp.absolute(acc_lims[0] - acc_lims[1]) / self.nwalkers

                    dif = sp.arange(self.nwalkers) * fact * np.random.uniform(0.8, 0.999)
                    for i in range(self.nwalkers):
                        pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)
                        #print(pos[i][j])

        pos = sp.array([pos for h in range(self.ntemps)])
        return pos

    def plot1(self, thetas, flattened, temp, kplanets, saveplace, ticknum=10):
        def gaussian(x, mu, sig):
            return np.exp(-np.power((x - mu)/sig, 2.)/2.)

        def plot(thetas, flattened, temp, kplanets, CORNER=False, ticknum=ticknum):
            ndim = 1 + 5 * kplanets + self.nins*2*(self.MOAV+1) + self.totcornum

            titles = sp.array(["Amplitude","Period","Longitude", "Phase","Eccentricity", 'Acceleration', 'Jitter', 'Offset', 'MACoefficient', 'MATimescale', 'Stellar Activity'])
            units = sp.array([" $[\\frac{m}{s}]$"," [Days]"," $[rad]$", " $[rads]$","", ' $[\\frac{m}{s^2}]$', ' $[\\frac{m}{s}]$', ' $[\\frac{m}{s}]$', ' $[\\frac{m}{s}]$', ' [Days]', ''])

            thetas = thetas[:-( len(thetas)%self.nwalkers )]
            flattened = flattened[:-( len(flattened)%self.nwalkers )]
            quasisteps = len(thetas)//self.nwalkers

            color = sp.arange(quasisteps)
            colores = sp.array([color for i in range(self.nwalkers)]).reshape(-1)
            i = 0
            sorting = sp.arange(len(thetas))

            subtitles, namen = sp.array([]), sp.array([])
            for k in range(kplanets):
                subtitles = sp.append(subtitles, [titles[i] + ' '+str(k+1)+units[i] for i in range(5)])
                namen = sp.append(namen, [titles[i] + '_'+str(k) for i in range(5)])

            subtitles = sp.append(subtitles, titles[5]+units[5])  # for acc
            namen = sp.append(namen, titles[5])  # for acc
            for i in range(self.nins):
                subtitles = sp.append(subtitles, [titles[ii] + ' '+str(i+1)+units[ii] for ii in sp.arange(2)+6])
                namen = sp.append(namen, [titles[ii] + '_'+str(i+1) for ii in sp.arange(2)+6])
                for j in range(self.MOAV):
                    subtitles = sp.append(subtitles, [titles[ii] + ' '+str(i+1) + ' '+str(j+1)+units[ii] for ii in sp.arange(2)+8])
                    namen = sp.append(namen, [titles[ii] + '_'+str(i+1) + '_'+str(j+1) for ii in sp.arange(2)+8])

            for h in range(self.totcornum):
                subtitles = sp.append(subtitles, titles[-1]+' '+str(h+1))
                namen = sp.append(namen, titles[-1]+'_'+str(h+1))

            print('\n PLOTTING CHAINS for temperature '+str(temp)+'\n')
            pbar_chain = tqdm(total=ndim)
            for i in range(ndim):  # chains
                fig, ax = plt.subplots(figsize=(12, 7))
                if subtitles[i][:3] == 'Per':
                    pass

                ydif = (max(thetas[:,i]) - min(thetas[:,i])) / 10.
                ax.set(ylim=(min(thetas[:,i]) - ydif, max(thetas[:,i]) + ydif))

                im = ax.scatter(sorting, thetas[:,i], c=colores, lw=0., cmap='gist_rainbow', alpha=0.8)
                plt.xlabel("N", fontsize=24)
                plt.ylabel(subtitles[i], fontsize=24)

                cb = plt.colorbar(im, ax=ax)
                lab = 'Step Number'

                if self.thin * self.draw_every_n != 1:
                    lab = 'Step Number * '+str(self.thin*draw_every_n)

                cb.set_label('Step Number')
                if self.PNG:
                    fig.savefig(saveplace+"/chains"+temp+'_'+str(i+1)+'_'+namen[i]+".png")
                if self.PDF:
                    fig.savefig(saveplace+"/chains"+temp+'_'+str(i+1)+'_'+namen[i]+".pdf")

                pbar_chain.update(1)
                plt.close('all')
            pbar_chain.close()

            print('\n PLOTTING POSTERIORS for temperature '+str(temp)+'\n')
            pbar_post = tqdm(total=ndim)
            for i in range(ndim):  # posteriors
                fig1, ax1 = plt.subplots(figsize=(12, 7))

                xdif1, ydif1 = (max(thetas[:,i]) - min(thetas[:,i])) / 10., (max(flattened) - min(flattened)) / 10.
                ax1.set(xlim=((min(thetas[:,i]) - xdif1), (max(thetas[:,i]) + xdif1)),
                        ylim=((min(flattened) - ydif1), (max(flattened) + ydif1)))

                im = ax1.scatter(thetas[:,i], flattened, s=10 , c=colores, lw=0., cmap='gist_rainbow', alpha=0.8)

                xaxis = ax1.get_xaxis()
                xaxis.set_major_locator(ticker.LinearLocator(numticks=ticknum))
                yaxis = ax1.get_yaxis()
                yaxis.set_major_locator(ticker.LinearLocator(numticks=ticknum))
                #yaxis.set_minor_locator(ticker.LinearLocator(numticks=5))
                '''
                if subtitles[i][:3] == 'Per':
                    ax1.set_xscale('log')
                    xaxis.set_major_locator(ticker.LogLocator(numticks=ticknum))
                '''
                ax1.axvline(thetas[sp.argmax(flattened), i], color='r', linestyle='--', linewidth=2, alpha=0.70)
                # ax1.invert_yaxis()

                plt.xlabel(subtitles[i], fontsize=24)
                plt.ylabel("Posterior", fontsize=24)

                cb = plt.colorbar(im, ax=ax1)
                lab = 'Step Number'
                if self.thin * self.draw_every_n != 1:
                    lab = 'Step Number * '+str(self.thin*self.draw_every_n)
                cb.set_label(lab)

                if self.PNG:
                    fig1.savefig(saveplace+"/posteriors"+temp+'_'+str(i+1)+'_'+namen[i]+".png")
                if self.PDF:
                    fig1.savefig(saveplace+"/posteriors"+temp+'_'+str(i+1)+'_'+namen[i]+".pdf")
                plt.close('all')

                pbar_post.update(1)
            pbar_post.close()

            if self.HISTOGRAMS:
                if kplanets == 0:
                    print 'Sorry! No histograms here yet! We are working on it ! '
                    pass
                print('\n PLOTTING HISTOGRAMS for temperature '+str(temp)+'\n')
                lab=['Amplitude [m/s]','Period [d]',r'$\phi$ [rads]',r'$\omega$ [rads]','Eccentricity','a [AU]',r'Msin(i) [$M_{\oplus}$]']
                params=len(lab)
                pbar_hist = tqdm(total=params*kplanets)
                num_bins = 25
                for k in range(kplanets):
                    per_s = thetas.T[5*k+1] * 24. * 3600.
                    if self.STARMASS:
                        semi = ((per_s**2.0) / ( (4.0*sp.pi**2.0) / (6.67e-11 * self.STARMASS * 1.99e30) ))**(1./3) / 1.49598e11 #AU!!
                        Mass = thetas.T[5*k] / ( (28.4/sp.sqrt(1. - thetas.T[5*k+4]**2.)) * (self.STARMASS**(-0.5)) * (semi**(-0.5)) ) * 317.8 #Me!!
                    else:
                        params = len(lab) - 2
                    for ii in range(params):
                        if ii < 5:
                            Per = thetas.T[5*k + ii]
                        if ii == 5:
                            Per = semi
                        if ii == 6:
                            Per = Mass

                        mu,sigma = norm.fit(Per)  # Mean and sigma of distribution!!
                        # first histogram of the data
                        n, bins, patches = plt.hist(Per, num_bins, normed=1)
                        plt.close("all")  # We don't need the plot just data!!

                        #Get the maximum and the data around it!!
                        maxi = Per[sp.where(flattened == sp.amax(flattened))][0]
                        dif = sp.fabs(maxi - bins)
                        his_max = bins[sp.where(dif == sp.amin(dif))]

                        res=sp.where(n == 0)[0]  # Find the zeros!!
                        if res.size:
                            if len(res) > 2:
                                for j in range(len(res)):
                                    if res[j+2] - res[j] == 2:
                                        sub=j
                                        break
                            else:
                                sub=res[0]

                            # Get the data subset!!
                            if bins[sub] > his_max:
                                post_sub=flattened[sp.where(Per <= bins[sub])]
                                Per_sub=Per[sp.where(Per <= bins[sub])]
                            else:
                                post_sub=flattened[sp.where(Per >= bins[sub])]
                                Per_sub=Per[sp.where(Per >= bins[sub])]

                        else:
                            Per_sub=Per
                            post_sub=flattened

                        plt.subplots(figsize=(12,7))  # Define the window size!!
                        # redo histogram of the subset of data
                        n, bins, patches = plt.hist(Per_sub, num_bins, normed=1, facecolor='blue', alpha=0.5)
                        mu, sigma = norm.fit(Per_sub)  # add a 'best fit' line
                        var = sigma**2.
                        #Some Stats!!
                        skew='%.4E' % Decimal(sp.stats.skew(Per_sub))
                        kurt='%.4E' % Decimal(sp.stats.kurtosis(Per_sub))
                        gmod='%.4E' % Decimal(bins[sp.where(n == sp.amax(n))][0])
                        med='%.4E' % Decimal(sp.median(Per_sub))
                        # print 'The skewness, kurtosis, mean, and median of the data are {} : {} : {} : {}'.format(skew,kurt,gmod,med)

                        #Make a model x-axis!!
                        span=bins[len(bins)-1] - bins[0]
                        bins_x=((sp.arange(num_bins*100.) / (num_bins*100.)) * span) + bins[0]

                        y = gaussian(bins_x, mu, sigma) * sp.amax(n) #Renormalised to the histogram maximum!!

                        axes = plt.gca()
                        #y = mlab.normpdf(bins, mu, sigma)
                        plt.plot(bins_x, y, 'r-',linewidth=3)

                        # Tweak spacing to prevent clipping of ylabel
                        plt.subplots_adjust(left=0.15)

                        #axes.set_xlim([])
                        axes.set_ylim([0.,sp.amax(n)+sp.amax(n)*0.7])

                        axes.set_xlabel(lab[ii],size=15)
                        axes.set_ylabel('Frequency',size=15)
                        axes.tick_params(labelsize=15)

                        plt.autoscale(enable=True, axis='x', tight=True)

                        #Get the axis positions!!
                        ymin, ymax = axes.get_ylim()
                        xmin, xmax = axes.get_xlim()

                        #Add a key!!
                        mu_o = '%.4E' % Decimal(mu)
                        sigma_o = '%.4E' % Decimal(sigma)
                        var_o = '%.4E' % Decimal(var)

                        axes.text(xmax - (xmax - xmin)*0.65, ymax - (ymax - ymin)*0.1, r"$\mathcal{N}(\mu_1,\sigma^2,\mu_3,\mu_4)$",size=25)
                        axes.text(xmax - (xmax - xmin)*0.8, ymax - (ymax - ymin)*0.180, r"$\mu_1 ={}$".format(mu_o),size=20)
                        axes.text(xmax - (xmax - xmin)*0.8, ymax - (ymax - ymin)*0.255, r"$\sigma^2 ={}$".format(var_o),size=20)
                        axes.text(xmax - (xmax - xmin)*0.8, ymax - (ymax - ymin)*0.330, r"$\mu_3 ={}$".format(skew),size=20)

                        axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.180, r"$\mu_4 ={}$".format(kurt),size=20)
                        axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.255, r"$Median ={}$".format(med),size=20)
                        axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.330, r"$Mode ={}$".format(gmod),size=20)

                        plt.savefig(saveplace+'/hist_test'+temp+'_'+str(k)+'_'+str(ii)+'.pdf') #,bbox_inches='tight')
                        plt.close('all')
                        pbar_hist.update(1)

                '''
                    if i < 5*kplanets and i%7==5:
                        plt.savefig(saveplace+"/histogram"+temp+'_'+str(i+1)+'_'+'SMA'+".pdf")
                    if i < 5*kplanets and i%7==6:
                        plt.savefig(saveplace+"/histogram"+temp+'_'+str(i+1)+'_'+'Mass'+".pdf")
                    else:
                        plt.savefig(saveplace+"/histogram"+temp+'_'+str(i+1)+'_'+namen[i]+".pdf")
                '''

                pbar_hist.close()


            if CORNER:
                try:
                    print('Plotting Corner Plot... May take a few seconds')
                    fig = corner.corner(thetas, labels=subtitles)
                    fig.savefig(saveplace+"/triangle.pdf")
                except:
                    print('Corner Plot Failed!!')
                    pass # corner
            try:
                plt.close('all')
            except:
                pass
            pass

        for i in range(self.ntemps):
            check_length = len(thetas[i])//self.nwalkers
            if check_length // self.draw_every_n < 100:
                self.draw_every_n = 1

            if i == 0:
                try:
                    plot(thetas[0][::self.draw_every_n], flattened[0][::self.draw_every_n], '0', kplanets, CORNER=self.CORNER)
                except:
                    print('Sample size insufficient to draw the posterior plots for the cold chain!')
                    pass
            else:
                try:
                    plot(thetas[i][::self.draw_every_n], flattened[i][::self.draw_every_n], str(i), kplanets)
                except:
                    print('Sample size insufficient to draw the posterior plots for temp '+str(i)+' ! !')
                    pass
        pass

    def plot2(self, fit, kplanets, saveplace, SHOW=False):

        def phasefold(TIME, RV, ERR, PER):
            phases = foldAt(TIME, PER, T0=0.0)
            sortIndi = sp.argsort(phases)  # sorts the points
            Phases = phases[sortIndi]  # gets the indices so we sort the RVs correspondingly(?)
            rv_phased = RV[sortIndi]
            time_phased = Phases * PER
            err_phased = ERR[sortIndi]
            return time_phased, rv_phased, err_phased

        def clear_noise(RV, ACC, theta_k, theta_i, theta_sa):
            '''
            This should clean offset, add jitter to err
            clear acc, red noise and stellar activity
            '''
            JITTER, OFFSET, MACOEF, MATS = sp.zeros(self.ndat), sp.zeros(self.ndat), sp.array([sp.zeros(self.ndat) for i in range(self.MOAV)]), sp.array([sp.zeros(self.ndat) for i in range(self.MOAV)])

            for i in range(self.ndat):
                jittpos = int(self.ins[i]*2*(self.MOAV+1))
                JITTER[i], OFFSET[i] = theta_i[jittpos], theta_i[jittpos + 1]
                for j in range(self.MOAV):
                    MACOEF[j][i], MATS[j][i] = theta_i[jittpos + 2*(j+1)], theta_i[jittpos + 2*(j+1)+1]

            ERR = sp.sqrt(self.err ** 2 + JITTER ** 2)
            tmin, tmax = np.min(self.time), np.max(self.time)

            #RV0 = RV - OFFSET - ACC * (time - time[0])
            if self.totcornum:
                COR = sp.array([sp.array([sp.zeros(self.ndat) for k in range(len(self.starflag[i]))]) for i in range(len(self.starflag))])
                assert len(theta_sa) == self.totcornum, 'error in correlations'
                AR = 0.0  # just to remember to add this
                counter = -1

                for i in range(self.nins):
                    for j in range(len(self.starflag[i])):
                        counter += 1
                        passer = -1
                        for k in range(self.ndat):
                            if self.starflag[i][j] == self.ins[k]:  #
                                passer += 1
                                COR[i][j][k] = theta_sa[counter] * self.staract[i][j][passer]

                FMC = 0
                for i in range(len(COR)):
                    for j in range(len(COR[i])):
                        FMC += COR[i][j]
            else:
                FMC = 0

            MODEL = OFFSET + ACC * (self.time - self.time[0]) + FMC

            for k in sp.arange(kplanets):
                MODEL += self.semimodel(theta_k[5*k:5*(k+1)], self.time)
            residuals, MA = sp.zeros(self.ndat), sp.zeros(self.ndat)
            for i in range(self.ndat):
                residuals = RV - MODEL
                for c in range(self.MOAV):
                    if i > c:
                        MA[i] = MACOEF[c][i] * sp.exp(-sp.fabs(self.time[i-1] - self.time[i]) / MATS[c][i]) * residuals[i-1]
                        MODEL[i] += MA[i]
                        residuals[i] -= MA[i]

            RV0 = RV - OFFSET - ACC * (self.time - self.time[0]) - FMC - MA

            return RV0, ERR, residuals

        ndim = 1 + 5 * kplanets + self.nins*2*(self.MOAV+1) + self.totcornum
        colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']
        letter = ['a', 'b', 'c', 'd', 'e', 'f']  # 'a' is just a placeholder

        theta_k = fit[:kplanets * 5]
        accel = fit[kplanets * 5]
        theta_i = fit[kplanets * 5 + 1:kplanets * 5 + self.nins*2*(self.MOAV+1) + 1]
        theta_sa = fit[kplanets * 5 + self.nins*2*(self.MOAV+1) + 1:]

        for k in range(kplanets):
            rv0, err0, residuals = clear_noise(self.rv, accel, theta_k, theta_i, theta_sa)
            rvk, errk = rv0, err0
            for kk in sp.arange(kplanets-1)+1:
                rvk -= self.semimodel(theta_k[5*kk:5*(kk+1)], self.time)
            t_p, rv_p, err_p = phasefold(self.time, rvk, errk, theta_k[1])
            t_p, res_p, err_p = phasefold(self.time, residuals, errk, theta_k[1])

            time_m = sp.linspace(min(self.time), max(self.time), int(1e4))
            rv_m = self.semimodel(theta_k[:5], time_m)

            for mode in range(2):
                fig = plt.figure(figsize=(20,10))
                gs = gridspec.GridSpec(3, 4)
                ax = fig.add_subplot(gs[:2, :])
                axr = fig.add_subplot(gs[-1, :])
                plt.subplots_adjust(hspace=0)

                for i in range(self.nins):  # printea datos separados por instrumento
                    x, y, yerr = sp.array([]), sp.array([]), sp.array([])
                    yr = sp.array([])
                    for j in range(len(self.ins)):
                        if self.ins[j] == i:
                            x = sp.append(x, self.time[j])
                            y = sp.append(y, rvk[j])
                            yerr = sp.append(yerr, errk[j])

                            yr = sp.append(yr, residuals[j])
                    if mode == 1:  # phasefolded
                        xp, yp, errp = phasefold(x, y, yerr, theta_k[1])  # phase fold
                        ax.errorbar(xp, yp, errp, color=colors[i], label='Data'+str(i), linestyle='', marker='o', alpha=0.75)  # phase fold
                        xpr, ypr, errpr = phasefold(x, yr, yerr, theta_k[1])
                        axr.errorbar(xpr, ypr, errpr, color=colors[i], fmt='o')
                        ax.set_xlim(min(xp), max(xp))
                        axr.set_xlim(min(xpr), max(xpr))

                    else:  # full
                        ax.errorbar(x, y, yerr, color=colors[i], label='Data'+str(i), linestyle='', marker='o', alpha=0.75)
                        axr.errorbar(x, yr, yerr, color=colors[i], fmt='o')
                        ax.set_xlim(min(x), max(x))
                        axr.set_xlim(min(x), max(x))

                # best_fit de el modelo completo en linea
                if mode == 1:  # phasefolded
                    time_m_p, rv_m_p, err_m_p = phasefold(time_m, rv_m, sp.zeros_like(time_m), theta_k[1])
                    ax.plot(time_m_p, rv_m_p, 'k', label='model')
                else:  # full
                    ax.plot(time_m, rv_m, '-k', label='model')
                # ax.minorticks_on()
                ax.set_ylabel('Radial Velocity (m/s)', fontsize=24)
                axr.axhline(0, color='k', linewidth=2)
                axr.get_yticklabels()[-1].set_visible(False)
                axr.minorticks_on()
                axr.set_ylabel('Residuals',fontsize=22)
                axr.set_xlabel('Time (Julian Days)',fontsize=22)
                if mode == 1:  # phasefolded
                    fig.savefig(saveplace+'/phasefold'+str(k)+'.pdf')
                else:  # full
                    fig.savefig(saveplace+'/fullmodel'+str(k)+'.pdf')
                if SHOW:
                    plt.show()

            theta_k = sp.roll(theta_k, -5)  # 5 del principio al final
        pass

    def instigator(self, chain, post, saveplace, kplanets):
        def mk_header(kplanets):
            h = []
            kepler = ['Amplitude               ', 'Period                  ', 'Phase                   ', 'Longitude               ', 'Eccentricity            ', 'Minimum Mass            ', 'SemiMajor Axis          ']
            telesc = ['Jitter                  ', 'Offset                  ']
            mov_ave = ['MA Coef                 ', 'Timescale               ']
            for i in range(kplanets):
                for item in kepler:
                    h.append(item)
            h.append('Acceleration            ')
            for j in range(self.nins):
                for item in telesc:
                    h.append(item)
                    for c in range(self.MOAV):
                        h.append(mov_ave[0])
                        h.append(mov_ave[1])
            for h in range(self.totcornum):
                h.append('Stellar Activity'+str(h))
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
        titles = sp.array(["Amplitude","Period","Longitude", "Phase","Eccentricity", 'Acceleration', 'Jitter', 'Offset', 'MACoefficient', 'MATimescale', 'Stellar Activity'])
        namen = sp.array([])
        ndim = kplanets * 5 + self.nins*2*(self.MOAV+1) + self.totcornum + 1

        RESU = sp.zeros((ndim, 5))
        for k in range(kplanets):
            namen = sp.append(namen, [titles[i] + '_'+str(k) for i in range(5)])
        namen = sp.append(namen, titles[5])  # for acc
        for i in range(self.nins):
            namen = sp.append(namen, [titles[ii] + '_'+str(i+1) for ii in sp.arange(2)+6])
            for c in range(self.MOAV):
                namen = sp.append(namen, [titles[ii] + '_'+str(i+1) + '_'+str(c+1) for ii in sp.arange(2)+8])
        for h in range(self.totcornum):
            namen = sp.append(namen, titles[-1]+'_'+str(h+1))

        alt_res = map(lambda v: (v[2], v[3]-v[2], v[2]-v[1], v[4]-v[2], v[2]-v[0]),
                      zip(*np.percentile(samples, [2, 16, 50, 84, 98], axis=0)))
        logdat = '\nAlternative results with uncertainties based on the 2nd, 16th, 50th, 84th and 98th percentiles of the samples in the marginalized distributions'
        logdat = '\nFormat is like median +- 1-sigma, +- 2-sigma'
        for res in range(ndim):
            logdat += '\n'+namen[res]+'     : '+str(alt_res[res][0])+' +- '+str(alt_res[res][1:3]) +'    2%   +- '+str(alt_res[res][3:5])
            RESU[res] = sp.percentile(samples, [2, 16, 50, 84, 98], axis=0)[:, res]
        print(logdat)
        return RESU

    def MCMC(self, kplanets, boundaries, inslims, acc_lims, sigmas_raw, pos0,
             logl, logp):

        ndim = 1 + 5 * kplanets + self.nins*2*(self.MOAV+1) + self.totcornum
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
            logdat += '\nN Number of Dimensions            : '+str(ndim)
            logdat += '\nN Moving Average                  : '+str(self.MOAV)
            logdat += '\nBeta Detail                       : '+str(self.betas)
            logdat += '\n-----------------------------------------------------'
            print(logdat)
            pass

        starinfo()
        sampler = PTSampler(self.ntemps, self.nwalkers, ndim, logl, logp, loglargs=[self.time, self.rv, self.err, self.ins, self.staract, self.starflag, kplanets, self.nins, self.MOAV, self.totcornum],
                            logpargs=[self.time, kplanets, self.nins, self.MOAV, self.totcornum, boundaries, inslims, acc_lims, sigmas_raw, self.eccprior, self.jittprior, self.jittmean, self.STARMASS, self.HILL, self.CHECK],
                            threads=self.cores, betas=self.betas)

        print('\n --------------------- BURN IN --------------------- \n')

        pbar = tqdm(total=self.burn_out)
        for p, lnprob, lnlike in sampler.sample(pos0, iterations=self.burn_out):
            pbar.update(1)
            pass
        pbar.close()

        p0, lnprob0, lnlike0 = p, lnprob, lnlike
        print("\nMean acceptance fraction: {0:.3f}".format(sp.mean(sampler.acceptance_fraction)))
        assert sp.mean(sampler.acceptance_fraction) != 0, 'Mean acceptance fraction = 0 ! ! !'
        sampler.reset()

        print('\n ---------------------- CHAIN ---------------------- \n')
        pbar = tqdm(total=self.nsteps)
        for p, lnprob, lnlike in sampler.sample(p0, lnprob0=lnprob0,
                                                   lnlike0=lnlike0,
                                                   iterations=self.nsteps,
                                                   thin=self.thin):
            pbar.update(1)
            pass
        pbar.close()


        assert sampler.chain.shape == (self.ntemps, self.nwalkers, self.nsteps/self.thin, ndim), 'something really weird happened'
        print("\nMean acceptance fraction: {0:.3f}".format(sp.mean(sampler.acceptance_fraction)))

        ln_post = sampler.lnprobability

        posteriors = sp.array([ln_post[i].reshape(-1) for i in range(self.ntemps)])
        chains = sampler.flatchain
        best_post = posteriors[0] == np.max(posteriors[0])

        thetas_raw = sp.array([chains[i] for i in range(self.ntemps)])
        thetas_hen = sp.array([self.henshin(chains[i], kplanets) for i in range(self.ntemps)])

        ajuste_hen = thetas_hen[0][best_post][0]
        ajuste_raw = thetas_raw[0][best_post][0]

        interesting_loc = sp.array([max(posteriors[temp]) - posteriors[temp] < self.bayes_factor for temp in sp.arange(self.ntemps)])
        interesting_thetas = sp.array([thetas_hen[temp][interesting_loc[temp]] for temp in sp.arange(self.ntemps)])
        interesting_thetas_raw = sp.array([thetas_raw[temp][interesting_loc[temp]] for temp in sp.arange(self.ntemps)])
        interesting_posts = sp.array([posteriors[temp][interesting_loc[temp]] for temp in range(self.ntemps)])
        sigmas = sp.array([ sp.std(interesting_thetas[0][:, i]) for i in range(ndim) ])
        sigmas_raw = sp.array([ sp.std(interesting_thetas_raw[0][:, i]) for i in range(ndim) ])
        #print('sigmas', sigmas)  # for testing
        #print('sigmas_raw', sigmas_raw)
        #print('mod_lims', boundaries)
        return thetas_raw, ajuste_raw, thetas_hen, ajuste_hen, p, lnprob, lnlike, posteriors, sampler.betas, interesting_thetas, interesting_posts, sigmas, sigmas_raw

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
        jitt_limiter = sp.amax(self.rv)
        if sp.amax(self.rv) ** 2 < sp.amin(self.rv) ** 2:
            jitt_limiter = sp.amin(self.rv)
        jitt_lim = 2 * abs(jitt_limiter)  # review this
        offs_lim = jitt_limiter  # review this
        ins_lims = sp.array([sp.append(sp.array([0.0001, jitt_lim, -offs_lim, offs_lim]), sp.array([sp.array([-1.0, 1.0, 0.1, 10]) for j in range(self.MOAV)])) for i in range(self.nins)]).reshape(-1)
        sqrta, sqrte = jitt_lim, 1.
        sqrta, sqrte = sqrta ** 0.5, sqrte ** 0.5
        ndim = kplan * 5 + self.nins*2*(self.MOAV+1) + self.totcornum + 1
        free_lims = sp.array([-sqrta, sqrta, sp.log(0.1), sp.log(3 * max(self.time)), -sqrta, sqrta, -sqrte, sqrte, -sqrte, sqrte])
        acc_lims = sp.array([-1., 1.])
        sigmas, sigmas_raw = sp.zeros(ndim), sp.zeros(ndim)
        pos0 = 0.
        thetas_hen, ajuste_hen = 0., 0.
        ajuste_raw = sp.array([0])
        oldlogpost = -999999999.
        interesting_thetas, interesting_posts = sp.array([]), sp.array([])
        thetas_raw = sp.array([])
        #####################################################
        START = chrono.time()
        while kplan <= to_k:
            mod_lims = sp.array([free_lims for i in range(kplan)]).reshape(-1)
            ins_lims = sp.array([sp.append(sp.array([0.0001, jitt_lim, -offs_lim, offs_lim]), sp.array([sp.array([-1.0, 1.0, 0.1, 10]) for j in range(self.MOAV)])) for i in range(self.nins)]).reshape(-1)
            #if LIL_JITT:
            #    ins_lims = ins_lims = sp.array([sp.append(sp.array([0.0001, 10.0, -offs_lim, offs_lim]), sp.array([sp.array([-1.0, 1.0, 0.1, 10]) for j in range(self.MOAV)])) for i in range(self.nins)]).reshape(-1)
            if kplan > 0:
                if self.CONSTRAIN and ajuste_raw[0]:
                    constrained = sp.array([])
                    for k in range(kplan - 1):
                        amp = 2.0
                        Ask, Pk, Ack, Sk, Ck = ajuste_raw[k*5:(k+1)*5]
                        Ask_std, Pk_std, Ack_std, Sk_std, Ck_std = sigmas_raw[5*k:5*(k+1)]
                        Ask_std, Ack_std, Sk_std, Ck_std = amp * sp.array([Ask_std, Ack_std, Sk_std, Ck_std])
                        aux = sp.array([Ask - Ask_std, Ask + Ask_std, Pk-Pk_std, Pk+Pk_std, Ack - Ack_std, Ack + Ack_std, Sk - Sk_std, Sk + Sk_std, Ck - Ck_std, Ck + Ck_std])

                        constrained = sp.append(constrained, aux)

                    for nin in range(self.nins):
                        ins_lims[0 + nin*4*(self.MOAV+1)] = 0.0001
                        ins_lims[1 + nin*4*(self.MOAV+1)] = ajuste_raw[(kplan - 1) * 5 + nin*2*(self.MOAV+1) + 1]
                    mod_lims = sp.append(constrained, free_lims)

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
            pos0 = self.pt_pos(kplan, mod_lims, ins_lims, acc_lims)
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

            alt_res = self.alt_results(interesting_thetas[0], kplan)
            saveplace = self.mklogfile(fit, bestlogpost, sample_sizes, sigmas, kplan, mod_lims, NEW_BIC, NEW_AIC, alt_res, START)
            self.instigator(interesting_thetas, interesting_posts, saveplace, kplan)

            if self.MUSIC:
                thybiding.play()
            if self.PLOT:
                self.plot2(fit, kplan, saveplace)
                self.plot1(interesting_thetas, interesting_posts, '0', kplan, saveplace)

            if OLD_BIC - NEW_BIC > self.BIC:
                print('\nBayes Information Criteria of %.2f requirement not met ! !' % self.BIC)
            if OLD_AIC - NEW_AIC > self.AIC:
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

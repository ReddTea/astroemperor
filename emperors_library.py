#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp

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


def normal_pdf(x, mean, variance):
    var = 2 * variance
    return ( - (x - mean) ** 2 / var)


def pt_pos(setup, kplanets, nins, boundaries, inslims, acc_lims, MOAV, totcornum,
           PACC):
    ntemps, nwalkers, nsteps = setup
    ndim = 1 + 5 * kplanets + nins*2*(MOAV+1) + totcornum + PACC
    pos = sp.array([sp.zeros(ndim) for i in range(nwalkers)])
    k = -2
    l = -2
    ll = -2  ##
    for j in range(ndim):
        if j < 5 * kplanets:
            k += 2
            if j%5==1:
                fact = sp.absolute(boundaries[k] - boundaries[k+1]) / nwalkers
            else:
                #fact = sp.absolute(boundaries[k]) / (self.nwalkers)
                fact = (sp.absolute(boundaries[k] - boundaries[k+1]) * 2) / (5 * nwalkers)
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
            for i in range(nwalkers):
                if j%5==1:
                    pos[i][j] = boundaries[k] + (dif[i] + fact/2.0)
                else:
                    #pos[i][j] = boundaries[k] * 0.5 + (dif[i] + fact/2.0)
                    pos[i][j] = (boundaries[k+1]+3*boundaries[k])/4 + (dif[i] + fact/2.0)
        if j == 5 * kplanets:  # acc
            fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
            for i in range(nwalkers):
                pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)
        if PACC:
            if j == 5 * kplanets + PACC:  # parabolic accel
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
                dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
                for i in range(nwalkers):
                    pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)

        # instruments
        if 5 * kplanets + PACC < j < 5*kplanets + nins*2*(MOAV+1) + 1 + PACC:
            l += 2
            fact = sp.absolute(inslims[l] - inslims[l+1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)

            if (j-5*kplanets-1-PACC) % nins*2*(MOAV+1) == 0:  # ojo aqui
                jitt_ini = sp.sort(sp.fabs(sp.random.normal(0, 1, nwalkers))) * 0.1
                dif = jitt_ini * sp.random.uniform(0.9, 0.999)

            for i in range(nwalkers):
                pos[i][j] = inslims[l] + (dif[i] + fact/2.0)
            #print(pos[j][:])
        if totcornum:
            if j > 5*kplanets + nins*2*(MOAV+1) + PACC:
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers

                dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.8, 0.999)
                for i in range(nwalkers):
                    pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)
                    #print(pos[i][j])

    pos = sp.array([pos for h in range(ntemps)])
    return pos


def pt_pos_PM(setup, kplanets, nins, boundaries, inslims, acc_lims, MOAV, totcornum,
              PACC):
    ntemps, nwalkers, nsteps = setup
    ndim = 1 + 4 * kplanets + nins*2*(MOAV+1) + totcornum + PACC
    pos = sp.array([sp.zeros(ndim) for i in range(nwalkers)])
    k = -2
    l = -2
    ll = -2  ##
    for j in range(ndim):
        if j < 4 * kplanets:
            k += 2
            if j%4==0 or j%4==3:  # period or ecc
                fact = sp.absolute(boundaries[k] - boundaries[k+1]) / nwalkers
            else:
                fact = (sp.absolute(boundaries[k] - boundaries[k+1]) * 2) / (5 * nwalkers)
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
            for i in range(nwalkers):
                if j%5==0 or j%4==3:  # period or ecc
                    pos[i][j] = boundaries[k] + (dif[i] + fact/2.0)
                else:
                    pos[i][j] = (boundaries[k+1]+3*boundaries[k])/4 + (dif[i] + fact/2.0)
        if j == 4 * kplanets:  # acc
            fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
            for i in range(nwalkers):
                pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)
        if PACC:
            if j == 4 * kplanets + PACC:  # parabolic accel
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers
                dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)
                for i in range(nwalkers):
                    pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)

            # instruments
        if 4 * kplanets + PACC < j < 4*kplanets + nins*2*(MOAV+1) + 1 + PACC:
            l += 2
            fact = sp.absolute(inslims[l] - inslims[l+1]) / nwalkers
            dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.9, 0.999)

            if (j-4*kplanets-1) % nins*2*(MOAV+1) == 0:  # ojo aqui
                jitt_ini = sp.sort(sp.fabs(sp.random.normal(0, 1, nwalkers))) * 0.1
                dif = jitt_ini * sp.random.uniform(0.9, 0.999)
            for i in range(nwalkers):
                pos[i][j] = inslims[l] + (dif[i] + fact/2.0)
        if totcornum:
            if j > 4*kplanets + nins*2*(MOAV+1) + PACC:
                fact = sp.absolute(acc_lims[0] - acc_lims[1]) / nwalkers

                dif = sp.arange(nwalkers) * fact * sp.random.uniform(0.8, 0.999)
                for i in range(nwalkers):
                    pos[i][j] = acc_lims[0] + (dif[i] + fact/2.0)
                    #print(pos[i][j])

    pos = sp.array([pos for h in range(ntemps)])
    return pos

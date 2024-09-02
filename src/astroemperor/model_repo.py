# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.8
# date 11 apr 2023

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT
# **FIN  : Finish this

import numpy as np
import kepler

def norm(x, mu, sig):
    izq = 1 / (sig*np.sqrt(2*np.pi))
    der = np.exp(-0.5*((x-mu)/sig)**2)

    return izq*der


def Line_Model(theta, *args, **kwargs):
    data = args[0]
    x, y, yerr = data

    m, b, logf = theta
    return m * x + b


def DG_Model(theta, *args, **kwargs):
    data = args[0]
    x, y, yerr = data
    mu1, mu2, sigma1, sigma2, logf = theta

    return norm(x, mu1, sigma1) + norm(x, mu2, sigma2), 0


def Keplerian_Model(theta, *args, **kwargs):
    # **EVAL, data vs just time
    per, A, phase, ecc, w = theta

    freq = 2. * np.pi / per  # freq space
    M = freq * args[0] + phase  # mean anomaly
    E = np.array([kepler.solve(m, ecc) for m in M])  # eccentric anomaly
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly
    model = A * (np.cos(f + w) + ecc * np.cos(w))

    return model, 0


def Keplerian_Model_1(theta, *args, **kwargs):
    P, As, Ac, S, C = theta

    per = np.exp(P)
    A = As ** 2 + Ac ** 2
    ecc = S ** 2 + C ** 2

    if ecc < 1e-5:
        w = 0
    else:
        w = np.arccos(C / (ecc ** 0.5))  # longitude of periastron
        if S < 0:
            w = 2 * np.pi - np.arccos(C / (ecc ** 0.5))

    if ecc > 1 or ecc < 0:
        print('\n HER HERE\n')
        print(theta)

    phase = np.arccos(Ac / (A ** 0.5))
    if As < 0:
        phase = 2 * np.pi - np.arccos(Ac / (A ** 0.5))


    freq = 2. * np.pi / per
    M = freq * args[0] + phase
    E = np.array([kepler.solve(m, ecc) for m in M])
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly

    model = A * (np.cos(f + w) + ecc * np.cos(w))

    return model, 0


def Keplerian_Model_2(theta, *args, **kwargs):
    #model = 0
    per, A, tp, ecc, w = theta
    freq = 2. * np.pi / per

    M = freq * (args[0] - tp)
    E = np.array([kepler.solve(m, ecc) for m in M])
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly
    model = A * (np.cos(f + w) + ecc * np.cos(w))

    return  model, 0


def Keplerian_Model_3(theta, *args, **kwargs):
    per, A, tp, S, C = theta

    ecc = S ** 2 + C ** 2

    if ecc < 1e-5:
        w = 0
    else:
        w = np.arccos(C / (ecc ** 0.5))  # longitude of periastron
        if S < 0:
            w = 2 * np.pi - np.arccos(C / (ecc ** 0.5))

    freq = 2. * np.pi / per
    M = freq * (args[0] - tp)
    E = np.array([kepler.solve(m, ecc) for m in M])
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly
    model = A * (np.cos(f + w) + ecc * np.cos(w))

    return  model, 0


def Instrument_Model(theta, *args, **kwargs):
    ins_no = args[0]
    flags = args[1]

    my_mask = (flags == ins_no)

    mod = theta[0] * my_mask  # OFFSET

    new_err = my_mask * theta[1] ** 2

    return mod, new_err


def Instrument_Moav_Model(theta, *args, **kwargs):
    #offset, jitter = theta

    ins_no = args[0]
    flags = args[1]
    maorder = args[2]

    time = args[3]
    residuals = args[4]

    my_mask = (flags == ins_no)

    mod = theta[0] * my_mask  # OFFSET

    if maorder > 0:
        theta_ma = theta[2:]
        t_ = time[my_mask]
        for i in range(len(t_)):
            for c in range(maorder):
                if i > c:
                    dt = abs(t_[i] - t_[i - 1 - c])
                    macoef = theta_ma[2 * c]
                    matime = theta_ma[2 * c + 1]
                    MA = macoef * np.exp(-dt / matime) * residuals[my_mask][i - 1 - c]
                    mod[my_mask][i] += MA
                    residuals[my_mask][i] -= MA


    new_err = my_mask * theta[1] ** 2

    return mod, new_err


def Instrument_Moav_SA_Model(theta, *args, **kwargs):
    # time and residuals should exclusively be the ones for this instrument
    ins_no = args[0]
    flags = args[1]
    maorder = args[2]
    time = args[3]

    staracts = args[4]
    cornum = args[5]

    #print('in model 2', offset*[flags == ins_no])

    my_mask = (flags == ins_no)
    mod = theta[0] * my_mask  # OFFSET
    #residuals = rv - mod

    if cornum:
        for j in range(cornum):
            mod[my_mask] += theta[2 * (maorder + 1) + j] * staracts[j]

    if maorder > 0:
        residuals = modargs[0][-1]
        theta_ma = theta[2:]
        t_ = time[my_mask]
        for i in range(len(t_)):
            for c in range(maorder):
                if i > c:
                    dt = abs(t_[i] - t_[i - 1 - c])
                    macoef = theta_ma[2 * c]
                    matime = theta_ma[2 * c + 1]
                    MA = macoef * np.exp(-dt / matime) * residuals[my_mask][i - 1 - c]
                    mod[my_mask][i] += MA
                    residuals[my_mask][i] -= MA

    new_err = my_mask * theta[1] ** 2

    return mod, new_err


def Instrument_Model_SA(theta, *args, **kwargs):
    # time and residuals should exclusively be the ones for this instrument
    ins_no = args[0]
    flags = args[1]
    maorder = args[2]
    time = args[3]

    staracts = args[4]
    cornum = args[5]

    #print('in model 2', offset*[flags == ins_no])

    my_mask = (flags == ins_no)
    mod = theta[0] * my_mask  # OFFSET
    #residuals = rv - mod

    if cornum:
        for j in range(cornum):
            mod[my_mask] += theta[2 * (maorder + 1) + j] * staracts[j]

    new_err = my_mask * theta[1] ** 2

    return mod, new_err


def Acceleration_Model(theta, *args, **kwargs):
    time = args[0]
    mod = np.polyval(np.concatenate([theta, [0]]), time-time[0])

    return mod, 0


def Empty_Model(theta, *args, **kwargs):
    return 0, 0


'''
# Notes on Keplerian Models

## Model 0 or vanilla
The easiest of them all, it uses
Period, Amplitude, Phase, Eccentricity, Longitude_Periastron

## Model 1 or hou
According to Hou's paper <<LINK>>
Mind that I use log Period here
lPeriod, Amp_sin, Amp_cos, Ecc_sin, Ecc_cos

## Model 2 or Tp
Pretty similar to vanilla
Period, Amplitude, Time_Periastron, Eccentricity, Longitude_Periastron

## Model 3 or Tp Hou
Combination of Time of periastron, circularising the eccentricity.

Period, Amplitude, Phase, Ecc_sin, Ecc_cos

'''










#

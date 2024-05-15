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

def unKeplerian_Model(Period, Amplitude, Phase, Eccentricity, Longitude_Periastron):
    # **EVAL, data vs just time
    Time = np.linspace(0, Period, 5000)

    freq = 2. * np.pi / Period  # freq space
    M = freq * Time + Phase  # mean anomaly
    E = np.array([kepler.solve(m, Eccentricity) for m in M])  # eccentric anomaly
    f = (np.arctan(((1. + Eccentricity) ** 0.5 / (1. - Eccentricity) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly
    model = Amplitude * (np.cos(f + Longitude_Periastron) + Eccentricity * np.cos(Longitude_Periastron))

    return Time, model


def unKeplerian_Model_1(lPeriod, Amp_sin, Amp_cos, Ecc_sin, Ecc_cos):
    per = np.exp(lPeriod)

    Time = np.linspace(0, per, 5000)
    A = Amp_sin ** 2 + Amp_cos ** 2
    ecc = Ecc_sin ** 2 + Ecc_cos ** 2

    if ecc < 1e-5:
        w = 0
    else:
        w = np.arccos(Ecc_cos / (ecc ** 0.5))  # longitude of periastron
        if Ecc_sin < 0:
            w = 2 * np.pi - np.arccos(Ecc_cos / (ecc ** 0.5))

    phase = np.arccos(Amp_cos / (A ** 0.5))
    if Amp_sin < 0:
        phase = 2 * np.pi - np.arccos(Amp_cos / (A ** 0.5))

    freq = 2. * np.pi / per
    M = freq * Time + phase
    E = np.array([kepler.solve(m, ecc) for m in M])
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly

    model = A * (np.cos(f + w) + ecc * np.cos(w))

    return Time, model


def unKeplerian_Model_2(Period, Amplitude, Time_Periastron, Eccentricity, Longitude_Periastron):
    Time = np.linspace(0, Period, 5000)

    freq = 2. * np.pi / Period
    M = freq * (Time - Time_Periastron)
    E = np.array([kepler.solve(m, Eccentricity) for m in M])
    f = (np.arctan(((1. + Eccentricity) ** 0.5 / (1. - Eccentricity) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly
    model = Amplitude * (np.cos(f + Longitude_Periastron) + Eccentricity * np.cos(Longitude_Periastron))

    return Time, model


def unKeplerian_Model_3(Period, Amplitude, Time_Periastron, Ecc_sin, Ecc_cos):
    Time = np.linspace(0, Period, 5000)
    ecc = Ecc_sin ** 2 + Ecc_cos ** 2
    if ecc < 1e-5:
        w = 0
    else:
        w = np.arccos(Ecc_cos / (ecc ** 0.5))  # longitude of periastron
        if Ecc_sin < 0:
            w = 2 * np.pi - np.arccos(Ecc_cos / (ecc ** 0.5))

    freq = 2. * np.pi / Period
    M = freq * (Time - Time_Periastron)
    E = np.array([kepler.solve(m, ecc) for m in M])
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)  # true anomaly
    model = Amplitude * (np.cos(f + w) + ecc * np.cos(w))

    return Time, model


#

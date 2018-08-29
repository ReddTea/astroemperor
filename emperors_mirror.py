#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
from PyAstronomy.pyasl import MarkleyKESolver

def RV_model(THETA, time, kplanets):
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
        M = freq * time + phase  # mean anomaly
        E = sp.array([MarkleyKESolver().getE(m, ecc) for m in M])  # eccentric anomaly
        f = (sp.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * sp.tan(E / 2.)) * 2.)  # true anomaly
        modelo += A * (sp.cos(f + w) + ecc * sp.cos(w))
    return  modelo


def mini_RV_model(params, time):
    A, P, phase, w, ecc = params
    freq = 2. * sp.pi / P
    M = freq * time + phase
    E = sp.array([MarkleyKESolver().getE(m, ecc) for m in M])
    f = (sp.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * sp.tan(E / 2.)) * 2.)
    modelo = A * (sp.cos(f + w) + ecc * sp.cos(w))
    return modelo


def PM_model(theta, time, kplanets):
    modelo = 0.0
    if kplanets == 0:
        return 0.0
    for i in range(kplanets):
        P, As, Ac, ecc = theta[4*i:4*(i+1)]
        A = As ** 2 + Ac ** 2
        phase = sp.arccos(Ac / (A ** 0.5))
        if As < 0:
            phase = 2 * sp.pi - sp.arccos(Ac / (A ** 0.5))
        per = sp.exp(P)
        freq = 2. * sp.pi / per
        M = freq * time + phase
        E = sp.array([MarkleyKESolver().getE(m, ecc) for m in M])
        f = (sp.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * sp.tan(E / 2.)) * 2.)
        #print '\r '+str(i)+str(per)  # testing purposes
        modelo += A * (sp.cos(f) + ecc)
    return modelo


def mini_PM_model(params, time):
    P, A, phase, ecc = params
    freq = 2. * sp.pi / P
    M = freq * time + phase
    E = sp.array([MarkleyKESolver().getE(m, ecc) for m in M])
    f = (sp.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * sp.tan(E / 2.)) * 2.)
    modelo = A * (sp.cos(f) + ecc)
    return modelo


def henshin(thetas, kplanets):
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


def henshin_PM(thetas, kplanets):
    for i in range(kplanets):
        Pk = thetas[:, i*4]
        Ask = thetas[:, i*4 + 1]
        Ack = thetas[:, i*4 + 2]
        ecck = thetas[:, i*4 + 3]

        Ak = Ask ** 2 + Ack ** 2
        Phasek = sp.arccos(Ack / (Ak ** 0.5))
        for j in range(len(Ask)):
            if Ask[j] < 0:
                Phasek[j] = 2 * sp.pi - sp.arccos(Ack[j] / (Ak[j] ** 0.5))

        thetas[:, i*4] = sp.exp(Pk)
        thetas[:, i*4 + 1] = Ak
        thetas[:, i*4 + 2] = Phasek
        thetas[:, i*4 + 3] = ecck
    return thetas

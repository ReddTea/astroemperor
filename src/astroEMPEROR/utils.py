# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.1.0
# date 19 jun 2021

import numpy as np
import os
import sys

def fold(x, y, yerr=None, per=None):
    if per == None:
        per = 2. * np.pi
    x_f = x % per
    order = np.argsort(x_f)
    if yerr is None:
        return x_f[order], y[order]
    else:
        return x_f[order], y[order], yerr[order]
    pass


def minmax(x):
    return np.amin(x), np.amax(x)


def flatten(t):
    return [item for sublist in t for item in sublist]


def cps(pers, amps, eccs, starmass):
    #sma, minmass = np.zeros(kplanets), np.zeros(kplanets)
    G = 6.674e-11  # m3 / (kg * s2)
    m2au = 6.685e-12  # au
    kg2sm = 5.03e-31  # solar masses
    s2d = 1.157e-5  # days
    G_ = G * m2au**3 / (kg2sm*s2d)  # au3 / (sm * d2)

    consts = 4*np.pi**2/(G*1.99e30)

    sma = ((pers*24*3600)**2 * starmass / consts)**(1./3) / 1.49598e11
    minmass = amps / ( (28.4329/np.sqrt(1. - eccs**2.)) * (starmass**(-0.5)) * (sma**(-0.5)) )

    return sma, minmass


def hill_check(p, a, e, sm=0.33):
    #kp = len(p)

    sma, minmass = cps(p, a, e, sm)
    o = np.argsort(sma)

    sma, minmass = sma[o], minmass[o]
    p, a, e = p[o], a[o], e[o]

    gamma = np.sqrt(1 - e**2)
    LHS, RHS = [], []
    for k in np.arange(kp):
        mm = np.array([minmass[k], minmass[k+1]])
        M = sm * 1047.56 + np.sum(mm)
        mu = mm / M
        alpha = np.sum(mu)
        delta = np.sqrt(sma[k+1] / sma[k])
        LHS.append(alpha**-3 * (mu[k] + (mu[k+1] / (delta**2))) * (mu[k] * gamma[k] + mu[k+1] * gamma[k+1] * delta)**2)
        RHS.append(1 + (3./alpha)**(4./3) * mu[k] * mu[k+1])

    return LHS, RHS


def amd(p, a, e, sm=0.33):
    kp = len(p)
    sma, minmass = cps(p, a, e, kp, sm)
    o = np.argsort(sma)

    sma, minmass = sma[o], minmass[o]
    p, a, e = p[o], a[o], e[o]

    mu = G * sm
    eps = np.sum(minmass) / sm
    gamma = minmass[0] / minmass[1]
    lambd = minmass * np.sqrt(mu*sma)
    alpha = sma[0] / sma[1]

    C = gamma * np.sqrt(alpha) * (1 - np.sqrt(1-e[0]**2)*1) + 1 - np.sqrt(1-e[1]**2)*1
    C = gamma * np.sqrt(alpha) + 1
    RHS = gamma * np.sqrt(alpha) + 1 - (1+gamma)**(1.5) * np.sqrt( (alpha/(gamma+alpha)) * (1+ (3**(4./3) * eps**(2./3) * gamma)/(1+gamma)**2))

    LHS = gamma * np.sqrt(alpha) * (1 - np.sqrt(1-e[0]**2)*np.cos(i[0])) + 1 - np.sqrt(1 - e[1]**2)*np.cos(i[1])

    pass


def delinearize(x, y):
    A = x**2 + y**2
    B = np.arccos(y / (A ** 0.5))
    if x < 0:
        B = 2 * np.pi - B
    return np.array([A, B])


def adelinearize(s, c):
    # x sine, y cosine
    A = s**2 + c**2

    B = np.arccos(c / (A ** 0.5))
    B[s<0] = 2 * np.pi - B[s<0]

    #where is slower
    #B = np.where(x>0, np.arccos(y / (A ** 0.5)), 2 * np.pi - np.arccos(y[x<0] / (A[x<0] ** 0.5)))
    return np.array([A, B])


def delinearize_pymc3(x, y):
    import aesara_theano_fallback.tensor as tt
    import exoplanet as xo

    A = x**2 + y**2
    B = tt.arccos(y / (A ** 0.5))
    if x < 0:
        B = 2 * np.pi - B
    return tt.array([A, B])

    #
'''
from utils import hill_check as hc
from utils import cps
import numpy as np
p = np.array([61.082, 30.126])
a = np.array([212.07, 88.34])
e = np.array([0.027, 0.25])

sma, mm = cps(p, a, e, 2, 0.33)
o = np.argsort(sma)
sma, mm, p, a, e = sma[o], mm[o], p[o], a[o], e[o]

sma_ = np.array([0.134, 0.214])
mm_ = np.array([0.65127405, 2.3941])
'''


def getExtremePoints(data, typeOfExtreme = None, maxPoints = None):
    """
    from https://towardsdatascience.com/modality-tests-and-kernel-density-estimations-3f349bb9e595
    """
    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange == 1)[0]

    if typeOfExtreme == 'max' and data[idx[0]] < data[idx[1]]:
        idx = idx[1:][::2]

    elif typeOfExtreme == 'min' and data[idx[0]] > data[idx[1]]:
        idx = idx[1:][::2]

    elif typeOfExtreme is not None:
        idx = idx[::2]

    # sort ids by min value
    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data) - 1) in idx:
        idx = np.delete(idx, len(data)-1)

    idx = idx[np.argsort(data[idx])]
    # If we have maxpoints we want to make sure the timeseries has a cutpoint
    # in each segment, not all on a small interval
    if maxPoints is not None:
        idx = idx[:maxPoints]
        if len(idx) < maxPoints:
            return (np.arange(maxPoints) + 1) * (len(data)//(maxPoints + 1))

    return idx


def plot_extreme(data, n=10):
    import matplotlib.pyplot as pl
    x, y = data

    idx = getExtremePoints(y, typeOfExtreme='max')
    pl.plot(x,y)
    ax = pl.gca()
    pl.scatter(x[idx], y[idx], s=40, c='red')

    for i in idx[:n]:
        ax.annotate(f' Max = {np.round(x[i], 2)}', (x[i], y[i]))
    pl.show()


def ensure_dir(name, loc=''):
    dr = loc+'datalogs/%s/run_1' % name
    while os.path.exists(dr):
        aux = int(dr.split('_')[-1]) + 1
        dr = dr.split('_')[0] + '_' + str(aux)

    os.makedirs(dr)
    os.makedirs(dr+'/histograms')
    os.makedirs(dr+'/posteriors')
    os.makedirs(dr+'/traces')
    return dr


class importer:
	def	__init__(self):
		self.c = 1
	def add_path(self, x):
		sys.path.insert(self.c, x)
		self.c += 1
	def restore_priority(self):
		self.c = 1


class binner:
    def __init__(self, x, y=None):
        pass
    pass

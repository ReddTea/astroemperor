# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/ /^\s*class/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.3
# date 14 nov 2022

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT
# **FIN  : Finish this

import numpy as np
import pandas as pd

import sys, os
from contextlib import contextmanager

from termcolor import colored
from sklearn.mixture import GaussianMixture


_ROOT = os.path.dirname(__file__)

def get_support(path):
    return os.path.join(_ROOT, 'support', path)


def fold(x, y, yerr=None, per=None):
    if per is None:
        per = 2. * np.pi
    x_f = x % per
    order = np.argsort(x_f)
    if yerr is None:
        return x_f[order], y[order]
    else:
        return x_f[order], y[order], yerr[order]


def fold_dataframe(df, per=None):
    if per is None:
        per = 2 * np.pi
    df['BJD'] = df['BJD'] % per
    return df.sort_values('BJD')


def minmax(x):
    return np.amin(x), np.amax(x)


def flatten(t):
    return [item for sublist in t for item in sublist]


def cps(pers, amps, eccs, starmass):
    #sma, minmass = np.zeros(kplanets), np.zeros(kplanets)
    G = 6.674e-11  # m3 / (kg * s2)
    #m2au = 6.685e-12  # au
    #kg2sm = 5.03e-31  # solar masses
    #s2d = 1.157e-5  # days
    #G_ = G * m2au**3 / (kg2sm*s2d)  # au3 / (sm * d2)

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
    for k in np.arange(len(p)):
        mm = np.array([minmass[k], minmass[k+1]])
        M = sm * 1047.56 + np.sum(mm)
        mu = mm / M
        alpha = np.sum(mu)
        delta = np.sqrt(sma[k+1] / sma[k])
        LHS.append(alpha**-3 * (mu[k] + (mu[k+1] / (delta**2))) * (mu[k] * gamma[k] + mu[k+1] * gamma[k+1] * delta)**2)
        RHS.append(1 + (3./alpha)**(4./3) * mu[k] * mu[k+1])

    return LHS, RHS

'''
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
'''

def delinearize(x, y):
    A = x**2 + y**2
    B = np.arccos(y / (A ** 0.5)) if A != 0 else 0
    if x < 0:
        B = 2 * np.pi - B
    return np.array([A, B])


def adelinearize(s, c):
    # x sine, y cosine
    A = s**2 + c**2
    B = np.zeros_like(A) if A.all() == 0 else np.arccos(c / (A ** 0.5))
    B[s<0] = 2 * np.pi - B[s<0]

    #where is slower
    #B = np.where(x>0, np.arccos(y / (A ** 0.5)), 2 * np.pi - np.arccos(y[x<0] / (A[x<0] ** 0.5)))
    return np.array([A, B])

'''
def delinearize_pymc3(x, y):
    import aesara_theano_fallback.tensor as tt
    import exoplanet as xo

    A = x**2 + y**2
    B = tt.arccos(y / (A ** 0.5))
    if x < 0:
        B = 2 * np.pi - B
    return tt.array([A, B])
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
    dr = f'{loc}datalogs/{name}/run_1'
    while os.path.exists(dr):
        aux = int(dr.split('_')[-1]) + 1
        dr = dr.split('_')[0] + '_' + str(aux)

    os.makedirs(dr)
    os.makedirs(f'{dr}/histograms')
    os.makedirs(f'{dr}/posteriors')
    os.makedirs(f'{dr}/posteriors/GMEstimates')
    os.makedirs(f'{dr}/likelihoods')
    os.makedirs(f'{dr}/chains')
    os.makedirs(f'{dr}/traces')
    os.makedirs(f'{dr}/models')
    os.makedirs(f'{dr}/models/uncertainpy')
    os.makedirs(f'{dr}/models/temp')
    os.makedirs(f'{dr}/temp')

    return dr


def set_pos0(setup, model_obj):
    ntemps, nwalkers, nsteps = setup
    ndim = model_obj.ndim__
    pos = np.zeros((ntemps, nwalkers, ndim))

    for t in range(ntemps):
        j = 0
        for b in model_obj:
            for p in b:
                if p.fixed is None:
                    m = (p.limits[1] + p.limits[0]) / 2
                    r = (p.limits[1] - p.limits[0]) / 2
                    dist = np.sort(np.random.uniform(0, 1, nwalkers))

                    if (
                        b.parameterisation != 0 and b.type_ == 'Keplerian'
                    ) and p != b[0]:
                        r *= 0.707

                    pos[t][:, j] = r * (2 * dist - 1) + m
                    np.random.shuffle(pos[t, :, j])
                    j += 1
    return list(pos)


def pos0_tested(setup, model_obj, max_repeats=100):
    p0 = set_pos0(setup, model_obj)

    ntemps, nwalkers, nsteps = setup

    is_bad_position = True
    repeat_number = 0
    while is_bad_position and repeat_number < max_repeats:
        is_bad_position = False
        for t in range(ntemps):
            for n in range(nwalkers):
                position_evaluated = p0[t][n]
                if model_obj.evaluate_logprior(position_evaluated) == -np.inf:
                    is_bad_position = True
                    p0[t][n] = set_pos0(setup, model_obj)[t][n]
        repeat_number += 1
    return p0


@contextmanager
def nullify_output(suppress_stdout=True, suppress_stderr=True, suppress_stdin=True):
    stdout = sys.stdout
    stderr = sys.stderr
    stdin = sys.stdin
    devnull = open(os.devnull, "w")
    try:
        if suppress_stdout:
            sys.stdout = devnull
        if suppress_stderr:
            sys.stderr = devnull
        if suppress_stdin:
            sys.stdin = devnull
        yield
    finally:
        if suppress_stdout:
            sys.stdout = stdout
        if suppress_stderr:
            sys.stderr = stderr
        if suppress_stdin:
            sys.stdin = stdin


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


class reddlog(object):
    def __init__(self):
        self.log = ''
        try:
            self.terminal_width = os.get_terminal_size().columns
        except:
            self.terminal_width = pd.get_option('display.width')

    def saveto(self, location):
        np.savetxt(f'{location}/log.dat', np.array([self.log]), fmt='%100s')

    def help(self):
        print('Colors: grey, red, green, yellow, blue, magenta, cyan, white')
        print('On_Colors: on_<color>')
        print('Attrs: bold, dark, underline, blink, reverse, concealed')


    def __call__(self, msg, center=False, save=True, c=None, oc=None, attrs=None):
        if attrs is None:
            attrs = []
        if save:
            self.log += msg
        if center:
            msg = msg.center(self.terminal_width)
        if c:
            msg = colored(msg, c, oc, attrs)
        print(msg)

'''
class KDE_estimator:
    def __init__(self):
        self.kde_estimator = []
        self.bandwidths = np.arange(0.1, 1.5, .1)
        self.train_length = 5000
        self.chain_length = 5000
        self.x_test = []

    def estimate(self, chain_p, p=0, thin_by=1):
        t_start = time.time()
        x_train = np.array([[c] for c in chain_p[::thin_by]])
        self.x_test.append(np.linspace(np.amin(x_train), np.amax(x_train), self.train_length)[:, np.newaxis])

        kde = KernelDensity(kernel='gaussian')
        grid = GridSearchCV(kde, {'bandwidth': self.bandwidths})

        grid.fit(x_train)
        self.kde_estimator.append(grid.best_estimator_)
        t_end = time.time()
        print(f'Time to estimate parameter {p} is {t_end - t_start} seconds')

    def auto_estimate(self, chain):
        nsteps, ndim = chain.shape
        thin_by_auto = nsteps // self.train_length
        for i in range(ndim):
            self.estimate(chain[:, i], p=i, thin_by=thin_by_auto)
'''

class GaussianMixture_addon(GaussianMixture):
    def __init__(self, name, unit, **kw):
        # this one is per parameter
        self.name = name
        self.unit = unit
        if self.unit is None:
            self.unit = ''
        self.means = None
        self.covariances = None
        self.sigmas = None
        self.weights = None
        self.mixture_mean = None
        self.mixture_variance = None
        self.mixture_sigma = None
        super().__init__(**kw)

    def update(self):
        self.means = self.means_.flatten()
        self.covariances = self.covariances_.flatten()
        self.sigmas = np.sqrt(self.covariances)
        self.weights = self.weights_
        self.mixture_mean = self.Mixture_Mean()
        self.mixture_variance = self.Mixture_Variance()
        self.mixture_sigma = np.sqrt(self.mixture_variance)

    def Mixture_Variance(self):
        s1 = np.sum(self.weights*self.covariances)
        s2 = np.sum(self.weights*self.means**2)
        s3 = np.sum(self.weights*self.means)**2
        return s1 + s2 - s3

    def Mixture_Mean(self):
        return np.sum(self.means*self.weights)


    def __repr__(self):
        return f'{self.means}; {self.sigmas}'


class GM_Estimator:
    def __init__(self):
        # This one is per param
        self.gm_estimator = None
        self.max_n_components = 5
        self.BIC0_ = np.inf
        self.BIC_Tolerance = 0
        self.name__ = 'GM_Estimator'


    def estimate(self, chain_p, name, unit):
        comp_bic = self.BIC0_
        gm0 = None
        chain_p = chain_p[:, np.newaxis]

        for n in range(self.max_n_components):
            gm = GaussianMixture_addon(name, unit, **{'n_components':n+1}).fit(chain_p)
            gm.update()
            mu = gm.means_
            sel_bic = gm.bic(mu)
            if sel_bic - comp_bic < self.BIC_Tolerance:
                comp_bic = sel_bic
                gm0 = gm

        self.gm_estimator = gm0
        return self.gm_estimator


class DataWrapper(object):
    def __init__(self, target_name, read_loc=''):
        self.target_name = target_name
        self.RV_PATH = f'{read_loc}datafiles/{self.target_name}/RV/'

        empty_lists = ['ndata', 'ncols', 'nsai', 'data', 'RV_labels',
                       'RV_sets', 'PM_sets']
        for attribute in empty_lists:
            setattr(self, attribute, [])


    def add_data__(self, filename):
        data = np.loadtxt('{0}{1}'.format(self.RV_PATH, filename))

        ndat, ncol = data.shape

        self.ndata.append(ndat)
        self.ncols.append(ncol)
        self.RV_labels.append(filename)

        names = ['BJD', 'RV', 'eRV']

        # identify and name SAI
        nsa = ncol - 3
        if nsa > 0:
            names.extend(f'Staract {j}' for j in range(nsa))
        self.nsai.append(nsa)

        df = pd.DataFrame(data, columns=names)

        # substract RV
        if abs(df.mean()['RV']) > 1e-6:
            df['RV'] -= df.mean()['RV']
        # create another column containing flags for the instrument
        df['Flag'] = np.ones(ndat, int) * len(self.ndata)
        self.data.append(df)

        return 'Reading data from {0}'.format(filename)


    def add_all__(self):
        my_files = list(np.sort(os.listdir(self.RV_PATH)))
        # mac os fix
        try:
            my_files.remove('.DS_Store')
        except:
            pass
        x = ''.join('\n'+self.add_data__(file) for file in my_files)
        return x+'\n'


    def get_data__(self, sortby='BJD'):
        return pd.concat(self.data).sort_values(sortby)


    def get_metadata__(self):
        return [getattr(self, attribute) for attribute in ['ndata', 'ncols', 'nsai', 'RV_labels']]


    def get_data_raw(self, sortby):
        holder = pd.concat(self.data).sort_values(sortby)
        x, y, yerr = holder.BJD, holder.RV, holder.eRV
        return [x, y, yerr]


class ModelWrapper(object):
    def __init__(self, func_model, fargs=None, fkwargs=None):
        if fargs is None:
            fargs = []
        if fkwargs is None:
            fkwargs = {}
        self.func = func_model
        self.fargs = fargs
        self.fkwargs = fkwargs

    def __call__(self, x):
        return self.func(x, *self.fargs, **self.fkwargs)




#

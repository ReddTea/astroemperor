# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT
# **FIN  : Finish this

import os
import shutil
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from termcolor import colored

from .globals import _OS_ROOT


def get_support(path):
    return os.path.join(_OS_ROOT, 'support', path)


class nuker(object):
    def __init__(self, loc=''):
        self._loc = loc
        self._target_folder = ''
        self._dr = ''
        self.IamSure = False
        self._nuclear = False

    @property
    def dr(self):
        return self._dr

    @dr.setter
    def dr(self, val):
        self._dr = val

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, val: str):
        self._loc = val


    def aim(self, target):
        if type(target) == str:
            self._target_folder = target
            self._dr = f'{self._loc}datalogs/{self._target_folder}/'
            self._target_list = os.listdir(self._dr)

        if type(target) == tuple:
            holder = []
            tmin, tmax = target[0], target[1] + 1
            for i in range(tmin, tmax):
                run_name = f'run_{i}'
                if run_name in self._target_list:
                    holder.append(run_name)
            self._target_list = holder

        if type(target) == bool and target:
            self._dr = f'{self._loc}datalogs/'
            self._nuclear = True
            self._target_list = os.listdir(self._dr)

        for x in self._target_list:
            if x[0] == '.':
                self._target_list.remove(x)


    def nuke(self):
        if not self.IamSure:
            print('You are about to delete the following directories:\n')
            for tg in self._target_list:
                print(f'    - {self._dr}{tg}\n')
            print('if you are really sure, set nuker.IamSure = True')
        else:
            if type(self._target_list) == str:
                shutil.rmtree(self._dr)
            if type(self._target_list) == list:
                for tg in self._target_list:
                    shutil.rmtree(f'{self._dr}{tg}')


def sec_to_clock(seconds):
    # Calculate hours, minutes, and seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)

    # Format the result as hh:mm:ss
    return f'{hours:02d}:{minutes:02d}:{remaining_seconds:02d}'


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
        idx = idx[-maxPoints:]
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


def ensure_dir(name, loc='', k=0, first=True):
    dr0 = f'{loc}datalogs/{name}/run_1'
    while os.path.exists(dr0):
        aux = int(dr0.split('_')[-1]) + 1
        dr0 = dr0.split('_')[0] + '_' + str(aux)
    
    if not first:
        aux = int(dr0.split('_')[-1]) - 1
        dr0 = dr0.split('_')[0] + '_' + str(aux)

    dr = dr0 + f'/k{k}'

    os.makedirs(dr)
    os.makedirs(f'{dr}/plots')
    os.makedirs(f'{dr}/plots/histograms')
    os.makedirs(f'{dr}/plots/GMEstimates')
    os.makedirs(f'{dr}/plots/traces')
    os.makedirs(f'{dr}/plots/models')
    os.makedirs(f'{dr}/plots/models/uncertainpy')
    os.makedirs(f'{dr}/plots/posteriors')
    os.makedirs(f'{dr}/plots/posteriors/scatter')
    os.makedirs(f'{dr}/plots/posteriors/hexbin')
    os.makedirs(f'{dr}/plots/posteriors/gaussian')
    os.makedirs(f'{dr}/plots/posteriors/chains')

    '''
    os.makedirs(f'{dr}/maxlike/plots/models')
    os.makedirs(f'{dr}/maxlike/plots/models/uncertainpy')
    os.makedirs(f'{dr}/maxlike/plots/posteriors')
    os.makedirs(f'{dr}/maxlike/plots/posteriors/scatter')
    os.makedirs(f'{dr}/maxlike/plots/posteriors/hexs')
    os.makedirs(f'{dr}/maxlike/plots/posteriors/gaussian')
    os.makedirs(f'{dr}/maxlike/temp')
    os.makedirs(f'{dr}/maxlike/temp/models')
    '''
    
    os.makedirs(f'{dr}/samples')
    os.makedirs(f'{dr}/samples/posteriors')
    os.makedirs(f'{dr}/samples/likelihoods')
    os.makedirs(f'{dr}/samples/chains')

    os.makedirs(f'{dr}/temp')
    os.makedirs(f'{dr}/temp/models')

    os.makedirs(f'{dr}/restore')
    os.makedirs(f'{dr}/restore/backends')

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


def parallel_func(func, foo, n_threads):
    import multiprocessing as mp
    pool = mp.Pool(processes=n_threads)
    results = pool.map(func, foo)
    pool.close()
    pool.join()
    return results


def find_common_integer_sequence(numbers):
    # Extract the integer part and convert to string
    str_numbers = [str(int(number)) for number in numbers]

    # Check the length of the shortest number to avoid index errors
    min_length = min(len(num) for num in str_numbers)

    # Find common sequence
    common_sequence = ''
    for i in range(min_length):
        # Check if this character is the same in all numbers
        if all(num[i] == str_numbers[0][i] for num in str_numbers):
            common_sequence += str_numbers[0][i]
        else:
            break

    # Convert the common sequence back to a number, filling with zeros
    if common_sequence:
        common_sequence = int(common_sequence + '0' * (len(str_numbers[0]) - len(common_sequence)))
    else:
        common_sequence = None

    return common_sequence


def find_confidence_intervals(sigma):
    from scipy.stats import norm
    if sigma <= 0:
        raise ValueError('sigma must be greater than 0')
    
    p = np.diff(norm.cdf([-sigma, sigma]))[0]
    return np.array([1-p, p]) * 100


def my_shell_env():
    try:
        #
        shl = get_ipython().__class__.__name__
        if shl == 'ZMQInteractiveShell':
            return 'jupyter-notebook'
        elif shl == 'TerminalInteractiveShell':
            return 'ipython-terminal'
        elif get_ipython().__class__.__module__ == 'google.colab._shell':
            return 'google-colab'
        
    except NameError:
        return 'python-terminal'


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
        self.baddies_list = ["\x1b[", '0m', '1m', '4m', '7m', '31m', '32m']

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
            msg0 = msg
            for b in self.baddies_list:
                msg0 = msg0.replace(b, '')

            self.log += msg0
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
            else:
                break                


        self.gm_estimator = gm0
        return self.gm_estimator


class DataWrapper1(object):
    def __init__(self, target_name, read_loc=''):
        self.target_name = target_name
        self.RV_PATH = f'{read_loc}datafiles/{self.target_name}/RV/'
        self.AM_PATH = f'{read_loc}datafiles/{self.target_name}/AM/'
        self.PM_PATH = f'{read_loc}datafiles/{self.target_name}/PM/'


        RV_empty_lists = ['ndata', 'ncols', 'nsai', 'data', 'RV_labels',
                          'RV_sets']
        
        for attribute in RV_empty_lists:
            setattr(self, attribute, [])

        AM_empty_lists = ['AM_labels','AM_sets',
                          'df_gost', 'df_hipgaia', 'df_hip',
                          ]

        for attribute in AM_empty_lists:
            setattr(self, attribute, [])


    def add_rv_data__(self, filename):
        data = np.loadtxt('{0}{1}'.format(self.RV_PATH, filename))

        ndat, ncol = data.shape

        self.ndata.append(ndat)
        self.ncols.append(ncol)
        self.RV_labels.append(filename)

        names = ['BJD', 'RV', 'eRV']

        # identify and name SAI
        nsa = ncol - 3
        if nsa > 0:
            names.extend(f'Staract {len(self.ndata)} {j}' for j in range(nsa))
        self.nsai.append(nsa)

        df = pd.DataFrame(data, columns=names)

        # substract RV
        if abs(df.mean()['RV']) > 1e-6:
            df['RV'] -= df.mean()['RV']

        for nam in names:
            if nam[:3] == 'Sta':
                #if abs(df.mean()[nam]) > 1e-6:
                df[nam] -= df.mean()[nam]
                df[nam] = (df[nam] - df.min()[nam]) /(df.max()[nam]-df.min()[nam]) * (df.max()['RV'] - df.min()['RV']) + df.min()['RV']
                #df[nam] = df[nam] / (df.max()[nam] - df.min()[nam])

        # create another column containing flags for the instrument
        df.insert(loc=3, column='Flag', value=np.ones(ndat, int) * len(self.ndata))
        self.data.append(df)

        return 'Reading data from {0}'.format(filename)


    def add_all__(self):
        # rv
        my_files = list(np.sort(os.listdir(self.RV_PATH)))
        # mac os fix
        try:
            my_files.remove('.DS_Store')
        except:
            pass
        x = ''.join('\n'+self.add_rv_data__(file) for file in my_files)

        # am
        self.AM_sets = list(np.sort(os.listdir(self.AM_PATH)))
        if len(self.AM_sets) > 0:
            for file in self.AM_sets:
                identifier = file.split('_')[-1]
                ff = self.AM_PATH+file

                if identifier == 'gost.csv':
                    self.df_gost = pd.read_csv(ff)

                if identifier == 'hipgaia.hg123':
                    self.df_hipgaia = pd.read_csv(ff, sep='\s+')

                if identifier == 'hip2.abs':
                    self.df_hip = pd.read_csv(ff, sep='\s+')

                x = ''.join('\nReading data from {0}'.format(file))

            self.make_readable()
            self.get_epochs()

        return x+'\n'


    def get_data__(self, sortby='BJD'):
        asd = pd.concat(self.data).sort_values(sortby)
        self.common_t = asd['BJD'].min()
        asd['BJD'] -= self.common_t
        return asd


    def get_metadata__(self):
        return [getattr(self, attribute) for attribute in ['ndata', 'ncols', 'nsai', 'RV_labels']]


    def get_data_raw(self, sortby):
        holder = pd.concat(self.data).sort_values(sortby)
        x, y, yerr = holder.BJD, holder.RV, holder.eRV
        return [x, y, yerr]


    def make_readable(self):
        lists = ['df_gost', 'df_hipgaia', 'df_hip']

        if len(self.df_gost) > 0:
            A = 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'
            B = 'scanAngle[rad]'
            C = 'parallaxFactorAlongScan'
            D = 'parallaxFactorAcrossScan'
            column_mapping = {A: 'BJD',
                              B: 'psi',
                              C: 'parf',
                              D: 'parx',
                              }
            self.df_gost = self.df_gost.rename(columns=column_mapping)


    def get_epochs(self):
        t = self.df_gost.BJD.values
        self.hipp_epoch = self.time_all_2jd(1991.25, fmt='decimalyear')  # 2448348.75

        self.gdr1_ref = 2457023.5  # self.time_all_2jd(2015, fmt='decimalyear')  # 2457023.5
        self.gdr2_ref = 2457206  # self.time_all_2jd(2015.5, fmt='decimalyear')  # 2457206 hardcode?
        self.gdr3_ref = 2457388.5  # self.time_all_2jd(2016, fmt='decimalyear')  # 2457388.5

        self.gdr1_epoch = [self.time_all_2jd('2014-07-25 10:30:00'),
                           self.time_all_2jd('2015-09-16 16:00:00')]
        self.gdr2_epoch = [self.time_all_2jd('2014-07-25 10:30:00'),
                           self.time_all_2jd('2016-05-23 11:35:00')]
        self.gdr3_epoch = [self.time_all_2jd('2014-07-25 10:30:00'),
                           self.time_all_2jd('2017-05-28 08:44:00')]

        self.mask_hipp = (self.hipp_epoch <= t) & (t <= self.gdr1_epoch[0])
        self.mask_gdr1 = (self.gdr1_epoch[0] <= t) & (t<= self.gdr1_epoch[1])
        self.mask_gdr2 = (self.gdr2_epoch[0] <= t) & (t<= self.gdr2_epoch[1])
        self.mask_gdr3 = (self.gdr3_epoch[0] <= t) & (t<= self.gdr3_epoch[1])

        self.iref = self.hipp_epoch


    def time_all_2jd(self, time_str, fmt='iso'):
        t = AstroTime(time_str, format=fmt)
        return t.to_value('jd')


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


from astropy.time import Time as AstroTime

class SDataWrapper(object):
    def __init__(self, target_name, read_loc=''):
        self.target_name = target_name
        self.modes = {'RV':{'PATH':f'{read_loc}datafiles/{self.target_name}/RV/',
                            },
                      'PM':{'PATH':f'{read_loc}datafiles/{self.target_name}/PM/',
                            },
                      'AM':{'PATH':f'{read_loc}datafiles/{self.target_name}/AM/',
                            },
                      }


        RV_empty_lists = ['ndata', 'ncols', 'nsai', 'data', 'RV_labels',
                          'RV_sets']
        
        PM_empty_lists = []

        AM_empty_lists = ['AM_labels','AM_sets',
                          'df_gost', 'df_hipgaia', 'df_hip',
                          ]



        for attribute in RV_empty_lists:
            setattr(self, attribute, [])

        for attribute in AM_empty_lists:
            setattr(self, attribute, [])


    def activate_modes(self):
        for p in self:
            if os.path.exists(p):
                filenames = list(np.sort(os.listdir(p)))
                if filenames > 0:
                    pass

    def add_rv_data__(self, filename):
        data = np.loadtxt('{0}{1}'.format(self.RV_PATH, filename))

        ndat, ncol = data.shape

        self.ndata.append(ndat)
        self.ncols.append(ncol)
        self.RV_labels.append(filename)

        names = ['BJD', 'RV', 'eRV']

        # identify and name SAI
        nsa = ncol - 3
        if nsa > 0:
            names.extend(f'Staract {len(self.ndata)} {j}' for j in range(nsa))
        self.nsai.append(nsa)

        df = pd.DataFrame(data, columns=names)

        # substract RV
        if abs(df.mean()['RV']) > 1e-6:
            df['RV'] -= df.mean()['RV']

        for nam in names:
            if nam[:3] == 'Sta':
                #if abs(df.mean()[nam]) > 1e-6:
                df[nam] -= df.mean()[nam]
                df[nam] = (df[nam] - df.min()[nam]) /(df.max()[nam]-df.min()[nam]) * (df.max()['RV'] - df.min()['RV']) + df.min()['RV']
                #df[nam] = df[nam] / (df.max()[nam] - df.min()[nam])

        # create another column containing flags for the instrument
        df.insert(loc=3, column='Flag', value=np.ones(ndat, int) * len(self.ndata))
        self.data.append(df)

        return 'Reading data from {0}'.format(filename)


    def add_all__(self):
        # rv
        my_files = list(np.sort(os.listdir(self.RV_PATH)))
        # mac os fix
        try:
            my_files.remove('.DS_Store')
        except:
            pass
        x = ''.join('\n'+self.add_rv_data__(file) for file in my_files)

        # am
        self.AM_sets = list(np.sort(os.listdir(self.AM_PATH)))
        if len(self.AM_sets) > 0:
            for file in self.AM_sets:
                identifier = file.split('_')[-1]
                ff = self.AM_PATH+file

                if identifier == 'gost.csv':
                    self.df_gost = pd.read_csv(ff)

                if identifier == 'hipgaia.hg123':
                    self.df_hipgaia = pd.read_csv(ff, sep='\s+')

                if identifier == 'hip2.abs':
                    self.df_hip = pd.read_csv(ff, sep='\s+')

                x = ''.join('\nReading data from {0}'.format(file))

            self.make_readable()
            self.get_epochs()

        return x+'\n'


    def get_data__(self, sortby='BJD'):
        asd = pd.concat(self.data).sort_values(sortby)
        self.common_t = asd['BJD'].min()
        asd['BJD'] -= self.common_t
        return asd


    def get_metadata__(self):
        return [getattr(self, attribute) for attribute in ['ndata', 'ncols', 'nsai', 'RV_labels']]


    def get_data_raw(self, sortby):
        holder = pd.concat(self.data).sort_values(sortby)
        x, y, yerr = holder.BJD, holder.RV, holder.eRV
        return [x, y, yerr]


    def make_readable(self):
        lists = ['df_gost', 'df_hipgaia', 'df_hip']

        if len(self.df_gost) > 0:
            A = 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'
            B = 'scanAngle[rad]'
            C = 'parallaxFactorAlongScan'
            D = 'parallaxFactorAcrossScan'
            column_mapping = {A: 'BJD',
                              B: 'psi',
                              C: 'parf',
                              D: 'parx',
                              }
            self.df_gost = self.df_gost.rename(columns=column_mapping)


    def get_epochs(self):
        t = self.df_gost.BJD.values
        self.hipp_epoch = self.time_all_2jd(1991.25, fmt='decimalyear')  # 2448348.75

        self.gdr1_ref = 2457023.5  # self.time_all_2jd(2015, fmt='decimalyear')  # 2457023.5
        self.gdr2_ref = 2457206  # self.time_all_2jd(2015.5, fmt='decimalyear')  # 2457206 hardcode?
        self.gdr3_ref = 2457388.5  # self.time_all_2jd(2016, fmt='decimalyear')  # 2457388.5

        self.gdr1_epoch = [self.time_all_2jd('2014-07-25 10:30:00'),
                           self.time_all_2jd('2015-09-16 16:00:00')]
        self.gdr2_epoch = [self.time_all_2jd('2014-07-25 10:30:00'),
                           self.time_all_2jd('2016-05-23 11:35:00')]
        self.gdr3_epoch = [self.time_all_2jd('2014-07-25 10:30:00'),
                           self.time_all_2jd('2017-05-28 08:44:00')]

        self.mask_hipp = (self.hipp_epoch <= t) & (t <= self.gdr1_epoch[0])
        self.mask_gdr1 = (self.gdr1_epoch[0] <= t) & (t<= self.gdr1_epoch[1])
        self.mask_gdr2 = (self.gdr2_epoch[0] <= t) & (t<= self.gdr2_epoch[1])
        self.mask_gdr3 = (self.gdr3_epoch[0] <= t) & (t<= self.gdr3_epoch[1])

        self.iref = self.hipp_epoch


    def time_all_2jd(self, time_str, fmt='iso'):
        t = AstroTime(time_str, format=fmt)
        return t.to_value('jd')


    def __repr__(self):
        return self.modes


    def __getitem__(self, string):
        return self.modes[string]

'''
amd = AMDataWrapper(sim.starname)
amd.AM_PATH
amd.filenames

amd = AMDataWrapper(sim.starname)
amd.AM_PATH

'''

#################

'''

from datetime import datetime
import math

def time_cal2jd(yr, mn, dy):
    """
    Convert Gregorian Calendar date to Julian Date.
    
    Input:
        cal - Calendar date with day fraction in the format [year, month, day_fraction].
    
    Output:
        JD - 2-part Julian Date.
    """
    y = yr
    m = mn
    
    # Adjust months for dates in Jan/Feb
    ind = mn <= 2
    y[ind] -= 1
    m[ind] += 12
    
    # Julian and Gregorian calendar switch dates
    date1 = 4.5 + 31 * (10 + 12 * 1582)  # Last day of Julian calendar (1582.10.04 Noon)
    date2 = 15.5 + 31 * (10 + 12 * 1582)  # First day of Gregorian calendar (1582.10.15 Noon)
    
    date = dy + 31 * (mn + 12 * yr)
    
    # Identify dates before and after the switch
    ind1 = date <= date1
    ind2 = date >= date2
    
    b = np.copy(y)
    b[ind1] = -2
    b[ind2] = np.trunc(y[ind2] / 400) - np.trunc(y[ind2] / 100)
    
    if not ind1.any() and not ind2.any():
        print('Dates between October 5 & 15, 1582 do not exist!')
    
    # Compute Julian Date
    jd = np.copy(y)
    jd[y > 0] = (np.trunc(365.25 * y[y > 0]) +
                 np.trunc(30.6001 * (m[y > 0] + 1)) +
                 b[y > 0] + 1720996.5 + dy[y > 0])
    
    jd[y < 0] = (np.trunc(365.25 * y[y < 0] - 0.75) +
                 np.trunc(30.6001 * (m[y < 0] + 1)) +
                 b[y < 0] + 1720996.5 + dy[y < 0])
    
    # Return 2-part Julian Date
    return np.vstack((jd // 1, jd % 1)).T


def doy_to_jd(year, doy):
    # Convert Day of Year to Julian Date for a given year
    date = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return cal_to_jd(date.year, date.month, date.day)

def time_yr2jd(yr):
    iyr = math.floor(yr)
    jd0 = cal_to_jd(iyr, 1, 1)
    days = cal_to_jd(iyr + 1, 1, 1) - jd0
    doy = (yr - iyr) * days + 1
    return doy_to_jd(iyr, doy)

def deg_to_mas(o1, o2):
    do = o2 - o1 
    do = do * 3.6e6
    do = do * np.cos(o2*np.pi/180)
    return do

'''

class DataWrapper(object):
    def __init__(self, target_name, read_loc=''):
        self.target_name = target_name
        self.modes = {'RV':{'PATH':f'{read_loc}datafiles/{self.target_name}/RV/',
                            'KEY':'RV',
                            },
                      'PM':{'PATH':f'{read_loc}datafiles/{self.target_name}/PM/',
                            'KEY':'PM',
                            },
                      'AM':{'PATH':f'{read_loc}datafiles/{self.target_name}/AM/',
                            'KEY':'AM',
                            },
                      }   

        self.RV_empty_lists = ['ndata', 'ncols', 'nsai', 'data', 'RV_labels',
                               'RV_sets']
        
        self.PM_empty_lists = []

        self.AM_empty_lists = ['AM_labels','AM_sets',
                               'df_gost', 'df_hipgaia', 'df_hip',
                               ]

        self.activate()


    def activate(self):
        for m in self:
            p = m['PATH']
            m['use'] = False
            if os.path.exists(p):
                # macos fix
                filenames = [fn for fn in sorted(os.listdir(p)) if fn != '.DS_Store']

                if len(filenames) > 0:
                    m['use'] = True
                    m['filenames'] = filenames
                    for attribute in getattr(self, f"{m['KEY']}_empty_lists"):
                        m[attribute] = []

        for m in self:
            if m['use']:
                m['logger_msg'] = getattr(self, f"mk_{m['KEY']}")()
        

    def mk_RV(self):
        m = self['RV']
        str2prt = ''
        for file in m['filenames']:
            data = np.loadtxt('{0}{1}'.format(m['PATH'], file))
            ndat, ncol = data.shape

            m['ndata'].append(ndat)
            m['ncols'].append(ncol)
            m['RV_labels'].append(file)

            names = ['BJD', 'RV', 'eRV']

            # identify and name SAI
            nsa = ncol - 3
            if nsa > 0:
                names.extend(f"Staract {len(m['ndata'])} {j}" for j in range(nsa))

            m['nsai'].append(nsa)

            # make dataframe
            df = pd.DataFrame(data, columns=names)
            
            if abs(df.mean()['RV']) > 1e-6:
                df['RV'] -= df.mean()['RV']

            for nam in names:
                if nam[:3] == 'Sta':
                    #if abs(df.mean()[nam]) > 1e-6:
                    df[nam] -= df.mean()[nam]
                    df[nam] = (df[nam] - df.min()[nam]) /(df.max()[nam]-df.min()[nam]) * (df.max()['RV'] - df.min()['RV']) + df.min()['RV']
                    #df[nam] = df[nam] / (df.max()[nam] - df.min()[nam])

            # create another column containing flags for the instrument
            df.insert(loc=3, column='Flag', value=np.ones(ndat, int) * len(m['ndata']))
            
            m['data'].append(df)
            str2prt += 'Reading data from {0}\n'.format(file)

        return str2prt


    def mk_AM(self):
        m = self['AM']
        str2prt = ''
        for file in m['filenames']:
            identifier = file.split('_')[-1]
            ff = m['PATH']+file

            if identifier == 'gost.csv':
                m['df_gost'] = pd.read_csv(ff)

            elif identifier == 'hipgaia.hg123':
                m['df_hipgaia'] = pd.read_csv(ff, sep='\s+')

            elif identifier == 'hip2.abs':
                m['df_hip'] = pd.read_csv(ff)

            else:
                print(f'File format not identified for {identifier}')

            str2prt += 'Reading data from {0}\n'.format(file)


        self.astrometry_human_r()
        self.astrometry_epochs()

        self.astro_gost()

        return str2prt


    def mk_PM(self):
        pass


    def get_data__(self, sortby='BJD'):
        m = self['RV']
        asd = pd.concat(m['data']).sort_values(sortby)
        m['common_t'] = asd['BJD'].min()
        asd['BJD'] -= m['common_t']
        return asd


    def astrometry_human_r(self):
        m = self['AM']
        if isinstance(m['df_gost'], pd.DataFrame):
            A = 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'
            B = 'scanAngle[rad]'
            C = 'parallaxFactorAlongScan'
            D = 'parallaxFactorAcrossScan'
            column_mapping = {A: 'BJD',
                              B: 'psi',
                              C: 'parf',
                              D: 'parx',
                              'ra[rad]':'RA',
                              'dec[rad]':'DEC',
                              }
            m['df_gost'] = m['df_gost'].rename(columns=column_mapping)


    def astrometry_epochs(self):
        m = self['AM']
        m0 = m['df_gost']
        if isinstance(m0, pd.DataFrame):
            t = m0.BJD.values
            m['hipp_epoch'] = self.time_all_2jd(1991.25, fmt='decimalyear')  # 2448348.75

            m['gdr1_ref'] = 2457023.5  # self.time_all_2jd(2015, fmt='decimalyear')  # 2457023.5
            m['gdr2_ref'] = 2457206  # self.time_all_2jd(2015.5, fmt='decimalyear')  # 2457206 hardcode?
            m['gdr3_ref'] = 2457388.5  # self.time_all_2jd(2016, fmt='decimalyear')  # 2457388.5

            m['gdr1_epoch'] = [self.time_all_2jd('2014-07-25 10:30:00'),
                            self.time_all_2jd('2015-09-16 16:00:00')]
            m['gdr2_epoch'] = [self.time_all_2jd('2014-07-25 10:30:00'),
                            self.time_all_2jd('2016-05-23 11:35:00')]
            m['gdr3_epoch'] = [self.time_all_2jd('2014-07-25 10:30:00'),
                            self.time_all_2jd('2017-05-28 08:44:00')]

            m['mask_hipp'] = (m['hipp_epoch'] <= t) & (t <= m['gdr1_epoch'][0])
            m['mask_gdr1'] = (m['gdr1_epoch'][0] <= t) & (t<= m['gdr1_epoch'][1])
            m['mask_gdr2'] = (m['gdr2_epoch'][0] <= t) & (t<= m['gdr2_epoch'][1])
            m['mask_gdr3'] = (m['gdr3_epoch'][0] <= t) & (t<= m['gdr3_epoch'][1])

            m['iref'] = m['hipp_epoch']


    def astro_gost(self):
        names = ['dra','ddec','parallax','pmra','pmdec']
        dra = 1
        ddec = 1

        data = []

        self['AM']['astro_gost'] = pd.DataFrame(data, columns=names)


    def time_all_2jd(self, time_str, fmt='iso'):
        t = AstroTime(time_str, format=fmt)
        return t.to_value('jd')


    def __repr__(self):
        x = ''
        for p in self:
            x += f'{p}\n'
        return x


    def __getitem__(self, string):
        return self.modes[string]


    def __iter__(self):
        for key in self.modes:
            yield self.modes[key]


#dw = minidw('HD209100')
#for p in dw:
#    print(p)

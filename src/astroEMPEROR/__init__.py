# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.5.1
# date 8 nov 2022

if True:
    import numpy as np
    import matplotlib.pyplot as pl
    from matplotlib import ticker
    import matplotlib.gridspec as gridspec
    from tqdm import tqdm
    import multiprocessing
    import sys, os
    import warnings
    import scipy.stats
    from copy import copy
    from tabulate import tabulate
    from PyAstronomy.pyasl import MarkleyKESolver
    ks = MarkleyKESolver()
    import kepler

    _ROOT = os.path.dirname(__file__)
    sys.path.insert(1, _ROOT)
    from utils import *
    #from model_repo import _model_keplerian, _model_sinusoid, _model_moav, _model_acc, _model_keplerian_pymc3, _model_scale
    import model_repo as mr

def _logp__(theta, my_model):
    return my_model.evaluate_prior(theta)

def _logl__(theta, my_model):
    # sin constantes!
    model, ferr2 = my_model.evaluate_model(theta)  # ferr is (errs**2)
    residuals = my_model.y - model
    #residuals, ferr = my_model.evaluate(theta)  # ferr is (errs**2)

    inv_sigma2 = 1.0 / (ferr2)

    lnl = np.sum(residuals**2 * inv_sigma2 - np.log(inv_sigma2)) + np.log(2*np.pi) * len(my_model.x)

    return -0.5 * lnl

def _ptform__(u, ptformargs):
    x = np.array(u)
    #return my_model.evaluate_prior(x)

    for i in range(len(x)):
        a, b = ptformargs[i]
        x[i] =  a * (2. * x[i] - 1) + b
    return x


class Parameter:
    def __init__(self, value, prior, limits, type=None, name=None, prargs=None, ptformargs=None):
        self.value = value
        self.prior = prior
        self.limits = limits
        self.type = type
        self.name = name
        self.prargs = prargs
        self.ptformargs = ptformargs

        self.fixed = None
        self.sigma = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self.name)


class Parameter_Block:
    def __init__(self, params, block_model=None, block_name=None, block_type=None):
        self.list_ = np.array(params)
        self.ndim_ = len(self.list_)
        self.name_ = block_name
        self.type_ = block_type
        self.model = block_model

        self.extra_args = []

        self.A_ = []  # anticoordinator
        self.C_ = []  # coordinator
        self.parametrization = 0
        self.hou_priors = []
        self.MOAV_ = False
        self.RV = True
        self.F = False
        self.batman_initialized = False


    def __repr__(self):
        return str([str(x) for x in self.list_])


    def __len__(self):
        return len(self.list_)


    def __getitem__(self, n):
        return self.list_[n]


    def __str__(self):
        return self.name_


    def _list_(self, *call):
        if len(call) == 1:
            return np.array([getattr(self.list_[i], call[0]) for i in range(len(self))])
        else:
            return np.array([[getattr(self.list_[i], c) for i in range(len(self))] for c in call], dtype=object)


    def _refresh_(self):
        self.C_ = []
        self.A_ = []
        ndim = len(self)


        for p in self:
            # fixed values
            if p.fixed is not None:
                self.C_.append(False)
                p.value = p.fixed
                p.prior = 'Fixed'
                p.limits = [np.nan, np.nan]
                ndim -= 1
            else:
                self.C_.append(True)
        self.ndim_ = ndim
        pass


    def _calc_priors_(self, theta):
        lp = 0.
        for i in range(len(self)):
            lp += getattr(my_stats, self[i].prior)(theta[i], self[i].limits, self[i].prargs)


        if self.hou_priors:
            for extra in self.hou_priors:
                name, pr, lims, pra = extra
                if name == 'Period':
                    x = np.exp(theta[0])
                if name == 'Amplitude':
                    x = theta[1]**2 + theta[2]**2
                if name == 'Eccentricity':
                    x = theta[3]**2 + theta[4]**2
                else:
                    continue
                lp += getattr(my_stats, pr)(x, lims, pra)

        return lp


    def _calc_priors_debug_(self, theta):
        lp = 0.
        flist = np.array([])
        for i in range(len(self)):
            lpv = getattr(my_stats, self[i].prior)(theta[i], self[i].limits, self[i].prargs)
            if lpv == -np.inf:
                print(self[i].name, theta[i], self[i].limits, lpv, '\n')
                flist = np.append(list, self[i].name)
            lp += lpv

        if self.hou_priors:
            for extra in self.hou_priors:
                name, pr, lims, pra = extra
                if name == 'Period':
                    x = np.exp(theta[0])
                if name == 'Amplitude':
                    x = theta[1]**2 + theta[2]**2
                if name == 'Eccentricity':
                    x = theta[3]**2 + theta[4]**2
                else:
                    continue
                lpv = getattr(my_stats, pr)(x, lims, pra)
                if lpv == -np.inf:
                    flist = np.append(list, name)
                    print('Hou, ', name, x, lims, lpv, '\n')  # debug
                lp += lpv
        return lp, flist


    def _init_batman_(self, time):
        if not self.batman_initialized:
            import batman
            # initialize batman model
            self.ld_dctn = {'uniform': 0,
                        'linear': 1,
                        'quadratic': 2,
                        'square-root': 2,
                        'logarithmic': 2,
                        'exponential': 2,
                        'power2': 2,
                        'nonlinear': 2
                        }  # dictionary with limb darkening dimentionality
            if self.type_ == 'Transit':
                self.bparams = batman.TransitParams()       #object to store transit parameters
                par_dctn = {'per':1., 'a':15., 't0':0., 'ecc':0., 'w':90., 'rp':0.1, 'inc':87.}
                for x in par_dctn:
                    setattr(self.bparams, x, par_dctn[x])
                #params.per = 1.                       #orbital period
                #params.a = 15.                        #semi-major axis (in units of stellar radii)
                #params.t0 = 0.                        #time of inferior conjunction
                #params.ecc = 0.                       #eccentricity
                #params.w = 90.                        #longitude of periastron (in degrees)
                #params.rp = 0.1                       #planet radius (in units of stellar radii)
                #params.inc = 87.                      #orbital inclination (in degrees)

                if self.extra_args[0]:
                    ldn = self.extra_args[0]
                    self.bparams.limb_dark = self.extra_args[0]        #limb darkening model
                    self.bparams.u = []      #limb darkening coefficients [u1, u2, u3, u4]
                    for i in range(self.ld_dctn[ldn]):
                        self.bparams.u.append(0.1)

                t = np.linspace(-0.025, 0.025, 1000)  #times at which to calculate light curve
                self.batman_m = batman.TransitModel(self.bparams, time)    #initializes model
                self.extra_args.append(self.batman_m)
                self.extra_args.append(self.bparams)
                self.batman_initialized = True

        pass


class Model:
    def __init__(self, data, bloques):
        self.bloques = bloques

        self.HILL = False
        self.starmass = None
        self.mod_fixed = []

        self.kplanets = 0
        for b in self.bloques:
            self.mod_fixed.append(b._list_('fixed'))
            if b.type_ == 'Keplerian':
                self.kplanets += 1

        self.x = data[0]
        self.y = data[1]
        self.yerr = data[2]
        self.flags = data[3]
        #self.ndim = flatten(self.bloques)


        self.fixed_values = flatten(self.mod_fixed)

        self.A_ = []
        self.C_ = []
        for i in range(len(self.fixed_values)):
            if self.fixed_values[i] != None:
                self.A_.append(i)
            else:
                self.C_.append(i)


    def evaluate_model(self, theta):
        n = 0
        mod0 = np.zeros_like(self.y)
        err20 = (self.yerr + 0) ** 2
        #jplanets = 0

        for a in self.A_:
             theta = np.insert(theta, a, self.fixed_values[a])

        for b in self:
            if b.MOAV_:
                b.extra_args[2] = self.y - mod0
            #if b.type_ == 'Transit':
            #    jplanets += 1
            thet = theta[n : n+len(b)]
            mod, ferr = b.model(thet, self.x, self.y, err20, self.flags, b.extra_args)
            mod0 += mod
            err20 += ferr
            n += len(b)
            #if jplanets > 1:
            #    mod0 -= 1

        return mod0, err20  # returns model and errors ** 2


    def evaluate_plot(self, theta, x):
        n = 0
        mod_total = np.zeros_like(x)
        ferr = 0
        #jplanets = 0
        for a in self.A_:
             theta = np.insert(theta, a, self.fixed_values[a])

        minx, maxx = minmax(x)

        for b in self:
            if b.MOAV_:
                b.extra_args[2] = np.zeros_like(x)
                extra_args_copy = [b.extra_args[0], b.extra_args[1], np.zeros_like(x)]
                thet = theta[n : n+len(b)]
                mod, ferr = b.model(thet, x, mod_total, ferr, np.zeros_like(x ), extra_args_copy)
            if b.type_ == 'Transit':
                import batman
        #        jplanets += 1
                extra_args_copy = [b.extra_args[0], b.extra_args[1], batman.TransitModel(b.extra_args[3], x), b.extra_args[3]]
                thet = theta[n : n+len(b)]
                mod, ferr = b.model(thet, x, mod_total, ferr, np.zeros_like(x ), extra_args_copy)

            else:
                thet = theta[n : n+len(b)]
                mod, ferr = b.model(thet, x, mod_total, ferr, np.zeros_like(x ), b.extra_args)

            mod_total += mod
            n += len(b)
        #    if jplanets > 1:
        #        mod_total -= 1
        return mod_total, ferr


    def evaluate_fmodel(self, theta):
        n = 0
        mod0 = np.zeros_like(self.y)
        err20 = (self.yerr + 0) ** 2
        transits = 0
        for a in self.A_:
             theta = np.insert(theta, a, self.fixed_values[a])

        for b in self:
            if b.MOAV_:
                b.extra_args[2] = self.y - mod0
            if b.type_ == 'Transit':
                transits += 1
            thet = theta[n : n+len(b)]
            mod, ferr = b.model(thet, self.x, self.y, err20, self.flags, b.extra_args)
            mod0 += mod

            err20 += ferr
            n += len(b)
        mod0 -= (transits - 1)

        return mod0, err20  # returns model and errors ** 2


    def evaluate_fplot(self, theta, x):
        n = 0
        mod_total = np.zeros_like(x)
        ferr = 0
        transits = 0
        for a in self.A_:
             theta = np.insert(theta, a, self.fixed_values[a])

        minx, maxx = minmax(x)

        for b in self:
            if b.MOAV_:
                b.extra_args[2] = np.zeros_like(x)
            if b.type_ == 'Transit':
                transits += 1
                import batman
                extra_args_copy = [b.extra_args[0], batman.TransitModel(b.extra_args[2], x), b.extra_args[2]]
            thet = theta[n : n+len(b)]
            mod, ferr = b.model(thet, x, mod_total, ferr, np.zeros_like(x ), extra_args_copy)
            mod_total += mod
            n += len(b)

            mod_total -= (transits - 1)
        return mod_total, ferr


    def evaluate_prior(self, theta):
        n = 0
        lp = 0.
        for a in self.A_:
             theta = np.insert(theta, a, self.fixed_values[a])

        for b in self:
            thet = theta[n : n+len(b)]
            lp += b._calc_priors_(thet)
            n += len(b)

        if self.HILL:
            lp += self.evaluate_hill(theta)
            pass

        return lp


    def evaluate_hill(self, theta):
        if self.kplanets > 1:
            periods = theta[np.arange(self.kplanets)*5]
            #print('\nTest0', periods)
            if self[0].parametrization == 1 or self[0].parametrization == 'hou':
                periods = np.exp(periods)
                amps = theta[np.arange(self.kplanets)*5 + 1] ** 2 + theta[np.arange(self.kplanets)*5 + 2] ** 2
                eccs = theta[np.arange(self.kplanets)*5 + 3] ** 2 + theta[np.arange(self.kplanets)*5 + 4] ** 2
            elif self[0].parametrization == 3 or self[0].parametrization == 'houtp':
                eccs = theta[np.arange(self.kplanets)*5 + 3] ** 2 + theta[np.arange(self.kplanets)*5 + 4] ** 2
            else:
                amps = theta[np.arange(self.kplanets)*5 + 1]
                eccs = theta[np.arange(self.kplanets)*5 + 3]

            gamma = np.sqrt(1 - eccs)

            #print('\nTest1', amps)
            #print('\nTest2', gamma)
            sma, minmass = cps(periods, amps, eccs, self.starmass)

            #print('\nTest3', sma)
            orden = np.argsort(sma)
            sma = sma[orden]  # in AU
            minmass = minmass[orden]  # in Earth Masses

            periods, amps, eccs = periods[orden], amps[orden], eccs[orden]

            M = self.starmass * 1047.56 + np.sum(minmass)  # jupiter masses
            mu = minmass / M
            for k in range(self.kplanets-1):
                alpha = mu[k] + mu[k+1]
                delta = np.sqrt(sma[k+1] / sma[k])

                LHS = alpha**-3 * (mu[k] + (mu[k+1] / (delta**2))) * (mu[k] * gamma[k] + mu[k+1] * gamma[k+1] * delta)**2
                RHS = 1 + (3./alpha)**(4./3) * (mu[k] * mu[k+1])
                #LHS = delta
                #RHS = 2.4*alpha**(1./3)
                if LHS > RHS:
                    return 0.
                else:
                    return -np.inf
                pass


    def debug_params(self, theta):
        n = 0
        lp = 0.
        flist = np.array([])
        for a in self.A_:
             theta = np.insert(theta, a, self.fixed_values[a])

        for b in self:
            thet = theta[n : n+len(b)]

            foo, bar = b._calc_priors_debug_(thet)
            lp += foo
            flist = np.append(flist, bar)
            n += len(b)

        return lp, flist


    def convert_dynesty(self, theta):
        pass


    def __getitem__(self, n):
        return self.bloques[n]


class Simulation:
    def __init__(self):
        self.blocks__ = []
        self.ndim__ = int(np.sum(self._list__('ndim_')))

        self.kplanets__ = 0
        self.nins__ = 0


        self.cores__ = multiprocessing.cpu_count() - 1

        Nonethings = ['starmass', 'starname', 'betas', 'saveplace']
        for c in Nonethings:
            setattr(self, c, None)

        Falsethings = ['']

        # plots and displays
        #self.saveplace = None
        self.plot_show = False
        self.plot_save = False
        self.plot_fmt = 'png'
        self.run_save = True

        self.chain_save = [0]
        self.run_log = ''
        self.save_loc = ''
        self.read_loc = None


        # stats
        self.bic_limit = 5
        self.post_max = -np.inf
        self.chi2 = np.inf
        self.chi2_red = np.inf
        self.BIC = np.inf
        self.AIC = np.inf


        self.conds = []

        self.eccentricity_limits = [[0, 1]]
        self.eccentricity_prargs = [[0, 0.1]]

        self.jitter_limits = [[0, 1]]
        self.jitter_prargs = [[5, 5]]

        print('Simulation Successfully Initialized \n')

        self.switch1 = True
        self.switch2 = True

        self.switch_RV = False
        self.switch_F = False
        self.switch_cherry = False
        self.switch_staract = False
        self.switch_evidence = False
        self.HILL = False

        self.switch_constrain = False
        self.developer_mode = False

        self.switch_sv_residuals = True

        self.bayes_factor = np.log(10000)  # switch_cherry
        self.minimum_samples = 1000  # switch_cherry


        self.transit = False  # switch_F
        self.jplanets__ = 0  # switch_F
        self.fnins__ = 0  # switch_F

        if self.transit:
            import batman
            self.ld_dctn = {'uniform': 0,
                        'linear': 1,
                        'quadratic': 2,
                        'square-root': 2,
                        'logarithmic': 2,
                        'exponential': 2,
                        'power2': 2,
                        'nonlinear': 2
                        }  # dictionary with limb darkening dimentionality

        self.gp = False
        if self.gp:
            import celerite
            from celerite import terms as cterms
            self.kernel_dctn = {'Constant': 1. ** 2,
                 'RealTerm':cterms.RealTerm(log_a=2., log_c=2.),
                 'ComplexTerm':cterms.ComplexTerm(log_a=2., log_b=2., log_c=2., log_d=2.),
                 'SHOTerm':cterms.SHOTerm(log_S0=2., log_Q=2., log_omega0=2.),
                 'Matern32Term':cterms.Matern32Term(log_sigma=2., log_rho=2.),
                 'JitterTerm':cterms.JitterTerm(log_sigma=1e-8)}

        pass


    def printsv(self, msg):
        print(msg)
        self.run_log += msg
        pass


    def _mk_keplerian__(self, param=0):
        self.kplanets__ += 1
        # auto
        sig_limiter = np.std(self.datay__)

        amp_limiter = sig_limiter * np.sqrt(3)
        per_limiter = max(self.datax__) - min(self.datax__)
        angle_limits = [0, 2*np.pi]

        try:
            ecc_limits = self.eccentricity_limits[self.kplanets__-1]
            ecc_prargs = self.eccentricity_prargs[self.kplanets__-1]
        except:
            ecc_limits = [0, 1]
            ecc_prargs = [0, 0.3]


        uni = 'Uniform'
        norm = 'Normal'
        ### Vanilla
        if param == 0 or param == 'vanilla':

            limits = [[0.1, per_limiter], [0, amp_limiter], angle_limits,
                       ecc_limits, angle_limits]

            #uni = my_stats.Uniform
            Period = Parameter(-np.inf, uni, limits[0], name='Period %s' %self.kplanets__)
            Amplitude = Parameter(-np.inf, uni, limits[1], name='Amplitude %s' %self.kplanets__)
            Phase = Parameter(-np.inf, uni, limits[2], name='Phase %s' %self.kplanets__)
            Eccentricity = Parameter(-np.inf, norm, limits[3], name='Eccentricity %s' %self.kplanets__, prargs=ecc_prargs)
            Longitude = Parameter(-np.inf, uni, limits[4], name='Longitude %s' %self.kplanets__)

            kepler_params = [Period, Amplitude, Phase, Eccentricity, Longitude]
            b_mod = mr._model_keplerian

            Kepler = Parameter_Block(kepler_params, block_model = b_mod, block_name='Kepler %s' %self.kplanets__, block_type='Keplerian')
            Kepler.hou_priors = []


        ### Hou
        elif param == 1 or param == 'hou':
            sqrta, sqrte = amp_limiter, 1  #(sqrt 0.5 ~ 0.707)
            sqrta, sqrte = sqrta ** 0.5, sqrte ** 2
            a_lims, e_lims = [-sqrta, sqrta], [-sqrte, sqrte]


            limits = [np.log([0.1, per_limiter*3]), a_lims, a_lims, e_lims, e_lims]

            Period = Parameter(-np.inf, uni, limits[0], name='lPeriod %s' %self.kplanets__)
            Amplitude = Parameter(-np.inf, uni, limits[1], name='Amp_sin %s' %self.kplanets__)
            Phase = Parameter(-np.inf, uni, limits[2], name='Amp_cos %s' %self.kplanets__)
            Eccentricity = Parameter(-np.inf, uni, limits[3], name='Ecc_sin %s' %self.kplanets__)
            Longitude = Parameter(-np.inf, uni, limits[4], name='Ecc_cos %s' %self.kplanets__)

            kepler_params = [Period, Amplitude, Phase, Eccentricity, Longitude]
            b_mod = mr._model_keplerian_hou
            Kepler = Parameter_Block(kepler_params, block_model = b_mod, block_name='Kepler %s' %self.kplanets__, block_type='Keplerian')

            Kepler.hou_priors = [['Period', uni, [1e-6, per_limiter*3],[]],
                                 ['Amplitude', uni, [0.1, amp_limiter], []],
                                 ['Phase', uni, [0, 2*np.pi], []],
                                 ['Eccentricity', norm, ecc_limits, ecc_prargs],
                                 ['Longitude', uni, [0, 2*np.pi], []]]

        ### t0
        elif param == 2 or param == 't0':
            t0_limiter = [min(self.datax__) - per_limiter, max(self.datax__) + per_limiter]
            limits = [[0.1, per_limiter], [0, amp_limiter], t0_limiter, ecc_limits, angle_limits]

            Period = Parameter(-np.inf, 'Uniform', limits[0], name='Period %s' %self.kplanets__)
            Amplitude = Parameter(-np.inf, 'Uniform', limits[1], name='Amplitude %s' %self.kplanets__)
            Phase = Parameter(-np.inf, 'Uniform', limits[2], name='T_0 %s' %self.kplanets__)
            Eccentricity = Parameter(-np.inf, norm, limits[3], name='Eccentricity %s' %self.kplanets__, prargs=ecc_prargs)
            Longitude = Parameter(-np.inf, 'Uniform', limits[4], name='Longitude %s' %self.kplanets__)

            kepler_params = [Period, Amplitude, Phase, Eccentricity, Longitude]
            b_mod = mr._model_keplerian_tp

            Kepler = Parameter_Block(kepler_params, block_model = b_mod, block_name='Kepler %s' %self.kplanets__, block_type='Keplerian')
            Kepler.hou_priors = []

        ### hou t0
        elif param == 3 or param == 'hout0':
            t0_limiter = [min(self.datax__) - per_limiter, max(self.datax__) + per_limiter]
            sqrte = 1  #(sqrt 0.5 ~ 0.707)
            sqrte = sqrte ** 2
            e_lims = [-sqrte, sqrte]

            limits = [[0.1, per_limiter], [0, amp_limiter], t0_limiter, e_lims, e_lims]

            Period = Parameter(-np.inf, uni, limits[0], name='Period %s' %self.kplanets__)
            Amplitude = Parameter(-np.inf, uni, limits[1], name='Amplitude %s' %self.kplanets__)
            Phase = Parameter(-np.inf, uni, limits[2], name='T_0 %s' %self.kplanets__)
            Eccentricity = Parameter(-np.inf, uni, limits[3], name='Ecc_sin %s' %self.kplanets__)
            Longitude = Parameter(-np.inf, uni, limits[4], name='Ecc_cos %s' %self.kplanets__)

            kepler_params = [Period, Amplitude, Phase, Eccentricity, Longitude]
            b_mod = mr._model_keplerian_houtp

            Kepler = Parameter_Block(kepler_params, block_model = b_mod, block_name='Kepler %s' %self.kplanets__, block_type='Keplerian')
            Kepler.hou_priors = [['Eccentricity', norm, ecc_limits, ecc_prargs],
                                 ['Longitude', uni, [0, 2*np.pi], []]]

            pass


        else:
            print('param=%s is not a valid parametrization' % param)


        if self.engine__.__name__ == 'pymc3':
            if param == 0 or param == 'vanilla':
                Kepler = Parameter_Block(kepler_params, block_model = mr._model_keplerian_pymc3, block_name='Kepler %s' %self.kplanets__, block_type='Keplerian')

            else:
                print('param=%s is not a valid parametrization yet pymc3' % param)

        Kepler.parametrization = param
        Kepler.signal_number = self.kplanets__

        self._refresh__()
        self.blocks__.insert((self.kplanets__ - 1), Kepler)
        self.printsv('%s block added \n' %Kepler.type_)


    def _mk_keplerian_scale__(self, kplan, param=0):

        # auto
        amp_limiter = np.amax(abs(self.datay__))
        per_limiter = max(self.datax__) - min(self.datax__)
        angle_limits = [0, 2*np.pi]

        kepler_params = []
        if param == 0 or param == 'vanilla':
            for k in range(kplan):
                self.kplanets__ += 1
                limits = [[1e-8, per_limiter], [0, amp_limiter * 2], angle_limits, [0, 1], angle_limits]

                #uni = my_stats.Uniform
                uni = 'Uniform'
                Period = Parameter(-np.inf, uni, limits[0], name='Period %s' %self.kplanets__)
                Amplitude = Parameter(-np.inf, uni, limits[1], name='Amplitude %s' %self.kplanets__)
                Phase = Parameter(-np.inf, uni, limits[2], name='Phase %s' %self.kplanets__)
                Eccentricity = Parameter(-np.inf, uni, limits[3], name='Eccentricity %s' %self.kplanets__)
                Longitude = Parameter(-np.inf, uni, limits[4], name='Longitude %s' %self.kplanets__)

                Params = [Period, Amplitude, Phase, Eccentricity, Longitude]
                for p in Params:
                    kepler_params.append(p)

            for i in range(self.nins__):
                Scale_Coef = Parameter(1, 'Uniform', [-3, 3], name='Instrument Scale %s' % str(i+1))
                kepler_params.append(Scale_Coef)

            b_mod = mr._model_keplerian_scale

            Kepler = Parameter_Block(kepler_params, block_model = b_mod, block_name='Kepler Scale %s' %self.kplanets__, block_type='Keplerian')

        Kepler.extra_args = [self.nins__, self.kplanets__]
        self._refresh__()
        self.blocks__.append(Kepler)

        print('%s block added \n' %Kepler.type_)

        pass


    def _mk_keplerian_joined__(self, param=0):


        pass


    def _mk_transit__(self, param=0, ldname=None):
        import batman
        self.jplanets__ += 1

        angle_limits = [0, 360]

        # Create Parameters
        uni = 'Uniform'

        Period = Parameter(10, uni, [0.1, max(self.fdatax__)], name='Period %s' %self.jplanets__)
        SMA = Parameter(2, uni, [1e-6, 1e3], name='SemiMajor Axis %s' %self.jplanets__)
        t0 = Parameter(245123., uni, [min(self.fdatax__), max(self.fdatax__)], name='T_0 %s' %self.jplanets__)
        Eccentricity = Parameter(0.1, uni, [0, 1], name='Eccentricity %s' %self.jplanets__)
        Longitude = Parameter(50, uni, [0, 360], name='Longitude %s' %self.jplanets__)
        Radius = Parameter(0.1, uni, [1e-6, 1], name='Planet Radius %s' %self.jplanets__)
        Inclination = Parameter(90, uni, [60, 120], name='Inclination %s' %self.jplanets__)

        transit_params = [Period, SMA, t0, Eccentricity, Longitude, Radius, Inclination]
        if ldname:
            for i in range(self.ld_dctn[ldname]):
                Coef = Parameter(0.1, uni, [-1, 1], name='LD Coefficient %i' % (i+1))
                transit_params.append(Coef)

        #transit_params =
        Transit = Parameter_Block(transit_params, block_model=mr._model_keplerian_transit,
                                  block_name='Transit %s' % str(self.jplanets__),
                                  block_type='Transit')

        Transit.extra_args = [ldname]
        Transit.F = True

        if self.jplanets__ > 1:
            Transit.extra_args.append(True)
        else:
            Transit.extra_args.append(False)
        #Transit.extra_args = []
        #raise Exception('debug')
        #if ldname:
        #    Transit.extra_args = [ldname]

        self._refresh__()
        self.blocks__.insert((self.jplanets__ - 1), Transit)
        #self.blocks__.append(Transit)

        print('%s block added \n' %Transit.type_)
        pass


    def _mk_sinusoid__(self):
        self.kplanets__ += 1

        # Sety automatic Limits
        amp_limiter = np.amax(abs(self.datay__))
        per_limiter = max(self.datax__) - min(self.datax__)
        angle_limits = [0, 2*np.pi]

        limits = [[0.1, per_limiter], [0, amp_limiter * 2], angle_limits]
        # Create Parameters
        Period = Parameter(10, 'Uniform', limits[0], name='Period %s' %self.kplanets__)
        Amplitude = Parameter(50, 'Uniform', limits[1], name='Amplitude %s' %self.kplanets__)
        Phase = Parameter(np.pi/2, 'Uniform', limits[2], name='Phase %s' %self.kplanets__)
        # Create Block
        kepler_params = [Period, Amplitude, Phase]
        Sinusoid = Parameter_Block(kepler_params, block_model=mr._model_sinusoid, block_name='Sinusoid %s' %self.kplanets__, block_type='Sinusoidal')

        self.blocks__.insert(0, Sinusoid)
        print('%s block added \n' %Sinusoid.type_)


    def _mk_noise_instrumental__(self, moav=0):
        # n = -1 no jitter


        for i in range(self.nins__):
            jit_limiter = np.amax(abs(self.datay__[self.dataflag__==i]))

            try:
                jitter_limits = self.jitter_limits[self.kplanets__-1]
                jitter_prargs = self.jitter_prargs[self.kplanets__-1]
            except:
                jitter_limits = [0, jit_limiter]
                jitter_prargs = [5, 5]


            new = []
            Offset = Parameter(10, 'Uniform', [-jit_limiter, jit_limiter], name='Offset %s' % str(i+1))
            new.append(Offset)

            Jitter = Parameter(5, 'Normal', jitter_limits, name='Jitter %s' % str(i+1), prargs=jitter_prargs)
            new.append(Jitter)

            if moav > 0:
                for j in np.arange(moav)+1:
                    MACoef = Parameter(0, 'Uniform', [-1, 1], name='MACoefficient %s Order %s' % (str(i+1), j))
                    MATime = Parameter(0, 'Uniform', [1e-8, 5], name='MATimescale %s Order %s' % (str(i+1), j))
                    new.append(MACoef)
                    new.append(MATime)


            Instrument = Parameter_Block(new, block_model=mr._model_moav,
                                         block_name='Instrument %s' % str(i+1),
                                         block_type='Instrumental')

            Instrument.extra_args = [i, moav]
            Instrument.instrument_name = self.ins_label__[i]

            if moav > 0:
                Instrument.MOAV_ = True
                Instrument.extra_args.append(np.zeros_like(self.datax__))

            self._refresh__()
            self.blocks__.append(Instrument)

            print('%s block added \n' %Instrument.type_)
        pass


    def _mk_noise_instrumental_sa__(self, n=0):
        # n= -1 no jitter
        for i in range(self.nins__):
            jit_limiter = np.amax(abs(self.datay__[self.dataflag__==i]))
            jit_mean = np.mean(self.datayerr__[self.dataflag__==i])


            new = []
            Offset = Parameter(10, 'Uniform', [-jit_limiter, jit_limiter], name='Offset %s' % str(i+1))
            new.append(Offset)

            Jitter = Parameter(5, 'Normal', [0, jit_limiter], name='Jitter %s' % str(i+1), prargs=[jit_mean, 3])
            new.append(Jitter)

            if n > 0:
                for j in np.arange(n)+1:
                    MACoef = Parameter(0, 'Uniform', [-1, 1], name='MACoefficient %s Order %s' % (str(i+1), j))
                    MATime = Parameter(0, 'Uniform', [1e-8, 5], name='MATimescale %s Order %s' % (str(i+1), j))
                    new.append(MACoef)
                    new.append(MATime)

            if self.cornums__[i]:
                for sa in range(self.cornums__[i]):
                    Staract = Parameter(1, 'Uniform', [-1, 1], name='Star Activity %s n%s' % (i+1, str(sa+1)))
                    new.append(Staract)

            Instrument = Parameter_Block(new, block_model=mr._model_moav_sa, block_name='Instrument %s' % str(i+1), block_type='Instrumental')

            Instrument.extra_args = [i, n, self.staract__[i], self.cornums__[i]]
            Instrument.instrument_name = self.ins_label__[i]

            if n > 0:
                Instrument.MOAV_ = True
                Instrument.extra_args.append(np.zeros_like(self.datax__))

            self._refresh__()
            self.blocks__.append(Instrument)

            print('%s block added \n' %Instrument.type_)
        pass


    def _mk_gaussian_process(self, kername=None):
        if kername:
            for i in range(self.kernel_dctn[kername]):
                Coef = Parameter(0.1, uni, [-1, 1], name='LD Coefficient %i' % (i+1))
                transit_params.append(Coef)

        pass


    def _mk_scale_instrumental__(self, n=0):
        for i in range(self.nins__):
            Scale_Coef = Parameter(1, 'Uniform', [-3, 3], name='Instrument Scale %s' % str(i+1))

            Scale = Parameter_Block([Scale_Coef], block_model=mr._model_scale, block_name='Instrument Scale %s' % str(i+1), block_type='InstrumentalScale')

            Scale.extra_args = [i]
            self._refresh__()
            self.blocks__.append(Scale)

            print('%s block added \n' %Scale.type_)
        pass


    def _mk_acceleration__(self, n=1):
        if n > 0:
            new = []
            for i in range(n):
                if i == 0:
                    Acc = Parameter(1, 'Uniform', [-0.1, 0.1], name='Acceleration')
                else:
                    Acc = Parameter(1, 'Uniform', [-0.1, 0.1], name='Acceleration Order %s' % str(i+1))
                new.append(Acc)
            Accel = Parameter_Block(new, block_model=mr._model_acc, block_name='Acceleration', block_type='General')
            self.blocks__.append(Accel)

            self.printsv('%s block added \n' %Accel.type_)
        pass


    def _mk_staract__(self):
        self.switch_staract = True
        '''
        for i in range(self.nins__):
            if self.cornums__[i]:
                new = []
                for sa in range(self.cornums__[i]):
                    Staract = Parameter(1, 'Uniform', [-1, 1], name='Star Activity %s' % str(sa+1))
                    new.append(Staract)

                SA_ins = Parameter_Block(new, block_model=mr._model_staract, block_name='Star Activity %s' % str(i+1), block_type='Instrumental Staract')

                SA_ins.extra_args = [i, self.staract__[i]]
                self._refresh__()
                self.blocks__.append(SA_ins)

                print('%s block added \n' % SA_ins.type_)
        '''


    def _calc_evidence__(self):
        if self.engine == 'emcee':
            if self.setup[0] > 5:
                self.switch_evidence = True
                print('\nEvidence calculations activated')
            else:
                print('\n Temperature number too low for good evidence estimates')

        if self.engine == 'dynesty':
            self.switch_evidence = True
            print('\nEvidence calculations activated')

        if self.engine == 'pymc3':
            self.switch_evidence = True
            print('\nEvidence calculations activated')

        pass


    def __getitem__(self, n):
        return self.blocks__[n]


    def __repr__(self):
        return str([self.blocks__[i].name_ for i in range(len(self.blocks__))])


    def __len__(self):
        return len(self.blocks__)


    def _refresh__(self):
        for b in self:
            b._refresh_()

        self.ndim__ = int(np.sum(self._list__('ndim_')))

        fixed_values = []
        for b in self:
            fixed_values.append(b._list_('fixed'))

        self.fixed_all = flatten(fixed_values)

        self.A_ = []
        self.C_ = []
        for i in range(len(self.fixed_all)):
            if self.fixed_all != None:
                self.A_.append(i)
            else:
                self.C_.append(i)
        pass


    def _set_engine__(self, eng):
        if eng == 'emcee':
            import emcee
            self.engine__ = emcee
        elif eng == 'dynesty':
            import dynesty
            self.engine__ = dynesty
        elif eng == 'pymc3':
            import pymc3 as pm
            self.engine__ = pm
        else:
            raise Exception('Failed to set engine properly. Try a string!')


    def _apply_conditions__(self):
        import itertools
        applied = []
        for b in self:
            for p in b:
                for c in self.conds:
                    if p.name == c[0]:
                        setattr(p, c[1], c[2])
                        if c not in applied:
                            applied.append(c)
                            print('\nFollowing condition has been applied: ', c)

        applied.sort()
        applied = list(applied for applied,_ in itertools.groupby(applied))

        for ap in applied[::-1]:
            self.conds.remove(ap)


    def _run__(self, setup, *args):

        self._apply_conditions__()

        self._refresh__()  # updates self.ndim__
        self.setup = setup

        if self.switch1:
            self.printsv('Current Engine is '+self.engine__.__name__+' '+self.engine__.__version__)
            self.switch1 = False

        if self.switch_RV:
            self.my_model = Model(self.data__, self.blocks__)
            #raise Exception('asdasd')

        elif self.switch_F:
            for b in self:
                if b.type_ == 'Transit':
                    b._init_batman_(self.fdatax__)

            self.my_model = Model(self.fdata__, self.blocks__)  # fblocks__ ?

        if self.kplanets__ > 1 and self.HILL:
            self.my_model.HILL = True
            self.my_model.starmass = self.starmass

        if True:
            if self.engine__.__name__ == 'emcee':
                if not self.developer_mode:
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                ntemps, nwalkers, nsteps = setup

                self.pos0__ = self._mk_pos0_exp__(setup)


                self.sampler = self.engine__.PTSampler(ntemps, nwalkers, self.ndim__, _logl__, _logp__,
                                                logpargs=[self.my_model],
                                                loglargs=[self.my_model],
                                                threads=self.cores__, betas=self.betas)


                print('\nRunning Burn-In')
                pbar = tqdm(total=nsteps//2)
                for p, lnprob, lnlike in self.sampler.sample(self.pos0__, iterations = nsteps//2):
                    pbar.update(1)
                    pass
                pbar.close()


                p0, lnprob0, lnlike0 = p, lnprob, lnlike
                self.sampler.reset()

                print('\nRunning Chain')
                pbar = tqdm(total=nsteps)  # loading bar
                for p, lnprob, lnlike in self.sampler.sample(p0, lnprob0=lnprob0,
                                                             lnlike0=lnlike0,
                                                             iterations=nsteps):
                    pbar.update(1)  # loading bar
                    pass
                pbar.close()

                self.printsv('\nRun Acceptance Fraction')
                tab_header = ['Chain', 'Mean', 'Minimum', 'Maximum']
                tab_all = []
                for t in range(ntemps):
                    maf = self.sampler.acceptance_fraction[t]
                    tab_all.append([t, np.mean(maf), np.amin(maf), np.amax(maf)])

                self.printsv('\n\n')
                self.printsv(tabulate(tab_all, headers=tab_header))
                self.printsv('\n--------------------------------------------------- \n')
                pass

            if self.engine__.__name__ == 'dynesty':
                nlive0 = setup[0]

                ptformargs0 = []
                for b in self:
                    for p in b:
                        if p.fixed == None:
                            if p.ptformargs == None:
                                l, h = p.limits
                                p.ptformargs = [(h-l)/2., (h+l)/2.]
                                if b.parametrization == 1 or b.parametrization == 'hou':
                                    if p.name[:3] == 'Ecc' and p.ptformargs[0] > 0.707:
                                        p.ptformargs[0] = 0.707
                                if b.parametrization == 3 or b.parametrization == 'hout0':
                                    if p.name[:3] == 'Ecc' and p.ptformargs[0] > 0.707:
                                        p.ptformargs[0] = 0.707
                            else:
                                s, c = p.ptformargs
                                p.limits = [c-s, c+s]
                            ptformargs0.append(p.ptformargs)


                #self.my_model = Model(DATA, self.blocks__)
                self.sampler = self.engine__.NestedSampler(_logl__, _ptform__, self.ndim__,
                                                        logl_args=[self.my_model],
                                                        ptform_args=[ptformargs0],
                                                        nlive=nlive0)
                self.sampler.run_nested()


                pass

            if False:
                ### should accept pymc3 kwargs for the param kwargs, as lower and upper
                ### different dist use different words :(

                #self.my_model = Model(DATA, self.blocks__)
                import pymc3_ext as pmx
                dr, tu, ch = setup
                with self.engine__.Model() as model_pymc3:

                    # Priors for unknown model parameters
                    theta_pm = []
                    #sma, mm = [], []

                    for b in self.blocks__:
                        for p in b:
                            if p.fixed == None:
                                if p.prior != 'Uniform':
                                    p.prior = 'Uniform'
                                if p.prior == 'Uniform':
                                    theta_pm.append(getattr(self.engine__, p.prior)(p.name, lower=p.limits[0], upper=p.limits[1]))
                                #if p.prior == 'Normal':
                                #    p.prior == 'TruncatedNormal'
                                #    theta_pm.append(getattr(self.engine__, p.prior)(p.name, mu=p.prargs[0], sigma=p.prargs[1], lower=p.limits[0], upper=p.limits[1]))



                    # Expected value of outcome
                    mu, err2 = self.my_model.evaluate_model(theta_pm)
                    ferr2 = self.datayerr__**2 + err2

                    if self.kplanets__:
                        rvmodel = self.engine__.Deterministic('rvmodel', mu)
                        for k in range(self.kplanets__):
                        #    e, w = delinearize_pymc3(theta_pm[self[k].ndim_*k+3], theta_pm[self[k].ndim_*k+4])
                            per = theta_pm[self[k].ndim_*k+0]
                            amp = theta_pm[self[k].ndim_*k+1]
                            ecc = theta_pm[self[k].ndim_*k+3]

                            sma, mm = cps(per, amp, ecc, self.starmass)

                            self.engine__.Deterministic('SMA %i' % (k+1), sma)
                            self.engine__.Deterministic('MinMass %i' % (k+1), mm)

                    #rverr = self.engine__.Deterministic('rverr', self.engine__.math.sqrt(ferr2))
                    rverr = self.datayerr__

                    #pm.Normal("obs", mu=mu, sd=pm.math.exp(logs), observed=y)
                    # Likelihood (sampling distribution) of observations
                    Y_obs = self.engine__.Normal("Y_obs", mu=mu, sd=rverr, observed=self.datay__)
                        #pm.Normal("obs", mu=rvmodel, sd=rv_err, observed=rv)
                    #self.sampler = pm.sample(draws=dr, tune=tu, chains=ch, cores=self.cores__, return_inferencedata=True)

                with model_pymc3:
                    self.map_params = pmx.optimize()

                with model_pymc3:
                    self.sampler = pmx.sample(draws=dr, tune=tu, start=self.map_params, chains=ch, cores=self.cores__, target_accept=0.95, return_inferencedata=True)

            if self.engine__.__name__ == 'pymc3':
                import aesara_theano_fallback.tensor as tt
                import exoplanet as xo
                import pymc3_ext as pmx
                dr, tu, ch = setup
                t, rv, rv_err = self.datax__, self.datay__, self.datayerr__
                t = t - np.mean(t)

                amp_limiter = np.amax(abs(self.datay__))
                per_limiter = max(self.datax__) - min(self.datax__)
                angle_limits = [0, 2*np.pi]

                with self.engine__.Model() as model_pymc3:
                    theta_pm = []
                    # Parameters
                    logK = self.engine__.Uniform(
                        "logK",
                        lower=0,
                        upper=np.log(amp_limiter*2),
                        testval=np.log(0.5 * (np.max(rv) - np.min(rv))),
                    )
                    lit_per = 4.230785
                    logP = self.engine__.Uniform(
                        "logP", lower=0, upper=np.log(per_limiter), testval=np.log(4)
                    )
                    phi = self.engine__.Uniform("phi", lower=0, upper=2 * np.pi, testval=0.1)

                    # Parameterize the eccentricity using:
                    #  h = sqrt(e) * sin(w)
                    #  k = sqrt(e) * cos(w)
                    hk = pmx.UnitDisk("hk", testval=np.array([0.01, 0.01]))
                    e = self.engine__.Deterministic("e", hk[0] ** 2 + hk[1] ** 2)
                    w = self.engine__.Deterministic("w", tt.arctan2(hk[1], hk[0]))

                    rv0 = self.engine__.Normal("rv0", mu=0.0, sd=10.0, testval=0.0)
                    rvtrend = self.engine__.Normal("rvtrend", mu=0.0, sd=10.0, testval=0.0)

                    # Deterministic transformations
                    n = 2 * np.pi * tt.exp(-logP)
                    P = self.engine__.Deterministic("P", tt.exp(logP))
                    K = self.engine__.Deterministic("K", tt.exp(logK))
                    cosw = tt.cos(w)
                    sinw = tt.sin(w)
                    t0 = (phi + w) / n

                    # The RV model
                    bkg = self.engine__.Deterministic("bkg", rv0 + rvtrend * t / 365.25)
                    M = n * t - (phi + w)

                    # This is the line that uses the custom Kepler solver
                    f = xo.orbits.get_true_anomaly(M, e + tt.zeros_like(M))
                    rvmodel = self.engine__.Deterministic(
                        "rvmodel", bkg + K * (cosw * (tt.cos(f) + e) - sinw * tt.sin(f))
                    )

                    # Condition on the observations
                    self.engine__.Normal("obs", mu=rvmodel, sd=rv_err, observed=rv)

                    # Compute the phased RV signal
                    phase = np.linspace(0, 1, 500)
                    M_pred = 2 * np.pi * phase - (phi + w)
                    f_pred = xo.orbits.get_true_anomaly(M_pred, e + tt.zeros_like(M_pred))
                    rvphase = self.engine__.Deterministic("rvphase", K * (cosw * (tt.cos(f_pred) + e) - sinw * tt.sin(f_pred)))


                with model_pymc3:
                    self.map_params = pmx.optimize()

                with model_pymc3:
                    self.sampler = pmx.sample(draws=dr, tune=tu, start=self.map_params, chains=ch, cores=self.cores__, target_accept=0.95, return_inferencedata=True)

                import arviz as az
                trace = self.sampler
                map_params = self.map_params

                print(az.summary(trace, var_names=["logK", "logP", "phi", "e", "w", "rv0", "rvtrend"]))

                fig, axes = pl.subplots(2, 1, figsize=(8, 8))

                period = map_params["P"]

                ax = axes[0]
                ax.errorbar(t, rv, yerr=rv_err, fmt=".k")
                ax.set_ylabel("radial velocity [m/s]")
                ax.set_xlabel("time [days]")

                ax = axes[1]
                ax.errorbar(t % period, rv - map_params["bkg"], yerr=rv_err, fmt=".k")
                ax.set_ylabel("radial velocity [m/s]")
                ax.set_xlabel("phase [days]")

                bkg = trace.posterior["bkg"].values
                rvphase = trace.posterior["rvphase"].values

                for ind in np.random.randint(np.prod(bkg.shape[:2]), size=25):
                    i = np.unravel_index(ind, bkg.shape[:2])
                    axes[0].plot(t, bkg[i], color="C0", lw=1, alpha=0.3)
                    axes[1].plot(phase * period, rvphase[i], color="C1", lw=1, alpha=0.3)

                axes[0].set_ylim(-110, 110)
                axes[1].set_ylim(-110, 110)

                pl.tight_layout()
                pl.show()
        pass


    def _run_auto__(self, setup, to_k, from_k=0, param=0, moav=0, acc=0, ldname=None):
        oldlogpost = -np.inf
        oldchi2 = np.inf
        oldbic = np.inf
        oldaic = np.inf

        if self.engine__.__name__ == 'emcee' or self.engine__.__name__ == 'dynesty' or self.engine__.__name__ == 'pymc3':

            if self.switch_RV:
                if self.switch_staract:
                    self._mk_noise_instrumental_sa__(n=moav)
                    pass
                else:
                    self._mk_noise_instrumental__(moav=moav)

            self._mk_acceleration__(n=acc)
            while from_k <= to_k:

                oldlogpost = self.post_max
                oldchi2 = self.chi2_red
                oldbic = self.BIC
                oldaic = self.AIC

                # run
                self._run__(setup)
                self._post_process__(setup)

                ### plots
                if self.switch_RV:
                    self.plotmodel()
                elif self.switch_F:
                    if self.jplanets__ > 0:
                        self.plotfmodel()

                self.plottrace()
                self.plotpost()
                self.plothistograms()
                try:
                    if self.chain_save is not None:
                        self.save_chain(self.chain_save)
                        self.save_posteriors(self.chain_save)
                except:
                    print('Chains not saved????')

                from_k += 1  # use kplanets__ instead

                if oldbic - self.BIC <  self.bic_limit:
                    print('\nBIC condition not met')
                    break

                if from_k > to_k:
                    break

                else:
                    print('\nBIC condition met!!')
                    print('past BIC - present BIC < 5')
                    print(str(oldbic)+'-'+str(self.BIC)+' < '+str(self.bic_limit))
                    self.printsv('\n######################################')
                    self.printsv('\n#   Proceeding with the next run !   #\n')
                    self.printsv('######################################\n')

                if self.switch_constrain == True:
                    for b in self:
                        if b.type_ == 'Keplerian':
                            for p in b:
                                if p.fixed == None:
                                    pval = p.value
                                    psig = p.sigma

                                    if False:
                                        limf, limc = pval - psig, pval + psig
                                        if psig / abs(pval) < 1e-6:
                                            self.conds.append([p.name, 'fixed', pval])
                                        else:
                                            if (limf > p.limits[0] and limc < p.limits[1]):
                                                self.conds.append([p.name, 'limits', [limf, limc]])
                                            elif limf > p.limits[0]:
                                                self.conds.append([p.name, 'limits', [limf, p.limits[1]]])
                                            elif limc < p.limits[1]:
                                                self.conds.append([p.name, 'limits', [p.limits[0], limc]])
                                    if True:
                                        if p.name.split(' ')[0] == 'Ecc_sin' or p.name.split(' ')[0] == 'Ecc_cos':
                                            if pval > 0:
                                                limf, limc = - psig, pval + psig
                                                if (limf > p.limits[0] and limc < p.limits[1]):
                                                    self.conds.append([p.name, 'limits', [limf, limc]])
                                                elif limf > p.limits[0]:
                                                    self.conds.append([p.name, 'limits', [limf, p.limits[1]]])
                                                elif limc < p.limits[1]:
                                                    self.conds.append([p.name, 'limits', [p.limits[0], limc]])
                                            else:
                                                limf, limc = pval - psig, psig
                                                if (limf > p.limits[0] and limc < p.limits[1]):
                                                    self.conds.append([p.name, 'limits', [limf, limc]])
                                                elif limf > p.limits[0]:
                                                    self.conds.append([p.name, 'limits', [limf, p.limits[1]]])
                                                elif limc < p.limits[1]:
                                                    self.conds.append([p.name, 'limits', [p.limits[0], limc]])

                                        else:
                                            limf, limc = pval - psig, pval + psig
                                            if psig / abs(pval) < 1e-6:
                                                self.conds.append([p.name, 'fixed', pval])
                                            else:
                                                if (limf > p.limits[0] and limc < p.limits[1]):
                                                    self.conds.append([p.name, 'limits', [limf, limc]])
                                                elif limf > p.limits[0]:
                                                    self.conds.append([p.name, 'limits', [limf, p.limits[1]]])
                                                elif limc < p.limits[1]:
                                                    self.conds.append([p.name, 'limits', [p.limits[0], limc]])
                                                pass
                        # should be just for jitter! >:(

                        if b.type_ == 'Instrumental':
                            for p in b:
                                if p.fixed == None and p.name.split(' ')[0] == 'Jitter':
                                    pval = p.value
                                    psig = p.sigma

                                    limc = pval + psig
                                    if limc < p.limits[1]:
                                        self.conds.append([p.name, 'limits', [p.limits[0], limc]])

                if self.switch_RV:
                    self._mk_keplerian__(param=param)
                elif self.switch_F:
                    self._mk_transit__(param=param, ldname=ldname)
                self.saveplace = ensure_dir(self.starname, loc=self.save_loc)



            print('\n\nFinished the run ! !')
            pass

        pass


    def _post_process__(self, setup=[]):
        if self.engine__.__name__ == 'emcee':
            ntemps, nwalkers, nsteps = setup

            raw_chain = self.sampler.flatchain
            raw_posts = np.array(
                            [self.sampler.lnprobability[i].reshape(-1) for i in range(ntemps)])

            self.chains = raw_chain
            self.posteriors = raw_posts

            if self.betas is None:
                self.betas = self.sampler.betas

            setup_info = 'Temperatures, Walkers, Steps   : '
            size_info = [len(self.chains[t]) for t in range(ntemps)]

            if self.switch_cherry:
                self.printsv('\nSelecting the sweetest cherries, with a bayes factor of %.3f' % self.bayes_factor)

                self.cherry_loc = [max(raw_posts[t]) - raw_posts[t] <= self.bayes_factor / self.betas[t] for t in range(ntemps)]
                self.chains = np.array(
                                [raw_chain[t][self.cherry_loc[t]] for t in range(ntemps)],
                                dtype=object)
                self.posteriors = np.array(
                                [raw_posts[t][self.cherry_loc[t]] for t in range(ntemps)],
                                dtype=object)

                size_info = [len(self.chains[t]) for t in range(ntemps)]

                while size_info[0] < self.minimum_samples:
                    self.bayes_factor *= 1.1
                    self.printsv('Sample size for cold chain insufficient %i. Ramping bayes_factor to %.3f' % (size_info[0], self.bayes_factor))
                    self.cherry_loc = [max(raw_posts[t]) - raw_posts[t] <= self.bayes_factor * self.betas[t] for t in range(ntemps)]
                    self.chains = np.array(
                                    [raw_chain[t][self.cherry_loc[t]] for t in range(ntemps)],
                                    dtype=object)
                    self.posteriors = np.array(
                                    [raw_posts[t][self.cherry_loc[t]] for t in range(ntemps)],
                                    dtype=object)

                    size_info = [len(self.chains[t]) for t in range(ntemps)]

            if self.switch_evidence:
                self.evidence = self.sampler.thermodynamic_integration_log_evidence()


            self.post_max = np.amax(self.posteriors[0])
            self.ajuste = self.chains[0][np.argmax(self.posteriors[0])]
            self.sigmas = np.std(self.chains[0], axis=0)


        if self.engine__.__name__ == 'dynesty':
            results = self.sampler.results

            setup_info = '\nLive Points                       : '
            size_info = results['niter']

            self.chains = np.array([results['samples']])
            self.posteriors = [results['logl']]
            self.evidence = (results['logz'][-1], results['logzerr'][-1])

            #self.printsv('\n\n------------- Not yet built ------------\n\n')
            #self.printsv('Trust the plots')
            ### aqui falta desarrollar los real results

            self.post_max = np.amax(self.posteriors[0])
            self.ajuste = self.chains[0][-1]
            self.sigmas = np.std(self.chains[0], axis=0)

            self.printsv('\n\n------------ Dynesty Summary -----------\n\n')
            self.printsv(str(results.summary()))


            pass


        if self.engine__.__name__ == 'pymc3':
            import arviz as az
            setup_info = '\nDraw, Tune, Chains                : '
            size_info = setup

            self.varnames = flatten([b._list_('name')[b.C_] for b in self])
            self.info = az.summary(self.sampler, var_names=self.varnames)
            self.ajuste = self.info['mean'].values
            #self.chains =

            #self.post_max = np.amax(self.posteriors[0])
            self.post_max = self.sampler.sample_stats.lp.max().values
            self.sigmas = self.info.sd.values
            #self.chains = [self.ajuste for _ in range(10)]

            self.printsv('\n\n------------- PyMC3 Summary ------------\n\n')
            self.printsv(str(self.info))
            pass


        if self.switch2:
            for i in range(self.nins__):
                self.printsv('\nInstrument %i corresponds to %s' % (i+1, self.ins_label__[i]))
                self.switch2 = False

        # the following if prints results (best fits, sigmas and planet signatures)
        if True:
            j = 0  # delete this ap1
            self.sma = np.zeros(self.kplanets__)
            self.mm = np.zeros(self.kplanets__)

            self.sma_sig = np.zeros(self.kplanets__)
            self.mm_sig = np.zeros(self.kplanets__)


            for b in self:
                for p in b:
                    if p.fixed == None:
                        p.value = self.ajuste[j]
                        p.sigma = self.sigmas[j]
                        j += 1

            self.printsv('\n\n--------------- Best Fit ---------------\n\n')


            n = 0
            n_ = []
            tab_3 = np.array([])
            switch_title = True
            for b in self:
                if switch_title:
                    self.printsv(tabulate(b._list_('name', 'value', 'prior', 'limits', 'sigma').T,
                                          headers=['Name            ', 'Value       ', 'Prior   ', 'Limits      ', 'Sigma']))
                    switch_title = False
                else:
                    self.printsv(tabulate(b._list_('name', 'value', 'prior', 'limits', 'sigma').T,
                                          headers=['                ', '            ', '        ', '            ', '     ']))


                # PLANET SIGNATURES
                if b.type_ == 'Keplerian':
                    my_params = [None, None, None, None, None]
                    for i in range(5):
                        if b[i].fixed == None:
                            my_params[i] = self.chains[0][:, n*b.ndim_ + i].T
                        else:
                            my_params[i] = b[i].fixed * np.ones(self.chains.shape[1])

                    # this handles different parametrizations
                    if self.engine__.__name__ == 'emcee' or self.engine__.__name__ == 'dynesty':
                        if b.parametrization == 'vanilla' or b.parametrization == 0:
                            per, A, phase, ecc, w = b._list_('value')
                            per_, A_, phase_, ecc_, w_ = my_params

                        if b.parametrization == 'hou' or b.parametrization == 1:
                            P, As, Ac, S, C = b._list_('value')
                            P_, As_, Ac_, S_, C_ = my_params
                            #self.chains[0][:, n*b.ndim_:(n+1)*b.ndim_].T

                            per = np.exp(P)
                            per_ = np.exp(P_)

                            A, phase = delinearize(As, Ac)
                            ecc, w = delinearize(S, C)

                            A_, phase_ = adelinearize(As_, Ac_)
                            ecc_, w_ = adelinearize(S_, C_)
                            ########################


                            names0 = ['Period', 'Amplitude', 'Phase', 'Eccentricity', 'Longitude']
                            names = [name + ' %s' % b.signal_number for name in names0]
                            values = [per, A, phase, ecc, w]
                            sigmas = np.std([per_, A_, phase_, ecc_, w_], axis=1)

                            priors = [x[1] for x in b.hou_priors]
                            limits = [x[2] for x in b.hou_priors]
                            #limits = [np.exp(b[0].limits), [0, b[1].limits[1]**2], [0, 2*np.pi], [0, 1], [0, 2*np.pi]]

                            self.printsv(tabulate(np.array([names, values, priors, limits, sigmas], dtype=object).T, headers=['                ', '            ', '        ', '            ', '            ']))

                        if b.parametrization == 't0' or b.parametrization == 2:
                            per, A, T0, ecc, w = b._list_('value')
                            per_, A_, T0_, ecc_, w_ = my_params
                            pass

                        if b.parametrization == 'hout0' or b.parametrization == 3:
                            per, A, tp, S, C = b._list_('value')
                            per_, A_, tp_, S_, C_ = my_params

                            #per = np.exp(P)
                            #per_ = np.exp(P_)
                            ecc, w = delinearize(S, C)
                            ecc_, w_ = adelinearize(S_, C_)

                            names0 = ['Eccentricity', 'Longitude']
                            names = [name + '%s' % b.signal_number for name in names0]

                            values = [ecc, w]
                            sigmas = np.std([ecc_, w_], axis=1)

                            priors = [x[1] for x in b.hou_priors]
                            limits = [x[2] for x in b.hou_priors]
                            self.printsv(tabulate(np.array([names, values, priors, limits, sigmas], dtype=object).T, headers=['                ', '            ', '        ', '            ', '            ']))

                        j += 1  # delete this ap1

                        # this part is for planet signatures
                        if self.starmass:
                            self.sma[n], self.mm[n] = cps(per, A, ecc, self.starmass)
                            self.sma_sig[n], self.mm_sig[n] = np.std(cps(per_, A_, ecc_, self.starmass), axis=1)


                    if self.engine__.__name__ == 'pymc3':
                        ps_names = ['SMA %i' % (n+1), 'MinMass %i' % (n+1)]

                        planet_sig = az.summary(sim.sampler, var_names=ps_names)
                        self.printsv(str(planet_sig))

                        self.sma[n], self.mm[n] = planet_sig['mean'].values
                        self.sma_sig[n], self.mm_sig[n] = planet_sig['sd'].values

                    n += 1
                    n_.append(n)

                    if n == self.kplanets__:
                        tab_3 = np.array([[n_, self.sma, self.sma_sig, self.mm, self.mm_sig]]).T
                        self.printsv('\n----------------- Planet Signatures -----------------\n')
                        tabh_3 = ['Signal Number', 'Semi-Major Axis [AU]', 'Sigma', 'Minimum Mass [M_J]', 'Sigma']
                        self.printsv(tabulate(tab_3, headers=tabh_3))

                    pass


        # this if prints statistics
        if self.switch_RV:
            #try:
            if True:
                if self.engine__.__name__ == 'pymc3':
                    self.printsv('\n \n Initializing Elegant Fix for PyMC3')
                    for b in self:
                        if b.type_ == 'Keplerian':
                            b.model = mr._model_keplerian

                ymod, err2 = self.my_model.evaluate_model(self.ajuste)
                ferr2 = err2 + self.datayerr__ ** 2
                residuals = self.datay__ - ymod
                self.degrees_of_freedom = len(self.datax__) - self.ndim__

                self.chi2 = np.sum(residuals**2) / self.degrees_of_freedom
                self.chi2_red = np.sum(residuals**2 / ferr2) / self.degrees_of_freedom

                self.BIC = np.log(len(self.datax__)) * self.ndim__ - 2 * self.post_max
                self.AIC = 2 * self.ndim__ - 2 * self.post_max

                #self.RMS = np.sqrt(np.sum(residuals**2))
                self.RMSE = np.sqrt(np.sum(residuals ** 2) / len(residuals))

                # SAVE RESIDUALS, switch?
                if self.switch_sv_residuals:
                    np.savez_compressed(self.saveplace + '/residuals', np.array([self.datax__, residuals, self.dataflag__]))

                tabh_1 = ['Run Info                         ', '                            ']

                tab_1 =    [['Star Name                      : ', self.starname],
                            ['The sample sizes are           : ', size_info],
                            [setup_info, setup],
                            ['Model used is                  : ', str(self)],
                            ['N data                         : ', len(self.datax__)],
                            ['Number of Dimensions           : ', self.ndim__],
                            ['Degrees of Freedom             : ', self.degrees_of_freedom]]


                self.printsv('\n\n')
                self.printsv(tabulate(tab_1, headers=tabh_1))
                self.printsv('\n--------------------------------------------------- \n')



                if self.engine__.__name__ == 'pymc3':
                    pass

                self.printsv('\n--------------------------------------------------- \n')

                tabh_2 = ['Statistical Details              ', 'Value']

                tab_2 = [['The maximum posterior is    :    ', self.post_max],
                         #['The chi2 is                 :    ', self.chi2],
                         ['The reduced chi2 is         :    ', self.chi2_red],
                         ['The BIC is                  :    ', self.BIC],
                         ['The AIC is                  :    ', self.AIC],
                         ['The RMSE is                 :    ', self.RMSE]]

                         #,
                         #['RMS Deviation is            :    ', self.RMSD]]

                if self.engine__.__name__ == 'emcee':
                    self.printsv('Beta Detail                     :   ' + str(self.sampler.betas))
                    if self.switch_evidence:
                        x = [['The evidence is             :    ', '%.3f +- %.3f' % self.evidence]]
                        tab_2 = np.vstack([x, tab_2])

                if self.engine__.__name__ == 'dynesty':
                    x = [['The evidence is             :    ', '%.3f +- %.3f' % self.evidence]]
                    tab_2 = np.vstack([x, tab_2])


                if self.engine__.__name__ == 'pymc3':
                    for b in self:
                        if b.type_ == 'Keplerian':
                            b.model = mr._model_keplerian_pymc3
                self.printsv(tabulate(tab_2, headers=tabh_2))
                self.printsv('\n --------------------------------------------------- \n')


            # dynesty debugging
            #except:
            else:
                self.printsv('\n \n Post-Processing Stats failed (pymc3 probably)')

                if self.engine__.__name__ == 'pymc3':
                    self.printsv('\n \n Initializing Elegant Fix for PyMC3')
                    for b in self:
                        if b.type_ == 'Keplerian':
                            b.model = mr._model_keplerian


                    logl = _logl__(self.ajuste, self.my_model)
                    logp = _logl__(self.ajuste, self.my_model)
                    self.post_max = logl + logp

                    DATA = [self.datax__, self.datay__, self.datayerr__, self.dataflag__]
                    ymod, err2 = Model(DATA, self.blocks__).evaluate_model(self.ajuste)
                    ferr2 = err2 + self.datayerr__ ** 2
                    residuals = self.datay__ - ymod
                    self.degrees_of_freedom = len(self.datax__) - self.ndim__

                    self.chi2 = np.abs(np.sum(residuals**2/self.datayerr__))
                    self.chi2_red = np.sum(residuals**2 / ferr2) / self.degrees_of_freedom

                    self.BIC = np.log(len(self.datax__)) * self.ndim__ - 2 * self.post_max
                    self.AIC = 2 * self.ndim__ - 2 * self.post_max

                    #self.RMS = np.sqrt(np.sum(residuals**2))
                    self.RMSE = np.sqrt(np.sum(residuals ** 2) / len(residuals))

                    tabh_1 = ['Run Info                         ', '                            ']

                    tab_1 =    [['Star Name                      : ', self.starname],
                                ['The sample sizes are           : ', size_info],
                                [setup_info],
                                ['Model used is                  : ', str(self)],
                                ['N data                         : ', len(self.datax__)],
                                ['Number of Dimensions           : ', self.ndim__],
                                ['Degrees of Freedom             : ', self.degrees_of_freedom]]


                    self.printsv('\n\n')
                    self.printsv(tabulate(tab_1, headers=tabh_1))

                    self.printsv('\n--------------------------------------------------- \n')

                    tabh_2 = ['Statistical Details              ', 'Value']
                    tab_2 = [['The maximum posterior is    :    ', self.post_max],
                            #['The chi2 is                 :    ', self.chi2],
                             ['The reduced chi2 is         :    ', self.chi2_red],
                             ['The BIC is                  :    ', self.BIC],
                             ['The AIC is                  :    ', self.AIC],
                             #['The RMS is                  :    ', self.RMS],
                             ['The RMSE is                 :    ', self.RMSE]]

                    self.printsv(tabulate(tab_2, headers=tabh_2))
                    self.printsv('\n --------------------------------------------------- \n')

                    for b in self:
                        if b.type_ == 'Keplerian':
                            b.model = mr._model_keplerian_pymc3


            if self.run_save:
                np.savetxt(self.saveplace+'/log.dat', np.array([self.run_log]), fmt='%100s')
            pass

        if self.switch_F:
            try:
                ymod, err2 = self.my_model.evaluate_model(self.ajuste)
                ferr2 = err2 + self.fdatayerr__ ** 2
                residuals = self.fdatay__ - ymod
                self.degrees_of_freedom = len(self.fdatax__) - self.ndim__

                self.chi2 = np.abs(np.sum(residuals**2/self.datayerr__))
                self.chi2_red = np.sum(residuals**2 / ferr2) / self.degrees_of_freedom

                self.BIC = np.log(len(self.fdatax__)) * self.ndim__ - 2 * self.post_max
                self.AIC = 2 * self.ndim__ - 2 * self.post_max

                #self.RMS = np.sqrt(np.sum(residuals ** 2))
                self.RMSE = np.sqrt(np.sum(residuals ** 2) / len(residuals))

                tabh_1 = ['Run Info                         ', '                            ']

                tab_1 =    [['Star Name                      : ', self.starname],
                            ['The sample sizes are           : ', size_info],
                            [setup_info, setup],
                            ['Model used is                  : ', str(self)],
                            ['N data                         : ', len(self.fdatax__)],
                            ['Number of Dimensions           : ', self.ndim__],
                            ['Degrees of Freedom             : ', self.degrees_of_freedom]]



                self.printsv('\n\n')
                self.printsv(tabulate(tab_1, headers=tabh_1))
                self.printsv('\n--------------------------------------------------- \n')

                if self.engine__.__name__ == 'emcee':
                    self.printsv('Beta Detail                       : ' + str(self.sampler.betas))
                self.printsv('\n--------------------------------------------------- \n')

                tabh_2 = ['Statistical Details              ', 'Value']

                tab_2 = [['The maximum posterior is    :    ', self.post_max],
                         #['The chi2 is                 :    ', self.chi2],
                         ['The reduced chi2 is         :    ', self.chi2_red],
                         ['The BIC is                  :    ', self.BIC],
                         ['The AIC is                  :    ', self.AIC],
                         ['The RMSE is                 :    ', self.RMSE]]

                if self.engine__.__name__ == 'dynesty':
                    x = [['The evidence is             :    ', self.evidence]]
                    for tab in tab_2:
                        x.append(tab)

                self.printsv(tabulate(tab_2, headers=tabh_2))
                self.printsv('\n --------------------------------------------------- \n')
            except:
                print('Post process F A I L E D')

            if self.run_save:
                np.savetxt(self.saveplace+'/log.dat', np.array([self.run_log]), fmt='%100s')


    def _add_data__(self, target_name):
        self.starname = target_name

        self.staract__ = []
        self.cornums__ = []
        self.ins_label__ = np.array([])
        self.fins_label__ = np.array([])

        # RVS
        try:
            col = ['datax__', 'datay__', 'datayerr__', 'dataflag__']
            for c in col:
                setattr(self, c, np.array([]))

            for file in np.sort(os.listdir('datafiles/%s/RV/' % target_name)):
                data = np.loadtxt('datafiles/%s/RV/%s' % (target_name, file))
                # mean rv substract
                my = np.mean(data[:, 1])
                if abs(my) > 1e-6:
                    data[:, 1] -= my
                # end mean rv substract
                dat = np.vstack([data.T[:3], np.ones(len(data.T[0])) * self.nins__])
                for i in range(len(col)):
                    setattr(self, col[i], np.append(getattr(self, col[i]), dat[i]))

                self.ins_label__ = np.append(self.ins_label__, file)
                self.nins__ += 1
                self.printsv('Reading data from %s \n' % file)

                staracts = data.T[3:]
                cornums = len(staracts)
                #if staracts.size:
                self.staract__.append(staracts)
                self.cornums__.append(cornums)

                self.switch_RV = True
                #else:
                #    self.staract__.append(None)
                #    self.cornums__.append(None)

        except Exception:
            print('\n No RV folder found on directory')

        try:
            col = ['fdatax__', 'fdatay__', 'fdatayerr__', 'fdataflag__']
            for c in col:
                setattr(self, c, np.array([]))

            for file in np.sort(os.listdir('datafiles/%s/FLUX/' % target_name)):
                data = np.loadtxt('datafiles/%s/FLUX/%s' % (target_name, file))
                dat = np.vstack([data.T[:3], np.ones(len(data.T[0])) * self.fnins__])
                for i in range(len(col)):
                    setattr(self, col[i], np.append(getattr(self, col[i]), dat[i]))

                self.fins_label__ = np.append(self.fins_label__, file)
                self.fnins__ += 1
                self.printsv('Reading data from %s \n' % file)

                self.switch_F = True


        except Exception:
            print('\n No FLUX folder found on directory')


        self.saveplace = ensure_dir(target_name, loc=self.save_loc)


    def _sort_data__(self):
        if self.nins__:
            col = ['datax__', 'datay__', 'datayerr__', 'dataflag__']
            ord = np.argsort(self.datax__)
            for c in col:
                setattr(self, c, getattr(self, c)[ord])

            self.printsv('All data sorted \n')

            # t0
            self.datax__ -= self.datax__[0]

            self.ndat__ = len(self.datax__)
            self.data__ = [self.datax__, self.datay__, self.datayerr__, self.dataflag__]

            # mean sa substract
            if np.shape(self.cornums__):
                for ins in range(self.nins__):
                    if self.cornums__[ins]:
                        for sa in range(len(self.staract__[ins])):
                            q = self.staract__[ins][sa]
                            mq = np.mean(q)
                            if abs(mq) > 1e-6:
                                self.staract__[ins][sa] = (q - mq)# / abs(mq)

        if self.fnins__:
            col = ['fdatax__', 'fdatay__', 'fdatayerr__', 'fdataflag__']
            ord = np.argsort(self.fdatax__)
            for c in col:
                setattr(self, c, getattr(self, c)[ord])

            self.printsv('All data sorted \n')

            # t0
            self.fdatax__ -= self.fdatax__[0]

            self.fndat__ = len(self.fdatax__)
            self.fdata__ = [self.fdatax__, self.fdatay__, self.fdatayerr__, self.fdataflag__]
        # end mean sa substract
            pass


    def _data__(self, name):
        self._add_data__(name)
        self._sort_data__()


    def _mk_pos0__(self, setup, reduce=0):
        if self.engine__.__name__ == 'emcee':
            ntemps, nwalkers, nsteps = setup
            pos = np.zeros((ntemps, nwalkers, self.ndim__))
            for t in range(ntemps):
                j = 0
                for b in self:
                    for p in b:
                        if p.fixed == None:
                            fact = abs(p.limits[1] - p.limits[0]) / nwalkers
                            if b.parametrization != 0 or b.parametrization != 'vanilla':
                                if p != b[0]:
                                    fact *= 0.707
                            dif = np.arange(nwalkers) * fact * np.random.uniform(0.999, 1)
                            for i in range(nwalkers):
                                pos[t][i][j] = p.limits[0] + dif[i]
                            np.random.shuffle(pos[t, :, j])
                            j += 1
            #pos = np.array([pos for _ in range(ntemps)])
            #pos *= np.random.uniform(0.999, 1, [ntemps, nwalkers, self.ndim__])
            return pos


    def _mk_pos0_exp__(self, setup):
        if self.engine__.__name__ == 'emcee':
            ntemps, nwalkers, nsteps = setup
            pos = np.zeros((ntemps, nwalkers, self.ndim__))
            for t in range(ntemps):
                j = 0
                for b in self:
                    for p in b:
                        if p.fixed == None:
                            m = (p.limits[1] + p.limits[0]) / 2
                            r = (p.limits[1] - p.limits[0]) / 2
                            dist = np.sort(np.random.uniform(0, 1, nwalkers))

                            if b.parametrization != 0 or b.parametrization != 'vanilla':
                                if p != b[0]:
                                    r *= 0.707
                            pos[t][:, j] = r * (2 * dist - 1) + m
                            np.random.shuffle(pos[t, :, j])
                            j += 1
        return pos


    def _list__(self, *call):
        if len(call) == 1:
            return np.array([getattr(self.blocks__[i], call[0]) for i in range(len(self))])
        else:
            return np.array([np.array([getattr(self.blocks__[i], c) for i in range(len(self))]) for c in call])


    def plotmodel(self):
        pl.rcParams['font.size'] = 16
        pl.rcParams['axes.linewidth'] = 2

        if self.engine__.__name__ == 'pymc3':
            for b in self:
                if b.type_ == 'Keplerian':
                    b.model = mr._model_keplerian

                if b.type_ == 'Sinusoidal':
                    b.model = mr._model_sinusoid

        # colors
        c = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        c = flatten([c,c,c,c,c])

        # draw points
        x_, y_, yerr_, flags_ = self.data__
        #DATA = [x_, y_, yerr_, flags_]

        #flatten([b._list_('value') for b in sim])

        y_m, err_m = self.my_model.evaluate_model(self.ajuste)

        # true residuals
        residuals = y_ - y_m

        ferr = np.sqrt(err_m)

        minx, maxx = minmax(self.datax__)
        x_line = np.linspace(minx, maxx, 1000)

        # instrumental blocks
        ins_blocks = [block for block in self if block.type_ != 'Keplerian']
        ins_theta = flatten([block._list_('value')[block.C_] for block in self if block.type_ != 'Keplerian'])

        ### this gets offsets, moavs and staracts
        y_ins, yerr_ins2 = Model(self.data__, ins_blocks).evaluate_model(ins_theta)
        yy = y_ - y_ins


        # signal blocks
        kep_blocks = [block for block in self if block.type_ == 'Keplerian']
        kep_theta = flatten([block._list_('value')[block.C_] for block in self if block.type_ == 'Keplerian'])

        y_kep, yerr_kep = Model(self.data__, kep_blocks).evaluate_model(kep_theta)
        #raise Exception('asdasd')
        for mode in range(2):  # 0 full, 1 phasefolded
            if mode == 0:  # full
                if True:
                    # initialize figure
                    fig = pl.figure(figsize=(8, 8))
                    gs = gridspec.GridSpec(3, 4)
                    ax = fig.add_subplot(gs[:2, :])
                    pl.subplots_adjust(hspace=0)

                    #ticks
                    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
                    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

                    axr = fig.add_subplot(gs[-1, :], sharex=ax)

                    axr.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                    axr.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
                    axr.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                    axr.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')

                    pl.minorticks_on()

                # draw points per ins
                for i in range(self.nins__):
                    sel = flags_ == i

                    # top
                    ax.errorbar(x_[sel], yy[sel], yerr=yerr_[sel], fmt='%so' % c[i],
                                alpha=0.8, markersize=10, mec='k',
                                ecolor='k', capsize=3, zorder=2)
                    # residuals
                    axr.errorbar(x_[sel], residuals[sel], yerr=ferr[sel], fmt='%so' % c[i],
                                alpha=0.8, markersize=10, mec='k',
                                ecolor='k', capsize=3, zorder=2)
                axr.set_xlabel('Time [days]')
                axr.set_ylabel(r'Residuals [$\frac{m}{s}$]')
                ax.set_ylabel(r'RV [$\frac{m}{s}$]')

                # draw red line

                DATA_line = [x_line, np.zeros_like(x_line), np.zeros_like(x_line), np.ones_like(x_line)*-1]
                y_line, err_line = Model(DATA_line, kep_blocks).evaluate_plot(kep_theta, x_line)
                ax.plot(x_line, y_line, 'C3-', lw=2, zorder=3)

                ### horizontal lines at 0
                ax.axhline(0, color='gray', linewidth=2, zorder=1)
                axr.axhline(0, color='gray', linewidth=2, zorder=1)

                ### plot save and show
                if self.plot_save:
                    pl.savefig(self.saveplace+'/model_full.%s' % self.plot_fmt)
                if self.plot_show:
                    pl.show()

            if mode == 1:  # phasefolded
                for b in self:
                    if b.type_ == 'Keplerian' or b.type_ == 'Sinusoidal':
                        if True:
                            # initialize figure
                            fig = pl.figure(figsize=(8, 8))
                            gs = gridspec.GridSpec(3, 4)
                            ax = fig.add_subplot(gs[:2, :])
                            pl.subplots_adjust(hspace=0)

                            #ticks
                            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
                            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

                            axr = fig.add_subplot(gs[-1, :], sharex=ax)

                            axr.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                            axr.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
                            axr.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                            axr.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')

                            pl.minorticks_on()

                        #########
                        if b.parametrization == 1 or b.parametrization == 'hou':
                            p_ = np.exp(b[0].value)
                        else:
                            p_ = b[0].value
                        other_blocks = [block for block in self if block is not b]
                        other_theta = flatten([block._list_('value')[block.C_] for block in self if block is not b])

                        y_others, yerr_others2 = Model(self.data__, other_blocks).evaluate_model(other_theta)

                        y_floor = y_ - y_others
                        yerr_floor = np.sqrt(yerr_**2 + yerr_others2)


                        #y_t, yerr_t = Model(self.data__, [b]).evaluate_plot(b._list_('value'), x_line)  #targets

                        ### draw points per ins
                        for i in range(self.nins__):
                            sel = flags_ == i

                            # model
                            xf, yf, yerrf = fold(x_[sel], y_floor[sel], per=p_, yerr=yerr_floor[sel])
                            ax.errorbar(xf, yf, yerr=yerrf, fmt='%so' % c[i],
                                        alpha=0.8, markersize=10, mec='k',
                                        ecolor='k', capsize=3, zorder=2)

                            # residuals
                            xfr, yfr, yerrfr = fold(x_[sel], residuals[sel], per=p_, yerr=ferr[sel])
                            axr.errorbar(xfr, yfr, yerr=yerrfr, fmt='%so' % c[i],
                                        alpha=0.8, markersize=10, mec='k',
                                        ecolor='k', capsize=3, zorder=2)

                        ### draw red line
                        minx = 0
                        x1f = np.linspace(minx, minx+p_, 1000)

                        # target thetas
                        y1f, err1f = Model(self.data__, [b]).evaluate_plot(b._list_('value')[b.C_], x1f)

                        axr.set_xlabel('Time [days]')
                        axr.set_ylabel(r'Residuals [$\frac{m}{s}$]')
                        ax.set_ylabel(r'RV [$\frac{m}{s}$]')

                        ax.plot(x1f, y1f, 'C3-', lw=2, zorder=3)

                        ### horizontal lines
                        ax.axhline(0, color='gray', linewidth=2, zorder=1)
                        axr.axhline(0, color='gray', linewidth=2, zorder=1)

                        ### plot save and show
                        if self.plot_save:
                            pl.savefig(self.saveplace+'/model_%s.%s' % (b.name_, self.plot_fmt))
                        if self.plot_show:
                            pl.show()


        if self.engine__.__name__ == 'pymc3':
            for b in self:
                if b.type_ == 'Keplerian':
                    b.model = mr._model_keplerian_pymc3

                if b.type_ == 'Sinusoidal':
                    b.model = mr._model_sinusoid
        pass


    def plotmodelscale(self):
        import copy
        pl.rcParams['font.size'] = 18
        pl.rcParams['axes.linewidth'] = 2

        if self.engine__.__name__ == 'pymc3':
            for b in self:
                if b.type_ == 'Keplerian':
                    b.model = mr._model_keplerian

                if b.type_ == 'Sinusoidal':
                    b.model = mr._model_sinusoid

        # colors
        c = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        c = flatten([c,c,c,c,c])

        # draw points
        x_, y_, yerr_, flags_ = self.data__
        #DATA = [x_, y_, yerr_, flags_]

        #flatten([b._list_('value') for b in sim])

        y_m, err_m = self.my_model.evaluate_model(self.ajuste)

        # true residuals
        residuals = y_ - y_m

        ferr = np.sqrt(err_m)
        minx, maxx = minmax(self.datax__)
        x_line = np.linspace(minx, maxx, 1000)


        scales = self[0]._list_('value')[-self.nins__:]
        offsets = [self[i+1][0].value for i in range(self.nins__)]

        # draw points per ins
        ins_blocks = [block for block in self if block.type_ != 'Keplerian']
        ins_theta = flatten([block._list_('value')[block.C_] for block in self if block.type_ != 'Keplerian'])

        ### this gets offsets
        y_ins, yerr_ins2 = Model(self.data__, ins_blocks).evaluate_model(ins_theta)

        y_floor = y_ - y_ins
        yerr_floor = np.sqrt(yerr_**2 + yerr_ins2)

        for mode in range(2):  # 0 full, 1 phasefolded
            if mode == 0:  # full

                if True:
                    # initialize figure
                    fig = pl.figure(figsize=(8, 8))
                    gs = gridspec.GridSpec(3, 4)
                    ax = fig.add_subplot(gs[:2, :])
                    pl.subplots_adjust(hspace=0)

                    #ticks
                    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
                    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

                    axr = fig.add_subplot(gs[-1, :], sharex=ax)

                    axr.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                    axr.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
                    axr.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                    axr.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')

                    pl.minorticks_on()

                # draw red line
                # choose redline model with flags
                DATA_line = [x_line, np.zeros_like(x_line), np.zeros_like(x_line), np.ones_like(x_line)*-1]
                y_line, err_line = Model(DATA_line, [self[0]]).evaluate_plot(self.ajuste, x_line)
                ax.plot(x_line, y_line, 'C3-', lw=2)

                for i in range(self.nins__):
                    sel = flags_ == i
                    # top
                    yy = (y_[sel] - y_ins[sel]) * (1. / scales[i])
                    ax.errorbar(x_[sel], yy, yerr=yerr_floor[sel], fmt='%so' % c[i], alpha=0.8, markersize=10)

                    # residuals
                    axr.errorbar(x_[sel], residuals[sel], yerr=ferr[sel], fmt='%so' % c[i], alpha=0.8, markersize=10)

                ### horizontal lines at 0
                ax.axhline(0, color='gray', linewidth=2)
                axr.axhline(0, color='gray', linewidth=2)

                ### plot save and show
                if self.plot_save:
                    pl.savefig(self.saveplace+'/model_full.%s' % self.plot_fmt)
                if self.plot_show:
                    pl.show()

                #raise Exception('debug')
            if mode == 1:  # phasefolded
                for b in self:
                    if b.type_ == 'Keplerian' or b.type_ == 'Sinusoidal':
                        for kplan in range(self.kplanets__):

                            if True:
                                # initialize figure
                                fig = pl.figure(figsize=(8, 8))
                                gs = gridspec.GridSpec(3, 4)
                                ax = fig.add_subplot(gs[:2, :])
                                pl.subplots_adjust(hspace=0)

                                #ticks
                                ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                                ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
                                ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                                ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

                                axr = fig.add_subplot(gs[-1, :], sharex=ax)

                                axr.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                                axr.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
                                axr.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                                axr.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')

                                pl.minorticks_on()

                            ## period for phase
                            if b.parametrization == 1 or b.parametrization == 'hou':
                                p_ = np.exp(b[5*kplan].value)
                            else:
                                p_ = b[5*kplan].value
                            ### create other block

                            if True:
                                kepler_params = b[:5*kplan]
                                kepler_params = np.append(kepler_params, b[5*(kplan+1):])

                                target_params = b[5*kplan:5*(kplan+1)]
                                target_params = np.append(target_params, b[-self.nins__:])

                                kp = copy.deepcopy(kepler_params)
                                tp = copy.deepcopy(target_params)

                                b_mod = mr._model_keplerian_scale
                                Support_block = Parameter_Block(kp, block_model = b_mod, block_name='Kepler Scale draw', block_type='Keplerian')

                                Support_block.extra_args = [self.nins__, self.kplanets__ - 1]
                                Support_block._refresh_()

                                # other blocks
                                ob = [Support_block]
                                #for x in ins_blocks:
                                #    ob.append(x)

                                # target block
                                Target_block = Parameter_Block(tp, block_model = b_mod, block_name='Kepler Scale draw', block_type='Keplerian')
                                Target_block.extra_args = [self.nins__, 1]
                                Target_block._refresh_()

                            #  other theta
                            ot = flatten([block._list_('value')[block.C_] for block in ob])

                            y_others, yerr_others2 = Model(self.data__, ob).evaluate_model(ot)

                            y_floor = y_ - y_others - y_ins
                            yerr_floor = np.sqrt(yerr_**2 + yerr_others2)

                            ### draw red line
                            minx = 0
                            x1f = np.linspace(minx, minx+p_, 1000)

                            # target thetas
                            target_theta = Target_block._list_('value')[Target_block.C_]
                            y1f, err1f = Model(self.data__, [Target_block]).evaluate_plot(target_theta, x1f)

                            ax.plot(x1f, y1f, 'C3-', lw=2)

                            ### draw points per ins
                            for i in range(self.nins__):
                                sel = flags_ == i

                                # model
                                xf, yf, yerrf = fold(x_[sel], y_floor[sel], per=p_, yerr=yerr_floor[sel])
                                ax.errorbar(xf, yf*(1./scales[i]), yerr=yerrf, fmt='%so' % c[i], alpha=0.8, markersize=10)

                                # residuals
                                xfr, yfr, yerrfr = fold(x_[sel], residuals[sel], per=p_, yerr=ferr[sel])
                                axr.errorbar(xfr, yfr, yerr=yerrfr, fmt='%so' % c[i], alpha=0.8, markersize=10)

                            ### horizontal lines
                            ax.axhline(0, color='gray', linewidth=2)
                            axr.axhline(0, color='gray', linewidth=2)

                            ### plot save and show
                            if self.plot_save:
                                pl.savefig(self.saveplace+'/model_%s.%s' % (kplan, self.plot_fmt))
                            if self.plot_show:
                                pl.show()

                            #raise Exception('debug debug debug')
        pass


    def plotfmodel(self):
        import batman
        pl.rcParams['font.size'] = 16
        pl.rcParams['axes.linewidth'] = 2
        c = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        c = flatten([c,c,c,c,c])
        x_, y_, yerr_, flags_ = self.fdata__

        y_m, err_m = self.my_model.evaluate_model(self.ajuste)
        residuals = y_ - y_m
        ferr = np.sqrt(err_m)

        minx, maxx = minmax(self.fdatax__)
        x_line = np.linspace(minx, maxx, 10000)

        ins_blocks = [block for block in self if block.type_ != 'Transit']
        ins_theta = flatten([block._list_('value')[block.C_] for block in self if block.type_ != 'Transit'])

        ### this gets offsets, moavs and staracts
        y_ins, yerr_ins2 = Model(self.fdata__, ins_blocks).evaluate_model(ins_theta)
        yy = y_ - y_ins

        kep_blocks = [block for block in self if block.type_ == 'Transit']
        kep_theta = flatten([block._list_('value')[block.C_] for block in self if block.type_ == 'Transit'])

        y_kep, yerr_kep = Model(self.fdata__, kep_blocks).evaluate_model(kep_theta)

        for mode in range(2):  # 0 full, 1 phasefolded
            if mode == 0:  # full
                if True:
                    # initialize figure
                    fig = pl.figure(figsize=(8, 8))
                    gs = gridspec.GridSpec(3, 4)
                    ax = fig.add_subplot(gs[:2, :])
                    pl.subplots_adjust(hspace=0)

                    #ticks
                    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
                    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

                    axr = fig.add_subplot(gs[-1, :], sharex=ax)

                    axr.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                    axr.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
                    axr.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                    axr.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')

                    pl.minorticks_on()

                # draw points per ins
                for i in range(self.fnins__):
                    sel = flags_ == i

                    # top
                    ax.errorbar(x_[sel], yy[sel], yerr=yerr_[sel], fmt='%so' % c[i],
                                alpha=0.5, markersize=5, mec='k',
                                ecolor='k', capsize=2, zorder=2)
                    # residuals
                    axr.errorbar(x_[sel], residuals[sel], yerr=ferr[sel], fmt='%so' % c[i],
                                alpha=0.5, markersize=5, mec='k',
                                ecolor='k', capsize=2, zorder=2)
                axr.set_xlabel('Time [days]')
                axr.set_ylabel(r'Residuals [$\frac{m}{s}$]')
                ax.set_ylabel(r'RV [$\frac{m}{s}$]')

                # draw red line

                DATA_line = [x_line, np.zeros_like(x_line), np.zeros_like(x_line), np.ones_like(x_line)*-1]
                y_line, err_line = Model(DATA_line, kep_blocks).evaluate_plot(kep_theta, x_line)
                ax.plot(x_line, y_line, 'C3-', lw=2, zorder=3)

                ### horizontal lines at 0
                ax.axhline(1, color='gray', linewidth=2, zorder=1)
                axr.axhline(0, color='gray', linewidth=2, zorder=1)

                ### plot save and show
                if self.plot_save:
                    pl.savefig(self.saveplace+'/model_full.%s' % self.plot_fmt)
                if self.plot_show:
                    pl.show()

            if mode == 1:  # phasefolded
                for b in self:
                    if b.type_ == 'Transit':
                        if True:
                            # initialize figure
                            fig = pl.figure(figsize=(8, 8))
                            gs = gridspec.GridSpec(3, 4)
                            ax = fig.add_subplot(gs[:2, :])
                            pl.subplots_adjust(hspace=0)

                            #ticks
                            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
                            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

                            axr = fig.add_subplot(gs[-1, :], sharex=ax)

                            axr.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
                            axr.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
                            axr.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
                            axr.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')

                            pl.minorticks_on()

                        #########
                        if b.parametrization == 1 or b.parametrization == 'hou':
                            p_ = np.exp(b[0].value)

                        else:
                            p_ = b[0].value
                        other_blocks = [block for block in self if block is not b]
                        other_theta = flatten([block._list_('value')[block.C_] for block in self if block is not b])

                        y_others, yerr_others2 = Model(self.fdata__, other_blocks).evaluate_model(other_theta)

                        y_floor = y_ - y_others
                        yerr_floor = np.sqrt(yerr_**2 + yerr_others2)


                        #y_t, yerr_t = Model(self.data__, [b]).evaluate_plot(b._list_('value'), x_line)  #targets

                        ### draw points per ins
                        for i in range(self.fnins__):
                            sel = flags_ == i

                            # model
                            xf, yf, yerrf = fold(x_[sel], y_floor[sel], per=p_, yerr=yerr_floor[sel])
                            ax.errorbar(xf, yf, yerr=yerrf, fmt='%so' % c[i],
                                        alpha=0.5, markersize=5, mec='k',
                                        ecolor='k', capsize=2, zorder=2)

                            # residuals
                            xfr, yfr, yerrfr = fold(x_[sel], residuals[sel], per=p_, yerr=ferr[sel])
                            axr.errorbar(xfr, yfr, yerr=yerrfr, fmt='%so' % c[i],
                                        alpha=0.5, markersize=5, mec='k',
                                        ecolor='k', capsize=2, zorder=2)

                        ### draw red line
                        minx = 0
                        x1f = np.linspace(minx, minx+p_, 10000)

                        # target thetas
                        y1f, err1f = Model(self.fdata__, [b]).evaluate_plot(b._list_('value')[b.C_], x1f)

                        axr.set_xlabel('Time [days]')
                        axr.set_ylabel(r'Residuals [$\frac{m}{s}$]')
                        ax.set_ylabel(r'RV [$\frac{m}{s}$]')

                        ax.plot(x1f, y1f, 'C3-', lw=2, zorder=3)

                        ### horizontal lines
                        ax.axhline(1, color='gray', linewidth=2, zorder=1)
                        axr.axhline(0, color='gray', linewidth=2, zorder=1)

                        ### plot save and show
                        if self.plot_save:
                            pl.savefig(self.saveplace+'/model_%s.%s' % (b.name_, self.plot_fmt))
                        if self.plot_show:
                            pl.show()

        pass


    def plotline(self):
        x = self.datax__
        y = self.datay__
        yerr = self.datayerr__

        pl.errorbar(x,y,yerr=yerr, fmt='o')

        xr = np.linspace(0, np.amax(x))

        DATA = [xr, np.zeros_like(xr), np.zeros_like(xr), np.zeros_like(xr)]
        yr, errr = Model(DATA, self.blocks__).evaluate_plot(np.array([-0.9594, 4.294, 0.534]), xr)

        pl.plot(xr, yr, 'k-')

        ym, errm = Model(DATA, self.blocks__).evaluate_plot(self.ajuste, xr)
        pl.plot(xr, ym, 'g-')

        if self.plot_save:
            pl.savefig(self.saveplace+'/line.%s' % self.plot_fmt)
        if self.plot_show:
            pl.show()


    def plottrace(self):
        import arviz as az

        svloc = self.saveplace+'/traces'
        if self.engine__.__name__ == 'emcee':
            i, j, k = 0, 0, 0

            axsize = np.amax(self._list__('ndim_'))

            for b in self.blocks__:
                for p in b:
                    if p.fixed == None:
                        if j % axsize == 0:
                            datadict = {}
                        datadict[p.name] = self.sampler.flatchain[:, :, i]
                        i += 1
                        j += 1
                        if j % axsize == 0 or i==self.ndim__:
                            az.plot_trace(datadict)
                            j = 0
                            k += 1
                            if self.plot_save:
                                pl.savefig(svloc+'/trace_%s.%s' % (k, self.plot_fmt))


        if self.engine__.__name__ == 'dynesty':
            from dynesty import plotting as dyplot
            pl.constrained_layout = False
            results = self.sampler.results

            i, j, k = 0, 0, 0
            axsize = np.amax(self._list__('ndim_'))
            axnames = flatten([b._list_('name') for b in self])

            rfig, raxes = dyplot.runplot(results)
            if self.plot_save:
                pl.savefig(svloc+'/runplot.%s' % self.plot_fmt)

            # Plot traces and 1-D marginalized posteriors.
            pl.rcParams['font.size'] = 10
            pl.rcParams['axes.linewidth'] = 2

            ceilq = self.ndim__//axsize+1
            if self.ndim__ % axsize == 0:
                ceilq = self.ndim__//axsize

            for n in range(int(np.ceil(ceilq))):
                j += 1
                top = (n+1)*axsize
                if top >= self.ndim__:
                    top = self.ndim__
                my_dims = range(n*axsize, top)
                tfig, taxes = dyplot.traceplot(results, show_titles=True, dims=my_dims)
                try:
                    tfig.tight_layout()
                except:
                    print('No tight_layout for dynesty traceplots')
                    pass

                if self.plot_save:
                    pl.savefig(svloc+'/trace%s.%s' % (n, self.plot_fmt))


        if self.engine__.__name__ == 'pymc3':
            #az.plot_trace(self.sampler)
            #'''
            k = 0
            for j in range(self.ndim__):
                if j % 5 == 0:
                    az.plot_trace(self.sampler, var_names=self.varnames[5*(k):5*(k+1)])
                    k += 1
                    if self.plot_save:
                        pl.savefig(svloc+'/trace_%s.%s' % (str(j//5), self.plot_fmt))
                if j==self.ndim__-1:
                    break
            #'''

        if self.plot_show:
            pl.show()

        pl.close('all')
        pass


    def plotpost(self):

        if self.engine__.__name__ == 'emcee':
            samples_h = self.chains[0]

        if self.engine__.__name__ == 'dynesty':
            samples_h = self.chains[0]

        if self.engine__.__name__ == 'pymc3':
            print('pymc3 posterior plots... soon')
            import arviz as az
            for b in self:
                az.plot_posterior(self.sampler, var_names=b._list_('name'))
                pl.savefig(self.saveplace+'/posteriors/post_%s.%s' % (b.name_, self.plot_fmt))
            return None

        c = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        c = flatten([c,c,c,c,c])
        alp, ms = 0.3, 2

        posts_h = self.posteriors[0]

        autothin = 1
        while len(samples_h[:, 0][::autothin]) > 3e6:
             autothin += 1

        i, j = 0, 0
        numplot = 0
        #axsize = np.amax(self._list__('ndim_'))
        #if self.ndim__ < axsize:
            #axsize = self.ndim__
        for b in self.blocks__:
            axsize = b.ndim_
            if self.ndim__ < axsize:  # ????
                axsize = self.ndim__

            if b.type_ == 'Keplerian':
                pass
            if b.type_ == 'Instrumental':
                if b.ndim_ > 4:
                    axsize = 4
            for p in b:
                if p.fixed == None:
                    if j % axsize == 0:
                        if numplot == self.ndim__//axsize:
                            fig, ax = pl.subplots(self.ndim__%axsize, 1, sharey=True, figsize=(12,10), constrained_layout=True)
                        else:
                            fig, ax = pl.subplots(axsize, 1, sharey=True, figsize=(12,10), constrained_layout=True)
                            numplot += 1
                        pl.ylabel('Posterior Prob', fontsize=22)
                        pl.title('Posteriors', fontsize=18)

                    if np.sum(np.array([ax]).shape) > 2:
                        ax[j].tick_params(direction='out', length=6, width=2, labelsize=14)
                        ax[j].plot(samples_h[:, i], posts_h, '%so' % c[i], alpha=alp, markersize=ms)
                        ax[j].set_xlabel(p.name, fontsize=22)
                        l1 = ax[j].axvline(p.value)
                    else:
                        ax.tick_params(direction='out', length=6, width=2, labelsize=14)
                        ax.plot(samples_h[:, i], posts_h, '%so' % c[i], alpha=alp, markersize=ms)
                        ax.set_xlabel(p.name, fontsize=22)
                        l1 = ax.axvline(p.value)
                    i += 1
                    j += 1
                    if j % axsize == 0 or i==self.ndim__:
                        if self.plot_save:
                            pl.savefig(self.saveplace+'/posteriors/post_%s.%s' % (numplot, self.plot_fmt))
                        if self.plot_show:
                            pl.show()
                        j = 0

        pl.close('all')
        pass


    def plotcorner(self, temp=0, dims=None):
        if self.engine__.__name__ == 'emcee':
            import corner
            try:
                print('Plotting Corner Plot... May take a while')
                subtitles = flatten([b._list_('name')[b.C_] for b in self])
                fig = corner.corner(self.chains[temp], labels=subtitles,
                                    title_kwargs={'fontsize': 6})
                if self.plot_save:
                    fig.savefig(self.saveplace+"/corner.pdf")

                pl.close('all')
            except:
                print('Corner Plot Failed!!')

        if self.engine__.__name__ == 'dynesty':
            try:
                fg, ax = dyplot.cornerplot(results, color='dodgerblue', show_titles=True,
                                                quantiles=None, max_n_ticks=3, dims=dims)
                if self.plot_save:
                    fig.savefig(self.saveplace+"/corner.pdf")
                pl.close('all')
            except Exception:
                print('\nCorner Plot Failed!!')


    def plothistograms(self):
        from scipy.stats import norm, skew, kurtosis
        from decimal import Decimal  # histograms
        if self.engine__.__name__ == 'emcee':
            #samples_h = self.chains[0]
            temp = 0
            ch = self.chains[temp]
            posts = self.posteriors[temp]

        if self.engine__.__name__ == 'dynesty':
            #samples_h = self.chains
            temp = 0  # trick
            ch = self.chains[temp]
            posts = self.posteriors[0]



        if self.engine__.__name__ == 'pymc3':
            print('pymc3 posterior plots... soon')
            return None

        c = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        c = flatten([c,c,c,c,c])
        alp, ms = 0.3, 2

        posts_h = self.posteriors[0]

        autothin = 1
        while len(ch[:, 0][::autothin]) > 3e6:
             autothin += 1

        num_bins = 16


        i, j = 0, 0
        for b in self.blocks__:
            if b.type_ == 'Keplerian':
                pass

            for p in b:
                if p.fixed == None:
                    target = ch[:, i]
                    mu, sigma = norm.fit(target)
                    n, bins, patches = pl.hist(target, num_bins, density=True)
                    pl.close("all")  # We don't need the plot just data!!

                    #Get the maximum and the data around it!!
                    maxi = target[posts == max(posts)][0]
                    dif = np.fabs(maxi - bins)
                    his_max = bins[dif == min(dif)][0]

                    # Find zeroes!!!
                    res = np.where(n == 0)[0]
                    if res.size:
                        sub = res[0]
                        if len(res) > 2:
                            for j in range(len(res)-2):
                                if res[j+2] - res[j] == 2:
                                    sub = j
                                    break


                        # Get the data subset!!
                        if bins[sub] > his_max:
                            post_sub = posts[np.where(target <= bins[sub])]
                            target_sub = target[np.where(target <= bins[sub])]
                        else:
                            post_sub = posts[np.where(target >= bins[sub])]
                            target_sub = target[np.where(target >= bins[sub])]

                    else:
                        target_sub = target
                        post_sub = posts

                    # the plot
                    pl.subplots(figsize=(10,7))  # Define the window size!!
                    n, bins, patches = pl.hist(target_sub, num_bins, density=True, alpha=0.5, ec='k')

                    mu, sigma = norm.fit(target_sub)  # add a 'best fit' line
                    var = sigma**2.
                    #Some Stats!!
                    skew_ = '%.4E' % Decimal(skew(target_sub))
                    kurt = '%.4E' % Decimal(kurtosis(target_sub))
                    gmod = '%.4E' % Decimal(bins[np.where(n == max(n))][0])
                    med = '%.4E' % Decimal(np.median(target_sub))

                    #Make a model x-axis!!
                    span = bins[len(bins)-1] - bins[0]
                    bins_x = ((np.arange(num_bins*100.) / (num_bins*100.)) * span) + bins[0]

                    gaussian = np.exp(-np.power((bins_x - mu)/sigma, 2.)/2.)

                    y = gaussian * np.max(n) #Renormalised to the histogram maximum!!

                    axes = pl.gca()

                    pl.plot(bins_x, y, 'C1-',lw=3)
                    pl.subplots_adjust(left=0.15)

                    axes.set_ylim([0., max(n)+ max(n)*0.7])
                    axes.set_xlabel(p.name, size=15)
                    axes.set_ylabel('Frequency',size=15)
                    axes.tick_params(labelsize=15)

                    pl.autoscale(enable=True, axis='x', tight=True)

                    ymin, ymax = axes.get_ylim()
                    xmin, xmax = axes.get_xlim()

                    #Add a key!!
                    mu_o = '%.4E' % Decimal(mu)
                    sigma_o = '%.4E' % Decimal(sigma)
                    var_o = '%.4E' % Decimal(var)

                    #Get the axis positions!!
                    axes.text(xmax - (xmax - xmin)*0.65, ymax - (ymax - ymin)*0.1, r"$\mathcal{N}(\mu_1,\sigma^2,\mu_3,\mu_4)$",size=25)
                    axes.text(xmax - (xmax - xmin)*0.9, ymax - (ymax - ymin)*0.180, r"$\mu_1 ={}$".format(mu_o),size=20)
                    axes.text(xmax - (xmax - xmin)*0.9, ymax - (ymax - ymin)*0.255, r"$\sigma^2 ={}$".format(var_o),size=20)
                    axes.text(xmax - (xmax - xmin)*0.9, ymax - (ymax - ymin)*0.330, r"$\mu_3 ={}$".format(skew_),size=20)

                    axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.180, r"$\mu_4 ={}$".format(kurt),size=20)
                    axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.255, r"$Median ={}$".format(med),size=20)
                    axes.text(xmax - (xmax - xmin)*0.5, ymax - (ymax - ymin)*0.330, r"$Mode ={}$".format(gmod),size=20)


                    pl.savefig(self.saveplace+'/histograms/hist_%s.%s' % (i, self.plot_fmt)) #,bbox_inches='tight')
                    #pl.show()
                    i += 1
                    #pl.close('all')
                    #pbar_hist.update(1)


        pass


    def best_fit(self):
        return flatten([b._list_('value') for b in self])


    def save_chain(self, temps=None):
        x = ''
        hea = flatten([b._list_('name')[b.C_] for b in self])
        for script in hea:
            x += str(script)+' \t'
        if self.engine__.__name__ == 'emcee':
            if temps == None:
                temps = range(self.setup[0])
            for temp in temps:
                #np.savetxt(self.saveplace + '/chain_'+str(temp)+'.dat', self.chains[temp], header=x)
                np.savez_compressed(self.saveplace + '/chain_'+str(temp), self.chains[temp])

        if self.engine__.__name__ == 'dynesty':
            #np.savetxt(self.saveplace + '/chain.dat', self.sampler.results.samples, header=x)
            np.savez_compressed(self.saveplace + '/chain', self.sampler.results.samples)


    def save_posteriors(self, temps=None):
        if self.engine__.__name__ == 'emcee':
            if temps == None:
                temps = range(self.setup[0])
            for temp in temps:
                #np.savetxt(self.saveplace + '/posterior_'+str(temp)+'.dat', self.posteriors[temp])
                np.savez_compressed(self.saveplace + '/posterior_'+str(temp), self.posteriors[temp])

        if self.engine__.__name__ == 'dynesty':
            #np.savetxt(self.saveplace + '/posteriors.dat', self.posteriors)
            #np.savetxt(self.saveplace + '/evidence.dat', self.evidence)
            np.savez_compressed(self.saveplace + '/posteriors', self.posteriors)
            np.savez_compressed(self.saveplace + '/evidence', self.evidence)


class my_stats:
    def Uniform(x, limits, args):
        if limits[0] <= x <= limits[1]:
            return 0.
        else:
            return -np.inf

    def Normal(x, limits, args):
        if limits[0] <= x <= limits[1]:
            mean, var = args[0], 2*args[1]
            return ( - (x - mean) ** 2 / var)
        else:
            return -np.inf

    def Fixed(x, limits, args):
        return 0.

    def Hou(x, limits, args):
        if limits[0] <= x <= limits[1]:
            return 0.
        else:
            return -np.inf

    def Jeffreys(x, limits, args):
        if lims[0] <= x <= lims[1]:
            return np.log(x**-1 / (np.log(lims[1]/lims[0])))
        else:
            return -np.inf

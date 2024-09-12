# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT
# **FIN  : Finish this

# sourcery skip: remove-redundant-if
if True:
    import itertools
    import multiprocessing
    import os
    import time
    import pickle
    import gc

    import numpy as np
    import pandas as pd
    from importlib import reload
    from tqdm import tqdm

    from tabulate import tabulate
    from termcolor import colored


    from .block import ReddModel
    from .canvas import plot_GM_Estimator, plot_beta_density, plot_trace2, plot_KeplerianModel, super_plots, plot_histograms, plot_betas, plot_rates
    from .globals import _PLATFORM_SYSTEM, _CORES, _TERMINAL_WIDTH, _OS_ROOT
    from .utils import *
    from .block_repo import *

    if _PLATFORM_SYSTEM == 'Darwin':
        multiprocessing.set_start_method('fork')  # not spawn
        pass
    else:
        pass

if True:
    import tracemalloc
    tracemalloc.start()

stat_names_ = ['chi2', 'chi2_red', 'AIC', 'BIC',
               'DIC', 'HQIC', 'RMSE', 'post_max',
               'like_max', 'BayesFactor']


class ModelSelectionObj:
    def __init__(self, crit):
        self._current_criteria = crit
        self._tolerance = 5
        self.compare_f = self.foobarmin
        self.dict = {'chi2':None, 'chi2_red':None, 'AIC':self.foobarmin, 'BIC':self.foobarmin,
                     'DIC':self.foobarmin, 'HQIC':self.foobarmin, 'RMSE':None, 'post_max':None,
                     'like_max':None, 'BayesFactor':self.foobarmax, 'Pass':self.foobarpass}
        self.msg = ''
        self.update()

    @property
    def criteria(self):
        return self._current_criteria

    @criteria.setter
    def set_criteria(self, val):
        self._current_criteria = val
        self.update()

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def set_tolerance(self, val):
        self._tolerance = val


    def foobarmin(self, foo, bar):
        foo_d = np.round(foo, 3)
        bar_d = np.round(bar, 3)
        self.msg = f'{foo_d} < {bar_d} - {self._tolerance}'

        return foo < bar - self._tolerance

    def foobarmax(self, foo, bar):
        foo_d = np.round(foo, 3)
        bar_d = np.round(bar, 3)
        self.msg = f'{foo_d} > {bar_d} + {self._tolerance}'

        return foo > bar + self._tolerance

    def foobarpass(self, foo, bar):
        self.msg = f'Criteria set to PASS'
        return True

    def compare(self, foo, bar):
        return self.compare_f(foo, bar)

    def update(self):
        self.compare_f = self.dict[self._current_criteria]


class Simulation(object):
    NONETHINGS = ['starname', 'betas', 'saveplace', 'ndim__', 'instrument_names']
    SWITCHES_F = ['switch_RV', 'switch_SA', 'switch_constrain',
                    'switch_dynamics', 'dynamics_already_included', 'debug_mode',
                    'switch_celerite', 'switch_AM', 'switch_PM', 'switch_inclination',
                    'FPTS', 'save_all', 'save_plots', 'use_c']
    SWITCHES_T = ['switch_first', 'switch_evidence', 'gaussian_mixtures_fit', 'switch_jitter',
                      'save_log', 'save_log_simple', 'save_backends']
    EMPTYLISTS = ['blocks__', 'model', 'conds', 'general_dependencies', 'model_dependencies']
    ZEROTHINGS = ['kplanets__', 'nins__','acceleration', 'keplerian_parameterisation']
    EMPTYSTRINGS = ['save_loc', 'read_loc']

    STATS_POSI = ['chi2', 'chi2_red', 'AIC', 'BIC',
                        'DIC', 'HQIC', 'RMSE', 'RMSi', 'Weights']
    STATS_NEGA = ['post_max', 'like_max', 'BayesFactor']

    def __init__(self, setup=None):
        if setup is None:
            setup = []
        # LOAD ATTRIBUTES
        self.time_init = time.time()
        self.cores__ = _CORES

        self.rounder_display = 3
        self.rounder_math = 8

        self.logger = reddlog()

        self.init_default_config()

        self.starmass = 1.

        self.model_constants = {'nan':'np.nan',
                                'gaussian_mixture_objects':'dict()',
                                }

        self.moav = {'order':0,
                     'global':False}

        self.eccentricity_limits = [0, 1]
        self.eccentricity_prargs = [0, 0.1]

        self.jitter_limits = [0, 1]
        self.jitter_prargs = [5, 5]


        self.multiprocess_method = 1

        # use as point estimate
        self.use_fit = 'max_post'


        # constrain
        self.constrain_sigma = 3
        self.constrain_method = 'sigma'  # 'sigma', 'GM'

        # posterior
        self.cherry = {'cherry':True,
                       'median':False,
                       'diff':20}
        self.posterior_fit_method = 'GM'  # KDE*soon

        self.posterior_dict = {'GM': 'Gaussian Mixtures',
                               'KDE': 'Kernel Density Estimation',
                               None: 'None'}


        # celerite

        self.my_kernel = {'terms':['SHOTerm'],
                          'params':[{'S0':0.0,
                                     'w0':1.0,
                                     'Q':0.25}]
                            }


        # Writing stuff
        self.dynesty_config = {'dlogz_init':0.05,
                                }


        self.ModelSelection = ModelSelectionObj('BIC')
        self.evidence = 0, 0

        # plots

        self.init_plot_config()


    def init_default_config(self):
        for c in self.NONETHINGS:
            setattr(self, c, None)

        for switch in self.SWITCHES_F:
            setattr(self, switch, False)

        for switch in self.SWITCHES_T:
            setattr(self, switch, True)

        for e in self.EMPTYLISTS:
            setattr(self, e, [])

        for nu in self.ZEROTHINGS:
            setattr(self, nu, 0)

        for e in self.EMPTYSTRINGS:
            setattr(self, e, '')

        for stat in self.STATS_POSI:
            setattr(self, stat, np.inf)

        for stat in self.STATS_NEGA:
            setattr(self, stat, -np.inf)


    def init_plot_config(self):
        axhline_kwargs = {'color':'gray', 'linewidth':2}
        errorbar_kwargs = {'marker':'o', 'ls':'', 'alpha':1.0, 'lw':1}
        fonts_kwargs = {}

        self.plot_all = {'plot':True,
                         'saveloc':'',
                         'paper_mode':True,
                         'time_to_plot':0,
                         'logger_level':'ERROR',
                         'format':'png'

                         }

        self.plot_posteriors = {'modes':[0, 1, 2, 3],
                                'dtp':None,    
                                'fs_supt':20,
                                'chain_alpha':0.2,
                                'temps':None,
                                'name':'plot_posteriors',
                                'function':super_plots,
                                'nice_name':'Plotting Posterior Scatter Plot',                         
                                }
        
        self.plot_histograms = {'axis_fs':18,
                                'title_fs':24,
                                'temps':None,
                                'name':'plot_histograms',
                                'function':plot_histograms,
                                'nice_name':'Plotting Histograms Plot',
                                }
        
        self.plot_keplerian_model = {'hist':True,
                                     'paper_mode':True,
                                     'uncertain':False,
                                     'errors':True,
                                     'periodogram':True,
                                     'gC':0,
                                     'use_c':False,
                                     'celerite':False,
                                     'axhline_kwargs':axhline_kwargs,
                                     'errorbar_kwargs':errorbar_kwargs,
                                     'fonts':fonts_kwargs,
                                     'name':'plot_KeplerianModel',
                                     'function':plot_KeplerianModel,
                                     'nice_name':'Plotting Keplerian Models',
                                    }

        self.plot_betas = {'title_fs':24,
                           'xaxis_fs':18,
                           'yaxis_fs':18,
                           'name':'plot_betas',
                           'function':plot_betas,
                           'nice_name':'Plotting E[log L](beta) Plot',
                           
                           }

        self.plot_rates = {'title_fs':24,
                           'xaxis_fs':18,
                           'yaxis_fs':18,
                           'window':1,
                           'name':'plot_rates',
                           'function':plot_rates,
                           'nice_name':'Plotting Temperature Rates',
                           
                           }

        self.plot_beta_density = {'name':'plot_beta_density',
                                  'function':plot_beta_density,
                                  'nice_name':'Plotting Beta Density',
                                  }

        self.plot_trace = {'modes':[0, 1, 2, 3],
                           'temps':None,
                           'name':'plot_trace',
                           'function':plot_trace2,
                           'nice_name':'PLOT ARVIZ',
                           }

        self.plot_periodogram = {'name':'plot_periodogram',
                                }
        
        self.plot_gaussian_mixtures = {
                                       'sig_factor':4,
                                       'plot_title':None,
                                       'plot_ylabel':None,
                                       'fill_cor':0,
                                       'plot_name':'',
                                       'temps':None,
                                       'name':'plot_GM_Estimator',
                                       'format':'png',
                                     }

        self.plot_all_list = [self.plot_posteriors,
                              self.plot_histograms,
                              self.plot_keplerian_model,
                              self.plot_periodogram,
                              self.plot_betas,  # 4
                              self.plot_beta_density,
                              self.plot_rates,
                              self.plot_gaussian_mixtures,
                              self.plot_trace,
                              ]

        self.parameter_histograms = False
        self.corner = (np.array(self.plot_trace['modes']) == 3).any()

        self.save_chains = None #  [0]
        self.save_likelihoods = [0]
        self.save_posteriors = [0]


    def set_engine(self, eng):
        setattr(self, f'{eng}_config', {'name':eng})

        if eng == 'emcee':
            import emcee
            self.engine__ = emcee

        elif eng == 'dynesty':
            import dynesty
            self.engine__ = dynesty
            self.engine_config = {'setup':np.array([]),
                                  'dlogz_init':0.05}

        elif eng == 'dynesty_dynamic':
            import dynesty
            self.engine__ = dynesty
            self.engine__args = 'dynamic'
            self.general_dependencies.append('dynesty')
            self.engine_config = {'setup':np.array([]),
                                  'dlogz_init':0.05}

        elif eng == 'pymc3':
            import pymc3 as pm
            self.engine__ = pm
 
        elif eng == 'reddemcee':
            import reddemcee
            self.engine__ = reddemcee
            self.general_dependencies.extend(['reddemcee', 'emcee', 'logging'])
            if self.FPTS:
                self.general_dependencies.extend(['astroemperor.fpts as fpts'])


            self.engine_config = {'setup':np.array([5, 100, 500, 2]),
                                  'ntemps':5,
                                  'betas':None,
                                  'adaptative':True,
                                  'moves':None,
                                  'config_adaptation_halflife':1000,
                                  'config_adaptation_rate':100,
                                  'config_adaptation_decay':0,
                                  }

            self.run_config = {'adapt_burnin':None,
                               'discard':0.5,
                               'thin':1,
                               'logger_level':'CRITICAL',
                               'burnin0':0,
                               'burnin0_runs':1,
                               }


        else:
            raise Exception(self.logger('Failed to set engine properly. Try a string!', center=True, c='red'))


    def load_data(self, folder_name):
        self.starname = folder_name
        self.data_wrapper = DataWrapper(folder_name, read_loc=self.read_loc)
        self.logger('\n')
        #self.logger(self.data_wrapper.add_all__(), center=True, c='blue')
        for m in self.data_wrapper:
            if m['use']:
                self.logger(m['logger_msg'], center=True, c='blue')
        self.logger('\n')

        dw_labels = self.data_wrapper['RV']['RV_labels']
        if self.instrument_names is None:
            self.instrument_names = dw_labels

        self.my_data = self.data_wrapper.get_data__()
        self.my_data_common_t = self.data_wrapper['RV']['common_t']

        #self.my_data_reduc = self.my_data.values[:, 0:3].T
        if len(dw_labels) > 0:
            self.nins__ = len(dw_labels)
            self.switch_RV = True

        self.cornums = self.data_wrapper['RV']['nsai']
        if self.switch_SA:
            if np.sum(self.cornums) > 0:
                pass
        else:
            self.my_data = self.my_data[['BJD', 'RV', 'eRV', 'Flag']]
            self.cornums = [0 for j in self.cornums]


        if self.data_wrapper['AM']['use']:
            self.switch_AM = True
            self.switch_inclination = True


    def add_keplerian_block(self):
        self.model_dependencies.append('kepler')
        self.kplanets__ += 1
        if self.switch_inclination:
            kb = mk_AstrometryKeplerianBlock(self.my_data,
                                             parameterisation=self.keplerian_parameterisation,
                                             number=self.kplanets__)
        else:
            kb = mk_KeplerianBlock(self.my_data,
                                   parameterisation=self.keplerian_parameterisation,
                                   number=self.kplanets__)


        prargs = [self.eccentricity_limits, self.eccentricity_prargs]
        kw = {'prargs':prargs, 'dynamics':self.switch_dynamics,
              'kplan':self.kplanets__, 'starmass':self.starmass,
              'dynamics_already_included':self.dynamics_already_included}
        SmartLimits(self.my_data, kb, **kw)  # sets limits and returns <extra_priors>


        kb.signal_number = self.kplanets__

        self.blocks__.insert((self.kplanets__ - 1), kb)

        msg = '{} {}, {}'.format(colored(kb.type_, 'green', attrs=['bold']),
                                 colored('block added', 'green'),
                                 colored(kb.name_, 'green'))
        msg = f'                              {msg}'
        self.logger(msg, center=True)
        self.logger('\n')


    def add_offset_block(self):
        ib = mk_OffsetBlock(self.my_data, nins=self.nins__)
        kw = {}
        SmartLimits(self.my_data, ib, **kw)
        self.blocks__.append(ib)
        msg = '{} {}, {}'.format(colored(ib.type_, 'green', attrs=['bold']),
                                    colored('block added', 'green'),
                                    colored(ib.name_, 'green'))
        msg = f'                              {msg}'
        self.logger(msg, center=True)
        self.logger('\n')


    def add_offset_am_block(self):
        ib = mk_AstrometryOffsetBlock(self.my_data, nins=1)
        kw = {}
        SmartLimits(self.my_data, ib, **kw)
        self.blocks__.append(ib)
        msg = '{} {}, {}'.format(colored(ib.type_, 'green', attrs=['bold']),
                                    colored('block added', 'green'),
                                    colored(ib.name_, 'green'))
        msg = f'                              {msg}'
        self.logger(msg, center=True)
        self.logger('\n')

        pass


    def add_sai_block(self):
        ib = mk_SAIBlock(self.my_data, nins=self.nins__, sa=self.cornums)
        ib.cornums = self.cornums
        kw = {}
        SmartLimits(self.my_data, ib, **kw)
        self.blocks__.append(ib)
        msg = '{} {}, {}'.format(colored(ib.type_, 'green', attrs=['bold']),
                                    colored('block added', 'green'),
                                    colored(ib.name_, 'green'))
        msg = f'                              {msg}'
        self.logger(msg, center=True)
        self.logger('\n')


    def add_jitter_block(self):
        ib = mk_JitterBlock(self.my_data, nins=self.nins__)
        kw = {}

        jitter_args = [self.jitter_limits, self.jitter_prargs]
        SmartLimits(self.my_data, ib, *jitter_args)
        self.blocks__.append(ib)
        msg = '{} {}, {}'.format(colored(ib.type_, 'green', attrs=['bold']),
                                    colored('block added', 'green'),
                                    colored(ib.name_, 'green'))
        msg = f'                              {msg}'
        self.logger(msg, center=True)
        self.logger('\n')


    def add_jitter_am_block(self):
        ib = mk_AstrometryJitterBlock(self.my_data, nins=1)
        kw = {}

        SmartLimits(self.my_data, ib, **kw)
        self.blocks__.append(ib)
        msg = '{} {}, {}'.format(colored(ib.type_, 'green', attrs=['bold']),
                                    colored('block added', 'green'),
                                    colored(ib.name_, 'green'))
        msg = f'                              {msg}'
        self.logger(msg, center=True)
        self.logger('\n')


    def add_acceleration_block(self):
        ab = mk_AccelerationBlock(self.my_data, accel=self.acceleration)
        self.blocks__.append(ab)

        msg = '{} {}, {}'.format(colored(ab.type_, 'green', attrs=['bold']),
                                 colored('block added', 'green'),
                                 colored(ab.name_, 'green'))
        msg = f'                              {msg}'
        self.logger(msg, center=True)
        self.logger('\n')


    def add_moav_block(self):
        ib = mk_MOAVBlock(self.my_data,
                          nins=self.nins__,
                          moav_args=self.moav)
        kw = {}

        SmartLimits(self.my_data, ib, **kw)
        self.blocks__.append(ib)
        msg = '{} {}, {}'.format(colored(ib.type_, 'green', attrs=['bold']),
                                    colored('block added', 'green'),
                                    colored(ib.name_, 'green'))
        msg = f'                              {msg}'
        self.logger(msg, center=True)
        self.logger('\n')


    def add_celerite_block(self):
        self.model_dependencies.append('celerite2')
        self.model_dependencies.append('celerite2.terms as cterms')
        self.plot_keplerian_model['celerite'] = True
        self.write_kernel()
        self.write_kernel(in_func=True)


        kw = self.my_kernel

        ib = mk_CeleriteBlock(self.my_data, nins=self.nins__, my_kernel=kw)

        cele_args = []

        SmartLimits(self.my_data, ib, *cele_args, **kw)
        self.blocks__.append(ib)
        msg = '{} {}, {}'.format(colored(ib.type_, 'green', attrs=['bold']),
                                    colored('block added', 'green'),
                                    colored(ib.name_, 'green'))
        msg = f'                              {msg}'
        self.logger(msg, center=True)
        self.logger('\n')
        '''
        '''


    def update_model(self):
        self.model = ReddModel(self.my_data, self)
        self.model.instrument_names = self.instrument_names
        self.model.refresh__()

        if self.switch_AM:
            self.model.switch_AM = True
            self.model.data_wrapper = self.data_wrapper


        self.model_constants['A_'] = f'{self.model.A_}'
        self.model_constants['mod_fixed_'] = f'{self.model.mod_fixed}'
        self.model_constants['cornums'] = f'{self.cornums}'


    def run(self, progress=True):
        time_run_init = time.time()
        
        ### assert errors!
        if self.constrain_method == 'GM':
            if not self.gaussian_mixtures_fit:
                msg = 'Invalid constrain_method = GM with .gaussian_mixtures_fit = False'
                raise SyntaxError(msg)        
        
        # PRE-CLEAN
        self.sampler = None

        self.debug_snapshot()
        self.debug_msg(f'run  : begin | {time.time()-self.time_init}')

        self.apply_conditions()
        self.saveplace = ensure_dir(self.starname, loc=self.save_loc, k=self.kplanets__, first=self.switch_first)

        # Merge general plot dict with each 
        for pi in range(len(self.plot_all_list)):
            self.plot_all_list[pi]['saveloc'] = self.saveplace
            self.plot_all_list[pi] = {**self.plot_all, **self.plot_all_list[pi]}

        self.temp_script = 'temp_script.py'

        self.update_model()

        self.prerun_logger()

        self.debug_msg(f'run  : init sampler | {time.time()-self.time_init}')

        if self.engine__.__name__ == 'reddemcee':
            
            # Constants for run
            setup = self.engine_config['setup']
            ntemps, nwalkers, nsweeps, nsteps = setup

            # Assert
            assert self.nins__ == len(self.instrument_names), f'instrument_names should have {self.nins__} items'
            if self.engine_config['betas'] is not None:
                assert len(self.engine_config['betas']) == setup[0], f'betas should have {setup[0]} items'
            # Write Script            
            self.debug_msg(f'run  : Write_script() | {time.time()-self.time_init}')
            self.write_script()
            self.debug_msg(f'run  : os <run temp_script.py> | {time.time()-self.time_init}')

            # Run Script
            self.logger('Generating Samples', center=True, c='green', attrs=['reverse'])
            self.logger.line()

            os.system(f'ipython {self.temp_script}')
            
            # RESTORE SAMPLER
            self.load_sampler_meta(target_dir='', setup=setup)


        if self.engine__.__name__ == 'dynesty':
            # TRANSFORM TO PTFORMARGS
            # This is missing additional_parameters ! !

            self.ptformargs0 = []
            for b in self:
                for p in b:
                    if p.fixed == None:
                        if p.ptformargs == None:
                            l, h = p.limits
                            p.ptformargs = [(h-l)/2., (h+l)/2.]
                            if b.parameterisation == 1:
                                if p.name[:3] == 'Ecc' and p.ptformargs[0] > 0.707:
                                    p.ptformargs[0] = 0.707
                            if b.parameterisation == 3:
                                if p.name[:3] == 'Ecc' and p.ptformargs[0] > 0.707:
                                    p.ptformargs[0] = 0.707
                        else:
                            s, c = p.ptformargs
                            p.limits = [c-s, c+s]
                        self.ptformargs0.append(p.ptformargs)


            if self.engine__args == 'dynamic':
                # SET SETUP
                nlive0, nlive_batch0 = setup
                # SET SAMPLER
                self.write_script()

                # RUN SAMPLER
                os.system('ipython {}'.format(self.temp_script))

                with open('sampler_pickle.pkl', 'rb') as sampler_metadata:
                    self.sampler_metadata_dict = pickle.load(sampler_metadata)
                os.system(f'mv sampler_pickle.pkl {self.saveplace}/restore/sampler_pickle.pkl')
            else:
                # SET SETUP
                # SET SAMPLER
                # RUN SAMPLER
                pass

        self.time_run = time.time() - time_run_init

        self.debug_snapshot()

        gc.collect()


    def run_auto(self, k_start=0, k_end=10, progress=True):
        self.debug_msg(f'run_auto : INIT run_auto | {time.time()-self.time_init}')
        self.auto_setup = self.engine_config['setup']

        ## Assert this is a valid run
        valid_engines = ['emcee', 'dynesty', 'pymc3', 'reddemcee']
        assert self.engine__.__name__ in valid_engines, f'Selected engine not valid!'
        assert k_start <= k_end, f'Invalid keplerian starting point: ({k_start}, {k_end})'

        ## Add blocks
        if True:
            if self.switch_RV:
                self.add_offset_block()
                if self.switch_SA and np.sum(self.cornums) > 0:
                    self.add_sai_block()

                if self.switch_first and k_start > 0:
                    for _ in range(k_start):
                        self.add_keplerian_block()

            if self.acceleration:
                self.add_acceleration_block()

            if self.switch_jitter:
                self.add_jitter_block()

            if self.moav['order']:
                self.add_moav_block()

            if self.switch_celerite:
                self.add_celerite_block()
                pass

            if self.switch_AM:
                self.add_offset_am_block()
                self.add_jitter_am_block()

        # make the run
        if True:
            while k_start <= k_end:
                # Init Stat Holders
                if True:
                    oldlike_max = self.like_max
                    oldpost_max = self.post_max
                    oldchi2 = self.chi2
                    oldAIC = self.AIC
                    oldBIC = self.BIC
                    oldDIC = self.DIC
                    oldHQIC = self.HQIC
                    oldRMSE = self.RMSE
                    oldRMSi = self.RMSi
                    oldWeights = self.Weights
                    oldBayesFactor = self.BayesFactor


                self.run(progress=progress)
                self.postprocess()  # change values
                #self.run_plot_routines()

                k_start += 1


                if self.switch_constrain:
                    if self.constrain_method == 'sigma':
                        for b in self:
                            if b.type_ == 'Keplerian':
                                for p in b:
                                    if p.fixed is None:
                                        pval = p.value
                                        psig = p.sigma

                                        limf = pval - self.constrain_sigma*psig
                                        limc = pval + self.constrain_sigma*psig


                                        if limc > p.limits[1]:
                                            limc = p.limits[1]

                                        if psig / abs(pval) < 1e-5:
                                            self.add_condition([p.name, 'fixed', pval])
                                        elif (limf > p.limits[0] and limc < p.limits[1]):
                                            self.add_condition([p.name, 'limits', [limf, limc]])
                                        elif limf > p.limits[0]:
                                            self.add_condition([p.name, 'limits', [limf, p.limits[1]]])
                                        elif limc < p.limits[1]:
                                            self.add_condition([p.name, 'limits', [p.limits[0], limc]])
                    if self.constrain_method == 'GM':
                        count = 0
                        for b in self:
                            if b.type_ == 'Keplerian':
                                for p in b[b.C_]:
                                    if p.GM_parameter.n_components == 1:
                                        prarg0 = [p.GM_parameter.means[0], p.GM_parameter.sigmas[0]]

                                        self.add_condition([p.name, 'prior', 'Normal'])
                                        self.add_condition([p.name, 'prargs', prarg0])

                                    elif p.GM_parameter.n_components > 1:
                                        self.add_condition([p.name, 'prior', 'GaussianMixture'])
                                        self.add_condition([p.name, 'prargs', [self.model.C_[count]]])
                                    count += 1

                if True:
                    run_metadata = {}

                    # to restore, we need
                    # my_data
                    # model? .evaluate_model?!?!?!
                    # ymod, ferr2, residuals

                    with open(f'{self.saveplace}/restore/run_pickle.pkl','wb') as md_save:
                        pickle.dump(run_metadata, md_save)


                # if not continue, model selec
                if self.ModelSelection.compare(self.BIC, oldBIC):
                    #oldbic_display = np.round(oldBIC, 3)
                    #newbic_display = np.round(self.BIC, 3)
                    self.logger('\nBIC condition met!!', c='blue', attrs=['bold'])
                    self.logger('\npresent BIC < past BIC - 5', c='blue')
                    self.logger(self.ModelSelection.msg, c='blue')
                else:
                    self.logger('\nBIC condition not met', c='blue')
                    self.logger('\npresent BIC < past BIC - 5', c='blue')
                    self.logger(self.ModelSelection.msg, c='blue')

                    self.logger.end_run_message()
                    break

                if k_start > k_end:
                    self.logger.end_run_message()
                    break

                

                
                # if passes:
                self.logger.next_run_message()
                if self.switch_RV:
                    self.add_keplerian_block()
                self.update_model()


    def postprocess(self):
        time_postprocess_init = time.time()

        self.debug_msg(f'postprocess() : INIT | {time.time()-self.time_init}')

        if self.engine__.__name__ == 'reddemcee':
            ntemps, nwalkers, nsweeps, nsteps = self.engine_config['setup']
            niter = int(nsteps * nsweeps)

            # BURN-IN config
            if True:
                self.reddemcee_discard = int(self.run_config['discard']*niter)
                self.reddemcee_thin = self.run_config['thin']

                reddemcee_dict = {'discard':self.reddemcee_discard,
                                  'thin':self.reddemcee_thin,
                                  'flat':True}

            if not self.FPTS:
                raw_chain0 = self.sampler.get_func('get_chain', kwargs=reddemcee_dict)
                raw_likes0 = self.sampler.get_func('get_blobs', kwargs=reddemcee_dict)
                raw_posts0 = self.sampler.get_func('get_log_prob', kwargs=reddemcee_dict)

                self.autocorr_time = self.sampler.get_autocorr_time(quiet=True,
                                                    tol=0,
                                                    discard=self.reddemcee_discard,
                                                    thin=self.reddemcee_thin)


            else:
                with open('sampler_flatchain.pkl', 'rb') as y:
                    raw_chain = pickle.load(y)
                raw_chain0 = list(raw_chain[:, self.reddemcee_discard:])
                with open('sampler_flatlogl.pkl', 'rb') as y:
                    raw_likes0 = pickle.load(y)
                raw_likes0 = list(np.array([raw_likes0[i].flatten(order='F')[self.reddemcee_discard:] for i in range(ntemps)]))
                with open('sampler_flatlogp.pkl', 'rb') as y:
                    raw_posts0 = pickle.load(y)
                raw_posts0 = list(np.array([raw_posts0[i].flatten(order='F')[self.reddemcee_discard:] for i in range(ntemps)]))

                os.system(f'mv sampler_flatchain.pkl {self.saveplace}/restore/sampler_flatchain.pkl')
                os.system(f'mv sampler_flatlogl.pkl {self.saveplace}/restore/sampler_flatlogl.pkl')
                os.system(f'mv sampler_flatlogp.pkl {self.saveplace}/restore/sampler_flatlogp.pkl')


            if self.cherry['cherry']:
                raw_chain = np.empty(ntemps, dtype=object)
                raw_likes = np.empty(ntemps, dtype=object)
                raw_posts = np.empty(ntemps, dtype=object)

                for t in range(ntemps):
                    if self.cherry['median']:
                        mask = raw_posts0[t] > np.median(raw_posts0[t])
                    elif self.cherry['diff']:
                        mask = max(raw_posts0[t]) - raw_posts0[t] <= self.cherry['diff']

                    raw_chain[t] = raw_chain0[t][mask]
                    raw_likes[t] = raw_likes0[t][mask]
                    raw_posts[t] = raw_posts0[t][mask]

            else:
                raw_chain = raw_chain0
                raw_likes = raw_likes0
                raw_posts = raw_posts0

            setup_info = 'Temperatures, Walkers, Sweeps, Steps   : '
            size_info = [len(raw_chain[t]) for t in range(ntemps)]


            if self.switch_evidence:
                if self.FPTS:
                    #self.evidence = self.sampler.
                    self.evidence = self.sampler_metadata_dict['thermodynamic_integration']
                else:
                    zaux = self.sampler.thermodynamic_integration(discard=self.reddemcee_discard)
                    self.evidence = zaux[0], zaux[1]

            self.best_loc_post = np.argmax(raw_posts[0])
            self.best_loc_like = np.argmax(raw_likes[0])

            self.post_max = raw_posts[0][self.best_loc_post]
            self.like_max = raw_likes[0][self.best_loc_post]
            self.prior_max_post = self.post_max - raw_likes[0][self.best_loc_post]

            self.sigmas = np.std(raw_chain[0], axis=0)

            self.fit_max = raw_chain[0][self.best_loc_post]
            self.fit_mean = np.mean(raw_chain[0], axis=0)
            self.fit_median = np.median(raw_chain[0], axis=0)

            self.fit_low1, self.fit_high1 = np.percentile(raw_chain[0], find_confidence_intervals(1), axis=0)
            self.fit_low2, self.fit_high2 = np.percentile(raw_chain[0], find_confidence_intervals(2), axis=0)
            self.fit_low3, self.fit_high3 = np.percentile(raw_chain[0], find_confidence_intervals(3), axis=0)

            self.fit_maxlike = raw_chain[0][self.best_loc_like]

            if self.use_fit == 'max_post':
                self.ajuste = self.fit_max
            elif self.use_fit == 'max_like':
                self.ajuste = self.fit_maxlike
            elif self.use_fit == 'mean':
                self.ajuste = self.fit_mean
            elif self.use_fit == 'median':
                self.ajuste = self.fit_median
            else:
                self.logger(f'Input error in use_fit = {self.use_fit}')
                self.ajuste = self.fit_max

        if self.engine__.__name__ == 'dynesty':
            # FAILSAFE CONFIGS
            self.plot_betas['plot'] = False
            self.plot_rates['plot'] = False
            self.plot_posteriors['chain_alpha'] = 1.0

            '''
            needs to save likes
            '''
            results = self.sampler_metadata_dict['results']
            self.sampler = [results]
            setup_info = '\nLive Points                       : '
            size_info = results['niter']

            raw_chain = np.array([results['samples']])
            raw_likes = [results['logl']]
            if self.switch_evidence:
                self.evidence = (results['logz'][-1], results['logzerr'][-1])

            raw_posts = raw_likes - self.evidence[0]
            ntemps = 1

            ## FIN
            ###### THIS CAN BE SHARED
            self.best_loc_post = np.argmax(raw_posts[0])
            self.best_loc_like = np.argmax(raw_likes[0])

            self.post_max = raw_posts[0][self.best_loc_post]
            self.like_max = raw_likes[0][self.best_loc_post]
            self.prior_max_post = 0.0  # calculate properly

            self.sigmas = np.std(raw_chain[0], axis=0)

            self.fit_max = raw_chain[0][self.best_loc_post]
            self.fit_mean = np.mean(raw_chain[0], axis=0)
            self.fit_median = np.median(raw_chain[0], axis=0)

            self.fit_low1, self.fit_high1 = np.percentile(raw_chain[0], find_confidence_intervals(1), axis=0)
            self.fit_low2, self.fit_high2 = np.percentile(raw_chain[0], find_confidence_intervals(2), axis=0)
            self.fit_low3, self.fit_high3 = np.percentile(raw_chain[0], find_confidence_intervals(3), axis=0)

            self.fit_maxlike = raw_chain[0][self.best_loc_like]


            if self.use_fit == 'max_post':
                self.ajuste = self.fit_max
            elif self.use_fit == 'max_like':
                self.ajuste = self.fit_maxlike
            elif self.use_fit == 'mean':
                self.ajuste = self.fit_mean
            elif self.use_fit == 'median':
                self.ajuste = self.fit_median
            else:
                self.logger(f'Input error in use_fit = {self.use_fit}')
                self.ajuste = self.fit_max

            print('\n\n------------ Dynesty Summary -----------\n\n')
            print(str(results.summary()))

        chains = raw_chain
        posts = raw_posts
        likes = raw_likes

        if True:
            self.ch = chains
            self.pt = posts
            self.lk = likes

        self.dic_aux = np.var(-2 * likes[0])

        ###########################################
        # GET STATS
        ###########################################
        self.debug_msg(f'postprocess() : GET_STATS | {time.time()-self.time_init}')

        if True:
            if self.switch_RV:
                ## not ajuste, but whole
                ## will fail with fixed stuff?
                ymod, err2 = self.temp_model_func(self.ajuste)
                residuals = self.my_data['RV'].values - ymod

                res_table = self.my_data.values.T
                res_table[1] = residuals


                np.savetxt(f'{self.saveplace}/restore/residuals.dat',
                            res_table)

                self.calculate_statistics(residuals, err2)

        ## SET P.VALUES P.SIGMA
        if True:
            ## **RED w/ set_attr
            j = 0  # delete this ap1
            for b in self:
                for p in b:
                    if p.fixed is None:
                        self._update_p_free(p, j)
                        j += 1
                    else:
                        self._update_p_fixed(p)


        # Get extra info. Parameter transformation and planet signatures
        if True:
            self.sma = []
            self.mm = []

            self.sma_sig = []
            self.mm_sig = []

            extra_names = []
            extra_chains = []


            # Get extra chains
            uptothisdim = 0
            for b in self:
                chain_counter = 0
                ## PLANET SIGNATURES
                if b.type_ == 'Keplerian':
                    my_params = [None, None, None, None, None]
                    if b.astrometry_bool:
                        my_params.extend([None, None])

                    for i in b.C_:
                        my_params[i] = chains[0].T[uptothisdim + chain_counter]
                        chain_counter += 1

                    for i in b.A_:
                        my_params[i] = b[i].fixed * np.ones(len(chains[0]))


                    if True:
                        if b.parameterisation == 0:
                            if b.astrometry_bool:
                                per, A, phase, ecc, w, inc, Ome = b.get_attr('value')
                                per_, A_, phase_, ecc_, w_, inc_, Ome_ = my_params
                            else:
                                per, A, phase, ecc, w = b.get_attr('value')
                                per_, A_, phase_, ecc_, w_ = my_params

                        elif b.parameterisation == 1:
                            P, As, Ac, S, C = b.get_attr('value')
                            P_, As_, Ac_, S_, C_ = my_params

                            per = np.exp(P)
                            per_ = np.exp(P_)

                            A, phase = delinearize(As, Ac)
                            ecc, w = delinearize(S, C)

                            A_, phase_ = adelinearize(As_, Ac_)
                            ecc_, w_ = adelinearize(S_, C_)

                            for thingy in ['Period', 'Amplitude', 'Phase', 'Eccentricity', 'Longitude_Periastron']:
                                extra_names.append(thingy+'_{}'.format(b.number_))

                            for thingy in [per_, A_, phase_, ecc_, w_]:
                                extra_chains.append(thingy)

                        elif b.parameterisation == 2:
                            per, A, t0, ecc, w = b.get_attr('value')
                            per_, A_, t0_, ecc_, w_ = my_params

                        elif b.parameterisation == 3:
                            per, A, t0, S, C = b.get_attr('value')
                            per_, A_, t0_, S_, C_ = my_params

                            ecc, w = delinearize(S, C)
                            ecc_, w_ = adelinearize(S_, C_)

                            for thingy in ['Eccentricity', 'Longitude_Periastron']:
                                extra_names.append(thingy)
                            for thingy in [ecc_, w_]:
                                extra_chains.append(thingy)

                        if self.starmass:
                            sma, mm = cps(per, A, ecc, self.starmass)
                            sma_, mm_ = cps(per_, A_, ecc_, self.starmass)

                            if b.astrometry_bool:
                                mm = mm/np.sin(inc)
                                mm_ = mm_/np.sin(inc_)

                            self.sma_sig.append(sma)
                            self.mm_sig.append(mm)

                            self.sma_sig.append(np.std(sma_))
                            self.mm_sig.append(np.std(mm_))

                            extra_names.append('Semi-Major Axis [AU]')
                            extra_names.append('Minimum Mass [M_J]')
                            extra_chains.append(sma_)
                            extra_chains.append(mm_)

                uptothisdim += b.ndim_
            ## Set p.values and sigma for extra params
            if True:
                ## **RED w/ set_attr\
                jj = 0
                for b in self:
                    for p in b.additional_parameters:
                        if p.has_posterior:
                            ch = extra_chains[jj]
                            self._update_p_extra(p, ch)
                            jj += 1

        # SAVE STUFF??
        self.debug_msg(f'postprocess() : SAVE STUFF | {time.time()-self.time_init}')


        if self.save_chains is not None:
            self.save_chain(chains)
            self.save_posterior(posts)
            self.save_loglikelihood(likes)

        self.update_model()

        self.debug_msg(f'postprocess() : gaussian_mixtures_fit() | {time.time()-self.time_init}')

        # SET GM
        if self.gaussian_mixtures_fit:
            self.logger('\n')
            self.logger('Calculating Gaussian Mixtures', center=True, c='green')
            self.set_gaussian_mixtures(chains[0], extra_chains)


        self.debug_msg(f'postprocess() : SET POSTERIORS | {time.time()-self.time_init}')


        # SET POSTERIORS
        if True:
            if self.posterior_fit_method == 'GM':
                for b in self:
                    for p in b[b.C_]:
                        p.posterior = p.GM_parameter
                        if p.GM_parameter.n_components == 1:
                            mu = np.round(p.GM_parameter.mixture_mean, 3)
                            sig = np.round(p.GM_parameter.mixture_sigma, 3)
                            p.display_posterior = f'~𝓝 ({mu}, {sig})'
                        elif p.GM_parameter.n_components > 1:
                            subscript_nums = ['', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
                            mu = np.round(p.GM_parameter.mixture_mean, 3)
                            sig = np.round(p.GM_parameter.mixture_sigma, 3)

                            p.display_posterior = '𝛴{}~~𝓝 ({}, {})'.format(subscript_nums[p.GM_parameter.n_components],
                                                                    mu, sig)
                        else:
                            print('Something really weird is going on! Error 110.')
                    for p in b[b.A_]:
                        p.posterior = p.GM_parameter
                        p.display_posterior = '~𝛿 (x - {})'.format(p.value)

                    jj = 0
                    for p in b.additional_parameters:
                        if p.has_posterior:
                            p.posterior = p.GM_parameter

                            if p.GM_parameter.n_components == 1:
                                mu = np.round(p.GM_parameter.mixture_mean, 3)
                                sig = np.round(p.GM_parameter.mixture_sigma, 3)
                                p.display_posterior = '~𝓝 ({}, {})'.format(mu, sig)
                            elif p.GM_parameter.n_components > 1:
                                subscript_nums = ['', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
                                mu = np.round(p.GM_parameter.mixture_mean, 3)
                                sig = np.round(p.GM_parameter.mixture_sigma, 3)

                                p.display_posterior = '𝛴{}~~𝓝 ({}, {})'.format(subscript_nums[p.GM_parameter.n_components],
                                                                        mu, sig)
                            else:
                                print('Something really weird is going on! Error 110.')

                        pass

        self.debug_msg(f'postprocess() : PRINT POSTERIORS | {time.time()-self.time_init}')
        # PRINT POSTERIORS
        if True:
            self.logger.line()
            self.logger('~~ Best Fit ~~', center=True, c='yellow', attrs=['bold', 'reverse'])
            self.logger.line()


            tab_3 = np.array([])
            switch_title = True

            for b in self:
                to_get = ['name',
                          'value_max',
                          'value_range',
                          'display_prior',
                          'limits']
                
                to_tab0 = b.get_attr(to_get)

                if len(b.additional_parameters):
                    mask = [x.has_posterior for x in b.additional_parameters]
                    for p in np.array(b.additional_parameters)[mask]:
                        p.display_prior = ''
                        for j in range(len(to_get)):
                            to_tab0[j] += [getattr(p, to_get[j])]

                to_tab0[2] = (np.array(to_tab0[2]).T - np.array(to_tab0[1])).T

                to_tab0 = adjust_table_tex(to_tab0, rounder=self.rounder_display)


                to_tab = list(zip(*to_tab0))

                df0 = pd.DataFrame(to_tab, columns=to_get)


                if switch_title:
                    headers1 = ['Parameter',
                                'Value (max)',
                                'Range (-+ sig)',
                                'Prior',
                                'Limits',
                                ]
                    switch_title = False
                else:
                    headers1 = [' ' * len(s) for s in headers1]

                self.logger(tabulate(df0,
                                     headers=headers1,
                                     showindex=False))


        # PRINT STATS
        if True:
            self.logger('\n\n')
            self.logger('~~ Run Info ~~', center=True, c='yellow', attrs=['bold', 'reverse'])
            self.logger('\n\n')

            tabh_1 = ['Info                             ', 'Value                       ']

            tab_1 =    [['Star Name                      : ', self.starname],
                        ['The sample sizes are           : ', size_info],
                        [setup_info, self.auto_setup.tolist()],
                        ['Model used is                  : ', str(self)],
                        ['N data                         : ', self.model.ndata],
                        ['Number of Dimensions           : ', len(self)],
                        ['Degrees of Freedom             : ', self.dof]
                        ]

            self.logger('\n\n')
            self.logger(tabulate(tab_1, headers=tabh_1))
            self.logger('\n')
            self.logger('---------------------------------------------------', center=True)
            self.logger('\n')


            tabh_2 = ['Statistic                        ', 'Value']

            tab_2 = [['The maximum posterior is    :    ', '{:.3f}'.format(self.post_max)],
                     ['The maximum likelihood is   :    ', '{:.3f}'.format(self.like_max)],
                     ['The BIC is                  :    ', '{:.3f}'.format(self.BIC)],
                     ['The AIC is                  :    ', '{:.3f}'.format(self.AIC)],
                     ['The DIC is                  :    ', '{:.3f}'.format(self.DIC)],
                     ['The HQIC is                 :    ', '{:.3f}'.format(self.HQIC)],
                     ['The Bayes Factor is         :    ', '{:.3f}'.format(self.BayesFactor)],
                     ['The chi2 is                 :    ', '{:.3f}'.format(self.chi2)],
                     ['The reduced chi2 is         :    ', '{:.3f}'.format(self.chi2_red)],
                     ['The RMSE is                 :    ', '{:.3f}'.format(self.RMSE)],
                     ['The RMSi is                 :    ', f'{np.round(self.RMSi, 3)}'],
                     ['The Weights are             :    ', f'{np.round(self.Weights, 3)}'],
                     ]

            if self.engine__.__name__ == 'reddemcee':
                adapt_ts = self.engine_config['config_adaptation_halflife']
                adapt_ra = self.engine_config['config_adaptation_rate']
                adapt_sc = self.engine_config['config_adaptation_decay']
                #af = np.mean(self.sampler.acceptance_fraction, axis=1)

                # should go on ln909? very beg of postprocess
                reddemcee_adapt = self.run_config['adapt_burnin']
                af = self.sampler.get_acceptance_fraction(nsteps=self.engine_config['setup'][-1],
                                                          discard=reddemcee_adapt)

                self.logger('\nDecay Timescale, Rate, Scheme   :   ' + f'{adapt_ts}, {adapt_ra}, {adapt_sc}')
                self.logger('\nBeta Detail                     :   ' + '[' + ', '.join('{:.4}'.format(x) for x in self.sampler.betas) + ']', save_extra_n=True)
                self.logger('\nMean Logl Detail                :   ' + '[' + ', '.join('{:.3f}'.format(np.mean(x)) for x in likes) + ']', save_extra_n=True)
                self.logger('\nTemperature Swap                :   ' + '[' + ', '.join('{:.3f}'.format(x) for x in self.sampler.ratios) + ']', save_extra_n=True)
                self.logger('\nMean Acceptance Fraction        :   ' + '[' + ', '.join('{:.3f}'.format(x) for x in af) + ']', save_extra_n=True)
                self.logger('\nAutocorrelation Time            :   ' + '[' + ', '.join('{:.3f}'.format(x) for x in self.autocorr_time[0]) + ']', save_extra_n=True)


                if self.switch_evidence:
                    x = [['The evidence is             :    ', '%.3f +- %.3f' % self.evidence]]
                    tab_2 = np.vstack([x, tab_2])

            if self.engine__.__name__ == 'dynesty':
                if self.switch_evidence:
                    x = [['The evidence is             :    ', '%.3f +- %.3f' % self.evidence]]
                    tab_2 = np.vstack([x, tab_2])

            self.logger('\n\n')
            self.logger('~~ Statistical Details ~~', center=True, c='yellow', attrs=['bold', 'reverse'])
            self.logger('\n\n')
            self.logger(tabulate(tab_2, headers=tabh_2))
            self.logger('\n\n')

            if self.save_log_simple:
                stats_log = np.array(['logZ',
                                      'logP',
                                      'logL',
                                      'BIC ',
                                      'AIC ',
                                      'X²  ',
                                      'X²v ',
                                      'RMSE',
                                      'RMSi',
                                      'Weights',
                                      ])
                
                stats_log = np.vstack([stats_log,
                                       ['%.3f +- %.3f' % self.evidence,
                                       '{:.3f}'.format(self.post_max),
                                       '{:.3f}'.format(self.like_max),
                                       '{:.3f}'.format(self.BIC),
                                       '{:.3f}'.format(self.AIC),
                                       '{:.3f}'.format(self.chi2),
                                       '{:.3f}'.format(self.chi2_red),
                                       '{:.3f}'.format(self.RMSE),
                                       f'{np.round(self.RMSi, 3)}',
                                       f'{np.round(self.Weights, 3)}',
                                       ],
                                       ])
                                      
                np.savetxt(f'{self.saveplace}/tables/stats.dat', stats_log.T, fmt='%s', delimiter='\t')

        # SAVE CHAIN SUMMARY
        if True:
            cs_names = self.get_attr_param('name', flat=True)
            max_length = max(len(item) for item in cs_names)
            cs_names = [item.ljust(max_length) for item in cs_names]

            cs = [cs_names,
                np.round(self.get_attr_param('value_max_lk', flat=True), 8),
                np.round(self.get_attr_param('value_max', flat=True), 8),
                np.round(self.get_attr_param('value_mean', flat=True), 8),
                np.round(self.get_attr_param('sigma', flat=True), 8),
                np.round(self.get_attr_param('value_low3', flat=True), 8),
                np.round(self.get_attr_param('value_low2', flat=True), 8),
                np.round(self.get_attr_param('value_low1', flat=True), 8),
                np.round(self.get_attr_param('value_median', flat=True), 8),
                np.round(self.get_attr_param('value_high1', flat=True), 8),
                np.round(self.get_attr_param('value_high2', flat=True), 8),
                np.round(self.get_attr_param('value_high3', flat=True), 8),
                ]
            
            cs_header=['Parameter',
                       'MaxLogl  ',
                       'MaxPost  ',
                       'Mean     ',
                       'Std      ',
                       '00.27    ',
                       '04.55    ',
                       '31.73    ',
                       'Median   ',
                       '68.26    ',
                       '95.44    ',
                       '99.73    ',
                       ]
            
            np.savetxt(f'{self.saveplace}/tables/chain_summary.dat',
                        np.vstack([cs_header, np.array(cs).T]),
                        fmt='%s',
                        delimiter='\t')
            
        # SAVE TEX 
        if True:
            par_box_names = self.get_attr_param('name', flat=True)
            
            v0 = np.array(self.get_attr_param('value_range', flat=True))[:, 0]
            v1 = self.get_attr_param('value_max', flat=True)
            v2 = np.array(self.get_attr_param('value_range', flat=True))[:, 1]

            par_box = [par_box_names,
                       v0,
                       v1,
                       v2,
                       ]
            
            par_box = adjust_table_tex(par_box, rounder=8)

            pb_header = ['Parameter',
                         'lower    ',
                         'value    ',
                         'higher   ']
            
            df0 = pd.DataFrame(np.array(par_box).T, columns=pb_header)

            df0.to_csv(f'{self.saveplace}/tables/param_minimal.dat',
                       sep='\t',
                       index=False)


            df0['value    '] = np.round(v1, self.rounder_math)
            df0['higher'] = np.round(v2-v1, self.rounder_math)
            df0['lower'] = np.round(v1-v0, self.rounder_math)

            df0['Value'] = df0.apply(lambda row: f"${row['value    ']}^{{+{row['higher']}}}_{{-{row['lower']}}}$", axis=1)


            df_latex = df0[['Parameter', 'Value']]
            
            latex_table = df_latex.to_latex(index=False, escape=False)

            with open(f'{self.saveplace}/tables/values.tex', 'w') as f:
                f.write(latex_table)

    
        #######################
        # PLOT GM PER PARAMETER

        self.debug_snapshot()

        self.debug_msg(f'postprocess() : run_plot_routines | {time.time()-self.time_init}')

        self.time_postprocess = time.time() - time_postprocess_init

        self.run_plot_routines(chains, likes, posts)

        self.debug_snapshot()

        # Time Table
        if True:
            self.logger(f'\nTime Table', save_extra_n=True)
            self.logger(f'Time RUN             :  {sec_to_clock(self.time_run)}', save_extra_n=True)
            self.logger(f'Time POSTPROCESS     :  {sec_to_clock(self.time_postprocess)}', save_extra_n=True)
            self.logger(f'Time CALCULATE GM    :  {sec_to_clock(self.time_calc_gm)}', save_extra_n=True)
            
            self.logger(f'Time Plot model      :  {sec_to_clock(self.time_plot_keplerian)}', save_extra_n=True)
            self.logger(f'Time Plot posteriors :  {sec_to_clock(self.time_plot_posteriors)}', save_extra_n=True)
            self.logger(f'Time Plot histograms :  {sec_to_clock(self.time_plot_histograms)}', save_extra_n=True)
            self.logger(f'Time Plot betas      :  {sec_to_clock(self.time_plot_betas)}', save_extra_n=True)
            self.logger(f'Time Plot arviz      :  {sec_to_clock(self.time_plot_trace)}', save_extra_n=True)
            self.logger(f'Time Plot GM         :  {sec_to_clock(self.time_plot_gm)}', save_extra_n=True)

        # SAVE LOG
        if self.save_log:
            self.logger.saveto(self.saveplace)

        if self.save_log_simple:
            simple_log = np.array([flatten(self.model.get_attr_param('name'))])
            simple_log = np.vstack([simple_log,
                                    ['{:.3f}'.format(x) for x in np.array(flatten(self.model.get_attr_param('value')))]
                                    ])

            np.savetxt(f'{self.saveplace}/best_fit.dat', simple_log.T, fmt='%s', delimiter='\t')


        self.clean_run()


    def run_plot_routines(self, chains, likes, posts):
        for pi in self.plot_all_list:
            if (pi['name'] == 'plot_posteriors' or
                pi['name'] == 'plot_histograms'):
                if pi['temps'] is None:
                    pi['temps'] = np.arange(self.auto_setup[0])

            if pi['name'] == 'plot_trace':
                if self.FPTS:
                    pi['plot'] = False
                else:
                    pi['temps'] = 0
                    pi['burnin'] = self.reddemcee_discard
                    pi['thin'] = self.reddemcee_thin

        res_max = flatten(self.get_attr_param('value_max'))


        self.plot_gaussian_mixtures['name'] = 'plot_GM_Estimator'
        self.plot_periodogram['name'] = 'plot_periodogram'

        
        my_plot_kwargs = {
            'plot_posteriors':{'chains':chains,
                               'posts':posts,
                               #'options':self.plot_posteriors,
                               'my_model':self.model,
                               'ncores':self.cores__,
                               },
            'plot_histograms':{'chains':chains,
                               'posts':posts,
                               #'options':self.plot_histograms,
                               'my_model':self.model,
                               'ncores':self.cores__,
                               },
            'plot_KeplerianModel':{'my_data':self.my_data,
                                   'my_model':self.model,
                                   'res':res_max,
                                   'common_t':self.my_data_common_t,
                                   #'options':self.plot_keplerian_model,
                                   },
            'plot_betas':{'betas':self.sampler.betas_history,
                          'logls':np.array([np.mean(likes[t]) for t in range(self.auto_setup[0])]),
                          'Z':self.evidence,
                          #'options':self.plot_betas,
                          'setup':self.auto_setup},

            'plot_rates':{'bh':self.sampler.betas_history,
                          'rh':self.sampler.ratios_history,
                          'afh':self.sampler.af_history,
                          'setup':self.auto_setup,
                          #'options':self.plot_rates,
                          },
            'plot_trace':{'sampler':self.sampler[0],
                          'eng_name':self.engine__.__name__,
                          'my_model':self.model,
                          },
            'plot_beta_density':{'betas':self.sampler.betas},

                          }


        for plot_func in self.plot_all_list:
            time_plot_init = time.time()
            plot_name = plot_func['name']
            if ((True) and
                (plot_name != 'plot_GM_Estimator') and
                (plot_name != 'plot_periodogram') and True):
                #(plot_name != 'plot_posteriors')):
                #plot_name != 'plot_trace'
                if plot_func['plot']:
                    self.debug_msg(f'run_plot_routines() : {plot_name} | {time.time()-self.time_init}')

                    self.logger('\n')
                    self.logger(plot_func['nice_name'], center=True, c='green')                    

                    plot_func['function'](options=plot_func,
                                          **my_plot_kwargs[plot_name])

                plot_func['time_to_plot'] = time.time() - time_plot_init    

                pass

        time_plot_init = time.time()
        
        self.time_plot_posteriors = self.plot_all_list[0]['time_to_plot']
        self.time_plot_histograms = self.plot_all_list[1]['time_to_plot']
        self.time_plot_keplerian = self.plot_all_list[2]['time_to_plot']
        self.time_plot_betas = self.plot_all_list[4]['time_to_plot']
        # 5 plot_beta_density
        self.time_plot_rates = self.plot_all_list[6]['time_to_plot']
        # 7 plot_gaussian_mixtures
        self.time_plot_trace = self.plot_all_list[8]['time_to_plot']



        if self.plot_all_list[7]['plot']:
            if self.gaussian_mixtures_fit:
                self.debug_msg(f'postprocess() : PLOT GM | {time.time()-self.time_init}')

                self.logger('\n')
                self.logger('Plotting Gaussian Mixtures', center=True, c='green')
                pbar_tot = self.model.ndim__
                pbar = tqdm(total=pbar_tot)
                for b in self:
                    self.plot_all_list[7]['fill_cor'] = b.bnumber_-1
                    for p in b[b.C_]:
                        self.plot_all_list[7]['plot_name'] = f'{b.bnumber_} {p.GM_parameter.name}'

                        plot_GM_Estimator(p.GM_parameter,
                                          options=self.plot_all_list[7])

                        pbar.update(1)
                    for p in b.additional_parameters:
                        if p.has_posterior:
                            self.plot_all_list[7]['plot_name'] = f'{b.bnumber_} {p.GM_parameter.name}'
                            plot_GM_Estimator(p.GM_parameter,
                                            options=self.plot_all_list[7])
                pbar.close()

        self.time_plot_gm = time.time() - time_plot_init
        gc.collect()


    def set_gaussian_mixtures(self, cold_chain, extra_chains):
        time_gm_init = time.time()

        pbartot = sum(b.ndim_ + len(b.additional_parameters) for b in self)
        pbar = tqdm(total=pbartot)
        
        for b in self:
            for p in b:
                if p.cpointer is not None:
                    p.GM_parameter = GM_Estimator().estimate(cold_chain[:, p.cpointer],
                                                            p.name, p.unit)
                    pbar.update(1)
                else:
                    p.GM_parameter = p.value
                
        # ADDITIONAL PARAMETERS
        for b in self:
            jj = 0
            for p in b.additional_parameters:
                if p.has_posterior:
                    ch = extra_chains[jj]
                    p.GM_parameter = GM_Estimator().estimate(ch, p.display_name, p.unit)
                    jj += 1
                    pbar.update(1)
            pass
        pbar.close()
        self.time_calc_gm = time.time() - time_gm_init
        
        pass


    def calculate_statistics(self, residuals, err2):
        self.dof = self.model.ndata - self.model.ndim__
        self.chi2 = np.sum(residuals**2 / err2)
        self.chi2_red = np.sum(residuals**2 / err2) / self.dof
        self.RMSE = np.sqrt(np.sum(residuals ** 2) / self.model.ndata)

        FLAGS_ = self.my_data.Flag
        self.RMSi = [np.sqrt(np.sum(residuals[FLAGS_==(i+1)] ** 2) / len(residuals[FLAGS_==(i+1)])) for i in range(self.nins__)]
        self.Weights = [len(FLAGS_==(i+1)) / self.RMSi[i]**2 for i in range(self.nins__)]
        self.Total_Weight = np.sum(self.Weights)
        self.Weights = self.Weights / self.Total_Weight

        self.AIC = 2 * self.model.ndim__ - 2 * self.like_max
        self.BIC = np.log(self.model.ndata) * self.model.ndim__ - 2 * self.like_max

        self.DIC = -2 * self.temp_like_func(self.fit_mean) + self.dic_aux

        self.post_true = self.post_max - self.evidence[0]
        self.BayesFactor = self.like_max - self.evidence[0]

        self.HQIC = 2 * self.model.ndim__ * np.log(np.log(self.model.ndata)) - 2 * self.like_max


    def prerun_logger(self):
        if self.switch_first:
            self.logger('~~ Setup Info ~~', center=True, c='blue', attrs=['reverse'])
            self.logger('\nCurrent Engine is            '+colored(self.engine__.__name__+' '+self.engine__.__version__, attrs=['bold']), c='blue')
            self.logger('Number of cores is           '+colored(self.cores__, attrs=['bold']), c='blue')
            self.logger('Save location is             '+colored(self.saveplace, attrs=['bold']), c='blue')

            if self.switch_dynamics:
                dyn_crit = 'Hill Stability'
            else:
                dyn_crit = 'None'
            self.logger('Dynamical Criteria is        '+colored(dyn_crit, attrs=['bold']), c='blue')
            self.logger('Posterior fit method is      '+colored(self.posterior_dict[self.posterior_fit_method], attrs=['bold']), c='blue')
            self.logger('Limits constrain method is   '+colored(self.constrain_method, attrs=['bold']), c='blue')
            self.logger('Model Selection method is    '+colored(self.ModelSelection.criteria, attrs=['bold']), c='blue')

            self.logger.line()
            self.logger('~~ Automatically Saving ~~', center=True, c='blue', attrs=['reverse'])

            
            saving0_ = ['Logger       ',
                        'Samples      ',
                        'Posteriors   ',
                        'Likelihoods  ',
                        'Plots: Posteriors           ',
                        'Plots: Keplerian Model      ',
                        'Plots: Gaussian Mixture     ',
                        'Plots: Parameter Histograms ',
                        'Plots: Corner               ']
            
            
            # self.plot_all_list
            checks0 = [self.save_log,
                       self.save_chains,
                       self.save_posteriors,
                       self.save_likelihoods,
                       self.plot_all_list[0]['plot'],  # posteriors
                       self.plot_all_list[2]['plot'],  # kep model
                       self.plot_all_list[7]['plot'],  # GM
                       self.plot_all_list[1]['plot'],  # histograms
                       self.corner
                       ]

            for i in range(len(checks0)):
                self.logger.check(saving0_[i], checks0[i])

            self.switch_first = False


        # Print Pre-Run
        if True:
            self.logger.line()
            self.logger('~~ Pre-Run Info ~~', center=True, c='yellow', attrs=['bold', 'reverse'])
            self.logger.line()

            tab_3 = np.array([])
            switch_title = True

            for b in self:
                to_tab0 = b.get_attr(['name', 'display_prior', 'limits'])
                to_tab0[2] = np.round(to_tab0[2], 3)
                to_tab = list(zip(*to_tab0))

                if switch_title:
                    self.logger(tabulate(to_tab,
                                          headers=['Parameter       ',
                                                   'Prior   ',
                                                   'Limits      ',
                                                   ]))
                    switch_title = False

                else:
                    self.logger(tabulate(to_tab,
                                          headers=['                ',
                                                   '        ',
                                                   '            ',
                                                   ]))

            self.logger.line()
            for b in self:
                self.logger('Math for {}:'.format(b.name_), c='yellow')
                self.logger('{}'.format(b.math_display_), center=True, c='yellow')

        pass


    def get_attr_param(self, call, flat=False, asarray=False):
        ret = [b.get_attr(call) for b in self]
        if flat:
            ret = flatten(ret)
        if asarray:
            ret = np.array(ret)
        return ret


    def get_attr_block(self, call, flat=False, asarray=False):
        ret = [getattr(b, call) for b in self]
        if flat:
            ret = flatten(ret)
        if asarray:
            ret = np.array(ret)
        return ret


    def save_chain(self, chains):
        temps = self.save_chains
        if self.engine__.__name__ == 'reddemcee':
            for temp in temps:
                np.savez_compressed(f'{self.saveplace}/samples/chains/chain_{str(temp)}', chains[temp])


    def save_posterior(self, posts):
        temps = self.save_posteriors
        if self.engine__.__name__ == 'reddemcee':
            for temp in temps:
                np.savez_compressed(
                    f'{self.saveplace}/samples/posteriors/posterior_{str(temp)}',
                    posts[temp],
                )


    def save_loglikelihood(self, likes):
        temps = self.save_likelihoods
        if self.engine__.__name__ == 'reddemcee':
            for temp in temps:
                np.savez_compressed(self.saveplace + '/samples/likelihoods/likelihood_'+str(temp), likes[temp])


    def apply_conditions(self):
        applied = []
        for b in self:
            for p in b:
                for c in self.conds:
                    if p.name == c[0]:
                        setattr(p, c[1], c[2])
                        if c not in applied:
                            applied.append(c)
                            msg = '\nCondition applied: Parameter {} attribute {} set to {}'.format(
                                        colored(c[0], attrs=['underline']),
                                        colored(c[1], attrs=['underline']),
                                        colored(c[2], attrs=['underline']))
                            self.logger(msg)

        self.logger.line()
        applied.sort()
        applied = [applied for applied,_ in itertools.groupby(applied)]

        for ap in applied[::-1]:
            self.conds.remove(ap)


    def add_condition(self, cond):
        self.conds.append(cond)


    def write_kernel(self, in_func=False):
        nterms = len(self.my_kernel['terms'])
        sumeq = '='
        param_counter = 0



        if in_func:
            sumeq = '='
            tail = '01'
            with open(f'temp_kernel{tail}', 'w') as f:
                for termi in range(nterms):
                    line_str = f'kernel {sumeq} cterms.'
                    term_name = self.my_kernel['terms'][termi]
                    param_dict = self.my_kernel['params'][termi]
                    c = 0

                    line_str += term_name
                    term_inputs = ''
                    for k in param_dict:
                        if param_dict[k] == None:
                            term_inputs += f'{k} = theta_gp[{c}], '
                            c += 1
                        else:
                            term_inputs += f'{k} = {param_dict[k]}, '
                    f.write(f'''
    {line_str}({term_inputs})
''')

                    sumeq = '+='
                f.write('''
    gp_.kernel = kernel

''')


        else:
            sumeq = '='
            tail = '00'
            with open(f'temp_kernel{tail}', 'w') as f:
                for termi in range(nterms):
                    line_str = f'kernel {sumeq} cterms.'
                    term_name = self.my_kernel['terms'][termi]
                    param_dict = self.my_kernel['params'][termi]

                    line_str += term_name
                    term_inputs = ''
                    for k in param_dict:
                        if param_dict[k] == None:
                            term_inputs += f'{k} = 1.0, '
                        else:
                            term_inputs += f'{k} = {param_dict[k]}, '
                    f.write(f'''
{line_str}({term_inputs})
''')

                    sumeq = '+='


    def write_script(self):
        # clean dependencies
        self.general_dependencies = np.unique(self.general_dependencies).tolist()
        self.model_dependencies = np.unique(self.model_dependencies).tolist()
        self.debug_msg(f'write_script() : INIT | {time.time()-self.time_init}')
        self.dir_work = os.path.dirname(os.path.realpath(__file__))
        self.dir_save = self.saveplace

        self.script_pool_opt = get_support(f'pools/0{self.multiprocess_method}.pool')

        ## GENERAL NEW
        ## START SCRIPT
        self.debug_msg(f'write_script() : open(temp_script.py, w) | {time.time()-self.time_init}')

        with open(self.temp_script, 'w') as f:
            f.write(open(get_support('init.scr')).read())
            self.debug_script(f,
f'''import time
debug_timer = {self.time_init}
print('temp_script.py   : INIT | ', time.time()-debug_timer)''')

            ## DEPENDENCIES
            for d in self.general_dependencies:
                f.write(f'''
import {d}
''')
            ## MODEL DEPENDENCIES
            for d in self.model_dependencies:
                f.write(f'''
import {d}
''')

            ## LOGGER
            if self.engine__.__name__ == 'reddemcee':
                f.write('''
logging.getLogger('emcee').setLevel('{}')
'''.format(self.run_config['logger_level']))


            ## CONSTANTS
            for c in self.model_constants:
                f.write(f'''{c} = {self.model_constants[c]}
''')


            ## MODEL
            f.write(open(self.model.write_model_(loc=self.saveplace)).read())

            ## LIKELIHOOD
            if self.switch_AM:
                f.write(open(get_support('likelihoods/a00.like')).read())

            else:
                if self.switch_celerite:
                    # KERNEL SETUP
                    #f.write(open(get_support('kernels/00.kernel')).read())

                    # LIKELIHOOD
                    f.write(open(get_support('likelihoods/02.like')).read())
                else:
                    f.write(open(get_support('likelihoods/00.like')).read())

            ## PRIOR & PTFORMARGS
            if self.debug_mode:
                f.write(f'''
print('temp_script.py   : prior_script.read() | ', time.time()-debug_timer)
''')

            if self.engine__.__name__ == 'reddemcee':
                for prior_script in os.listdir(get_support('priors')):
                    f.write(open(get_support(f'priors/{prior_script}')).read())
                    f.write('''
''')

                count = 0
                for b in self:
                    for p in b[b.C_]:
                        first = True
                        attributes = ['weights_',
                                      'means_',
                                      'covariances_',
                                      'precisions_',
                                      'precisions_cholesky_',
                                      'converged_',
                                      'n_iter_',
                                      'lower_bound_',
                                      'n_features_in_',
                                      ]
                        if p.prior == 'GaussianMixture':
                            if first:
                                f.write('''

from sklearn.mixture import GaussianMixture as skl_gm

''')
                            first = False
                            f.write('''
{0} = skl_gm()
gaussian_mixture_objects[{1}] = {0}
'''.format('gaussian_mixture_{}'.format(self.model.C_[count]), self.model.C_[count]))

                            for at in attributes:
                                coso = getattr(p.GM_parameter, at)
                                if type(coso) == np.ndarray:
                                    coso = coso.tolist()
                                    f.write('''
setattr(gaussian_mixture_objects[{0}], '{1}', np.array({2}))'''.format(self.model.C_[count], at, coso))

                                else:
                                    f.write('''
setattr(gaussian_mixture_objects[{0}], '{1}', {2})'''.format(self.model.C_[count], at, coso))


                        count += 1

                f.write('''

def my_prior(theta):
    for a in A_:
        theta = np.insert(theta, a, mod_fixed_[a])
    lp = 0.
''')
                count = 0
                for b in self:
                    for p in b[b.C_]:
                        f.write(f'''
    lp += {p.prior}(theta[{self.model.C_[count]}], {p.limits}, {p.prargs})
''')
                        count += 1

                    f.write('''
    if lp == -np.inf:
        return lp

''')

                    if len(b.additional_parameters):
                        for p in b.additional_parameters:
                            if p.has_prior:
                                if p.name == 'Amplitude':
                                    f.write('''
    x = theta[{0}][1]**2 + theta[{0}][2]**2
    '''.format(b.slice))

                                elif p.name == 'Eccentricity':
                                    f.write('''
    x = theta[{0}][3]**2 + theta[{0}][4]**2
    '''.format(b.slice))

                                elif p.name == 'Hill' and p.prargs[0] > 1:
                                    f.write('''
    b_len = {}
    kplanets = {}
    '''.format(len(b)   , p.prargs[0]))
                                    f.write(open(get_support(f'PAE/0{b.parameterisation}.pae')).read())

                                else:
                                    continue

                                f.write('''
    lp += {}(x, {}, {})
'''.format(p.prior, p.limits, p.prargs))

                        f.write('''
    if lp == -np.inf:
        return lp

''')

                f.write('''

    return lp


''')

            if self.engine__.__name__ == 'dynesty':
                # this could go on constants
                f.write(f'''
ptformargs = {self.ptformargs0}
''')

                f.write('''
def my_prior(theta):
#    for a in A_:
#        theta = np.insert(theta, a, mod_fixed_[a])
    x = np.array(theta)
    for i in range(len(x)):
        a, b = ptformargs[i]
        x[i] =  a * (2. * x[i] - 1) + b
    return x

''')

                # this could go on constants
                f.write(f'''

nlive, nlive_batch = {self.auto_setup[0]}, {self.auto_setup[1]}
ndim = {self.model.ndim__}
''')

            ## MULTIPROCESSING
            if self.debug_mode:
                f.write(f'''
print('write_script() : import multiprocessing pool | ', time.time()-debug_timer)
''')
                

            if self.engine__.__name__ == 'reddemcee':
                f.write(open(self.script_pool_opt).read().format(self.cores__))

            if self.engine__.__name__ == 'dynesty':
                pool_bool = False
                if self.multiprocess_method:
                    pool_bool = True

            ## MORE CONSTANTS
            if self.engine__.__name__ == 'reddemcee':
                f.write(f'''

ntemps, nwalkers, nsweeps, nsteps = {self.auto_setup[0]}, {self.auto_setup[1]}, {self.auto_setup[2]}, {self.auto_setup[3]}
setup = ntemps, nwalkers, nsweeps, nsteps
ndim = {self.model.ndim__}
''')

            ## BACKENDS
            if self.engine__.__name__ == 'reddemcee':
                if not self.FPTS:
                    if self.engine_config['betas'] is not None:
                        aux_betas_str = f'np.array({str(list(self.engine_config["betas"]))})'
                    else:
                        aux_betas_str = 'None'
                    f.write(f'''
betas = {aux_betas_str}
backends = [None for _ in range(ntemps)]
for t in range(ntemps):
    filename = '{self.saveplace}/restore/backends/emperor_backend_' +str(t)+'.h5'
    backends[t] = emcee.backends.HDFBackend(filename)
    backends[t].reset(nwalkers, ndim)

sampler = reddemcee.PTSampler(nwalkers,
                             ndim,
                             my_likelihood,
                             my_prior,
                             ntemps=ntemps,
                             pool=mypool,
                             backend=backends,
                             betas=betas,
                             adaptative={str(self.engine_config["adaptative"])},
                             config_adaptation_halflife={self.engine_config["config_adaptation_halflife"]},
                             config_adaptation_rate={self.engine_config["config_adaptation_rate"]},
                             config_adaptation_decay={self.engine_config["config_adaptation_decay"]},
                             )

''')
                else:
                    f.write(f'''
sampler = fpts.Sampler(nwalkers, ndim,
                             my_likelihood,
                             my_prior,
                             ntemps=ntemps,
                             threads={self.cores__},
                             pool=mypool)

sampler0 = reddemcee.PTSampler(nwalkers, ndim,
                             my_likelihood,
                             my_prior,
                             ntemps=ntemps,
                             adaptative={str(self.engine_config["adaptative"])},
                             config_adaptation_halflife={self.engine_config["config_adaptation_halflife"]},
                             config_adaptation_rate={self.engine_config["config_adaptation_rate"]},
                             config_adaptation_decay={self.engine_config["config_adaptation_decay"]},
''')
            ## SETUP POS0
            if self.engine__.__name__ == 'reddemcee':
                pos0_bool = 'False'
                if self.debug_mode:
                    pos0_bool = 'True'
                    f.write(f'''
print('temp_script.py   : set_init()  pos0 | ', time.time()-debug_timer)
''')

                # POS 0
                f.write('''
def set_init():
    pos = np.zeros((ntemps, nwalkers, ndim))

    for t in range(ntemps):
        j = 0
''')

                for b in self.model:
                    for p in b:
                        if p.fixed == None:
                            r = 1
                            if p.is_hou:
                                r = 0.707
                            f.write('''
        m = ({0} + {1}) / 2
        r = ({0} - {1}) / 2 * {2}
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1

'''.format(p.limits[1], p.limits[0], r))

                if self.FPTS:
                    f.write('''
    return pos


''')
                else:
                    f.write('''
    return list(pos)


''')

                # TEST P0
                f.write('''
def test_init(max_repeats={0}):
    p0 = set_init()

    is_bad_position = True
    repeat_number = 0

    while is_bad_position and (repeat_number < max_repeats):
        is_bad_position = False
        for t in range(ntemps):
            for n in range(nwalkers):
                position_evaluated = p0[t][n]
                if my_prior(position_evaluated) == -np.inf:
                    is_bad_position = True
                    p0[t][n] = set_init()[t][n]
        repeat_number += 1
        if {1}:
            print('test_init ', repeat_number)

    if is_bad_position:
        print('COULDNT FIND VALID INITIAL POSITION')
    return p0

'''.format(100, pos0_bool))

                if self.debug_mode:
                    f.write(f'''
print('temp_script.py   : test_init()  pos0 | ', time.time()-debug_timer)
''')
                f.write('''

p1 = test_init()

''')

            # MAKE THE RUN
            if self.debug_mode:
                f.write(f'''
print('temp_script.py   : run __main__ | ', time.time()-debug_timer)
''')

            if self.engine__.__name__ == 'reddemcee':
                if not self.FPTS:
                    if self.run_config['adapt_burnin']:
                        f.write(f'''
adapt_burnin = {self.run_config['adapt_burnin']}
''')
                        f.write(open(get_support('endit2.scr')).read())
                    else:
                        f.write(open(get_support('endit.scr')).read())
                else:
                    f.write(open(get_support('endit_fpts.scr')).read())

            if self.engine__.__name__ == 'dynesty':
                nested_dict = self.dynesty_config
                nested_args = ''
                for key in [*self.engine_config]:
                    nested_args += f'{key} = {self.engine_config[key]}, '

                f.write(open(get_support('endit_dyn.scr')).read().format(pool_bool, self.cores__, nested_args))

            # SAVE PICKLE
            f.write(open(get_support('save_metadata.scr')).read().format(f'{self.saveplace}/restore/'))
            # __NAME__==
            f.write(open(get_support('run_main.scr')).read())
            # load just the model into emperor

        self.debug_msg(f'write_script() : reloads into emp| {time.time()-self.time_init}')

        import temp_script
        temp_script = reload(temp_script)


        self.temp_model_func = temp_script.my_model
        self.temp_like_func = temp_script.my_likelihood
        self.temp_prior_func = temp_script.my_prior

        self.debug_msg(f'write_script() : END | {time.time()-self.time_init}')


    def load_sampler_meta(self, target_dir='', setup=[]):
        from emcee.backends import HDFBackend

        ntemps, nwalkers, nsweeps, nsteps = setup

        restore_path = f'{self.saveplace}/restore/'
        #os.system(f'mv sampler_pickle.pkl {self.saveplace}/restore/sampler_pickle.pkl')
        

        # Restore Sampler
        self.sampler = self.engine__.PTSampler(nwalkers, self.model.ndim__,
                                     self.temp_like_func,
                                     self.temp_prior_func,
                                     logl_args=[], logl_kwargs={},
                                     logp_args=[], logp_kwargs={},
                                     ntemps=ntemps,
                                     pool=None,
                                     )
        
        

        with open(f'{restore_path}sampler_pickle.pkl', 'rb') as sampler_metadata:
            self.sampler_metadata_dict = pickle.load(sampler_metadata)

        for attr, value in self.sampler_metadata_dict.items():
            setattr(self.sampler, attr, value)

        self.engine_config["betas"] = self.sampler.betas

        
        #if not target_dir:
        #    target_dir = self.saveplace+'/restore/'
        #filename = f'{target_dir}sampler_pickle.pkl'

        #with open(filename, 'rb') as sampler_metadata:
        #    metadata_dict = pickle.load(sampler_metadata)

        #for attr, value in metadata_dict.items():
        #    setattr(self.sampler, attr, value)

        if not self.FPTS:
            for t in range(ntemps):
                loc_t = '{}emperor_backend_{}.h5'.format(self.saveplace+'/restore/backends/', t)
                self.sampler[t] = HDFBackend(loc_t)
        else:
            print('Method not implemented yet!')


    def clean_run(self):
        self.debug_msg(f'clean_run() : CLEANING.. | {time.time()-self.time_init}')

        os.system(f'mv {self.temp_script} {self.saveplace}/temp/{self.temp_script}')
        gc.collect()
        pass


    def _update_p_extra(self, p, ch):
        p.value = ch[self.best_loc_post]  # self.use_fit
        p.sigma = np.std(ch)
        p.sigma_frac_mean = 0

        p.value_max = ch[self.best_loc_post]
        p.value_mean = np.mean(ch)
        p.value_median = np.median(ch)

        p.value_max_lk = ch[self.best_loc_like]

        p.value_low1, p.value_high1 = hdi_of_samples(ch, 0.36)
        p.value_low2, p.value_high2 = hdi_of_samples(ch, 0.90)
        p.value_low3, p.value_high3 = hdi_of_samples(ch, 0.95)

        if p.value_low1 < p.value_max:
            a = p.value_low1
        else:
            a = p.value_max - p.sigma

        if p.value_high1 > p.value_max:
            b = p.value_high1
        else:
            b = p.value_max + p.sigma

        if a < p.limits[0]:
            a = p.limits[0]
        if b > p.limits[1]:
            b = p.limits[1]

        p.value_range = [a, b]


    def _update_p_free(self, p, j):
        p.value = self.ajuste[j]
        p.sigma = self.sigmas[j]
        p.sigma_frac_mean = 0

        p.value_max = self.fit_max[j]
        p.value_mean = self.fit_mean[j]
        p.value_median = self.fit_median[j]

        p.value_max_lk = self.fit_maxlike[j]

        p.value_low1, p.value_high1 = self.fit_low1[j], self.fit_high1[j]
        p.value_low2, p.value_high2 = self.fit_low2[j], self.fit_high2[j]
        p.value_low3, p.value_high3 = self.fit_low3[j], self.fit_high3[j]

        if p.value_low1 < p.value_max:
            a = p.value_low1
        else:
            a = p.value_max - p.sigma

        if p.value_high1 > p.value_max:
            b = p.value_high1
        else:
            b = p.value_max + p.sigma

        if a < p.limits[0]:
            a = p.limits[0]
        if b > p.limits[1]:
            b = p.limits[1]

        p.value_range = [a, b]


    def _update_p_fixed(self, p):
        p.sigma = np.nan
        p.value_max = p.value
        p.value_mean = p.value
        p.value_median = p.value

        p.value_max_lk = p.value

        p.value_low1, p.value_high1 = np.nan, np.nan
        p.value_low2, p.value_high2 = np.nan, np.nan
        p.value_low3, p.value_high3 = np.nan, np.nan
        p.value_range = [np.nan, np.nan]        
        pass


    def debug_script(self, f, msg):
        if self.debug_mode:
            f.write(f'''
{msg}
''')


    def debug_snapshot(self):
        if self.debug_mode:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            print("----[ Top 30 ]----")
            for stat in top_stats[:30]:
                print(stat)
            print("----[ Top 30 ]----")
        else:
            pass
        
    
    def debug_msg(self, msg):
        if self.debug_mode:
            print(msg)


    def __getitem__(self, n):
        return self.blocks__[n]


    def __repr__(self):
        #return self.model
        return str([self.blocks__[i].name_ for i in range(len(self.blocks__))])


    def __len__(self):
        return np.sum([len(b) for b in self])

#
'''
TODO

PERFORMANCE
---------------
P0 -> set an init option, ball around value
constrain_sigma -> find a better way, cuts off from k1 to k2


PLOTS
---------------
trace       -> corner without burnin
rc.colors   -> cyclers and rc.hl



ORDER
---------------
move self.model_constants to model


READ
---------------
Phase to M0?


OUTDATED
---------------
plot_trace2

'''
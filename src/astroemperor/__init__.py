# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.7.1
# date 17 jan 2023

__version__ = '0.7.7'
__name__ = 'astroemperor'
__all__ = ['support']
# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT
# **FIN  : Finish this

# sourcery skip: remove-redundant-if
if True:
    import numpy as np
    import pandas as pd
    
    import os
    import pickle

    import itertools
    import multiprocessing

    from tabulate import tabulate
    from termcolor import colored
    from tqdm import tqdm


    from .utils import *
    from .canvas import *
    from .block import ReddModel
    from .block_repo import *

    try:
        terminal_width = os.get_terminal_size().columns
    except:
        print('I couldnt grab the terminal size! Trying with pandas...')
        terminal_width = pd.get_option('display.width')

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
                     'like_max':None, 'BayesFactor':self.foobarmax}
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

    def compare(self, foo, bar):
        return self.compare_f(foo, bar)

    def update(self):
        self.compare_f = self.dict[self._current_criteria]


class Simulation(object):
    def __init__(self, setup=None):
        if setup is None:
            setup = []
        # LOAD ATTRIBUTES
        self.blocks__ = []
        self.ndim__ = None
        self.model = []

        self.kplanets__ = 0
        self.nins__ = 0

        self.logger = reddlog()
        self.cores__ = multiprocessing.cpu_count() - 1

        self.save_loc = ''
        self.read_loc = ''

        self.starmass = 1.
        Nonethings = ['starname', 'betas', 'saveplace']
        for c in Nonethings:
            setattr(self, c, None)

        switches = ['switch_first', 'switch_RV', 'switch_SA', 'switch_constrain',
                    'switch_dynamics', 'dynamics_already_included']
        for switch in switches:
            setattr(self, switch, False)

        switches_T = ['switch_evidence', 'gaussian_mixtures_fit']
        for switch in switches_T:
            setattr(self, switch, True)


        self.conds = []

        self.eccentricity_limits = [0, 1]
        self.eccentricity_prargs = [0, 0.1]

        self.jitter_limits = [0, 1]
        self.jitter_prargs = [5, 5]


        self.multiprocess_method = 1

        # constrain
        self.constrain_sigma = 3
        self.constrain_method = 'sigma'  # 'sigma', 'GM'

        # posterior
        self.posterior_fit_method = 'GM'  # KDE*soon

        self.posterior_dict = {'GM': 'Gaussian Mixtures',
                               'KDE': 'Kernel Density Estimation'}
        # stats
        stats_names_posi = ['chi2', 'chi2_red', 'AIC', 'BIC',
                            'DIC', 'HQIC', 'RMSE']
        for stat in stats_names_posi:
            setattr(self, stat, np.inf)

        stats_names_nega = ['post_max', 'like_max', 'BayesFactor']
        for stat in stats_names_nega:
            setattr(self, stat, -np.inf)


        # Writing stuff
        self.general_dependencies = []
        self.dynesty_config = {'dlogz_init':0.05,
                                }
        self.reddemcee_config = {'burnin':'half',
                                'thinby':1,
                                }

        #self.ModelSelection = 'BIC'
        self.ModelSelection = ModelSelectionObj('BIC')
        self.evidence = 0, 0


        self.save_all = False
        self.save_plots = False
        self.save_log = True
        self.save_plots_fmt = 'png'

        # plots
        self.instrument_names = None


        self.plot_gaussian_mixtures = {'plot':True,
                                        'sig_factor':4,
                                        'plot_title':None,
                                        'plot_ylabel':None}
        #self.plot_gaussian_mixtures = True
        self.plot_keplerian_model = {'plot':True,
                                     'hist':True,
                                     'uncertain':True,
                                     'errors':False,
                                     'format':'png',
                                     'logger_level':'ERROR',
                                     'gC':0}

        self.plot_trace = {'plot':True,
                           'modes':[0, 1, 2, 3],
                           'temp':0}

        self.parameter_histograms = False
        self.corner = (np.array(self.plot_trace['modes']) == 3).any()

        self.save_chains = [0]
        self.save_likelihoods = [0]
        self.save_posteriors = [0]
        self.logger('   ', center=True, save=False, c='green', attrs=['bold', 'reverse'])
        self.logger('~~ Simulation Successfully Initialized ~~', center=True, save=False, c='green', attrs=['bold', 'reverse'])
        self.logger('   ', center=True, save=False, c='green', attrs=['bold', 'reverse'])


    def set_engine(self, eng):
        setattr(self, f'{eng}_config', {'name':eng})

        if eng == 'emcee':
            import emcee
            self.engine__ = emcee
        elif eng == 'dynesty':
            import dynesty
            self.engine__ = dynesty

        elif eng == 'dynesty_dynamic':
            import dynesty
            self.engine__ = dynesty
            self.engine__args = 'dynamic'
            self.general_dependencies.append('dynesty')

            self.dynesty_config['dlogz_init'] = 0.05

        elif eng == 'pymc3':
            import pymc3 as pm
            self.engine__ = pm
        elif eng == 'reddemcee':
            import reddemcee
            self.engine__ = reddemcee
            self.general_dependencies.append('reddemcee')
            self.general_dependencies.append('emcee')

            self.reddemcee_config['burnin'] = 'half'
            self.reddemcee_config['thinby'] = 1
        else:
            raise Exception(self.logger('Failed to set engine properly. Try a string!', center=True, c='red'))


    def load_data(self, folder_name):
        self.starname = folder_name
        self.data_wrapper = DataWrapper(folder_name, read_loc=self.read_loc)
        self.logger('\n')
        self.logger(self.data_wrapper.add_all__(), center=True, c='blue')
        self.logger('\n')

        if self.instrument_names is None:
            self.instrument_names = self.data_wrapper.RV_labels
        self.my_data = self.data_wrapper.get_data__()

        self.my_data_reduc = self.my_data.values[:, 0:3].T
        if len(self.data_wrapper.RV_labels) > 0:
            self.nins__ = len(self.data_wrapper.RV_labels)
            self.switch_RV = True


    def add_keplerian_block(self, parameterisation=0):
        self.kplanets__ += 1
        kb = mk_KeplerianBlock(self.my_data, parameterisation=parameterisation,
                                  number=self.kplanets__)

        prargs = [self.eccentricity_limits, self.eccentricity_prargs]
        kw = {'prargs':prargs, 'dynamics':self.switch_dynamics,
              'kplan':self.kplanets__, 'starmass':self.starmass,
              'dynamics_already_included':self.dynamics_already_included}
        SmartLimits(self.my_data, kb, **kw)  # sets limits and returns <extra_priors>

        #if parameterisation == 1 or parameterisation == 3:
        #    br.add_additional_priors(extra_priors)

        kb.signal_number = self.kplanets__

        kb.slice = slice((self.kplanets__ - 1)*kb.ndim_, self.kplanets__*kb.ndim_)

        self.blocks__.insert((self.kplanets__ - 1), kb)

        msg = '{} {}, {}'.format(colored(kb.type_, 'green', attrs=['bold']),
                                 colored('block added', 'green'),
                                 colored(kb.name_, 'green'))
        msg = f'                              {msg}'
        self.logger(msg, center=True)
        self.logger('\n')


    def add_instrumental_blocks(self, moav=0, offset=True):
        for i in range(self.nins__):
            ib = mk_InstrumentBlock(self.my_data, number=i+1, moav=moav)

            jitter_args = [self.jitter_limits, self.jitter_prargs]
            SmartLimits(self.my_data, ib, *jitter_args)

            ib.ins_no = int(i+1)
            ib.slice = slice(self.kplanets__ * 5 + (i) * ib.ndim_,
                             self.kplanets__ * 5 + (i+1) * ib.ndim_)

            ib.instrument_label = self.instrument_names[i]
            self.blocks__.append(ib)

            msg = '{} {}, {}'.format(colored(ib.type_, 'green', attrs=['bold']),
                                     colored('block added', 'green'),
                                     colored(ib.name_, 'green'))
            msg = f'                              {msg}'
            self.logger(msg, center=True)
            self.logger('\n')


    def add_acceleration_block(self, accel=1):
        ab = mk_AccelerationBlock(self.my_data, n=accel)
        self.blocks__.append(ab)

        msg = '{} {}, {}'.format(colored(ab.type_, 'green', attrs=['bold']),
                                 colored('block added', 'green'),
                                 colored(ab.name_, 'green'))
        msg = f'                              {msg}'
        self.logger(msg, center=True)
        self.logger('\n')


    def update_model(self):
        self.model = ReddModel(self.my_data, self)
        #if self.model.update_additional_priors():
        #    self.update_additional_priors_block()
        self.model.refresh__()


    def run(self, setup, progress=True):
        ### assert errors!
        ##
        if self.constrain_method == 'GM':
            if not self.gaussian_mixtures_fit:
                msg = 'Invalid constrain_method = GM with .gaussian_mixtures_fit = False'
                raise SyntaxError(msg)

        self.apply_conditions()
        self.saveplace = ensure_dir(self.starname, loc=self.save_loc)
        #self.temp_script = self.saveplace+'/temp/temp_script.py'
        self.temp_script = 'temp_script.py'

        if not self.switch_first:
            self.logger('\n')
            self.logger('~~ Setup Info ~~', center=True, c='blue', attrs=['reverse'])
            self.logger('\nCurrent Engine is  '+colored(self.engine__.__name__+' '+self.engine__.__version__, attrs=['bold']), c='blue')
            self.logger('Number of cores is '+colored(self.cores__, attrs=['bold']), c='blue')
            self.logger('Save location is   '+colored(self.saveplace, attrs=['bold']), c='blue')

            if self.switch_dynamics:
                dyn_crit = 'Hill Stability'
            else:
                dyn_crit = 'None'
            self.logger('Dynamical Criteria is      '+colored(dyn_crit, attrs=['bold']), c='blue')
            self.logger('Posterior fit method is    '+colored(self.posterior_dict[self.posterior_fit_method], attrs=['bold']), c='blue')
            self.logger('Limits constrain method is '+colored(self.constrain_method, attrs=['bold']), c='blue')
            self.logger('Model Selection method is  '+colored(self.ModelSelection.criteria, attrs=['bold']), c='blue')

            self.logger('\n')
            self.logger('~~ Automatically Saving ~~', center=True, c='blue', attrs=['reverse'])

            saving_ = ['save_log', 'save_chains', 'save_posteriors',
                       'save_likelihoods', 'plot_gaussian_mixtures',
                       'plot_keplerian_model', 'parameter_histograms',
                       'corner']
            saving0_ = ['Logger       ',
                        'Samples      ',
                        'Posteriors   ',
                        'Likelihoods  ',
                        'Plots: Gaussian Mixture     ',
                        'Plots: Keplerian Model      ',
                        'Plots: Parameter Histograms ',
                        'Plots: Corner               ']
            checks_ = []
            for thing in saving_:
                self_attr = getattr(self, thing)
                if type(self_attr) == dict:
                    if self_attr['plot']:
                        checks_.append(colored('✔ ', attrs=['reverse'], color='green'))
                    else:
                        checks_.append(colored('✘ ', attrs=['reverse'], color='red'))
                    pass
                else:
                    if self_attr:
                        checks_.append(colored('✔ ', attrs=['reverse'], color='green'))
                    else:
                        checks_.append(colored('✘ ', attrs=['reverse'], color='red'))

            self.logger('')
            for i in range(4):
                self.logger('{}: {}'.format(saving0_[i], checks_[i]), c='blue')
            self.logger('')
            for i in range(4, 8):
                self.logger('{}: {}'.format(saving0_[i], checks_[i]), c='blue')


            self.switch_first = True

        self.update_model()

        # Print Pre-Run
        if True:
            self.logger('\n\n')
            self.logger('~~ Pre-Run Info ~~', center=True, c='yellow', attrs=['bold', 'reverse'])
            self.logger('\n\n')

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

            self.logger('\n')
            for b in self:
                self.logger('Math for {}:\n'.format(b.name_), c='yellow')
                self.logger('{}'.format(b.math_display_), center=True, c='yellow')

            self.logger('\n')


        if self.engine__.__name__ == 'reddemcee':
            from emcee.backends import HDFBackend
            ntemps, nwalkers, nsteps = setup

            self.write_model()

            os.system('ipython {}'.format(self.temp_script))

            self.sampler = self.engine__.PTSampler(nwalkers, self.model.ndim__,
                                         self.model.evaluate_loglikelihood,
                                         self.model.evaluate_logprior,
                                         logl_args=[], logl_kwargs={},
                                         logp_args=[], logp_kwargs={},
                                         ntemps=ntemps, pool=None)

            #self.sampler = [None for _ in range(ntemps)]
            with open('sampler_pickle.pkl', 'rb') as sampler_metadata:
                self.sampler_metadata_dict = pickle.load(sampler_metadata)

            for t in range(ntemps):
                loc_t = '{}emperor_backend_{}.h5'.format(self.saveplace+'/temp/', t)
                self.sampler[t] = HDFBackend(loc_t)

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
                self.write_model()

                # RUN SAMPLER
                os.system('ipython {}'.format(self.temp_script))

                with open('sampler_pickle.pkl', 'rb') as sampler_metadata:
                    self.sampler_metadata_dict = pickle.load(sampler_metadata)
                pass
            else:
                # SET SETUP
                # SET SAMPLER
                # RUN SAMPLER
                pass


    def run_auto(self, setup, k_start=0, k_end=10, parameterisation=1, moav=0, accel=0, progress=True):
        self.auto_setup = setup
        if self.engine__.__name__ in ['emcee', 'dynesty', 'pymc3', 'reddemcee']:

            if self.switch_RV and not self.switch_SA:
                self.add_instrumental_blocks(moav=moav)
            if accel:
                self.add_acceleration_block(accel=accel)

            while k_start <= k_end:
                if not self.switch_first and k_start > 0 and self.switch_RV:
                    for _ in range(k_start):
                        self.add_keplerian_block(parameterisation=parameterisation)


                oldlike_max = self.like_max
                oldpost_max = self.post_max
                oldchi2 = self.chi2
                oldAIC = self.AIC
                oldBIC = self.BIC
                oldDIC = self.DIC
                oldHQIC = self.HQIC
                oldRMSE = self.RMSE
                oldBayesFactor = self.BayesFactor


                self.run(self.auto_setup, progress=progress)
                self.postprocess(self.auto_setup)  # change values

                k_start += 1


                if self.switch_constrain:
                    if self.constrain_method == 'sigma':
                        for b in self:
                            if b.type_ == 'Keplerian':
                                for p in b:
                                    if p.fixed is None:
                                        pval = p.value
                                        psig = p.sigma

                                        limf, limc = pval - self.constrain_sigma*psig, pval + self.constrain_sigma*psig

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
                # if not continue, model selec
                if self.ModelSelection.compare(self.BIC, oldBIC):
                    oldbic_display = np.round(oldBIC, 3)
                    newbic_display = np.round(self.BIC, 3)
                    self.logger('\nBIC condition met!!', c='blue', attrs=['bold'])
                    self.logger('present BIC < past BIC - 5', c='blue')
                    self.logger(self.ModelSelection.msg, c='blue')
                else:
                    self.logger('\nBIC condition not met', c='blue')
                    self.logger('present BIC < past BIC - 5', c='blue')
                    self.logger(self.ModelSelection.msg, c='blue')

                    self.logger('\n')
                    self.logger('   ', center=True, c='magenta', attrs=['bold', 'reverse'])
                    self.logger('~~ Ending the run ~~', center=True, c='magenta', attrs=['bold', 'reverse'])
                    self.logger('   ', center=True, c='magenta', attrs=['bold', 'reverse'])
                    self.logger('\n', center=True, c='magenta')
                    break

                if k_start > k_end:
                    self.logger('\n')
                    self.logger('   ', center=True, c='magenta', attrs=['bold', 'reverse'])
                    self.logger('~~ Run came to an end ~~', center=True, c='magenta', attrs=['bold', 'reverse'])
                    self.logger('   ', center=True, c='magenta', attrs=['bold', 'reverse'])
                    self.logger('\n', center=True, c='magenta')
                    break

                # if passes:
                self.logger('\n\n')
                self.logger('   ', center=True, c='magenta', attrs=['bold', 'reverse'])
                self.logger('~~ Proceeding with the next run ! ~~', center=True, c='magenta', attrs=['bold', 'reverse'])
                self.logger('   ', center=True, c='magenta', attrs=['bold', 'reverse'])
                self.logger('\n')
                if self.switch_RV:
                    self.add_keplerian_block(parameterisation=parameterisation)
                self.update_model()


    def postprocess(self, setup=[]):
        if self.engine__.__name__ == 'reddemcee':
            ntemps, nwalkers, nsteps = setup

            self.acceptance_fraction = self.sampler_metadata_dict['acceptance_fraction']
            self.autocorr_time = self.sampler_metadata_dict['get_autocorr_time']

            self.sampler.betas = self.sampler_metadata_dict['betas']
            self.sampler.ratios = self.sampler_metadata_dict['ratios']


            if True:
                if type(self.reddemcee_config['burnin']) == str:
                    if self.reddemcee_config['burnin'] == 'half':
                        self.reddemcee_discard = nsteps // 2
                    elif self.reddemcee_config['burnin'] == 'auto':
                        print('method not yet implemented!! CODE 26')
                        self.reddemcee_discard = 0
                    else:
                        self.reddemcee_discard = 0
                elif type(self.reddemcee_config['burnin']) == int:
                    self.reddemcee_discard = self.reddemcee_config['burnin']

                elif type(self.reddemcee_config['burnin']) == float:
                    self.reddemcee_discard = int(self.reddemcee_config['burnin'] * nsteps)

                else:
                    print('method not understood!! CODE 27')
                    self.reddemcee_discard = 0
            if True:
                if type(self.reddemcee_config['thinby']) == str:
                    if self.reddemcee_config['thinby'] == 'half':
                        self.reddemcee_thin = 2

                    elif self.reddemcee_config['thinby'] == 'auto':
                        print('method not yet implemented!! CODE 28')
                        self.reddemcee_thin = 1
                    else:
                        self.reddemcee_thin = 1
                elif type(self.reddemcee_config['thinby']) == int:
                    self.reddemcee_thin = self.reddemcee_config['thinby']

                else:
                    print('method not understood!! CODE 29')
                    self.reddemcee_thin = 1

            reddemcee_dict = {'discard':self.reddemcee_discard,
                              'thin':self.reddemcee_thin,
                              'flat':True}

            raw_chain = self.sampler.get_func('get_chain', kwargs=reddemcee_dict)
            raw_posts = self.sampler.get_func('get_log_prob', kwargs=reddemcee_dict)
            raw_likes = self.sampler.get_func('get_blobs', kwargs=reddemcee_dict)

            setup_info = 'Temperatures, Walkers, Steps   : '
            size_info = [len(raw_chain[t]) for t in range(ntemps)]


            if self.switch_evidence:
                self.evidence = self.sampler_metadata_dict['thermodynamic_integration']

            best_loc = np.argmax(raw_posts[0])

            self.post_max = raw_posts[0][best_loc]
            self.like_max = raw_likes[0][best_loc]
            self.prior_max = self.post_max - self.like_max

            self.ajuste = raw_chain[0][best_loc]
            self.sigmas = np.std(raw_chain, axis=1)[0]

            self.fit_max = raw_chain[0][best_loc]
            self.fit_mean = np.mean(raw_chain[0], axis=0)
            self.fit_median = np.median(raw_chain[0], axis=0)

        if self.engine__.__name__ == 'dynesty':
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

            ## FIN
            best_loc = -1

            self.post_max = raw_posts[0][best_loc]
            self.like_max = raw_likes[0][best_loc]
            self.prior_max = 0

            self.ajuste = raw_chain[0][best_loc]
            self.sigmas = np.std(raw_chain[0], axis=0)

            self.fit_max = raw_chain[0][best_loc]
            self.fit_mean = raw_chain[0][best_loc]
            self.fit_median = raw_chain[0][best_loc]

            print('\n\n------------ Dynesty Summary -----------\n\n')
            print(str(results.summary()))


        chains = raw_chain
        posts = raw_posts
        likes = raw_likes


        ###########################################
        ###########################################
        # GET STATS
        if True:
            if self.switch_RV:
                ymod, err2 = self.model.evaluate_model(self.ajuste)
                ferr2 = err2 + self.my_data['eRV'].values ** 2
                residuals = self.my_data['RV'].values - ymod

                ndim = self.model.ndim__
                ndat = self.model.ndata

                rss = np.sum(residuals**2)

                self.dof = ndat - ndim
                self.chi2 = rss / self.dof
                self.chi2_red = np.sum(residuals**2 / ferr2) / self.dof
                self.RMSE = np.sqrt(np.sum(residuals ** 2) / len(residuals))

                self.AIC = 2 * ndim - 2 * self.like_max
                self.BIC = np.log(ndat) * ndim - 2 * self.like_max

                tm = np.mean(raw_chain[0], axis=0)
                self.DIC = -2 * self.model.evaluate_loglikelihood(tm) + np.var(-2 * likes[0])

                self.post_true = self.post_max - self.evidence[0]
                self.BayesFactor = self.like_max - self.evidence[0]

                self.HQIC = 2 * ndim * np.log(np.log(ndat)) - 2 * self.like_max

        ## SET P.VALUES P.SIGMA
        if True:
            ## **RED w/ set_attr
            j = 0  # delete this ap1
            for b in self:
                for p in b:
                    if p.fixed == None:
                        p.value = self.ajuste[j]
                        p.sigma = self.sigmas[j]
                        p.sigma_frac_mean = 0

                        p.value_max = self.fit_max[j]
                        p.value_mean = self.fit_mean[j]
                        p.value_median = self.fit_median[j]
                        j += 1
                    else:
                        p.sigma = np.nan
                        p.value_max = p.value
                        p.value_mean = p.value
                        p.value_median = p.value

        # Get extra info. Parameter transformation and planet signatures
        if True:
            self.sma = []
            self.mm = []

            self.sma_sig = []
            self.mm_sig = []

            self.extra_names = []
            self.extra_chains = []


            # Get extra chains
            uptothisdim = 0
            for b in self:
                chain_counter = 0
                ## PLANET SIGNATURES
                if b.type_ == 'Keplerian':
                    my_params = [None, None, None, None, None]

                    for i in b.C_:
                        my_params[i] = chains[0].T[uptothisdim + chain_counter]
                        chain_counter += 1

                    for i in b.A_:
                        my_params[i] = b[i].fixed * np.ones(len(chains[0]))


                    if True:
                        if b.parameterisation == 0:
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
                                self.extra_names.append(thingy+'_{}'.format(b.number_))

                            for thingy in [per_, A_, phase_, ecc_, w_]:
                                self.extra_chains.append(thingy)

                        elif b.parameterisation == 2:
                            per, A, t0, ecc, w = b.get_attr('value')
                            per_, A_, t0_, ecc_, w_ = my_params

                        elif b.parameterisation == 3:
                            per, A, t0, S, C = b.get_attr('value')
                            per_, A_, t0_, S_, C_ = my_params

                            ecc, w = delinearize(S, C)
                            ecc_, w_ = adelinearize(S_, C_)

                            for thingy in ['Eccentricity', 'Longitude_Periastron']:
                                self.extra_names.append(thingy)
                            for thingy in [ecc_, w_]:
                                self.extra_chains.append(thingy)

                        if self.starmass:
                            sma, mm = cps(per, A, ecc, self.starmass)
                            sma_, mm_ = cps(per_, A_, ecc_, self.starmass)

                            self.sma_sig.append(sma)
                            self.mm_sig.append(mm)

                            self.sma_sig.append(np.std(sma_))
                            self.mm_sig.append(np.std(mm_))

                            self.extra_names.append('Semi-Major Axis [AU]')
                            self.extra_names.append('Minimum Mass [M_J]')
                            self.extra_chains.append(sma_)
                            self.extra_chains.append(mm_)

                uptothisdim += b.ndim_
            ## Set p.values and sigma for extra params
            if True:
                ## **RED w/ set_attr\
                jj = 0
                for b in self:
                    for p in b.additional_parameters:
                        if p.has_posterior:
                            ch = self.extra_chains[jj]
                            p.value = ch[best_loc]
                            p.sigma = np.std(ch)
                            p.sigma_frac_mean = 0

                            p.value_max = ch[best_loc]
                            p.value_mean = np.mean(ch)
                            p.value_median = np.median(ch)
                            jj += 1

        # SAVE STUFF??
        if True:
            if self.save_chains is not None:
                self.save_chain(chains)
                self.save_posterior(posts)
                self.save_loglikelihood(likes)

            self.chains = chains[0]
            self.posts = posts[0]
            self.likes = likes[0]

            self.update_model()

        # SET GM
        if True:
            # REGULAR PARAMETERS
            if self.gaussian_mixtures_fit:
                self.model.get_GMEstimates(self.chains)

            # ADDITIONAL PARAMETERS
            for b in self:
                jj = 0
                for p in b.additional_parameters:
                    if p.has_posterior:
                        ch = self.extra_chains[jj]
                        p.GM_parameter = GM_Estimator().estimate(ch, p.display_name, p.unit)
            pass

        # SET POSTERIORS
        if True:
            if self.posterior_fit_method == 'GM':
                for b in self:
                    for p in b[b.C_]:
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

        # PRINT POSTERIORS
        if True:
            self.logger('\n\n')
            self.logger('~~ Best Fit ~~', center=True, c='yellow', attrs=['bold', 'reverse'])
            self.logger('\n\n')


            tab_3 = np.array([])
            switch_title = True

            for b in self:
                to_tab0 = b.get_attr(['name', 'display_posterior', 'value_max',
                                      'value_mean', 'sigma', 'limits'])

                to_tab0[2] = np.round(to_tab0[2], 3)
                to_tab0[3] = np.round(to_tab0[3], 3)
                to_tab0[4] = np.round(to_tab0[4], 3)
                to_tab0[5] = np.round(to_tab0[5], 3)


                to_tab = list(zip(*to_tab0))
                if switch_title:
                    self.logger(tabulate(to_tab,
                                          headers=['Parameter       ',
                                                   'Posterior       ',
                                                   'Value (max)',
                                                   'Value (mean)',
                                                   'Sigma',
                                                   'Limits      ',
                                                   ]))
                    switch_title = False
                else:
                    self.logger(tabulate(to_tab,
                                          headers=['                ',
                                                   '                ',
                                                   '           ',
                                                   '           ',
                                                   '     ',
                                                   '           ',
                                                   ]))

                if len(b.additional_parameters):
                    mask = [x.has_posterior for x in b.additional_parameters]

                    pnames = [p.display_name for p in np.array(b.additional_parameters)[mask]]
                    #pdisp = [p.limits for p in np.array(b.additional_parameters)[mask]]
                    pdisp = [p.display_posterior for p in np.array(b.additional_parameters)[mask]]
                    pvalmax = [p.value_max for p in np.array(b.additional_parameters)[mask]]
                    pvalmean = [p.value_mean for p in np.array(b.additional_parameters)[mask]]
                    psig = [p.sigma for p in np.array(b.additional_parameters)[mask]]
                    plims = [p.limits for p in np.array(b.additional_parameters)[mask]]

                    to_tab1 = [pnames, pdisp, pvalmax, pvalmean, psig, plims]

                    to_tab1[2] = np.round(to_tab1[2], 3)
                    to_tab1[3] = np.round(to_tab1[3], 3)
                    to_tab1[4] = np.round(to_tab1[4], 3)
                    to_tab1[5] = np.round(to_tab1[5], 3)

                    to_tab = list(zip(*to_tab1))
                    self.logger(tabulate(to_tab,
                                          headers=['                ',
                                                   '                ',
                                                   '           ',
                                                   '           ',
                                                   '     ',
                                                   '           ',
                                                   ]))

        # PRINT STATS
        if True:
            self.logger('\n\n')
            self.logger('~~ Run Info ~~', center=True, c='yellow', attrs=['bold', 'reverse'])
            self.logger('\n\n')

            tabh_1 = ['Info                             ', 'Value                       ']

            tab_1 =    [['Star Name                      : ', self.starname],
                        ['The sample sizes are           : ', size_info],
                        [setup_info, setup],
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
                     ['The chi2 is                 :    ', '{:.3f}'.format(self.chi2)],
                     ['The reduced chi2 is         :    ', '{:.3f}'.format(self.chi2_red)],
                     ['The AIC is                  :    ', '{:.3f}'.format(self.AIC)],
                     ['The BIC is                  :    ', '{:.3f}'.format(self.BIC)],
                     ['The DIC is                  :    ', '{:.3f}'.format(self.DIC)],
                     ['The HQIC is                 :    ', '{:.3f}'.format(self.HQIC)],
                     ['The Bayes Factor is         :    ', '{:.3f}'.format(self.BayesFactor)],
                     ['The RMSE is                 :    ', '{:.3f}'.format(self.RMSE)]
                     ]

            if self.engine__.__name__ == 'reddemcee':
                self.logger('Beta Detail                     :   ' + str(['{:.3f}'.format(x) for x in self.sampler.betas]))
                self.logger('Temperature Swap                :   ' + str(['{:.3f}'.format(x) for x in self.sampler.ratios]))
                self.logger('Mean Acceptance Fraction        :   ' + str(['{:.3f}'.format(x) for x in np.mean(self.acceptance_fraction, axis=1)]))
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

        #######################
        # PLOT GM PER PARAMETER
        if self.gaussian_mixtures_fit:
            if self.plot_gaussian_mixtures['plot']:
                self.logger('Plotting Gaussian Mixtures', center=True, c='green')
                pbar_tot = self.model.ndim__
                pbar = tqdm(total=pbar_tot)
                for b in self:
                    for p in b[b.C_]:
                        plot_GM_Estimator(p.GM_parameter,
                                        saveloc=self.saveplace,
                                        fmt=self.save_plots_fmt,
                                        sig_factor=self.plot_gaussian_mixtures['sig_factor'],
                                        fill_cor=b.bnumber_-1,
                                        plot_name='{} '.format(b.bnumber_) + p.GM_parameter.name,
                                        plot_title=self.plot_gaussian_mixtures['plot_title'],
                                        plot_ylabel=self.plot_gaussian_mixtures['plot_ylabel'])
                        pbar.update(1)
                    for p in b.additional_parameters:
                        if p.has_posterior:
                            plot_GM_Estimator(p.GM_parameter,
                                            saveloc=self.saveplace,
                                            fmt=self.save_plots_fmt,
                                            sig_factor=self.plot_gaussian_mixtures['sig_factor'],
                                            fill_cor=b.bnumber_-1,
                                            plot_name='{} '.format(b.bnumber_) + p.GM_parameter.name,
                                            plot_title=self.plot_gaussian_mixtures['plot_title'],
                                            plot_ylabel=self.plot_gaussian_mixtures['plot_ylabel'])
                pbar.close()

        # PLOT Keplerian Model and uncertainties
        if self.kplanets__ > 0:
            if self.plot_keplerian_model['plot']:
                res_max = flatten(self.get_attr_param('value_max'))
                self.logger('Plotting Keplerian Models', center=True, c='green')
                if True:
                    plot_KeplerianModel(self.my_data, self.model,
                                        res_max, saveloc=self.saveplace,
                                        options = self.plot_keplerian_model)
                else:
                    print('Model plot failed miserably :(\n')

        # PLOT stuff with arviz, including corner!!

        if self.plot_trace['plot']:
            self.logger('Plotting Trace', center=True, c='green')

            plot_traces(self.sampler[self.plot_trace['temp']],
                                self.engine__.__name__,
                                self.model, saveloc=self.saveplace,
                                trace_modes=self.plot_trace['modes'], fmt='png')

        # SAVE LOG
        if self.save_log:
            self.logger.saveto(self.saveplace)


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
                np.savez_compressed(f'{self.saveplace}/chains/chain_{str(temp)}', chains[temp])


    def save_posterior(self, posts):
        temps = self.save_posteriors
        if self.engine__.__name__ == 'reddemcee':
            for temp in temps:
                np.savez_compressed(
                    f'{self.saveplace}/posteriors/posterior_{str(temp)}',
                    posts[temp],
                )


    def save_loglikelihood(self, likes):
        temps = self.save_likelihoods
        if self.engine__.__name__ == 'reddemcee':
            for temp in temps:
                np.savez_compressed(self.saveplace + '/likelihoods/likelihood_'+str(temp), likes[temp])


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

                            #toprint = [colored(x, attrs=['bold', 'underline']) for x in c]
                            #print('\nCondition applied: Parameter {} attribute {} set to {}'.format(*toprint))

        applied.sort()
        applied = [applied for applied,_ in itertools.groupby(applied)]

        for ap in applied[::-1]:
            self.conds.remove(ap)


    def add_condition(self, cond):
        self.conds.append(cond)


    def write_model(self):

        self.dir_work = os.path.dirname(os.path.realpath(__file__))
        self.dir_save = self.saveplace

        self.script_pool_opt = get_support(f'pools/0{self.multiprocess_method}.pool')

        if self.engine__.__name__ == 'reddemcee':


            ntemps, nwalkers, nsteps = self.auto_setup
            #self.my_data.to_csv('{}/temp/temp_data.csv'.format(self.saveplace))
            with open(self.temp_script, 'w') as f:
                f.write(open(get_support('init.scr')).read())

                # DEPENDENCIES
                for d in self.general_dependencies:
                    f.write('''
import {}
'''.format(d))

                f.write('''
import kepler
''')
                # CONSTANTS
                f.write('''
nan = np.nan
A_ = {}
mod_fixed_ = {}
gaussian_mixture_objects = dict()

'''.format(self.model.A_, self.model.mod_fixed))

                # MODEL
                f.write(open(self.model.write_model_(loc=self.saveplace)).read())

                # LIKELIHOOD
                f.write(open(get_support('likelihoods/00.like')).read())

                # PRIOR
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
                        f.write('''
    lp += {}(theta[{}], {}, {})
'''.format(p.prior, self.model.C_[count], p.limits, p.prargs))
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

                # MULTIPROCESSING
                f.write(open(self.script_pool_opt).read())

                # SETUP CONSTS
                f.write('''

ntemps, nwalkers, nsteps = {0}, {1}, {2}
setup = ntemps, nwalkers, nsteps
ndim = {3}
'''.format(ntemps, nwalkers, nsteps, self.model.ndim__))

                # BACKENDS
                f.write('''
backends = [None for _ in range(ntemps)]
for t in range(ntemps):
    filename = '{}emperor_backend_' +str(t)+'.h5'
    backends[t] = emcee.backends.HDFBackend(filename)
    backends[t].reset(nwalkers, ndim)

sampler = reddemcee.PTSampler(nwalkers, ndim,
                             my_likelihood,
                             my_prior,
                             ntemps=ntemps, pool=mypool,
                             backend=backends)
'''.format(self.saveplace+'/temp/'))

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


                f.write('''
    return list(pos)


''')

                f.write('''
def test_init(max_repeats={0}):
    p0 = set_init()

    is_bad_position = True
    repeat_number = 0

    while is_bad_position and repeat_number < max_repeats:
        is_bad_position = False
        for t in range(ntemps):
            for n in range(nwalkers):
                position_evaluated = p0[t][n]
                if my_prior(position_evaluated) == -np.inf:
                    is_bad_position = True
                    p0[t][n] = set_init()[t][n]
        repeat_number += 1
    return p0

'''.format(100))
            # RUN
                f.write('''

p1 = test_init()

''')
                f.write(open(get_support('endit.scr')).read())

        if self.engine__.__name__ == 'dynesty':
            if self.engine__args == 'dynamic':
                nlive, nlive_batch = self.auto_setup
                with open(self.temp_script, 'w') as f:
                    f.write(open(get_support('init.scr')).read())

                    # DEPENDENCIES
                    for d in self.general_dependencies:
                        f.write('''
import {}
    '''.format(d))

                    f.write('''
import kepler
    ''')
                    # CONSTANTS
                    f.write('''
nan = np.nan
A_ = {}
mod_fixed_ = {}
    '''.format(self.model.A_, self.model.mod_fixed))

                    # MODEL
                    f.write(open(self.model.write_model_(loc=self.saveplace)).read())

                    # LIKELIHOOD
                    f.write(open(get_support('likelihoods/00.like')).read())

                    # PTFORMARGS
                    f.write('''
ptformargs = {}
'''.format(self.ptformargs0))
                    # PRIOR


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
                    # SETUP CONSTS
                    f.write('''

nlive, nlive_batch = {}, {}
ndim = {}
'''.format(nlive, nlive_batch, self.model.ndim__))

                    # SAMPLER /BACKENDS?

                    # END
                    pool_bool = False
                    if self.multiprocess_method:
                        pool_bool = True
                    nested_dict = self.dynesty_config
                    nested_args = ''
                    for key in [*nested_dict]:
                        nested_args += f'{key} = {nested_dict[key]}, '

                    f.write(open(get_support('endit_dyn.scr')).read().format(pool_bool, self.cores__, nested_args))

        pass


    def __getitem__(self, n):
        return self.blocks__[n]


    def __repr__(self):
        #return self.model
        return str([self.blocks__[i].name_ for i in range(len(self.blocks__))])


    def __len__(self):
        return np.sum([len(b) for b in self])






#

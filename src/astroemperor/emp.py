# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT
# **FIN  : Finish this

import numpy as np
import pandas as pd
import os
import sys
import gc
import itertools
from .qol_utils import DataWrapper, reddlog, Debugger, create_directory_structure, get_support, GM_Estimator, adjust_table_tex, sec_to_clock
from .math_utils import find_confidence_intervals, cps, delinearize, adelinearize, hdi_of_samples, hdi_of_chain

from .block_repo import *
from .emp_model  import ReddModel
from .canvas import plot_GM_Estimator, plot_beta_density, plot_trace2, plot_KeplerianModel, super_plots, plot_histograms, plot_betas, plot_rates
from .globals import _CORES, _PLATFORM_SYSTEM

from tabulate import tabulate
from termcolor import colored
from importlib import reload, import_module
from tqdm import tqdm
import time


class emp_retainer(object):
    NONETHINGS = ['starname', 'betas', 'saveplace', 'ndim__',
                  'instrument_names_RV', 'instrument_names_AM', 'instrument_names_PM']
    SWITCHES_F = ['switch_RV', 'switch_SA', 'switch_SA_pro',
                    'switch_dynamics', 'dynamics_already_included',
                    'switch_celerite', 'switch_AM', 'switch_PM', 'switch_inclination',
                    'FPTS', 'save_all', 'save_plots', 'use_c']
    SWITCHES_T = ['switch_first_run', 'switch_evidence', 'gaussian_mixtures_fit', 'switch_jitter',
                      'save_log', 'save_log_simple', 'save_backends', 'debug_first']
    EMPTYLISTS = ['blocks__', 'model', 'conds', 'general_dependencies', 'model_dependencies']
    ZEROTHINGS = ['kplanets__', 'nins__','acceleration', 'keplerian_parameterisation']
    EMPTYSTRINGS = ['save_loc', 'read_loc']
    STATS_POSI = ['chi2', 'chi2_red', 'AIC', 'BIC',
                  'DIC', 'HQIC', 'RMSE', 'RMSi', 'Weights']
    STATS_NEGA = ['post_max', 'like_max', 'BayesFactor', 'Evidence']

    def _init_default_config(self):
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


    def _init_others_config(self):
        self.rounder_display = 3
        self.rounder_math = 2
        self.rounder_tables = 2
        self.starmass = 1.
        self.starmass_err = None
        self.multiprocess_method = 1
        self.cores__ = _CORES
        self.evidence = (-np.inf, np.inf)
        self.debug = Debugger()


    def _init_model_config(self):
        self.moav = {'order':0,
                     'global':False}

        self.sinusoid = 0
        self.magnetic_cycle = 0
        self.rotation_period = 0


        self.eccentricity_limits = [0, 1]
        self.eccentricity_prargs = [0, 0.1]

        self.jitter_limits = [0, 1]
        self.jitter_prargs = [5, 5]


    def _init_postprocess_config(self):
        # use as point estimate
        self.use_fit = 'max_post'

        # constrain
        self.constrain = {'method':'range',
                          'sigma':3,
                          'types':['Keplerian', 'Jitter'],
                          'tol':1e-4,
                          'known_methods':['sigma', 'GM', 'range', 'None']}
    
        # posterior
        self.cherry = {'cherry':False,
                       'median':False,
                       'diff':20}
        
        self.posterior_fit_method = 'GM'  # KDE*soon

        self.posterior_dict = {'GM': 'Gaussian Mixtures',
                               'KDE': 'Kernel Density Estimation',
                               None: 'None'}

        self.evidence_method = 'thermodynamic_integration'
        #self.evidence_method = 'stepping_stones'
        #self.evidence_method = 'hybrid_evidence'


    def _sampler_config(self):
        pass


class emp_scribe(object):
    def write_script(self):
        self._init_write_script()

        with open(self.temp_script, 'w') as f:
            f.write(open(get_support(f'init_{self.engine__.__name__}.scr')).read())
            self.debug.debug_script(f,
f'''import time
debug_timer = {self.time_init}
print('{self.temp_script}   : INIT | ', time.time()-debug_timer)''')

            # Dependencies
            self._set_dependencies_constants(f)

            # Set Logging Level
            self._set_logging_level(f)

            ## MODEL
            f.write(open(self.model.write_model(loc=self.saveplace)).read())

            self._write_likelihood(f)

            self._write_prior(f)

            self._set_multiprocessing(f)

            self._set_qol_constants(f)

            self._set_backends(f)

            self._set_pos0(f)

            getattr(self, f'_set_sampler_{self.engine__.__name__}')(f)

            self._set_run(f)

            self._save_backend(f)
            #self._save_pkl(f)

            f.write(open(get_support('run_main.scr')).read())


    def _write_prior(self, f):
        getattr(self, f'_write_prior_{self.engine__.__name__}')(f)


    def _write_prior_reddemcee(self, f):
        for prior_script in os.listdir(get_support('priors')):
            f.write(open(get_support(f'priors/{prior_script}')).read())
            f.write('''
''')

        f.write('''

def my_prior(theta):
    for a in A_:
        theta = np.insert(theta, a, mod_fixed_[a])
    lp = 0.
''')
        # TODO replace with cpointer
        count = 0
        for b in self.model:
            for p in b[b.C_]:
                f.write(f'''
    lp += {p.prior}(theta[{self.model.C_[count]}], {p.limits}, {p.prargs})
''')
                count += 1
            f.write('''
    if lp == -np.inf:
        return lp

''')

            for p in b.additional_parameters:
                if p.has_prior:
                    if p.name[:3] == 'Amp':
                        f.write('''
    x = theta[{0}][1]**2 + theta[{0}][2]**2
    '''.format(b.slice))

                    elif p.name[:3] == 'Ecc':
                        f.write('''
    x = theta[{0}][3]**2 + theta[{0}][4]**2
'''.format(b.slice))

                    elif p.name == 'Hill' and p.prargs[0] > 1:
                        f.write(f'''
    b_len = {len(b)}
    kplanets = {p.prargs[0]}
''')
                        f.write(open(get_support(f'PAE/0{b.parameterisation}.pae')).read())

                    else:
                        continue

                    f.write('''
    lp += {}(x, {}, {})
'''.format(p.prior, p.limits, p.prargs))


        f.write('''

    return lp


''')

        
    def _write_prior_dynesty(self, f):
        # this should go on constants
        f.write(f'''
from scipy.stats import truncnorm
ptformargs = {self.ptformargs0}
''')

        # TODO fixed params!
        # TODO make it priors
        
        f.write('''
def my_prior(theta):
    x = np.array(theta)
''')
        for b in self.model:
            for p in b[b.C_]:
                if p.prior == 'Uniform':
#                    if (p.is_circular and
#                        p.ptformargs[0]==np.pi and
#                        p.ptformargs[1]==np.pi):
#                            f.write(f'''
#    x[{p.cpointer}] =  (x[{p.cpointer}] % 1.) * 2 * np.pi
#''')
#                    else:
                    f.write(f'''
    a, b = {p.ptformargs}
    x[{p.cpointer}] =  a * (2. * x[{p.cpointer}] - 1) + b
''')
                elif p.prior == 'Normal':
                    f.write(f'''
    low, high = {p.limits}  # lower and upper bounds
    m, s, _ = {p.prargs}  # mean and standard deviation
    low_n, high_n = (low - m) / s, (high - m) / s  # standardize
    x[{p.cpointer}] = truncnorm.ppf(theta[{p.cpointer}], low_n, high_n, loc=m, scale=s)
''')
                    
                elif p.prior == 'Beta':
                    f.write(f'''
NOT INCLUDED BETA, DEBUG EMP.PY LN 272
''')
                # PERIOD AS LOG
                # add fking jeffries
                elif p.prior == 'Jeffreys':
                    f.write(f'''
#    low, high = {p.limits}
#    #x[{p.cpointer}] = np.exp(low + x[{p.cpointer}] * (high - low))
#    x[{p.cpointer}] = low * np.exp(x[{p.cpointer}] * np.log(high / low))  # Jeffreys
    a, b = {p.ptformargs}
    x[{p.cpointer}] =  a * (2. * x[{p.cpointer}] - 1) + b
''')
                    

        f.write('''              
    return x


''')


    def _write_likelihood(self, f):
        if self.switch_AM:
            aux_str = 'a00'
        elif self.switch_celerite:
            aux_str = '02'
        else:
            aux_str = '00'
        
        f.write(open(get_support(f'likelihoods/{aux_str}.like')).read())


    def _init_write_script(self):
        self.debug(f'write_script() : INIT | {time.time()-self.time_init}')
        self.dir_work = os.path.dirname(os.path.realpath(__file__))
        self.dir_save = self.saveplace

        self.script_pool_opt = get_support(f'pools/0{self.multiprocess_method}.pool')
        ## START SCRIPT
        self.debug(f'write_script() : open({self.temp_script}, w) | {time.time()-self.time_init}')        


    def _write_kernel(self, in_func=False):
        nterms = len(self.my_kernel['terms'])
        sumeq = '='


        # goes in my_model
        if in_func:
            tail = '01'
            with open(f'temp_kernel{tail}', 'w') as f:
                c = 0
                for termi in range(nterms):
                    line_str = f'kernel {sumeq} cterms.'
                    term_name = self.my_kernel['terms'][termi]
                    param_dict = self.my_kernel['params'][termi]

                    line_str += term_name
                    term_inputs = ''
                    for k in param_dict:
                        if param_dict[k] == None:
                            term_inputs += f'{k}=theta_gp[{c}], '
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
                            term_inputs += f'{k}=1.0, '
                        else:
                            term_inputs += f'{k}={param_dict[k]}, '
                    f.write(f'''
{line_str}({term_inputs})
''')

                    sumeq = '+='


    def _write_kernel_spe(self, in_func=False):
        if in_func:
            with open('temp_kernel01', 'w') as f:
                f.write(f'''
    #S0_1 = theta_gp[2] * theta_gp[0] ** 2 / (2*theta_gp[1]*np.pi**2)
    #S0_2 = theta_gp[3] * theta_gp[0] ** 2 / (8*theta_gp[1]*np.pi**2)
                        
    kernel = cterms.SHOTerm(S0=theta_gp[2], rho=theta_gp[0], tau=theta_gp[1])
    kernel += cterms.SHOTerm(S0=theta_gp[3], rho=theta_gp[0]*0.5, tau=theta_gp[1])
    gp_.kernel = kernel
''')
        else:
            with open('temp_kernel00', 'w') as f:
                f.write(f'''
kernel = cterms.SHOTerm(S0=1, rho=4., tau=8.)
kernel += cterms.SHOTerm(S0=1, rho=2., tau=8.)
''')           


    def _set_dependencies_constants(self, f): 
        # clean dependencies
        self.general_dependencies = np.unique(self.general_dependencies).tolist()
        self.model_dependencies = self.model.get_dependencies().tolist()
        self.all_dependencies = self.general_dependencies + self.model_dependencies
        
        self.model_constants = self.model.get_constants()
        # PRIORS
        my_priors = self.model.get_attr_param('prior', flat=True)
        if self.engine__.__name__== 'reddemcee':
            # TODO test this works properly
            if 'Beta' in my_priors:
                self.model_dependencies.append('from scipy.stats import beta as betapdf')

        if self.engine__.__name__== 'dynesty':
            # TODO test this works properly
            if 'Beta' in my_priors:
                self.model_dependencies.append('from scipy.stats import beta')
            if 'Normal' in my_priors:
                self.model_dependencies.append('from scipy.stats import truncnorm')



        ## DEPENDENCIES
        for d in self.all_dependencies:
            f.write(f'''
{d}''')
        
        f.write(f'''
''')
        self._set_gm_for_prior(f)

        ## CONSTANTS
        for c in self.model_constants:
            f.write(f'''{c} = {self.model_constants[c]}
''')

        if self.engine__.__name__ == 'dynesty':
            f.write(f'''
ptformargs = {self.ptformargs0}
''')


    def _set_gm_for_prior(self, f):
        if 'GaussianMixture' not in self.model.get_attr_param('prior', flat=True):
            return
        f.write('''
from sklearn.mixture import GaussianMixture as skl_gm''')
        
        for b in self.model:
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
            for p in b[b.C_]:
                if p.prior == 'GaussianMixture':
                    f.write(f'''
gaussian_mixture_{p.gC_} = skl_gm()
gaussian_mixture_objects[{p.gC_}] = gaussian_mixture_{p.gC_}
''')

                    for at in attributes:
                        coso = getattr(p.GM_parameter, at)
                        # TODO what is this doing?
                        if type(coso) == np.ndarray:
                            coso = coso.tolist()
                            f.write(f'''
setattr(gaussian_mixture_objects[{p.gC_}], '{at}', np.array({coso}))''')

                        else:
                            f.write(f'''
setattr(gaussian_mixture_objects[{p.gC_}], '{at}', {coso})''')


    def _set_logging_level(self, f):
        if self.engine__.__name__ == 'reddemcee':
            f.write(f'''
logging.getLogger('emcee').setLevel('{self.run_config['logger_level']}')
''')


    def _set_multiprocessing(self, f):
        # TODO test different mp schemes
        if self.debug.debugging:
            f.write('''
print('write_script() : import multiprocessing pool | ', time.time()-debug_timer)
''')
        
        if self.engine__.__name__ in ['reddemcee', 'dynesty']:
            aux_args = self.cores__
            if self.multiprocess_method == 1:
                aux_args = self.cores__, False
                if _PLATFORM_SYSTEM == 'Darwin':
                    aux_args = self.cores__, True
                
            f.write(open(self.script_pool_opt).read().format(aux_args))

        #if self.engine__.__name__ == 'dynesty':
            # TODO this goes in engine config
            #self.dynesty_pool_bool = False
            #if self.multiprocess_method:
            #    self.dynesty_pool_bool = True


    def _set_qol_constants(self, f):
        if self.engine__.__name__ == 'reddemcee':
            betas0 = self.engine_config['betas']
            ntemps, nwalkers, nsweeps, nsteps = self.engine_config['setup']

            if betas0 is not None:
                aux_betas_str = f'np.array({list(betas0)})'
                aux_betas_str = f'{list(betas0)}'
            else:
                aux_betas_str = 'None'

            
            f.write(f'''

ntemps, nwalkers, nsweeps, nsteps = {ntemps}, {nwalkers}, {nsweeps}, {nsteps}
setup = ntemps, nwalkers, nsweeps, nsteps
ndim = {self.model.ndim__}
betas = {aux_betas_str}
progress = {self.engine_config['progress']}

''')
        elif self.engine__.__name__ == 'dynesty':
            f.write(f'''

ndim = {self.model.ndim__}
''')           


    def _set_backends(self, f):
        self.backend_bool = False
        self.backend_name = 'backend_savetest'

        if self.backend_bool:
            if self.engine__.__name__ == 'reddemcee':
                f.write(f'''
from reddemcee.hdf import PTHDFBackend

my_backend = PTHDFBackend("{self.backend_name}.h5")
my_backend.reset(ntemps, nwalkers, ndim,
                 tsw_hist={self.engine_config["tsw_history"]},
                 smd_hist={self.engine_config["smd_history"]})
''')
        else:
            f.write('''
my_backend = None
''')


    def _set_sampler_reddemcee(self, f):
        f.write(f'''
sampler = reddemcee.PTSampler(nwalkers,
                             ndim,
                             my_likelihood,
                             my_prior,
                             ntemps=ntemps,
                             pool=mypool,
                             backend=my_backend,
                             betas=betas,
                             tsw_history={self.engine_config["tsw_history"]},
                             smd_history={self.engine_config["smd_history"]},
                             adapt_tau={self.engine_config["adapt_tau"]},
                             adapt_nu={self.engine_config["adapt_nu"]},
                             adapt_mode={self.engine_config["adapt_mode"]}
                             )

''')
        self._set_sampler_D_(f)
    def _set_sampler_D_(self, f):
        limits = np.array(self.model.get_attr_param('limits',
                                                    flat=True))[self.model.C_]
        D_ = np.diff(limits).flatten()
        f.write(f'''
sampler.D_ = np.array({np.array2string(D_, separator=', ')})

''')
        pass


    def _set_sampler_dynesty(self, f):
        f.write(f'''
sampler = dynesty.DynamicNestedSampler(my_likelihood,
                                       my_prior,
                                       ndim,
                                       pool=mypool,
                                       **{self.engine_config})

''')


    def _set_pos0(self, f):
        if self.engine__.__name__ == 'dynesty':
            return
        self.debug.debug_script(f, f'print("{self.temp_script}   : set_init()  pos0 | time.time()-debug_timer")')

        f.write('''
def set_init():
    pos = np.zeros((ntemps, nwalkers, ndim))

    for t in range(ntemps):
        j = 0
''')

        for b in self.model:
            for p in b[b.C_]:
                r = 0.707 if p.is_hou else 1

                b = p.limits[0] if p.init_pos[0] is None else np.round(p.init_pos[0], 8)

                a = p.limits[1] if p.init_pos[1] is None else np.round(p.init_pos[1], 8)

                f.write(f'''
        m = ({a} + {b}) / 2
        r = ({a} - {b}) / 2 * {r}
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1

''')

        f.write('''
    return pos


''')

        ##############
        # TEST P0

        f.write(f'''
def test_init(max_repeats={100}):
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
        if {str(self.debug.debugging)}:
            print('test_init ', repeat_number)

    if is_bad_position:
        print('COULDNT FIND VALID INITIAL POSITION')
    return p0

    
p1 = test_init()
''')

        self.debug.debug_script(f, f'print("{self.temp_script}   : test_init()  pos0 | time.time()-debug_timer")')


    def _set_run(self, f):
        self.debug.debug_script(f, f'print("{self.temp_script}   : run __main__ | time.time()-debug_timer")')

        # TODO restructure this
        if self.engine__.__name__ == 'reddemcee':
            adapt_batches = self.run_config['adaptation_batches']
            adapt_nsweeps = self.run_config['adaptation_nsweeps']
            if (adapt_batches and adapt_nsweeps):
                f.write(f'''
adaptation_batches = {adapt_batches}
adaptation_nsweeps = {adapt_nsweeps}

# freeze1
adaptation_batches = 1
adaptation_nsweeps = {self.reddemcee_discard}

''')
                self.endit_script = 'endit_freeze1.scr'
            else:
                self.endit_script = 'endit_reddemcee.scr'
            
            f.write(open(get_support(self.endit_script)).read())

        if self.engine__.__name__ == 'dynesty':
            nested_args = ''.join(
                f'{key} = {self.run_config[key]}, '
                for key in [*self.run_config]
            )
            f.write(open(get_support('endit_dynesty.scr')).read().format(nested_args))
  

    def _save_pkl(self, f):
        f.write(open(get_support('save_metadata.scr')).read().format(f'{self.saveplace}/restore/'))


    def _save_backend(self, f):
        self.backend_bool = True
        # TODO rename backend to something with star&run?
        self.backend_name = f'{self.starname}_{self.saveplace_run}'

        if self.backend_bool:
            if self.engine__.__name__ == 'reddemcee':
                f.write(f'''
    from reddemcee.hdf import PTHDFBackend

    saver = PTHDFBackend('{self.backend_name}.h5')
    saver.reset(ntemps, nwalkers, ndim,
                tsw_hist={self.engine_config["tsw_history"]},
                smd_hist={self.engine_config["smd_history"]})

    ntot = sampler.backend.iteration
    saver.grow(ntot)
    with saver.open("a") as f:
        g = f[saver.name]
        g.attrs["iteration"] = ntot


        if sampler.backend.tsw_history_bool:
            g["tsw_history"].resize(ntot, axis=0)
            g["tsw_history"][:] = sampler.backend.tsw_history

        if sampler.backend.smd_history_bool:
            g["smd_history"].resize(ntot, axis=0)
            g["smd_history"][:] = sampler.backend.smd_history


    ntot = sampler.backend[0].iteration
    for t in range(ntemps):
        saver[t].grow(ntot, None)

        with saver[t].open("a") as f:
            g = f[saver[t].name]
            g.attrs["iteration"] = ntot

            g["chain"][:, :, :] = sampler.backend[t].get_chain()
            g["log_like"][:, :] = sampler.backend[t].get_log_like()
            g["log_prob"][:, :] = sampler.backend[t].get_log_prob()
            g["beta_history"][:] = sampler.backend[t].get_betas()
            g["accepted"][:] = sampler.backend[t].accepted
''')


            if self.engine__.__name__ == 'dynesty':
                f.write(f'''
    with open('{self.backend_name}.pkl','wb') as md_save:
        pickle.dump(sampler.results, md_save)
''')


    def _load_sampler(self):
        self.debug(f'run  : _load_sampler() | {time.time()-self.time_init}')
        getattr(self, f'_load_sampler_{self.engine__.__name__}')()


    def _load_sampler_reddemcee(self):
        '''Loads temporary model script and Backends'''
        self._load_script_models()

        from reddemcee.hdf import PTHDFBackend, HDFBackend_plus

        ntemps, nwalkers, nsweeps, nsteps = self.engine_config['setup']
        reader = PTHDFBackend(f'{self.saveplace}/restore/backends/{self.backend_name}.h5')

        reader.backends = [HDFBackend_plus(f'{self.saveplace}/restore/backends/{self.backend_name}_{t}.h5') for t in range(ntemps)]


        self.betas = list(reader.get_last_sample().betas)
        self.engine_config["betas"] = self.betas
        self.engine_config["ntemps"] = len(self.betas)


        self.sampler = self.engine__.PTSampler(nwalkers,
                                 self.model.ndim__,
                                 self.temp_like_func,
                                 self.temp_prior_func,
                                 ntemps=ntemps,
                                 pool=None,
                                 backend=reader,
                                 betas=self.engine_config["betas"],
                                 tsw_history=self.engine_config["tsw_history"],
                                 smd_history=self.engine_config["smd_history"],
                                 adapt_tau=self.engine_config["adapt_tau"],
                                 adapt_nu=self.engine_config["adapt_nu"],
                                 adapt_mode=self.engine_config["adapt_mode"],
                                 )
                             

    def _load_sampler_dynesty(self):
        self._load_script_models()

        import pickle
        with open(f'{self.saveplace}/restore/backends/{self.backend_name}.pkl', 'rb') as sampler_metadata:
            self.sampler = pickle.load(sampler_metadata)


    def _load_script_models(self):
        sys.path.insert(0, f'{self.saveplace}/temp')
        
        module = import_module(self.temp_script.split('.')[0])
        module = reload(module)

        self.temp_model_func = module.my_model
        self.temp_like_func = module.my_likelihood
        self.temp_prior_func = module.my_prior

        sys.path.pop(0)


class emp_painter(object):
    def _init_plot_config(self):
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
                                'temps':None,
                                'name':'plot_posteriors',
                                'scatter_kwargs':{'alpha':0.7,
                                                  's':10},
                                'hexbin_kwargs':{'gridsize':(60,10),
                                                 'mincnt':1},
                                'chain_kwargs':{'marker':'o',
                                                'markersize':2,
                                                'alpha':0.2,
                                                'lw':0},
                                'vlines_kwargs':{'lw':4,
                                                 'alpha':0.75},
                                'legend_kwargs':{'framealpha':0,
                                                 'loc':1,
                                                 'fontsize':28},
                                'label_kwargs':{'fontsize':44},
                                'tick_params_kwargs':{'labelsize':40},
                                'colorbar_kwargs':{'fontsize':28},
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
                                     'name':'plot_keplerian_model',
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
                                 'nice_name':'Plotting Periodogram',
                                }
        
        self.plot_gaussian_mixtures = {'sig_factor':4,
                                       'plot_title':None,
                                       'plot_ylabel':None,
                                       'fill_cor':0,
                                       'plot_name':'',
                                       'temps':None,
                                       'name':'plot_gaussian_mixtures',
                                       'nice_name':'Plotting Gaussian Mixtures',
                                       'format':'png',
                                     }

        self.parameter_histograms = False
        self.corner = (np.array(self.plot_trace['modes']) == 3).any()


        # TODO is this still used? if not, DEL
        self.save_chains = None #  [0]
        self.save_likelihoods = [0]
        self.save_posteriors = [0]


    def run_plot_routines(self):
        self._plot_routines_check()

        res_max = self.model.get_attr_param('value_max', flat=True)

        #self.plot_gaussian_mixtures['name'] = 'plot_GM_Estimator'
        #self.plot_periodogram['name'] = 'plot_periodogram'

        if self.engine__.__name__ == 'reddemcee':
            setup = self.engine_config['setup']

            plot_betas_betas_last = self.sampler.betas
            plot_betas_betas_disc = self.sampler.get_betas(discard=self.reddemcee_discard)
            plot_betas_betas_full = self.sampler.get_betas()
            plot_betas_tsw_full = self.sampler.get_tsw()
            plot_betas_smd_full = self.sampler.get_smd()

            plot_trace_backend = self.sampler.backend[0]

        if self.engine__.__name__ == 'dynesty':
            setup = [1, 0, 0, 0]
            plot_betas_betas_last = None
            plot_betas_betas_disc = None
            plot_betas_betas_full = None
            plot_betas_tsw_full = None
            plot_betas_smd_full = None

            plot_trace_backend = self.sampler


        my_plot_kwargs = {
            'plot_posteriors':{'chains':self.chain,
                               'posts':self.posts,
                               #'options':self.plot_posteriors,
                               'my_model':self.model,
                               'ncores':self.cores__,
                               },
            'plot_histograms':{'chains':self.chain,
                               'posts':self.posts,
                               #'options':self.plot_histograms,
                               'my_model':self.model,
                               'ncores':self.cores__,
                               },
            'plot_keplerian_model':{'model_':self.model
                                   },
            'plot_betas':{'betas':plot_betas_betas_disc,
                          'logls':np.array([np.mean(self.likes[t]) for t in range(setup[0])]),
                          'Z':self.evidence,
                          #'options':self.plot_betas,
                          'setup':setup,
                          },

            'plot_rates':{'betas':plot_betas_betas_full,
                          'tsw':plot_betas_tsw_full,
                          'smd':plot_betas_smd_full,
                          'setup':setup,
                          'run_config':self.run_config,
                          },
            'plot_trace':{'sampler':plot_trace_backend,
                          'eng_name':self.engine__.__name__,
                          'my_model':self.model,
                          },
            'plot_beta_density':{'betas':plot_betas_betas_last},

                          }

        # TODO test arviz... again
        for plot_func in self.plot_all_list:
            time_plot_init = time.time()
            plot_name = plot_func['name']
            if plot_name not in ['plot_gaussian_mixtures', 'plot_periodogram']:
                if plot_func['plot']:
                    self.debug(f'run_plot_routines() : {plot_name} | {time.time()-self.time_init}')

                    self.logger('\n', save=False)
                    self.logger(plot_func['nice_name'], center=True, c='green', save=False)

                    plot_func['function'](options=plot_func,
                                          **my_plot_kwargs[plot_name])

                plot_func['time_to_plot'] = time.time() - time_plot_init    

        time_plot_init = time.time()


        if self.plot_gaussian_mixtures['plot'] and self.gaussian_mixtures_fit:
            self._plot_GM_per_block()

        self.time_plot_gm = time.time() - time_plot_init
        gc.collect()


    def _plot_GM_per_block(self):
        self.debug(f'postprocess() : PLOT GM | {time.time()-self.time_init}')

        self.logger('\n', save=False)
        self.logger(self.plot_gaussian_mixtures['nice_name'], center=True, c='green', save=False)
        pbar_tot = np.sum(b.ndim_ + len(b.additional_parameters) for b in self.model)

        pbar = tqdm(total=pbar_tot)
        for b in self.model:
            self.plot_gaussian_mixtures['fill_cor'] = b.bnumber_-1
            for p in b[b.C_]:
                self.plot_gaussian_mixtures['plot_name'] = f'{b.bnumber_} {p.GM_parameter.name}'

                plot_GM_Estimator(p.GM_parameter,
                                  options=self.plot_gaussian_mixtures)

                pbar.update(1)

            for p in b.additional_parameters:
                if p.has_posterior:
                    self.plot_gaussian_mixtures['plot_name'] = f'{b.bnumber_} {p.GM_parameter.name}'
                    plot_GM_Estimator(p.GM_parameter,
                                    options=self.plot_gaussian_mixtures)
                pbar.update(1)
        pbar.close()


    def _plot_routines_check(self):

        for pi in self.plot_all_list:
            if pi['name'] == 'plot_keplerian_model':
                    pi['use_c'] = self.use_c
                    pi['celerite'] = self.switch_celerite
                    pi['common_t'] = self.my_data_common_t


        if self.engine__.__name__ == 'reddemcee':
            for pi in self.plot_all_list:
                if pi['name'] in ['plot_posteriors', 'plot_histograms']:
                    valid_temps = np.arange(self.engine_config['setup'][0])
                    if pi['temps'] is None:
                        pi['temps'] = valid_temps
                    elif not np.all([t in valid_temps for t in pi['temps']]):
                        pi['temps'] = valid_temps


                if pi['name'] == 'plot_trace':
                    if self.FPTS:
                        pi['plot'] = False
                    else:
                        pi['temps'] = 0
                        pi['burnin'] = self.reddemcee_discard
                        pi['thin'] = self.reddemcee_thin

                


        if self.engine__.__name__ == 'dynesty':
            self.plot_posteriors['chain_alpha'] = 1.0
            for pi in self.plot_all_list:
                if pi['name'] in ['plot_posteriors', 'plot_histograms']:
                    pi['temps'] = np.arange(1)

                if pi['name'] in ['plot_betas',
                                  'plot_rates',
                                  'plot_beta_density']:
                    pi['plot'] = False

                if pi['name'] == 'plot_trace':
                    pi['temps'] = 0
                    pi['burnin'] = 0
                    pi['thin'] = 1
            

class emp_stats(object):
    def _init_criteria(self):
        self._crit_dict = {'chi2':self.foobarmin,
                           'chi2_red':self.foobarmin,
                           'AIC':self.foobarmin,
                           'BIC':self.foobarmin,
                           'DIC':self.foobarmin,
                           'HQIC':self.foobarmin,
                           'RMSE':self.foobarmin,
                           'post_max':self.foobarmax,
                           'like_max':self.foobarmax,
                           'BayesFactor':self.foobarmax,
                           'Evidence':self.foobarmax,
                           'Pass':self.foobarpass}

        self._crit_tolerance = 5
        self._crit_current = 'BIC'
        self._crit_compare_f = self.foobarmin
        self._crit_msg = ''

    def set_comparison_criteria(self, val):
        if val in [*self._crit_dict]:
            self._crit_current = val
            self._crit_compare_f = self._crit_dict[val]

        self.logger(f'Comparison Criteria set to {val}')
        

    def set_tolerance(self, val):
        self._crit_tolerance = val
        self.logger(f'Comparison Criteria Tolerance set to {val}')

    def foobarmin(self, new, old):
        new_d = np.round(new, 3)  # new
        old_d = np.round(old, 3)  # old

        self.logger(f'\npast {self._crit_current} - present {self._crit_current} > 5', c='blue')
        self._crit_msg = f'{old_d} - {new_d} > {self._crit_tolerance}'

        return old - new > self._crit_tolerance

    def foobarmax(self, new, old):
        new_d = np.round(new, 3)
        old_d = np.round(old, 3)

        self.logger(f'\npresent {self._crit_current} - past {self._crit_current} > 5', c='blue')
        self._crit_msg = f'{new_d} - {old_d} > {self._crit_tolerance}'

        return new - old > self._crit_tolerance

    def foobarpass(self, new, old):
        self._crit_msg = 'Criteria set to PASS'
        return True

    def criteria_compare(self, new, old):
        return self._crit_compare_f(new, old)

    def _stat_holder_update(self):
        self.stats_old_like_max = self.like_max
        self.stats_old_post_max = self.post_max
        self.stats_old_chi2 = self.chi2
        self.stats_old_AIC = self.AIC
        self.stats_old_BIC = self.BIC
        self.stats_old_DIC = self.DIC
        self.stats_old_HQIC = self.HQIC
        self.stats_old_RMSE = self.RMSE
        self.stats_old_RMSi = self.RMSi
        self.stats_old_Weights = self.Weights
        self.stats_old_BayesFactor = self.BayesFactor
        self.stats_old_Evidence = self.evidence[0]

    def _stats_calculate_statistics(self, residuals, err2):
        self.dof = self.model.ndata - self.model.ndim__
        self.chi2 = np.sum(residuals**2 / err2)
        self.chi2_red = np.sum(residuals**2 / err2) / self.dof
        self.RMSE = np.sqrt(np.sum(residuals ** 2) / self.model.ndata)

        FLAGS_ = self.my_data.Flag
        self.RMSi = [np.sqrt(np.sum(residuals[FLAGS_==(i+1)] ** 2) / len(residuals[FLAGS_==(i+1)])) for i in range(self.nins__)]
        self.Weights = [len(FLAGS_==(i+1)) / self.RMSi[i]**2 for i in range(self.nins__)]
        self.Total_Weight = np.sum(self.Weights)
        self.Weights /= self.Total_Weight

        self.AIC = 2 * self.model.ndim__ - 2 * self.like_max
        self.BIC = np.log(self.model.ndata) * self.model.ndim__ - 2 * self.like_max

        self.DIC = -2 * self.temp_like_func(self.fit_mean) + self.dic_aux

        self.Evidence = self.evidence[0]
        self.post_true = self.post_max - self.Evidence
        self.BayesFactor = self.like_max - self.Evidence

        self.HQIC = 2 * self.model.ndim__ * np.log(np.log(self.model.ndata)) - 2 * self.like_max


class model_manager(object):
    def add_keplerian(self):  # sourcery skip: remove-pass-body
        self.kplanets__ += 1
        if self.switch_inclination:
            kb = AstrometryKeplerianBlock(self.keplerian_parameterisation,
                                          self.kplanets__,
                                          self.use_c)

        else:
            kb = KeplerianBlock(self.keplerian_parameterisation,
                                self.kplanets__,
                                self.use_c)


        prargs = [self.eccentricity_limits, self.eccentricity_prargs]
        kw = {'prargs':prargs, 'dynamics':self.switch_dynamics,
              'kplan':self.kplanets__, 'starmass':self.starmass,
              'starmass_err':self.starmass_err,
              'dynamics_already_included':self.dynamics_already_included}
        
        # I like to insert in a specific position :s
        #self._limits_append_block(kb, kwargs=kw)
        
        self.SmartSetter(kb, **kw)  # sets limits and returns <extra_priors>
        self.blocks__.insert((self.kplanets__ - 1), kb)
        self.logger._add_block_message(kb.type_, kb.name_)


    def add_offset(self):
        ib = OffsetBlock(self.nins__)
        self._limits_append_block(ib)


    def add_jitter(self):
        ib = JitterBlock(self.nins__)
        jitter_args = [self.jitter_limits, self.jitter_prargs]
        self._limits_append_block(ib, args=jitter_args)


    def add_moav(self):
        ib = MOAVBlock(self.nins__, self.moav)
        
        self._limits_append_block(ib)

    def add_sai(self):
        if self.switch_SA_pro:
            self.cornum_signals = 1
            self.cornum_pro = 1

            ib = SAIPROBlock(self.cornum_signals, self.cornum_pro)
            ib.cornum_pro = self.cornum_pro
            self._limits_append_block(ib)
        else:    
            ib = SAIBlock(self.nins__, sa=self.cornums)
            ib.cornums = self.cornums
            self._limits_append_block(ib)


    def add_acceleration(self):
        ab = AccelerationBlock(self.acceleration)
        self._limits_append_block(ab)


    def add_sinusoid(self):
        sb = SinusoidBlock(self.sinusoid)
        self._limits_append_block(sb)

    def add_magnetic_cycle(self):
        sb = MagneticCycleBlock(self.magnetic_cycle)
        self._limits_append_block(sb)


    def add_celerite(self):
        if self.my_kernel['terms'][0] == 'GonzRotationTerm':
            self._write_kernel_spe()
            self._write_kernel_spe(in_func=True)
        else:
            self._write_kernel()
            self._write_kernel(in_func=True)

        kw = self.my_kernel
        ib = CeleriteBlock(nins=self.nins__, my_kernel=kw)
        
        self.SmartSetter(ib, *[], **kw)
        self.blocks__.append(ib)
        self.logger._add_block_message(ib.type_, ib.name_)



    def add_jitter_am(self):
        ib = AstrometryJitterBlock(2)  # Hip, Gaia
        # TODO proper jitter args
        # jitter_args = [self.jitter_limits, self.jitter_prargs]
        self._limits_append_block(ib)


    def add_offset_am(self):
        # TODO proper offset args
        ib = AstrometryOffsetBlock(5)
        self._limits_append_block(ib)


    def _limits_append_block(self, block, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        self.SmartSetter(block, *args, **kwargs)
        self.blocks__.append(block)
        self.logger._add_block_message(block.type_, block.name_)


class emp_counsil(object):
    def postprocess(self):
        time_postprocess_init = time.time()
        self.debug(f'postprocess() : INIT | {time.time()-self.time_init}')
        #def _postprocess_init(self):
        
        self._postprocess_setup()

        self._postprocess_set_samples()

        self._get_fit()

        self._calculate_statistics()

        self._set_p_values()

        self._set_additional_parameters()

        self._set_posteriors()

        self._print_posteriors()

        self._print_postproc_log()

        #self._save_stuff()

        self._save_chain_summary()
        
        self._save_latex()
        

        ##self._save_stats()

        #self.plots()

        #self.time_table()

        #self.log()

        #self.clean_run()

        self.time_postprocess = time.time() - time_postprocess_init


    def _postprocess_setup(self):
        getattr(self, f'_postprocess_setup_{self.engine__.__name__}')()
    

    def _postprocess_setup_reddemcee(self):
        # TODO test autocorr without hdf backend
        try:
            self.autocorr_time = self.sampler.get_autocorr_time(discard=self.reddemcee_discard,
                                                                thin=self.reddemcee_thin,
                                                                quiet=True,)
            self.autocorr_time_warn = False
            print('Autocorrelation tolerance=50 fails. Setting to 0.')
        except Exception:
            self.autocorr_time_warn = True
            self.autocorr_time = self.sampler.get_autocorr_time(discard=self.reddemcee_discard,
                                                                thin=self.reddemcee_thin,
                                                                quiet=True,
                                                                tol=0,)
        

    def _postprocess_setup_dynesty(self):
        # FAILSAFE CONFIGS
        # TODO move?
        #self.plot_betas['plot'] = False
        #self.plot_rates['plot'] = False
        #self.plot_posteriors['chain_alpha'] = 1.0
        pass

    
    def _postprocess_set_samples(self):
        getattr(self, f'_postprocess_set_samples_{self.engine__.__name__}')()
        

    def _postprocess_set_samples_reddemcee(self):
        ntemps = self.engine_config['setup'][0]
        self.reddemcee_dict = {'discard':self.reddemcee_discard,
                               'thin':self.reddemcee_thin,
                               'flat':True}

        raw_chain0 = self.sampler.get_chain(**self.reddemcee_dict)
        raw_likes0 = self.sampler.get_log_like(**self.reddemcee_dict)
        raw_posts0 = self.sampler.get_log_prob(**self.reddemcee_dict)

        if self.cherry['cherry']:
            self.chain = np.empty(ntemps, dtype=object)
            self.likes = np.empty(ntemps, dtype=object)
            self.posts = np.empty(ntemps, dtype=object)

            for t in range(ntemps):
                if self.cherry['median']:
                    mask = raw_posts0[t] > np.median(raw_posts0[t])
                elif self.cherry['diff']:
                    mask = max(raw_posts0[t]) - raw_posts0[t] <= self.cherry['diff']

                self.chain[t] = raw_chain0[t][mask]
                self.likes[t] = raw_likes0[t][mask]
                self.posts[t] = raw_posts0[t][mask]

        else:
            self.chain = raw_chain0
            self.likes = raw_likes0
            self.posts = raw_posts0

        # TODO move to... ?
        if self.switch_evidence:
            try:
                #zaux = self.sampler.thermodynamic_integration(discard=self.reddemcee_discard)
                zaux = getattr(self.sampler, self.evidence_method)(discard=self.reddemcee_discard)[:2]

            except Exception:
                print('Thermodynamic Integration interpolation failed! Using classic method...')
                zaux = self.sampler.thermodynamic_integration_classic(discard=self.reddemcee_discard)
            self.evidence = zaux[0], zaux[1]

        self.best_loc_post = np.argmax(self.posts[0])
        self.best_loc_like = np.argmax(self.likes[0])

    def _postprocess_set_samples_dynesty(self):
        if self.switch_evidence:
            self.evidence = (self.sampler['logz'][-1], self.sampler['logzerr'][-1])

        self.chain = [self.sampler['samples']]
        self.likes = [self.sampler['logl']]
        self.posts = [self.sampler['logl'] - self.evidence[0]]

        self.best_loc_post = -1
        self.best_loc_like = -1

    def _get_fit(self):
        getattr(self, f'_get_fit_{self.engine__.__name__}')()


    def _get_fit_reddemcee(self):
        self.post_max = self.posts[0][self.best_loc_post]
        self.like_max = self.likes[0][self.best_loc_post]
        self.prior_max_post = self.post_max - self.likes[0][self.best_loc_post]

        self.sigmas = np.std(self.chain[0], axis=0)

        self.fit_max = self.chain[0][self.best_loc_post]
        self.fit_mean = np.mean(self.chain[0], axis=0)
        self.fit_median = np.median(self.chain[0], axis=0)


        self.fit_low1, self.fit_high1 = hdi_of_chain(self.chain[0], 0.36).T
        self.fit_low2, self.fit_high2 = hdi_of_chain(self.chain[0], 0.90).T
        self.fit_low3, self.fit_high3 = hdi_of_chain(self.chain[0], 0.95).T


        self.fit_maxlike = self.chain[0][self.best_loc_like]

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


    def _get_fit_dynesty(self):
        # TODO make a different method, leave for now
        ###### THIS CAN BE SHARED
        self._get_fit_reddemcee()
        r = self.sampler
        samples = r.samples
        weights = np.exp(r.logwt - r.logz[-1])
        weights /= np.sum(weights)

        from dynesty.utils import quantile

        all_quants = []
        all_quants.extend(
            quantile(samples[:, i], [0.16, 0.84, 0.1, 0.9, 0.05, 0.95], weights)
            for i in range(samples.shape[1]))

        all_quants = np.array(all_quants)
        self.fit_low1, self.fit_high1 = all_quants[:, 0], all_quants[:, 1]
        self.fit_low2, self.fit_high2 = all_quants[:, 2], all_quants[:, 3]
        self.fit_low3, self.fit_high3 = all_quants[:, 4], all_quants[:, 5]


        '''
        r = sim.sampler
        weights = np.exp(r.logwt - r.logz[-1])
        weights /= np.sum(weights)

        for i in range(8):
            ...:     print(dynesty.utils.quantile(samples[:, i], [0.36, 0.5, 0.64], weights))
        vals = [[4.2307702768102, 4.230783913236789, 4.230797624141973],
        [55.5163041392088, 55.709485945761706, 55.9056472781712],
        [7.180579681830706, 7.3553242075764675, 7.585150916162191],
        [0.0072205583259223535, 0.010166281858242752, 0.013384756270929339],
        [0.8254052515045534, 1.0848266869015135, 1.4278938384403816],
        [-0.004576031733818643, -0.004389921887157595, -0.004205252616377426],
        [5.1502626959312945, 5.3115763243581355, 5.474122164860442],
        [0.7824368659647362, 1.0536187976773366, 1.326083745360509],
        ]

        for v in vals:
            print(f'{v[1]:.9f}+{v[2]-v[1]:.9f}-{v[1]-v[0]:.9f}')
        '''


        pass


    def _calculate_statistics(self):
        if self.switch_RV:
            self.dic_aux = np.var(-2 * self.likes[0])
            ymod, err2 = self.temp_model_func(self.ajuste)
            residuals = self.my_data['RV'].values - ymod

            res_table = self.my_data.values.T
            res_table[1] = residuals

            np.savetxt(f'{self.saveplace}/restore/residuals.dat',
                        res_table)

            self._stats_calculate_statistics(residuals, err2)


    def _set_p_values(self):
        for b in self.blocks__:
            for p in b[b.C_]:
                self._update_p_free(p)
            
            for p in b[b.A_]:
                self._update_p_fixed(p)


    def _set_additional_parameters(self):
        # TODO clean this up, looks awful
        # probably use get_PAE method from model
        self.sma = []
        self.mm = []

        self.sma_sig = []
        self.mm_sig = []

        extra_names = []
        extra_chains = []

        # Get extra chains

        ch0 = self.chain[0]
        for b in self.model:
            ## PLANET SIGNATURES
            if b.type_ == 'Keplerian':
                my_params = [None, None, None, None, None]
                if b.astrometry_bool:
                    my_params.extend([None, None])

                for p in b[b.C_]:
                    my_params[p.C_] = ch0.T[p.cpointer]

                for p in b[b.A_]:
                    my_params[p.A_] = p.value * np.ones(len(ch0))

                if True:
                    if b.parameterisation == 0:
                        if b.astrometry_bool:
                            per, A, phase, ecc, w, inc, Ome = b.get_attr('value')
                            per_, A_, phase_, ecc_, w_, inc_, Ome_ = my_params
                        else:
                            per, A, phase, ecc, w = b.get_attr('value')
                            per_, A_, phase_, ecc_, w_ = my_params

                    elif b.parameterisation == 1:
                        per, A, phase, S, C = b.get_attr('value')
                        per_, A_, phase_, S_, C_ = my_params

                        ecc, w = delinearize(S, C)
                        ecc_, w_ = adelinearize(S_, C_)

                        for thingy in ['Eccentricity', 'Longitude_Periastron']:
                            extra_names.append(thingy+'_{}'.format(b.number_))

                        for thingy in [ecc_, w_]:
                            extra_chains.append(thingy)


                    elif b.parameterisation == 2:
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

                    elif b.parameterisation == 3:
                        per, A, t0, ecc, w = b.get_attr('value')
                        per_, A_, t0_, ecc_, w_ = my_params

                    elif b.parameterisation == 4:
                        per, A, t0, S, C = b.get_attr('value')
                        per_, A_, t0_, S_, C_ = my_params

                        ecc, w = delinearize(S, C)
                        ecc_, w_ = adelinearize(S_, C_)

                        for thingy in ['Eccentricity', 'Longitude_Periastron']:
                            extra_names.append(thingy)
                        for thingy in [ecc_, w_]:
                            extra_chains.append(thingy)

                    elif b.parameterisation == 6:
                        per, A, phase, ecc, w = b.get_attr('value')
                        per_, A_, phase_, ecc_, w_ = my_params

                        per = np.exp(per)
                        per_ = np.exp(per_)

                        for thingy in ['Period']:
                            extra_names.append(thingy+'_{}'.format(b.number_))

                        for thingy in [per_]:
                            extra_chains.append(thingy)


                    elif b.parameterisation == 7:
                        P, A, phase, S, C = b.get_attr('value')
                        P_, A_, phase_, S_, C_ = my_params

                        per = np.exp(P)
                        per_ = np.exp(P_)

                        ecc, w = delinearize(S, C)
                        ecc_, w_ = adelinearize(S_, C_)

                        for thingy in ['Period', 'Eccentricity', 'Longitude_Periastron']:
                            extra_names.append(thingy+'_{}'.format(b.number_))

                        for thingy in [per_, ecc_, w_]:
                            extra_chains.append(thingy)


                    if self.starmass:
                        sma, mm = cps(per, A, ecc, self.starmass)  # No starmass error
                        sma_, mm_ = cps(per_, A_, ecc_, self.starmass, self.starmass_err)

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

        ## Set p.values and sigma for extra params
        if True:
            ## **RED w/ set_attr\
            jj = 0
            for b in self.model:
                for p in b.additional_parameters:
                    if p.has_posterior:
                        ch = extra_chains[jj]
                        self._update_p_extra(p, ch)
                        jj += 1

        self.extra_chains = extra_chains


    def _update_p_free(self, p):
        j = p.cpointer
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


    def _set_posteriors(self):
        # TODO postprocess config dict?
        subscript_nums = ['', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
        if self.posterior_fit_method == 'GM':
            self._set_gaussian_mixtures(self.chain[0], self.extra_chains)
            for b in self.blocks__:
                for p in b[b.C_]:
                    p.posterior = p.GM_parameter
                    if p.GM_parameter.n_components == 1:
                        mu = np.round(p.GM_parameter.mixture_mean, 3)
                        sig = np.round(p.GM_parameter.mixture_sigma, 3)
                        p.display_posterior = f'~𝓝 ({mu}, {sig})'
                    elif p.GM_parameter.n_components > 1:
                        mu = np.round(p.GM_parameter.mixture_mean, 3)
                        sig = np.round(p.GM_parameter.mixture_sigma, 3)

                        p.display_posterior = f'𝛴{subscript_nums[p.GM_parameter.n_components]}~~𝓝 ({mu}, {sig})'
                    else:
                        print('Something really weird is going on! Error 110.')
                for p in b[b.A_]:
                    p.posterior = p.GM_parameter
                    p.display_posterior = '~𝛿 (x - {p.value})'

                for p in b.additional_parameters:
                    if p.has_posterior:
                        p.posterior = p.GM_parameter
                        if p.GM_parameter.n_components == 1:
                            mu = np.round(p.GM_parameter.mixture_mean, 3)
                            sig = np.round(p.GM_parameter.mixture_sigma, 3)
                            p.display_posterior = '~𝓝 ({}, {})'.format(mu, sig)
                        elif p.GM_parameter.n_components > 1:
                            mu = np.round(p.GM_parameter.mixture_mean, 3)
                            sig = np.round(p.GM_parameter.mixture_sigma, 3)

                            p.display_posterior = f'𝛴{subscript_nums[p.GM_parameter.n_components]}~~𝓝 ({mu}, {sig})'
                        else:
                            print('Something really weird is going on! Error 110.')

                    pass
            pass


    def _set_gaussian_mixtures(self, cold_chain, extra_chains):
        time_gm_init = time.time()
        self.logger('Calculating Gaussian Mixtures', center=True, c='green', save=False)

        pbartot = np.sum(b.ndim_ + len(b.additional_parameters) for b in self.model)
        pbar = tqdm(total=pbartot)
        
        for b in self.blocks__:
            for p in b[b.C_]:
                p.GM_parameter = GM_Estimator().estimate(cold_chain[:, p.cpointer],
                                                            p.name, p.unit)
                pbar.update(1)
            for p in b[b.A_]:
                p.GM_parameter = p.value
                
        # ADDITIONAL PARAMETERS
        for b in self.model:
            jj = 0
            for p in b.additional_parameters:
                if p.has_posterior:
                    ch = extra_chains[jj]
                    p.GM_parameter = GM_Estimator().estimate(ch, p.display_name, p.unit)
                    jj += 1
                    pbar.update(1)

        pbar.close()
        self.time_calc_gm = time.time() - time_gm_init


    def _print_posteriors(self):
        self.debug(f'postprocess() : PRINT POSTERIORS | {time.time()-self.time_init}')
        self.logger._add_subtitle('Best Fit', postprocess=True)

        switch_title = True

        #param_length = max(len(n) for n in self.model.get_attr_param('name', flat=True))
        #value_length = max(len(n) for n in self.model.get_attr_param('value_max', flat=True))
        #range_length = max(len(str(ran)) for ran in self.model.get_attr_param('value_range', flat=True))

        #prior_length = max(len(n) for n in self.model.get_attr_param('display_prior', flat=True))
        #limits_length = max(len(str(lim)) for lim in self.model.get_attr_param('limits', flat=True))


        to_get = ['name',
                    'value_max',
                    'value_range',
                    'display_prior',
                    'limits']
        for b in self.model:
            
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


    def _print_postproc_log(self):
        self._print_1_basic_info()

        self._print_2_run_info()

        self._print_3_statistics()


    def _print_1_basic_info(self):
        self.logger._add_subtitle('Run Info', double=True, postprocess=True)
        # LOGGER
        if self.engine__.__name__ == 'reddemcee':
            setup_info = 'Temps, Walkers, Sweeps, Steps  : '
            setup_info2 = list(self.engine_config['setup'])
            size_info = [len(self.chain[t]) for t in range(len(self.chain))]
        elif self.engine__.__name__ == 'dynesty':
            setup_info = 'Live Points                       : '
            setup_info2 = list(self.sampler['batch_nlive'])
            size_info = self.sampler['niter']

        tabh_1 = ['Info                             ', 'Value                       ']
        tab_1 =    [['Star Name                      : ', self.starname],
                    ['The sample sizes are           : ', size_info],
                    [setup_info, setup_info2],
                    ['Model used is                  : ', str(self.model)],
                    ['N data                         : ', self.model.ndata],
                    ['t0 epoch is                    : ', self.my_data_common_t],
                    ['Number of Dimensions           : ', self.model.ndim__],
                    ['Degrees of Freedom             : ', self.dof]
                    ]

        self.logger.line()

        self.logger(tabulate(tab_1, headers=tabh_1))
        self.logger._add_subtitle('----------------------------------------', center=False)


    def _print_2_run_info(self):
        if self.engine__.__name__ != 'reddemcee':
            return
        adapt_ts = self.engine_config['adapt_tau']
        adapt_ra = self.engine_config['adapt_nu']
        adapt_sc = self.engine_config['adapt_mode']

        # TODO acceptance fraction with discard!!
        discard0 = self.reddemcee_discard
        nsteps0 = self.engine_config['setup'][-1]

        mean_af = np.mean(self.sampler.acceptance_fraction, axis=1)

        self.logger('\nDecay Timescale, Rate, Scheme   :   ' + f'{adapt_ts}, {adapt_ra}, {adapt_sc}', save_extra_n=True)
        self.logger('\nBeta Detail                     :   ' + '[' + ', '.join('{:.4}'.format(x) for x in self.sampler.betas) + ']', save_extra_n=True)
        self.logger('\nMean Logl Detail                :   ' + '[' + ', '.join('{:.3f}'.format(np.mean(x)) for x in self.likes) + ']', save_extra_n=True)
        self.logger('\nMean Acceptance Fraction        :   ' + '[' + ', '.join('{:.3f}'.format(x) for x in mean_af) + ']', save_extra_n=True)
        self.logger('\nAutocorrelation Time            :   ' + '[' + ', '.join('{:.3f}'.format(x) for x in self.autocorr_time[0]) + ']', save_extra_n=True)

        if self.engine_config['tsw_history']:
            mean_tsw = np.mean(self.sampler.get_tsw(discard=discard0//nsteps0), axis=0)
            self.logger('\nTemperature Swap Rate           :   ' + '[' + ', '.join('{:.3f}'.format(x) for x in mean_tsw) + ']', save_extra_n=True)
        if self.engine_config['smd_history']:
            mean_smd = np.mean(self.sampler.get_smd(discard=discard0//nsteps0), axis=0)
            self.logger('\nMean Swap Distance              :   ' + '[' + ', '.join('{:.3f}'.format(x) for x in mean_smd) + ']', save_extra_n=True)
            

    def _print_3_statistics(self):
        tabh_2 = ['Statistic                  ', 'Value']

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
        
        if self.switch_evidence:
            x = [['The evidence is             :    ', '%.3f +- %.3f' % self.evidence]]
            tab_2 = np.vstack([x, tab_2])

        self.logger._add_subtitle('Statistical Details', double=True, postprocess=True)
        self.logger(tabulate(tab_2, headers=tabh_2))
        self.logger.dline()


    def _save_log(self):
        if self.save_log:
            self.logger.saveto(self.saveplace,
                               name=f"_{self.saveplace_run.split('_')[-1]}_{self.starname}")


    def _save_simple_log(self):
        if not self.save_log_simple:
            return

        self._save_simple_stats()

        self._save_simple_fit()


    def _save_simple_stats(self):
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

        stat_box_names = np.array([r'$\log{Z}$',
                                   r'$\log{P}$',
                                   r'$\log{L}$',
                                    'BIC ',
                                   r'$chi^{2}_{\nu}$',
                                    'RMSE'])
        stat_box_value = np.array([f"{self.evidence[0]:.3f} \\pm {self.evidence[1]:.3f}",
                                   f'{self.post_max:.3f}',
                                   f'{self.like_max:.3f}',
                                   f'{self.BIC:.3f}',
                                   f'{self.chi2_red:.3f}',
                                   f'{self.RMSE:.3f}'])
        stat_box = [stat_box_names,
                    stat_box_value]
        stat_box = adjust_table_tex(stat_box, rounder=8)
        stat_box_header = ['Stat', 'Value']

        df0 = pd.DataFrame(np.array(stat_box).T, columns=stat_box_header)
        
        latex_table = df0.to_latex(index=False, escape=False)
        with open(f'{self.saveplace}/tables/stats.tex', 'w') as f:
            f.write(latex_table)


    def _save_simple_fit(self):
        simple_log = np.array(self.model.get_attr_param('name', flat=True))
        simple_log = np.vstack([simple_log,
                                ['{:.3f}'.format(x) for x in np.array(self.model.get_attr_param('value', flat=True))]
                                ])

        np.savetxt(f'{self.saveplace}/best_fit.dat', simple_log.T, fmt='%s', delimiter='\t')


    def _save_chain_summary(self):
        cs_names = self.model.get_attr_param('name', flat=True)
        max_length = max(len(item) for item in cs_names)
        cs_names = np.array([item.ljust(max_length) for item in cs_names])
        #summary_rounder = 8
        #attrs = ["value_max_lk", "value_max", "value_mean", "sigma",
        #         "value_low3", "value_low2", "value_low1",
        #         "value_median", "value_high1", "value_high2", "value_high3"]
        #get = self.model.get_attr_param


        cs = np.array([cs_names,
            np.round(self.model.get_attr_param('value_max_lk', flat=True), 8),
            np.round(self.model.get_attr_param('value_max', flat=True), 8),
            np.round(self.model.get_attr_param('value_mean', flat=True), 8),
            np.round(self.model.get_attr_param('sigma', flat=True), 8),
            np.round(self.model.get_attr_param('value_low3', flat=True), 8),
            np.round(self.model.get_attr_param('value_low2', flat=True), 8),
            np.round(self.model.get_attr_param('value_low1', flat=True), 8),
            np.round(self.model.get_attr_param('value_median', flat=True), 8),
            np.round(self.model.get_attr_param('value_high1', flat=True), 8),
            np.round(self.model.get_attr_param('value_high2', flat=True), 8),
            np.round(self.model.get_attr_param('value_high3', flat=True), 8),
            ])
        

        for b in self.model:
            if len(b.additional_parameters):
                mask = [x.has_posterior for x in b.additional_parameters]
                for p in np.array(b.additional_parameters)[mask]:
                    h = np.array([p.name.ljust(max_length),
                         np.round(p.value_max_lk, 8),
                         np.round(p.value_max, 8),
                         np.round(p.value_mean, 8),
                         np.round(p.sigma, 8),
                         np.round(p.value_low3, 8),
                         np.round(p.value_low2, 8),
                         np.round(p.value_low1, 8),
                         np.round(p.value_median, 8),
                         np.round(p.value_high1, 8),
                         np.round(p.value_high2, 8),
                         np.round(p.value_high3, 8),
                         ])[:, None]
                    cs = np.hstack([cs, h]).tolist()
                    


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


    def _save_latex(self):
        # save param_minimal.dat
        par_box_names = self.model.get_attr_param('name', flat=True)
        
        v0 = np.array(self.model.get_attr_param('value_range', flat=True))[:, 0]
        v1 = self.model.get_attr_param('value_max', flat=True)
        v2 = np.array(self.model.get_attr_param('value_range', flat=True))[:, 1]

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

        # save latex
        df_latex = df0[['Parameter', 'Value']]
        
        latex_table = df_latex.to_latex(index=False, escape=False)

        with open(f'{self.saveplace}/tables/values.tex', 'w') as f:
            f.write(latex_table)


    def _get_time_table(self):
        label_width = 27
        self.logger.dline()
        self.logger('Time Table', save_extra_n=True)
        self.logger.message_width("Time RUN", f": {sec_to_clock(self.time_run)}", label_width)
        self.logger.message_width("Time POSTPROCESS", f": {sec_to_clock(self.time_postprocess)}", label_width)
        self.logger.message_width("Time CALCULATE GM", f": {sec_to_clock(self.time_calc_gm)}", label_width)


        for plot_dict in self.plot_all_list:
            if plot_dict['time_to_plot']:
                self.logger.message_width(f"Time {plot_dict['name']}", f": {sec_to_clock(plot_dict['time_to_plot'])}", label_width)


class Simulation(emp_retainer, model_manager,
                 emp_painter, emp_scribe, emp_stats, emp_counsil):
    def __init__(self):
        self.time_init = time.time()

        self._init_default_config()
    
        self.logger = reddlog()

        self._init_others_config()

        self._init_model_config()

        self._init_plot_config()

        self._init_postprocess_config()

        self._init_criteria()
        self._stat_holder_update()

        self._sampler_config()


    def load_data(self, folder_name):
        self.starname = folder_name
        self.data_wrapper = DataWrapper(folder_name,
                                        read_loc=self.read_loc)
        self.logger('\n')

        for m in self.data_wrapper:
            if m['use']:
                setattr(self, f"switch_{m['KEY']}", True)
                self.logger(m['logger_msg'], center=True, c='blue')
            self.logger('\n')

        if self.data_wrapper['RV']['use']:
            self._load_data_RV()

        if self.data_wrapper['AM']['use']:
            self.switch_AM = True
            self.switch_inclination = True
            
            self.data_AM = self.data_wrapper['AM']
            #to_get = ['ref_epoch','ra','dec','parallax','pmra','pmdec','radial_velocity']
            #self.AM_astro_array = self.data_wrapper['AM']['df_hg123'][to_get].values
            #self.data_AM = {'astro_array':self.AM_astro_array,
            #                }
            #self.switch_AM_cata = self.data_wrapper['AM']['AM_cata']
        else:
            self.data_AM = None

        # TODO: make sure inputs correspond with RV|AM|PM
        self.SmartSetter = SmartSetter(self.my_data)


    def _load_data_RV(self):
        if not self.switch_RV:
            return
        
        dw_labels = self.data_wrapper['RV']['labels']
        if len(dw_labels) > 0:
            self.nins__ = len(dw_labels)

        if self.instrument_names_RV is None:
            self.instrument_names_RV = dw_labels

        self.my_data = self.data_wrapper.get_data__()
        self.my_data_common_t = self.data_wrapper['RV']['common_t']

        #self.my_data_reduc = self.my_data.values[:, 0:3].T

        self.cornums = self.data_wrapper['RV']['nsai']

        if self.switch_SA:
            if np.sum(self.cornums) > 0:
                pass
        else:
            self.my_data = self.my_data[['BJD', 'RV', 'eRV', 'Flag']]
            self.cornums = [0 for _ in self.cornums]


    def set_engine(self, eng):
        setattr(self, f'{eng}_config', {'name':eng})

        if eng == 'emcee':
            import emcee
            self.engine__ = emcee

        elif eng == 'dynesty':
            import dynesty
            self.engine__ = dynesty
            self.engine_config = {'nlive':None,
                                  'bound':'multi',
                                  'sample':'auto'}

        elif eng == 'dynesty_dynamic':
            self._set_engine_dynesty()
            

        elif eng == 'pymc3':
            import pymc3 as pm
            self.engine__ = pm
 
        elif eng == 'reddemcee':
            self._set_engine_reddemcee()


        else:
            print(f'Failed to set {eng} as engine.')


    def _set_engine_dynesty(self):
        import dynesty
        self.engine__ = dynesty
        self.engine__args = 'dynamic'
        self.general_dependencies.append('import dynesty')
        self.engine_config = {'nlive':None,
                                'bound':'multi',
                                'sample':'auto'}
        
        self.run_config = {'maxiter':None,
                           'maxcall':None,
                           'dlogz_init':0.01,
                           'print_progress':True}


    def _set_engine_reddemcee(self):
        import reddemcee
        self.engine__ = reddemcee
        self.general_dependencies.extend(['import reddemcee',
                                          'import emcee',
                                          'import logging'])
        if self.FPTS:
            self.general_dependencies.extend(['import astroemperor.fpts as fpts'])


        self.engine_config = {'setup':np.array([5, 100, 500, 2]),
                              'ntemps':5,
                              'betas':None,
                              'moves':None,
                              'tsw_history':True,
                              'smd_history':True,
                              'adapt_tau':1000,
                              'adapt_nu':1,
                              'adapt_mode':0,
                              'progress':True,
                              }
        # TODO burnin in int is for sweeps, should be in niter
        self.run_config = {'burnin':None,
                           'adaptation_batches':None,
                           'adaptation_nsweeps':None,
                           'thin':1,
                           'discard':0.5,
                           'logger_level':'CRITICAL',
                            }
        

    def update_model(self):
        self.model = ReddModel(self.blocks__,
                               data_RV=self.my_data,
                               data_AM=self.data_AM)
        self.model.instrument_names_RV = self.instrument_names_RV
        self.model.refresh__()

        if self.switch_AM:
            self.model.switch_AM = True
            self.model.data_wrapper = self.data_wrapper
            self.model.starmass = self.starmass

        # additional dependencies?
        # TODO add beta, and some for dynesty
        # additional constants?
        # TODO set this in just one place
        #self.model_constants['A_'] = list(self.model.A_)
        #self.model_constants['mod_fixed_'] = list(self.model.mod_fixed)
        #self.model_constants['cornums'] = list(self.cornums)

        #self.model.model_constants['cornums'] = list(self.cornums)


    def autorun(self, k_start=0, k_end=10):
        self.k_start = k_start
        self.k_end = k_end

        self._autorun_check()

        while self.k_start <= self.k_end:
            self._autorun_add_blocks()

            self._stat_holder_update()

            self.run()  # also loads

            self.postprocess()

            self.run_plot_routines()
            
            self._get_time_table()

            self._save_simple_log()
            self._save_log()

            self.k_start += 1



            if self._check_continue_run():
                self.logger.end_run_message()
                break

            # free some memory
            self._delete_backends()
            self._clear_samples()


            self.logger.next_run_message()
            #self.logger._add_subtitle('~~~~~~~~~~~~~~~')
            self._set_init_pos()

            self._constrain_next_run()

            self._load_next_model()



            # CLEAN RUNS
            # CLEAN LOGS! or make a clear separator


    def run(self):
        time_run_init = time.time()

        self._prepare_run()

        self.prerun_logger()

        self._run_engine_specific()  # also loads

        # move the run files
        self._run_clean()

        # Load the sampler
        self._load_sampler()

        self.time_run = time.time() - time_run_init


    def _prepare_run(self):
        self.debug.debug_snapshot()
        self.debug(f'run  : begin | {time.time()-self.time_init}')

        if self.constrain['method'] not in self.constrain['known_methods']:
            msg = f"Invalid constrain[method] = {self.constrain['method']}"
            raise SyntaxError(msg)

        if self.constrain['method'] == 'GM' and not self.gaussian_mixtures_fit:
            msg = "Invalid constrain[method] = GM with .gaussian_mixtures_fit = False"
            raise SyntaxError(msg)

        # PRE-CLEAN
        self.sampler = None
        # BURN-IN and THIN

        self.saveplace = create_directory_structure(self.starname, base_path=self.save_loc, k=self.kplanets__, first=self.switch_first_run)
        self.saveplace_run = self.saveplace.split('/')[-2]
        # Merge general plot dict with each 
        for plot_dict in self.plot_all_list:
            getattr(self, plot_dict['name'])['saveloc'] = self.saveplace
            setattr(self, plot_dict['name'], {**self.plot_all, **getattr(self, plot_dict['name'])})

        self.temp_script = f'temp_script_{self.starname}_{self.saveplace_run}.py'

        self.apply_conditions()
        self.update_model()

        getattr(self, f'_prepare_run_{self.engine__.__name__}')()
        

    def _prepare_run_reddemcee(self):
        ntemps, nwalkers, nsweeps, nsteps = self.engine_config['setup']
        niter = int(nsteps * nsweeps)
        if type(self.run_config['burnin']) == float:
            self.reddemcee_discard = int(self.run_config['burnin']*niter)
        elif type(self.run_config['burnin']) == int:
            self.reddemcee_discard = self.run_config['burnin']
        elif self.run_config['burnin'] is None:
            self.reddemcee_discard = 0

        if False:#self.run_config['adaptation_batches']:
            adapt_iter = self.run_config['adaptation_nsweeps'] * nsteps
            self.adaptation_niter = self.run_config['adaptation_batches'] * adapt_iter
            self.reddemcee_discard += self.adaptation_niter

        self.reddemcee_thin = self.run_config['thin']


    def _prepare_run_dynesty(self):
        # TODO: ptformargs here!
        # TODO: add other priors?
        self.ptformargs0 = []
        self.ptformargs1 = []
        for b in self.blocks__:
            for p in b[b.C_]:
                # PERIOD AS LOG
                if False:
                    if p.name[:3] == 'Per':
                        p.prior = 'Jeffreys'

                if p.ptformargs == None:
                    l, h = p.limits
                    p.ptformargs = [(h-l)/2., (h+l)/2.]
                    if b.parameterisation in [1, 2, 4, 7]:
                        if p.name[:3] == 'Ecc' and p.ptformargs[0] > 0.707:
                            p.ptformargs[0] = 0.707
                else:
                    s, c = p.ptformargs
                    p.limits = [c-s, c+s]
                self.ptformargs0.append(p.ptformargs)


    def _run_engine_specific(self):
        self.debug(f'run  : init sampler | {time.time()-self.time_init}')
        getattr(self, f'_run_engine_{self.engine__.__name__}')()


    def _run_engine_reddemcee(self):
        '''Writes the script based on Model object.
        Loads the written model for... plotting?
        Runs the script.
        Loads the sampler from script to emp.
        '''
        # Constants for run
        ntemps, nwalkers, nsweeps, nsteps = self.engine_config['setup']
        
        # Assert
        assert self.nins__ == len(self.instrument_names_RV), f'instrument_names should have {self.nins__} items'
        if self.engine_config['betas'] is not None:
            assert len(self.engine_config['betas']) == ntemps, f'betas should have {ntemps} items'
        # Write Script
        
        self.debug(f'run  : write_script() | {time.time()-self.time_init}')
        self.write_script()
        

        # Run Script
        self.debug(f'run  : os <run temp_script.py> | {time.time()-self.time_init}')
        self.logger('Generating Samples', center=True, save=False, c='green', attrs=['reverse'])
        self.logger.line()
        os.system(f'ipython {self.temp_script}')


    def _run_engine_dynesty(self):
        self.write_script()

        # RUN SAMPLER
        os.system(f'ipython {self.temp_script}')

        
    def _run_clean(self):
        # temp script
        self.debug(f'clean_run() : CLEANING.. | {time.time()-self.time_init}')
        os.system(f'mv {self.temp_script} {self.saveplace}/temp/{self.temp_script}')

        # backends
        if self.engine__.__name__ == 'reddemcee':
            os.system(f'mv {self.backend_name}.h5 {self.saveplace}/restore/backends/{self.backend_name}.h5')
            for t in range(self.engine_config['setup'][0]):
                os.system(f'mv {self.backend_name}_{t}.h5 {self.saveplace}/restore/backends/{self.backend_name}_{t}.h5')

        if self.engine__.__name__ == 'dynesty':
            os.system(f'mv {self.backend_name}.pkl {self.saveplace}/restore/backends/{self.backend_name}.pkl')

        gc.collect()


    def _autorun_check(self):
        self.debug(f'run_auto : INIT run_auto | {time.time()-self.time_init}')
        assert self.k_start <= self.k_end, f'Invalid keplerian starting point: ({self.k_start}, {self.k_end})'


    def _autorun_add_blocks(self):
        if self.switch_first_run:
            if self.switch_RV:
                self._autorun_add_RV_ins()

                if self.k_start > 0:
                    for _ in range(self.k_start):
                        self.add_keplerian()

            if self.switch_AM:
                self.add_offset_am()
                self.add_jitter_am()


    def _autorun_add_RV_ins(self):
        if self.acceleration:
            self.add_acceleration()

        self.add_offset()

        if self.switch_jitter:
            self.add_jitter()

        if self.moav['order']:
            self.add_moav()

        if self.switch_celerite:
            self.add_celerite()
                
        if self.switch_SA and np.sum(self.cornums) > 0:
            self.add_sai()

        if self.sinusoid:
            self.add_sinusoid()

        if self.magnetic_cycle:
            self.add_magnetic_cycle()



    def _check_continue_run(self):
        new_crit = getattr(self, f'{self._crit_current}')
        old_crit = getattr(self, f'stats_old_{self._crit_current}')
        if self.criteria_compare(new_crit, old_crit):
            self.logger(f'\n {self._crit_current} condition met!!', c='blue', attrs=['bold'])
            self.logger(self._crit_msg, c='blue')
        else:
            self.logger('\nBIC condition not met', c='blue')
            self.logger(self._crit_msg, c='blue')
            return True

        if self.k_start > self.k_end:
            return True
        
        return False


    def _set_init_pos(self):
        for b in self.blocks__:
            if b.type_ == 'Keplerian':
                for p in b[b.C_]:
                    p.init_pos = p.value_range


    def _delete_backends(self):
        if not self.save_backends:
            if self.engine__.__name__ == 'reddemcee':
                ext = 'h5'
            elif self.engine__.__name__ == 'dynesty':
                ext = 'pkl'
            else:
                print(f'BACKENDS NOT DELETED FOR ENGINE {self.engine__.__name__}')

            os.system(f'rm {self.saveplace}/restore/backends/*.{ext}')


    def _clear_samples(self):
        getattr(self, f'_clear_samples_{self.engine__.__name__}')()
        

    def _clear_samples_reddemcee(self):
        ntemps = self.engine_config['setup'][0]
        self.chain = np.empty(ntemps, dtype=object)
        self.likes = np.empty(ntemps, dtype=object)
        self.posts = np.empty(ntemps, dtype=object)

    def _clear_samples_dynesty(self):
        ntemps = 1
        self.chain = np.empty(ntemps, dtype=object)
        self.likes = np.empty(ntemps, dtype=object)
        self.posts = np.empty(ntemps, dtype=object)


    def _constrain_next_run(self):
        # TODO constrain options as dict
        constrain_method = self.constrain['method']
        if constrain_method=='None':
            return
        if constrain_method in self.constrain['known_methods']:
            getattr(self, f'_apply_constrain_{constrain_method}')()
        else:
            print(f'ERROR: Constrain method {constrain_method} not identified')


    def _apply_constrain_sigma(self):
        for b in self.blocks__:
            if b.type_ in self.constrain['types']:
                for p in b[b.C_]:
                    pval = p.value
                    psig = p.sigma

                    limf = pval - self.constrain['sigma']*psig
                    limc = pval + self.constrain['sigma']*psig

                    limc = min(limc, p.limits[1])
                    if (limf < p.limits[0] or
                        b.type_ == 'Jitter'):
                        limf = p.limits[0]

                    rang = limc - limf

                    if rang / abs(pval) < self.constrain['tol']:
                        print(f'Not further constraining {p.name}.')
                    else:
                        self.add_condition([p.name, 'limits', [limf, limc]])


    def _apply_constrain_GM(self):
        # TODO replace count with p.cpoint?
        count = 0
        for b in self.blocks__:
            if b.type_ in self.constrain['types']:
                for p in b[b.C_]:
                    if p.GM_parameter.n_components == 1:
                        prarg0 = [p.GM_parameter.means[0], p.GM_parameter.sigmas[0]]

                        self.add_condition([p.name, 'prior', 'Normal'])
                        self.add_condition([p.name, 'prargs', prarg0])

                    elif p.GM_parameter.n_components > 1:
                        self.add_condition([p.name, 'prior', 'GaussianMixture'])
                        self.add_condition([p.name, 'prargs', [self.model.C_[count]]])
                    count += 1


    def _apply_constrain_range(self):
        for b in self.blocks__:
            if b.type_ in self.constrain['types']:
                for p in b[b.C_]:
                    limf = getattr(p, f"value_low{self.constrain['sigma']}")
                    limc = getattr(p, f"value_high{self.constrain['sigma']}")

                    limc = min(limc, p.limits[1])
                    if (limf < p.limits[0] or
                        b.type_ == 'Jitter'):
                        limf = p.limits[0]

                    rang = limc - limf

                    if rang / abs(p.value) < self.constrain['tol']:
                        #self.add_condition([p.name, 'fixed', p.value])
                        print(f'Not further constraining {p.name}.')
                        pass
                    else:
                        self.add_condition([p.name, 'limits', [limf, limc]])


    def _load_next_model(self):
        # TODO prepare for astrometry
        # TODO add custom list option
        if self.switch_RV:
            self.add_keplerian()

        self.update_model()


    def apply_conditions(self):
        applied = []
        for b in self.blocks__:
            for p in b:
                for c in self.conds:
                    if p.name == c[0]:
                        setattr(p, c[1], c[2])
                        if c not in applied:
                            applied.append(c)
                            self.logger._apply_condition_message(c)

        self.logger.line()
        applied.sort()
        applied = [applied for applied,_ in itertools.groupby(applied)]

        for ap in applied[::-1]:
            self.conds.remove(ap)


    def add_condition(self, cond):
        self.conds.append(cond)


    def prerun_logger(self):
        if self.switch_first_run:
            self._prerun_logger_first()

        self.logger._add_subtitle('Pre-Run Info', postprocess=True)

        switch_title = True

        param_length = max(len(n) for n in self.model.get_attr_param('name', flat=True))
        prior_length = max(len(n) for n in self.model.get_attr_param('display_prior', flat=True))
        limits_length = max(len(str(lim)) for lim in self.model.get_attr_param('limits', flat=True))

        for b in self.model:
            to_tab0 = b.get_attr(['name', 'display_prior', 'limits'])

            to_tab0[2] = np.round(to_tab0[2], 3)
            to_tab = list(zip(*to_tab0))

            if switch_title:
                headers0 = [f"{'Parameter':<{param_length}}",
                            f"{'Prior':<{prior_length}}",
                            f"{'Limits':<{limits_length}}",
                            ]
                switch_title = False

            else:
                headers0 = [f"{'':<{param_length}}",
                            f"{'':<{prior_length}}",
                            f"{'':<{limits_length}}",
                            ]
            self.logger(tabulate(to_tab,
                                headers=headers0))

        # logger math models
        self.logger.dline()
        for b in self.model:
            self.logger(f'Math for {b.name_}: ', c='yellow', )
            self.logger(f'{b.math_display_}', center=True, c='yellow', save_extra_n=True)


    def _prerun_logger_first(self):
        label_width = 29
        dyn_crit = 'Hill Stability' if self.switch_dynamics else 'None'

        self.logger('~~ Setup Info ~~', center=True, c='blue', attrs=['reverse'])
        self.logger.line()

        self.logger.message_width('Current Engine is', f"{self.engine__.__name__} {self.engine__.__version__}", label_width)
        self.logger.message_width('Number of cores is', f"{self.cores__}", label_width)
        self.logger.message_width('Save location is', self.saveplace, label_width)

        self.logger.message_width('Dynamical Criteria is', dyn_crit, label_width)
        self.logger.message_width('Posterior fit method is', self.posterior_dict[self.posterior_fit_method], label_width)
        self.logger.message_width('Limits constrain method is', self.constrain['method'], label_width)
        self.logger.message_width('Model Selection method is', self._crit_current, label_width)

        self.logger.line()
        self.logger('~~ Automatically Saving ~~', center=True, c='blue', attrs=['reverse'])
        self.logger.line()


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

        self.switch_first_run = False


    @property
    def plot_all_list(self):
        return [
            self.plot_posteriors,
            self.plot_histograms,
            self.plot_keplerian_model,
            self.plot_periodogram,
            self.plot_betas,  # 4
            self.plot_beta_density,
            self.plot_rates,
            self.plot_gaussian_mixtures,
            self.plot_trace,
        ]


#
# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT
# **FIN  : Finish this

import itertools
import numpy as np

from . import param_repo as pr
from .emp_model import Parameter, Parameter_Block

subscript_nums = ['', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
supscript_nums = ['', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']


def KeplerianBlock(parameterisation, number, use_c):
    params_param = [['dPeriod', 'dAmplitude', 'dPhase', 'dEccentricity', 'dLongitude'],
                    ['dPeriod', 'dAmplitude', 'dPhase', 'dEcc_sin', 'dEcc_cos'],
                    ['dlPeriod', 'dAmp_sin', 'dAmp_cos', 'dEcc_sin', 'dEcc_cos'],
                    ['dPeriod', 'dAmplitude', 'dT_0', 'dEccentricity', 'dLongitude'],
                    ['dPeriod', 'dAmplitude', 'dT_0', 'dEcc_sin', 'dEcc_cos'],
                    ['dPeriod', 'dAmplitude', 'dM0', 'dEccentricity', 'dLongitude'],
                    ['dlPeriod', 'dAmplitude', 'dPhase', 'dEccentricity', 'dLongitude'],
                    ['dlPeriod', 'dAmplitude', 'dPhase', 'dEcc_sin', 'dEcc_cos'],
                    ]
    param_list = params_param[parameterisation]
    Kep_Block = {'model_script':f'kep0{parameterisation}.model',
                 'math_display_':f'K⋅(cos(ν(t,P,𝜙,e)+𝜔 )+e⋅cos(𝜔 ))|{subscript_nums[number]}',
                 }

    if parameterisation == 5:
        Kep_Block = {'model_script':'akep00.model',
                    'math_display_':f'K⋅(cos(ν(t,P,𝜙,e)+𝜔 )+e⋅cos(𝜔 ))|{subscript_nums[number]}',
                    }

    my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]
    if number:
        for p in my_params:
            p.name += f' {number}'
            p.mininame += f' {number}'

    Empty_Block = {'name_':f'KeplerianBlock {number}',
                   'type_':'Keplerian',
                   'class_':'Model',
                   'is_iterative':True,
                   'display_on_data_':True,
                   'dynamics_bool':False,
                   'astrometry_bool':False,
                   'parameterisation':parameterisation,
                   'number_':number,
                   'write_args':['slice'],
                   'dependencies':['import kepler']
                    }
    block_attributes = {**Empty_Block, **Kep_Block}

    if use_c:
        calc_rv_sup = {0:0,
                       1:0,
                       2:0,
                       3:1,
                       4:1}

        block_attributes['model_script'] = f'clib/ckep0{parameterisation}.model'
        block_attributes['dependencies'] = [f'from fast_kepler import calc_rv{calc_rv_sup[parameterisation]}']

    return Parameter_Block(my_params, block_attributes)


def SinusoidBlock(number):
    param_list = ['dPeriod', 'dAmplitude', 'dPhase']
    my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]

    if number:
        for p in my_params:
            p.name += f' {number}'
            p.mininame += f' {number}'

    block_attributes = {'name_':'SinusoidBlock',
                        'type_':'Sinusoid',
                        'class_':'Model',
                        'is_iterative':False,
                        'display_on_data_':True,
                        'parameterisation':None,
                        'number_':number,
                        
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'math_display_':'K⋅(cos(𝜔 t + 𝜙)',
                        'model_script':'sinusoid00.model',
                        'write_args':['slice'],
                        }
    return Parameter_Block(my_params, block_attributes)


def OffsetBlock(number, marginalize=False):
    my_params = []
    if marginalize:
        model_to_use = 'offset01.model'
    else:
        for i in range(number):
            offset_dict = pr.make_parameter(getattr(pr, 'dOffset'))
            offset_dict['name'] += f' {i+1}'
            offset_dict['mininame'] += f' {i+1}'
            my_params.append(Parameter(offset_dict))
        model_to_use = 'offset00.model'
    

    block_attributes = {'name_':'OffsetBlock',
                        'type_':'Offset',
                        'class_':'Data',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':number,
                        
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'math_display_':'γ₀|ᵢ',
                        'model_script':model_to_use,
                        'write_args':['number_', 'slice'],
                        }


    return Parameter_Block(my_params, block_attributes)


def AccelerationBlock(accel):
    my_params = []
    for i in range(accel):
        accel_dict = pr.make_parameter(getattr(pr, 'dAcceleration'))
        if i == 0:
            accel_dict['name'] = 'Acceleration'
            accel_dict['unit'] = r'($\frac{m}{s day}$)'
        else:
            accel_dict['name'] = f'Acceleration Order {str(i + 1)}'
            accel_dict['unit'] = r'($\frac{m}{s day^%s}$)' % str(i + 1)
            accel_dict['mininame'] += f' {i+1}'

        my_params.append(Parameter(accel_dict))
    
    math_display = f'γ{subscript_nums[1]}'
    for j in range(accel-1):
        math_display += f' + γ{subscript_nums[2 + j]}'

    block_attributes = {'name_':f'AccelerationBlock o{accel}',
                        'type_':'Acceleration',
                        'class_':'Data',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':accel,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'acc.model',
                        'write_args':['slice'],
                        'math_display_':math_display,
                        }

    return Parameter_Block(my_params, block_attributes)


def JitterBlock(number):
    my_params = []
    for i in range(number):
        jitter_dict = pr.make_parameter(getattr(pr, 'dJitter'))
        jitter_dict['name'] += f' {i+1}'
        jitter_dict['mininame'] += rf'_{i+1}'

        my_params.append(Parameter(jitter_dict))
    
    block_attributes = {'name_':'JitterBlock',
                        'type_':'Jitter',
                        'class_':'Data',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':number,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'jitter00.model',
                        'write_args':['number_', 'slice'],
                        'math_display_':'𝝈ᵢ',
                        }

    return Parameter_Block(my_params, block_attributes)


def MOAVBlock(number, moav_args=None):
    moav_args = moav_args or {}
    # number +1?
    my_params = []
    order = moav_args['order']
    is_global = moav_args['global']
    param_range = 1 if is_global else number

    for i, j in itertools.product(range(param_range), range(order)):
        MA_coef_dict = pr.make_parameter(getattr(pr, 'dMACoefficient'))
        MA_coef_dict['name'] += f' {i+1} Order {j+1}'

        MA_time_dict = pr.make_parameter(getattr(pr, 'dMATimescale'))
        MA_time_dict['name'] += f' {i+1} Order {j+1}'

        my_params.extend((Parameter(MA_coef_dict), Parameter(MA_time_dict)))
    block_attributes = {
        'name_': 'MOAVBlock',
        'type_': 'MOAV',
        'class_': 'Data',
        'is_iterative': False,
        'display_on_data_': False,
        'parameterisation': None,
        'number_': number,
        'bnumber_': 0,
        'moav': order,
        'is_global': is_global,
        'slice': None,
        'additional_priors_bool': None,
        'dynamics_bool': None,
        'write_args': ['number_', 'slice', 'moav'],
        'math_display_': '𝛴ᵢ𝛴ₘ 𝛷ₘ⋅exp((t₍ᵢ₋ₘ₎-tᵢ)/𝜏ₘ)⋅𝜀(t₍ᵢ₋ₘ₎)',
        'model_script': 'moav01.model' if is_global else 'moav00.model',
    }
    return Parameter_Block(my_params, block_attributes)  


def SAIBlock(number, sa=False):
    my_params = []
    for i in range(number):
        for si in range(sa[i]):
            d0 = pr.make_parameter(getattr(pr, 'dStaract'))
            d0['name'] += f' {i+1} {si+1}'
            d0['mininame'] += f'_{i+1} {si+1}'
            my_params.append(Parameter(d0))


    block_attributes = {'name_':'StellarActivityBlock',
                        'type_':'StellarActivity',
                        'class_':'Data',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':number,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'sai00.model',
                        'write_args':['number_', 'slice'],
                        'math_display_':'𝜁ₘ⋅𝓐ᵢ',
                        }


    return Parameter_Block(my_params, block_attributes)


def MagneticCycleBlock(number):
    param_list = ['dPeriod', 'dAmplitude', 'dAmplitude', 'dPhase', 'dPhase']
    my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]

    if True:
        my_params[0].name += f' S{1}'
        my_params[0].mininame += f' S{1}'

        my_params[1].name += f' S{1}'
        my_params[1].mininame += f' S{1}'
        my_params[2].name += f' S{2}'
        my_params[2].mininame += f' S{2}'

        my_params[3].name += f' S{1}'
        my_params[3].mininame += f' S{1}'
        my_params[4].name += f' S{2}'
        my_params[4].mininame += f' S{2}'

        if False:
            for p in my_params:
                p.name += f' S{number}'
                p.mininame += f' S{number}'
            pass

    block_attributes = {'name_':'MagneticCycleBlock',
                        'type_':'MagneticCycle',
                        'class_':'Model',
                        'is_iterative':False,
                        'display_on_data_':True,
                        'parameterisation':None,
                        'number_':number,
                        
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'math_display_':'K₁⋅(cos(𝜔 ₁t + 𝜙₁) + K₂⋅(cos(2⋅𝜔 ₁t + 𝜙₂)',
                        'model_script':'magneticcycle00.model',
                        'write_args':['slice'],
                        }
    return Parameter_Block(my_params, block_attributes)


def SAIPROBlock(number, cornum_pro):
    my_params = []
    param_list = ['dPeriod', 'dAmplitude', 'dPhase', 'dOffset']
    my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]
    for p in my_params:
        p.name = f'SAI {p.name}'
        p.mininame = f'SAI {p.mininame}'        
        p.name += f' {number}'
        p.mininame += f' {number}'
        
    block_attributes = {'name_':'StellarActivityPROBlock',
                        'type_':'StellarActivityPRO',
                        'class_':'Data',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':number,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'saipro00.model',
                        'write_args':['slice'],
                        'math_display_':'',
                        }


    return Parameter_Block(my_params, block_attributes)


def CeleriteBlock(nins, my_kernel):
    my_params = []
    nterms = len(my_kernel['terms'])
    

    for current_term, current_params in zip(my_kernel['terms'],
                                            my_kernel['params']):
        
        if current_term == 'RealTerm':
            term_params = ['dRealTerm_a',
                           'dRealTerm_c']
        elif current_term == 'Matern32Term':
            term_params = ['dMatern32Term_sigma',
                           'dMatern32Term_rho']
        elif current_term == 'RotationTerm':
            term_params = ['dRotationTerm_period',
                           'dRotationTerm_sigma',
                           'dRotationTerm_Q0',
                           'dRotationTerm_dQ',
                           'dRotationTerm_f']
            
        # TODO: you can simplify all this as it is done in SHO Term
        elif current_term == 'SHOTerm':
            term_params = []
            for param in current_params:
                term_params.append(f'd{current_term}_{param}')

        elif current_term == 'GonzRotationTerm':
            term_params = ['dGonzRotationTerm_rho',
                           'dGonzRotationTerm_tau',
                           'dGonzRotationTerm_A1',
                           'dGonzRotationTerm_A2']


        else:
            print(f'ERROR: Current kernel {current_term} not identified. -RT')
            term_params = []

        my_params.extend(
            Parameter(pr.make_parameter(getattr(pr, param)))
            for param in term_params
        )

    if nterms > 1:
        nterms_counter = 1
        params_counter = 0
        for t in range(nterms):
            for _ in my_kernel['params'][t]:
                my_params[params_counter].name += str(nterms_counter)
                params_counter += 1
            nterms_counter += 1

    block_attributes = {'name_':'CeleriteBlock',
                        'type_':'Celerite2',
                        'class_':'Model',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':nins,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'real_term00.model',
                        'math_display_':'',
                        'dependencies':['import celerite2',
                                        'import celerite2.terms as cterms']
                        }

    return Parameter_Block(my_params, block_attributes)




def AstrometryKeplerianBlock(parameterisation, number, use_c):
    init_Block = KeplerianBlock(parameterisation,
                                number,
                                use_c)
                                   
    Empty_Block = {'name_':f'AstrometryKeplerianBlock {number}',
                    'model_script':'akep00.model',
                    'math_display_':f'K⋅sin(I)⋅(cos(ν(t,P,𝜙,e)+𝜔 )+e⋅cos(𝜔 ))|{subscript_nums[number]}',
                    'astrometry_bool':True,
                    }
    for change in Empty_Block:
        setattr(init_Block, change, Empty_Block[change])

    param_list = ['dInclination', 'dOmega']
    my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]

    if number:
        for p in my_params:
            p.name += f' {number}'
    
    init_Block.list_ = np.append(init_Block.list_, my_params)

    return init_Block


def AstrometryOffsetBlock(number):
    param_list = ['dOffset_ra', 'dOffset_de', 'dOffset_plx',
                  'dOffset_pm_ra', 'dOffset_pm_de']

    my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]
    
    block_attributes = {'name_':'AstrometryOffsetBlock',
                        'type_':'AstrometryOffset',
                        'class_':'Data',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':number,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'am_offset00.model',
                        'write_args':['slice'],
                        'math_display_':'γ₀|ᵢ',  # put proper utf8
                        }

    return Parameter_Block(my_params, block_attributes)


def AstrometryJitterBlock(number):
    param_list = ['dJitterH', 'dJitterG']

    my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]
    
    block_attributes = {'name_':'AstrometryJitterBlock',
                        'type_':'AstrometryJitter',
                        'class_':'Data',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':number,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'am_jitter00.model',
                        'write_args':['slice'],
                        'dependencies':['from numpy.linalg import inv, slogdet'],
                        'math_display_':'',
                        }

    return Parameter_Block(my_params, block_attributes)


class SmartSetter(object):
    def __init__(self, data):
        self.data = data
        self.uni = 'Uniform'
        self.norm = 'Normal'
        self.unit_limiter = [0, 1]
        self.angular_limiter = [0, 2*np.pi]

        # TODO set acceleration
        # TODO set astrometry
        # TODO set GP

    def set_Keplerian(self, b, *args, **kwargs):
        sig_limiter = self.data['RV'].std(ddof=0)
        per_limiter = self.data['BJD'].max() - self.data['BJD'].min()

        amp_limiter = sig_limiter * np.sqrt(4)

        ecc_limits, ecc_prargs = kwargs['prargs']

        if b.parameterisation == 0:
            lims = [[1.5, per_limiter],
                    [1e-6, amp_limiter],
                    self.angular_limiter,
                    ecc_limits,
                    self.angular_limiter]
            
            priors = [self.uni, self.uni, self.uni, self.norm, self.uni]
            prargs = [None, None, None, ecc_prargs, None]

            if b.astrometry_bool:
                lims.extend([[0, np.pi], self.angular_limiter])  # Ome
                priors.extend([self.uni, self.uni])
                prargs.extend([None, None])

        if b.parameterisation == 1:
            lims = [[1.5, per_limiter],
                    [1e-6, amp_limiter],
                    self.angular_limiter,
                    [-1, 1],
                    [-1, 1]]
            
            priors = [self.uni, self.uni, self.uni, self.uni, self.uni]
            prargs = [None, None, None, None, None]
            self.ecc_w_add_additional(b)

        if b.parameterisation == 2:
            kamp = np.sqrt(amp_limiter/2)
            lims = [[np.log(1.5), np.log(per_limiter)],
                    [-kamp, kamp],
                    [-kamp, kamp],
                    [-1, 1],
                    [-1, 1]]
            
            priors = [self.uni, self.uni, self.uni, self.uni, self.uni]
            prargs = [None, None, None, None, None]
            self.ecc_w_add_additional(b)

        if b.parameterisation == 3:
            lims = [[1.5, per_limiter],
                    [1e-6, amp_limiter],
                    [-1000, 1000],
                    ecc_limits,
                    self.angular_limiter]
            
            priors = [self.uni, self.uni, self.uni, self.norm, self.uni]
            prargs = [None, None, None, ecc_prargs, None]

        if b.parameterisation == 4:
            lims = [[1.5, per_limiter],
                    [1e-6, amp_limiter],
                    [-1000, 1000],
                    [-1, 1],
                    [-1, 1]]
            
            priors = [self.uni, self.uni, self.uni, self.uni, self.uni]
            prargs = [None, None, None, None, None]
            self.ecc_w_add_additional(b)

        if b.parameterisation == 6:
            lims = [[np.log(1.5), np.log(per_limiter)],
                    [1e-6, amp_limiter],
                    self.angular_limiter,
                    ecc_limits,
                    self.angular_limiter]
            
            priors = ['Jeffreys', self.uni, self.uni, self.norm, self.uni]
            prargs = [None, None, None, ecc_prargs, None]
            self.per_add_additional(b, per_limiter=per_limiter)

        if b.parameterisation == 7:
            lims = [[np.log(1.5), np.log(per_limiter)],
                    [1e-6, amp_limiter],
                    self.angular_limiter,
                    [-1, 1],
                    [-1, 1]]
            
            priors = ['Jeffreys', self.uni, self.uni, self.uni, self.uni]
            prargs = [None, None, None, None, None]

            self.per_add_additional(b, per_limiter=per_limiter)
            self.ecc_w_add_additional(b, prargs=ecc_prargs)


        if kwargs['starmass']:
            # TODO
            self.sma_minmass_add_additional(b)
        
            if kwargs['dynamics']:
                if not kwargs['dynamics_already_included'] and d['kplan'] > 1:
                    d0 = {'name': 'Hill',
                        'display_name': 'Dynamical Criteria',
                        'prior': 'Hill',
                        'limits':[None, None],
                        'prargs':[d['kplan'], d['starmass']],
                        'has_prior':True,
                        'has_posterior':False,
                        'unit':None,
                        'fixed':None,
                        }
                    kwargs['dynamics_already_included'] = True
                    b.add_additional_parameters(d0)

        return lims, priors, prargs


    def sma_minmass_add_additional(self, b):
        param_list = ['dSMA', 'dMinM']
        my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]

        
        for additional_parameter in my_params:
            additional_parameter.has_prior = False
            additional_parameter.has_posterior = True
            additional_parameter.limits = [1e-5, 1000]
            if b.number_:
                additional_parameter.name += f' {b.number_}'
                additional_parameter.mininame += f' {b.number_}'
            additional_parameter.display_name = additional_parameter.name
            b.add_additional_parameters(additional_parameter)

    def ecc_w_add_additional(self, b, prargs=None):
        param_list = ['dEccentricity', 'dLongitude']
        my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]
        
        my_params[0].has_prior = True
        my_params[0].limits = [0, 1]
        if prargs is not None:
            my_params[0].prior = self.norm
            my_params[0].prargs = prargs


        my_params[1].has_prior = False
        my_params[1].limits = self.angular_limiter 


        for additional_parameter in my_params:
            additional_parameter.has_posterior = True
            if b.number_:
                additional_parameter.name += f' {b.number_}'
                additional_parameter.mininame += f' {b.number_}'
            additional_parameter.display_name = additional_parameter.name
            b.add_additional_parameters(additional_parameter)

    def per_add_additional(self, b, per_limiter=None):
        my_params = Parameter(pr.make_parameter(getattr(pr, 'dPeriod')))
        my_params.has_prior = False
        my_params.limits = [0, per_limiter]
        my_params.has_posterior = True
        if b.number_:
            my_params.name += f' {b.number_}'
            my_params.mininame += f' {b.number_}'
            my_params.display_name = my_params.name
            b.add_additional_parameters(my_params)



    def set_Offset(self, b, *args, **kwargs):
        lims, priors, prargs = [], [], []
        for nin in range(b.number_):
            mask = self.data['Flag'] == (nin + 1)
            jit_limiter = self.data[mask]['RV'].abs().max()
            lims.append([-jit_limiter, jit_limiter])
            priors.append(self.uni)
            prargs.append(None)

        return lims, priors, prargs


    def set_Jitter(self, b, *args, **kwargs):
        lims, priors, prargs = [], [], []
        
        jit_limits, jit_prargs = args
        for nin in range(b.number_):
            mask = self.data['Flag'] == (nin + 1)
            jit_limiter = self.data[mask]['RV'].abs().max()
            
            lims.append([1e-5, jit_limiter])
            priors.append(self.norm)
            prargs.append(jit_prargs)

        return lims, priors, prargs


    def set_MOAV(self, b, *args, **kwargs):
        lims, priors, prargs = [], [], []
        
        for _ in range(b.number_):
            for _ in range(b.moav):
                lims.append([0.0, 0.3])
                lims.append([5, 25])
               
                priors.append(self.uni)
                priors.append(self.uni)

                prargs.append(None)
                prargs.append(None)

        return lims, priors, prargs


    def set_StellarActivity(self, b, *args, **kwargs):
        lims, priors, prargs = [], [], []
        for cornum in b.cornums:
            for _ in range(cornum):
                lims.append([-1., 1.])
                priors.append(self.uni)
                prargs.append(None)
        return lims, priors, prargs


    def set_StellarActivityPRO(self, b, *args, **kwargs):
        lims, priors, prargs = [], [], []
        for _ in range(b.cornum_pro):
            lims.append([-1., 1.])
            priors.append(self.uni)
            prargs.append(None)
        return lims, priors, prargs    


    def set_Celerite2(self, b, *args, **kwargs):
        lims, priors, prargs = [], [], []
        d = kwargs
        pc = 0
        for i in range(len(d['terms'])):
            current_term = d['terms'][i]
            current_params = d['params'][i]
            for param in current_params:
                if current_params[param] == None:
                    if current_term in ['RealTerm',
                                        'SHOTerm',
                                        'Matern32Term',
                                        'RotationTerm',
                                        'GonzRotationTerm',]:
                        if param == 'f':
                            lims.append([1e-5, 1])
                        else:
                            lims.append([1e-5, 5])
                        priors.append('Uniform')
                    else:
                        print(f'Unidentified kernel term {current_term} !')

                else:
                    lims.append([np.nan, np.nan])
                    priors.append('Fixed')
                    b[pc].fixed = current_params[param]

                prargs.append(None)
                pc += 1
        return lims, priors, prargs


    def set_Acceleration(self, b, *args, **kwargs):
        lims, priors, prargs = [], [], []
        acc_lims = [-1, 1]
        for _ in range(b.number_):
            lims.append(acc_lims)
            priors.append(self.uni)
            prargs.append(None)


        return lims, priors, prargs


    def set_Sinusoid(self, b, *args, **kwargs):
        sig_limiter = self.data['RV'].std(ddof=0)
        amp_limiter = sig_limiter * np.sqrt(4)
        per_limiter = self.data['BJD'].max() - self.data['BJD'].min()
        lims = [[1.5, per_limiter],
                [1e-6, amp_limiter],
                self.angular_limiter,
                ]
            
        priors = [self.uni, self.uni, self.uni]
        prargs = [None, None, None]
        return lims, priors, prargs


    def set_MagneticCycle(self, b, *args, **kwargs):
        sig_limiter = self.data['RV'].std(ddof=0)
        amp_limiter = sig_limiter * np.sqrt(4)
        per_limiter = self.data['BJD'].max() - self.data['BJD'].min()
        lims = [[1.5, per_limiter],
                [1e-6, amp_limiter],
                [1e-6, amp_limiter],
                self.angular_limiter,
                self.angular_limiter,
                ]
            
        priors = [self.uni, self.uni, self.uni, self.uni, self.uni]
        prargs = [None, None, None, None, None]
        return lims, priors, prargs


    def set_AstrometryJitter(self, b, *args, **kwargs):
        lims = [[0, 20], [0, 10]]  # hip, gaia
        priors = ['Uniform', 'Uniform']
        prargs = [[], []]

        return lims, priors, prargs


    def set_AstrometryOffset(self, b, *args, **kwargs):
        #lims, priors, prargs = [], [], []
        off_lims = [-1e1, 1e1]
        lims = [off_lims, off_lims, off_lims, off_lims, off_lims]  # hip, gaia
        priors = ['Uniform', 'Uniform', 'Uniform', 'Uniform', 'Uniform']
        prargs = [[], [], [], [], []]

        return lims, priors, prargs


    def add_constant(self, lims, priors, prargs):
        for i, pr in enumerate(priors):
            if pr == 'Uniform':
                low, high = lims[i]


    def __call__(self, block, *args, **kwargs):
        #try:
        lims, priors, prargs = getattr(self, f'set_{block.type_}')(block, *args, **kwargs)
        #except Exception:
        #    print(f'Unidentified block type {block.type_}')

        block.set_attr('limits', lims, silent=True)
        block.set_attr('prior', priors, silent=True)
        block.set_attr('prargs', prargs, silent=True)

#
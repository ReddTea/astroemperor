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
from .block import Parameter, Parameter_Block
from .utils import ModelWrapper
from .model_repo import *

from .globals import _OS_ROOT

## RED please re-do this import
import sys
sys.path.insert(1, _OS_ROOT)
import param_repo as pr


subscript_nums = ['', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
supscript_nums = ['', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']

def mk_KeplerianBlock(my_data, parameterisation=0, number=1, use_c=False):
    Empty_Block = {'name_':f'KeplerianBlock {number}',
                    'type_':'Keplerian',
                    'is_iterative':True,
                    'display_on_data_':True,
                    'parameterisation':parameterisation,
                    'number_':number,
                    'use_c':use_c,
                    
                    'bnumber_':0,
                    'moav':0,
                    'slice':None,
                    'additional_priors_bool':None,
                    'dynamics_bool':None,
                    'astrometry_bool':False,
                    }
    
    if parameterisation == 0:
        param_list = ['dPeriod', 'dAmplitude', 'dPhase', 'dEccentricity', 'dLongitude']
        Kep_Block = {'model_script':'kep00.model',
                    'math_display_':f'K⋅(cos(ν(t,P,𝜙,e)+𝜔)+e⋅cos(𝜔))|{subscript_nums[number]}',
                    }
        
    if parameterisation == 1:
        param_list = ['dlPeriod', 'dAmp_sin', 'dAmp_cos', 'dEcc_sin', 'dEcc_cos']
        Kep_Block = {'model_script':'kep01.model',
                    'math_display_':f'K⋅(cos(ν(t,P,𝜙,e)+𝜔)+e⋅cos(𝜔))|{subscript_nums[number]}',
                    }
        
    if parameterisation == 2:
        param_list = ['dPeriod', 'dAmplitude', 'dT_0', 'dEccentricity', 'dLongitude']
        Kep_Block = {'model_script':'kep02.model',
                    'math_display_':f'K⋅(cos(ν(t,P,T₀,e)+𝜔)+e⋅cos(𝜔))|{subscript_nums[number]}',
                    }

    if parameterisation == 3:
        param_list = ['dPeriod', 'dAmplitude', 'dT_0', 'dEcc_sin', 'dEcc_cos']
        Kep_Block = {'model_script':'kep03.model',
                    'math_display_':f'K⋅(cos(ν(t,P,T₀,e)+𝜔)+e⋅cos(𝜔))|{subscript_nums[number]}',
                    }

    if parameterisation == 999:
        param_list = ['dPeriod', 'dAmplitude', 'dM0', 'dEccentricity', 'dLongitude']
        Kep_Block = {'model_script':'akep00.model',
                    'math_display_':f'K⋅(cos(ν(t,P,𝜙,e)+𝜔)+e⋅cos(𝜔))|{subscript_nums[number]}',
                    }


    my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]
    if number:
        for p in my_params:
            p.name += f' {number}'
            p.mininame += f' {number}'
    block_attributes = {**Empty_Block, **Kep_Block}

    return Parameter_Block(my_params, block_attributes)


def mk_OffsetBlock(my_data, nins=1):
    my_params = []
    for i in range(nins):
        offset_dict = pr.make_parameter(getattr(pr, 'dOffset'))
        offset_dict['name'] += f' {i+1}'
        offset_dict['mininame'] += f' {i+1}'
        my_params.append(Parameter(offset_dict))
    
    block_attributes = {'name_':f'OffsetBlock',
                        'type_':'Offset',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':nins,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'offset00.model',
                        'math_display_':'γ₀|ᵢ',
                        }


    return Parameter_Block(my_params, block_attributes)


def mk_SAIBlock(my_data, nins=1, sa=False):
    my_params = []
    for i in range(nins):
        for si in range(sa[i]):
            d0 = pr.make_parameter(getattr(pr, 'dStaract'))
            d0['name'] += f' {i+1} {si+1}'
            d0['mininame'] += f'_{i+1} {si+1}'
            my_params.append(Parameter(d0))


    block_attributes = {'name_':'StellarActivityBlock',
                        'type_':'StellarActivity',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':nins,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'sai00.model',
                        'math_display_':'',
                        }


    return Parameter_Block(my_params, block_attributes)    


def mk_AccelerationBlock(my_data, accel=1):
    acmod = Acceleration_Model
    
    my_params = []
    #punits = []
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
    '''
    pvalues = [-np.inf for _ in range(bdim)]
    ppriors = ['Uniform' for _ in range(bdim)]
    daily = 1 # 1/365.25
    plimits = [[-daily, daily] for _ in range(bdim)]

    ptypes = [None for _ in range(bdim)]
    prargs = [None for _ in range(bdim)]
    ptformargs = [None for _ in range(bdim)]
    pfixed = [None for _ in range(bdim)]
    psigma = [None for _ in range(bdim)]

    pGM_parameter = [None for _ in range(bdim)]
    pposterior = [None for _ in range(bdim)]

    pstds = [None for _ in range(bdim)]
    pis_circular = [False for _ in range(bdim)]
    pis_hou = [False for _ in range(bdim)]

    
    for i in range(bdim):
        d0 = {'name':pnames[i], 'prior':ppriors[i], 'value':pvalues[i],
                  'limits':plimits[i], 'unit':punits[i], 'prargs':prargs[i],
                  'type':ptypes[i], 'ptformargs':ptformargs[i], 'fixed':pfixed[i],
                  'sigma':psigma[i], 'GM_parameter':pGM_parameter[i],
                  'posterior':pposterior[i], 'std':pstds[i],
                  'is_circular':pis_circular[i],
                  'is_hou':pis_hou[i]}


        my_params.append(Parameter(d0))


    b_mod = ModelWrapper(acmod, [my_data.values[:, 0]])
    b_name = f'AccelerationBlock o{accel}'
    b_script = 'acc.model'
    '''

    block_attributes = {'name_':f'AccelerationBlock o{accel}',
                        'type_':'Acceleration',
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
                        'math_display_':math_display,
                        }

    return Parameter_Block(my_params, block_attributes)


def mk_MOAVBlock(my_data, nins=1, moav_args={}):
    my_params = []
    order = moav_args['order']
    is_global = moav_args['global']
    param_range = 1 if is_global else nins
    for i in range(param_range):
        for j in range(order):
            MA_coef_dict = pr.make_parameter(getattr(pr, 'dMACoefficient'))
            MA_coef_dict['name'] += f' {i+1} Order {j+1}'

            MA_time_dict = pr.make_parameter(getattr(pr, 'dMATimescale'))
            MA_time_dict['name'] += f' {i+1} Order {j+1}'


            my_params.append(Parameter(MA_coef_dict))
            my_params.append(Parameter(MA_time_dict))
    
    block_attributes = {'name_':f'MOAVBlock',
                        'type_':'MOAV',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':nins,
                        
                        'bnumber_':0,
                        'moav':order,
                        'is_global':is_global,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'math_display_':'𝛴ᵢ𝛴ₘ 𝛷ₘ⋅exp((t₍ᵢ₋ₘ₎-tᵢ)/𝜏ₘ)⋅𝜀(t₍ᵢ₋ₘ₎)',
                        }
    if is_global:
        block_attributes['model_script'] = 'moav01.model'
    else:
        block_attributes['model_script'] = 'moav00.model'


    return Parameter_Block(my_params, block_attributes)    


def mk_JitterBlock(my_data, nins=1):
    my_params = []
    for i in range(nins):
        jitter_dict = pr.make_parameter(getattr(pr, 'dJitter'))
        jitter_dict['name'] += f' {i+1}'
        jitter_dict['mininame'] += rf'_{i+1}'

        my_params.append(Parameter(jitter_dict))
    
    block_attributes = {'name_':f'JitterBlock',
                        'type_':'Jitter',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':nins,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'jitter00.model',
                        'math_display_':'',
                        }

    return Parameter_Block(my_params, block_attributes)


def mk_CeleriteBlock(my_data, nins=1, my_kernel={}):
    my_params = []

    for current_term in my_kernel['terms']:
        if current_term == 'RealTerm':
        # real term
            my_params.append(Parameter(pr.make_parameter(getattr(pr, 'dRealTerm_a'))))
            my_params.append(Parameter(pr.make_parameter(getattr(pr, 'dRealTerm_c'))))
        elif current_term == 'RotationTerm':
            my_params.append(Parameter(pr.make_parameter(getattr(pr, 'dRotationTerm_sigma'))))
            my_params.append(Parameter(pr.make_parameter(getattr(pr, 'dRotationTerm_period'))))
            my_params.append(Parameter(pr.make_parameter(getattr(pr, 'dRotationTerm_Q0'))))
            my_params.append(Parameter(pr.make_parameter(getattr(pr, 'dRotationTerm_dQ'))))
            my_params.append(Parameter(pr.make_parameter(getattr(pr, 'dRotationTerm_f'))))
        elif current_term == 'Matern32Term':
            my_params.append(Parameter(pr.make_parameter(getattr(pr, 'dMatern32Term_sigma'))))
            my_params.append(Parameter(pr.make_parameter(getattr(pr, 'dMatern32Term_rho'))))
        else:
            print('ERROR: Current kernel not identified. -RT')
        pass

    block_attributes = {'name_':f'CeleriteBlock',
                        'type_':'Celerite2',
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
                        }

    return Parameter_Block(my_params, block_attributes)


def mk_AdditionalPriorsBlock(my_data):
    my_params = []
    empmod = Empty_Model
    b_mod = ModelWrapper(empmod, [my_data.values[:, 0]])
    b_name = 'AdditionalPriorsBlock'

    math_display = ''

    block_attributes = {'name_':f'AdditionalPrior',
                        'type_':'AdditionalPriors',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':nins,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'',
                        'math_display_':'',
                        }

    return Parameter_Block(my_params, block_attributes)


def mk_AstrometryKeplerianBlock(my_data, parameterisation=0, number=1):
    init_Block = mk_KeplerianBlock(my_data,
                                   parameterisation=0,
                                   number=number)
                                   
    Empty_Block = {'name_':f'AstrometryKeplerianBlock {number}',
                    #'type_':'AstrometryKeplerian',
                    'model_script':'akep00.model',
                    'math_display_':f'K⋅sin(I)⋅(cos(ν(t,P,𝜙,e)+𝜔)+e⋅cos(𝜔))|{subscript_nums[number]}',
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


def mk_AstrometryOffsetBlock(my_data, nins=1):
    param_list = ['dOffset_ra', 'dOffset_de',
                  'dOffset_pm_ra', 'dOffset_pm_de']

    my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]
    
    block_attributes = {'name_':f'AstrometryOffsetBlock',
                        'type_':'AstrometryOffset',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':nins,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'aoffset00.model',
                        'math_display_':'γ₀|ᵢ',  # put proper utf8
                        }

    return Parameter_Block(my_params, block_attributes)


def mk_AstrometryJitterBlock(my_data, nins=1):
    param_list = ['dJitterH', 'dJitterG']

    my_params = [Parameter(pr.make_parameter(getattr(pr, par))) for par in param_list]
    
    block_attributes = {'name_':f'AstrometryJitterBlock',
                        'type_':'AstrometryJitter',
                        'is_iterative':False,
                        'display_on_data_':False,
                        'parameterisation':None,
                        'number_':nins,
                        
                        'bnumber_':0,
                        'moav':0,
                        'slice':None,
                        'additional_priors_bool':None,
                        'dynamics_bool':None,
                        'model_script':'ajitter00.model',
                        'math_display_':'',
                        }

    return Parameter_Block(my_params, block_attributes)


def SmartLimits(my_data, b, *args, **kwargs):
    uni = 'Uniform'
    norm = 'Normal'
    # ecc limits and prargs
    additional_priors = []
    lims = []
    priors = []
    prargs = []
    d = kwargs

    if b.type_ == 'Keplerian':
        sig_limiter = my_data['RV'].std(ddof=0)
        per_limiter = my_data['BJD'].max() - my_data['BJD'].min()
        amp_limiter = sig_limiter * np.sqrt(4)
        angle_limits = [0, 2*np.pi]

        ecc_limits, ecc_prargs = d['prargs']

        if b.parameterisation == 0:
            lims = [[1.5, per_limiter], [1e-6, amp_limiter], angle_limits,
                       ecc_limits, angle_limits]
            priors = [uni, uni, uni, norm, uni]
            prargs = [None, None, None, ecc_prargs, None]

            if b.astrometry_bool:
                lims.extend([angle_limits, angle_limits])  # Ome
                priors.extend([uni, uni])
                prargs.extend([None, None])



        if b.parameterisation == 1:
            sqrta, sqrte = amp_limiter, 0.707  #(sqrt 0.5 ~ 0.707)
            sqrta, sqrte = sqrta ** 0.5, sqrte ** 2
            a_lims, e_lims = [-sqrta, sqrta], [-sqrte, sqrte]

            lims = [np.log([0.1, per_limiter*3]), a_lims, a_lims, e_lims, e_lims]
            priors = [uni, uni, uni, uni, uni]
            prargs = [None, None, None, None, None]

            b.additional_parameters = []
            pnames = ['Period', 'Amplitude', 'Phase', 'Eccentricity', 'Longitude']
            ndim = len(pnames)

            pdisplay_names = [f'{x} {b.number_}' for x in pnames]
            ppriors = [uni, uni, uni, uni, uni]
            plims = [[1e-6, per_limiter*3], [0.1, amp_limiter], [0, 2*np.pi], ecc_limits, [0, 2*np.pi]]
            pprargs = [[], [], [], ecc_prargs, []]
            phas_prior = [False, True, False, True, False]
            phas_posterior = [True, True, True, True, True]
            punits = ['(Days)', r'($\frac{m}{s}$)', '(rad)', '', '(rad)']


            for i in range(ndim):
                d0 = {'name': pnames[i],
                      'display_name': pdisplay_names[i],
                      'prior': ppriors[i],
                      'limits':plims[i],
                      'prargs':pprargs[i],
                      'has_prior':phas_prior[i],
                      'has_posterior':phas_posterior[i],
                      'unit':punits[i],
                      'fixed':None,
                      }

                b.add_additional_parameters(d0)

        if b.parameterisation == 2:
            t0_limiter = [my_data['BJD'].min(), my_data['BJD'].min() + per_limiter]

            lims = [[0.1, per_limiter], [0, amp_limiter], t0_limiter, ecc_limits, angle_limits]
            priors = [uni, uni, uni, norm, uni]
            prargs = [None, None, None, ecc_prargs, None]

        if b.parameterisation == 3:

            t0_limiter = [my_data['BJD'].min() - per_limiter, my_data['BJD'].max() + per_limiter]
            sqrte = 1  #(sqrt 0.5 ~ 0.707)
            sqrte **= 2
            e_lims = [-sqrte, sqrte]

            lims = [[0.1, per_limiter], [0, amp_limiter], t0_limiter, e_lims, e_lims]
            priors = [uni, uni, uni, uni, uni]
            prargs = [None, None, None, None, None]

            b.additional_parameters = []
            pnames = ['Eccentricity', 'Longitude']
            pdisplay_names = [f'{x} {b.number_}' for x in pnames]
            plims = [ecc_limits, angle_limits]
            pprargs = [ecc_prargs, []]
            phas_prior = [True, False]
            phas_posterior = [True, True]
            punits = ['', '(rad)']

            ndim = len(pnames)
            ppriors = [uni, uni]
            for i in range(ndim):
                d0 = {'name': pnames[i],
                      'display_name': pdisplay_names[i],
                      'prior': ppriors[i],
                      'limits':plims[i],
                      'prargs':pprargs[i],
                      'has_prior':phas_prior[i],
                      'has_posterior':phas_posterior[i],
                      'unit':punits[i],
                      'fixed':None,
                      }

                b.add_additional_parameters(d0)

            '''
            b.add_additional_priors([['Eccentricity', norm, ecc_limits, ecc_prargs],
                                   ['Longitude', uni, [0, 2*np.pi], []],
                                   ])
            '''
            
        if d['starmass']:
            sma_minmass_add_additional(b, uni)
        
            if d['dynamics']:
                if not d['dynamics_already_included'] and d['kplan'] > 1:
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
                    d['dynamics_already_included'] = True
                    b.add_additional_parameters(d0)


    elif b.type_ == 'Instrumental':
        jit_limits, jit_prargs = args

        mask = my_data['Flag']==b.number_
        jit_limiter = my_data[mask]['RV'].abs().max()

        lims = [[-jit_limiter, jit_limiter], [1e-5, jit_limiter]]
        priors = [uni, norm]
        prargs = [None, jit_prargs]

        if b.moav > 0:
            for _ in range(b.moav):
                lims.append([0.0, 0.3])
                lims.append([5, 25])

                priors.append(uni)
                priors.append(uni)

                prargs.append(None)
                prargs.append(None)

        if b.cornum > 0:
            for _ in range(b.cornum):
                lims.append([-1., 1.])
                priors.append(uni)
                prargs.append(None)


    elif b.type_ == 'Offset':
        for nin in range(b.number_):
            mask = my_data['Flag'] == (nin + 1)
            jit_limiter = my_data[mask]['RV'].abs().max()
            lims.append([-jit_limiter, jit_limiter])
            priors.append(uni)
            prargs.append(None)


    elif b.type_ == 'StellarActivity':
        for cornum in b.cornums:
            for _ in range(cornum):
                lims.append([-1., 1.])
                priors.append(uni)
                prargs.append(None)
          

    elif b.type_ == 'MOAV':
        for nin in range(b.number_):
            for _ in range(b.moav):
                lims.append([0.0, 0.3])
                lims.append([5, 25])
               
                priors.append(uni)
                priors.append(uni)

                prargs.append(None)
                prargs.append(None)


    elif b.type_ == 'Jitter':
        jit_limits, jit_prargs = args
        for nin in range(b.number_):
            mask = my_data['Flag'] == (nin + 1)
            jit_limiter = my_data[mask]['RV'].abs().max()
            lims.append([1e-5, jit_limiter])
            priors.append(norm)
            prargs.append(jit_prargs)


    elif b.type_ == 'Celerite2':
        pc = 0
        for i in range(len(d['terms'])):
            current_term = d['terms'][i]
            current_params = d['params'][i]

            if current_term == 'RealTerm':

                for param in current_params:
                    if current_params[param] == None:
                        lims.append([1e-5, 5])
                        priors.append(uni)
                        prargs.append(None)
                    else:
                        lims.append([np.nan, np.nan])
                        priors.append('Fixed')
                        prargs.append(None)
                        b[pc].fixed = current_params[param]
                    pc += 1
            elif current_term == 'Matern32Term':

                for param in current_params:
                    if current_params[param] == None:
                        lims.append([1e-5, 5])
                        priors.append(uni)
                        prargs.append(None)
                    else:
                        lims.append([np.nan, np.nan])
                        priors.append('Fixed')
                        prargs.append(None)
                        b[pc].fixed = current_params[param]
                    pc += 1

            elif current_term == 'RotationTerm':
                 for param in current_params:
                    if current_params[param] == None:
                        if param == 'f':
                            lims.append([1e-5, 1])
                        else:
                            lims.append([1e-5, 5])
                        priors.append(uni)
                        prargs.append(None)
                    else:
                        lims.append([np.nan, np.nan])
                        priors.append('Fixed')
                        prargs.append(None)
                        b[pc].fixed = current_params[param]
                    pc += 1               


    elif b.type_ == 'AstrometryKeplerian':
        pass


    elif b.type_ == 'AstrometryOffset':
        #['dOffset_ra', 'dOffset_dec',
        #          'dOffset_pm_ra', 'dOffset_pm_dec']
        
        off_lim = 1e6
        for i in range(2):
            lims.append([-off_lim, off_lim])
            priors.append(uni)
            prargs.append(None)

        for i in range(2):
            lims.append([-off_lim, off_lim])
            priors.append(uni)
            prargs.append(None)


    elif b.type_ == 'AstrometryJitter':
        # 'dJitterH', 'dJitterG'
        
        jit_lim = np.exp(12)
        for i in range(2):
            lims.append([-jit_lim, jit_lim])
            priors.append(uni)
            prargs.append(None)


    elif b.type_ == 'Acceleration':
        for nin in range(b.number_):
            daily = 1 # 1/365.25
            lims.append([-daily, daily])
            priors.append(uni)
            prargs.append(None)

    else:
        print(f'type_ {b.type_} not recognised. \nSmartLimits failed')



    b.set_attr('limits', lims, silent=True)
    b.set_attr('prior', priors, silent=True)
    b.set_attr('prargs', prargs, silent=True)


# TODO Rename this here and in `SmartLimits`
def sma_minmass_add_additional(b, uni):
    pnames = ['Semi-Major Axis', 'Minimum Mass']
    pdisplay_names = [f'{x} {b.number_}' for x in pnames]
    ppriors = [uni, uni]
    plims = [[1e-5, 1000], [1e-5, 1000]]
    pprargs = [[], []]
    phas_prior = [False, False]
    phas_posterior = [True, True]
    punits = ['(AU)', 'Mj']
    ndim = len(pnames)
    for i in range(ndim):
        d0 = {'name': pnames[i],
              'display_name': pdisplay_names[i],
              'prior': ppriors[i],
              'limits':plims[i],
              'prargs':pprargs[i],
              'has_prior':phas_prior[i],
              'has_posterior':phas_posterior[i],
              'unit':punits[i],
              'fixed':None,
              }

        b.add_additional_parameters(d0)

# https://en.wikipedia.org/wiki/Mathematical_operators_and_symbols_in_Unicode
# sun symbol ⊙
# earth symbol ⊕
#


# maths ≤ ≥ ∼ ∄	∊ ∏ ∑ ∫ ≪	≫
# letters 𝒜 𝒞	𝒟
# 𝒢			𝒥	𝒦			𝒩	𝒪	𝒫	𝒬		𝒮	𝒯
# 𝒰	𝒱	𝒲	𝒳	𝒴	𝒵	𝒶	𝒷	𝒸	𝒹		𝒻		𝒽	𝒾	𝒿
#	𝓀	𝓁	𝓂	𝓃		𝓅	𝓆	𝓇	𝓈	𝓉	𝓊	𝓋	𝓌	𝓍	𝓎	𝓏
#	𝓐	𝓑	𝓒	𝓓	𝓔	𝓕	𝓖	𝓗	𝓘	𝓙	𝓚	𝓛	𝓜	𝓝	𝓞	𝓟
#	𝓠	𝓡	𝓢	𝓣	𝓤	𝓥	𝓦	𝓧	𝓨	𝓩	𝓪	𝓫	𝓬	𝓭	𝓮	𝓯
#	𝓰	𝓱	𝓲	𝓳	𝓴	𝓵	𝓶	𝓷	𝓸	𝓹	𝓺	𝓻	𝓼	𝓽	𝓾	𝓿
#	𝔀	𝔁	𝔂	𝔃	𝔄	𝔅		𝔇	𝔈	𝔉	𝔊			𝔍	𝔎	𝔏
#	𝔐	𝔑	𝔒	𝔓	𝔔		𝔖	𝔗	𝔘	𝔙	𝔚	𝔛	𝔜

#   𝛢	𝛣	𝛤	𝛥	𝛦	𝛧	𝛨	𝛩	𝛪	𝛫	𝛬	𝛭	𝛮	𝛯
#	𝛰	𝛱	𝛲	𝛳	𝛴	𝛵	𝛶	𝛷	𝛸	𝛹	𝛺	𝛻	𝛼	𝛽	𝛾	𝛿
#	𝜀	𝜁	𝜂	𝜃	𝜄	𝜅	𝜆	𝜇	𝜈	𝜉	𝜊	𝜋	𝜌	𝜍	𝜎	𝜏
#	𝜐	𝜑	𝜒	𝜓	𝜔	𝜕	𝜖	𝜗	𝜘	𝜙	𝜚	𝜛	𝜜	𝜝	𝜞	𝜟
#	𝜠	𝜡	𝜢	𝜣	𝜤	𝜥	𝜦	𝜧	𝜨	𝜩	𝜪	𝜫	𝜬	𝜭	𝜮	𝜯
#	𝜰	𝜱	𝜲	𝜳	𝜴	𝜵	𝜶	𝜷	𝜸	𝜹	𝜺	𝜻	𝜼	𝜽	𝜾	𝜿
#	𝝀	𝝁	𝝂	𝝃	𝝄	𝝅	𝝆	𝝇	𝝈	𝝉	𝝊	𝝋	𝝌	𝝍	𝝎	𝝏
#	𝝐	𝝑	𝝒	𝝓	𝝔	𝝕
# ⏰	⏱	⏲	⏳ ⭐	⭑	⭒	⭓	⭔	⭕

# subscript_nums = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
# supscript_nums = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']
# ⁱ ⁺ ⁻ ⁿ ⁽	⁾
# ᵢ ₘ ₊ ₋ ₍	₎   ₍ᵢ₋₁₎
# ~𝓤()
# ~𝓝()
# ~𝓖()
# ~ 𝓙()

# ✅ ✔ ✓
# ◯
# ❎ ❌ ✘ ✗ ☒

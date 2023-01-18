# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
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
from .block import Parameter, Parameter_Block
from .utils import ModelWrapper
from .model_repo import *


def mk_KeplerianBlock(my_data, parameterisation=0, number=1):
    my_params = []
    subscript_nums = ['', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']

    if parameterisation == 0:
        kepmod = Keplerian_Model
        pnames = ['Period', 'Amplitude', 'Phase', 'Eccentricity', 'Longitude']
        punits = ['(Days)', r'($\frac{m}{s}$)', '(rad)', '', '(rad)']
        math_display = f'K⋅(cos(ν(t,P,𝜙,e)+𝜔)+e⋅cos(𝜔))|{subscript_nums[number]}'
        b_script = 'kep00.model'
        pis_circular = [False, False, True, False, True]
        pis_hou = [False, False, False, False, False]

    elif parameterisation == 1:
        kepmod = Keplerian_Model_1
        pnames = ['lPeriod', 'Amp_sin', 'Amp_cos', 'Ecc_sin', 'Ecc_cos']
        punits = ['(Days)', r'($\frac{m}{s}$)', '(rad)', '', '(rad)']
        math_display = f'K⋅(cos(ν(t,P,𝜙,e)+𝜔)+e⋅cos(𝜔))|{subscript_nums[number]}'
        b_script = 'kep01.model'
        pis_circular = [False, False, False, False, False]
        pis_hou = [False, True, True, True, True]

    elif parameterisation == 2:
        kepmod = Keplerian_Model_2
        pnames = ['Period', 'Amplitude', 'T_0', 'Eccentricity', 'Longitude']
        punits = ['(Days)', r'($\frac{m}{s}$)', '(Days)', '', '(rad)']
        math_display = f'K⋅(cos(ν(t,P,T₀,e)+𝜔)+e⋅cos(𝜔))|{subscript_nums[number]}'
        b_script = 'kep02.model'
        pis_circular = [False, False, False, False, True]
        pis_hou = [False, False, False, False, False]

    elif parameterisation == 3:
        kepmod = Keplerian_Model_3
        pnames = ['Period', 'Amplitude', 'T_0', 'Ecc_sin', 'Ecc_cos']
        punits = ['(Days)', r'($\frac{m}{s}$)', '(Days)', '', '(rad)']
        math_display = f'K⋅(cos(ν(t,P,T₀,e)+𝜔)+e⋅cos(𝜔))|{subscript_nums[number]}'
        b_script = 'kep03.model'
        pis_circular = [False, False, False, False, False]
        pis_hou = [False, False, False, True, True]

    if number:
        pnames = [pnam+' %i' % number for pnam in pnames]

    bdim = len(pnames)

    pvalues = [-np.inf for _ in range(bdim)]
    ppriors = ['Uniform' for _ in range(bdim)]
    plimits = [[None, None] for _ in range(bdim)]

    ptypes = [None for _ in range(bdim)]
    prargs = [None for _ in range(bdim)]
    ptformargs = [None for _ in range(bdim)]
    pfixed = [None for _ in range(bdim)]
    psigma = [None for _ in range(bdim)]

    pGM_parameter = [None for _ in range(bdim)]
    pposterior = [None for _ in range(bdim)]

    pstds = [None for _ in range(bdim)]

    for i in range(bdim):
        d0 = {'name':pnames[i], 'prior':ppriors[i], 'value':pvalues[i],
                  'limits':plimits[i], 'unit':punits[i], 'prargs':prargs[i],
                  'type':ptypes[i], 'ptformargs':ptformargs[i], 'fixed':pfixed[i],
                  'sigma':psigma[i], 'GM_parameter':pGM_parameter[i],
                  'posterior':pposterior[i], 'std':pstds[i],
                  'is_circular':pis_circular[i],
                  'is_hou':pis_hou[i]}

        my_params.append(Parameter(d0))

    b_mod = ModelWrapper(kepmod, [my_data.values[:, 0]])
    b_name = f'KeplerianBlock {number}'

    return Parameter_Block(my_params, block_model=b_mod,
                           block_name=b_name, block_type='Keplerian',
                           model_script=b_script,
                           is_iterative=True,
                           math_display=math_display, display=True,
                           parameterisation=parameterisation, number=number)


def mk_InstrumentBlock(my_data, number=1, moav=0, sa=False):
    my_params = []
    #subscript_nums = ['', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
    supscript_nums = ['', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']

    pnames = ['Offset', 'Jitter']
    punits = [r'($\frac{m}{s}$)', r'($\frac{m}{s}$)']
    math_display = f'γ₀|{supscript_nums[number]}'
    if number:
        pnames = [pnam+' %i' % number for pnam in pnames]
    if moav > 0:
        for j in range(moav):
            if number:
                pnames.append(f'MACoefficient {number} Order {j + 1}')
                pnames.append(f'MATimescale {number} Order {j+1}')

                punits.append('(Days)')
            else:
                pnames.extend((f'MACoefficient Order {j + 1}', f'MATimescale Order {j + 1}'))
        math_display = f'{math_display} + 𝛴ᵢ𝛴ₘ 𝛷ₘ⋅exp((t₍ᵢ₋ₘ₎-tᵢ)/𝜏ₘ)⋅𝜀(t₍ᵢ₋ₘ₎)'

    bdim = len(pnames)

    pvalues = [-np.inf for _ in range(bdim)]
    ppriors = ['Uniform' for _ in range(bdim)]
    plimits = [[None, None] for _ in range(bdim)]

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


    if moav > 0:
        insmod = Instrument_Moav_Model
        b_mod = ModelWrapper(insmod, [number, my_data.values[:, 3], moav,
                             my_data.values[:, 0], my_data.values[:, 1]])
        b_script = 'ins01.model'
        if sa:
            insmod = Instrument_Moav_SA_Model
            b_mod = ModelWrapper(insmod, [number, my_data.values[:, 3], moav,
                                 my_data.values[:, 0], my_data.values[:, 4:],
                                 my_data.shape[1]-4])
            b_script = None
    else:
        insmod = Instrument_Model
        b_mod = ModelWrapper(insmod, [number, my_data.values[:, 3]])
        b_script = 'ins00.model'

    b_name = f'InstrumentalBlock {number}'

    return Parameter_Block(my_params, block_model=b_mod,
                           block_name=b_name, block_type='Instrumental',
                           model_script=b_script,
                           is_iterative=False,
                           math_display=math_display, display=False,
                           parameterisation=None, number=number,
                           moav=moav)


def mk_AccelerationBlock(my_data, n=1):
    my_params = []
    subscript_nums = ['', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
    #supscript_nums = ['', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']
    punits = []
    acmod = Acceleration_Model
    for i in range(n):
        if i == 0:
            pnames = ['Acceleration']
            punits.append(r'($\frac{m}{s^2}$)')
        else:
            pnames.append(f'Acceleration Order {str(i + 1)}')
            punits.append(r'($\frac{m}{s^%s}$)' % str(i+2))

    bdim = len(pnames)


    pvalues = [-np.inf for _ in range(bdim)]
    ppriors = ['Uniform' for _ in range(bdim)]
    yearly = 1/365.25
    plimits = [[-yearly, yearly] for _ in range(bdim)]

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

    math_display = f'γ{subscript_nums[1]}'
    for j in range(bdim-1):
        math_display += f' + γ{subscript_nums[2 + j]}'
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
    b_name = f'AccelerationBlock o{n}'
    b_script = 'acc.model'

    return Parameter_Block(my_params, block_model=b_mod,
                           block_name=b_name, block_type='Acceleration',
                           model_script=b_script,
                           is_iterative=False,
                           math_display=math_display, display=False,
                           parameterisation=None, number=n)


def mk_AdditionalPriorsBlock(my_data):
    my_params = []
    empmod = Empty_Model
    b_mod = ModelWrapper(empmod, [my_data.values[:, 0]])
    b_name = 'AdditionalPriorsBlock'

    math_display = ''
    return Parameter_Block(my_params, block_model=b_mod,
                           block_name=b_name, block_type='AdditionalPriors',
                           is_iterative=False,
                           math_display=math_display, display=False,
                           parameterisation=None, number=None)


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
        amp_limiter = sig_limiter * np.sqrt(3)
        angle_limits = [0, 2*np.pi]

        ecc_limits, ecc_prargs = d['prargs']

        if b.parameterisation == 0:
            lims = [[0.1, per_limiter], [0, amp_limiter], angle_limits,
                       ecc_limits, angle_limits]
            priors = [uni, uni, uni, norm, uni]
            prargs = [None, None, None, ecc_prargs, None]

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
            _extracted_from_SmartLimits_122(b, uni)
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
        '''
            if d['dynamics']:
                if d['kplan'] > 1 and b.number_ == d['kplan']:
                    prarg = [d['kplan'], d['starmass']]
                    b.add_additional_priors([['Hill', 'Hill', [None, None], prarg]])
                    b.dynamics_bool = True
                else:
                    b.dynamics_bool = False
            '''

    elif b.type_ == 'Instrumental':
        jit_limits, jit_prargs = args

        mask = my_data['Flag']==b.number_
        jit_limiter = my_data[mask]['RV'].abs().max()

        lims = [[-jit_limiter, jit_limiter], [1e-5, jit_limiter]]
        priors = [uni, norm]
        prargs = [None, jit_prargs]

        if b.moav > 0:
            for _ in range(b.moav):
                lims.extend(([0.5, 1], [15, 25]))
                priors.extend((uni, uni))
                prargs.extend((None, None))
        '''
            elif b.type_ == 'AdditionalPriors':
                my_params = []

                pnames = []
                punits = []
                pvalues = []
                ppriors = []
                plimits = []

                ptypes = []
                prargs = []
                ptformargs = []
                pfixed = []
                psigma = []
                if False:
                    # All CVs go here!!!
                    sig_limiter = my_data['RV'].std(ddof=0)
                    per_limiter = my_data['BJD'].max() - my_data['BJD'].min()
                    amp_limiter = sig_limiter * np.sqrt(3)
                    angle_limits = [0, 2*np.pi]

                    ecc_limits, ecc_prargs = d['prargs']
                if True:
                    #Hill goes here
                    if d['dynamics']:
                        if d['kplan'] > 1:
                            prarg = [d['kplan'], d['starmass']]

                            pnames.append('Hill')
                            ppriors.append('Hill')
                            pvalues.append(None)
                            plimits.append([None, None])
                            punits.append(None)
                            prargs.append(prarg)
                            ptypes.append(None)
                            ptformargs.append(None)
                            pfixed.append(None)
                            psigma.append(None)

                bdim = len(pnames)
                for i in range(bdim):
                    d0 = {'name':pnames[i], 'prior':ppriors[i], 'value':pvalues[i],
                              'limits':plimits[i], 'unit':punits[i], 'prargs':prargs[i],
                              'type':ptypes[i], 'ptformargs':ptformargs[i], 'fixed':pfixed[i],
                              'sigma':psigma[i]}
                    my_params.append(Parameter(d0))
                b.list_ = np.array(my_params)
                return
            '''
    elif b.type_ != 'Acceleration':
        print(f'type_ {b.type_} not recognised. \nSmartLimits failed')

    b.set_attr('limits', lims, silent=True)
    b.set_attr('prior', priors, silent=True)
    b.set_attr('prargs', prargs, silent=True)


# TODO Rename this here and in `SmartLimits`
def _extracted_from_SmartLimits_122(b, uni):
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

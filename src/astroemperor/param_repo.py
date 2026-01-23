# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 1.0

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT
# **FIN  : Finish this

import numpy as np

def make_parameter(target):
    return {**dEmpty, **target}

dEmpty = { 'prior':'Uniform',
          'limits':[None, None],
          'init_pos':[None, None],
          'value':-np.inf,
          'value_max':-np.inf,
          'value_mean':-np.inf,
          'value_median':-np.inf,
          'value_range':[None, None],
          
          'fixed':None,
          'prargs':None,
          'type':None,
          'ptformargs':None,
          'sigma':None,
          'GM_parameter':None,
          'posterior':None,
          'std':None,
          'sigma_frac_mean':None,
          
          'display_prior':'',
          'display_posterior':'',
          }


dPeriod = {'name':'Period',         
            'unit':'(Days)',        
            'is_circular':False,
            'is_hou':False,
            'mininame':'P',
            'texname':r'P',
            }

dAmplitude = {'name':'Amplitude',         
            'unit':r'($\frac{m}{s}$)',        
            'is_circular':False,
            'is_hou':False,
            'mininame':'K',
            'texname':r'K',
            }

dPhase = {'name':'Phase',         
            'unit':'(rad)',
            'is_circular':True,
            'is_hou':False,
            'mininame':'M₀',
            'texname':r'M_0',
            }

dEccentricity = {'name':'Eccentricity',         
            'unit':'',
            'is_circular':False,
            'is_hou':False,
            'mininame':'e',
            'texname':r'e',
            }

dLongitude = {'name':'Longitude',         
            'unit':'(rad)',
            'is_circular':True,
            'is_hou':False,
            'mininame':'ω',
            'texname':r'\omega',
            }

######

dlPeriod = {'name':'lPeriod',         
            'unit':'(Days)',
            'is_circular':False,
            'is_hou':False,
            'mininame':r'\ln_P',
            }

dAmp_sin = {'name':'Amp_sin',
            'unit':r'($\frac{m}{s}$)',
            'is_circular':False,
            'is_hou':True,
            'mininame':'Kₛᵢₙ',
            'texname':r'K_{\sin}',
            }


dAmp_cos = {'name':'Amp_cos',
            'unit':'(rad)',
            'is_circular':False,
            'is_hou':True,
            'mininame':'Kₖₒₛ',
            'texname':r'K_{\cos}',
            }

dEcc_sin = {'name':'Ecc_sin',
            'unit':'',
            'is_circular':False,
            'is_hou':True,
            'mininame':'eₛᵢₙ',
            'texname':r'e_{\sin}',
            }

dEcc_cos = {'name':'Ecc_cos',
            'unit':'(rad)',
            'is_circular':False,
            'is_hou':True,
            'mininame':'eₖₒₛ',
            'texname':r'e_{\cos}',
            }

#######

dT_0 = {'name':'T_0',         
            'unit':'(Days)',
            'is_circular':False,
            'is_hou':False,
            'mininame':'T₀',
            'texname':r'T_0',
            }

dM0 = {'name':'M0',
       'unit':'(Days)',
       'is_circular':False,
       'is_hou':False,
       'mininame':'M₀',
       'texname':r'M_0',
       }

#######

dSMA = {'name':'Semi-Major Axis',
        'unit':'(AU)',
        'is_circular':False,
        'is_hou':False,
        'mininame':r'a',
        'texname':r'a',
        }

dMinM = {'name':'Minimum Mass',
         'unit':'(Mj)',
         'is_circular':False,
         'is_hou':False,
         'mininame':'Mₘᵢₙ',
         'texname':r'M_{\text{min}}',
         }

#######

dOffset = {'name':'Offset',
           'unit':r'($\frac{m}{s}$)',
           'is_circular':False,
           'is_hou':False,
           'mininame':'γ₀',
           'texname':r'\gamma_0',
            }

dJitter = {'name':'Jitter',
           'unit':r'($\frac{m}{s}$)',
           'is_circular':False,
           'is_hou':False,
           'mininame':'J',
           'texname':r'J',
            }

dMACoefficient = {'name':'MACoefficient',
           'unit':r'($\frac{m}{s}$)',
           'is_circular':False,
           'is_hou':False,
           'mininame':'Φ',
           'texname':r'\Phi',
            }

dMATimescale = {'name':'MATimescale',
           'unit':'(Days)',
           'is_circular':False,
           'is_hou':False,
           'mininame':'τ',
           'texname':r'\tau',
            }

dStaract = {'name':'Staract',
           'unit':r'($\frac{m}{s}$)',
           'is_circular':False,
           'is_hou':False,
           'mininame':r'𝓐',
           'texname':r'\mathcal{A}',
            }

#######

dAcceleration = {'name':'Acceleration',
                 'unit':r'($\frac{m}{s day}$)',
                 'is_circular':False,
                 'is_hou':False,
                 'mininame':'γ⁽¹⁾',
                 'texname':r'\dot{\gamma}',
                }

#######

dCeleJitter = {'name':'Jitter Term',
            'unit':r'($\frac{m}{s}$)',
            'is_circular':False,
            'is_hou':False,
            'mininame':r'',
             }

#######

dRealTerm_a = {'name':'Real Term a',
            'unit':r'($\frac{m}{s}$)',
            'is_circular':False,
            'is_hou':False,
            'mininame':'aᵣ',
            'texname':r'a_{\rm real}',
             }

dRealTerm_c = {'name':'Real Term c',
            'unit':'',
            'is_circular':False,
            'is_hou':False,
            'mininame':'cᵣ',
            'texname':r'c_{\rm real}',
             }

########

dRotationTerm_sigma = {'name':'Rotation Term sigma',
            'unit':r'($\frac{m}{s}$)',
            'is_circular':False,
            'is_hou':False,
            'mininame':'σᵣₒₜ',
            'texname':r'\sigma_{\rm rot}',
             }

dRotationTerm_period = {'name':'Rotation Term period',
            'unit':'(days)',
            'is_circular':False,
            'is_hou':False,
            'mininame':'Pᵣₒₜ',
            'texname':r'\P_{\rm rot}',
             }

dRotationTerm_Q0 = {'name':'Rotation Term Q0',
            'unit':r'',
            'is_circular':False,
            'is_hou':False,
            'mininame':'Q₀ᵣₒₜ',
            'texname':r'Q_{0,\rm rot}',
             }

dRotationTerm_dQ = {'name':'Rotation Term dQ',
            'unit':'',
            'is_circular':False,
            'is_hou':False,
            'mininame':'dQᵣₒₜ',
            'texname':r'dQ_{\rm rot}',
             }

dRotationTerm_f = {'name':'Rotation Term f',
            'unit':r'',
            'is_circular':False,
            'is_hou':False,
            'mininame':'fᵣₒₜ',
            'texname':r'f_{\rm rot}',
             }

########
dMatern32Term_sigma = {'name':'Matern32 Term sigma',
            'unit':r'($\frac{m}{s}$)',
            'is_circular':False,
            'is_hou':False,
            'mininame':'σₘ₃₂',
            'texname':r'\sigma_{M(3/2)}',
             }

dMatern32Term_rho = {'name':'Matern32 Term rho',
            'unit':r'',
            'is_circular':False,
            'is_hou':False,
            'mininame':'ρₘ₃₂',
            'texname':r'\rho_{M(3/2)}',
             }

########

dSHOTerm_sigma = {'name':'SHO Term σ',
            'unit':r'($\frac{m}{s}$)',
            'is_circular':False,
            'is_hou':False,
            'mininame':'σₛₕₒ',
            'texname':r'\sigma_{\rm SHO}',
             }

dSHOTerm_rho = {'name':'SHO Term ρ',
            'unit':r'(days)',
            'is_circular':False,
            'is_hou':False,
            'mininame':'ρₛₕₒ',
            'texname':r'\rho_{\rm SHO}',
             }

dSHOTerm_tau = {'name':'SHO Term τ',
            'unit':r'(days)',
            'is_circular':False,
            'is_hou':False,
            'mininame':'τₛₕₒ',
            'texname':r'\tau_{\rm SHO}',
             }

dSHOTerm_S0 = {'name':'SHO Term S0',
            'unit':r'($\frac{m}{s}$)',
            'is_circular':False,
            'is_hou':False,
            'mininame':'S₀ₛₕₒ',
            'texname':r'S_{0,\rm SHO}',
             }

dSHOTerm_w0 = {'name':'SHO Term w0',
            'unit':r'',
            'is_circular':False,
            'is_hou':False,
            'mininame':'ω₀ₛₕₒ',
            'texname':r'\omega_{0,\rm SHO}',
             }

dSHOTerm_Q = {'name':'SHO Term Q',
            'unit':r'',
            'is_circular':False,
            'is_hou':False,
            'mininame':'Qₛₕₒ',
            'texname':r'Q_{\rm SHO}',
             }

########

dGonzRotationTerm_rho = {'name':'GRot Term rho',
            'unit':r'(days)',
            'is_circular':False,
            'is_hou':False,
            'mininame':r'',
            'texname':r'\rho_{\rm grot}',
             }

dGonzRotationTerm_tau = {'name':'GRot Term tau',
            'unit':r'(days)',
            'is_circular':False,
            'is_hou':False,
            'mininame':r'',
            'texname':r'\tau_{\rm grot}',
             }

dGonzRotationTerm_A1 = {'name':'GRot Term A1',
            'unit':r'($\frac{m}{s}$)',
            'is_circular':False,
            'is_hou':False,
            'mininame':r'',
            'texname':r'A_{1,\rm grot}',
             }

dGonzRotationTerm_A2 = {'name':'GRot Term A2',
            'unit':r'($\frac{m}{s}$)',
            'is_circular':False,
            'is_hou':False,
            'mininame':r'',
            'texname':r'A_{2,\rm grot}',
             }



########

# astrometry, orbital
dInclination = {'name':'Inclination',
            'unit':'(rad)',
            'is_circular':True,
            'is_hou':False,
            'mininame':r'I',
            'texname':r'I',
            }

dOmega = {'name':'Omega',
          'unit':'(rad)',
          'is_circular':True,
          'is_hou':False,
          'mininame':r'Ω',
          'texname':r'\Omega',
          }

dInc1 = {'name':'Inc1',
            'unit':'(rad)',
            'is_circular':True,
            'is_hou':False,
            'mininame':r'I₁',
            }

dInc2 = {'name':'Inc2',
            'unit':'(rad)',
            'is_circular':True,
            'is_hou':False,
            'mininame':r'I₂',
            }

dInc0 = {'name':'Inclination',
            'unit':'(rad)',
            'is_circular':True,
            'is_hou':False,
            'mininame':r'I₀',
            }


dIncD = {'name':'Inclination',
            'unit':'(rad)',
            'is_circular':True,
            'is_hou':False,
            'mininame':r'I',
            }

########

# astrometry, instrumental
dOffset_ra = {'name':'Offset RA*',
           'unit':'(mas)',
           'is_circular':False,
           'is_hou':False,
           'mininame':'Δα',
           'texname':r'\Delta \alpha_*',
            }

dOffset_de = {'name':'Offset DE',
           'unit':'(mas)',
           'is_circular':False,
           'is_hou':False,
           'mininame':'Δδ',
           'texname':r'\Delta \delta',
            }

dOffset_plx = {'name':'Offset PLX',
           'unit':'(mas)',
           'is_circular':False,
           'is_hou':False,
           'mininame':'Δω',
           'texname':r'\Delta \varpi',
            }


dOffset_pm_ra = {'name':'Offset pm RA*',
           'unit':'(mas)',
           'is_circular':False,
           'is_hou':False,
           'mininame':'Δµᵅ',
           'texname':r'\Delta \mu_{\alpha *}',
            }

dOffset_pm_de = {'name':'Offset pm DE',
           'unit':'(mas)',
           'is_circular':False,
           'is_hou':False,
           'mininame':'Δµᵟ',
           'texname':r'\Delta \mu_{\delta}',
            }


dJitterH = {'name':'Jitter Hipp',
           'unit':'',
           'is_circular':False,
           'is_hou':False,
           'mininame':r'J⁽ᴴ⁾',
           'texname':r'J_{\rm Hipp}',
            }

dJitterG = {'name':'Jitter Gaia',
           'unit':r'',
           'is_circular':False,
           'is_hou':False,
           'mininame':r'S⁽ᴳ⁾',
           'texname':r'S_{\rm Gaia}',
            }


'''
free parameters in combined RV and astrometry:
    - orbital period (P)
    - RV semi-amplitude (K)
    - eccentricity (e),
    - argument of periastron (ω)
    - mean anomaly (M0) at t0, replazable with T0?
    
    - inclination (I)
    - longitude of ascending node (𝛺)
    

    - RV jitter (σJ)
    - time-scale (τ)
    - amplitude (𝜙) of the MA model

    - offset in α (𝛥α)
    - offset in δ (𝛥δ)
    - offset in μα (𝛥μα)
    - offset in μδ (𝛥μδ)
    
    - log jitter in Gaia (ln Jgaia)
    - log jitter in Hipp (ln Jhip)
    
    == Companion mass (mp), semimajor axis (a), and the epoch at the periastron (TP ) are derived.

    
    + GOST for smearing?
    + instrumental free parameters go in the logl?

    + model and likelihood in
    ++ mcmc_func.R -> RV.kepler
    ++ mcmc_func.R -> RV.kepler

    + orbit in 
    ++ orbit.R -> kepler.classic
    ++ orbit.R -> kepler.PN


    def model():
        I: inclination
        Vkep = K * (angular_things)

        K = RV * np.sin(I)
        RV = 
'''

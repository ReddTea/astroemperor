# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import astroemperor as emp

np.random.seed(1234)


sim = emp.Simulation()
target_folder = 'synth'

setup = np.array([2, 100, 300])

burnin = 0.25

my_moav = 0
my_accel = 0
my_parameterisation = 0

sim.jitter_prargs = [0, 5]

thinby = 1


switch_SA = False
switch_celerite = False
sim.switch_dynamics = False
sim.switch_constrain = True
sim.constrain_method = 'sigma'#, 'GM'

sim.constrain_sigma = 1




if True:
    sim.cores__ = 8
    sim.FPTS = False
    sim.cherry['cherry'] = True
    sim.cherry['diff'] = 20
    sim.acceleration = my_accel
    sim.keplerian_parameterisation = my_parameterisation

    sim.set_engine('reddemcee')
    sim.reddemcee_config['iterations'] = 1
    sim.reddemcee_config['burnin'] = burnin
    sim.reddemcee_config['thinby'] = thinby
    sim.reddemcee_config['logger_level'] = 'CRITICAL'

    sim.multiprocess_method = 1  # 0 no mp, 1 multiprocessing

    sim.save_loc = ''
    sim.switch_SA = switch_SA
    sim.switch_celerite = switch_celerite

    sim.my_kernel = {'terms':['RealTerm'],
                     'params':[{'a':None,
                                'c':None}]
                            }

    sim.my_kernel = {'terms':['RotationTerm'],
                     'params':[{'sigma':None,
                                'period':None,
                                'Q0':None,
                                'dQ':None,
                                'f':None}]
                            }

    sim.my_kernel = {'terms':['Matern32Term'],
                     'params':[{'sigma':None,
                                'rho':None}]
                            }

    # PLOT OPTIONS
    sim.plot_paper_mode = True  # superseeds all others
    sim.save_plots_fmt = 'pdf'

    sim.gaussian_mixtures_fit = True
    sim.plot_gaussian_mixtures['plot'] = True

    sim.posterior_fit_method = 'GM'  # None, 'GM'

    if True:
        sim.plot_keplerian_model['plot'] = True  # hist, uncertain
        sim.plot_keplerian_model['hist'] = True
        sim.plot_keplerian_model['errors'] = True
        sim.plot_keplerian_model['periodogram'] = True
        sim.plot_keplerian_model['uncertain'] = True
        sim.plot_keplerian_model['logger_level'] = 'CRITICAL'  # ERROR, CRITICAL

    if True:
        sim.plot_posteriors = {'plot':True,
                                'modes':[0, 1, 2],
                                'dtp':None,
                                'format':'png',
                                'logger_level':'DEBUG',
                                'saveloc':'',
                                'fs_supt':20,
                                }
        
    if True:
        sim.plot_histograms = {'plot':True,
                                'format':'png',
                                'logger_level':'ERROR',
                                'saveloc':'',
                                'fs_supt':16,
                                'paper_mode':False,
                                  }

    if True:
        sim.plot_trace['plot'] = True
        sim.plot_trace['modes'] = [0]#,1,2,3]  # 0:trace, 1:norm_post, 2:dens_interv, 3:corner

    sim.debug_mode = False  # DEBUG MODE
    sim.ModelSelection.set_criteria = 'BIC'  # default is BIC
    sim.ModelSelection.set_tolerance = 5

    sim.ModelSelection.update()

    sim.load_data(target_folder)


    if target_folder == 'synth':
        sim.instrument_names_RV = ['Synth Data 1', 'Synth Data 2']

        if my_parameterisation == 0:
            sim.add_condition(['Period 1', 'limits', [40, 80]])
            sim.add_condition(['Period 2', 'limits', [17, 32]])

            sim.add_condition(['Amplitude 1', 'limits', [150, 250]])
            sim.add_condition(['Amplitude 2', 'limits', [50, 150]])

            sim.add_condition(['Phase 1', 'limits', [np.pi/4-1., np.pi/4+1.]])
            sim.add_condition(['Phase 2', 'limits', [np.pi/3-1., np.pi/3+1.]])

            sim.add_condition(['Eccentricity 1', 'fixed', 0])
            sim.add_condition(['Eccentricity 2', 'fixed', 0])

            sim.add_condition(['Longitude 1', 'fixed', 0])
            sim.add_condition(['Longitude 2', 'fixed', 0])

            #sim.add_condition(['Offset 1', 'limits', [99, 101]])
            #sim.add_condition(['Offset 2', 'limits', [-101, -99]])

            # substract mean for RVs in utils
            #sim.add_condition(['Offset 1', 'fixed', 100])
            #sim.add_condition(['Offset 2', 'fixed', -100])

            sim.add_condition(['Acceleration', 'fixed', 0])




sim.run_auto(setup, k_start=0, k_end=2, moav=my_moav)

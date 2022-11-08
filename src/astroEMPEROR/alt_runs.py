
    # keplerian
    if False:

        sim = Simulation()
        sim._data__('Test Keplerian')

        sim.starmass = 0.37

        theta_ = [61.082, 212.07, 0.8, 0.027, 0.6126,
                  30.126, 88.34, 0.18, 0.25, 0.9006,
                  0.00, 5.1212]

    # keplerian emcee
    if False:
        # engine

        sim.plot_save = True
        sim.run_save = True
        #sim.plot_show = True
        sim._set_engine__('emcee')

        #sim._mk_noise_instrumental__()

        #sim._mk_keplerian__()
        #sim._mk_keplerian__()

        # periods
        #sim[0][0].limits = [56, 66]
        #sim[1][0].limits = [25, 35]

        sim.conds.append(['Period 1', 'limits', [56, 66]])
        sim.conds.append(['Period 2', 'limits', [25, 35]])

        sim.conds.append(['Amplitude 1', 'limits', [180, 220]])
        sim.conds.append(['Amplitude 2', 'limits', [80, 120]])

        sim.conds.append(['Eccentricity 1', 'fixed', 0])
        sim.conds.append(['Eccentricity 2', 'fixed', 0])

        sim.conds.append(['Longitude 1', 'fixed', 0])
        sim.conds.append(['Longitude 2', 'fixed', 0])

        sim.conds.append(['Offset 1', 'fixed', 0])
        sim.conds.append(['Offset 2', 'fixed', 0])


        sim.conds.append(['Jitter 1', 'limits', [0.1, 10]])
        sim.conds.append(['Jitter 2', 'limits', [0.1, 10]])

        sim._mk_acceleration__()

        sim.constrain_run = True
        sim.chain_save = [0]
        # ntemps, nwalkers, nsteps
        setup = np.array([3, 500, 2000])
        #sim.betas = np.array([1, 0.5, 0.1])
        #sim._run__(setup)
        #sim._post_process__(setup)

        #'Period 1', 'limits', [56, 66]
        sim._run_auto__(setup, 3, moav=1)
        #sim.plotpost()

        pl.close('all')

    # keplerian dynesty
    if False:


        sim._set_engine__('dynesty')

        #sim._mk_keplerian__()
        #sim._mk_keplerian__()
        #sim._mk_noise_instrumental__()

        sim.conds.append(['Period 1', 'ptformargs', [5, 61]])
        sim.conds.append(['Period 2', 'ptformargs', [5, 30]])

        sim.conds.append(['Amplitude 1', 'limits', [190, 210]])
        sim.conds.append(['Amplitude 2', 'limits', [95, 105]])

        sim.conds.append(['Phase 1', 'ptformargs', [0.1, np.pi/3.]])
        sim.conds.append(['Phase 2', 'ptformargs', [0.1, 4.*np.pi/3]])

        sim.conds.append(['Eccentricity 1', 'fixed', 0])
        sim.conds.append(['Eccentricity 2', 'fixed', 0])

        sim.conds.append(['Longitude 1', 'fixed', 0])
        sim.conds.append(['Longitude 2', 'fixed', 0])

        sim.conds.append(['Offset 1', 'fixed', 0])
        sim.conds.append(['Offset 2', 'fixed', 0])


        sim.conds.append(['Jitter 1', 'ptformargs', [2, 5]])
        sim.conds.append(['Jitter 2', 'ptformargs', [2, 5]])



        setup = np.array([2000])
        #sim._run__(setup)
        #sim._post_process__(setup)
        sim.plot_save = True
        sim.chain_save = [0]  # default
        sim.constrain_run = True

        sim._run_auto__(setup, 3, moav=1)

        # PLOTS
        #sim.plotmodel()
        #sim.plottrace()
        #sim.plotpost()

    # keplerian pymc3
    if False:
        sim._set_engine__('pymc3')

        #sim._mk_keplerian__()
        #sim._mk_keplerian__()
        #sim._mk_noise_instrumental__()

        sim.conds.append(['Period 1', 'limits', [55, 65]])
        sim.conds.append(['Period 2', 'limits', [25, 35]])

        sim.conds.append(['Amplitude 1', 'limits', [190, 210]])
        sim.conds.append(['Amplitude 2', 'limits', [95, 105]])

        sim.conds.append(['Phase 1', 'limits', [1., 1.1]])
        sim.conds.append(['Phase 2', 'limits', [4.1, 4.3]])

        sim.conds.append(['Eccentricity 1', 'fixed', 0])
        sim.conds.append(['Eccentricity 2', 'fixed', 0])

        sim.conds.append(['Longitude 1', 'fixed', 0])
        sim.conds.append(['Longitude 2', 'fixed', 0])

        sim.conds.append(['Offset 1', 'fixed', 0])
        sim.conds.append(['Offset 2', 'fixed', 0])

        sim.conds.append(['Jitter 1', 'prior', 'Uniform'])
        sim.conds.append(['Jitter 2', 'prior', 'Uniform'])

        sim.conds.append(['Jitter 1', 'limits', [3, 7]])
        sim.conds.append(['Jitter 2', 'limits', [3, 7]])




        # draw, tune, chains
        sim.plot_save = True
        sim.chain_save = [0]  # default


        sim.constrain_run = True

        setup = np.array([2000, 2000, 3])

        sim._run_auto__(setup, 3)
        #sim._run__(setup)
        #sim._post_process__(setup)

        # PLOTS
        #sim.plotmodel()
        #sim.plottrace()
        #sim.plotpost()


        pass


    # line
    if False:
        sim = Simulation()
        sim._add_data__('Test_line', label='Test Data 1')
        sim._sort_data__()

        m_ = -0.9594
        n_ = 4.294
        j_ = 0.534
        theta_ = np.array([m_, n_, j_])


    # line dynesty
    if False:

        import dynesty
        sim._set_engine__(dynesty)

        sim._mk_acceleration__()
        sim._mk_noise_instrumental__()


        sim[0][0].ptformargs = [2, 0]
        sim[1][0].ptformargs = [8, 0]
        sim[1][1].ptformargs = [0.5, 0.5]

        setup = np.array([1000])
        sim._run__(setup)
        sim._post_process__(setup)

    # line emcee
    if False:

        import emcee
        sim._set_engine__(emcee)

        sim._mk_acceleration__()
        sim._mk_noise_instrumental__()

        sim[0][0].limits = [-3, 3]  # m
        sim[1][0].limits = [-8, 8]  # n

        sim[1][1].prior = 'Uniform'
        sim[1][1].limits = [0, 1]  # j

        # ntemps, nwalkers, nsteps
        setup = np.array([3, 100, 10000])
        #sim.betas = np.array([1, 0.5, 0.1])
        sim._run__(setup)
        sim._post_process__(setup)
        sim.plotline()

    # line pymc3
    if False:

        import pymc3 as pm
        sim._set_engine__(pm)

        sim._mk_acceleration__()
        sim._mk_noise_instrumental__()

        sim[0][0].limits = [-2, 2]  # m
        sim[1][0].limits = [-8, 8]  # n

        sim[1][1].prior = 'Uniform'
        sim[1][1].limits = [0, 1]  # j




        # dr, tn, ch
        setup = np.array([1000, 5000, 3])
        sim._run__(setup)
        sim._post_process__(setup)
        sim.plotline()

    # TOI1634
    if False:

        # Simulation setup
        sim = Simulation()  # Creates the instance
        sim.developer_mode = True  # displays a couple of extra messages

        #sim.plot_show = True  # shows plot, requires to manually close them
        sim.plot_save = True  # saves plots
        # self.plot_fmt = 'png'  # default is png.

        #sim.run_save = True  # Saves log. Default is True
        sim.starmass = 0.497  # optional, required for planet signatures

        # data
        sim._data__('TOI1634')  # Target folder name in /datafiles/

        # engine

        import emcee
        sim._set_engine__(emcee)

        # model
        kplan = 1
        sim._mk_keplerian_scale__(kplan)
        #sim._mk_keplerian__()
        #sim._mk_keplerian__()

        sim._mk_noise_instrumental__()

        # parameter conditions

        sim[0][0].limits = [0.9, 1.1]  # period
        sim[0][1].limits = [5, 7]  # amplitude

        #sim[0][3].limits = [0.0, 0.4]  # ecc
        #sim[0][4].limits = [0.0, 0.1]  # w

        #sim[0][3].fixed = 0  # ecc
        #sim[0][4].fixed = 0  # w

        sim[0][5*(kplan)].fixed = 1  # scale 1 for 1 planet
        #sim[0][5*(kplan)+1].limits = [0, 3]

        #sim[0][5*(kplan)].limits = [0, 3]  # scale 1 for  planet
        #sim[0][5*(kplan)+1].fixed = 1

        # DATASET 1
        sim[1][0].limits = [-1, 1]  # offset

        sim[1][1].limits = [0, 3]  # jitter
        sim[1][1].prargs = [2, 2]  # jitter

        # DATASET 2
        #sim[2][0].limits = [-3, 3]  # offset
        #sim[2][1].limits = [0, 3]  # jitter
        #sim[1][1].prargs = [2, 2]  # jitter

        setup = np.array([5, 1000, 4000])
        #sim.betas = np.array([1, 0.5, 0.1])


        # RUN THE THING
        sim._run__(setup)
        sim._post_process__(setup)


        # PLOTS
        #sim.plotmodel()
        sim.plotmodelscale()
        sim.plottrace()
        sim.plotpost()
        #sim.plotcorner()
        # SAVE
        sim.save_chain([0])
        sim.save_posteriors([0])


        pl.close('all')

    # scale testing
    if False:
        sim = Simulation()

        sim.developer_mode = True
        #sim.plot_show = True
        sim.plot_save = True
        sim.run_save = True

        sim.starmass = 1.

        sim._data__('Test scale')

        import emcee
        sim._set_engine__(emcee)

        #sim._mk_keplerian__(param='hout0')
        sim._mk_keplerian_scale__(2)

        sim._mk_noise_instrumental__()
        #sim._mk_scale_instrumental__()

        sim[0][0].limits = [55, 65]  # period
        sim[0][1].limits = [180, 220]  # amp
        sim[0][2].limits = [1., 1.1]  # phase

        sim[0][0+5].limits = [25, 35]  # period
        sim[0][1+5].limits = [80, 120]  # amp
        sim[0][2+5].limits = [4.1, 4.2]  # phase

        sim[0][3].fixed =  0  # ecc
        sim[0][4].fixed = 0  # w
        sim[0][3+5].fixed =  0  # ecc
        sim[0][4+5].fixed = 0  # w
        #sim[0][3].limits = [0.0, 0.01]  # ecc
        #sim[0][4].limits = [0.0, 0.01]  # w


        sim[0][10].fixed = 1  # scale dataset 1
        #sim[0][11].fixed = 0.3  # scale dataset 2

        # DATASET 1
        #sim[1][0].limits = [-1, 1]  # offset
        #sim[1][1].limits = [4, 6]  # jitter


        sim[1][0].fixed = 0  # offset
        sim[1][1].fixed = 5  # jitter


        #sim[3][0].limits = [0.99, 1.01]  # scale

        # DATASET 2
        sim[2][0].limits = [25, 35]  # offset
        #sim[2][1].limits = [3, 7]  # jitter
        #sim[2][0].fixed = 30  # offset
        sim[2][1].fixed = 5  # jitter

        #sim[4][0].limits = [1.9, 2.1]

        setup = np.array([2, 400, 600])
        #sim.betas = np.array([1, 0.5, 0.1])
        sim._run__(setup)
        sim._post_process__(setup)

        sim.plotmodelscale()
        sim.plottrace()
        sim.plotpost()

        #sim.plotcorner()

        #sim.save_chain()
        #sim.save_posteriors()
        pl.close('all')

        #theta_ = np.array([60.0, 200.00, np.pi/3, 0.5, 0.00, 5])
        theta_ = np.array([60.0, 200.00, np.pi/3, 0, 0, 0.0, 5])

    # HIP18606

    if False:

        # Simulation setup

        sim = Simulation()  # Creates the instance
        sim.developer_mode = True  # displays a couple of extra messages

        #sim.plot_show = True  # shows plot, requires to manually close them
        sim.plot_save = True  # saves plots
        # self.plot_fmt = 'png'  # default is png.

        #sim.run_save = True  # Saves log. Default is True
        sim.starmass = 1  # optional, required for planet signatures

        # data
        sim._data__('HIP18606')  # Target folder name in /datafiles/

        # engine

        import emcee
        sim._set_engine__(emcee)

        # model
        sim._mk_keplerian__()
        #sim._mk_keplerian__()
        #sim._mk_keplerian__()

        sim._mk_noise_instrumental__()
        #sim._mk_scale_instrumental__()


        # parameter conditions

        #sim[0][0].limits = [0.1, 2]  # period
        #sim[0][1].limits = [2, 7]  # amplitude

        #sim[0][3].limits = [0.0, 0.1]  # ecc
        #sim[0][4].limits = [0.0, 0.1]  # w

        #sim[0][3].fixed = 0  # ecc
        #sim[0][4].fixed = 0  # w

        # DATASET 1
        #sim[1][0].limits = [-3, 3]  # offset

        sim[1][1].limits = [0, 3]  # jitter
        #sim[1][1].prargs = [2, 2]  # jitter

        # DATASET 2
        #sim[2][0].limits = [-3, 3]  # offset
        #sim[2][1].limits = [0, 5]  # jitter


        # Scales
        # harps
        sim[4][0].fixed = 1  # scale  # harps
        #sim[4][0].limits = [0.99, 1.01]  # scale  # harps
        #sim[2][0].limits = [0.99, 1.01]  # scale  # harps

        # subaru
        sim[3][0].limits = [0.1, 1.0]

        setup = np.array([5, 1000, 3000])
        #sim.betas = np.array([1, 0.5, 0.1])


        # RUN THE THING
        sim._run__(setup)
        sim._post_process__(setup)


        # PLOTS
        sim.plotmodel()
        sim.plottrace()
        sim.plotpost()
        #sim.plotcorner()
        # SAVE
        #sim.save_chain()
        #sim.save_posteriors()


        pl.close('all')

    # HIP111909

    if False:

        # Simulation setup

        sim = Simulation()  # Creates the instance
        sim.developer_mode = True  # displays a couple of extra messages

        sim.plot_save = True  # saves plots
        sim.run_save = True
        sim.constrain_run = True
        sim.chain_save = [0]
        #sim.plot_show = True  # shows plot, requires to manually close them
        # self.plot_fmt = 'png'  # default is png.

        #sim.run_save = True  # Saves log. Default is True
        sim.starmass = 1  # optional, required for planet signatures

        # data
        sim._data__('HIP111909')  # Target folder name in /datafiles/

        # engine

        sim._set_engine__('emcee')

        # model
        sim._mk_acceleration__()
        # parameter conditions

        sim.conds.append(['Period 1', 'limits', [450, 550]])
        sim.conds.append(['Amplitude 1', 'limits', [20, 40]])

        setup = np.array([5, 1000, 3000])
        #sim.betas = np.array([1, 0.5, 0.1])


        # RUN THE THING
        sim._run_auto__(setup, 2, moav=1)


        pl.close('all')

    # GJ357
    if False:
        pass






#

